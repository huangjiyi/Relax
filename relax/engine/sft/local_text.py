# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Node-local plain text SFT input for single-node Megatron training."""

import hashlib
import json
import mmap
import os
from argparse import Namespace
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import numpy as np

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def local_text_sft(args: Namespace) -> bool:
    return bool(getattr(args, "sft_local_text_dir", None))


def validate_local_text_sft(args: Namespace) -> int:
    from relax.engine.sft.runtime import is_sft_mode

    if (
        not is_sft_mode(args)
        or args.train_backend != "megatron"
        or args.tensor_model_parallel_size != 1
        or args.pipeline_model_parallel_size != 1
        or args.context_parallel_size != 1
        or args.qkv_format != "thd"
        or args.num_epoch != 1
        or getattr(args, "sft_training_mode", "sft") != "sft"
        or getattr(args, "task_type", "causal_lm") != "causal_lm"
        or args.input_key != "messages"
        or any(
            getattr(args, name, None)
            for name in (
                "multimodal_keys",
                "custom_dataset_class_path",
                "eval_size",
                "eval_prompt_data",
                "conversation_key_map",
                "system_prompt",
                "apply_chat_template_kwargs",
                "label_key",
                "sft_loss_last_turn_only",
                "sft_ignore_empty_think",
                "sft_mask_reasoning",
                "sft_wholepack",
                "use_routing_replay",
                "use_rollout_routing_replay",
                "use_rollout_indexer_replay",
            )
        )
    ):
        raise ValueError("Local text input requires plain Megatron THD text SFT for one epoch (TP/PP/CP=1).")

    directory = Path(args.sft_local_text_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    source = Path(args.prompt_data)
    source_stat = source.stat()
    if (str(source), source_stat.st_size, source_stat.st_mtime_ns, args.seed) != (
        manifest["source"],
        manifest["source_bytes"],
        manifest["source_mtime_ns"],
        manifest["seed"],
    ):
        raise ValueError("Local text index does not match --prompt-data or the shuffle seed.")
    size = manifest["rows"]
    if (
        (directory / "source.jsonl").stat().st_size != source_stat.st_size
        or (directory / "source-offsets.u64").stat().st_size != size * 8
        or (directory / f"epoch0-seed{args.seed}.u32").stat().st_size != size * 4
    ):
        raise ValueError("Local text source, index, or shuffle order is incomplete.")
    prepared = getattr(args, "sft_local_prepared_batches", None)
    if prepared:
        metadata = json.loads((Path(prepared) / "batch-manifest.json").read_text())
        if any(metadata.get(key) != value for key, value in manifest.items()):
            raise ValueError("Prepared batches do not match the original indexed dataset.")
        model_source = getattr(args, "_model_source_original_hf_checkpoint", None) or args.hf_checkpoint
        if (metadata["gbs"], metadata["steps"], metadata["model"], args.max_tokens_per_gpu) != (
            args.global_batch_size,
            size // args.global_batch_size,
            model_source,
            16384,
        ):
            raise ValueError("Prepared batches do not match the training recipe.")
    return size


class _IndexedJSONL:
    def __init__(self, directory: Path) -> None:
        self.offsets = np.memmap(directory / "source-offsets.u64", mode="r", dtype="<u8")
        with (directory / "source.jsonl").open("rb") as source:
            self.bytes = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self.offsets[index])
        end = self.bytes.find(b"\n", start)
        return json.loads(self.bytes[start : end if end >= 0 else len(self.bytes)])


class LocalTextBatches:
    def __init__(self, args: Namespace, tokenizer: Any) -> None:
        from relax.components.sft import _resolve_pad_token_ids_from_config
        from relax.engine.sft.dataset.streaming import SFTStreamingDataset

        # This route runs one native tokenizer batch per data-parallel process.
        # Keep each Rayon pool from claiming all cores on the shared node.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        directory = Path(args.sft_local_text_dir)
        prepared = getattr(args, "sft_local_prepared_batches", None)
        self.prepared = Path(prepared) if prepared else None
        if self.prepared:
            self.prepared_dp = json.loads((self.prepared / "batch-manifest.json").read_text())["dp"]
        self.order = np.memmap(directory / f"epoch0-seed{args.seed}.u32", mode="r", dtype="<u4")
        self.dataset = SFTStreamingDataset(
            str(directory / "empty.jsonl"),
            tokenizer=tokenizer,
            capacity=args.max_tokens_per_gpu,
            prompt_key=args.input_key,
            metadata_key=args.metadata_key,
            tool_key=getattr(args, "tool_key", None),
            pad_token_ids=_resolve_pad_token_ids_from_config(args.hf_checkpoint),
            oversize_strategy=args.sft_oversize_strategy,
            invalid_multimodal_strategy=args.sft_invalid_multimodal_strategy,
            prefetch_max_cached=0,
        )
        self.dataset.reader = _IndexedJSONL(directory)
        self.global_batch_size = args.global_batch_size
        logger.info("Local text SFT input enabled: %s rows, memory-mapped source and order", len(self.order))

    def load(self, step: int, dp_rank: int, dp_size: int) -> dict[str, Any]:
        from relax.engine.sft.batched_text_encode import encode_text_batch
        from relax.engine.sft.dataset.streaming import pack_samples_for_tq

        local_size, rest = divmod(self.global_batch_size, dp_size)
        if rest:
            raise ValueError("Local SFT requires global batch size divisible by DP size.")
        start = step * self.global_batch_size
        stop = start + self.global_batch_size
        batch_ids = self.order[start:stop]
        if len(batch_ids) != self.global_batch_size:
            raise IndexError(f"Local text data exhausted before full batch for step {step}.")
        if dp_rank == 0 and step < 32:
            boundary = hashlib.sha256(json.dumps(batch_ids.tolist()).encode()).hexdigest()
            logger.info("Local text data boundary next_step=%s source_ids_sha256=%s", step + 1, boundary)
        if self.prepared:
            return self._load_prepared(step, dp_rank, dp_size, local_size)
        shard_ids = batch_ids[dp_rank * local_size : (dp_rank + 1) * local_size]
        samples = encode_text_batch(self.dataset, shard_ids)
        if any(s is None or s.multimodal_train_inputs is not None for s in samples):
            raise ValueError(
                "The node-local text route requires all rows to be valid pure text under the active policy."
            )
        return pack_samples_for_tq(samples)

    def _load_prepared(self, step: int, dp_rank: int, dp_size: int, local_size: int) -> dict[str, Any]:
        import torch

        if dp_size != self.prepared_dp:
            raise ValueError("Prepared batches were balanced for a different DP size.")
        batch_file = self.prepared / f"{step:06d}.npz"
        deadline = monotonic() + 180
        while not batch_file.is_file():
            if (self.prepared / "failed").exists() or monotonic() > deadline:
                raise RuntimeError(f"Local batch producer did not publish step {step}.")
            sleep(0.01)
        with np.load(batch_file, allow_pickle=False) as data:
            lengths = data["lengths"][dp_rank].tolist()
            offsets = data["token_offsets"]
            start, end = offsets[dp_rank : dp_rank + 2]
            tokens = data["tokens"][start:end].astype(np.int64)
            masks = data["masks"][start:end].astype(np.int64)
        if len(lengths) != local_size or sum(lengths) != len(tokens) or len(tokens) != len(masks):
            raise ValueError(f"Invalid prepared batch geometry at optimizer step {step}.")
        return {
            "tokens": torch.from_numpy(tokens).split(lengths),
            "loss_masks": torch.from_numpy(masks).split(lengths),
            "total_lengths": lengths,
            "response_lengths": lengths,
        }
