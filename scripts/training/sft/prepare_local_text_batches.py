# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Produce every batch in the original epoch order, balanced as in
TransferQueue."""

import argparse
import fcntl
import json
import multiprocessing as mp
import shutil
from itertools import islice
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np


def jsonl_offsets(source: Path):
    with source.open("rb", buffering=1024 * 1024) as stream:
        offset = 0
        for line in stream:
            if line.strip():
                yield offset
            offset += len(line)


def build_index(source: Path, directory: Path, seed: int) -> None:
    import torch

    stat = source.stat()
    identity = {"source": str(source), "source_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns, "seed": seed}
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing index belongs to another dataset or shuffle seed.")
        return

    temporary = directory / "source.jsonl.tmp"
    shutil.copyfile(source, temporary)
    copied = directory / "source.jsonl"
    temporary.replace(copied)
    offsets = jsonl_offsets(copied)
    chunk_size = 2**20
    rows = 0
    with (directory / "source-offsets.u64").open("wb") as f:
        while len(chunk := np.fromiter(islice(offsets, chunk_size), dtype="<u8")):
            chunk.tofile(f)
            rows += len(chunk)

    if not 0 < rows <= np.iinfo(np.uint32).max:
        raise ValueError("The source must fit the uint32 row index.")
    dataset_seed = int(np.random.RandomState(seed).randint(0, np.iinfo(np.int32).max))
    dataset_order = np.random.default_rng(dataset_seed).permutation(rows)
    sampler = torch.randperm(rows, generator=torch.Generator().manual_seed(0)).numpy()
    with (directory / f"epoch0-seed{seed}.u32").open("wb") as f:
        for start in range(0, rows, chunk_size):
            dataset_order[sampler[start : start + chunk_size]].astype("<u4").tofile(f)
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError("The source changed while the local cache was built.")
    (directory / "empty.jsonl").touch()
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps({**identity, "rows": rows}) + "\n")
    temporary.replace(manifest)


def encode_step(step: int) -> int:
    import torch
    from transfer_queue.sampler.seqlen_balanced_sampler import get_seqlen_balanced_partitions

    final = output / f"{step:06d}.npz"
    if final.exists():
        return step
    data = batches.load(step, 0, 1)
    lengths = data["total_lengths"]
    partitions = get_seqlen_balanced_partitions(lengths, dp_size, equal_size=True)
    layout = np.asarray(partitions, dtype=np.int32)
    expected = (dp_size, batches.global_batch_size // dp_size)
    if layout.shape != expected:
        raise RuntimeError(f"Invalid balanced layout {layout.shape}, expected {expected}")
    order = layout.ravel().tolist()
    balanced_lengths = np.asarray(lengths, dtype="<u2")[layout]
    token_offsets = np.r_[0, balanced_lengths.sum(axis=1, dtype=np.int64).cumsum()]
    tokens = torch.cat([data["tokens"][i] for i in order]).numpy().astype("<u4")
    masks = torch.cat([data["loss_masks"][i] for i in order]).numpy().astype("u1")
    if tokens.size != token_offsets[-1] or len(tokens) != len(masks):
        raise RuntimeError("Encoded steps must contain exactly one mask value per token.")
    temp = final.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez(f, lengths=balanced_lengths, token_offsets=token_offsets, tokens=tokens, masks=masks)
    temp.replace(final)
    return step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Original JSONL. Builds the index if needed.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--gbs", type=int, default=8192)
    parser.add_argument("--dp", type=int, default=8)
    parser.add_argument("--oversize-strategy", default="skip")
    parser.add_argument("--invalid-multimodal-strategy", default="skip")
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    with (options.output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(options)


def prepare(options: argparse.Namespace) -> None:
    global output, batches, dp_size

    import torch
    from transformers import AutoTokenizer

    from relax.engine.sft.local_text import LocalTextBatches

    if options.source is not None:
        build_index(options.source, options.local, options.seed)
    source_manifest = json.loads((options.local / "manifest.json").read_text())
    output = options.output
    dp_size = options.dp
    if options.gbs % dp_size:
        raise ValueError("Global batch size must be a multiple of DP size.")
    args = SimpleNamespace(
        sft_local_text_dir=str(options.local),
        seed=source_manifest["seed"],
        max_tokens_per_gpu=16384,
        input_key="messages",
        metadata_key="metadata",
        tool_key="tools",
        hf_checkpoint=options.model,
        sft_oversize_strategy=options.oversize_strategy,
        sft_invalid_multimodal_strategy=options.invalid_multimodal_strategy,
        global_batch_size=options.gbs,
    )
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(options.model, trust_remote_code=True)
    batches = LocalTextBatches(args, tokenizer)
    epoch_steps = len(batches.order) // options.gbs
    blueprint = {
        **source_manifest,
        "gbs": options.gbs,
        "dp": dp_size,
        "steps": epoch_steps,
        "model": options.model,
        "loss_mode": "original_text",
    }
    manifest = output / "batch-manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != blueprint:
        raise RuntimeError("Existing prepared batches use another input recipe.")
    temp = manifest.with_suffix(".tmp")
    temp.write_text(json.dumps(blueprint, sort_keys=True) + "\n")
    temp.replace(manifest)
    failure = output / "failed"
    if failure.exists():
        failure.unlink()
    started = monotonic()
    print(f"Preparing {epoch_steps} full-epoch batches, workers={options.workers}", flush=True)
    try:
        with mp.get_context("fork").Pool(options.workers) as pool:
            for count, step in enumerate(pool.imap_unordered(encode_step, range(epoch_steps)), 1):
                if count % 32 == 0:
                    print(f"ready batches={count} last={step} elapsed={monotonic() - started:.1f}s", flush=True)
    except BaseException as exc:
        failure.write_text(type(exc).__name__ + "\n")
        raise


if __name__ == "__main__":
    main()
