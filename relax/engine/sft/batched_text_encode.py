# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Batch the native Qwen template for ordinary plain-text SFT conversations."""

from itertools import chain

import torch

from relax.engine.sft.dataset.chat_template import (
    HAS_GENERATION_MARKER,
    _prepare_chat_messages,
    _resolve_sft_template_kwargs,
    _thread_local_chat_template_kwargs,
    is_qwen38_chat_template,
    raw_text_concat_enabled,
)
from relax.engine.sft.dataset.streaming import ProcessedSample


def encode_text_batch(dataset, indices):
    tokenizer = dataset.tokenizer
    groups = {}
    samples = [None] * len(indices)
    for position, index in enumerate(indices):
        index = int(index)
        row = dataset.reader[index]
        metadata = row.get(dataset.metadata_key)
        if raw_text_concat_enabled() or metadata:
            samples[position] = dataset._process_one(index)
            continue
        sample = dataset.get_canonical_sample(index, row=row)
        result = _resolve_sft_template_kwargs(sample, tokenizer=tokenizer, apply_chat_template_kwargs=None)
        if (
            result.patch_name != "qwen_history_thinking"
            or not HAS_GENERATION_MARKER(result.template)
            or is_qwen38_chat_template(result.template)
            or sample.tools is not None
            or sample.images
            or sample.videos
            or sample.audios
            or set(result.kwargs) - {"chat_template", "preserve_thinking"}
            or any(
                (m.role not in ("system", "user", "assistant"))
                or (m.role == "assistant" and (not m.learn or m.tool_calls))
                for m in sample.messages
            )
        ):
            samples[position] = dataset._process_one(index)
            continue
        key = (result.template, result.kwargs.get("preserve_thinking"))
        messages, _ = _prepare_chat_messages(sample, result.kwargs, last_turn_only=False)
        groups.setdefault(key, []).append((position, index, messages))

    for (template, preserve), rows in groups.items():
        kw = {"chat_template": template}
        if preserve is not None:
            kw["preserve_thinking"] = preserve
        kw = _thread_local_chat_template_kwargs(tokenizer, kw)
        batch = tokenizer.apply_chat_template(
            [row[2] for row in rows],
            tools=None,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **kw,
        )
        tokens, masks = batch["input_ids"], batch["assistant_masks"]
        lengths = [len(row) for row in tokens]
        if len(lengths) != len(rows) or [len(row) for row in masks] != lengths:
            raise RuntimeError("Batched Qwen tokenizer returned incomplete tokens or loss masks.")
        token_tensors = torch.tensor(list(chain.from_iterable(tokens)), dtype=torch.long).split(lengths)
        mask_tensors = torch.tensor(list(chain.from_iterable(masks)), dtype=torch.long).split(lengths)
        for (position, index, _), ids, mask, length in zip(rows, token_tensors, mask_tensors, lengths, strict=True):
            if length > dataset.capacity:
                samples[position] = dataset._process_one(index)
                continue
            samples[position] = ProcessedSample(
                tokens=ids,
                loss_mask=mask,
                total_length=length,
                multimodal_train_inputs=None,
                source_idx=index,
            )
    return samples
