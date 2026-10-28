# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read existing Megatron communication groups without running collectives."""

import json
import sys
from typing import Any

from relax.utils import tracing


GROUPS = (
    ("pp", "get_pipeline_model_parallel_group", {}),
    ("dp", "get_data_parallel_group", {"with_context_parallel": False}),
    ("cp", "get_context_parallel_group", {}),
    ("ep", "get_expert_model_parallel_group", {}),
    ("tp", "get_tensor_model_parallel_group", {}),
)


def _group_metadata(dist: Any, group: Any, global_rank: int) -> dict:
    # None means WORLD to torch.distributed, not an absent/singleton group.
    if group is None:
        raise ValueError("group unavailable")
    rank = int(dist.get_rank(group=group))
    size = int(dist.get_world_size(group=group))
    if size < 1 or rank < 0 or rank >= size:
        raise ValueError("rank outside group")
    result = {"rank": rank, "size": size}
    # On older Torch, retain the measured rank/size if member lookup is absent.
    if hasattr(dist, "get_process_group_ranks"):
        members = [int(value) for value in dist.get_process_group_ranks(group)]
        if len(members) != size or len(set(members)) != size or members[rank] != global_rank:
            raise ValueError("group membership inconsistent")
        result["ranks"] = members
    return result


def read_megatron_topology(dist: Any, parallel_state: Any) -> dict:
    """Injectable adapter; no rank arithmetic or dependency on group creation order."""
    if not dist.is_available() or not dist.is_initialized():
        return {}
    if not parallel_state.model_parallel_is_initialized():
        return {}
    global_rank = int(dist.get_rank())
    attributes = {
        "relax.global_rank": global_rank,
        "relax.world_size": int(dist.get_world_size()),
        "relax.topology.schema_version": 1,
        "relax.topology.source": "megatron_process_groups",
        "relax.topology.rank_base": 0,
        "relax.dp_scope": "without_context_parallel",
    }
    complete = True
    for axis, name, kwargs in GROUPS:
        try:
            group = getattr(parallel_state, name)(**kwargs)
            groups = group if isinstance(group, (list, tuple)) else [group]
            if not groups:
                raise ValueError("no groups")
            metadata = [_group_metadata(dist, item, global_rank) for item in groups]
            if len(metadata) == 1:
                for field, value in metadata[0].items():
                    attributes[f"relax.{axis}_{field}"] = value
            else:
                # A process may belong to multiple physical PP groups. Preserve
                # all of them instead of selecting an arbitrary group-local rank.
                attributes[f"relax.{axis}_groups"] = json.dumps(metadata, separators=(",", ":"))
            attributes[f"relax.{axis}_status"] = "collected"
        except Exception as error:
            complete = False
            attributes[f"relax.{axis}_status"] = f"unavailable:{type(error).__name__}"
    attributes["relax.topology.status"] = "complete" if complete else "partial"
    return attributes


def capture_megatron_topology() -> dict:
    """Best effort; disabled mode imports neither Megatron, Torch, nor OTel."""
    if not tracing.enabled():
        return {}
    try:
        import torch.distributed as dist
        from megatron.core import parallel_state

        attributes = read_megatron_topology(dist, parallel_state)
        attributes.update(capture_gpu_identity())
        if attributes:
            tracing.set_process_attributes(attributes)
        return attributes
    except Exception as error:
        tracing.logger.warning(f"OTel topology capture unavailable ({type(error).__name__})")
        return {}


def read_sglang_topology(parallel_state: Any) -> dict:
    """Read SGLang GroupCoordinator metadata, not CUDA-device local_rank."""
    attributes = {
        "relax.topology.schema_version": 1,
        "relax.topology.source": "sglang_group_coordinator",
        "relax.topology.rank_base": 0,
        "relax.dp_scope": "moe_data_parallel",
        "relax.cp_scope": "attention_context_parallel",
    }
    complete = True
    for axis, name in (("pp", "get_pp_group"), ("dp", "get_moe_dp_group"),
                       ("cp", "get_attn_cp_group"), ("ep", "get_moe_ep_group"),
                       ("tp", "get_tp_group")):
        try:
            group = getattr(parallel_state, name)()
            rank, size = int(group.rank_in_group), int(group.world_size)
            members, global_rank = [int(value) for value in group.ranks], int(group.rank)
            if size < 1 or rank < 0 or rank >= size or len(members) != size or len(set(members)) != size or members[rank] != global_rank:
                raise ValueError("group membership inconsistent")
            if "relax.global_rank" in attributes and attributes["relax.global_rank"] != global_rank:
                raise ValueError("global rank differs between coordinators")
            attributes.update({"relax.global_rank": global_rank, f"relax.{axis}_rank": rank,
                               f"relax.{axis}_size": size, f"relax.{axis}_ranks": members,
                               f"relax.{axis}_status": "collected"})
        except Exception as error:
            complete = False
            attributes[f"relax.{axis}_status"] = f"unavailable:{type(error).__name__}"
    attributes["relax.topology.status"] = "complete" if complete else "partial"
    return attributes


def capture_sglang_topology() -> dict:
    """Native thread spans own export; do not create another Relax provider."""
    if not tracing.enabled():
        return {}
    try:
        # Inference initializes this module before requests. Never import a
        # distributed runtime just to trace a Tokenizer/probe with no groups.
        parallel_state = sys.modules.get("sglang.srt.distributed.parallel_state")
        if parallel_state is None:
            return {}
        return {**read_sglang_topology(parallel_state), **capture_gpu_identity()}
    except Exception:
        return {}


def capture_gpu_identity() -> dict:
    """Read the worker's already initialized CUDA device; never initialize CUDA for tracing."""
    if not tracing.enabled():
        return {}
    try:
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            return {}
        return {
            "relax.gpu.model": torch.cuda.get_device_name(torch.cuda.current_device()),
            "relax.gpu.model_source": "torch.cuda.get_device_name",
        }
    except Exception:
        return {}
