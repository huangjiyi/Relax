# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dense Qwen3 spec for Megatron runtimes without the legacy grouped GEMM
argument."""

from argparse import Namespace
from typing import Any


def build_layer_spec(args: Namespace, _config: Any, _vp_stage: int | None) -> Any:
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )

    if args.num_experts:
        raise ValueError("This layer spec is for dense Qwen3 models.")
    factory = get_gpt_layer_local_spec
    if args.transformer_impl == "transformer_engine":
        factory = get_gpt_layer_with_transformer_engine_spec
    return factory(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        multi_latent_attention=args.multi_latent_attention,
    )
