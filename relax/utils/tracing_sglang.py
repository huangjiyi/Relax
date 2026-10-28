# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Configure SGLang's native OTLP spans without taking its global provider."""

import os

from relax.utils.env import Envs
from relax.utils.tracing import enabled


def configure(server_args: dict) -> dict:
    minimum_level = 2 if Envs.RELAX_OTEL_SGLANG_LIFECYCLE else 3
    if not enabled() or Envs.RELAX_OTEL_LEVEL < minimum_level or Envs.RELAX_OTEL_EXPORTER not in ("otlp", "both"):
        return server_args
    endpoint = Envs.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
    if not endpoint:
        raise ValueError("Native SGLang tracing requires an OTLP traces endpoint")
    # Resource.create in each native subprocess merges these attributes.
    attributes = Envs.OTEL_RESOURCE_ATTRIBUTES
    os.environ["SGLANG_TRACE_LEVEL"] = str(Envs.SGLANG_TRACE_LEVEL)
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(
        filter(
            None,
            (
                attributes,
                f"relax.run_id={Envs.RELAX_OTEL_RUN_ID}",
                "relax.role=rollout",
                "relax.timing=host",
                "relax.level=3",
            ),
        )
    )
    return {**server_args, "enable_trace": True, "otlp_traces_endpoint": endpoint}
