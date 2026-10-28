# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicitly opt in with ETTR_TEST_RAY_ADDRESS; uses CPU actors only."""

import asyncio
import json
import os
import random
import uuid

import pytest


@pytest.mark.skipif(not os.environ.get("ETTR_TEST_RAY_ADDRESS"), reason="Requires an explicitly selected Ray cluster")
def test_ray_carriers_preserve_distinct_parents(tmp_path, monkeypatch):
    ray = pytest.importorskip("ray")
    pytest.importorskip("opentelemetry.sdk.trace")
    from relax.utils import tracing

    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    monkeypatch.setenv("RELAX_OTEL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_OTEL_RUN_ID", uuid.uuid4().hex)
    monkeypatch.delenv("RELAX_OTEL_TRACEPARENT", raising=False)

    @ray.remote(num_cpus=0, num_gpus=0)
    class Worker:
        @tracing.traced("probe.remote", step_arg="step")
        async def work(self, step, *, _trace_context=None):
            random.seed(1234)
            with tracing.stage("probe.seeded"):
                await asyncio.sleep(0.01)
            return step

    ray.init(
        address=os.environ["ETTR_TEST_RAY_ADDRESS"],
        runtime_env={"env_vars": {**tracing.runtime_environment(), "PYTHONPATH": os.environ.get("PYTHONPATH", "")}},
    )
    workers = [Worker.remote(), Worker.remote()]
    try:
        refs = []
        with tracing.stage("probe.job"):
            for step in (0, 1):
                with tracing.stage(f"probe.parent-{step}"):
                    refs.append(workers[step].work.remote(step, **tracing.remote_context()))
            assert ray.get(refs) == [0, 1]
        records = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
        assert len({span["context"]["span_id"] for span in records}) == len(records)
        parents = {span["name"]: span for span in records if span["name"].startswith("probe.parent-")}
        children = [span for span in records if span["name"] == "probe.remote"]
        assert len(children) == 2
        assert len({span["resource"]["attributes"]["process.pid"] for span in children}) == 2
        for span in children:
            step = span["attributes"]["relax.step"]
            parent = parents[f"probe.parent-{step}"]
            assert span["parent_id"] == parent["context"]["span_id"]
            assert span["context"]["trace_id"] == parent["context"]["trace_id"]
    finally:
        for worker in workers:
            ray.kill(worker)
        ray.shutdown()
        if tracing._provider is not None:
            tracing._provider.shutdown()
            tracing._stream.close()
            tracing._provider = None
            tracing._provider_pid = None
