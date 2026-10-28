# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Collect run allocation and worker hardware without changing execution state."""

import random
import sys
from types import SimpleNamespace as NS

import pytest

from relax.utils import tracing, tracing_topology
from tests.utils.test_tracing_compact import records as _records


records = _records


@pytest.mark.parametrize("mode,roles,shared,total", [
    ("colocate", {"actor": 8, "rollout": 8}, ["actor", "rollout"], 8),
    ("async", {"actor": 8, "rollout": 8}, [], 16),
    ("hybrid", {"actor": 8, "rollout": 8}, [], 16),
    ("colocate", {"actor": 8, "rollout": 8, "critic": 8, "reference": 4},
     ["actor", "rollout", "critic"], 12),
])
def test_run_allocation_is_exported_once_without_inheritance(records, mode, roles, shared, total):
    before = random.getstate()
    tracing.start_run()
    config = NS(fully_async=mode in ("async", "hybrid"), hybrid=mode == "hybrid")
    with tracing.stage("controller.init"):
        tracing.capture_run_configuration(config, roles, shared)
        with tracing.stage("service.deploy"):
            pass
    tracing.finish_run()
    spans = records()
    root = next(s for s in spans if s["name"] == "job.run")
    assert root["attributes"]["relax.run.execution_mode"] == mode
    assert root["attributes"]["relax.run.gpu_count"] == total
    assert root["attributes"]["relax.run.actor.gpu_count"] == 8
    assert root["attributes"]["relax.run.rollout.gpu_count"] == 8
    assert all(not any(k.startswith("relax.run.") for k in s["attributes"])
               for s in spans if s is not root)
    assert random.getstate() == before


def test_hardware_is_worker_init_only_and_native_capture_uses_same_adapter(records, monkeypatch):
    calls = []
    cuda = NS(is_initialized=lambda: True, current_device=lambda: 3,
              get_device_name=lambda device: calls.append(device) or "NVIDIA H20")
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=cuda))
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.parallel_state", NS())
    monkeypatch.setattr(tracing_topology, "read_sglang_topology", lambda _: {"relax.global_rank": 3})
    with tracing.stage("worker.init", **{"relax.role": "actor", "relax.rank": 3}):
        tracing.set_process_attributes(tracing_topology.capture_gpu_identity())
        __import__("time").sleep(.002)
        with tracing.stage("worker.init.model_optimizer"):
            pass
    with tracing.stage("train.worker"):
        pass
    spans = records()
    assert next(s for s in spans if s["name"] == "worker.init")["attributes"]["relax.gpu.model"] == "NVIDIA H20"
    assert all("relax.gpu.model" not in s["attributes"] for s in spans if s["name"] != "worker.init")
    assert tracing_topology.capture_sglang_topology()["relax.gpu.model"] == "NVIDIA H20"
    assert calls == [3, 3]


def test_no_cuda_initialization_or_probe_when_disabled(monkeypatch):
    def forbidden():
        raise AssertionError("must not probe a device")
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_initialized=lambda: False, current_device=forbidden)))
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    assert tracing_topology.capture_gpu_identity() == {}
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "0")
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_initialized=forbidden)))
    assert tracing_topology.capture_gpu_identity() == {}
    tracing.capture_run_configuration(NS(), {"actor": 8}, [])


def test_device_probe_failure_is_nonfatal(monkeypatch):
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_initialized=lambda: True)))
    assert tracing_topology.capture_gpu_identity() == {}
