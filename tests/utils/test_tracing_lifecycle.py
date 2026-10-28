# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise actual native wrappers and RPC dispatch without GPUs or SGLang imports."""

import ast
import contextvars
import functools
import os
import random
import re
import secrets
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, id_generator
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from relax.utils import tracing
from relax.utils import tracing_sglang_lifecycle as native_lifecycle
from relax.utils.tracing_descriptions import DESCRIPTION_ATTRIBUTE, NATIVE_DESCRIPTIONS, description_processor
from tests.utils.test_tracing_async_phases import records as _records
from tests.utils.test_tracing_async_phases import source_class


records = _records
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    monkeypatch.setenv("RELAX_OTEL_LEVEL", "2")
    monkeypatch.setenv("RELAX_OTEL_SGLANG_LIFECYCLE", "1")
    exporter = InMemorySpanExporter()
    module = ModuleType("test_native_overlay")
    module.__dict__.update(os=os, sys=sys, functools=functools, id_generator=id_generator,
                           random=random, secrets=secrets, time=time)
    module.threads_info = {threading.get_native_id(): SimpleNamespace(
        host_id="test-host", pid=threading.get_native_id(), thread_label="Scheduler", tp_rank=2, pp_rank=0, dp_rank=0,
    )}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    file = ROOT / "patches/ettr/sglang-trace.py"
    tree = ast.parse(file.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {
        "trace_lifecycle", "trace_weight_update", "trace_weight_resume", "TraceCustomIdGenerator",
    }]
    exec(compile(tree, str(file), "exec"), module.__dict__)
    provider = TracerProvider(shutdown_on_exit=False, id_generator=module.TraceCustomIdGenerator())
    provider.add_span_processor(description_processor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    module.tracer = provider.get_tracer("sglang-test")
    yield module, exporter
    native_lifecycle.close_pending()
    native_lifecycle._updates.clear()
    provider.shutdown()


def receiver(module):
    class Updater:
        @module.trace_weight_update
        def update(self, request):
            self.current = trace.get_current_span().get_span_context()
            if getattr(request, "raise_error", False):
                raise ValueError("receiver failed")
            return SimpleNamespace(success=getattr(request, "success", True))

    class Scheduler:
        def __init__(self):
            self.weight_updater = Updater()

        @module.trace_weight_resume
        def resume(self, fail=False):
            if fail:
                raise RuntimeError("resume failed")
            return "resumed"

    return Scheduler()


def request(parent, version="1", **kwargs):
    carrier = {}
    TraceContextTextMapPropagator().inject(carrier, context=trace.set_span_in_context(parent))
    carrier["relax-step"] = "7"
    return SimpleNamespace(relax_trace_context=carrier, weight_version=version, **kwargs)


def test_buckets_merge_into_receiver_versions_without_leaking_context(native):
    module, exporter = native
    scheduler = receiver(module)
    rng = random.getstate()
    with module.tracer.start_as_current_span("weights.transfer_apply") as parent:
        req = request(parent)
        for _ in range(4):
            assert scheduler.weight_updater.update(req).success
            assert trace.get_current_span() is parent
        assert not exporter.get_finished_spans()
        assert scheduler.resume() == "resumed"
        scheduler.weight_updater.update(request(parent, "2"))
        scheduler.resume()
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["rollout.weights.receive_apply"] * 2 + ["weights.transfer_apply"]
    for span, version, buckets in zip(spans[:2], ("1", "2"), (4, 1), strict=True):
        assert span.parent.span_id == parent.get_span_context().span_id
        assert span.attributes["relax.weights.version"] == version
        assert span.attributes["relax.weights.bucket_count"] == buckets
        assert span.attributes["relax.weights.apply_host_ns"] > 0
        assert span.attributes["relax.weights.end_reason"] == "resume"
        assert span.attributes["relax.step"] == 7
        assert span.attributes["tp_rank"] == 2
        assert span.attributes["relax.level"] == 2
        assert "Includes inter-bucket waits" in span.attributes[DESCRIPTION_ATTRIBUTE]
        assert span.attributes["code.function.name"].endswith("Updater.update")
        assert parent.start_time <= span.start_time <= span.end_time <= parent.end_time
    assert not native_lifecycle._updates
    assert random.getstate() == rng


@pytest.mark.parametrize("failure", ["raise", "returned", "resume", "superseded", "shutdown"])
def test_incomplete_receiver_update_is_error(native, failure):
    module, exporter = native
    scheduler = receiver(module)
    with module.tracer.start_as_current_span("weights.transfer_apply") as parent:
        if failure == "raise":
            with pytest.raises(ValueError, match="receiver failed"):
                scheduler.weight_updater.update(request(parent, raise_error=True))
        elif failure == "returned":
            assert not scheduler.weight_updater.update(request(parent, success=False)).success
        else:
            scheduler.weight_updater.update(request(parent))
            if failure == "resume":
                with pytest.raises(RuntimeError, match="resume failed"):
                    scheduler.resume(fail=True)
            elif failure == "superseded":
                scheduler.weight_updater.update(request(parent, "2"))
                scheduler.resume()
            else:
                native_lifecycle.close_pending()
    failed = exporter.get_finished_spans()[0]
    assert failed.name == "rollout.weights.receive_apply"
    assert failed.status.status_code == StatusCode.ERROR
    assert failed.attributes["relax.weights.bucket_count"] == 1
    assert failed.attributes["relax.weights.apply_host_ns"] > 0
    assert not native_lifecycle._updates


def test_native_init_nested_context_and_actual_source(native, monkeypatch):
    module, exporter = native
    parent = module.tracer.start_span("rollout.init.server_launch")
    carrier = request(parent).relax_trace_context
    monkeypatch.setenv("RELAX_OTEL_TRACEPARENT", carrier["traceparent"])

    @module.trace_lifecycle("sglang.init.load_model")
    def load():
        return "loaded"

    @module.trace_lifecycle("sglang.init.scheduler")
    def initialize():
        return load()

    assert initialize() == "loaded"
    parent.end()
    load_span, init_span, _ = exporter.get_finished_spans()
    assert load_span.parent.span_id == init_span.context.span_id
    assert init_span.parent.span_id == parent.context.span_id
    assert load_span.attributes["code.file.path"] == __file__
    assert load_span.attributes["code.function.name"].endswith("load")
    assert not trace.get_current_span().get_span_context().is_valid


def test_unsampled_and_untraced_requests_do_not_create_receiver_spans(native):
    module, exporter = native
    scheduler = receiver(module)
    with module.tracer.start_as_current_span("weights.transfer_apply") as parent:
        req = request(parent)
        req.relax_trace_context["traceparent"] = req.relax_trace_context["traceparent"][:-2] + "00"
        scheduler.weight_updater.update(req)
        scheduler.weight_updater.update(SimpleNamespace(weight_version="2"))
        scheduler.resume()
    assert [s.name for s in exporter.get_finished_spans()] == ["weights.transfer_apply"]
    assert not native_lifecycle._updates


def test_startup_boundary_context_and_continuous_weight_control(records, monkeypatch):
    monkeypatch.setenv("RELAX_OTEL_LEVEL", "2")
    monkeypatch.setenv("RELAX_OTEL_SGLANG_LIFECYCLE", "1")
    monkeypatch.setenv("RELAX_OTEL_TRACEPARENT", "original")
    runtime = {"env_vars": {"X": "1"}}
    with tracing.stage("service.deploy", **{"relax.step": 7}):
        with ThreadPoolExecutor(max_workers=1) as pool:
            env = pool.submit(contextvars.copy_context().run, tracing.child_runtime_environment, runtime).result()
        assert env is not runtime and env["env_vars"] is not runtime["env_vars"]
        assert runtime == {"env_vars": {"X": "1"}}
        assert env["env_vars"]["RELAX_OTEL_TRACEPARENT"] == tracing.capture_context()["traceparent"]
        with pytest.raises(ValueError):
            with tracing.subprocess_context():
                assert os.environ["RELAX_OTEL_TRACEPARENT"] == tracing.capture_context()["traceparent"]
                raise ValueError("spawn error")
        assert os.environ["RELAX_OTEL_TRACEPARENT"] == "original"
        control = tracing.weight_context()["relax_trace_context"]
        assert control["relax-step"] == "7"
        assert int(control["traceparent"][-2:], 16) & 1
        assert not int(tracing.request_context()["traceparent"][-2:], 16) & 1


def test_actual_dcs_dispatch_adds_only_weight_control_context(records, monkeypatch):
    monkeypatch.setenv("RELAX_OTEL_SGLANG_LIFECYCLE", "1")
    calls = []
    engine = SimpleNamespace(make_request=SimpleNamespace(remote=lambda endpoint, payload: calls.append(
        (endpoint, payload)
    )))
    backend = source_class(
        "relax/distributed/checkpoint_service/backends/device_direct.py", "DeviceDirectBackend", {"_batch_request"},
        {"weight_context": tracing.weight_context},
    )()
    backend.rollout_engines = {"0": engine}
    payload = {"weight_version": "3", "names": ["weight"]}
    with tracing.stage("weights.transfer_apply", **{"relax.step": 8}):
        backend._batch_request("/update_weights_from_distributed", payload)
        backend._batch_request("/continue_generation")
    assert "relax_trace_context" not in payload
    assert calls[0][1]["relax_trace_context"]["relax-step"] == "8"
    assert calls[1] == ("/continue_generation", None)
    monkeypatch.setenv("RELAX_OTEL_SGLANG_LIFECYCLE", "0")
    backend._batch_request("/update_weights_from_distributed", payload)
    assert calls[-1][1] is payload
    assert tracing.weight_context() == {}


def test_native_init_patch_names_are_documented():
    patch = (ROOT / "patches/ettr/sglang-v0.5.17-lifecycle.patch").read_text()
    names = set(re.findall(r'\+\s+@trace_lifecycle\("([^"]+)"\)', patch))
    assert len(names) == 12
    assert names <= NATIVE_DESCRIPTIONS.keys()


@pytest.mark.parametrize("transport", ["distributed", "tensor"])
def test_patched_native_handlers_with_cpu_stubs(native, transport):
    overlay = os.environ.get("ETTR_NATIVE_LIFECYCLE_OVERLAY")
    if not overlay:
        pytest.skip("Supply a hash-verified private native overlay to test the native handler bodies")
    module, exporter = native
    root = Path(overlay) / "sglang/srt"
    method = "update_weights_from_" + transport
    events = []
    scope = {
        "trace_weight_update": module.trace_weight_update,
        "trace_weight_resume": module.trace_weight_resume,
        "UpdateWeightsFromDistributedReqOutput": SimpleNamespace,
        "UpdateWeightsFromTensorReqOutput": SimpleNamespace,
        "torch": SimpleNamespace(distributed=SimpleNamespace(barrier=lambda **kw: events.append("barrier"))),
        "DisaggregationMode": SimpleNamespace(DECODE="decode"),
    }
    updater = source_class(
        str(root / "managers/scheduler_components/weight_updater.py"),
        "SchedulerWeightUpdaterManager", {method}, scope,
    )()
    updater._observe_weight_load = lambda mode: nullcontext()
    updater.tp_worker = SimpleNamespace(**{method: lambda req: (True, "applied")})
    updater.tp_cpu_group = "cpu-test-group"
    updater.flush_cache_after_weight_update = lambda req: events.append("flush")
    scheduler = source_class(str(root / "managers/scheduler.py"), "Scheduler", {"continue_generation"}, scope)()
    scheduler.weight_updater = updater
    scheduler.disaggregation_mode = "none"
    scheduler._engine_paused = True
    with module.tracer.start_as_current_span("weights.transfer_apply") as parent:
        for _ in range(3):
            result = getattr(updater, method)(request(parent, disable_draft_model=True))
            assert result.success and result.message == "applied"
        scheduler.continue_generation(SimpleNamespace(torch_empty_cache=False))
    assert scheduler._engine_paused is False
    assert events == (["flush"] if transport == "distributed" else ["flush", "barrier"]) * 3
    spans = exporter.get_finished_spans()
    assert len(spans) == 2 and spans[0].attributes["relax.weights.bucket_count"] == 3
    assert spans[0].attributes["code.function.name"].endswith(method)
    assert spans[0].parent.span_id == parent.context.span_id


def test_disabled_helpers_do_not_import_sdk_or_change_calls():
    import subprocess

    code = '''
import sys
from relax.utils import tracing
from relax.utils.tracing_sglang_lifecycle import lifecycle, weight_update
from types import SimpleNamespace
assert tracing.weight_context() == {}
runtime = {"env_vars": {"X": "1"}}
assert tracing.child_runtime_environment(runtime) is runtime
with tracing.subprocess_context(), lifecycle(None, "unused", None), weight_update(None, None, None, None):
    pass
assert not any(n.startswith("opentelemetry") for n in sys.modules)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                            env={**os.environ, "RELAX_OTEL_ENABLED": "0"})
    assert result.returncode == 0, result.stderr
