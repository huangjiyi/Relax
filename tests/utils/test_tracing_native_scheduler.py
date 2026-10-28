# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the native request hierarchy with real OTel and CPU-only SGLang stubs."""

import importlib.util
import json
import os
import random
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from relax.utils import tracing_topology
from relax.utils.tracing_descriptions import DESCRIPTION_ATTRIBUTE, description_processor


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    for name in ("sglang", "sglang.srt", "sglang.srt.utils", "sglang.srt.runtime_context"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["sglang.srt.utils"].get_int_env_var = lambda name, default: int(os.getenv(name, default))
    resources = SimpleNamespace(trace_level=1)
    sys.modules["sglang.srt.runtime_context"].get_resources = lambda: resources
    monkeypatch.setattr(tracing_topology, "capture_sglang_topology", lambda: {
        "relax.topology.status": "complete", "relax.topology.tp.world_size": 2,
    })
    path = Path(__file__).resolve().parents[2] / "patches/ettr/sglang-trace.py"
    spec = importlib.util.spec_from_file_location("native_scheduler_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False, resource=Resource.create({"service.name": "sglang"}),
                              id_generator=module.TraceCustomIdGenerator())
    provider.add_span_processor(description_processor())
    from relax.utils.tracing_policy import CollectionProcessor
    collection = CollectionProcessor()
    collection.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(collection)
    module.tracer = provider.get_tracer("scheduler-test")
    module.opentelemetry_initialized = True
    yield module, exporter
    provider.shutdown()


def set_thread(module, label, rank=None, pid=None):
    pid = threading.get_native_id() if pid is None else pid
    module.threads_info[pid] = module.TraceThreadInfo("test-host", pid, label, rank, None, 0 if rank is not None else None)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_scheduler_children_keep_parents_identity_links_and_times(native, tmp_path, level):
    module, exporter = native
    module.set_global_trace_level(level)
    set_thread(module, "Tokenizer")
    rng = random.getstate()
    with module.tracer.start_as_current_span("rollout.generate"):
        for sample in (6, 7):
            with module.tracer.start_as_current_span("rollout.engine_request", attributes={
                "relax.group_index": 3, "relax.sample_index": sample,
            }) as parent:
                carrier = {}
                module._trace_context_propagator.inject(carrier)
                request = module.TraceReqContext(str(sample), external_trace_header=carrier)
                request.trace_req_start(ts=100)
                request_id = parent.context.span_id
                assert request.root_span is None
                for rank in (0, 1):
                    # Exercise the same serialized context used across scheduler processes.
                    worker = module.TraceReqContext.__new__(module.TraceReqContext)
                    worker.__setstate__(request.__getstate__())
                    worker.pid = 200 + rank
                    set_thread(module, "Scheduler", rank, worker.pid)
                    worker.rebuild_thread_context(ts=110)
                    worker.trace_set_thread_attrs({"request_batch": sample})
                    worker.trace_slice_start("prefill_forward", 1, ts=120)
                    worker.trace_slice_start("nested_compute", 1, ts=121)
                    worker.trace_slice_end("nested_compute", 1, ts=122)
                    worker.trace_event("prefill_done", 1, ts=125)
                    worker.trace_slice_end("prefill_forward", 1, ts=130)
                    worker.trace_slice(module.TraceSliceContext("decode_forward", 140, 180), thread_finish_flag=True)
                    assert worker.thread_context is None
                request.trace_req_finish(ts=190)
                phases = [s for s in exporter.get_finished_spans()
                          if s.parent and s.parent.span_id == request_id and s.name.endswith("_forward")]
                assert len(phases) == 4
                for span in phases:
                    assert (span.start_time, span.end_time) == ((120, 130) if span.name == "prefill_forward" else (140, 180))
                    assert span.attributes["pid"] == 200 + span.attributes["tp_rank"]
                    assert span.attributes["thread_label"] == "Scheduler"
                    assert span.attributes["request_batch"] == sample
                    assert "relax.topology.tp.world_size" not in span.attributes
                    assert span.attributes[DESCRIPTION_ATTRIBUTE]
                    assert span.attributes["code.function.name"]
                for rank in (0, 1):
                    prefill = next(s for s in phases if s.name == "prefill_forward" and s.attributes["tp_rank"] == rank)
                    decode = next(s for s in phases if s.name == "decode_forward" and s.attributes["tp_rank"] == rank)
                    assert decode.links[0].context.span_id == prefill.context.span_id
                    assert prefill.events[0].name == "prefill_done"
                    assert any(s.name == "nested_compute" and s.parent.span_id == prefill.context.span_id
                               for s in exporter.get_finished_spans())
    spans = exporter.get_finished_spans()
    assert not any(s.name.startswith("Scheduler") for s in spans)
    assert not any(s.name.startswith("Tokenizer") or " Req " in s.name or s.name == "worker.init" for s in spans)
    assert all(s.resource.attributes["relax.topology.tp.world_size"] == 2
               for s in spans if s.name.endswith("_forward"))
    ids = {s.context.span_id for s in spans}
    assert len(ids) == len(spans)
    assert all(not s.parent or s.parent.span_id in ids for s in spans)
    assert random.getstate() == rng
    (tmp_path / "scheduler-flat.jsonl").write_text("\n".join(json.dumps(json.loads(s.to_json())) for s in spans) + "\n")


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("aborted", [False, True])
def test_flat_scheduler_retains_abort_and_pending_events(native, active, aborted):
    module, exporter = native
    set_thread(module, "Scheduler", 0)
    request = module.TraceReqContext("abort")
    request.trace_req_start(ts=100)
    if active:
        request.trace_slice_start("decode_forward", 1, ts=110)
    request.trace_event("failure_detail", 1, ts=120, attrs={"reason": "failed"})
    request.abort(ts=130, abort_info={"reason": "request failed"} if aborted else None)
    request.trace_req_finish(ts=140)
    spans = exporter.get_finished_spans()
    errors = [s for s in spans if s.status.status_code == StatusCode.ERROR]
    # Genuine failures also mark the owning Req; a normal thread end does not.
    assert len(errors) == int(aborted)
    name = "decode_forward" if active else "scheduler.abort" if aborted else "scheduler.events"
    diagnostic = next(s for s in spans if s.name == name)
    if aborted:
        assert diagnostic.attributes["reason"] == "request failed"
    assert diagnostic.events[0].name == "failure_detail"
    assert diagnostic.end_time == 130
    if not active:
        assert diagnostic.start_time == 130


def test_scheduler_wrapper_is_unchanged_outside_relax(native, monkeypatch):
    module, exporter = native
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "0")
    set_thread(module, "Scheduler", 0)
    request = module.TraceReqContext("native")
    request.trace_req_start()
    request.trace_slice_start("prefill_forward", 1)
    request.trace_slice_end("prefill_forward", 1)
    request.trace_req_finish()
    spans = exporter.get_finished_spans()
    wrapper = next(s for s in spans if s.name.startswith("Scheduler"))
    assert next(s for s in spans if s.name == "prefill_forward").parent.span_id == wrapper.context.span_id


def test_unsampled_scheduler_emits_nothing(native):
    module, exporter = native
    set_thread(module, "Scheduler", 0)
    request = module.TraceReqContext("unsampled", external_trace_header={
        "traceparent": "00-12345678901234567890123456789012-1234567890123456-00",
    })
    request.trace_req_start()
    request.trace_slice_start("prefill_forward", 1)
    request.trace_slice_end("prefill_forward", 1, thread_finish_flag=True)
    request.trace_req_finish()
    assert not exporter.get_finished_spans()
    assert not module._worker_metadata_emitted


def test_waiting_slices_are_absent_and_still_finish_thread(native):
    module, exporter = native
    set_thread(module, "Scheduler", 0)
    request = module.TraceReqContext("waiting")
    request.trace_req_start(ts=1)
    request.trace_slice_start("prefill_waiting", 1, ts=2)
    request.trace_slice_end("prefill_waiting", 1, ts=3)
    request.trace_slice_start("prefill_forward", 1, ts=4)
    request.trace_slice_end("prefill_forward", 1, ts=5)
    request.trace_slice(module.TraceSliceContext("decode_waiting", 6, 7), thread_finish_flag=True)
    assert request.thread_context is None
    request.trace_req_finish(ts=8)
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["prefill_forward"]
    assert spans[0].parent is None
