# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cross-process shutdown receipts with real SDK exporters and native owners."""

import json
import os
import subprocess
import sys
import time

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from relax.utils.tracing_shutdown import track_active_spans
from relax.utils.tracing_shutdown_control import _directory, _write, finalize_run_traces
from tests.utils.test_tracing_native_scheduler import native as _native
from tests.utils.test_tracing_native_scheduler import set_thread


native = _native


@pytest.fixture
def control(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    monkeypatch.setenv("RELAX_OTEL_RUN_ID", "shutdown-test")
    monkeypatch.setenv("RELAX_OTEL_SHUTDOWN_DIR", str(tmp_path))
    return _directory()


def test_native_live_request_parents_export_before_kill(control, native):
    module, exporter = native
    # Use the actual patched TraceReqContext with a fresh instrumented provider.
    provider = TracerProvider(shutdown_on_exit=False, id_generator=module.TraceCustomIdGenerator())
    module._active_span_processor = track_active_spans(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    module.tracer = provider.get_tracer("native-test")
    set_thread(module, "Scheduler", 0)
    request = module.TraceReqContext("interrupted")
    request.trace_req_start()
    request.trace_slice_start("decode_forward", 1)
    already_finished = {s.context.span_id for s in exporter.get_finished_spans()}
    report = finalize_run_traces("SIGTERM")
    assert report["verified"]
    spans = exporter.get_finished_spans()
    ids = {s.context.span_id for s in spans}
    assert len(spans) == 1
    assert all(s.parent is None or s.parent.span_id in ids for s in spans)
    interrupted = [s for s in spans if s.context.span_id not in already_finished]
    assert len(interrupted) == 1
    assert all(s.attributes["relax.span.outcome"] == "cancelled" for s in interrupted)
    assert all(s.attributes["relax.cancel.reason"] == "SIGTERM" for s in interrupted)
    request.trace_slice_start("decode_forward", 1)
    request.trace_event("after_shutdown", 1)
    request.trace_req_finish()
    new_request = module.TraceReqContext("new-after-stop")
    new_request.trace_req_start()
    assert not new_request.copy_for_thread().tracing_enable
    assert len(exporter.get_finished_spans()) == len(spans)
    provider.shutdown()


def test_shared_request_closes_a_separate_process_before_hard_exit(control, tmp_path):
    script = r'''
import os, time
from pathlib import Path
from relax.utils import tracing
tracing.start_run()
with tracing.stage("service.run"):
    with tracing.stage("rollout.sample"):
        Path(os.environ["READY"]).touch()
        while not Path(os.environ["EXIT"]).exists():
            time.sleep(.02)
        os._exit(0)
'''
    ready, exit_file = tmp_path / "ready", tmp_path / "exit"
    env = {**os.environ, "RELAX_OTEL_ENABLED": "1", "RELAX_OTEL_EXPORTER": "file",
           "RELAX_OTEL_OUTPUT_DIR": str(tmp_path / "spans"), "READY": str(ready), "EXIT": str(exit_file)}
    env.pop("RELAX_OTEL_TRACEPARENT", None)
    child = subprocess.Popen([sys.executable, "-c", script], env=env)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert child.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.02)
        report = finalize_run_traces("SIGTERM")
        assert report["verified"]
        assert report["participants"][0]["pid"] == child.pid
        assert report["participants"][0]["ended_spans"] == 2
        records = [json.loads(line) for p in (tmp_path / "spans").glob("*.jsonl") for line in p.read_text().splitlines()]
        ids = {s["context"]["span_id"] for s in records}
        assert len(records) == 2
        assert all(s["parent_id"] is None or s["parent_id"] in ids for s in records)
        exit_file.touch()
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_missing_ack_is_failure(control):
    _write(control / "participants" / "dead.json", {"provider_id": "dead", "state": "active"})
    report = finalize_run_traces("SIGTERM", timeout=.1)
    assert not report["verified"]
    assert report["pending"][0]["provider_id"] == "dead"


def test_other_run_cannot_finalize_this_provider(control, monkeypatch):
    provider = TracerProvider(shutdown_on_exit=False)
    track_active_spans(provider)
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    span = provider.get_tracer("test").start_span("still-running")
    with monkeypatch.context() as context:
        context.setenv("RELAX_OTEL_RUN_ID", "unrelated-run")
        assert not finalize_run_traces("SIGTERM", timeout=.1)["verified"]
    assert span.is_recording()
    span.end()
    provider.shutdown()


def test_real_error_and_failed_export_are_preserved(control, monkeypatch):
    provider = TracerProvider(shutdown_on_exit=False)
    processor = track_active_spans(provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("test").start_span("failed-before-stop")
    span.set_status(Status(StatusCode.ERROR, "real OOM"))
    monkeypatch.setattr(provider, "force_flush", lambda **kwargs: False)
    report = finalize_run_traces("SIGTERM")
    assert not report["verified"]
    assert len(report["failed"]) == 1
    assert exporter.get_finished_spans()[0].status.status_code == StatusCode.ERROR
    assert not processor.active
    provider.shutdown()
