# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import contextvars
import inspect
import json
import random
import subprocess
import sys

import pytest

from relax.utils import tracing


@pytest.fixture
def output(monkeypatch, tmp_path):
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    monkeypatch.setenv("RELAX_OTEL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_OTEL_RUN_ID", "test")
    monkeypatch.setenv("RELAX_OTEL_EXPORTER", "file")
    monkeypatch.setenv("RELAX_OTEL_LEVEL", "1")
    monkeypatch.delenv("RELAX_OTEL_DETAIL_STEPS", raising=False)
    monkeypatch.delenv("RELAX_OTEL_TRACEPARENT", raising=False)
    yield tmp_path
    tracing.finish_run()
    if tracing._provider is not None:
        tracing._provider.shutdown()
    if tracing._stream is not None:
        tracing._stream.close()
    tracing._provider = None
    tracing._provider_pid = None
    tracing._root_span = None
    tracing._initialization_failed = False


def spans(output):
    return [json.loads(line) for path in output.glob("*.jsonl") for line in path.read_text().splitlines()]


def test_disabled_does_not_import_sdk_or_change_call_result():
    code = """
import builtins, os
os.environ['RELAX_OTEL_ENABLED'] = '0'
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('opentelemetry'):
        raise AssertionError('disabled tracing imported OpenTelemetry')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from relax.utils.tracing import traced, stage, capture_context
@traced('unused')
def work(value):
    return value + 1
with stage('unused'):
    assert work(3) == 4
assert capture_context() == {}
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_exception_is_preserved_and_context_is_restored(output):
    error = ValueError("business failure")
    with tracing.stage("parent", **{"relax.role": "actor"}):
        with pytest.raises(ValueError) as caught:
            with tracing.stage("child", **{"relax.step": 7}):
                raise error
        assert caught.value is error
        with tracing.stage("sibling"):
            pass
    records = {span["name"]: span for span in spans(output)}
    assert records["child"]["status"]["status_code"] == "ERROR"
    assert records["child"]["parent_id"] == records["parent"]["context"]["span_id"]
    assert records["sibling"]["parent_id"] == records["parent"]["context"]["span_id"]
    assert records["sibling"]["attributes"].get("relax.step") is None


async def test_async_carriers_remain_isolated_and_cover_actual_await(output):
    @tracing.traced("remote", step_arg="step")
    async def work(step, *, _trace_context=None):
        with tracing.stage("nested"):
            await asyncio.sleep(0.01)
        return step

    assert inspect.iscoroutinefunction(work)
    assert "_trace_context" in inspect.signature(work).parameters
    tasks = []
    for step in (0, 1):
        with tracing.stage(f"parent-{step}"):
            carrier = tracing.capture_context()
            # A new context simulates a receiving process with no active span.
            tasks.append(contextvars.Context().run(asyncio.create_task, work(step, _trace_context=carrier)))
    assert await asyncio.gather(*tasks) == [0, 1]
    records = spans(output)
    parents = {span["name"]: span for span in records if span["name"].startswith("parent-")}
    for span in (span for span in records if span["name"] == "remote"):
        step = span["attributes"]["relax.step"]
        assert span["parent_id"] == parents[f"parent-{step}"]["context"]["span_id"]
        nested = next(item for item in records if item["parent_id"] == span["context"]["span_id"])
        assert span["start_time"] <= nested["start_time"] <= nested["end_time"] <= span["end_time"]


async def test_cancellation_ends_span_and_does_not_leak_context(output):
    started = asyncio.Event()

    @tracing.traced("cancelled")
    async def work():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(work())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with tracing.stage("after"):
        pass
    records = {span["name"]: span for span in spans(output)}
    assert records["cancelled"]["status"]["status_code"] == "UNSET"
    assert records["cancelled"]["attributes"]["relax.span.outcome"] == "cancelled"
    assert records["after"]["parent_id"] is None


def test_run_context_reaches_process_fallback_and_finishes_with_error(output, monkeypatch):
    tracing.start_run()
    carrier = tracing.capture_context()
    assert carrier["traceparent"] == tracing.runtime_environment()["RELAX_OTEL_TRACEPARENT"]
    with tracing.stage("worker"):
        pass
    tracing.finish_run(error=True)
    records = {span["name"]: span for span in spans(output)}
    assert records["worker"]["parent_id"] == records["job.run"]["context"]["span_id"]
    assert records["job.run"]["status"]["status_code"] == "ERROR"


async def test_component_thread_copies_parent_context(output):
    from relax.components.background_loop import BackgroundLoopRunner

    def work():
        with tracing.stage("thread"):
            pass

    with tracing.stage("caller"):
        await BackgroundLoopRunner().run(work)
    records = {span["name"]: span for span in spans(output)}
    assert records["thread"]["parent_id"] == records["caller"]["context"]["span_id"]


def test_feature_settings_use_the_environment_registry(monkeypatch):
    from relax.utils.env import Envs, known_env_names

    assert set(tracing._ENV_NAMES) <= known_env_names()
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "on")
    assert tracing.enabled()
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "off")
    assert not tracing.enabled()
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "ture")
    with pytest.raises(ValueError, match="RELAX_OTEL_ENABLED"):
        tracing.enabled()
    monkeypatch.setenv("RELAX_OTEL_TRACEPARENT", "carrier")
    assert Envs.RELAX_OTEL_TRACEPARENT == "carrier"


def test_spans_are_unique_under_reseeding_and_preserve_training_rng(output):
    original = random.getstate()
    try:
        random.seed(1234)
        expected = random.getstate()
        with tracing.stage("parent"):
            assert random.getstate() == expected
            for name in ("child-a", "child-b"):
                random.seed(1234)
                with tracing.stage(name):
                    pass
                assert random.getstate() == expected
        records = spans(output)
        assert len({span["context"]["span_id"] for span in records}) == len(records)
        assert len({span["context"]["trace_id"] for span in records}) == 1
    finally:
        random.setstate(original)


def test_detail_window_and_parent_flags_preserve_routing_headers(output, monkeypatch):
    monkeypatch.setenv("RELAX_OTEL_LEVEL", "3")
    monkeypatch.setenv("RELAX_OTEL_DETAIL_STEPS", "1,3-4")
    headers = {"X-SMG-Routing-Key": "session"}
    for step in range(6):
        with tracing.stage("step", **{"relax.step": step}):
            with tracing.stage("phase", level=2):
                pass
            with tracing.stage("detail", level=3):
                carrier = tracing.request_context(headers)
                assert carrier["X-SMG-Routing-Key"] == "session"
                assert bool(int(carrier["traceparent"][-2:], 16) & 1) == (step in (1, 3, 4))
    details = [span for span in spans(output) if span["name"] == "detail"]
    assert [s["attributes"]["relax.step"] for s in details] == [1, 3, 4]
    assert all(s["attributes"]["relax.level"] == 3 for s in details)
    assert headers == {"X-SMG-Routing-Key": "session"}


@pytest.mark.parametrize("retry", [False, True])
def test_otlp_http_exports_same_ids_as_file(output, monkeypatch, retry):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    requests = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            request = ExportTraceServiceRequest()
            request.ParseFromString(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, request))
            self.send_response(503 if retry and len(requests) == 1 else 200)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("RELAX_OTEL_EXPORTER", "both")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"http://127.0.0.1:{server.server_port}/v1/traces")
    try:
        original = random.getstate()
        with tracing.stage("parent"):
            with tracing.stage("child"):
                pass
        assert tracing.flush()
        assert random.getstate() == original
        received = [s for _, r in requests for rs in r.resource_spans for ss in rs.scope_spans for s in ss.spans]
        assert {s.span_id.hex() for s in received} == {s["context"]["span_id"][2:] for s in spans(output)}
        assert len(received) == (4 if retry else 2)
        assert all(path == "/v1/traces" for path, _ in requests)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_otlp_grpc_retry_preserves_training_rng(output, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import grpc
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2, trace_service_pb2_grpc

    requests = []

    class Receiver(trace_service_pb2_grpc.TraceServiceServicer):
        def Export(self, request, context):
            requests.append(request)
            if len(requests) == 1:
                context.abort(grpc.StatusCode.UNAVAILABLE, "retry probe")
            return trace_service_pb2.ExportTraceServiceResponse()

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(Receiver(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    monkeypatch.setenv("RELAX_OTEL_EXPORTER", "both")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"http://127.0.0.1:{port}")
    try:
        state = random.getstate()
        with tracing.stage("grpc"):
            pass
        assert tracing.flush()
        assert len(requests) == 2
        assert random.getstate() == state
    finally:
        server.stop(grace=0).wait(timeout=5)


def test_megatron_adapter_uses_real_call_arguments_and_does_not_infer_backward_id(output, monkeypatch):
    from relax.utils.tracing_megatron import _calls, _wrap

    monkeypatch.setenv("RELAX_OTEL_LEVEL", "3")
    monkeypatch.setenv("RELAX_OTEL_DETAIL_STEPS", "0")

    def forward(value, current_microbatch=None):
        return value

    def backward(value):
        return value

    token = _calls.set({"forward": 0, "backward": 0})
    try:
        with tracing.stage("schedule", **{"relax.step": 0}):
            assert _wrap(forward, "forward")(3, current_microbatch=7) == 3
            assert _wrap(backward, "backward")(5) == 5
    finally:
        _calls.reset(token)
    records = {s["name"]: s for s in spans(output)}
    assert records["megatron.microbatch.forward"]["attributes"]["relax.microbatch"] == 7
    assert "relax.microbatch" not in records["megatron.microbatch.backward"]["attributes"]
    assert records["megatron.microbatch.backward"]["parent_id"] == records["schedule"]["context"]["span_id"]


def test_native_sglang_configuration_preserves_user_arguments(output, monkeypatch):
    from relax.utils.tracing_sglang import configure

    original = {"model_path": "model", "port": 12345}
    assert configure(original) is original
    monkeypatch.setenv("RELAX_OTEL_LEVEL", "3")
    monkeypatch.setenv("RELAX_OTEL_EXPORTER", "both")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector/v1/traces")
    result = configure(original)
    assert result["model_path"] == original["model_path"]
    assert result["enable_trace"] is True
    assert result["otlp_traces_endpoint"] == "http://collector/v1/traces"
    assert "enable_trace" not in original
