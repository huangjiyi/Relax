# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real SDK exports for cancellation, missing parents and process shutdown."""

import ast
import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.engine.rollout.request_permit import GenerationAborted
from relax.utils import tracing
from tests.utils.test_tracing import output as _output
from tests.utils.test_tracing import spans
from tests.utils.test_tracing_native_scheduler import native as _native
from tests.utils.test_tracing_native_scheduler import set_thread


output = _output
native = _native


@pytest.mark.parametrize("error", [GenerationAborted("weight_switch"), asyncio.CancelledError("user_stop"),
                                   RuntimeError("CUDA out of memory"), SystemExit(1)])
@pytest.mark.parametrize("async_call", [False, True])
async def test_remote_exception_is_preserved_and_flushed_after_end(output, monkeypatch, error, async_call):
    flushed = []
    monkeypatch.setattr(tracing, "flush", lambda: flushed.extend(spans(output)))

    @tracing.traced("remote")
    def sync(*, _trace_context=None):
        raise error

    @tracing.traced("remote")
    async def asynchronous(*, _trace_context=None):
        raise error

    with pytest.raises(type(error)) as caught:
        if async_call:
            await asynchronous()
        else:
            sync()
    assert caught.value is error
    assert len(flushed) == 1
    record = flushed[0]
    expected = isinstance(error, (GenerationAborted, asyncio.CancelledError))
    assert record["status"]["status_code"] == ("UNSET" if expected else "ERROR")
    if expected:
        assert record["attributes"]["relax.cancel.reason"] == str(error)
        assert record["events"][0]["name"] == "cancellation"


def test_abort_retains_http_cause_without_reclassifying_generic_http_errors(output):
    cause = RuntimeError("SGLang returned 499 while pausing generation")
    cause.response = SimpleNamespace(status_code=499)
    with pytest.raises(GenerationAborted):
        with tracing.stage("request"):
            raise GenerationAborted("weight_switch") from cause
    record = spans(output)[0]
    assert record["attributes"]["http.response.status_code"] == 499
    assert record["attributes"]["relax.cancel.cause"] == str(cause)
    with pytest.raises(RuntimeError):
        with tracing.stage("unexpected_http"):
            raise cause
    assert spans(output)[1]["status"]["status_code"] == "ERROR"


def test_shutdown_ends_actual_parents_once_and_preserves_failures(output, monkeypatch):
    from opentelemetry.trace import Status, StatusCode

    monkeypatch.setattr(tracing, "_stopped_pid", None)
    tracing.start_run()
    with tracing.stage("service.run"):
        with tracing.stage("already_failed") as failed:
            failed.set_status(Status(StatusCode.ERROR, "real failure"))
            assert tracing.shutdown_tracing("SIGTERM")
            assert tracing.shutdown_tracing("SIGTERM")
            with tracing.stage("after_observation_end") as after:
                assert after is None
    records = spans(output)
    assert len(records) == 3
    ids = {s["context"]["span_id"] for s in records}
    assert all(s["parent_id"] is None or s["parent_id"] in ids for s in records)
    assert len({s["end_time"] for s in records}) == 1
    assert next(s for s in records if s["name"] == "already_failed")["status"]["status_code"] == "ERROR"


def test_job_failure_does_not_mark_unrelated_open_work_as_failed(output, monkeypatch):
    monkeypatch.setattr(tracing, "_stopped_pid", None)
    tracing.start_run()
    with tracing.stage("unrelated_inflight"):
        tracing.shutdown_tracing("job_exit_1", error=True)
    records = {s["name"]: s for s in spans(output)}
    assert records["job.run"]["status"]["status_code"] == "ERROR"
    assert records["unrelated_inflight"]["status"]["status_code"] == "UNSET"
    assert records["unrelated_inflight"]["attributes"]["relax.span.outcome"] == "cancelled"


@pytest.mark.parametrize("info,is_error", [({"reason": "Aborted"}, False),
    ({"type": "abort", "status_code": 503, "message": "unavailable"}, True),
    ({"reason": "CUDA out of memory"}, True)])
def test_native_only_known_cancellation_is_non_error(native, info, is_error):
    module, exporter = native
    set_thread(module, "Scheduler", 0)
    request = module.TraceReqContext("status")
    request.trace_req_start()
    request.abort(abort_info=info)
    diagnostic = next(s for s in exporter.get_finished_spans() if s.name == "scheduler.abort")
    assert (diagnostic.status.status_code.name == "ERROR") == is_error


@pytest.mark.parametrize("explicit", [False, True])
def test_native_cancel_retains_scheduler_without_deleted_wrappers(native, explicit):
    module, exporter = native
    set_thread(module, "Tokenizer")
    request = module.TraceReqContext("cancelled-request")
    request.trace_req_start()
    assert request.root_span is None
    copy = request.copy_for_thread()
    set_thread(module, "Scheduler", 0)
    copy.rebuild_thread_context()
    copy.trace_slice_start("decode_forward", 1)
    copy.abort(abort_info={"status_code": 499, "message": "weight_switch"})
    if explicit:
        request.abort(abort_info={"status_code": 499, "message": "weight_switch"})
    else:
        request.__del__()
    request.__del__()
    records = exporter.get_finished_spans()
    assert len(records) == 1 and records[0].name == "decode_forward"
    assert records[0].attributes["relax.span.outcome"] == "aborted"
    assert records[0].parent is None
    assert records[0].status.status_code.name != "ERROR"


@pytest.mark.parametrize("enabled", ["1", "true", "on"])
def test_native_tokenizer_finish_and_late_abort_emit_no_wrappers(native, monkeypatch, enabled):
    module, exporter = native
    monkeypatch.setenv("RELAX_OTEL_ENABLED", enabled)
    set_thread(module, "Tokenizer")
    request = module.TraceReqContext("late-abort")
    request.trace_req_start()
    request.abort()
    request.abort(abort_info={"status_code": 499, "message": "weight_switch"})
    request.trace_req_finish()
    assert request.root_span is None and request.thread_context is None
    assert not exporter.get_finished_spans()


def test_tracer_obtained_before_shutdown_cannot_leave_a_late_open_span(output, monkeypatch):
    monkeypatch.setattr(tracing, "_stopped_pid", None)
    tracer = tracing._get_tracer()
    tracing.shutdown_tracing("controller_shutdown")
    late = tracer.start_span("late_racing_thread")
    assert not late.is_recording()
    assert not tracing._active_spans.active
    assert spans(output)[0]["attributes"]["relax.cancel.reason"] == "controller_shutdown"


def test_startup_wait_racing_shutdown_is_noop(output, monkeypatch):
    owner = SimpleNamespace(_otel_startup_wait=("wait", SimpleNamespace()))
    monkeypatch.setenv("RELAX_OTEL_LEVEL", "2")
    monkeypatch.setattr(tracing, "_get_tracer", lambda: None)
    tracing._finish_startup_wait(owner, SimpleNamespace(start_time=10))
    assert not spans(output)


async def test_service_finalize_closes_outer_run_without_waiting_for_inner_task(output, monkeypatch):
    from relax.components.base import Base

    monkeypatch.setattr(tracing, "_stopped_pid", None)
    started = asyncio.Event()
    release = asyncio.Event()

    @tracing.traced("service.run", role="rollout")
    async def run():
        started.set()
        await release.wait()

    running = asyncio.create_task(run())
    await started.wait()
    assert await Base().finish_tracing("controller_shutdown")
    assert not running.done()  # No business cancellation or stop side effects.
    release.set()
    await running
    records = spans(output)
    assert len(records) == 1
    assert records[0]["attributes"]["relax.span.outcome"] == "cancelled"


def test_controller_finalization_dispatches_all_services_and_bounds_waits():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "relax/core/controller.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Controller")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_finalize_service_traces")
    events = []
    responses = []
    clock = [0]

    def dispatch(role):
        events.append(role)
        response = Mock()
        def result(timeout_s):
            assert events == ["actor", "rollout"]
            assert 0 <= timeout_s <= 5
            clock[0] += 5
            if role == "actor":
                raise TimeoutError("unreachable")
            return False
        response.result.side_effect = result
        responses.append(response)
        return response

    services = {role: SimpleNamespace(handle=SimpleNamespace(finish_tracing=SimpleNamespace(
        remote=lambda reason, role=role: dispatch(role)))) for role in ("actor", "rollout")}
    ns = {"Envs": SimpleNamespace(RELAX_OTEL_ENABLED=True), "time": SimpleNamespace(monotonic=lambda: clock[0]),
          "logger": Mock()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "controller.py", "exec"), ns)
    ns["_finalize_service_traces"](SimpleNamespace(serve_dict=services))
    assert all(response.result.call_count == 1 for response in responses)
    ns["Envs"].RELAX_OTEL_ENABLED = False
    events.clear()
    ns["_finalize_service_traces"](SimpleNamespace(serve_dict=services))
    assert events == []


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_driver_exports_before_hung_cleanup_and_hard_exit(tmp_path, sig):
    root = Path(__file__).resolve().parents[2]
    script = r'''
import ast, os, signal, sys, threading
from unittest.mock import Mock
from relax.utils import tracing
tracing.start_run()
tree = ast.parse(open("relax/entrypoints/train.py").read())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_hard_exit", "_graceful_shutdown")]
ctrl = Mock()
ctrl.shutdown.side_effect = threading.Event().wait
ns = dict(os=os, signal=signal, sys=sys, threading=threading, logger=Mock(), _ctrl=ctrl,
    _shutdown_done=False, _kernel_cache_heartbeat=None, _SHUTDOWN_TIMEOUT_SEC=.5,
    _NORMAL_SHUTDOWN_TIMEOUT_SEC=.5, shutdown_tracing=tracing.shutdown_tracing,
    stage=tracing.stage, finish_run=tracing.finish_run, ray=Mock(), serve=Mock())
exec(compile(tree, "train.py", "exec"), ns)
sig = int(sys.argv[1])
signal.signal(sig, ns["_graceful_shutdown"])
with tracing.stage("controller.loop"):
    with tracing.stage("completed_child"):
        pass
    os.kill(os.getpid(), sig)
'''
    env = {**os.environ, "RELAX_OTEL_ENABLED": "1", "RELAX_OTEL_EXPORTER": "file",
           "RELAX_OTEL_OUTPUT_DIR": str(tmp_path), "RELAX_OTEL_LEVEL": "2"}
    env.pop("RELAX_OTEL_TRACEPARENT", None)
    result = subprocess.run([sys.executable, "-c", script, str(sig)], cwd=root, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 128 + sig, result.stderr
    records = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert {s["name"] for s in records} == {"job.run", "controller.loop", "completed_child"}
    ids = {s["context"]["span_id"] for s in records}
    assert all(s["parent_id"] is None or s["parent_id"] in ids for s in records)
    for record in records:
        assert record["status"]["status_code"] != "ERROR"
        if record["name"] != "completed_child":
            assert record["attributes"]["relax.cancel.reason"] == signal.Signals(sig).name
