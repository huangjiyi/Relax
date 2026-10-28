# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in SGLang startup and receiver tracing, called by the versioned overlay.

Weight buckets are observed without exporting a span per bucket. A pending
version is never installed as the ambient context between control requests.
"""

import atexit
import os
import threading
import time
from contextlib import contextmanager

from relax.utils.env import Envs
from relax.utils.tracing_code import function_location


_updates = {}
_exit_registered = False


def _enabled(native):
    return (Envs.RELAX_OTEL_ENABLED and Envs.RELAX_OTEL_SGLANG_LIFECYCLE
            and Envs.RELAX_OTEL_LEVEL >= 2 and native.tracer is not None
            and not getattr(native, "_tracing_stopped", lambda: False)())


def _attributes(native, function):
    values = {"relax.role": "rollout", "relax.level": 2, **function_location(function)}
    info = native.threads_info.get(threading.get_native_id())
    if info is not None:
        values.update({"host_id": info.host_id, "pid": info.pid, "thread_label": info.thread_label})
        for key in ("tp_rank", "pp_rank", "dp_rank"):
            value = getattr(info, key, None)
            if value is not None:
                values[key] = value
    return values


def _context(carrier=None):
    from opentelemetry import trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    if not carrier and trace.get_current_span().get_span_context().is_valid:
        return None
    return TraceContextTextMapPropagator().extract(
        carrier or {"traceparent": os.environ.get("RELAX_OTEL_TRACEPARENT", "")}
    )


@contextmanager
def lifecycle(native, name, function):
    if not _enabled(native):
        yield
        return
    with native.tracer.start_as_current_span(name, context=_context(), attributes=_attributes(native, function)):
        yield


def _key(owner):
    return os.getpid(), threading.get_native_id(), id(owner)


def _finish(key, reason, error=None):
    state = _updates.pop(key, None)
    if state is None:
        return
    from opentelemetry.trace import Status, StatusCode

    span = state["span"]
    span.set_attributes({
        "relax.weights.bucket_count": state["buckets"],
        "relax.weights.apply_host_ns": state["active_ns"],
        "relax.weights.end_reason": reason,
    })
    if error:
        span.set_status(Status(StatusCode.ERROR, error))
    span.end()


def close_pending():
    """Export unfinished updates as errors on graceful process shutdown."""
    for key in list(_updates):
        _finish(key, "shutdown", "Process exited before generation resumed")


@contextmanager
def weight_update(native, owner, request, function):
    global _exit_registered
    carrier = getattr(request, "relax_trace_context", None)
    if not _enabled(native) or not carrier:
        yield None
        return
    from opentelemetry import trace

    key = _key(owner)
    version = getattr(request, "weight_version", None)
    # Routed expert updates can omit the published version until the final
    # control request; their Actor transfer parent is the update identity.
    identity = str(version) if version is not None else carrier.get("traceparent", "")
    state = _updates.get(key)
    if state is not None and state["identity"] != identity:
        _finish(key, "superseded", "A new update arrived before generation resumed")
        state = None
    if state is None:
        attrs = _attributes(native, function)
        if version is not None:
            attrs["relax.weights.version"] = str(version)
        step = carrier.get("relax-step", "")
        if step.lstrip("-").isdigit():
            attrs["relax.step"] = int(step)
        span = native.tracer.start_span(
            "rollout.weights.receive_apply", context=_context(carrier), attributes=attrs
        )
        if not span.is_recording():
            span.end()
            yield None
            return
        state = {"span": span, "identity": identity, "buckets": 0, "active_ns": 0}
        _updates[key] = state
        if not _exit_registered:
            atexit.register(close_pending)
            _exit_registered = True
    started = time.perf_counter_ns()
    error = None
    try:
        with trace.use_span(state["span"], end_on_exit=False, record_exception=False, set_status_on_exception=False):
            yield state
    except BaseException as exc:
        state["span"].record_exception(exc)
        error = type(exc).__name__
        raise
    finally:
        state["buckets"] += 1
        state["active_ns"] += time.perf_counter_ns() - started
        if error:
            _finish(key, "error", error)


def weight_result(owner, result):
    if not getattr(result, "success", True):
        _finish(_key(owner), "error", "SGLang weight update returned success=False")


def weight_resume(owner, error=None):
    _finish(_key(owner), "resume" if error is None else "error", error)
