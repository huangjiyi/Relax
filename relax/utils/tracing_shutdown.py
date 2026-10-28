# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cancellation semantics and process-local span finalization. No SDK import when disabled."""

import asyncio
import atexit
import os
import threading
import time
from concurrent.futures import CancelledError as FutureCancelledError
from typing import Any


def mark_interrupted(span: Any, outcome: str, reason: str, **attributes: Any) -> None:
    """Retain a real error even when shutdown subsequently interrupts its owner."""
    if not span.is_recording():
        return
    values = {"relax.span.outcome": outcome, "relax.cancel.reason": reason, **attributes}
    span.add_event("cancellation", attributes=values)
    if getattr(getattr(getattr(span, "status", None), "status_code", None), "name", None) != "ERROR":
        span.set_attributes(values)


def record_exception(span: Any, exc: BaseException) -> None:
    from opentelemetry.trace import Status, StatusCode

    from relax.engine.rollout.request_permit import GenerationAborted

    if not span.is_recording():
        return
    if isinstance(exc, (GenerationAborted, asyncio.CancelledError, FutureCancelledError, KeyboardInterrupt)) or (
        isinstance(exc, SystemExit) and exc.code in (None, 0)
    ):
        outcome = "aborted" if isinstance(exc, GenerationAborted) else "cancelled"
        reason = str(exc) or type(exc).__name__
        details = {"exception.type": f"{type(exc).__module__}.{type(exc).__qualname__}"}
        if exc.__cause__ is not None:
            details["relax.cancel.cause_type"] = type(exc.__cause__).__name__
            details["relax.cancel.cause"] = str(exc.__cause__)
            response = getattr(exc.__cause__, "response", None)
            if response is not None:
                details["http.response.status_code"] = response.status_code
        mark_interrupted(span, outcome, reason, **details)
    else:
        span.record_exception(exc)
        span.set_attribute("relax.span.outcome", "error")
        span.set_status(Status(StatusCode.ERROR, type(exc).__name__))


class ActiveSpanProcessor:
    """Keep only live spans, so an explicit shutdown can export real observed boundaries."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.pid = os.getpid()
        self.active: dict[int, Any] = {}
        self.lock = threading.RLock()
        self.closing: str | None = None
        self.listener: Any = None
        self.listener_started = False
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork)

    def _after_fork(self) -> None:
        self.pid = os.getpid()
        self.active = {}
        self.lock = threading.RLock()
        self.closing = None
        self.listener = None
        self.listener_started = False

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        with self.lock:
            reason = self.closing
            if reason is None:
                self.active[span.context.span_id] = span
            # The first span starts after exporters have been installed. The
            # native and Relax providers both use this hook, including children.
            if not self.listener_started:
                self.listener_started = True
                try:
                    from relax.utils.tracing_shutdown_control import start_listener

                    self.listener = start_listener(self)
                except Exception as exc:
                    from relax.utils.logging_utils import get_logger

                    get_logger(__name__).warning(f"OTel shutdown registration failed: {exc}")
        if reason is not None:
            # A caller may have obtained its tracer before shutdown began.
            # End observation at this actual callback, never leave a new orphan.
            mark_interrupted(span, "cancelled", reason)
            span.set_attribute("relax.span.end_reason", reason)
            span.end()

    def _on_ending(self, span: Any) -> None:
        pass

    def on_end(self, span: Any) -> None:
        with self.lock:
            self.active.pop(span.context.span_id, None)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def close(self, reason: str, *, completed: bool = False) -> int:
        if os.getpid() != self.pid:
            return 0
        with self.lock:
            if self.closing is not None:
                return 0
            self.closing = reason
            active = sorted(self.active.values(), key=lambda s: s.start_time, reverse=True)
            boundary = time.time_ns()
            for span in active:
                if not span.is_recording():
                    continue
                # This is an observed shutdown boundary, not a successful operation end.
                span.set_attribute("relax.span.end_reason", reason)
                if not completed:
                    mark_interrupted(span, "cancelled", reason)
                span.end(end_time=boundary)
            return len(active)

    def shutdown(self) -> None:
        # Must run before exporter.shutdown() in provider processor order.
        if self.listener is not None:
            self.listener.finish("process_exit")
        else:
            self.close("process_exit")

    def at_exit(self) -> None:
        if os.getpid() != self.pid:
            return
        try:
            if self.listener is not None:
                self.listener.finish("process_exit")
            else:
                self.close("process_exit")
                self.provider.force_flush(timeout_millis=2000)
        except Exception:
            # Interpreter teardown cannot change the process's exit outcome.
            pass


def track_active_spans(provider: Any) -> ActiveSpanProcessor:
    processor = ActiveSpanProcessor(provider)
    provider.add_span_processor(processor)
    # Registered after the provider's own hook: atexit runs in reverse order.
    atexit.register(processor.at_exit)
    return processor
