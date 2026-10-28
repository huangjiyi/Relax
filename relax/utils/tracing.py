# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in business spans. These measure host intervals, not GPU kernel time."""

import contextvars
import functools
import inspect
import os
import secrets
import socket
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.tracing_code import caller_location, function_location


logger = get_logger(__name__)
_provider: Any = None
_provider_pid: int | None = None
_root_span: Any = None
_stream: Any = None
_initialization_failed = False
_active_spans: Any = None
_stopped_pid: int | None = None
_lock = threading.Lock()
_attributes: contextvars.ContextVar[dict] = contextvars.ContextVar("relax_trace_attributes", default={})
_process_attributes: dict = {}
_process_attributes_pid: int | None = None
_ENV_NAMES = (
    "RELAX_OTEL_ENABLED",
    "RELAX_OTEL_OUTPUT_DIR",
    "RELAX_OTEL_SHUTDOWN_DIR",
    "RELAX_OTEL_RUN_ID",
    "RELAX_OTEL_TRACEPARENT",
    "RELAX_OTEL_EXPORTER",
    "RELAX_OTEL_LEVEL",
    "RELAX_OTEL_DETAIL_STEPS",
    "RELAX_OTEL_SGLANG_LIFECYCLE",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_RESOURCE_ATTRIBUTES",
    "SGLANG_TRACE_LEVEL",
)


def enabled() -> bool:
    return Envs.RELAX_OTEL_ENABLED


def set_process_attributes(attributes: dict) -> None:
    """Store rank metadata once on worker.init; never inherit it into child spans."""
    global _process_attributes, _process_attributes_pid
    if not enabled():
        return
    pid = os.getpid()
    if _process_attributes_pid != pid:
        _process_attributes = {}
        _process_attributes_pid = pid
    _process_attributes = {key: value for key, value in attributes.items() if value is not None}
    if _get_tracer() is not None:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording() and getattr(span, "name", None) == "worker.init":
            span.set_attributes(_process_attributes)


def selected(level: int, attributes: dict | None = None) -> bool:
    """L1/L2 are continuous; L3 requires an explicit rollout-step window."""
    if not enabled() or level > Envs.RELAX_OTEL_LEVEL:
        return False
    if level < 3:
        return True
    step = {**_attributes.get(), **{k: v for k, v in (attributes or {}).items() if v is not None}}.get("relax.step")
    window = Envs.RELAX_OTEL_DETAIL_STEPS
    if step is None or not window:
        return False
    for part in window.split(","):
        bounds = part.strip().split("-")
        start, end = int(bounds[0]), int(bounds[-1])
        if start <= int(step) <= end:
            return True
    return False


def _get_tracer() -> Any:
    global _provider, _provider_pid, _stream, _initialization_failed, _active_spans
    if not enabled() or _stopped_pid == os.getpid():
        return None
    pid = os.getpid()
    if _provider_pid == pid and _active_spans is not None and _active_spans.closing is not None:
        return None
    with _lock:
        if _provider_pid != pid:
            _provider = None
            _initialization_failed = False
            _provider_pid = pid
        if _initialization_failed:
            return None
        if _provider is None:
            try:
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
                from opentelemetry.sdk.trace.id_generator import IdGenerator

                class SystemIdGenerator(IdGenerator):
                    # Megatron resets Python random identically on multiple ranks.
                    # IDs must neither repeat nor advance the training RNG stream.
                    def generate_span_id(self) -> int:
                        return secrets.randbits(64) or self.generate_span_id()

                    def generate_trace_id(self) -> int:
                        return secrets.randbits(128) or self.generate_trace_id()

                    def is_trace_id_random(self) -> bool:
                        return True

                run_id = Envs.RELAX_OTEL_RUN_ID or os.environ.get("RUN_NAME") or uuid.uuid4().hex
                directory = Path(Envs.RELAX_OTEL_OUTPUT_DIR or Path(tempfile.gettempdir()) / "relax-otel" / run_id)
                os.environ["RELAX_OTEL_RUN_ID"] = run_id
                os.environ["RELAX_OTEL_OUTPUT_DIR"] = str(directory)
                exporters = Envs.RELAX_OTEL_EXPORTER
                if exporters not in ("file", "otlp", "both"):
                    raise ValueError("RELAX_OTEL_EXPORTER must be file, otlp or both")
                # Use a private provider; Ray or another library may own the global provider.
                _provider = TracerProvider(
                    resource=Resource.create(
                        {
                            "service.name": "relax",
                            "service.instance.id": f"{socket.gethostname()}:{pid}",
                            "host.name": socket.gethostname(),
                            "process.pid": pid,
                            "relax.run_id": run_id,
                            "relax.timing": "host",
                        }
                    ),
                    shutdown_on_exit=True,
                    id_generator=SystemIdGenerator(),
                )
                from relax.utils.tracing_shutdown import track_active_spans

                _active_spans = track_active_spans(_provider)
                from relax.utils.tracing_policy import CollectionProcessor

                collection = CollectionProcessor()
                _provider.add_span_processor(collection)
                if exporters in ("file", "both"):
                    directory.mkdir(parents=True, exist_ok=True)
                    _stream = (directory / f"{socket.gethostname()}-{pid}.jsonl").open("a", buffering=1)
                    collection.add_span_processor(
                        SimpleSpanProcessor(
                            ConsoleSpanExporter(out=_stream, formatter=lambda span: span.to_json(indent=None) + "\n")
                        )
                    )
                if exporters in ("otlp", "both"):
                    from relax.utils.tracing_otlp import isolate_exporter_rng

                    endpoint = Envs.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
                    if not endpoint:
                        raise ValueError("Set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT for OTLP export")
                    protocol = Envs.OTEL_EXPORTER_OTLP_TRACES_PROTOCOL
                    if protocol == "http/protobuf":
                        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                        exporter = OTLPSpanExporter(endpoint=endpoint, timeout=2)
                    elif protocol == "grpc":
                        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                        exporter = OTLPSpanExporter(endpoint=endpoint, timeout=2)
                    else:
                        raise ValueError("Unsupported OTLP trace protocol")
                    collection.add_span_processor(
                        BatchSpanProcessor(
                            isolate_exporter_rng(exporter),
                            max_queue_size=8192,
                            max_export_batch_size=256,
                            schedule_delay_millis=500,
                        )
                    )
            except Exception as exc:
                if _provider is not None:
                    _provider.shutdown()
                    _provider = None
                _initialization_failed = True
                logger.warning(f"OTel initialization failed ({type(exc).__name__}); continuing without spans")
                return None
    return _provider.get_tracer("relax.ettr")


def _parent_context(carrier: dict | None = None) -> Any:
    from opentelemetry import trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    if carrier:
        return TraceContextTextMapPropagator().extract(carrier)
    if trace.get_current_span().get_span_context().is_valid:
        return None  # Keep the context of this task, rather than the process fallback.
    return TraceContextTextMapPropagator().extract({"traceparent": Envs.RELAX_OTEL_TRACEPARENT})


def capture_context() -> dict[str, str]:
    if _get_tracer() is None:
        return {}
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    carrier: dict[str, str] = {}
    from relax.utils.tracing_policy import carrier_context

    TraceContextTextMapPropagator().inject(carrier, context=carrier_context(_parent_context()))
    return carrier


def remote_context() -> dict:
    """Explicit keyword for instrumented Ray methods, empty when disabled."""
    carrier = capture_context()
    return {"_trace_context": carrier} if carrier else {}


def weight_context() -> dict:
    """Business context for weight-control RPCs, independent of the L3 window."""
    if not Envs.RELAX_OTEL_SGLANG_LIFECYCLE or not selected(2):
        return {}
    carrier = capture_context()
    if carrier:
        step = _attributes.get().get("relax.step")
        if step is not None:
            carrier["relax-step"] = str(step)
    return {"relax_trace_context": carrier} if carrier else {}


def child_runtime_environment(runtime_env: dict | None) -> dict | None:
    """Copy a Ray environment and bind the child's startup to the current span."""
    carrier = capture_context()
    if not carrier:
        return runtime_env
    result = dict(runtime_env or {})
    result["env_vars"] = {**result.get("env_vars", {}), "RELAX_OTEL_TRACEPARENT": carrier["traceparent"]}
    return result


@contextmanager
def subprocess_context() -> Iterator[None]:
    """Carry startup context through spawn; restore the actor environment afterward."""
    carrier = capture_context()
    if not carrier:
        yield
        return
    previous = os.environ.get("RELAX_OTEL_TRACEPARENT")
    os.environ["RELAX_OTEL_TRACEPARENT"] = carrier["traceparent"]
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RELAX_OTEL_TRACEPARENT", None)
        else:
            os.environ["RELAX_OTEL_TRACEPARENT"] = previous


def request_context(headers: dict | None = None) -> dict | None:
    """Forward W3C context; native backend details follow the same L3 window."""
    carrier = capture_context()
    if not carrier:
        return headers
    if not selected(3):
        # ParentBased samplers in SGLang suppress details outside the window.
        flags = int(carrier["traceparent"][-2:], 16) & ~1
        carrier["traceparent"] = carrier["traceparent"][:-2] + f"{flags:02x}"
    return {**(headers or {}), **carrier}


def flush() -> bool:
    if _provider is None or _provider_pid != os.getpid():
        return True
    try:
        return _provider.force_flush(timeout_millis=5000)
    except Exception as exc:
        logger.warning(f"OTel flush failed ({type(exc).__name__})")
        return False


def shutdown_tracing(reason: str, *, error: bool = False, completed: bool = False) -> bool:
    """End this process's observation at shutdown before remote teardown can block.

    Does not stop work or cancel tasks. Further instrumentation is disabled in
    this process; interrupted spans retain the observed boundary and reason.
    """
    global _stopped_pid, _root_span
    if _provider is None or _provider_pid != os.getpid():
        return True
    _stopped_pid = os.getpid()
    try:
        if error and _root_span is not None and _root_span.is_recording():
            from opentelemetry.trace import Status, StatusCode

            _root_span.set_attribute("relax.span.outcome", "error")
            _root_span.set_status(Status(StatusCode.ERROR, reason))
        if _active_spans is not None:
            _active_spans.close(reason, completed=completed)
        _root_span = None
        return flush()
    except Exception as exc:
        logger.warning(f"OTel shutdown failed ({type(exc).__name__})")
        return False


def runtime_environment() -> dict[str, str]:
    if not enabled():
        return {}
    return {name: os.environ[name] for name in _ENV_NAMES if name in os.environ}


@contextmanager
def stage(name: str, *, carrier: dict | None = None, level: int = 1,
          _code_location: dict | None = None, **attributes: Any) -> Iterator[Any]:
    if not selected(level, attributes):
        yield None
        return
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    from opentelemetry import trace

    from relax.utils.tracing_shutdown import record_exception

    # Inherit only business scope, not fixed rank metadata or the parent's code.
    scope = {**_attributes.get(), **{key: value for key, value in attributes.items() if value is not None}}
    values = {**scope, **(_code_location or caller_location())}
    values["relax.level"] = level
    if name == "worker.init":
        if _process_attributes_pid == os.getpid():
            values.update(_process_attributes)
        values["relax.detail_steps"] = Envs.RELAX_OTEL_DETAIL_STEPS
    token = _attributes.set(scope)
    try:
        from relax.utils.tracing_policy import REMOVED

        if name in REMOVED:
            # Retain task-local business IDs and explicit RPC ancestry, but do
            # not create a wrapper or let caller attributes mutate its parent.
            parent = trace.get_current_span(_parent_context(carrier)).get_span_context()
            with trace.use_span(trace.NonRecordingSpan(parent), record_exception=False,
                                set_status_on_exception=False):
                yield None
            return
        span = tracer.start_span(
            name,
            context=_parent_context(carrier),
            attributes=values,
        )
        with trace.use_span(
            span,
            record_exception=False,
            set_status_on_exception=False,
        ):
            try:
                yield span
            except BaseException as exc:
                record_exception(span, exc)
                raise
            finally:
                if span.is_recording():
                    span.end()
    finally:
        _attributes.reset(token)


def _finish_startup_wait(owner: Any, run_span: Any) -> None:
    """Export the observed init-to-run interval without installing an ambient span."""
    pending = getattr(owner, "_otel_startup_wait", None)
    if pending is None:
        return
    owner._otel_startup_wait = None
    if getattr(run_span, "start_time", None) is None or not selected(2):
        return
    tracer = _get_tracer()
    if tracer is None:
        return
    from opentelemetry import context, trace

    name, initialized = pending
    parent = context.Context()
    if initialized.parent is not None:
        parent = trace.set_span_in_context(trace.NonRecordingSpan(initialized.parent), parent)
    values = {key: value for key, value in initialized.attributes.items()
              if key.startswith("code.") or key in ("relax.role", "relax.code.location")}
    values.update({"relax.level": 2, "relax.wait.reason": "controller_startup",
                   "relax.wait.end_reason": "run_called"})
    span = tracer.start_span(
        name, context=parent, start_time=initialized.end_time, attributes=values,
        links=[trace.Link(initialized.get_span_context()), trace.Link(run_span.get_span_context())],
    )
    span.end(end_time=run_span.start_time)


def traced(name: str, *, role: str | None = None, step_arg: str | None = None, level: int = 1,
           group_arg: str | None = None, sample_arg: str | None = None,
           startup_wait: str | None = None, finish_startup_wait: bool = False) -> Callable:
    """Preserve sync/async behavior and accept a declared Ray carrier keyword."""

    def decorate(function: Callable) -> Callable:
        signature = inspect.signature(function)
        location = function_location(function)

        def scope(args: tuple, kwargs: dict) -> Any:
            bound = signature.bind_partial(*args, **kwargs).arguments
            owner = bound.get("self")
            step = bound.get(step_arg) if step_arg else getattr(owner, "step", None)
            group = bound.get(group_arg) if group_arg else None
            sample = bound.get(sample_arg) if sample_arg else None
            group_sample = group[0] if group else sample
            return stage(
                name,
                carrier=bound.get("_trace_context"),
                level=level,
                _code_location=location,
                **{
                    "relax.role": role or bound.get("role") or getattr(owner, "role", None),
                    "relax.rank": getattr(owner, "_rank", getattr(owner, "rank", None)),
                    "relax.step": step,
                    "relax.group_index": getattr(group_sample, "group_index", None),
                    "relax.sample_index": getattr(sample, "index", None),
                },
            )

        if inspect.iscoroutinefunction(function):

            @functools.wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not enabled():
                    return await function(*args, **kwargs)
                try:
                    with scope(args, kwargs) as span:
                        if finish_startup_wait and args:
                            _finish_startup_wait(args[0], span)
                        result = await function(*args, **kwargs)
                finally:
                    if "_trace_context" in signature.parameters:
                        flush()
                return result

            return async_wrapper

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled():
                return function(*args, **kwargs)
            try:
                with scope(args, kwargs) as span:
                    result = function(*args, **kwargs)
            finally:
                if "_trace_context" in signature.parameters:
                    flush()
            if startup_wait and args and getattr(span, "end_time", None) is not None and selected(2):
                args[0]._otel_startup_wait = (startup_wait, span)
            return result

        return wrapper

    return decorate


def start_run() -> None:
    global _root_span
    tracer = _get_tracer()
    if tracer is None or _root_span is not None:
        return
    from opentelemetry import trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    _root_span = tracer.start_span(
        "job.run", context=_parent_context(), attributes={
            "relax.role": "controller", "relax.detail_steps": Envs.RELAX_OTEL_DETAIL_STEPS,
            "relax.otel.level": Envs.RELAX_OTEL_LEVEL, **caller_location(),
        }
    )
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier, context=trace.set_span_in_context(_root_span))
    os.environ["RELAX_OTEL_TRACEPARENT"] = carrier["traceparent"]


def finish_run(*, error: bool = False) -> None:
    global _root_span
    if _root_span is None:
        return
    from opentelemetry.trace import Status, StatusCode

    if error:
        _root_span.set_status(Status(StatusCode.ERROR))
    try:
        _root_span.end()
        _provider.force_flush(timeout_millis=5000)
    except Exception as exc:
        logger.warning(f"OTel finalization failed ({type(exc).__name__})")
    finally:
        _root_span = None


def capture_run_configuration(config: Any, role_gpus: dict[str, int], shared_roles: list[str]) -> None:
    """Record the resolved service allocation once, without changing child context."""
    if not enabled() or _root_span is None or not _root_span.is_recording():
        return
    try:
        shared = [count for role, count in role_gpus.items() if role in shared_roles]
        total = max(shared, default=0) + sum(count for role, count in role_gpus.items() if role not in shared_roles)
        mode = "hybrid" if getattr(config, "hybrid", False) else (
            "async" if getattr(config, "fully_async", False) else "colocate" if shared_roles else "separated"
        )
        _root_span.set_attributes({
            "relax.run.metadata_version": 1,
            "relax.run.execution_mode": mode,
            "relax.run.gpu_count": total,
            "relax.run.gpu_count_source": "service_resource_config",
            **{f"relax.run.{role}.gpu_count": count for role, count in role_gpus.items()},
        })
    except Exception as error:
        logger.warning(f"OTel run configuration capture unavailable ({type(error).__name__})")
