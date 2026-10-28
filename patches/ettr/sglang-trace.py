# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""package for sglang requests tracing"""

from __future__ import annotations

import functools
import logging
import os
import random
import secrets
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from sglang.srt.utils import get_int_env_var

logger = logging.getLogger(__name__)
opentelemetry_imported = False
opentelemetry_initialized = False
_trace_context_propagator = None
tracer: Optional[trace.Tracer] = None
_active_span_processor = None


def _tracing_stopped():
    return _active_span_processor is not None and _active_span_processor.closing is not None



# Modules allowed to emit spans (from --trace-modules); None means no filtering.
global_trace_modules: Optional[List[str]] = None

TRACE_HEADERS = ["traceparent", "tracestate"]

try:
    from opentelemetry import context, propagate, trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPSpanExporter,
    )
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider, id_generator
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _trace_context_propagator = TraceContextTextMapPropagator()

    opentelemetry_imported = True
except ImportError:

    class id_generator:
        class IdGenerator:
            pass

    logger.debug("opentelemetry package is not installed, tracing disabled")


def extract_trace_headers(headers: Mapping[str, str]) -> Optional[Dict]:
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


def get_global_trace_level() -> int:
    from sglang.srt.runtime_context import get_resources

    resources = get_resources()
    if resources.trace_level is None:
        resources.trace_level = get_int_env_var("SGLANG_TRACE_LEVEL", 3)
    return resources.trace_level


def set_global_trace_level(level: int):
    from sglang.srt.runtime_context import get_resources

    get_resources().trace_level = level


@dataclass
class TraceThreadInfo:
    host_id: str
    pid: int
    thread_label: str
    tp_rank: int
    dp_rank: int
    pp_rank: int


@dataclass
class TraceEvent:
    event_name: str
    ts: int
    attrs: Dict[str, Any]


@dataclass
class TraceSliceContext:
    slice_name: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    span: Optional[trace.span.Span] = None
    level: int = 1
    attrs: Optional[Dict[str, Any]] = None
    events: Optional[List[TraceEvent]] = None


@dataclass
class TraceThreadContext:
    thread_info: TraceThreadInfo
    cur_slice_stack: Optional[List[TraceSliceContext]] = None
    thread_span: Optional[trace.span.Span] = None
    thread_attrs: Optional[Dict[str, Any]] = None


class TraceCustomIdGenerator(id_generator.IdGenerator):
    """
    The default IdGenerator may produce duplicate trace IDs across multiple TP scheduler processes,
    hence a custom IdGenerator is implemented.
    """

    def __init__(self):
        super().__init__()
        self.local_random = random.Random()
        self.local_random.seed(time.time())

    def generate_trace_id(self) -> int:
        if os.environ.get("RELAX_OTEL_ENABLED") == "1":
            return secrets.randbits(128) or self.generate_trace_id()
        return self.local_random.getrandbits(64)

    def generate_span_id(self) -> int:
        if os.environ.get("RELAX_OTEL_ENABLED") == "1":
            return secrets.randbits(64) or self.generate_span_id()
        return self.local_random.getrandbits(64)


# global variables
threads_info: Dict[int, TraceThreadInfo] = {}
_worker_metadata_emitted = set()


def _code_attributes():
    if os.environ.get("RELAX_OTEL_ENABLED") != "1":
        return {}
    from relax.utils.tracing_code import caller_location

    return caller_location()


def _ettr_enabled():
    return os.environ.get("RELAX_OTEL_ENABLED", "").strip().lower() in ("1", "true", "on", "yes")


get_cur_time_ns = lambda: int(time.time() * 1e9)
if hasattr(time, "time_ns"):
    get_cur_time_ns = lambda: int(time.time_ns())


def _get_host_id() -> str:
    """
    In distributed tracing systems, obtain a unique node identifier
    and inject it into all subsequently generated spans
    to prevent PID conflicts between threads on different nodes.
    """
    if os.path.exists("/etc/machine-id"):
        try:
            with open("/etc/machine-id", "r") as f:
                return f.read().strip()
        except:
            pass

    mac = uuid.getnode()
    if mac != 0:
        return uuid.UUID(int=mac).hex

    return "unknown"


# Should be called by each tracked process.
def process_tracing_init(
    otlp_endpoint, server_name, trace_modules: Optional[str] = None
):
    global opentelemetry_initialized
    global get_cur_time_ns
    global tracer
    global _active_span_processor
    global global_trace_modules

    if trace_modules is not None:
        global_trace_modules = [
            module.strip() for module in trace_modules.split(",") if module.strip()
        ]

    if not opentelemetry_imported:
        opentelemetry_initialized = False
        raise RuntimeError(
            "opentelemetry package is not installed!!! Please not enable tracing or install opentelemetry"
        )

    try:
        resource = Resource.create(
            attributes={
                SERVICE_NAME: server_name,
            }
        )
        tracer_provider = TracerProvider(
            resource=resource, id_generator=TraceCustomIdGenerator()
        )

        schedule_delay_millis = get_int_env_var(
            "SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", 500
        )
        max_export_batch_size = get_int_env_var(
            "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", 64
        )

        span_exporter = get_otlp_span_exporter(otlp_endpoint)
        if os.environ.get("RELAX_OTEL_ENABLED", "").strip().lower() in ("1", "true", "on", "yes"):
            from relax.utils.tracing_otlp import isolate_exporter_rng
            from relax.utils.tracing_shutdown import track_active_spans

            _active_span_processor = track_active_spans(tracer_provider)
            span_exporter = isolate_exporter_rng(span_exporter)
        processor = BatchSpanProcessor(
            span_exporter=span_exporter,
            schedule_delay_millis=schedule_delay_millis,
            max_export_batch_size=max_export_batch_size,
        )
        if _ettr_enabled():
            from relax.utils.tracing_policy import CollectionProcessor

            collection = CollectionProcessor()
            collection.add_span_processor(processor)
            tracer_provider.add_span_processor(collection)
        else:
            tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)
    except Exception as e:
        opentelemetry_initialized = False
        raise RuntimeError(
            f"initialize opentelemetry error:{e}. Please set correct otlp endpoint."
        )

    opentelemetry_initialized = True
    tracer = trace.get_tracer("sglang server")


def get_global_tracing_enabled():
    return opentelemetry_initialized and not _tracing_stopped()


def get_otlp_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    supported_protocols = {"grpc", "http/protobuf"}

    if protocol not in supported_protocols:
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' configured. "
            f"Supported protocols are: {', '.join(sorted(supported_protocols))}"
        )

    if protocol == "grpc":
        return GRPCSpanExporter(endpoint=endpoint, insecure=True)
    elif protocol == "http/protobuf":
        return HTTPSpanExporter(endpoint=endpoint)


# Should be called by each tracked thread.
def trace_set_thread_info(
    thread_label: str,
    tp_rank: Optional[int] = None,
    dp_rank: Optional[int] = None,
    pp_rank: Optional[int] = None,
):
    if not opentelemetry_initialized:
        return

    pid = threading.get_native_id()
    if pid in threads_info:
        return

    threads_info[pid] = TraceThreadInfo(
        host_id=_get_host_id(),
        pid=pid,
        thread_label=thread_label,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
    )


class TraceReqContext:
    def __init__(
        self,
        rid,
        bootstrap_room=None,
        role="unified",
        module_name="",
        external_trace_header: Optional[Dict[str, str]] = None,
    ):
        self.rid: str = str(rid)
        self.trace_level = get_global_trace_level()
        self.tracing_enable: bool = opentelemetry_initialized and self.trace_level > 0 and not _tracing_stopped()

        # Filter by --trace-modules only for explicitly named modules; contexts
        # created with the default empty module_name are always traced.
        if (
            module_name
            and global_trace_modules is not None
            and module_name not in global_trace_modules
        ):
            self.tracing_enable = False

        if not self.tracing_enable or _tracing_stopped():
            return

        self.start_time_ns: Optional[int] = None
        self.thread_context: Optional[TraceThreadContext] = None
        self.bootstrap_room: Optional[int] = bootstrap_room
        self.role: str = role
        self.module_name = module_name

        # Indicates whether this instance is a replica from the main process.
        # When True, root_span is None and only root_span_context is preserved.
        self.is_copy: bool = False
        self.root_span: Optional[trace.span.Span] = None
        self.root_span_context: Optional[context.Context] = None
        # Record the most recently completed span as the previous span for the next span to be created.
        self.last_span_context: Optional[trace.span.SpanContext] = None
        self.external_trace_header: Optional[Dict[str, str]] = external_trace_header

        self.events_cache: List[TraceEvent] = []

        self.pid: int = threading.get_native_id()

    def is_tracing_enabled(self) -> bool:
        return self.tracing_enable and not _tracing_stopped()

    def __create_thread_context(self, ts: int):
        if self.pid not in threads_info:
            trace_set_thread_info("unknown")

        thread_info = threads_info[self.pid]
        thread_context = TraceThreadContext(
            thread_info=thread_info,
            cur_slice_stack=[],
        )

        thread_name = f"{thread_info.thread_label}"
        if thread_info.tp_rank is not None:
            thread_name += f" [TP {thread_info.tp_rank}] "
        if thread_info.pp_rank is not None:
            thread_name += f" [PP {thread_info.pp_rank}] "
        if thread_info.dp_rank is not None:
            thread_name += f" [DP {thread_info.dp_rank}] "
        thread_name += f"(host:{thread_info.host_id[:8]} | pid:{self.pid})"
        flat_scheduler = (
            _ettr_enabled()
            and thread_info.thread_label in ("Scheduler", "Tokenizer")
        )
        if not flat_scheduler:
            thread_context.thread_span = tracer.start_span(
                name=thread_name,
                start_time=ts,
                context=self.root_span_context,
                attributes=_code_attributes(),
            )

        rank_attrs = {}
        if thread_info.tp_rank is not None:
            rank_attrs["tp_rank"] = thread_info.tp_rank
        if thread_info.pp_rank is not None:
            rank_attrs["pp_rank"] = thread_info.pp_rank
        if thread_info.dp_rank is not None:
            rank_attrs["dp_rank"] = thread_info.dp_rank
        if rank_attrs and thread_context.thread_span:
            thread_context.thread_span.set_attributes(rank_attrs)

        identity = {"host_id": thread_info.host_id, "pid": thread_info.pid,
                    "thread_label": thread_info.thread_label}
        if flat_scheduler:
            thread_context.thread_attrs = {**identity, **rank_attrs}
        if _ettr_enabled() and thread_info.thread_label == "Scheduler":
            from relax.utils.tracing_topology import capture_sglang_topology
            from relax.utils.tracing_policy import register_native_metadata

            topology = getattr(thread_info, "_relax_topology", None)
            if topology is None:
                topology = capture_sglang_topology()
                if topology.get("relax.topology.status") == "complete":
                    thread_info._relax_topology = topology
            register_native_metadata(topology)

        if thread_context.thread_span:
            thread_context.thread_span.set_attributes(identity)

        return thread_context

    def __getstate__(self) -> Optional[Dict[str, Any]]:
        if not self.tracing_enable or _tracing_stopped():
            return {"tracing_enable": False}

        if self.root_span_context is None:
            return {"tracing_enable": False}

        state = {
            "tracing_enable": self.tracing_enable,
            "rid": self.rid,
            "bootstrap_room": self.bootstrap_room,
            "start_time_ns": self.start_time_ns,
            "role": self.role,
            "trace_level": self.trace_level,
            "module_name": self.module_name,
            "is_copy": self.is_copy,
            "pid": self.pid,
            "thread_context": None,
            "root_span": None,
            "last_span_context": None,
        }

        carrier: dict[str, str] = {}
        propagate.inject(carrier, self.root_span_context)
        state["root_span_context"] = carrier

        prev_span_context = self.last_span_context
        if self.thread_context and self.thread_context.cur_slice_stack:
            cur_slice = self.thread_context.cur_slice_stack[0]
            if cur_slice.span:
                prev_span_context = cur_slice.span.get_span_context()

        if prev_span_context:
            state["last_span_context"] = {
                "span_id": prev_span_context.span_id,
                "trace_id": prev_span_context.trace_id,
            }

        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        if not opentelemetry_initialized:
            self.tracing_enable = False
        if not self.tracing_enable or _tracing_stopped():
            return

        self.is_copy = True
        self.pid = threading.get_native_id()
        self.root_span_context = propagate.extract(self.root_span_context)
        if self.last_span_context:
            self.last_span_context = trace.span.SpanContext(
                trace_id=self.last_span_context["trace_id"],
                span_id=self.last_span_context["span_id"],
                is_remote=True,
            )
        self.events_cache = []

    def copy_for_thread(self) -> TraceReqContext:
        """
        Create a copy of this context for use in another thread.

        The copy shares the same root_span_context but has its own thread_context.
        This is useful for propagating trace context across threads (e.g., worker threads).

        Usage:
            # Sender (main thread)
            trace_ctx_copy = trace_ctx.copy_for_thread()
            queue.put(TransferKVChunk(..., trace_ctx=trace_ctx_copy))

            # Receiver (worker thread)
            kv_chunk = queue.get()
            kv_chunk.trace_ctx.rebuild_thread_context()
        """
        # Fast path: not tracing
        if not self.tracing_enable or _tracing_stopped() or self.root_span_context is None:
            return TraceNullContext()

        # Extract prev_span_context from current thread state
        prev_span_context = self.last_span_context
        if self.thread_context and self.thread_context.cur_slice_stack:
            cur_slice = self.thread_context.cur_slice_stack[0]
            if cur_slice.span:
                prev_span_context = cur_slice.span.get_span_context()

        # Create new instance with shared state
        copied = TraceReqContext.__new__(TraceReqContext)
        copied.tracing_enable = self.tracing_enable
        copied.rid = self.rid
        copied.bootstrap_room = self.bootstrap_room
        copied.start_time_ns = self.start_time_ns
        copied.role = self.role
        copied.trace_level = self.trace_level
        copied.module_name = self.module_name
        copied.is_copy = True  # Mark as copy
        copied.pid = self.pid

        # thread_context is None, will be rebuilt via rebuild_thread_context()
        copied.thread_context = None
        copied.root_span = None

        # Share root_span_context (already a context, no need to serialize)
        copied.root_span_context = self.root_span_context

        # Set prev_span_context for linking spans
        if prev_span_context:
            copied.last_span_context = trace.span.SpanContext(
                trace_id=prev_span_context.trace_id,
                span_id=prev_span_context.span_id,
                is_remote=True,
            )
        else:
            copied.last_span_context = None

        copied.events_cache = []

        return copied

    def rebuild_thread_context(self, ts: Optional[int] = None):
        if not self.tracing_enable or _tracing_stopped():
            return

        ts = ts or get_cur_time_ns()
        self.thread_context = self.__create_thread_context(ts)

    def trace_req_start(
        self,
        ts: Optional[int] = None,
    ):
        if not self.tracing_enable or _tracing_stopped():
            return

        ts = ts or get_cur_time_ns()

        # create req context and root span
        self.start_time_ns = ts

        external_trace_context = _trace_context_propagator.extract(
            self.external_trace_header or {}
        )

        if _ettr_enabled():
            # Requests/Tokenizer are transport context only. Preserve the real
            # engine_request parent through serialization without a phantom ID.
            self.root_span_context = external_trace_context
            self.thread_context = self.__create_thread_context(ts)
            return

        # Drop the worker_id added by MultiTokenizer
        orig_rid = self.rid.split("_")[-1]
        role = "" if self.role == "unified" else self.role
        attrs = {"rid": orig_rid, "module": f"sglang::{self.module_name}", **_code_attributes()}
        if self.bootstrap_room:
            attrs["bootstrap_room"] = str(hex(self.bootstrap_room))
        root_span = tracer.start_span(
            name=f"{role} Req {orig_rid[:8]}",
            start_time=ts,
            context=external_trace_context,
            attributes=attrs,
        )

        self.root_span = root_span
        self.root_span_context = trace.set_span_in_context(root_span)

        # create thread context and thread span
        self.thread_context = self.__create_thread_context(ts)

    def trace_req_finish(
        self, ts: Optional[int] = None, attrs: Optional[Dict[str, Any]] = None
    ):
        if not self.tracing_enable or _tracing_stopped():
            return

        if not self.root_span and not _ettr_enabled():
            return

        ts = ts or get_cur_time_ns()

        # End all unclosed thread spans.
        self.abort()

        if self.root_span is not None:
            if attrs:
                self.root_span.set_attributes(attrs)
            self.root_span.end(end_time=ts)
            self.root_span = None

    def __check_fast_return(self, level=None):
        if not self.tracing_enable or _tracing_stopped():
            return True

        if not self.thread_context:
            return True

        if _ettr_enabled() and self.thread_context.thread_info.thread_label == "Tokenizer":
            return True

        if level and level > self.trace_level:
            return True

        return False

    def trace_slice_start(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
    ):
        if self.__check_fast_return(level):
            return
        if _ettr_enabled() and name in ("prefill_waiting", "decode_waiting"):
            return

        ts = ts or get_cur_time_ns()

        cur_slice = TraceSliceContext(
            slice_name=name,
            start_time_ns=ts,
            level=level,
            attrs={},
            events=[],
        )

        parent_span = self.thread_context.thread_span
        prev_span_context = None
        if not self.thread_context.cur_slice_stack:
            if self.last_span_context:
                prev_span_context = self.last_span_context
        else:
            parent_span = self.thread_context.cur_slice_stack[-1].span

        parent_span_context = trace.set_span_in_context(parent_span) if parent_span else self.root_span_context

        span = tracer.start_span(
            name=cur_slice.slice_name,
            start_time=cur_slice.start_time_ns,
            context=parent_span_context,
            attributes={**((self.thread_context.thread_attrs or {}) if parent_span is None else {}), **_code_attributes()},
        )
        cur_slice.span = span

        if prev_span_context:
            span.add_link(prev_span_context)

        self.thread_context.cur_slice_stack.append(cur_slice)

    def trace_slice_end(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        thread_finish_flag: bool = False,
    ):
        if self.__check_fast_return(level):
            return

        if _ettr_enabled() and name in ("prefill_waiting", "decode_waiting"):
            if thread_finish_flag:
                self.abort(ts)
            return

        if not self.thread_context.cur_slice_stack:
            logger.warning(
                f"No matching with the SLICE_START event {name} is required."
            )
            return

        cur_slice = self.thread_context.cur_slice_stack[-1]
        ts = ts or get_cur_time_ns()

        # check if slice_name matching and level matching
        # unlikely path, excepting error API usage
        if cur_slice.slice_name != name or cur_slice.level != level:
            logger.warning(
                f"Slice name mismatch: {name} != {cur_slice.slice_name} or level mismatch: {level} != {cur_slice.level}"
            )
            self.thread_context.cur_slice_stack.pop()
            return

        span = cur_slice.span

        if attrs:
            span.set_attributes(attrs)

        if self.events_cache:
            new_events_cache = []
            for event in self.events_cache:
                if event.ts >= cur_slice.start_time_ns and event.ts < ts:
                    span.add_event(
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                else:
                    new_events_cache.append(event)
            self.events_cache = new_events_cache

        span.end(end_time=ts)

        self.thread_context.cur_slice_stack.pop()
        # only for first level slice
        if not self.thread_context.cur_slice_stack:
            self.last_span_context = span.get_span_context()

        if thread_finish_flag:
            self.abort(ts)

    def trace_slice(
        self,
        slice: TraceSliceContext,
        thread_finish_flag: bool = False,
    ):
        if self.__check_fast_return(slice.level):
            return
        if _ettr_enabled() and slice.slice_name in ("prefill_waiting", "decode_waiting"):
            if thread_finish_flag:
                self.abort(slice.end_time_ns)
            return

        parent_span = self.thread_context.thread_span
        prev_span_context = None
        if not self.thread_context.cur_slice_stack:
            if self.last_span_context:
                prev_span_context = self.last_span_context
        else:
            parent_span = self.thread_context.cur_slice_stack[-1].span

        parent_span_context = trace.set_span_in_context(parent_span) if parent_span else self.root_span_context

        span = tracer.start_span(
            name=slice.slice_name,
            start_time=slice.start_time_ns,
            context=parent_span_context,
            attributes={**((self.thread_context.thread_attrs or {}) if parent_span is None else {}), **_code_attributes()},
        )

        if prev_span_context:
            span.add_link(prev_span_context)

        if slice.attrs:
            span.set_attributes(slice.attrs)

        if slice.events:
            for event in slice.events:
                span.add_event(
                    name=event.event_name, timestamp=event.ts, attributes=event.attrs
                )

        if self.events_cache:
            new_events_cache = []
            for event in self.events_cache:
                if event.ts >= slice.start_time_ns and event.ts < slice.end_time_ns:
                    span.add_event(
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                else:
                    new_events_cache.append(event)
            self.events_cache = new_events_cache

        span.end(end_time=slice.end_time_ns)

        # only for first level slice
        if not self.thread_context.cur_slice_stack:
            self.last_span_context = span.get_span_context()

        if thread_finish_flag:
            self.abort(slice.end_time_ns)

    # Add event to the current slice on the same thread with the same rid.
    def trace_event(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
        attrs: Dict[str, Any] = None,
    ):
        if self.__check_fast_return(level):
            return

        ts = ts or get_cur_time_ns()

        if attrs is None:
            attrs = {}
        self.events_cache.append(TraceEvent(name, ts, attrs))

    def trace_set_root_attrs(self, attrs: Dict[str, Any]):
        if not self.tracing_enable or _tracing_stopped():
            return

        if self.root_span:
            self.root_span.set_attributes(attrs)

    def trace_set_thread_attrs(self, attrs: Dict[str, Any]):
        if self.__check_fast_return():
            return

        if self.thread_context.thread_span:
            self.thread_context.thread_span.set_attributes(attrs)
        elif self.thread_context.thread_attrs is not None:
            self.thread_context.thread_attrs.update(attrs)
            for slice in self.thread_context.cur_slice_stack:
                slice.span.set_attributes(attrs)

    def abort(self, ts=None, abort_info: Optional[Dict] = None):
        if not self.tracing_enable or _tracing_stopped():
            return

        # close all slice spans (unlikely, except error API usage)
        ts = ts or get_cur_time_ns()
        if abort_info and not isinstance(abort_info, dict):
            abort_info = abort_info.to_json()

        def record_abort(span):
            if not abort_info or not span.is_recording():
                return
            span.set_attributes(abort_info)
            if _ettr_enabled() and (
                abort_info.get("status_code") == 499
                or abort_info.get("reason") == "have unclosed span, auto closed"
                or (abort_info.get("reason") == "Aborted" and abort_info.get("status_code") in (None, 499))
            ):
                from relax.utils.tracing_shutdown import mark_interrupted

                mark_interrupted(span, "aborted", str(abort_info.get("reason") or abort_info.get("message")
                                                         or "generation_cancelled_http_499"))
            else:
                span.set_status(Status(StatusCode.ERROR))

        if (_ettr_enabled() and self.thread_context is not None
                and self.thread_context.thread_info.thread_label == "Tokenizer"):
            self.events_cache = []
            self.thread_context = None
            return

        # Only the Tokenizer owner holds this root. Copies must not end or
        # modify another process's Req; normal thread completion also calls abort.
        if abort_info and _ettr_enabled() and self.root_span is not None:
            record_abort(self.root_span)
        if self.thread_context is None:
            if abort_info and _ettr_enabled() and self.root_span is not None:
                self.root_span.end(end_time=ts)
                self.root_span = None
            return
        if self.thread_context.thread_span is None and (abort_info or self.events_cache):
            # With no Scheduler wrapper, retain diagnostics on the active slice.
            # If all slices ended, emit an instantaneous diagnostic at this boundary.
            stack = self.thread_context.cur_slice_stack
            diagnostic = stack[-1].span if stack else tracer.start_span(
                "scheduler.abort" if abort_info else "scheduler.events",
                start_time=ts, context=self.root_span_context,
                attributes={**(self.thread_context.thread_attrs or {}), **_code_attributes()},
            )
            if abort_info:
                record_abort(diagnostic)
            for event in self.events_cache:
                diagnostic.add_event(event.event_name, timestamp=event.ts, attributes=event.attrs)
            self.events_cache = []
            if not stack:
                diagnostic.end(end_time=ts)
        while len(self.thread_context.cur_slice_stack) > 0:
            if self.thread_context.cur_slice_stack[-1].span:
                self.thread_context.cur_slice_stack[-1].span.end(end_time=ts)
            self.thread_context.cur_slice_stack.pop()

        # set abort info into thread span
        if self.thread_context.thread_span:
            if abort_info:
                record_abort(self.thread_context.thread_span)

            if self.events_cache:
                for event in self.events_cache:
                    self.thread_context.thread_span.add_event(
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                self.events_cache = []

            self.thread_context.thread_span.end(end_time=ts)
        self.thread_context = None

        if abort_info and _ettr_enabled() and self.root_span is not None:
            self.root_span.end(end_time=ts)
            self.root_span = None

    def __del__(self):
        try:
            ts = get_cur_time_ns()
            self.abort(ts=ts, abort_info={"reason": "have unclosed span, auto closed"})
            if _ettr_enabled() and getattr(self, "root_span", None) is not None:
                # The HTTP cancellation path may discard the owner without
                # trace_req_finish. Close the actual root, preserving its ID.
                from relax.utils.tracing_shutdown import mark_interrupted

                mark_interrupted(self.root_span, "aborted", "request_context_discarded_without_finish")
                self.root_span.end(end_time=ts)
                self.root_span = None
        except Exception:
            # Best effort during interpreter teardown; never change a request result.
            pass


@dataclass
class TraceNullContext:
    tracing_enable: bool = False

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_USAGE_CACHED_TOKENS = "gen_ai.usage.cached_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"


# Relax's opt-in lifecycle overlay. These wrappers do not import Relax or the
# SDK when disabled, and never create a span for an individual weight bucket.
def trace_lifecycle(name):
    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if os.environ.get("RELAX_OTEL_ENABLED") != "1":
                return function(*args, **kwargs)
            from relax.utils.tracing_sglang_lifecycle import lifecycle

            with lifecycle(sys.modules[__name__], name, function):
                return function(*args, **kwargs)
        return wrapper
    return decorate


def trace_weight_update(function):
    @functools.wraps(function)
    def wrapper(self, recv_req, *args, **kwargs):
        if os.environ.get("RELAX_OTEL_ENABLED") != "1":
            return function(self, recv_req, *args, **kwargs)
        from relax.utils.tracing_sglang_lifecycle import weight_result, weight_update

        with weight_update(sys.modules[__name__], self, recv_req, function):
            result = function(self, recv_req, *args, **kwargs)
        weight_result(self, result)
        return result
    return wrapper


def trace_weight_resume(function):
    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        if os.environ.get("RELAX_OTEL_ENABLED") != "1":
            return function(self, *args, **kwargs)
        from relax.utils.tracing_sglang_lifecycle import weight_resume

        try:
            result = function(self, *args, **kwargs)
        except BaseException as exc:
            weight_resume(self.weight_updater, type(exc).__name__)
            raise
        weight_resume(self.weight_updater)
        return result
    return wrapper
