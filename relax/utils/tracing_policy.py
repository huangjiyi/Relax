# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Permanent collection policy shared by file and OTLP processors.

Only undecided local subtrees are buffered, never the run. RPC carriers skip an
undecided parent so another process cannot refer to an ID we may later remove.
"""

import copy
import os
import threading

REMOVED = frozenset({
    "prefill_waiting", "decode_waiting", "megatron.scheduler",
    "rollout.check_completion", "rollout.report_progress", "rollout.save_data",
    "rollout.finalize_step", "rollout.group", "rollout.sample",
    "worker.init.model_optimizer", "controller.init", "controller.init.services",
    "controller.init.service_phase", "weights.send", "weights.receiver_ack",
})
SHORT = frozenset({"worker.init", "rollout.wait_resume", "worker.init.checkpoint_prepare",
                   "sglang.init.ipc", "rollout.init.checkpoint_prepare"})
_candidates = {}
_resolved = {}
_lock = threading.RLock()
_metadata = {}


def register_native_metadata(values):
    _metadata[(os.getpid(), threading.get_native_id())] = {
        **values, "relax.metadata.source": "native_process",
    }


def carrier_context(context=None):
    from opentelemetry import trace

    current = trace.get_current_span(context).get_span_context()
    with _lock:
        while current is not None:
            key = (os.getpid(), current.trace_id, current.span_id)
            if key in _candidates:
                current = _candidates[key]
            elif key in _resolved:
                current = _resolved[key]
            else:
                break
    return trace.set_span_in_context(trace.NonRecordingSpan(current) if current else trace.INVALID_SPAN, context)



class CollectionProcessor:
    """Resolve conditional parents once, before either exporter sees children."""

    def __init__(self):
        self.sinks = []
        self.lock = _lock
        self.groups = {}
        self.owner = {}
        self.open = {}

    def add_span_processor(self, processor):
        self.sinks.append(processor)

    def on_start(self, span, parent_context=None):
        from opentelemetry.sdk.resources import Resource

        if not span.is_recording():
            return
        values = _metadata.get((os.getpid(), threading.get_native_id()))
        if values and span.resource.attributes.get("service.name") == "sglang":
            span._resource = span.resource.merge(Resource(values))
        with self.lock:
            if not span.is_recording():
                return
            parent_context = span.parent
            while parent_context is not None:
                parent_key = (os.getpid(), parent_context.trace_id, parent_context.span_id)
                if parent_key not in _resolved:
                    break
                parent_context = _resolved[parent_key]
            if parent_context != span.parent:
                span._parent = parent_context
            key = (span.context.trace_id, span.context.span_id)
            parent = (span.parent.trace_id, span.parent.span_id) if span.parent else None
            candidate = span.name in SHORT or span.name in REMOVED or span.name == "megatron.iteration"
            if candidate:
                _candidates[(os.getpid(), *key)] = span.parent
            root = self.owner.get(parent)
            if root is None and candidate:
                root = key
                self.groups[root] = []
                self.open[root] = set()
            if root is not None:
                self.owner[key] = root
                self.open[root].add(key)

    def _on_ending(self, span):
        pass

    @staticmethod
    def _drop(span):
        if span.name in REMOVED:
            return True
        if span.name == "megatron.iteration":
            return span.status.status_code.name != "ERROR"
        return span.name in SHORT and span.end_time - span.start_time < 1_000_000

    def on_end(self, span):
        key = (span.context.trace_id, span.context.span_id)
        with self.lock:
            root = self.owner.get(key)
            if root is None:
                ready = [span]
            else:
                self.groups[root].append(span)
                self.open[root].discard(key)
                if self.open[root]:
                    return
                ready = self.groups.pop(root)
                del self.open[root]
                for item in ready:
                    self.owner.pop((item.context.trace_id, item.context.span_id), None)
            removed = {item.context.span_id: item.parent for item in ready if self._drop(item)}
            for item in ready:
                key = (os.getpid(), item.context.trace_id, item.context.span_id)
                if item.context.span_id in removed:
                    _resolved[key] = item.parent
                _candidates.pop(key, None)
            for item in ready:
                if item.context.span_id in removed:
                    continue
                parent = item.parent
                while parent is not None and parent.span_id in removed:
                    parent = removed[parent.span_id]
                if parent != item.parent:
                    item = copy.copy(item)
                    item._parent = parent
                for sink in self.sinks:
                    sink.on_end(item)

    def force_flush(self, timeout_millis=30000):
        return all([sink.force_flush(timeout_millis) for sink in self.sinks])

    def shutdown(self):
        for sink in self.sinks:
            sink.shutdown()
