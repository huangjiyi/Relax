# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from io import StringIO

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from relax.utils.tracing_policy import CollectionProcessor, carrier_context


@pytest.mark.parametrize("duration,keep", [(999999, False), (1000000, True), (50000000, True)])
def test_conditional_parent_resolved_before_both_exporters(duration, keep):
    provider = TracerProvider(shutdown_on_exit=False)
    output, memory = StringIO(), InMemorySpanExporter()
    policy = CollectionProcessor()
    policy.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=output)))
    policy.add_span_processor(BatchSpanProcessor(memory))
    provider.add_span_processor(policy)
    tracer = provider.get_tracer("policy-test")
    with tracer.start_as_current_span("root") as root:
        parent = tracer.start_span("worker.init", start_time=0)
        with trace.use_span(parent):
            assert trace.get_current_span(carrier_context()).get_span_context() == root.context
            with tracer.start_as_current_span("child") as child:
                with tracer.start_as_current_span("worker.init.checkpoint_prepare", start_time=0) as inner:
                    with tracer.start_as_current_span("grandchild"):
                        pass
                    inner.end(end_time=3)
            assert not memory.get_finished_spans()
            copied = trace.set_span_in_context(parent)
        parent.end(end_time=duration)
        with tracer.start_as_current_span("late_child", context=copied):
            pass
        assert trace.get_current_span(carrier_context(copied)).get_span_context() == (parent.context if keep else root.context)
    provider.force_flush()
    spans = memory.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert ("worker.init" in by_name) == keep
    assert by_name["child"].parent == (parent.context if keep else root.context)
    assert by_name["grandchild"].parent == child.context
    assert by_name["late_child"].parent == (parent.context if keep else root.context)
    assert "worker.init.checkpoint_prepare" not in by_name
    decoder = json.JSONDecoder()
    remaining, file_spans = output.getvalue().strip(), []
    while remaining:
        value, end = decoder.raw_decode(remaining)
        file_spans.append(value)
        remaining = remaining[end:].strip()
    assert {s["context"]["span_id"]: s["parent_id"] for s in file_spans} == {
        f"0x{s.context.span_id:016x}": f"0x{s.parent.span_id:016x}" if s.parent else None for s in spans}
    assert not policy.groups and not policy.owner
    provider.shutdown()


@pytest.mark.parametrize("error", [False, True])
def test_iteration_only_retains_real_error(error):
    provider = TracerProvider(shutdown_on_exit=False)
    policy, output = CollectionProcessor(), InMemorySpanExporter()
    policy.add_span_processor(SimpleSpanProcessor(output))
    provider.add_span_processor(policy)
    tracer = provider.get_tracer("policy-test")
    with tracer.start_as_current_span("train.compute") as root:
        with tracer.start_as_current_span("megatron.iteration") as iteration:
            with tracer.start_as_current_span("megatron.forward_backward"):
                pass
            if error:
                iteration.set_status(trace.StatusCode.ERROR)
    by_name = {s.name: s for s in output.get_finished_spans()}
    assert ("megatron.iteration" in by_name) == error
    assert by_name["megatron.forward_backward"].parent == (iteration.context if error else root.context)
    provider.shutdown()


def test_shutdown_before_candidate_registration_does_not_leave_pending_group():
    from relax.utils.tracing_shutdown import ActiveSpanProcessor

    provider = TracerProvider(shutdown_on_exit=False)
    active = ActiveSpanProcessor(provider)
    provider.add_span_processor(active)
    policy, output = CollectionProcessor(), InMemorySpanExporter()
    policy.add_span_processor(SimpleSpanProcessor(output))
    provider.add_span_processor(policy)
    active.close("test_shutdown")
    span = provider.get_tracer("policy-test").start_span("worker.init")
    assert not span.is_recording()
    assert not policy.groups and not policy.owner and not policy.open
    provider.shutdown()
