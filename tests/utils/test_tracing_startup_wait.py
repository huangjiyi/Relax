# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check service startup boundaries, causal context and opt-out behavior."""

import asyncio
import random

import pytest
from opentelemetry import trace

from relax.utils import tracing
from tests.utils.test_tracing_async_phases import records as _records


records = _records


class Service:
    @tracing.traced("service.init", role="rollout", startup_wait="rollout.wait_start")
    def __init__(self, fail=False):
        if fail:
            raise ValueError("init failed")
        self.calls = 0

    @tracing.traced("service.run", role="rollout", finish_startup_wait=True)
    async def run(self, *, _trace_context=None):
        self.calls += 1
        with tracing.stage("rollout.step"):
            await asyncio.sleep(0)
        return self.calls


def test_startup_wait_uses_observed_boundaries_without_context_leak(records):
    rng = random.getstate()
    with tracing.stage("job.run") as parent:
        service = Service()
        assert trace.get_current_span() is parent
        with tracing.stage("weights.initialize"):
            assert not any(s['name'] == 'rollout.wait_start' for s in records())
        assert asyncio.run(service.run()) == 1
        assert asyncio.run(service.run()) == 2
        assert trace.get_current_span() is parent
    spans = records()
    init = next(s for s in spans if s['name'] == 'service.init')
    runs = [s for s in spans if s['name'] == 'service.run']
    waits = [s for s in spans if s['name'] == 'rollout.wait_start']
    assert len(waits) == 1
    wait = waits[0]
    assert wait['start_time'] == init['end_time']
    assert wait['end_time'] == runs[0]['start_time']
    assert wait['parent_id'] == init['parent_id']
    assert {link['context']['span_id'] for link in wait['links']} == {
        init['context']['span_id'], runs[0]['context']['span_id']}
    assert wait['attributes']['code.function.name'].endswith('Service.__init__')
    assert 'relax.span.description' not in wait['attributes']
    assert 'relax.step' not in wait['attributes']
    assert all(s['parent_id'] in {r['context']['span_id'] for r in runs}
               for s in spans if s['name'] == 'rollout.step')
    assert random.getstate() == rng


def test_no_startup_wait_for_failed_init_or_unstarted_service(records):
    with pytest.raises(ValueError):
        Service(fail=True)
    Service()
    assert not any(s['name'] == 'rollout.wait_start' for s in records())


@pytest.mark.parametrize('settings', [{'RELAX_OTEL_ENABLED': '0'}, {'RELAX_OTEL_LEVEL': '1'}])
def test_startup_wait_disabled_and_level_one_preserve_behavior(records, monkeypatch, settings):
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    service = Service()
    assert not hasattr(service, '_otel_startup_wait')
    assert asyncio.run(service.run()) == 1
    assert not any(s['name'] == 'rollout.wait_start' for s in records())


def test_unsampled_parent_preserves_init_and_run(records, monkeypatch):
    monkeypatch.setenv("RELAX_OTEL_TRACEPARENT", "00-" + "1" * 32 + "-" + "2" * 16 + "-00")
    service = Service()
    assert not hasattr(service, "_otel_startup_wait")
    assert asyncio.run(service.run()) == 1
    assert records() == []


def test_unsampled_first_run_does_not_extend_wait_to_a_later_run(records):
    with tracing.stage("job.run"):
        service = Service()
        carrier = {"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-00"}
        assert asyncio.run(service.run(_trace_context=carrier)) == 1
        assert asyncio.run(service.run()) == 2
    assert service._otel_startup_wait is None
    assert not any(s['name'] == 'rollout.wait_start' for s in records())
