"""Keep datasource identifiers stable across interleaved rollout tasks."""

import asyncio
import json
from types import SimpleNamespace

from relax.utils import tracing
import pytest


@pytest.fixture
def records(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    monkeypatch.setenv("RELAX_OTEL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_OTEL_RUN_ID", "compact-code-test")
    monkeypatch.setenv("RELAX_OTEL_EXPORTER", "file")
    monkeypatch.setenv("RELAX_OTEL_LEVEL", "3")
    monkeypatch.setenv("RELAX_OTEL_DETAIL_STEPS", "0")
    monkeypatch.delenv("RELAX_OTEL_TRACEPARENT", raising=False)
    tracing._process_attributes = {}
    tracing._process_attributes_pid = None
    yield lambda: [json.loads(line) for p in tmp_path.glob("*.jsonl") for line in p.read_text().splitlines()]
    tracing.finish_run()
    if tracing._provider is not None:
        tracing._provider.shutdown()
    if tracing._stream is not None:
        tracing._stream.close()
    tracing._provider = tracing._provider_pid = tracing._stream = tracing._root_span = None
    tracing._initialization_failed = False
    tracing._process_attributes = {}
    tracing._process_attributes_pid = None


async def test_rollout_identifiers_follow_sample_context_without_cross_task_leak(records):
    @tracing.traced('rollout.sample', level=3, sample_arg='sample')
    async def generate(sample):
        await asyncio.sleep(0)
        with tracing.stage('rollout.engine_request', level=3):
            await asyncio.sleep(0)
        with tracing.stage('rollout.reward', level=3):
            pass

    @tracing.traced('rollout.group', level=3, group_arg='group')
    async def group_task(group):
        await asyncio.gather(*(generate(s) for s in group))

    groups = [[SimpleNamespace(group_index=g, index=g * 2 + i) for i in range(2)] for g in (0, 7)]
    with tracing.stage('rollout.generate', **{'relax.step': 0}):
        await asyncio.gather(*(group_task(g) for g in groups))
        with tracing.stage('after.groups'):
            pass
    spans = records()
    assert not any(s['name'] in ('rollout.group', 'rollout.sample') for s in spans)
    lookup = {s['context']['span_id']: s for s in spans}
    for s in spans:
        if s['name'] in ('rollout.sample', 'rollout.engine_request', 'rollout.reward'):
            a = s['attributes']
            assert a['relax.group_index'] in (0, 7)
            assert a['relax.sample_index'] in (a['relax.group_index'] * 2, a['relax.group_index'] * 2 + 1)
            if s['name'] != 'rollout.sample':
                assert lookup[s['parent_id']]['name'] == 'rollout.generate'
        if s['name'] == 'after.groups':
            assert 'relax.group_index' not in s['attributes'] and 'relax.sample_index' not in s['attributes']


def test_all_step_detail_window_and_native_sampling_flag(records, monkeypatch):
    monkeypatch.setenv('RELAX_OTEL_DETAIL_STEPS', '0-9')
    for step in range(10):
        with tracing.stage('rollout.generate', **{'relax.step': step}):
            assert tracing.selected(3)
            assert int(tracing.request_context()['traceparent'][-2:], 16) & 1
            with tracing.stage('detail', level=3):
                pass
    with tracing.stage('rollout.generate', **{'relax.step': 10}):
        assert not tracing.selected(3)
        assert not int(tracing.request_context()['traceparent'][-2:], 16) & 1
    assert {s['attributes']['relax.step'] for s in records() if s['name'] == 'detail'} == set(range(10))
