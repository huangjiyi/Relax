"""Contract tests for compact worker metadata and real Python code locations."""

import inspect
import json
import os
import random
import time
from pathlib import Path

import pytest

from relax.utils import tracing


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


def test_metadata_only_on_init_including_nested_child_and_future_span(records):
    before = random.getstate()
    metadata = {"relax.tp_rank": 1, "relax.tp_ranks": [0, 1, 2, 3], "relax.world_size": 8}
    with tracing.stage("worker.init", **{"relax.rank": 1}):
        tracing.set_process_attributes(metadata)
        time.sleep(.002)  # Actual initialization is retained above the 1ms threshold.
        with tracing.stage("nested.child"):
            pass
    with tracing.stage("later"):
        pass
    spans = {s["name"]: s for s in records()}
    assert all(spans['worker.init']['attributes'][k] == v for k, v in metadata.items())
    for name in ('nested.child', 'later'):
        assert not set(metadata) & spans[name]['attributes'].keys()
        assert 'relax.detail_steps' not in spans[name]['attributes']
    assert random.getstate() == before


def test_contextmanager_uses_business_callsite_and_does_not_inherit_parent_code(records):
    with tracing.stage("parent"):
        line = inspect.currentframe().f_lineno + 1
        with tracing.stage("child"):
            pass
    spans = {s['name']: s for s in records()}
    attrs = spans['child']['attributes']
    assert attrs['code.file.path'] == __file__
    assert attrs['code.line.number'] == line
    assert attrs['code.function.name'].endswith('test_contextmanager_uses_business_callsite_and_does_not_inherit_parent_code')
    assert attrs['code.line.number'] != spans['parent']['attributes']['code.line.number']


async def test_async_decorator_reports_target_function_and_keeps_context(records):
    @tracing.traced('decorated', step_arg='step')
    async def task(step):
        with tracing.stage('inside'):
            pass
    await task(0)
    spans = {s['name']: s for s in records()}
    assert spans['decorated']['attributes']['code.function.name'].endswith('.task')
    assert spans['decorated']['attributes']['relax.code.location'] == 'function_definition'
    assert spans['inside']['attributes']['relax.step'] == 0
    assert spans['inside']['attributes']['code.function.name'].endswith('.task')
    assert spans['inside']['attributes']['relax.code.location'] == 'callsite'


def test_root_has_code_and_run_settings_without_copying_them(records):
    tracing.start_run()
    with tracing.stage('child'):
        pass
    tracing.finish_run()
    spans = {s['name']: s for s in records()}
    assert spans['job.run']['attributes']['relax.detail_steps'] == '0'
    for s in spans.values():
        attrs = s['attributes']
        assert Path(attrs['code.file.path']).is_file()
        assert attrs['code.line.number'] > 0 and attrs['code.function.name']
    assert 'relax.detail_steps' not in spans['child']['attributes']


def test_megatron_wrapper_points_to_original_function(records):
    from relax.utils import tracing_megatron

    def forward_step(current_microbatch):
        return current_microbatch

    token = tracing_megatron._calls.set({'forward': 0, 'backward': 0})
    try:
        with tracing.stage('train', **{'relax.step': 0}):
            assert tracing_megatron._wrap(forward_step, 'forward')(2) == 2
    finally:
        tracing_megatron._calls.reset(token)
    span = next(s for s in records() if s['name'] == 'megatron.microbatch.forward')
    assert span['attributes']['code.function.name'].endswith('.forward_step')
    assert span['attributes']['code.line.number'] == forward_step.__code__.co_firstlineno


def test_decorator_resolves_real_timer_wrapper_to_business_source(records):
    from relax.utils.timer import timer

    def operation():
        return 42

    decorated = tracing.traced('timed')(timer(operation))
    assert decorated() == 42
    attrs = next(s for s in records() if s['name'] == 'timed')['attributes']
    assert attrs['code.file.path'] == operation.__code__.co_filename
    assert attrs['code.line.number'] == operation.__code__.co_firstlineno
    assert attrs['code.function.name'] == operation.__module__ + '.' + operation.__qualname__
