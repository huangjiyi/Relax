# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise actual RayTrainGroup dispatch with fresh-process trace consumers."""

import ast
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from relax.utils import tracing


def group_class():
    path = Path(__file__).resolve().parents[2] / "relax/distributed/ray/actor_group.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayTrainGroup")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("_trace_context", "train_fully_async")]
    scope = {"remote_context": tracing.remote_context}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), scope)
    return scope["RayTrainGroup"]


@pytest.fixture
def output(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    monkeypatch.setenv("RELAX_OTEL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_OTEL_RUN_ID", "async-rpc-test")
    monkeypatch.setenv("RELAX_OTEL_EXPORTER", "file")
    monkeypatch.setenv("RELAX_OTEL_LEVEL", "3")
    monkeypatch.setenv("RELAX_OTEL_DETAIL_STEPS", "0-9")
    monkeypatch.delenv("RELAX_OTEL_TRACEPARENT", raising=False)
    yield tmp_path
    tracing.finish_run()
    if tracing._provider is not None:
        tracing._provider.shutdown()
    if tracing._stream is not None:
        tracing._stream.close()
    tracing._provider = tracing._provider_pid = tracing._stream = tracing._root_span = None
    tracing._initialization_failed = False


CHILD = """
import json, random, sys
from relax.utils import tracing
request = json.load(sys.stdin)
class Worker:
    role = 'actor'
    def __init__(self, rank): self._rank = rank
    @tracing.traced('train.worker', step_arg='rollout_id')
    def train_async(self, rollout_id, *, _trace_context=None):
        with tracing.stage('train.compute'):
            pass
state = random.getstate()
Worker(request['rank']).train_async(request['step'], **request['kwargs'])
assert random.getstate() == state
assert tracing.flush()
"""


def test_fully_async_dispatch_carries_parent_rank_and_step_across_processes(output):
    group = group_class()()
    group.args = SimpleNamespace(train_backend="megatron")

    def handler(rank):
        def remote(step, **kwargs):
            env = dict(os.environ)
            env.pop("RELAX_OTEL_TRACEPARENT", None)
            subprocess.run(
                [sys.executable, "-c", CHILD], input=json.dumps({"step": step, "rank": rank, "kwargs": kwargs}),
                text=True, env=env, check=True, timeout=30,
            )
        return SimpleNamespace(train_async=SimpleNamespace(remote=remote))

    group._actor_handlers = [handler(0), handler(1)]
    state = random.getstate()
    with tracing.stage("train.step", **{"relax.step": 3}):
        group.train_fully_async(3)
    assert random.getstate() == state
    spans = [json.loads(line) for file in output.glob("*.jsonl") for line in file.read_text().splitlines()]
    parent = next(s for s in spans if s["name"] == "train.step")
    workers = [s for s in spans if s["name"] == "train.worker"]
    assert len(workers) == 2
    assert {s["attributes"]["relax.rank"] for s in workers} == {0, 1}
    assert all(s["parent_id"] == parent["context"]["span_id"] for s in workers)
    assert {s["context"]["trace_id"] for s in spans} == {parent["context"]["trace_id"]}
    assert all(s["attributes"]["relax.step"] == 3 for s in spans)
    by_id = {s["context"]["span_id"]: s for s in spans}
    for span in [s for s in spans if s["name"] == "train.compute"]:
        assert by_id[span["parent_id"]]["name"] == "train.worker"


def test_disabled_or_other_backend_keeps_original_rpc_signature(output, monkeypatch):
    group = group_class()()
    group._actor_handlers = [SimpleNamespace(train_async=SimpleNamespace(remote=lambda step: step))]
    group.args = SimpleNamespace(train_backend="fsdp")
    with tracing.stage("train.step", **{"relax.step": 2}):
        assert group.train_fully_async(2) == [2]
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "0")
    group.args = SimpleNamespace(train_backend="megatron")
    assert group.train_fully_async(4) == [4]
