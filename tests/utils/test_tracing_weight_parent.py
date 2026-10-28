# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check initial weight-sync context across the real service and group dispatch."""

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace

from relax.utils import tracing
from tests.utils.test_tracing_async_phases import source_class
from tests.utils.test_tracing_async_rpc import output as _output


output = _output


CHILD = """
import json, sys
from types import SimpleNamespace
from relax.utils import tracing
from tests.utils.test_tracing_async_phases import source_class
from tests.utils.test_tracing_async_phases import worker
request = json.load(sys.stdin)
group_cls = source_class('relax/distributed/ray/actor_group.py', 'RayTrainGroup',
    {'_trace_context', 'update_weights_fully_async'},
    {'remote_context': tracing.remote_context, 'ray': SimpleNamespace(get=lambda refs: refs)})
group = group_cls()
group.args = SimpleNamespace(train_backend='megatron')
group._actor_handlers = []
for rank in range(2):
    actor = worker(rank, [])
    group._actor_handlers.append(SimpleNamespace(update_weights_fully_async=SimpleNamespace(
        remote=actor.update_weights_fully_async)))
component_cls = source_class('relax/components/actor.py', 'Actor', {'update_weights_fully_async'},
    {'traced': tracing.traced})
component = component_cls()
component.actor_model = group
component.update_weights_fully_async(**request)
assert tracing.flush()
"""


def test_initial_sync_has_real_parent_across_service_process(output):
    def remote(**kwargs):
        env = dict(os.environ)
        env.pop("RELAX_OTEL_TRACEPARENT", None)
        subprocess.run(
            [sys.executable, "-B", "-c", CHILD], input=json.dumps(kwargs), text=True, env=env, check=True, timeout=40
        )
        return "unawaited-remote-result"

    cls = source_class(
        "relax/core/service.py", "Service", {"update_weights_fully_async"}, {"remote_context": tracing.remote_context}
    )
    service = cls()
    service.handle = SimpleNamespace(update_weights_fully_async=SimpleNamespace(remote=remote))
    with tracing.stage("controller.loop"):
        assert asyncio.run(service.update_weights_fully_async()) == "unawaited-remote-result"
    spans = [json.loads(line) for path in output.glob("*.jsonl") for line in path.read_text().splitlines()]
    by_id = {s["context"]["span_id"]: s for s in spans}
    dispatch = next(s for s in spans if s["name"] == "weights.initialize")
    assert by_id[dispatch["parent_id"]]["name"] == "controller.loop"
    workers = [s for s in spans if s["name"] == "weights.sync"]
    assert len(workers) == 2
    assert all(s["parent_id"] == dispatch["context"]["span_id"] for s in workers)
    assert len({s["context"]["trace_id"] for s in spans}) == 1


def test_non_megatron_initial_dispatch_preserves_rpc_signature(output, monkeypatch):
    cls = source_class(
        "relax/distributed/ray/actor_group.py",
        "RayTrainGroup",
        {"_trace_context", "update_weights_fully_async"},
        {"remote_context": tracing.remote_context, "ray": SimpleNamespace(get=lambda x: x)},
    )
    group = cls()
    calls = []
    group._actor_handlers = [
        SimpleNamespace(
            update_weights_fully_async=SimpleNamespace(
                remote=lambda step, rollout_only, actor_fwd_only: calls.append((step, rollout_only, actor_fwd_only))
            )
        )
    ]
    group.args = SimpleNamespace(train_backend="fsdp")
    with tracing.stage("weights.initialize"):
        group.update_weights_fully_async(0, rollout_only=True)
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "0")
    group.args = SimpleNamespace(train_backend="megatron")
    group.update_weights_fully_async(0, actor_fwd_only=True)
    assert calls == [(0, True, False), (0, False, True)]
