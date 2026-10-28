# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the actual rollout loop with asynchronous CPU service doubles."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
from opentelemetry import trace

from relax.utils import tracing
from tests.utils.test_tracing_async_phases import source_class
from tests.utils.test_tracing_compact import records as _records


records = _records


REAL_SLEEP = asyncio.sleep


def rollout(monkeypatch, events, *, peer=False, fail=False, stop_wait=False):
    async def partitions():
        events.append("partitions")
        return len([e for e in events if e == "partitions"]) > 1

    async def generate(rollout_id, **kwargs):
        events.append(("generate", rollout_id))
        if fail:
            raise ValueError("generation failure")

    async def no_op(*args, **kwargs):
        events.append("remote")

    logger = SimpleNamespace(**{n: lambda *a, **k: None for n in ("info", "warning", "exception")})
    cls = source_class(
        "relax/components/rollout.py",
        "Rollout",
        {"run", "_async_run", "_maybe_save_data", "_run_eval_with_mark"},
        {
            "traced": tracing.traced,
            "stage": tracing.stage,
            "remote_context": tracing.remote_context,
            "asyncio": asyncio,
            "satisfy_staleness": lambda ready, *a: ready,
        },
    )
    obj = cls()
    obj.config = SimpleNamespace(
        rollout_global_dataset=True,
        num_rollout=2,
        max_staleness=2,
        eval_interval=None,
        eval_prompt_data=None,
        offload_rollout=peer,
        fully_async=not peer,
        use_health_check=False,
        save=None,
        save_interval=None,
    )
    obj._run_task = None
    obj._stop_event = asyncio.Event()
    obj.step = 0
    obj.status = "paused"
    obj._logger = logger
    obj.rollout_manager = SimpleNamespace(
        load=SimpleNamespace(remote=no_op),
        generate=SimpleNamespace(remote=generate),
        offload=SimpleNamespace(remote=no_op),
    )
    obj.healthy = SimpleNamespace(
        report_error=SimpleNamespace(remote=lambda *a: events.append("error")),
        update_heartbeat=SimpleNamespace(remote=lambda *a: events.append("heartbeat")),
    )
    obj.data_system_client = SimpleNamespace(async_get_partition_list=partitions)
    obj._peer_barrier = None

    async def pending(step):
        events.append("peer")
        return [("actor", 0)] if events.count("peer") == 1 else []

    if peer:
        obj._peer_barrier = SimpleNamespace(is_empty=lambda: False, list_pending_async=pending)

    async def complete(partition):
        events.append("complete")
        return True

    obj._async_check_partition_production_complete = complete
    obj.eval_handler = no_op()

    async def sleep(delay):
        events.append("sleep")
        obj.status = "running"
        if stop_wait and "partitions" in events:
            obj._stop_event.set()
        await REAL_SLEEP(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setitem(sys.modules, "relax.engine.sft.runtime", SimpleNamespace(is_offline_mode=lambda args: False))
    return obj


@pytest.mark.parametrize("peer", [False, True])
async def test_real_loop_covers_waits_and_preserves_operations(monkeypatch, records, peer):
    events = []
    obj = rollout(monkeypatch, events, peer=peer)
    await obj.run()
    enabled_events = events[:]
    assert obj.step == 2
    spans = records()
    root = next(s for s in spans if s["name"] == "service.run")
    by_name = {}
    for span in spans:
        by_name.setdefault(span["name"], []).append(span)
        assert "relax.span.description" not in span["attributes"]
        if span["name"].startswith("rollout.") and span["name"] != "rollout.offload":
            assert span["parent_id"] == root["context"]["span_id"]
    for name in ("rollout.wait_staleness",):
        assert {s["attributes"]["relax.step"] for s in by_name[name]} == {0, 1}
    assert not {"rollout.report_progress", "rollout.save_data", "rollout.finalize_step"} & by_name.keys()
    assert "rollout.check_completion" not in by_name
    assert len(by_name.get("rollout.wait_resume", [])) <= 3
    assert len(by_name["rollout.wait_eval"]) == len(by_name["rollout.restore_data"]) == 1
    assert ("rollout.offload" in by_name) == peer
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "0")
    events.clear()
    obj = rollout(monkeypatch, events, peer=peer)
    await obj.run()
    assert events == enabled_events


@pytest.mark.parametrize("mode", ["stop", "error"])
async def test_wait_stop_and_error_keep_original_control_flow(monkeypatch, records, mode):
    events = []
    obj = rollout(monkeypatch, events, fail=mode == "error", stop_wait=mode == "stop")
    if mode == "error":
        with pytest.raises(ValueError, match="generation failure"):
            await obj.run()
        assert any(s["status"]["status_code"] == "ERROR" for s in records())
    else:
        await obj.run()
        assert obj.step == 0
        assert any(s["name"] == "rollout.wait_staleness" for s in records())
    assert not trace.get_current_span().get_span_context().is_valid
