# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run real async worker/DCS method bodies with CPU transport doubles."""

import ast
import asyncio
import json
import os
import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from opentelemetry import trace

from relax.utils import async_utils, tracing
from relax.utils.logging_utils import get_logger
from relax.utils.timer import Timer, inverse_timer, timer


ROOT = Path(__file__).resolve().parents[2]
ACTOR = "relax/backends/megatron/actor.py"
CLIENT = "relax/distributed/checkpoint_service/client/engine.py"
UPDATER = "relax/backends/megatron/weight_update/update_weight_from_tensor.py"


def source_class(path: str, name: str, methods: set[str], scope: dict) -> type:
    """Keep production bodies, decorators and source lines; avoid GPU imports."""
    source = ROOT / path
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.bases, cls.decorator_list = [], []
    cls.body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods]
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    scope = {"__name__": path.removesuffix(".py").replace("/", "."), **scope}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope[name]


@pytest.fixture
def records(monkeypatch, tmp_path):
    for key, value in {
        "RELAX_OTEL_ENABLED": "1", "RELAX_OTEL_OUTPUT_DIR": str(tmp_path),
        "RELAX_OTEL_RUN_ID": "async-phases-test", "RELAX_OTEL_EXPORTER": "file",
        "RELAX_OTEL_LEVEL": "3", "RELAX_OTEL_DETAIL_STEPS": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("RELAX_OTEL_TRACEPARENT", raising=False)
    yield lambda: [json.loads(line) for p in tmp_path.glob("*.jsonl") for line in p.read_text().splitlines()]
    async_utils.shutdown_async_loop()
    tracing.finish_run()
    if tracing._provider is not None:
        tracing._provider.shutdown()
    if tracing._stream is not None:
        tracing._stream.close()
    tracing._provider = tracing._provider_pid = tracing._stream = tracing._root_span = None
    tracing._initialization_failed = False
    Timer().__init__()


def worker(rank: int, events: list, transfer_error: Exception | None = None):
    def event(name, result=None):
        span = trace.get_current_span()
        events.append((name, getattr(span, "name", None), threading.get_ident()))
        return result

    def gather(out, value, **kwargs):
        event("gather")
        out[:] = [value]

    dist = SimpleNamespace(
        get_rank=lambda *a, **k: rank, get_world_size=lambda **k: 1,
        barrier=lambda **k: event("barrier"), all_gather_object=gather,
        all_reduce=lambda *a, **k: event("all_reduce"), ReduceOp=SimpleNamespace(MAX=0),
    )
    logger = get_logger(__name__)
    common = {"traced": tracing.traced, "stage": tracing.stage, "timer": timer,
              "dist": dist, "get_gloo_group": lambda: "gloo", "logger": logger}

    async def topology(url):
        await asyncio.sleep(0)
        event("topology")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})

    def transfer(*flags):
        event("transfer")
        if transfer_error:
            raise transfer_error

    client = source_class(CLIENT, "CheckpointEngineClient", {"update_weights_for_rollout"}, common)()
    client.coordinator_url = "http://unused.invalid"
    client._http_client = SimpleNamespace(get=topology)
    client.rank = rank
    client._backend = SimpleNamespace(
        init_process_group_for_rollout=lambda data: event("connect"), update_weights_for_rollout=transfer,
        _lora_adapter_mode=False, _is_pp_src_rank=True,
        rollout_topology={0: {"metadata": {"actor_id": "engine-0"}}},
    )

    async def connect_ref(step):
        await asyncio.sleep(0)
        event("connect_ref")

    client.init_process_groups_for_actor_fwd_ref = connect_ref

    class Iterator:
        tq_wait_times = [0.01]

        def step(self, partition):
            event("iterator_step")

        def get_buffer(self):
            event("buffer")
            return {"total_lengths": [2], "loss_masks": [[1, 1]]}

    @tracing.traced("train.compute", step_arg="rollout_id")
    def compute(rollout_id, *args):
        event("compute")

    def http_get(url, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        event(endpoint)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"done": True} if endpoint == "is_eval_done" else True)

    scope = {
        **common, "Timer": Timer, "inverse_timer": inverse_timer, "torch": torch, "time": time, "os": os,
        "run": async_utils.run, "ray": SimpleNamespace(get=lambda ref, **kw: ref),
        "print_memory": lambda *a, **k: None, "requests": SimpleNamespace(get=http_get),
        "get_serve_url": lambda role: "http://unused.invalid/" + role,
        "is_offline_mode": lambda args: False, "StreamingTQIterator": Iterator,
        "mpu": SimpleNamespace(get_data_parallel_rank=lambda: 0, get_data_parallel_world_size=lambda **kw: 1,
                               get_data_parallel_group=lambda **kw: "dp"),
        "create_stream_dataloader": lambda *a, **k: event("create_iterator", ([Iterator()], 1)),
        "train": compute, "merge_dict_list": lambda data: data,
        "log_rollout_data": lambda *a: event("log_rollout"),
        "train_dump_utils": SimpleNamespace(save_debug_train_data=lambda *a, **kw: event("dump")),
        "evaluation_step_for_rollout": lambda args, step: step + 1,
        "log_perf_data": lambda *a, **kw: event("log_perf"),
        "tracking_utils": SimpleNamespace(flush_metrics=lambda *a: event("flush_metrics")),
        "compute_rollout_step": lambda args, step: step,
    }
    cls = source_class(ACTOR, "MegatronTrainRayActor", {
        "train_async", "update_weights_fully_async", "_check_services_health",
        "_wait_for_previous_eval", "_run_step_evaluation",
    }, scope)
    actor = cls()
    actor._rank, actor.role = rank, "actor"
    actor.args = SimpleNamespace(
        use_routing_replay=False, advantage_estimator="grpo", true_on_policy_mode=True,
        kl_coef=0, use_kl_loss=False, multimodal_keys=None, use_opd=False,
        num_rollout=10, rollout_http_timeout=1,
    )
    actor.data_iterator = None
    actor.model = actor.optimizer = actor.opt_param_scheduler = actor.tokenizer = actor.flops_counter = None
    actor.prof = SimpleNamespace(step=lambda **kw: event("profiler"))
    actor.checkpoint_engine_client = client
    actor.rollout_manager = SimpleNamespace(
        register_recovered_engines=SimpleNamespace(remote=lambda ids: event("publish")),
    )
    actor._weight_sync_lock = SimpleNamespace(
        acquire=SimpleNamespace(remote=lambda: event("acquire", True)),
        release=SimpleNamespace(remote=lambda: event("release")),
    )
    return actor


@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_all_steps_keep_phase_parent_rank_and_source_across_dcs_thread(records, monkeypatch, rank, level):
    monkeypatch.setenv("RELAX_OTEL_LEVEL", str(level))
    events = []
    actor = worker(rank, events)
    before = random.getstate()
    Timer().start("train_wait")
    for step in range(10):
        with tracing.stage("train.step", **{"relax.role": "actor", "relax.step": step}):
            actor.train_async(step)
    assert random.getstate() == before
    spans = records()
    by_id = {s["context"]["span_id"]: s for s in spans}
    assert len(by_id) == len(spans)
    for span in spans:
        if span["name"] == "train.step":
            continue
        attrs = span["attributes"]
        assert attrs["relax.rank"] == rank and attrs["relax.role"] == "actor"
        assert attrs["relax.step"] in range(10)
        parent = by_id[span["parent_id"]]
        assert parent["attributes"]["relax.step"] == attrs["relax.step"]
        assert span["context"]["trace_id"] == parent["context"]["trace_id"]
        assert span["status"]["status_code"] != "ERROR"
        assert Path(attrs["code.file.path"]).is_file()
        assert attrs["code.line.number"] > 0 and attrs["code.function.name"]
        assert not any(key in attrs for key in ("relax.tp_ranks", "relax.world_size", "relax.detail_steps"))
    required = {"train.worker", "train.compute", "weights.sync"}
    if level >= 2:
        required |= {"train.prepare_data", "train.postprocess", "train.wait_eval", "weights.wait_ready",
                     "weights.connect", "weights.transfer_apply", "train.finalize", "train.log_metrics"}
    for name in required:
        assert {s["attributes"]["relax.step"] for s in spans if s["name"] == name} == set(range(10))
    assert {s["name"] for s in spans} == required | {"train.step"}
    transfers = [event for event in events if event[0] == "transfer"]
    assert all(event[2] != threading.get_ident() for event in transfers)
    assert {event[1] for event in transfers} == {"weights.transfer_apply" if level >= 2 else "weights.sync"}
    for s in spans:
        if s["name"] in {"weights.connect", "weights.transfer_apply"}:
            assert by_id[s["parent_id"]]["name"] == "weights.sync"


def test_tracing_toggle_preserves_training_and_transport_call_order(records, monkeypatch):
    observed = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("RELAX_OTEL_ENABLED", enabled)
        Timer().__init__()
        events = []
        actor = worker(0, events)
        Timer().start("train_wait")
        for step in (0, 9):
            assert actor.train_async(step) is None
        observed.append([event[0] for event in events])
    assert observed[0] == observed[1]


def test_transfer_failure_preserves_exception_releases_lock_and_marks_parent(records):
    error = RuntimeError("transport failed")
    events = []
    actor = worker(0, events, transfer_error=error)
    with tracing.stage("train.worker", **{"relax.step": 9, "relax.role": "actor", "relax.rank": 0}):
        with pytest.raises(RuntimeError) as caught:
            actor.update_weights_fully_async(9)
        assert caught.value is error
        with tracing.stage("after.failure"):
            pass
    assert events[-1][0] == "release"
    spans = {s["name"]: s for s in records()}
    assert spans["weights.transfer_apply"]["status"]["status_code"] == "ERROR"
    assert spans["weights.sync"]["status"]["status_code"] == "ERROR"
    assert spans["after.failure"]["parent_id"] == spans["train.worker"]["context"]["span_id"]


@pytest.mark.parametrize("distributed", [False, True])
def test_weight_chunks_keep_transport_results_without_leaf_spans(records, distributed):
    waited = []
    cls = source_class(UPDATER, "UpdateWeightFromTensor", {"_send_hf_params", "_wait_for_ipc"}, {
        "traced": tracing.traced, "ray": SimpleNamespace(get=lambda refs: waited.extend(refs)),
        "_send_to_colocated_engine": lambda *a, **kw: (["ipc"], "keep-alive-buffer"),
        "update_weights_from_distributed": lambda *a: ["nccl"],
    })
    updater = cls()
    updater._ipc_engine = None if distributed else object()
    updater._ipc_gather_src = updater._ipc_gather_group = None
    updater._route_experts = False
    updater.use_distribute = distributed
    updater._is_distributed_src_rank = True
    updater._group_name, updater._model_update_groups, updater.weight_version = "group", {}, 1
    updater.distributed_rollout_engines = []
    with tracing.stage("weights.sync", **{"relax.step": 0}):
        with tracing.stage("weights.transfer_apply", level=2):
            for _ in range(5):
                refs, buffer = updater._send_hf_params([])
                assert refs == (["nccl"] if distributed else ["ipc"])
                assert buffer == (None if distributed else "keep-alive-buffer")
                updater._wait_for_ipc(refs)
    assert len(waited) == 5
    assert {s["name"] for s in records()} == {"weights.sync", "weights.transfer_apply"}
