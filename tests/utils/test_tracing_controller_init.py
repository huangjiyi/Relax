# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check exported startup parents through real serial and parallel Controller methods."""

import concurrent.futures
import contextvars
import threading
from enum import Enum
from types import SimpleNamespace

import pytest

from relax.utils import tracing
from relax.utils.logging_utils import get_logger
from tests.utils.test_tracing_async_phases import records as _records
from tests.utils.test_tracing_async_phases import source_class


records = _records


class Roles(str, Enum):
    actor = "actor"
    rollout = "rollout"

    def __str__(self) -> str:
        return self.value


@pytest.mark.parametrize("fully_async", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_startup_children_attach_to_job_without_controller_wrappers(records, monkeypatch, fully_async, enabled):
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1" if enabled else "0")
    calls = []
    rendezvous = threading.Barrier(2) if fully_async else None

    class Service:
        def __init__(self, cls, *, role, **kwargs):
            self.role = role

        @tracing.traced("service.deploy", level=2)
        def deploy(self):
            if rendezvous is not None:
                rendezvous.wait(timeout=5)
            calls.append(str(self.role))
            # Exercise the same startup carrier consumed by remote service constructors.
            self.carrier = tracing.child_runtime_environment({"env_vars": {"EXISTING": "1"}})
            with tracing.stage("service.init", level=2):
                pass

    def noop(*args, **kwargs):
        pass

    controller_type = source_class(
        "relax/core/controller.py", "Controller",
        {"__init__", "register_all_serve", "_start_services", "_run_service_phase", "_create_service_task"},
        {
            "traced": tracing.traced, "stage": tracing.stage, "Service": Service,
            "concurrent": concurrent, "contextvars": contextvars, "threading": threading,
            "logger": get_logger(__name__),
            "ServiceStartupPhase": SimpleNamespace(DEPLOY=SimpleNamespace(value="deploy")),
            "require_control_plane_resource": noop, "resolve_sft_num_rollout": noop,
            "HealthManager": lambda **kwargs: SimpleNamespace(status=None, mark_healthy=noop),
            "create_dcs_deployment": lambda **kwargs: (None, "test-coordinator"),
            "with_control_plane_affinity": lambda config, options: options,
            "validate_ppo_config": noop, "validate_sft_resource": noop,
            "maybe_start_managed_opd_teacher": lambda *args, **kwargs: (None, None),
            "resolve_sft_algo_key": lambda config: "test", "process_role": lambda config: Roles,
            "ALGOS": {"test": {role: object for role in Roles}},
            "register_extra_roles": lambda *args: [], "load_function": lambda path: object,
            "create_data_source_actor": lambda *args: object(),
            "relax_utils": SimpleNamespace(report_train_start=noop), "_build_train_start_config": lambda config: {},
            "_actor_rollout_pg_roles": lambda config: list(Roles),
            "capture_run_configuration": tracing.capture_run_configuration,
            "create_placement_group": lambda **kwargs: object(), "_uses_s3_model_prefetch": lambda config: False,
        },
    )
    controller_type._initialize_data_system = noop
    controller_type._maybe_resolve_num_rollout = noop
    controller_type._validate_gpu_resources = noop
    controller_type._report_error_to_metrics_service = noop
    config = SimpleNamespace(
        fully_async=fully_async, colocate=not fully_async, hybrid=False, enable_affinity=False,
        use_agentic_rollout=False, resource={role: (1, 8) for role in Roles}, data_source_path="test-source",
    )

    tracing.start_run()
    controller = controller_type(config, runtime_env={})
    tracing.finish_run()
    assert sorted(calls) == ["actor", "rollout"]
    if not fully_async:
        assert calls == ["actor", "rollout"]
    assert set(controller.serve_dict) == set(Roles)
    spans = records()
    if not enabled:
        assert spans == []
        assert all(s.carrier == {"env_vars": {"EXISTING": "1"}} for s in controller.serve_dict.values())
        return

    assert not {"controller.init", "controller.init.services", "controller.init.service_phase"} & {
        span["name"] for span in spans
    }
    root = next(span for span in spans if span["name"] == "job.run")
    children = [span for span in spans if span["name"] in {
        "controller.init.health", "controller.init.checkpoint_service", "service.deploy",
    }]
    assert len(children) == 4
    assert all(span["parent_id"] == root["context"]["span_id"] for span in children)
    deployments = [span for span in children if span["name"] == "service.deploy"]
    deployment_ids = {span["context"]["span_id"] for span in deployments}
    initializers = [span for span in spans if span["name"] == "service.init"]
    assert len(initializers) == 2
    assert {span["parent_id"] for span in initializers} == deployment_ids
    for service in controller.serve_dict.values():
        traceparent = service.carrier["env_vars"]["RELAX_OTEL_TRACEPARENT"]
        assert "0x" + traceparent.split("-")[2] in deployment_ids
        assert service.carrier["env_vars"]["EXISTING"] == "1"
    if fully_async:
        assert max(span["start_time"] for span in deployments) < min(
            span["end_time"] for span in deployments
        )
