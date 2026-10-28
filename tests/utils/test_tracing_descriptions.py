# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Catalog semantics stay available without duplicating text in default
spans."""

import ast
import os
import random
import secrets
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, id_generator
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from relax.utils import tracing
from relax.utils.tracing_descriptions import DESCRIPTION_ATTRIBUTE, DESCRIPTIONS, describe_span, description_processor
from tests.utils.test_tracing_compact import records as _records


records = _records


def test_default_spans_omit_repeated_descriptions_and_keep_lookup_identity(records):
    before = random.getstate()
    tracing.start_run()
    with tracing.stage("rollout.generate", **{"relax.group_index": 0}):
        with tracing.stage("rollout.engine_request", **{"relax.sample_index": 0}):
            with tracing.stage("rollout.reward"):
                pass
    tracing.finish_run()
    spans = {s["name"]: s for s in records()}
    assert set(spans) == {"job.run", "rollout.generate", "rollout.engine_request", "rollout.reward"}
    assert all(DESCRIPTION_ATTRIBUTE not in s["attributes"] for s in spans.values())
    assert len({s["context"]["span_id"] for s in spans.values()}) == 4
    for name in ("rollout.engine_request", "rollout.reward"):
        text = describe_span(name, spans[name]["attributes"])
        assert "G: global prompt-group index" in text
        assert "S: global sample index, not within-group." in text
        assert "Both start at 0 and continue across steps." in text
        assert "\nG:" in text and "\nS:" in text
        assert len(text.split()) <= 60
        assert "GPU kernel time" not in text
        assert text.isascii()
    assert spans["rollout.engine_request"]["attributes"]["relax.sample_index"] == 0
    assert spans["rollout.engine_request"]["attributes"]["relax.group_index"] == 0
    assert random.getstate() == before


def test_processor_describes_native_dynamic_names_snapshot_and_unknown():
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(description_processor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("native-test")
    cases = [
        (" Req abcdef12", {}, "Req is its request ID"),
        ("Scheduler [TP 0] [PP 0] (host:abc | pid:42)", {}, "concurrent requests overlap"),
        ("Tokenizer(host:abc | pid:42)", {}, "including output waits"),
        ("decode_forward", {}, "not one token or one kernel"),
        ("worker.init", {"relax.init.kind": "metadata_snapshot"}, "not model initialization time"),
        ("worker.init", {}, "Initialize one backend worker"),
        ("extension.operation", {"code.function.name": "custom.function"}, "custom.function"),
    ]
    for name, attrs, _ in cases:
        with tracer.start_as_current_span(name, attributes=attrs):
            pass
    assert provider.force_flush()
    for span, (name, _, expected) in zip(exporter.get_finished_spans(), cases, strict=True):
        assert span.name == name
        assert expected in span.attributes[DESCRIPTION_ATTRIBUTE]
        assert span.attributes[DESCRIPTION_ATTRIBUTE].isascii()
    provider.shutdown()


def test_native_provider_omits_description_text_and_preserves_parents(monkeypatch):
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    before = random.getstate()
    exporter = InMemorySpanExporter()
    providers = []
    source = Path(__file__).resolve().parents[2] / "patches/ettr/sglang-trace.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *[
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in {"TraceCustomIdGenerator", "_ettr_enabled", "process_tracing_init"}
        ],
    ]
    scope = {
        "os": os,
        "random": random,
        "secrets": secrets,
        "time": time,
        "id_generator": id_generator,
        "Resource": Resource,
        "SERVICE_NAME": "service.name",
        "TracerProvider": TracerProvider,
        "BatchSpanProcessor": BatchSpanProcessor,
        "opentelemetry_imported": True,
        "get_int_env_var": lambda name, default: int(os.environ.get(name, default)),
        "get_otlp_span_exporter": lambda endpoint: exporter,
        "trace": SimpleNamespace(
            set_tracer_provider=providers.append,
            get_tracer=lambda name: providers[0].get_tracer(name),
        ),
    }
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), scope)
    scope["process_tracing_init"]("memory-only", "sglang")
    provider = providers[0]
    try:
        with scope["tracer"].start_as_current_span("rollout.weights.receive_apply") as parent:
            with scope["tracer"].start_as_current_span("decode_forward"):
                pass
        assert provider.force_flush()
        spans = {span.name: span for span in exporter.get_finished_spans()}
        assert set(spans) == {"rollout.weights.receive_apply", "decode_forward"}
        assert all(DESCRIPTION_ATTRIBUTE not in span.attributes for span in spans.values())
        assert spans["decode_forward"].parent == parent.context
        assert random.getstate() == before
    finally:
        provider.shutdown()


def test_production_literal_stage_names_have_specific_descriptions():
    root = Path(__file__).resolve().parents[2] / "relax"
    names = set()
    for file in root.rglob("*.py"):
        source = file.read_text()
        if "utils.tracing import" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("stage", "traced")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.add(node.args[0].value)
    assert names <= DESCRIPTIONS.keys(), names - DESCRIPTIONS.keys()
    assert all(not describe_span(name).startswith("Instrumented operation") for name in names)


def test_disabled_description_module_does_not_import_sdk():
    code = """
import sys
from relax.utils.tracing_descriptions import describe_span
from relax.utils import tracing
with tracing.stage('rollout.group'):
    pass
assert 'global prompt-group' in describe_span('rollout.group')
assert not any(n.startswith('opentelemetry') for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "RELAX_OTEL_ENABLED": "0"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
