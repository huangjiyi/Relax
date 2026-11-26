# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import fcntl
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[3] / "scripts" / "training" / "sft" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_script("relax_sft_train_and_publish.py")


def test_new_job_preserves_training_arguments(tmp_path):
    original = tmp_path / "original"
    spec_path = tmp_path / "job.json"
    spec_path.write_text(
        json.dumps(
            {
                "submission_id": "current",
                "entrypoint": shlex.join(
                    [
                        "python3",
                        "-m",
                        "relax.entrypoints.train",
                        "--save",
                        str(original / "output"),
                        "--save-hf",
                        str(original / "hf" / "iter_{rollout_id}"),
                        "--manual-gc",
                        "--global-batch-size",
                        "8192",
                    ]
                ),
                "runtime_env": {"env_vars": {"PYTHONPATH": str(original)}},
            }
        )
    )
    run_dir = tmp_path / "new run"
    cache_dir = tmp_path / "cache"
    spec, checkpoint, template = launcher.prepare_spec(spec_path, run_dir, cache_dir)
    argv = shlex.split(spec["entrypoint"])
    assert spec["submission_id"] != "current"
    assert spec["runtime_env"] == json.loads(spec_path.read_text())["runtime_env"]
    assert argv[argv.index("--global-batch-size") + 1] == "8192"
    assert argv[argv.index("--sft-local-prepared-batches") + 1] == str(cache_dir / "batches")
    assert "--manual-gc" in argv
    assert checkpoint == run_dir / "output"
    assert template == str(run_dir / "output_hf" / "iter_{rollout_id}")
    assert not run_dir.exists()


def test_final_export_requires_complete_shards(tmp_path):
    artifact = tmp_path / "hf" / "iter_7"
    artifact.mkdir(parents=True)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7\n")
    (artifact / "config.json").write_text("{}")
    (artifact / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"parameter": "model-00001-of-00001.safetensors"}})
    )
    template = str(tmp_path / "hf" / "iter_{rollout_id}")
    with pytest.raises(FileNotFoundError, match="Complete HF weights"):
        launcher.final_hf_directory(tmp_path, template)
    (artifact / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    assert launcher.final_hf_directory(tmp_path, template) == artifact


@pytest.mark.parametrize("status", ["FAILED", "STOPPED"])
def test_training_must_succeed(monkeypatch, status):
    monkeypatch.setattr(launcher, "ray_request", lambda *_: {"status": status})
    with pytest.raises(RuntimeError, match=status):
        launcher.wait_for_training("http://localhost", {"submission_id": "test"}, submit=False)


@pytest.mark.parametrize("previous_run", ["idle", "locked", "active Ray job", "unrecognized"])
def test_overwrite_run_resets_only_an_idle_training_directory(tmp_path, monkeypatch, previous_run):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "output"
    checkpoint.mkdir(parents=True)
    old_weights = checkpoint / "old-weights"
    old_weights.write_bytes(b"old checkpoint")
    launch_log = run_dir / "launch-current.log"
    launch_log.write_text("current launch")
    sibling = tmp_path / "keep"
    sibling.write_text("outside the training directory")
    (run_dir / "link").symlink_to(sibling)
    job_spec = tmp_path / "next.json"
    entrypoint = [
        "python3",
        "-m",
        "relax.entrypoints.train",
        "--save",
        str(checkpoint),
        "--save-hf",
        str(run_dir / "output_hf"),
    ]
    job_spec.write_text(json.dumps({"entrypoint": shlex.join(entrypoint)}))
    prev_spec, _, _ = launcher.prepare_spec(job_spec, run_dir)
    recorded = run_dir / "ray-submit.json"
    if previous_run != "unrecognized":
        recorded.write_text(json.dumps(prev_spec))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch",
            "--job-spec",
            str(job_spec),
            "--run-dir",
            str(run_dir),
            "--overwrite-run-dir",
            "--launch-log-name",
            launch_log.name,
        ],
    )
    ray_jobs = [{"status": "RUNNING"}] if previous_run == "active Ray job" else []
    monkeypatch.setattr(launcher, "ray_request", lambda *_: ray_jobs)

    class RaySubmissionReached(Exception):
        pass

    def wait_for_training(dashboard, spec, submit):
        assert submit
        assert spec["submission_id"] != prev_spec["submission_id"]
        assert json.loads(recorded.read_text()) == spec
        raise RaySubmissionReached

    monkeypatch.setattr(launcher, "wait_for_training", wait_for_training)
    lock_path = run_dir / "train-and-publish.lock"
    with lock_path.open("a") as lock:
        if previous_run == "locked":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError):
                launcher.main()
        elif previous_run == "active Ray job":
            with pytest.raises(RuntimeError, match="active job"):
                launcher.main()
        elif previous_run == "unrecognized":
            with pytest.raises(ValueError, match="without an existing job spec"):
                launcher.main()
        else:
            with pytest.raises(RaySubmissionReached):
                launcher.main()
            assert lock_path.stat().st_ino == os.fstat(lock.fileno()).st_ino

    assert old_weights.exists() == (previous_run != "idle")
    assert launch_log.read_text() == "current launch"
    assert sibling.read_text() == "outside the training directory"


@pytest.mark.parametrize("success", [True, False])
def test_publishing_writes_receipt_and_avoids_duplicate_upload(tmp_path, monkeypatch, success):
    calls = []

    def push_model(**kwargs):
        calls.append(kwargs["model_name"])
        return None if success else False

    monkeypatch.setitem(sys.modules, "model_tools", SimpleNamespace(push_model=push_model))
    receipt = tmp_path / "publish.json"
    publisher = {"user": "test-user", "token": "test-value"}
    if success:
        launcher.publish(tmp_path, "test-model", receipt, "BF16", publisher)
        launcher.publish(tmp_path, "test-model", receipt, "BF16", publisher)
        assert json.loads(receipt.read_text())["status"] == "published"
    else:
        with pytest.raises(RuntimeError, match="upload failed"):
            launcher.publish(tmp_path, "test-model", receipt, "BF16", publisher)
        with pytest.raises(RuntimeError, match="existing publication receipt"):
            launcher.publish(tmp_path, "test-model", receipt, "BF16", publisher)
        assert json.loads(receipt.read_text())["status"] == "started"
    assert calls == ["test-model"]


def test_index_preserves_nonempty_rows_and_seed(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    preparer = load_script("prepare_local_text_batches.py")
    source = tmp_path / "data.jsonl"
    source.write_bytes(b'{"id":1}\n\n{"id":2}\n  \n{"id":3}')
    local = tmp_path / "indexed"
    preparer.build_index(source, local, 1234)
    manifest = json.loads((local / "manifest.json").read_text())
    assert manifest["rows"] == 3
    assert manifest["seed"] == 1234
    expected_offsets = [0, 10, 22]
    assert np.fromfile(local / "source-offsets.u64", dtype="<u8").tolist() == expected_offsets
    order = np.fromfile(local / "epoch0-seed1234.u32", dtype="<u4")
    assert sorted(order.tolist()) == list(range(3))
    preparer.build_index(source, local, 1234)
