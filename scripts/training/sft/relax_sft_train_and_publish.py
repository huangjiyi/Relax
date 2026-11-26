#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run native Relax SFT on a QS debug node and publish its final HF
checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def prepare_spec(path: Path, run_dir: Path | None, prepare_dir: Path | None = None) -> tuple[dict, Path, str]:
    spec = json.loads(sys.stdin.read() if path == Path("-") else path.read_text())
    argv = shlex.split(spec["entrypoint"])
    if argv[1:3] != ["-m", "relax.entrypoints.train"]:
        raise ValueError("The job spec must launch python -m relax.entrypoints.train.")
    options = {}
    for flag in ("--save", "--save-hf"):
        if argv.count(flag) != 1:
            raise ValueError(f"The job spec must contain exactly one {flag}.")
        options[flag] = argv.index(flag) + 1
    if run_dir is not None:
        argv[options["--save"]] = str(run_dir / "output")
        argv[options["--save-hf"]] = str(run_dir / "output_hf" / "iter_{rollout_id}")
        if prepare_dir is not None:
            for flag, value in (
                ("--sft-local-text-dir", str(prepare_dir / "indexed")),
                ("--sft-local-prepared-batches", str(prepare_dir / "batches")),
            ):
                if flag in argv:
                    argv[argv.index(flag) + 1] = value
                else:
                    argv.extend((flag, value))
        spec["entrypoint"] = shlex.join(argv)
        digest = hashlib.sha256(str(run_dir).encode()).hexdigest()[:16]
        spec["submission_id"] = f"relax-sft-{digest}"
    checkpoint = Path(argv[options["--save"]])
    hf_template = argv[options["--save-hf"]]
    if not checkpoint.is_absolute() or not Path(hf_template).is_absolute():
        raise ValueError("Both checkpoint paths must be absolute and visible from this node.")
    if not spec.get("submission_id"):
        raise ValueError("The job spec must include a submission_id.")
    return spec, checkpoint, hf_template


def ray_request(dashboard: str, path: str = "", body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    request = Request(
        f"{dashboard.rstrip('/')}/api/jobs/{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def lookup_job(dashboard: str, job_id: str) -> dict | None:
    try:
        return ray_request(dashboard, job_id)
    except HTTPError as error:
        if error.code != 404:
            raise
        return None


def require_idle_cluster(dashboard: str) -> None:
    if any(job.get("status") in ("PENDING", "RUNNING") for job in ray_request(dashboard)):
        raise RuntimeError("The Ray cluster already has an active job.")


def reset_run_directory(run_dir: Path, checkpoint: Path, keep_names: set[str]) -> None:
    # Keep the lock inode while removing the previous run's data.
    contents = [path for path in run_dir.iterdir() if path.name not in keep_names]
    if not contents:
        return
    records = [
        path for path in contents if path.match("ray-submit*.json") and not path.is_symlink() and path.is_file()
    ]
    for record in records:
        entrypoint = json.loads(record.read_text()).get("entrypoint", "")
        argv = shlex.split(entrypoint)
        if "--save" not in argv:
            continue
        save_index = argv.index("--save") + 1
        if argv[save_index : save_index + 1] == [str(checkpoint)]:
            break
    else:
        raise ValueError(f"Refusing to overwrite {run_dir} without an existing job spec for this directory.")

    # Delete the records last so a failed cleanup can be retried.
    for path in sorted(contents, key=lambda path: path in records):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def wait_for_training(dashboard: str, spec: dict, submit: bool) -> None:
    job_id = quote(spec["submission_id"], safe="")
    job = lookup_job(dashboard, job_id)
    if job is None:
        if not submit:
            raise RuntimeError(f"Ray job {spec['submission_id']} was not found.")
        require_idle_cluster(dashboard)
        ray_request(dashboard, body=spec)
        job = ray_request(dashboard, job_id)
    print(f"Waiting for Ray job {spec['submission_id']}", flush=True)
    while job["status"] in ("PENDING", "RUNNING"):
        time.sleep(15)
        job = ray_request(dashboard, job_id)
    if job["status"] != "SUCCEEDED":
        raise RuntimeError(f"Ray job ended with status {job['status']}.")


def start_batch_preparation(spec: dict, directory: Path, record_dir: Path, workers: int) -> None:
    argv = shlex.split(spec["entrypoint"])

    def option(flag: str, default: str | None = None) -> str:
        if flag in argv:
            return argv[argv.index(flag) + 1]
        if default is not None:
            return default
        raise ValueError(f"{flag} is required for local batch preparation.")

    source = Path(option("--prompt-data"))
    model = Path(option("--hf-checkpoint"))
    if not source.is_absolute() or not model.is_absolute() or not model.is_dir():
        raise ValueError("Local preparation needs absolute paths to the JSONL and HF model directory.")
    if option("--max-tokens-per-gpu") != "16384":
        raise ValueError("Prepared text batches currently require --max-tokens-per-gpu 16384.")
    actor_gpus = json.loads(option("--resource"))["actor"][1]
    command = [
        sys.executable,
        str(Path(__file__).with_name("prepare_local_text_batches.py")),
        "--source",
        str(source),
        "--model",
        str(model),
        "--local",
        str(directory / "indexed"),
        "--output",
        str(directory / "batches"),
        "--seed",
        option("--seed", "1234"),
        "--gbs",
        option("--global-batch-size"),
        "--dp",
        str(actor_gpus),
        "--workers",
        str(workers),
        "--oversize-strategy",
        option("--sft-oversize-strategy", "skip"),
        "--invalid-multimodal-strategy",
        option("--sft-invalid-multimodal-strategy", "error"),
    ]
    directory.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **spec.get("runtime_env", {}).get("env_vars", {})}
    repo = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo), env.get("PYTHONPATH", "")) if p)
    log_path = record_dir / "prepare-local-batches.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"Preparing local batches | {log_path}", flush=True)
    batches = directory / "batches"
    manifest = batches / "batch-manifest.json"
    while True:
        if (status := process.poll()) not in (None, 0):
            raise RuntimeError(f"Local batch preparation exited with status {status}. See {log_path}.")
        if manifest.is_file():
            steps = json.loads(manifest.read_text())["steps"]
            if steps > 0 and all((batches / f"{step:06d}.npz").is_file() for step in range(min(steps, 16))):
                return
        if status == 0:
            raise RuntimeError(
                f"Local batch preparation exited without publishing the startup batches. See {log_path}."
            )
        time.sleep(2)


def final_hf_directory(checkpoint: Path, template: str) -> Path:
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    rollout_id = int(tracker.read_text().strip())
    artifact = Path(template.format(rollout_id=rollout_id))
    if "{rollout_id" not in template:
        artifact /= f"iter_{rollout_id:07d}"
    if not isinstance(json.loads((artifact / "config.json").read_text()), dict):
        raise ValueError(f"Invalid HF config in {artifact}.")
    for filename in ("model.safetensors", "pytorch_model.bin"):
        weight = artifact / filename
        if weight.is_file() and weight.stat().st_size > 0:
            return artifact
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = artifact / filename
        if not index.is_file():
            continue
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if shards and all(
            Path(name).name == name and (artifact / name).is_file() and (artifact / name).stat().st_size > 0
            for name in shards
        ):
            return artifact
    raise FileNotFoundError(f"Complete HF weights are missing from {artifact}.")


def publish(artifact: Path, model_name: str, receipt: Path, precision: str, publisher: dict[str, str]) -> None:
    from model_tools import push_model

    identity = {"model_name": model_name, "artifact": str(artifact)}
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous == {**identity, "status": "published"}:
            print(f"Already published {artifact} as {model_name}.", flush=True)
            return
        raise RuntimeError(f"Check the model registry and the existing publication receipt at {receipt}.")

    write_json(receipt, {**identity, "status": "started"})
    result = push_model(
        model_path=artifact,
        model_name=model_name,
        framework="Relax",
        model_format="hf",
        precision=precision.lower(),
        gpu_type="NoLimit",
        **publisher,
    )
    if result is False:  # This SDK returns None on success.
        raise RuntimeError(f"QS upload failed. Check the model registry and {receipt}.")
    write_json(receipt, {**identity, "status": "published"})
    print(f"Published {artifact} as {model_name}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-spec", type=Path, required=True, help="Native Relax Ray job JSON (file or '-').")
    parser.add_argument("--model-name", help="Destination model name in the QS registry.")
    parser.add_argument("--run-dir", type=Path, help="Submit a new run here. Otherwise wait for the existing job.")
    parser.add_argument("--attempt", help="Name a new Ray attempt that resumes from the same run directory.")
    parser.add_argument(
        "--overwrite-run-dir", action="store_true", help="Delete the previous run and start a new one."
    )
    parser.add_argument("--launch-log-name", help="Name of the current launcher log to keep during overwrite.")
    parser.add_argument("--prepare-dir", type=Path, help="Build local text caches for a new run from its JSONL input.")
    parser.add_argument("--prep-workers", type=int, default=14)
    parser.add_argument("--ray-dashboard", default="http://127.0.0.1:8265")
    parser.add_argument("--precision", default="BF16")
    parser.add_argument(
        "--publisher-user-env", default="MODEL_STORE_USER", help="Environment variable for the uploader login."
    )
    parser.add_argument(
        "--publisher-token-env", default="MODEL_STORE_TOKEN", help="Environment variable for the uploader token."
    )
    parser.add_argument("--publisher-environment", default="prod")
    parser.add_argument("--publisher-zone", default="cn")
    parser.add_argument("--publisher-region", default="")
    parser.add_argument("--dry-run", action="store_true", help="Show the selected job and artifact paths.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir else None
    prepare_dir = args.prepare_dir.resolve() if args.prepare_dir else None
    if args.overwrite_run_dir and (run_dir is None or args.attempt or args.run_dir.is_symlink()):
        parser.error("--overwrite-run-dir requires a real --run-dir and cannot be combined with --attempt.")
    if args.launch_log_name and not Path(args.launch_log_name).match("launch-*.log"):
        parser.error("--launch-log-name must match launch-*.log.")
    if prepare_dir is not None and (run_dir is None or args.prep_workers < 1):
        parser.error("--prepare-dir requires --run-dir and a positive --prep-workers value.")
    spec, checkpoint, template = prepare_spec(args.job_spec, run_dir, prepare_dir)
    recorded_name = "ray-submit.json"
    if args.attempt:
        argv = shlex.split(spec["entrypoint"])
        if run_dir is None or "--load" not in argv or Path(argv[argv.index("--load") + 1]).resolve() != checkpoint:
            parser.error("--attempt requires --run-dir and --load pointing to its checkpoint directory.")
        attempt_key = hashlib.sha256(args.attempt.encode()).hexdigest()[:12]
        spec["submission_id"] += f"-{attempt_key}"
        recorded_name = f"ray-submit-{attempt_key}.json"
    elif args.overwrite_run_dir:
        argv = shlex.split(spec["entrypoint"])
        for flag in ("--hf-checkpoint", "--ref-load", "--prompt-data", "--load"):
            if flag not in argv:
                continue
            value = argv[argv.index(flag) + 1]
            if value == str(checkpoint) and flag == "--load":
                continue
            if "://" not in value and Path(value).resolve().is_relative_to(run_dir):
                parser.error(f"--overwrite-run-dir would delete the input to {flag}.")
        spec["submission_id"] += f"-{uuid4().hex[:12]}"
    print(f"Ray job {spec['submission_id']} | checkpoint {checkpoint} | HF {template}")
    if args.model_name:
        print(f"QS model {args.model_name} | precision {args.precision}")
    if args.dry_run:
        print(spec["entrypoint"])
        return
    publisher = None
    if args.model_name:
        user = os.getenv(args.publisher_user_env)
        token = os.getenv(args.publisher_token_env)
        if not user or not token or importlib.util.find_spec("model_tools") is None:
            raise RuntimeError(
                "Provide uploader credentials via the selected environment variables and install model_tools."
            )
        publisher = {
            "user": user,
            "token": token,
            "env": args.publisher_environment,
            "zone": args.publisher_zone,
            "region": args.publisher_region,
        }

    record_dir = run_dir or checkpoint.parent
    record_dir.mkdir(parents=True, exist_ok=True)
    lock_path = record_dir / "train-and-publish.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if run_dir is not None:
            if args.overwrite_run_dir:
                require_idle_cluster(args.ray_dashboard)
                keep_names = {lock_path.name}
                if args.launch_log_name:
                    keep_names.add(args.launch_log_name)
                reset_run_directory(run_dir, checkpoint, keep_names)
            recorded_spec = run_dir / recorded_name
            if recorded_spec.exists():
                if json.loads(recorded_spec.read_text()) != spec:
                    raise ValueError(f"A different job spec already occupies {run_dir}.")
            else:
                if not args.attempt and (checkpoint.exists() or (run_dir / "output_hf").exists()):
                    raise ValueError(f"Checkpoint artifacts already occupy {run_dir}.")
                write_json(recorded_spec, spec)
        if prepare_dir is not None and lookup_job(args.ray_dashboard, quote(spec["submission_id"], safe="")) is None:
            require_idle_cluster(args.ray_dashboard)
            start_batch_preparation(spec, prepare_dir, record_dir, args.prep_workers)
        wait_for_training(args.ray_dashboard, spec, submit=run_dir is not None)
        artifact = final_hf_directory(checkpoint, template)
        print(f"Final HF export: {artifact}", flush=True)
        if publisher is not None:
            publish(artifact, args.model_name, record_dir / "qs-model-publish.json", args.precision, publisher)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        sys.exit(str(error))
