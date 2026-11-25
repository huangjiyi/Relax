#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare a pinned QS Hugging Face model before starting SFT."""

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def validate_checkpoint(directory: Path) -> None:
    config = json.loads((directory / "config.json").read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Invalid Hugging Face config in {directory}.")

    def populated(name: str) -> bool:
        path = directory / name
        return path.is_file() and path.stat().st_size > 0

    if not (populated("tokenizer.json") or (populated("vocab.json") and populated("merges.txt"))):
        raise ValueError(f"Qwen3 tokenizer files are missing from {directory}.")
    for filename in ("model.safetensors", "pytorch_model.bin"):
        if populated(filename):
            return
        index = directory / f"{filename}.index.json"
        if index.is_file():
            weights = json.loads(index.read_text())["weight_map"]
            if (
                isinstance(weights, dict)
                and weights
                and all(
                    isinstance(name, str) and Path(name).name == name and populated(name) for name in weights.values()
                )
            ):
                return
    raise ValueError(f"Complete Hugging Face weights are missing from {directory}.")


def prepare_model(args: argparse.Namespace) -> Path:
    if not args.output_dir.is_absolute():
        raise ValueError("--output-dir must be an absolute path.")
    target = args.output_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = {"name": args.model_name, "version": args.version, "subversion": args.subversion}
    marker_name = ".qs_meta.json"

    with (target.parent / f".{target.name}.download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"Model output path is not a directory: {target}.")
            if any(target.iterdir()):
                marker = target / marker_name
                if not marker.is_file() or json.loads(marker.read_text()) != identity:
                    raise ValueError(
                        f"{target} contains a different or unmanaged model. Choose a new output directory."
                    )
                validate_checkpoint(target)
                return target

        user = os.getenv(args.user_env)
        token = os.getenv(args.token_env)
        if not user or not token:
            raise ValueError(f"Set the model download credentials in {args.user_env} and {args.token_env}.")

        from model_tools import download_model

        with TemporaryDirectory(prefix=f".{target.name}.download-", dir=target.parent) as staged_dir:
            staged = Path(staged_dir)
            result = download_model(
                save_path=str(staged),
                model_name=args.model_name,
                version=args.version,
                subversion=args.subversion,
                user=user,
                token=token,
                env=args.environment,
                zone=args.zone,
                region=args.region,
            )
            if result is False:
                raise RuntimeError(f"QS download failed for {args.model_name} {args.version}/{args.subversion}.")
            validate_checkpoint(staged)
            (staged / marker_name).write_text(json.dumps(identity) + "\n")
            staged.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--subversion", type=int, required=True)
    parser.add_argument("--user-env", default="MODEL_STORE_USER")
    parser.add_argument("--token-env", default="MODEL_STORE_TOKEN")
    parser.add_argument("--environment", default="prod")
    parser.add_argument("--zone", default="cn")
    parser.add_argument("--region", default="")
    args = parser.parse_args()
    print(prepare_model(args), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        sys.exit(str(error))
