# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build a new private SGLang overlay; never patch the installed package in place."""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(native, output):
    package = Path(__file__).resolve().parent
    native, output = native.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(native) or native.is_relative_to(output):
        raise ValueError("Choose a new overlay directory outside the installed package")
    trace = json.loads((package / "manifest.json").read_text())
    lifecycle = json.loads((package / "lifecycle-manifest.json").read_text())
    records = [trace, *lifecycle["files"]]
    for item in records:
        relative = Path(item["target"]).relative_to("python/sglang")
        if digest(native / relative) != item["before_sha256"]:
            raise ValueError(f"Source mismatch: {relative}; use the matching native version")
    patches = [
        ("sglang-v0.5.17-ettr.patch", trace["patch_sha256"]),
        (lifecycle["patch"], lifecycle["patch_sha256"]),
    ]
    for name, expected in patches:
        if digest(package / name) != expected:
            raise ValueError(f"Patch checksum mismatch: {name}")
    with tempfile.TemporaryDirectory(prefix="ettr-native-build-") as directory:
        staging = Path(directory)
        for item in records:
            target = staging / item["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(native / Path(item["target"]).relative_to("python/sglang"), target)
        for name, _ in patches:
            subprocess.run(["patch", "--batch", "-p1", "-i", str(package / name)], cwd=staging, check=True)
        replacements = {}
        for item in records:
            target = staging / item["target"]
            if digest(target) != item["after_sha256"]:
                raise ValueError(f"Patched checksum mismatch: {item['target']}")
            ast.parse(target.read_text())
            replacements[Path(item["target"]).relative_to("python/sglang")] = target

        def mirror(relative):
            destination = output / "sglang" / relative
            destination.mkdir(parents=True, exist_ok=True)
            for source in (native / relative).iterdir():
                if source.name == "__pycache__":
                    continue
                child = relative / source.name
                if child in replacements:
                    shutil.copyfile(replacements[child], destination / source.name)
                elif any(path.is_relative_to(child) for path in replacements):
                    mirror(child)
                else:
                    (destination / source.name).symlink_to(source, target_is_directory=source.is_dir())

        output.mkdir(parents=True, exist_ok=False)
        mirror(Path())
    result = {
        "native_root": str(native), "overlay": str(output), "files": records,
        "required_environment": {"RELAX_OTEL_SGLANG_LIFECYCLE": "1"},
        "validation": "Source hashes, patch round-trip and Python syntax only; no GPU validation",
    }
    (output / "overlay-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.sglang_root, args.output), indent=2))
