# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run-scoped observation shutdown on a shared directory, before process teardown.

This channel ends telemetry, not model work. Every provider acknowledges its own
flush; missing acknowledgements remain failures rather than fabricated spans.
"""

import hashlib
import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _directory() -> Path | None:
    if not Envs.RELAX_OTEL_ENABLED:
        return None
    root = Envs.RELAX_OTEL_SHUTDOWN_DIR
    run_id = Envs.RELAX_OTEL_RUN_ID
    if not root or not run_id:
        return None
    return Path(root) / hashlib.sha256(run_id.encode()).hexdigest()[:24]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(path)


class ShutdownListener:
    def __init__(self, processor: Any, directory: Path):
        self.processor = processor
        self.directory = directory
        self.run_id = os.environ["RELAX_OTEL_RUN_ID"]
        try:
            start = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        except OSError:
            start = str(time.time_ns())
        self.identity = {
            "run_id": self.run_id, "host": socket.gethostname(), "pid": os.getpid(),
            "process_start": start, "provider_id": uuid.uuid4().hex,
        }
        self.path = directory / "participants" / (self.identity["provider_id"] + ".json")
        self.done = threading.Event()
        self.lock = threading.Lock()
        _write(self.path, {**self.identity, "state": "active"})
        threading.Thread(target=self._watch, name="otel-shutdown", daemon=True).start()

    def finish(self, reason: str, nonce: str | None = None, completed: bool = False) -> None:
        with self.lock:
            if self.done.is_set():
                return
            result = {**self.identity, "state": "finalized", "reason": reason, "nonce": nonce}
            try:
                result["observed_at_ns"] = time.time_ns()
                result["ended_spans"] = self.processor.close(reason, completed=completed)
                result["flush_completed"] = bool(self.processor.provider.force_flush(timeout_millis=5000))
            except Exception as exc:
                result.update(flush_completed=False, error=f"{type(exc).__name__}: {exc}")
            _write(self.path, result)
            self.done.set()

    def _watch(self) -> None:
        while not self.done.wait(0.2):
            try:
                request = json.loads((self.directory / "request.json").read_text())
                if request["run_id"] != self.run_id:
                    continue
                self.finish(request["reason"], request["nonce"], request.get("completed", False))
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.warning(f"OTel shutdown listener failed ({type(exc).__name__}): {exc}")
                return


def start_listener(processor: Any) -> ShutdownListener | None:
    directory = _directory()
    if directory is None:
        return None
    return ShutdownListener(processor, directory)


def finalize_run_traces(reason: str, *, completed: bool = False, timeout: float = 10.0) -> dict:
    """Request all registered providers to end observation, then await real flushes."""
    directory = _directory()
    if directory is None:
        return {"enabled": False}
    request = {
        "run_id": os.environ["RELAX_OTEL_RUN_ID"], "nonce": uuid.uuid4().hex,
        "reason": reason, "completed": completed, "requested_at_ns": time.time_ns(),
    }
    _write(directory / "request.json", request)
    deadline = time.monotonic() + timeout
    previous = None
    stable = 0
    while True:
        participants = [json.loads(path.read_text()) for path in sorted((directory / "participants").glob("*.json"))]
        pending = [p for p in participants if p.get("state") != "finalized"]
        failed = [p for p in participants if p.get("state") == "finalized" and not p.get("flush_completed")]
        # Providers that exited normally before this request have already flushed.
        mismatched = [p for p in participants if p.get("nonce") not in (None, request["nonce"])]
        ids = {p["provider_id"] for p in participants}
        stable = stable + 1 if ids == previous else 0
        previous = ids
        if (participants and not pending and stable >= 2) or time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    result = {
        "enabled": True, **request, "participants": participants,
        "pending": pending, "failed": failed, "mismatched": mismatched,
        "verified": bool(participants) and not (pending or failed or mismatched),
        "verification_scope": "Provider flush completion; Collector completeness requires a separate audit",
        "finished_at_ns": time.time_ns(),
    }
    _write(directory / "result.json", result)
    return result
