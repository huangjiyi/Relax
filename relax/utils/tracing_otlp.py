# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Isolate SDK retry jitter from training RNG without changing SDK globals."""

import functools
import secrets
import types
from typing import Any


def isolate_exporter_rng(exporter: Any) -> Any:
    """Keep the installed SDK's encoding, timeout, retry and shutdown logic.

    SDK 1.44 HTTP export draws random.uniform even on successful requests;
    gRPC does so on retries. Bind those methods with a private globals mapping
    whose random alias uses SystemRandom. This leaves the SDK module, other
    exporter instances and Python's training RNG untouched. Validated against
    SDK 1.44; transport success and retry paths have regression tests.
    """
    for name in ("export", "_export"):
        bound = getattr(exporter, name, None)
        function = getattr(bound, "__func__", None)
        if function is None or "random" not in function.__code__.co_names:
            continue
        isolated = types.FunctionType(
            function.__code__,
            {**function.__globals__, "random": secrets.SystemRandom()},
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        isolated.__kwdefaults__ = function.__kwdefaults__
        functools.update_wrapper(isolated, function)
        setattr(exporter, name, types.MethodType(isolated, exporter))
    return exporter
