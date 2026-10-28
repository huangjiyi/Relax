# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Capture code provenance without SDK imports, stack formatting, or RNG use."""

import inspect
import sys


_ADAPTERS = frozenset({
    __name__, "relax.utils.tracing", "relax.utils.tracing_megatron",
    "contextlib", "sglang.srt.observability.trace",
})


def caller_location() -> dict:
    frame = sys._getframe(1)
    try:
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            generic_native = module == "sglang.srt.observability.req_time_stats" and frame.f_code.co_name == "trace_slice"
            if module not in _ADAPTERS and not generic_native:
                return {
                    "code.file.path": frame.f_code.co_filename,
                    "code.line.number": frame.f_lineno,
                    "code.function.name": f"{module}.{frame.f_code.co_qualname}",
                    "relax.code.location": "callsite",
                }
            frame = frame.f_back
        raise RuntimeError("No application frame for tracing code location")
    finally:
        del frame


def function_location(function) -> dict:
    # functools.wraps preserves the business name but leaves __code__ on the
    # wrapper (for example Relax's @timer). Resolve both from the same target.
    function = inspect.unwrap(function)
    return {
        "code.file.path": function.__code__.co_filename,
        "code.line.number": function.__code__.co_firstlineno,
        "code.function.name": f"{function.__module__}.{function.__qualname__}",
        "relax.code.location": "function_definition",
    }
