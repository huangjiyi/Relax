# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional adapter at MCore host schedule boundaries; no CUDA synchronization."""

import contextvars
import functools
import inspect
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from relax.utils.tracing import selected, stage
from relax.utils.tracing_code import function_location


_calls: contextvars.ContextVar[dict | None] = contextvars.ContextVar("relax_megatron_calls", default=None)


def _wrap(function: Any, direction: str) -> Any:
    signature = inspect.signature(function)
    location = function_location(function)

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        calls = _calls.get()
        if calls is None or not selected(3):
            return function(*args, **kwargs)
        index = calls[direction]
        calls[direction] += 1
        values = signature.bind_partial(*args, **kwargs).arguments
        attributes = {
            "relax.backend": "megatron",
            "relax.direction": direction,
            "relax.microbatch.call_index": index,
            "relax.microbatch": values.get("current_microbatch"),
        }
        # Native backward_step has no microbatch argument. Do not infer its
        # identity from the forward order under pipeline scheduling.
        with stage(f"megatron.microbatch.{direction}", level=3, _code_location=location, **attributes):
            return function(*args, **kwargs)

    wrapped._relax_otel_schedule = True
    return wrapped


def _install() -> None:
    from megatron.core.pipeline_parallel import schedules

    for name, direction in (("forward_step", "forward"), ("backward_step", "backward")):
        original = getattr(schedules, name)
        if getattr(original, "_relax_otel_schedule", False):
            continue
        wrapped = _wrap(original, direction)
        setattr(schedules, name, wrapped)
        # Streaming schedules retain direct imports. Cover already imported
        # aliases too; future imports see the wrapped MCore functions.
        streaming = sys.modules.get("relax.backends.megatron.streaming_schedules")
        if streaming is not None and getattr(streaming, name, None) is original:
            setattr(streaming, name, wrapped)


@contextmanager
def schedule_scope(*, forward_only: bool = False, **attributes: Any) -> Iterator[None]:
    """Cover the actual schedule invocation, including gradient finalization."""
    token = None
    if selected(3):
        _install()
        token = _calls.set({"forward": 0, "backward": 0})
    try:
        name = "megatron.forward_only" if forward_only else "megatron.forward_backward"
        with stage(name, level=2, **{"relax.forward_only": forward_only, **attributes}):
            yield
    finally:
        if token is not None:
            _calls.reset(token)
