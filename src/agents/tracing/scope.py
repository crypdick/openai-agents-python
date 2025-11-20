# Holds the current active span
import contextvars
from typing import TYPE_CHECKING, Any

from ..logger import logger

if TYPE_CHECKING:
    from .spans import Span
    from .traces import Trace

_current_span: contextvars.ContextVar["Span[Any] | None"] = contextvars.ContextVar(
    "current_span", default=None
)

_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "current_trace", default=None
)

_remote_span_updates: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "remote_span_updates", default=None
)


class Scope:
    """
    Manages the current span and trace in the context.
    """

    @classmethod
    def get_current_span(cls) -> "Span[Any] | None":
        return _current_span.get()

    @classmethod
    def set_current_span(cls, span: "Span[Any] | None") -> "contextvars.Token[Span[Any] | None]":
        return _current_span.set(span)

    @classmethod
    def reset_current_span(cls, token: "contextvars.Token[Span[Any] | None]") -> None:
        _current_span.reset(token)

    @classmethod
    def get_current_trace(cls) -> "Trace | None":
        return _current_trace.get()

    @classmethod
    def set_current_trace(cls, trace: "Trace | None") -> "contextvars.Token[Trace | None]":
        logger.debug(f"Setting current trace: {trace.trace_id if trace else None}")
        return _current_trace.set(trace)

    @classmethod
    def reset_current_trace(cls, token: "contextvars.Token[Trace | None]") -> None:
        logger.debug("Resetting current trace")
        _current_trace.reset(token)

    @classmethod
    def get_remote_span_updates(cls) -> dict[str, Any] | None:
        return _remote_span_updates.get()

    @classmethod
    def set_remote_span_updates(
        cls, updates: dict[str, Any] | None
    ) -> "contextvars.Token[dict[str, Any] | None]":
        return _remote_span_updates.set(updates)

    @classmethod
    def add_remote_span_update(cls, key: str, value: Any) -> None:
        updates = cls.get_remote_span_updates()
        if updates is None:
            updates = {}
            cls.set_remote_span_updates(updates)
        updates[key] = value
