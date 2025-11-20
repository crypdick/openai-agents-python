from __future__ import annotations

import asyncio
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from agents.setup_ray import use_ray

from .logger import logger
from .tool import FunctionTool
from .tool_context import ToolContext

if use_ray():
    import ray

    from agents.setup_ray import ensure_ray_initialized
    from agents.tracing.ray_exporter import setup_distributed_tracing

    _ray_aggregator = ensure_ray_initialized()
else:
    ray = None  # type: ignore[assignment]


@dataclass
class RayToolError:
    """
    Formats errors from Ray tool executions in the format the library expects.
    """

    error_type: str
    error_message: str
    tool_name: str
    original_traceback: str | None = None

    def to_exception(self) -> Exception:
        """Convert back to an exception for re-raising in the main process."""
        # Try to recreate the original exception type if it's a builtin
        import builtins

        exc_class = getattr(builtins, self.error_type, None)
        if exc_class and issubclass(exc_class, Exception):
            return exc_class(self.error_message)  # type: ignore[no-any-return]

        # Try to import from agents.exceptions
        try:
            from . import exceptions

            exc_class = getattr(exceptions, self.error_type, None)
            if exc_class and issubclass(exc_class, Exception):
                return exc_class(self.error_message)  # type: ignore[no-any-return]
        except (ImportError, AttributeError):
            pass

        # Fall back to a generic Exception with type info
        return Exception(f"{self.error_type}: {self.error_message}")


class ToolInvocationBackend(ABC):
    """Interface for executing tool calls."""

    @abstractmethod
    async def invoke(
        self,
        func_tool: FunctionTool,
        tool_context: ToolContext[Any],
        tool_arguments: str,
    ) -> Any:
        """Execute the tool and return its result."""


class AsyncToolInvocationBackend(ToolInvocationBackend):
    """Default backend that executes tool calls inline."""

    async def invoke(
        self,
        func_tool: FunctionTool,
        tool_context: ToolContext[Any],
        tool_arguments: str,
    ) -> Any:
        return await func_tool.on_invoke_tool(tool_context, tool_arguments)


@dataclass
class _RayToolCallPayload:
    func_tool: FunctionTool
    tool_context: ToolContext[Any]
    tool_arguments: str

    async def run(self) -> Any | RayToolError:
        """
        Execute the tool and return result or RayToolError on exception.
        We return errors instead of raising to avoid Ray's verbose stack traces.
        """
        try:
            return await self.func_tool.on_invoke_tool(self.tool_context, self.tool_arguments)
        except Exception as e:
            # Capture the clean error information before Ray wraps it
            return RayToolError(
                error_type=type(e).__name__,
                error_message=str(e),
                tool_name=self.func_tool.name,
                original_traceback=traceback.format_exc(),
            )


if use_ray():

    @ray.remote
    def _ray_execute_function_tool(payload: _RayToolCallPayload) -> Any:
        """Execute an async function tool in a Ray worker."""
        setup_distributed_tracing()
        return asyncio.run(payload.run())


class RayToolInvocationBackend(ToolInvocationBackend):
    """Ray-based backend that executes tools inside Ray tasks."""

    def __init__(self, *, ray_remote_args: dict[str, Any] | None = None):
        self._fallback_backend = AsyncToolInvocationBackend()
        self._ray_remote_args = ray_remote_args or {}

    async def invoke(
        self,
        func_tool: FunctionTool,
        tool_context: ToolContext[Any],
        tool_arguments: str,
    ) -> Any:
        # Check if tool explicitly bypasses Ray backend (e.g., MCP tools)
        if getattr(func_tool, "_bypass_ray_backend", False):
            logger.debug(f"Tool {func_tool.name} bypasses Ray backend; executing inline.")
            return await self._fallback_backend.invoke(func_tool, tool_context, tool_arguments)

        if not ray:
            logger.debug("Ray is unavailable; falling back to inline tool execution.")
            return await self._fallback_backend.invoke(func_tool, tool_context, tool_arguments)

        payload = _RayToolCallPayload(
            func_tool=func_tool,
            tool_context=tool_context,
            tool_arguments=tool_arguments,
        )

        try:
            object_ref = _ray_execute_function_tool.options(**self._ray_remote_args).remote(payload)
        except TypeError as exc:
            # Ray throws TypeError when serialization fails during task submission
            logger.warning(
                "Failed to serialize tool payload for Ray, falling back to default backend: %s", exc
            )
            return await self._fallback_backend.invoke(func_tool, tool_context, tool_arguments)
        except Exception as exc:
            # Catch-all for other Ray submission errors
            logger.warning("Failed to submit Ray task, falling back to default backend: %s", exc)
            return await self._fallback_backend.invoke(func_tool, tool_context, tool_arguments)

        return await self._ray_get_async(object_ref)

    async def _ray_get_async(self, object_ref: ray.ObjectRef[Any]) -> Any:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, lambda: ray.get(object_ref))

            if isinstance(result, RayToolError):
                # Convert back to exception and raise with clean message
                raise result.to_exception()

            return result
        except Exception:
            logger.exception("Ray task execution failed; reraising.")
            raise
