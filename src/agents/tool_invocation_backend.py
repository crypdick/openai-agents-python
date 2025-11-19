from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


from .logger import logger
from .tool import FunctionTool
from .tool_context import ToolContext

try:  # pragma: no cover - ray may not be installed in some environments
    import ray  # type: ignore[unused-ignore]
except Exception:  # pragma: no cover - gracefully handle missing ray
    ray = None  # type: ignore[assignment]


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

    async def run(self) -> Any:
        return await self.func_tool.on_invoke_tool(self.tool_context, self.tool_arguments)


if ray:

    @ray.remote  # type: ignore[misc]
    def _ray_execute_function_tool(payload: _RayToolCallPayload) -> Any:
        """Execute an async function tool in a Ray worker."""
        return asyncio.run(payload.run())


class RayToolInvocationBackend(ToolInvocationBackend):
    """Ray-based backend that executes tools inside Ray tasks."""

    def __init__(self, *, auto_init: bool = True):
        self._auto_init = auto_init
        self._fallback_backend = AsyncToolInvocationBackend()

    async def invoke(
        self,
        func_tool: FunctionTool,
        tool_context: ToolContext[Any],
        tool_arguments: str,
    ) -> Any:
        if not ray or "_ray_execute_function_tool" not in globals():
            logger.debug("Ray is unavailable; falling back to inline tool execution.")
            return await self._fallback_backend.invoke(func_tool, tool_context, tool_arguments)

        self._ensure_ray_initialized()

        payload = _RayToolCallPayload(
            func_tool=func_tool,
            tool_context=tool_context,
            tool_arguments=tool_arguments,
        )

        try:
            object_ref = _ray_execute_function_tool.remote(payload)  # type: ignore[name-defined]
        except TypeError as exc:
            # Ray throws TypeError when serialization fails during task submission
            logger.warning(
                "Failed to serialize tool payload for Ray, running inline instead: %s", exc
            )
            return await self._fallback_backend.invoke(func_tool, tool_context, tool_arguments)
        except Exception as exc:
            # Catch-all for other Ray submission errors
            logger.warning(
                "Failed to submit Ray task, running inline instead: %s", exc
            )
            return await self._fallback_backend.invoke(func_tool, tool_context, tool_arguments)

        return await self._ray_get_async(object_ref)

    def _ensure_ray_initialized(self) -> None:
        if self._auto_init and not ray.is_initialized():  # type: ignore[union-attr]
            logger.debug("Initializing Ray for tool invocation backend.")
            ray.init(ignore_reinit_error=True)  # type: ignore[union-attr]

    async def _ray_get_async(self, object_ref: "ray.ObjectRef[Any]") -> Any:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: ray.get(object_ref))
        except Exception:
            logger.exception("Ray task execution failed; reraising.")
            raise
