"""Tests for tool invocation backend when Ray is disabled (RAY_BACKEND != 1).

Run these tests with: pytest tests/test_ray_backend_disabled.py
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.tool import FunctionTool
from agents.tool_context import ToolContext
from agents.tool_invocation_backend import AsyncToolInvocationBackend, RayToolInvocationBackend

# Skip all tests in this file if TEST_RAY_BACKEND=1
pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_RAY_BACKEND") == "1",
    reason="Skipping disabled backend tests when TEST_RAY_BACKEND=1",
)


@pytest.mark.asyncio
async def test_ray_backend_disabled_by_default():
    """Test that Async backend is used by default when RAY_BACKEND is not set."""
    from agents.run import RunConfig

    config = RunConfig()
    assert isinstance(config.tool_invocation_backend, AsyncToolInvocationBackend)


@pytest.mark.asyncio
async def test_ray_backend_fallback_when_ray_not_available():
    """Test that RayToolInvocationBackend falls back to inline execution when ray is not available."""
    # When RAY_BACKEND is not set, ray should be None at module level,
    # so RayToolInvocationBackend should always fall back
    backend = RayToolInvocationBackend()

    async def mock_invoke(*args):
        return "fallback_result"

    backend._fallback_backend = MagicMock()
    backend._fallback_backend.invoke = AsyncMock(side_effect=mock_invoke)

    tool = MagicMock(spec=FunctionTool)
    context = MagicMock(spec=ToolContext)

    result = await backend.invoke(tool, context, "{}")
    assert result == "fallback_result"
    backend._fallback_backend.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_async_backend_direct_invocation():
    """Test that AsyncToolInvocationBackend invokes tools directly."""
    backend = AsyncToolInvocationBackend()

    # Create a mock tool that returns a specific value
    async def mock_on_invoke_tool(ctx, args):
        return "direct_result"

    tool = MagicMock(spec=FunctionTool)
    tool.on_invoke_tool = AsyncMock(side_effect=mock_on_invoke_tool)

    context = MagicMock(spec=ToolContext)

    result = await backend.invoke(tool, context, "{}")
    assert result == "direct_result"
    tool.on_invoke_tool.assert_called_once_with(context, "{}")

