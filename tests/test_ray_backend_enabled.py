"""Tests for tool invocation backend when Ray is enabled (RAY_BACKEND=1)."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.tool import FunctionTool
from agents.tool_context import ToolContext

# Check if we should run these tests
TEST_RAY_BACKEND = os.environ.get("TEST_RAY_BACKEND") == "1"

# Skip all tests in this file if TEST_RAY_BACKEND is not set
pytestmark = pytest.mark.skipif(
    not TEST_RAY_BACKEND,
    reason="Skipping Ray backend tests. Set TEST_RAY_BACKEND=1 to run these tests.",
)

# Only import if we're actually running these tests
if TEST_RAY_BACKEND:
    try:
        import ray
    except ImportError as e:
        raise RuntimeError("Ray needs to be installed to run the Ray backend tests.") from e

    # Import after setting RAY_BACKEND to ensure the module loads correctly
    from agents.tool_invocation_backend import (
        RayToolInvocationBackend,
        _ray_execute_function_tool,
        _RayToolCallPayload,
    )
else:
    # Dummy imports to avoid collection errors
    ray = None  # type: ignore[assignment]
    RayToolInvocationBackend = None  # type: ignore[assignment, misc]
    _RayToolCallPayload = None  # type: ignore[assignment, misc]
    _ray_execute_function_tool = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_ray_backend_initialization():
    """Test that RayToolInvocationBackend initializes Ray if needed."""
    import agents.setup_ray

    # Save the original state
    original_initialized = agents.setup_ray._ray_initialized

    try:
        # Reset the initialization flag
        agents.setup_ray._ray_initialized = False

        # Mock ray.init to avoid actual initialization
        with patch("agents.setup_ray.ray") as mock_ray:
            mock_ray.is_initialized.return_value = False
            from agents.setup_ray import ensure_ray_initialized

            ensure_ray_initialized()
            mock_ray.init.assert_called_once()

            # Reset for second test
            agents.setup_ray._ray_initialized = False
            mock_ray.is_initialized.return_value = True
            ensure_ray_initialized()
            # Should not call init again since ray.is_initialized() returns True
            assert mock_ray.init.call_count == 1
    finally:
        # Restore original state
        agents.setup_ray._ray_initialized = original_initialized


@pytest.mark.asyncio
async def test_ray_backend_invoke_fallback_if_ray_unavailable():
    """Test fallback to inline execution if Ray module becomes unavailable."""
    import agents.tool_invocation_backend as tib_module

    # Temporarily set ray to None to simulate it not being available
    original_ray = tib_module.ray  # type: ignore[attr-defined]
    try:
        tib_module.ray = None  # type: ignore[attr-defined, assignment]

        backend = RayToolInvocationBackend()

        async def mock_invoke(*args):
            return "fallback_result"

        backend._fallback_backend = MagicMock()
        backend._fallback_backend.invoke = AsyncMock(side_effect=mock_invoke)

        tool = MagicMock(spec=FunctionTool)
        tool.name = "mock_tool"
        context = MagicMock(spec=ToolContext)

        result = await backend.invoke(tool, context, "{}")
        assert result == "fallback_result"
        backend._fallback_backend.invoke.assert_called_once()
    finally:
        # Restore original ray module
        tib_module.ray = original_ray  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ray_backend_invoke_with_resource_args():
    """Test that ray_remote_args are passed to .options()."""
    import agents.tool_invocation_backend as tib_module

    # Mock the global _ray_execute_function_tool
    mock_remote_func = MagicMock()
    mock_options_ret = MagicMock()
    mock_obj_ref = MagicMock()

    mock_remote_func.options.return_value = mock_options_ret
    mock_options_ret.remote.return_value = mock_obj_ref

    # Save original and replace
    original_func = tib_module._ray_execute_function_tool
    try:
        tib_module._ray_execute_function_tool = mock_remote_func

        backend = RayToolInvocationBackend(ray_remote_args={"num_cpus": 2})

        # Mock _ray_get_async
        with patch.object(backend, "_ray_get_async", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = "ray_result"

            tool = MagicMock(spec=FunctionTool)
            tool.name = "mock_tool"
            tool._bypass_ray_backend = False  # Ensure it doesn't bypass
            context = MagicMock(spec=ToolContext)

            result = await backend.invoke(tool, context, "{}")

            assert result == "ray_result"
            mock_remote_func.options.assert_called_once_with(num_cpus=2)
            mock_options_ret.remote.assert_called_once()

            # Verify payload construction
            call_args = mock_options_ret.remote.call_args
            payload = call_args[0][0]
            assert isinstance(payload, _RayToolCallPayload)
            assert payload.func_tool == tool
            assert payload.tool_arguments == "{}"
    finally:
        # Restore original
        tib_module._ray_execute_function_tool = original_func


@pytest.mark.asyncio
async def test_ray_backend_serialization_failure_fallback():
    """Test fallback if Ray task submission fails with TypeError."""
    import agents.tool_invocation_backend as tib_module

    mock_remote_func = MagicMock()
    mock_remote_func.options.return_value.remote.side_effect = TypeError("Pickle error")

    # Save original and replace
    original_func = tib_module._ray_execute_function_tool
    try:
        tib_module._ray_execute_function_tool = mock_remote_func

        backend = RayToolInvocationBackend()

        async def mock_invoke(*args):
            return "fallback_result"

        backend._fallback_backend = MagicMock()
        backend._fallback_backend.invoke = AsyncMock(side_effect=mock_invoke)

        tool = MagicMock(spec=FunctionTool)
        tool.name = "mock_tool"
        context = MagicMock(spec=ToolContext)

        result = await backend.invoke(tool, context, "{}")

        assert result == "fallback_result"
        backend._fallback_backend.invoke.assert_called_once()
    finally:
        # Restore original
        tib_module._ray_execute_function_tool = original_func


@pytest.mark.asyncio
async def test_ray_backend_enabled_via_env_var():
    """Test that Ray backend is used when RAY_BACKEND=1 is set."""

    from agents.run import RunConfig

    # Create a new RunConfig - it should use Ray backend
    config = RunConfig()
    assert isinstance(config.tool_invocation_backend, RayToolInvocationBackend)
