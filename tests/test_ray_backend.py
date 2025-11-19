from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.tool import FunctionTool
from agents.tool_context import ToolContext

# Check if ray is importable for conditional skipping
try:
    import ray  # noqa

    from agents.tool_invocation_backend import RayToolInvocationBackend, _RayToolCallPayload
except ImportError:
    ray = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_ray_backend_initialization():
    """Test that RayToolInvocationBackend initializes Ray if needed."""
    if not ray:
        pytest.skip("Ray is not installed")

    with patch("agents.tool_invocation_backend.ray") as mock_ray:
        mock_ray.is_initialized.return_value = False
        with patch("agents.setup_ray.ray", mock_ray):
            _backend = RayToolInvocationBackend(auto_init=True)
            import agents.setup_ray

            agents.setup_ray._ray_initialized = False
            from agents.setup_ray import ensure_ray_initialized

            ensure_ray_initialized()
            mock_ray.init.assert_called_once()

            mock_ray.is_initialized.return_value = True
            ensure_ray_initialized()
            # Should not call init again
            assert mock_ray.init.call_count == 1


@pytest.mark.asyncio
async def test_ray_backend_invoke_fallback_if_no_ray():
    """Test fallback to inline execution if Ray is not present."""
    # Mock ray being None in tool_invocation_backend module
    with patch("agents.tool_invocation_backend.ray", None):
        backend = RayToolInvocationBackend()

        async def mock_invoke(*args):
            return "fallback_result"

        backend._fallback_backend = MagicMock()
        backend._fallback_backend.invoke.side_effect = mock_invoke

        tool = MagicMock(spec=FunctionTool)
        context = MagicMock(spec=ToolContext)

        result = await backend.invoke(tool, context, "{}")
        assert result == "fallback_result"
        backend._fallback_backend.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_ray_backend_invoke_with_resource_args():
    """Test that ray_remote_args are passed to .options()."""
    if not ray:
        pytest.skip("Ray is not installed")

    with patch("agents.tool_invocation_backend.ray") as mock_ray:
        mock_ray.is_initialized.return_value = True
        # Mock the global _ray_execute_function_tool
        mock_remote_func = MagicMock()
        mock_options_ret = MagicMock()
        mock_obj_ref = MagicMock()

        mock_remote_func.options.return_value = mock_options_ret
        mock_options_ret.remote.return_value = mock_obj_ref

        # We need to patch the global in the module
        with patch("agents.tool_invocation_backend._ray_execute_function_tool", mock_remote_func):
            backend = RayToolInvocationBackend(ray_remote_args={"num_cpus": 2})

            # Mock _ray_get_async
            with patch.object(backend, "_ray_get_async", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = "ray_result"

                tool = MagicMock(spec=FunctionTool)
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


@pytest.mark.asyncio
async def test_ray_backend_serialization_failure_fallback():
    """Test fallback if Ray task submission fails with TypeError."""
    if not ray:
        pytest.skip("Ray is not installed")

    with patch("agents.tool_invocation_backend.ray") as mock_ray:
        mock_ray.is_initialized.return_value = True

        mock_remote_func = MagicMock()
        mock_remote_func.options.return_value.remote.side_effect = TypeError("Pickle error")

        with patch("agents.tool_invocation_backend._ray_execute_function_tool", mock_remote_func):
            backend = RayToolInvocationBackend()

            async def mock_invoke(*args):
                return "fallback_result"

            backend._fallback_backend = MagicMock()
            backend._fallback_backend.invoke.side_effect = mock_invoke

            tool = MagicMock(spec=FunctionTool)
            context = MagicMock(spec=ToolContext)

            result = await backend.invoke(tool, context, "{}")

            assert result == "fallback_result"
            backend._fallback_backend.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_ray_backend_enabled_via_env_var(monkeypatch):
    """Test that Ray backend is used when RAY_BACKEND=1 is set."""
    if not ray:
        pytest.skip("Ray is not installed")

    monkeypatch.setenv("RAY_BACKEND", "1")

    from agents.run import RunConfig

    # Create a new RunConfig - it should use Ray backend
    config = RunConfig()
    assert isinstance(config.tool_invocation_backend, RayToolInvocationBackend)


@pytest.mark.asyncio
async def test_ray_backend_disabled_by_default(monkeypatch):
    """Test that Async backend is used by default when RAY_BACKEND is not set."""
    monkeypatch.delenv("RAY_BACKEND", raising=False)

    from agents.run import RunConfig
    from agents.tool_invocation_backend import AsyncToolInvocationBackend

    # Create a new RunConfig - it should use Async backend
    config = RunConfig()
    assert isinstance(config.tool_invocation_backend, AsyncToolInvocationBackend)
