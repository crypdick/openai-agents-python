import asyncio
import json
import shutil
from typing import Any

from mcp import Tool as MCPTool
from mcp.types import (
    CallToolResult,
    Content,
    GetPromptResult,
    ListPromptsResult,
    PromptMessage,
    TextContent,
)

from agents.mcp import MCPServer
from agents.mcp.server import _MCPServerWithClientSession
from agents.mcp.util import ToolFilter
from agents.setup_ray import use_ray

tee = shutil.which("tee") or ""
assert tee, "tee not found"


def _get_ray_execute_fake_mcp_tool():
    """Lazily create the Ray remote function when needed."""
    import ray

    @ray.remote
    def _execute_fake_mcp_tool(tool_name: str, arguments: dict[str, Any] | None) -> str:
        """Execute the fake MCP tool computation in a Ray worker."""
        return f"result_{tool_name}_{json.dumps(arguments)}"

    return _execute_fake_mcp_tool


# Added dummy stream classes for patching stdio_client to avoid real I/O during tests
class DummyStream:
    async def send(self, msg):
        pass

    async def receive(self):
        raise Exception("Dummy receive not implemented")


class DummyStreamsContextManager:
    async def __aenter__(self):
        return (DummyStream(), DummyStream())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class _TestFilterServer(_MCPServerWithClientSession):
    """Minimal implementation of _MCPServerWithClientSession for testing tool filtering"""

    def __init__(self, tool_filter: ToolFilter, server_name: str):
        # Initialize parent class properly to avoid type errors
        super().__init__(
            cache_tools_list=False,
            client_session_timeout_seconds=None,
            tool_filter=tool_filter,
        )
        self._server_name: str = server_name
        # Override some attributes for test isolation
        self.session = None
        self._cleanup_lock = asyncio.Lock()

    def create_streams(self):
        raise NotImplementedError("Not needed for filtering tests")

    @property
    def name(self) -> str:
        return self._server_name


class FakeMCPServer(MCPServer):
    def __init__(
        self,
        tools: list[MCPTool] | None = None,
        tool_filter: ToolFilter = None,
        server_name: str = "fake_mcp_server",
        use_ray_for_tools: bool = False,
    ):
        
        super().__init__(use_structured_content=False, use_ray_for_tools=use_ray_for_tools)
        self.tools: list[MCPTool] = tools or []
        self.tool_calls: list[str] = []
        self.tool_results: list[str] = []
        self.tool_filter = tool_filter
        self._server_name = server_name
        self._custom_content: list[Content] | None = None

    def add_tool(self, name: str, input_schema: dict[str, Any]):
        self.tools.append(MCPTool(name=name, inputSchema=input_schema))

    async def connect(self):
        pass

    async def cleanup(self):
        pass

    async def list_tools(self, run_context=None, agent=None):
        tools = self.tools

        # Apply tool filtering using the REAL implementation
        if self.tool_filter is not None:
            # Use the real _MCPServerWithClientSession filtering logic
            filter_server = _TestFilterServer(self.tool_filter, self.name)
            tools = await filter_server._apply_tool_filter(tools, run_context, agent)

        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        # Track tool calls in main process
        self.tool_calls.append(tool_name)

        # Delegate the actual computation to Ray if enabled for this server
        if self.use_ray_for_tools and use_ray():
            import ray

            execute_func = _get_ray_execute_fake_mcp_tool()
            loop = asyncio.get_running_loop()
            object_ref = execute_func.remote(tool_name, arguments)
            result_text = await loop.run_in_executor(None, lambda: ray.get(object_ref))
        else:
            result_text = f"result_{tool_name}_{json.dumps(arguments)}"

        self.tool_results.append(result_text)

        # Allow testing custom content scenarios
        if self._custom_content is not None:
            return CallToolResult(content=self._custom_content)

        return CallToolResult(
            content=[TextContent(text=result_text, type="text")],
        )

    async def list_prompts(self, run_context=None, agent=None) -> ListPromptsResult:
        """Return empty list of prompts for fake server"""
        return ListPromptsResult(prompts=[])

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        """Return a simple prompt result for fake server"""
        content = f"Fake prompt content for {name}"
        message = PromptMessage(role="user", content=TextContent(type="text", text=content))
        return GetPromptResult(description=f"Fake prompt: {name}", messages=[message])

    @property
    def name(self) -> str:
        return self._server_name
