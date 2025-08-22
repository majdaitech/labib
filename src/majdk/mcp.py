"""MCP (Model Context Protocol) integration helpers.

This module provides a thin adapter to connect to MCP servers and register their
exposed tools into an `Agent` using `Agent.add_external_tool`.

Design goals:
- Optional dependency: only required when you use MCP.
- Async under the hood, simple sync API to call from your code.
- Tools discovered from MCP are mapped to the Agent tool schema and executed via
  the MCP session when invoked by the Agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
import asyncio
import os

try:
    # Minimal expected API from the Python MCP SDK
    from mcp import ClientSession
    from mcp.transport.stdio import StdioClientTransport
except Exception:  # broad: optional dependency
    ClientSession = None  # type: ignore
    StdioClientTransport = None  # type: ignore

from .agent import Agent


class MCPNotAvailableError(RuntimeError):
    pass


class MCPManager:
    """Manage a connection to an MCP server and call its tools.

    Usage:
        mgr = MCPManager(command=["your-mcp-server-binary"], env=os.environ)
        mgr.connect()
        tools = mgr.list_tools()
        result = mgr.call_tool("tool_name", {"param": "value"})
    """

    def __init__(
        self,
        command: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.command = command
        self.env = env or os.environ.copy()
        self._session: Optional[ClientSession] = None  # type: ignore
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _require_sdk(self) -> None:
        if ClientSession is None or StdioClientTransport is None:
            raise MCPNotAvailableError(
                "Python MCP SDK not found. Install with: pip install mcp"
            )

    def connect(self) -> None:
        """Establish the MCP session (sync wrapper)."""
        self._require_sdk()
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_connect())

    async def _async_connect(self) -> None:
        assert self.command, "command must be provided to start MCP server (stdio)"
        transport = StdioClientTransport(command=self.command, env=self.env)
        self._session = ClientSession(transport)
        await self._session.__aenter__()  # enter async context manually

    def close(self) -> None:
        if self._session and self._loop:
            self._loop.run_until_complete(self._session.__aexit__(None, None, None))
            self._session = None

    def list_tools(self) -> List[Dict[str, Any]]:
        self._ensure_connected()
        assert self._loop is not None
        return self._loop.run_until_complete(self._async_list_tools())

    async def _async_list_tools(self) -> List[Dict[str, Any]]:
        assert self._session is not None
        # Expected shape: [{"name": str, "description": str, "input_schema": {...}}]
        tools = await self._session.list_tools()  # type: ignore[attr-defined]
        # Normalize to plain dicts
        normalized: List[Dict[str, Any]] = []
        for t in tools:
            name = getattr(t, "name", None) or t.get("name")
            desc = getattr(t, "description", None) or t.get("description", "")
            schema = getattr(t, "input_schema", None) or t.get("input_schema", {})
            normalized.append({"name": name, "description": desc, "input_schema": schema})
        return normalized

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        self._ensure_connected()
        assert self._loop is not None
        return self._loop.run_until_complete(self._async_call_tool(name, arguments))

    async def _async_call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        assert self._session is not None
        # Some SDKs return structured results; we stringify for the agent
        result = await self._session.call_tool(name, arguments)  # type: ignore[attr-defined]
        return result

    def _ensure_connected(self) -> None:
        if self._session is None:
            raise RuntimeError("MCP session is not connected. Call connect() first.")


def _wrap_schema_for_agent(name: str, description: str, input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap MCP input_schema into the Agent's function tool schema."""
    params = input_schema or {"type": "object", "properties": {}, "required": []}
    if params.get("type") != "object":
        # Best effort: wrap as object with free-form
        params = {"type": "object", "properties": {}, "required": []}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }


essential_tool_filter = Callable[[str, Dict[str, Any]], bool]


def register_mcp_tools(
    agent: Agent,
    manager: MCPManager,
    namespace: Optional[str] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    tool_filter: Optional[essential_tool_filter] = None,
) -> List[str]:
    """Discover MCP tools and register them into an Agent as external tools.

    Returns a list of registered tool names.
    """
    tools = manager.list_tools()
    registered: List[str] = []

    for t in tools:
        name = t["name"]
        if include and name not in include:
            continue
        if exclude and name in exclude:
            continue
        if tool_filter and not tool_filter(name, t):
            continue

        tool_name = f"{namespace}.{name}" if namespace else name
        schema = _wrap_schema_for_agent(tool_name, t.get("description", ""), t.get("input_schema", {}))

        def make_handler(n: str) -> Callable[..., Any]:
            def _handler(**kwargs: Any) -> Any:
                # Strip namespace when calling MCP
                raw_name = n.split(".", 1)[1] if "." in n else n
                return manager.call_tool(raw_name, kwargs)
            return _handler

        agent.add_external_tool(
            name=tool_name,
            description=t.get("description", ""),
            parameters=schema,  # already wrapped
            handler=make_handler(tool_name),
        )
        registered.append(tool_name)

    return registered
