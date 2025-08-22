"""MAJD Agent Kit - Lightweight AI Agent SDK"""

import warnings

# Deprecation notice for package rename
warnings.warn(
    "Package 'majdk' has been renamed to 'labib'. Please switch imports to 'labib'.",
    DeprecationWarning,
    stacklevel=2,
)

from .agent import Agent, tool
from .models import OpenAIModel, OllamaModel, MockModel
from .orchestrator import Orchestrator, chain, map_reduce
from .mcp import MCPManager, register_mcp_tools, MCPNotAvailableError
from .auto import auto_orchestrate, auto_plan_orchestrate

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "tool",
    "OpenAIModel",
    "OllamaModel",
    "MockModel",
    "Orchestrator",
    "chain",
    "map_reduce",
    "MCPManager",
    "register_mcp_tools",
    "MCPNotAvailableError",
    "auto_orchestrate",
    "auto_plan_orchestrate",
]
