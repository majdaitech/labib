"""Demonstrate one-line auto multi-agent orchestration.

Usage examples:
  # Mock model, sequential, built-in tools only
  python examples/auto_orchestration.py --goal "Summarize recent LLM evals" --model-type mock --mode sequential

  # OpenAI model, parallel, with MCP tools
  python examples/auto_orchestration.py --goal "Compare 3 web frameworks and recommend one" \
      --model-type openai --openai-model gpt-4o-mini --mode parallel \
      --sources builtin,mcp --mcp-cmd "path/to/mcp-server --flag"

  # Ollama model (no tool-calls), map-reduce
  python examples/auto_orchestration.py --goal "Pros/cons of Python for backend" \
      --model-type ollama --ollama-model llama3 --mode map_reduce
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from typing import List

from labib import (
    auto_orchestrate,
    auto_plan_orchestrate,
    OpenAIModel,
    OllamaModel,
    MockModel,
)

# MCP is optional; import in a guarded way
try:
    from labib import MCPManager
except Exception:  # pragma: no cover
    MCPManager = None  # type: ignore


def build_model_factory(args):
    if args.model_type == "mock":
        return lambda: MockModel()
    elif args.model_type == "openai":
        api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[error] OPENAI_API_KEY not set and --openai-key not provided", file=sys.stderr)
            sys.exit(2)
        model_name = args.openai_model or "gpt-4o-mini"
        return lambda: OpenAIModel(api_key=api_key, model=model_name)
    elif args.model_type == "ollama":
        return lambda: OllamaModel(model=args.ollama_model, base_url=args.ollama_url)
    else:
        print(f"[error] unsupported model_type: {args.model_type}", file=sys.stderr)
        sys.exit(2)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Auto multi-agent orchestration demo")
    p.add_argument("--goal", required=True, help="High-level task/goal for the agents")
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--mode", choices=["sequential", "parallel", "map_reduce"], default="sequential")
    p.add_argument("--sources", default="builtin", help="Comma list of sources: builtin,mcp")
    p.add_argument("--verbose", action="store_true", help="Print details about created agents and tools")
    p.add_argument("--auto-plan", action="store_true", help="Let the model plan agents/roles/mode automatically")
    p.add_argument("--max-agents", type=int, default=4, help="Max agents when using --auto-plan")

    # Model selection
    p.add_argument("--model-type", choices=["mock", "openai", "ollama"], default="mock")
    # OpenAI
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--openai-key", default=None)
    # Ollama
    p.add_argument("--ollama-model", default="gemma3:1b")
    p.add_argument("--ollama-url", default="http://localhost:11434")

    # MCP
    p.add_argument("--mcp-cmd", default=None, help="Command to start MCP server (quoted string)")
    p.add_argument("--mcp-namespace", default="mcp")

    args = p.parse_args(argv)

    sources = tuple([s.strip() for s in args.sources.split(",") if s.strip()])
    model_factory = build_model_factory(args)

    mcp_mgr = None
    if "mcp" in sources:
        if MCPManager is None:
            print("[warning] MCP extra not installed. Run: pip install -e .[mcp]", file=sys.stderr)
        elif not args.mcp_cmd:
            print("[warning] --mcp-cmd not provided; MCP tools won't be registered", file=sys.stderr)
        else:
            cmd = shlex.split(args.mcp_cmd)
            mcp_mgr = MCPManager(command=cmd)
            mcp_mgr.connect()

    if args.model_type == "ollama" and "mcp" in sources:
        print("[note] Ollama backend in this kit does not make tool calls; MCP tools won't be invoked.", file=sys.stderr)

    if args.auto_plan:
        orch = auto_plan_orchestrate(
            goal=args.goal,
            model_factory=model_factory,
            sources=sources,
            mode=args.mode,  # optional override
            max_agents=args.max_agents,
            pass_history=True,
            preserve_query=True,
            mcp_manager=mcp_mgr,
            mcp_namespace=args.mcp_namespace,
            verbose=args.verbose,
        )
    else:
        orch = auto_orchestrate(
            goal=args.goal,
            num_agents=args.num_agents,
            model_factory=model_factory,
            mode=args.mode,
            sources=sources,
            mcp_manager=mcp_mgr,
            mcp_namespace=args.mcp_namespace,
            pass_history=True,
            preserve_query=True,
            verbose=args.verbose,
        )

    out = orch.run(args.goal)
    print("=== OUTPUT ===")
    print(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
