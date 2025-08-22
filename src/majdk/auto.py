"""Auto orchestration helpers to create multi-agent pipelines in one line.

Provides a convenience `auto_orchestrate(...)` that creates N agents from a
high-level goal, attaches tools from selected sources (builtin, MCP), and
returns an `Orchestrator` configured for sequential, parallel, or map-reduce.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Any, Dict
import json

from .agent import Agent
from .models import BaseModel
from .orchestrator import Orchestrator, map_reduce

try:
    # Optional MCP integration
    from .mcp import MCPManager, register_mcp_tools
except Exception:  # pragma: no cover - optional
    MCPManager = None  # type: ignore
    register_mcp_tools = None  # type: ignore


_DEFAULT_ROLES: List[str] = [
    "You are a web researcher focused on factual evidence. Goal: {goal}. Keep answers concise.",
    "You are an analyst; identify trade-offs and risks. Goal: {goal}. Be precise.",
    "You are a planner; outline clear, actionable steps. Goal: {goal}. Keep it practical.",
    "You are a synthesizer; produce a final, clear answer. Goal: {goal}. Be brief and definitive.",
]


def _make_agent(
    model_factory: Callable[[], BaseModel],
    system_prompt: str,
    sources: Sequence[str],
) -> Agent:
    load_builtin = "builtin" in sources
    agent = Agent(
        model=model_factory(),
        system_prompt=system_prompt,
        load_builtin_tools=load_builtin,
        max_iterations=4,
    )
    return agent


def auto_orchestrate(
    goal: str,
    num_agents: int,
    model_factory: Callable[[], BaseModel],
    mode: str = "sequential",  # 'sequential' | 'parallel' | 'map_reduce'
    sources: Sequence[str] = ("builtin",),  # e.g., ("builtin", "mcp")
    pass_history: bool = True,
    preserve_query: bool = True,
    mcp_manager: Optional["MCPManager"] = None,
    mcp_namespace: str = "mcp",
    roles: Optional[List[str]] = None,
    verbose: bool = False,
) -> Orchestrator:
    """Create an orchestrator from a goal, number of agents, and tool sources.

    - model_factory: callable that returns a new model instance per agent.
    - sources: include "builtin" for built-in tools, and/or "mcp" to register MCP tools.
    - mode: 'sequential', 'parallel', or 'map_reduce'.
    - When using MCP, pass an initialized MCPManager (connected) and optionally a namespace.
    """
    assert num_agents >= 1, "num_agents must be >= 1"
    roles = roles or _DEFAULT_ROLES

    # Build agents
    built: List[Tuple[str, Agent]] = []
    created_names: List[str] = []
    for i in range(num_agents):
        role_template = roles[i % len(roles)]
        prompt = role_template.format(goal=goal)
        a = _make_agent(model_factory, prompt, sources)

        # Attach MCP tools per agent (optional)
        mcp_enabled = "mcp" in sources and mcp_manager is not None and register_mcp_tools is not None
        mcp_tools_count = 0
        if mcp_enabled:
            try:
                names = register_mcp_tools(a, mcp_manager, namespace=mcp_namespace)
                mcp_tools_count = len(names)
            except Exception:
                mcp_tools_count = 0

        agent_name = f"agent_{i+1}"
        built.append((agent_name, a))
        created_names.append(agent_name)
        if verbose:
            print(f"[auto] Created {agent_name} | builtin_tools={'yes' if 'builtin' in sources else 'no'} | mcp_tools={mcp_tools_count} | role='{prompt}'")

    # Optionally create a synthesizer for parallel reduce
    synthesizer = None  # type: ignore
    if mode == "map_reduce" or (mode == "parallel"):
        synth_prompt = (
            f"You are a synthesizer; merge multiple agent results into a single, clear answer. Goal: {goal}."
        )
        synthesizer = _make_agent(model_factory, synth_prompt, sources)
        if "mcp" in sources and mcp_manager is not None and register_mcp_tools is not None:
            try:
                names = register_mcp_tools(synthesizer, mcp_manager, namespace=mcp_namespace)
                if verbose:
                    print(f"[auto] Synthesizer created | mcp_tools={len(names)} | role='{synth_prompt}'")
            except Exception:
                if verbose:
                    print("[auto] Synthesizer created | mcp_tools=0 (registration failed)")
        elif verbose:
            print(f"[auto] Synthesizer created | role='{synth_prompt}'")

    if verbose:
        print(f"[auto] Built {len(created_names)} agents: {', '.join(created_names)} | mode={mode}")

    # Assemble orchestrator
    if mode == "sequential":
        return Orchestrator(built, mode="sequential", pass_history=pass_history, preserve_query=preserve_query)
    elif mode == "parallel":
        return Orchestrator(built, mode="parallel", pass_history=pass_history, preserve_query=preserve_query, final_agent=synthesizer)
    elif mode == "map_reduce":
        # Use map_reduce helper
        return map_reduce([a for _, a in built], reduce_agent=synthesizer)
    else:
        raise ValueError("mode must be one of 'sequential', 'parallel', 'map_reduce'")


def auto_plan_orchestrate(
    goal: str,
    model_factory: Callable[[], BaseModel],
    sources: Sequence[str] = ("builtin",),
    mode: Optional[str] = None,  # if None, planner may choose
    max_agents: int = 4,
    pass_history: bool = True,
    preserve_query: bool = True,
    mcp_manager: Optional["MCPManager"] = None,
    mcp_namespace: str = "mcp",
    verbose: bool = False,
) -> Orchestrator:
    """Ask the model to propose agents and roles from the goal and available tools.

    Returns an Orchestrator built according to the plan. Falls back to a 3-agent
    sequential pipeline with default roles if planning fails.
    """
    # Build a temporary agent to collect tool schemas (and optionally register MCP)
    temp_agent = _make_agent(model_factory, "Planner agent (temporary)", sources)
    mcp_tool_names: List[str] = []
    if "mcp" in sources and mcp_manager is not None and register_mcp_tools is not None:
        try:
            mcp_tool_names = register_mcp_tools(temp_agent, mcp_manager, namespace=mcp_namespace)
        except Exception:
            mcp_tool_names = []

    # Collect available tools for the planner prompt
    try:
        schemas = temp_agent._get_tool_schemas()  # type: ignore[attr-defined]
        available_tools = [s.get("function", {}).get("name", "") for s in schemas]
        available_tools = [t for t in available_tools if t]
    except Exception:
        available_tools = []

    # Planner system prompt
    system = (
        "You are an orchestration planner. Given a high-level GOAL and a list of available tools, "
        "produce a JSON plan describing agents to create. Output ONLY valid JSON, no prose.\n\n"
        "JSON schema:\n"
        "{\n  \"mode\": \"sequential|parallel|map_reduce\",\n  \"agents\": [\n    {\n      \"name\": string,\n      \"role\": string,\n      \"use_tools\": [string, ...]  // names from available tools (optional)\n    }\n  ],\n  \"synthesizer\": { \"role\": string } | null\n}\n\n"
        f"Available tools: {available_tools}\n"
        f"Max agents: {max_agents}"
    )
    user = f"GOAL: {goal}"

    # Call the model directly to get a plan
    model = model_factory()
    resp = model.generate([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    content = getattr(resp, "content", "") or ""

    # Extract JSON
    plan_obj: Dict[str, Any]
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            plan_text = content[start : end + 1]
        else:
            plan_text = content
        plan_obj = json.loads(plan_text)
    except Exception:
        if verbose:
            print("[auto-plan] Failed to parse plan JSON. Falling back to defaults.")
        return auto_orchestrate(
            goal=goal,
            num_agents=min(3, max_agents),
            model_factory=model_factory,
            mode=(mode or "sequential"),
            sources=sources,
            pass_history=pass_history,
            preserve_query=preserve_query,
            mcp_manager=mcp_manager,
            mcp_namespace=mcp_namespace,
            verbose=verbose,
        )

    # Build agents from plan
    planned_mode = (plan_obj.get("mode") or mode or "sequential").lower()
    agents_spec = plan_obj.get("agents", [])[:max_agents]
    built: List[Tuple[str, Agent]] = []
    for idx, spec in enumerate(agents_spec):
        name = spec.get("name") or f"agent_{idx+1}"
        role = spec.get("role") or "You are a helpful expert."
        role_full = f"{role} Goal: {goal}."
        a = _make_agent(model_factory, role_full, sources)

        # MCP selective registration using include
        if "mcp" in sources and mcp_manager is not None and register_mcp_tools is not None:
            include = spec.get("use_tools") if isinstance(spec.get("use_tools"), list) else None
            try:
                register_mcp_tools(a, mcp_manager, namespace=mcp_namespace, include=include)
            except Exception:
                pass

        built.append((name, a))
        if verbose:
            used = spec.get("use_tools") if isinstance(spec.get("use_tools"), list) else []
            print(f"[auto-plan] Created {name} | role='{role_full}' | use_tools={used}")

    # Synthesizer
    synthesizer = None  # type: ignore
    synth_spec = plan_obj.get("synthesizer")
    if planned_mode in ("parallel", "map_reduce"):
        synth_role = None
        if isinstance(synth_spec, dict):
            synth_role = synth_spec.get("role")
        synth_prompt = synth_role or (
            f"You are a synthesizer; merge multiple agent results into a single, clear answer. Goal: {goal}."
        )
        synthesizer = _make_agent(model_factory, synth_prompt, sources)
        if "mcp" in sources and mcp_manager is not None and register_mcp_tools is not None:
            try:
                register_mcp_tools(synthesizer, mcp_manager, namespace=mcp_namespace)
            except Exception:
                pass
        if verbose:
            print(f"[auto-plan] Synthesizer created | role='{synth_prompt}'")

    if verbose:
        names = ", ".join([n for n, _ in built])
        print(f"[auto-plan] Built {len(built)} agents: {names} | mode={planned_mode}")

    # Assemble orchestrator
    if planned_mode == "sequential":
        return Orchestrator(built, mode="sequential", pass_history=pass_history, preserve_query=preserve_query)
    elif planned_mode == "parallel":
        return Orchestrator(built, mode="parallel", pass_history=pass_history, preserve_query=preserve_query, final_agent=synthesizer)
    elif planned_mode == "map_reduce":
        return map_reduce([a for _, a in built], reduce_agent=synthesizer)
    else:
        # Fallback
        if verbose:
            print(f"[auto-plan] Unknown mode '{planned_mode}', defaulting to sequential")
        return Orchestrator(built, mode="sequential", pass_history=pass_history, preserve_query=preserve_query)
