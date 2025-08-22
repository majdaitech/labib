"""Simple multi-agent orchestration utilities"""

from typing import List, Tuple, Optional, Callable, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .agent import Agent


class Orchestrator:
    """Route a query through multiple Agents (sequential or parallel)."""

    def __init__(
        self,
        agents: List[Tuple[str, Agent]],
        mode: str = "sequential",
        pass_history: bool = False,
        preserve_query: bool = True,
        final_agent: Optional[Agent] = None,
        aggregator: Optional[Callable[[List[Tuple[str, str]]], str]] = None,
        max_workers: int = 4,
    ):
        if not agents:
            raise ValueError("At least one agent is required")
        self.agents = agents
        if mode not in ("sequential", "parallel"):
            raise ValueError("mode must be 'sequential' or 'parallel'")
        self.mode = mode
        self.pass_history = pass_history
        self.preserve_query = preserve_query
        self.final_agent = final_agent
        self.aggregator = aggregator
        self.max_workers = max_workers

    def run(self, query: str) -> str:
        if self.mode == "sequential":
            return self._run_sequential(query)
        else:
            return self._run_parallel(query)

    def _run_sequential(self, query: str) -> str:
        original = query
        last_output = ""
        history: List[Tuple[str, str]] = []  # (name, output)
        for name, agent in self.agents:
            agent.memory.log_step("thought", f"[Orchestrator] → {name}: processing")
            extra = None
            if self.pass_history and history:
                summary = self._format_history(history)
                extra = [{"role": "system", "content": f"Context from previous agents:\n{summary}"}]
            prompt_query = original if self.preserve_query else (last_output or original)
            out = agent.run(prompt_query, extra_messages=extra)
            history.append((name, out))
            last_output = out
        return last_output

    def _run_parallel(self, query: str) -> str:
        results: List[Tuple[str, str]] = []  # (name, output)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(agent.run, query): name for name, agent in self.agents}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    out = f"[Error from {name}]: {e}"
                results.append((name, out))

        # If a final_agent is provided, synthesize a final answer with all outputs as context
        if self.final_agent is not None:
            context = self._format_history(results)
            extra = [{"role": "system", "content": f"Parallel agents produced these results:\n{context}"}]
            return self.final_agent.run(query, extra_messages=extra)

        # Or use an aggregator function over named outputs
        if self.aggregator is not None:
            return self.aggregator(results)

        # Default: join outputs
        joined = "\n\n".join([f"[{name}] {text}" for name, text in results])
        return joined

    @staticmethod
    def _format_history(pairs: List[Tuple[str, str]]) -> str:
        return "\n".join([f"- {name}: {text}" for name, text in pairs])


def chain(*agents: Agent, pass_history: bool = True, preserve_query: bool = True) -> Orchestrator:
    """Sequential chain of Agents (auto-named)."""
    named = [(f"agent_{i+1}", a) for i, a in enumerate(agents)]
    return Orchestrator(named, mode="sequential", pass_history=pass_history, preserve_query=preserve_query)


def map_reduce(map_agents: List[Agent], reduce_agent: Agent) -> Orchestrator:
    """Parallel map with a reduce synthesizer Agent."""
    named = [(f"map_{i+1}", a) for i, a in enumerate(map_agents)]
    return Orchestrator(named, mode="parallel", final_agent=reduce_agent)
