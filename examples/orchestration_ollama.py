"""Multi-agent orchestration using Ollama backend"""

from labib import Agent, OllamaModel, chain, map_reduce


def build_agents(model: str = "gemma3:1b", base_url: str = "http://localhost:11434"):
    researcher = Agent(
        model=OllamaModel(model=model, base_url=base_url),
        system_prompt=(
            "You are a diligent researcher. Provide concise, factual bullets."
        ),
        load_builtin_tools=False,  # Ollama backend here does not perform tool calls
        max_iterations=2,
    )

    analyst = Agent(
        model=OllamaModel(model=model, base_url=base_url),
        system_prompt=(
            "You are an analyst. Weigh trade-offs and give practical insight in 1-2 sentences."
        ),
        load_builtin_tools=False,
        max_iterations=2,
    )

    synthesizer = Agent(
        model=OllamaModel(model=model, base_url=base_url),
        system_prompt=(
            "Synthesize prior notes into a single, clear answer (max 3 sentences)."
        ),
        load_builtin_tools=False,
        max_iterations=2,
    )

    return researcher, analyst, synthesizer


def main():
    researcher, analyst, synthesizer = build_agents()
    query = (
        "List key pros and cons of using Python for backend services, then give a short recommendation."
    )

    # Sequential chain with history passing
    seq = chain(researcher, analyst, pass_history=True)
    seq_out = seq.run(query)
    print("=== Sequential Output (Ollama) ===")
    print(seq_out)

    # Parallel map with reduce synthesizer
    mr = map_reduce([researcher, analyst], reduce_agent=synthesizer)
    par_out = mr.run(query)
    print("\n=== Parallel (Map-Reduce) Output (Ollama) ===")
    print(par_out)


if __name__ == "__main__":
    main()
