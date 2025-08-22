"""Multi-agent orchestration demo for LABIB"""

from labib import Agent, MockModel, chain, map_reduce
from labib.tools.web import fetch  # tool name: "web.fetch"
from labib.tools.calculator import calculate  # tool name: "calculator"


def build_agents():
    researcher = Agent(
        model=MockModel(),
        system_prompt="You are a diligent web researcher. Use web.fetch when needed and return concise factual notes.",
        load_builtin_tools=False,
        tools=[fetch],
        max_iterations=3,
    )

    analyst = Agent(
        model=MockModel(),
        system_prompt="You are an analyst. Use calculator for quick math and return one-sentence insights.",
        load_builtin_tools=False,
        tools=[calculate],
        max_iterations=3,
    )

    synthesizer = Agent(
        model=MockModel(),
        system_prompt="Synthesize the following inputs into a short, clear answer.",
        load_builtin_tools=False,
        max_iterations=3,
    )

    return researcher, analyst, synthesizer


def main():
    researcher, analyst, synthesizer = build_agents()
    query = "Fetch https://example.com and estimate 12*7; then provide a single-sentence takeaway."

    # Sequential chain with history passing
    seq = chain(researcher, analyst, pass_history=True)
    seq_out = seq.run(query)
    print("=== Sequential Output ===")
    print(seq_out)

    # Parallel map with reduce synthesizer
    mr = map_reduce([researcher, analyst], reduce_agent=synthesizer)
    par_out = mr.run(query)
    print("\n=== Parallel (Map-Reduce) Output ===")
    print(par_out)


if __name__ == "__main__":
    main()
