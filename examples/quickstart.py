"""Quick start example for MAJD Agent Kit"""

import os
from labib import Agent, tool, MockModel, OpenAIModel


# Custom tool example
@tool("greet", "Greet someone by name")
def greet_person(name: str) -> str:
    """Greet a person with their name"""
    return f"Hello, {name}! Nice to meet you."


def main():
    """Quick start demo"""
    print("🤖 LABIB - Quick Start Demo\n")
    
    # Create agent with mock model (no API keys needed)
    model = MockModel()
    agent = Agent(model)
    
    # Add custom tool
    agent.add_tool(greet_person)
    
    # Example queries
    queries = [
        "What is 15 * 24?",
        "Calculate the square root of 144",
        "Greet Alice",
        "List the files in the current directory",
        "What's 2 + 2?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"📝 Query {i}: {query}")
        response = agent.run(query)
        print(f"💡 Response: {response}\n")
        print("-" * 50)
    
    # Show logged steps
    print("\n📊 Agent Steps Log:")
    steps = agent.memory.get_steps()
    for step in steps[-5:]:  # Show last 5 steps
        print(f"[{step['step_type'].upper()}] {step['content']}")
    
    print(f"\n✅ Full log saved to: {agent.memory.log_file}")


def openai_example():
    """Example using OpenAI (requires API key)"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable to run this example")
        return
    
    model = OpenAIModel(api_key)
    agent = Agent(model)
    
    response = agent.run("What's the weather like? Just give a general response.")
    print(f"OpenAI Response: {response}")


if __name__ == "__main__":
    main()
    
    # Uncomment to test OpenAI
    # openai_example()
