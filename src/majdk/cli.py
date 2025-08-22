"""Command line interface for MAJD Agent Kit"""

import argparse
import sys
import os
from pathlib import Path
from .agent import Agent
from .models import OpenAIModel, OllamaModel, MockModel
from .memory import Memory


def create_agent(model_type: str, **kwargs) -> Agent:
    """Create agent with specified model"""
    if model_type == "openai":
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key required. Use --api-key or set OPENAI_API_KEY")
            sys.exit(1)
        model = OpenAIModel(api_key, kwargs.get("model", "gpt-3.5-turbo"))
    elif model_type == "ollama":
        model = OllamaModel(kwargs.get("model", "llama2"), kwargs.get("base_url", "http://localhost:11434"))
    elif model_type == "mock":
        model = MockModel()
    else:
        print(f"Error: Unknown model type '{model_type}'. Use: openai, ollama, or mock")
        sys.exit(1)
    
    memory = Memory(kwargs.get("log_file", "agent_logs.jsonl"))
    return Agent(model, memory)


def interactive_mode(agent: Agent):
    """Run agent in interactive mode"""
    print("MAJD Agent Kit - Interactive Mode")
    print("Type 'quit' or 'exit' to stop, 'clear' to clear memory\n")
    
    while True:
        try:
            query = input("🤖 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                break
            elif query.lower() == 'clear':
                agent.memory.clear()
                print("Memory cleared.")
                continue
            elif not query:
                continue
            
            print("\n🧠 Thinking...")
            response = agent.run(query)
            print(f"💡 Response: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="MAJD Agent Kit CLI")
    
    # Model configuration
    parser.add_argument("--model-type", choices=["openai", "ollama", "mock"], 
                       default="mock", help="Model type to use")
    parser.add_argument("--model", default=None, help="Specific model name")
    parser.add_argument("--api-key", default=None, help="API key for OpenAI")
    parser.add_argument("--base-url", default="http://localhost:11434", 
                       help="Base URL for Ollama")
    
    # Logging
    parser.add_argument("--log-file", default="agent_logs.jsonl", 
                       help="Log file path")
    
    # Execution modes
    parser.add_argument("--query", help="Single query to execute")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Create agent
    agent = create_agent(
        args.model_type,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        log_file=args.log_file
    )
    
    if args.query:
        # Single query mode
        print(f"Query: {args.query}")
        response = agent.run(args.query)
        print(f"Response: {response}")
    elif args.interactive:
        # Interactive mode
        interactive_mode(agent)
    else:
        # Default to interactive
        interactive_mode(agent)


if __name__ == "__main__":
    main()
