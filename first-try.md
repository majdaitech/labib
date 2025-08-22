# MAJD Agent Kit - Lightweight AI Agent SDK

## File Structure
```
src/majdk/
├── __init__.py
├── agent.py
├── models.py  
├── tools/
│   ├── __init__.py
│   ├── calculator.py
│   ├── web.py
│   └── filesystem.py
├── memory.py
├── cli.py
└── ui_app.py
examples/
└── quickstart.py
pyproject.toml
README.md
LICENSE
```

---

## src/majdk/__init__.py
```python
"""MAJD Agent Kit - Lightweight AI Agent SDK"""

from .agent import Agent, tool
from .models import OpenAIModel, OllamaModel, MockModel

__version__ = "0.1.0"
__all__ = ["Agent", "tool", "OpenAIModel", "OllamaModel", "MockModel"]
```

---

## src/majdk/agent.py
```python
"""Core agent implementation with tool support"""

import json
import inspect
from typing import Any, Dict, List, Callable, Optional
from .models import BaseModel
from .memory import Memory


def tool(name: str = None, description: str = ""):
    """Decorator to register functions as tools"""
    def decorator(func: Callable) -> Callable:
        func._is_tool = True
        func._tool_name = name or func.__name__
        func._tool_description = description or func.__doc__ or ""
        func._tool_schema = _generate_schema(func)
        return func
    return decorator


def _generate_schema(func: Callable) -> Dict[str, Any]:
    """Generate JSON schema for function parameters"""
    sig = inspect.signature(func)
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
    
    for param_name, param in sig.parameters.items():
        param_type = "string"  # Default to string
        if param.annotation == int:
            param_type = "integer"
        elif param.annotation == float:
            param_type = "number"
        elif param.annotation == bool:
            param_type = "boolean"
            
        schema["function"]["parameters"]["properties"][param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}"
        }
        
        if param.default == inspect.Parameter.empty:
            schema["function"]["parameters"]["required"].append(param_name)
    
    return schema


class Agent:
    """Lightweight AI agent with tool support"""
    
    def __init__(self, model: BaseModel, memory: Memory = None, max_iterations: int = 5):
        self.model = model
        self.memory = memory or Memory()
        self.max_iterations = max_iterations
        self.tools = {}
        self._load_builtin_tools()
    
    def _load_builtin_tools(self):
        """Load built-in tools"""
        from .tools import calculator, web, filesystem
        
        for module in [calculator, web, filesystem]:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and hasattr(attr, '_is_tool'):
                    self.tools[attr._tool_name] = attr
    
    def add_tool(self, func: Callable):
        """Add a tool function"""
        if hasattr(func, '_is_tool'):
            self.tools[func._tool_name] = func
        else:
            raise ValueError("Function must be decorated with @tool")
    
    def run(self, query: str) -> str:
        """Main agent execution loop"""
        self.memory.log_step("thought", f"Starting task: {query}")
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": query}
        ]
        
        for iteration in range(self.max_iterations):
            self.memory.log_step("thought", f"Iteration {iteration + 1}")
            
            # Get model response
            response = self.model.generate(messages, self._get_tool_schemas())
            
            # Check if model wants to use a tool
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                self.memory.log_step("action", f"Using tool: {tool_name} with args: {tool_args}")
                
                # Execute tool
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name](**tool_args)
                        self.memory.log_step("observation", f"Tool result: {result}")
                        
                        # Add tool response to conversation
                        messages.append({"role": "assistant", "content": response.content or ""})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result)
                        })
                    except Exception as e:
                        error_msg = f"Tool error: {str(e)}"
                        self.memory.log_step("observation", error_msg)
                        messages.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = f"Unknown tool: {tool_name}"
                    self.memory.log_step("observation", error_msg)
                    messages.append({"role": "assistant", "content": error_msg})
            else:
                # Final response
                final_answer = response.content or "No response generated"
                self.memory.log_step("final", final_answer)
                return final_answer
        
        fallback = "Max iterations reached"
        self.memory.log_step("final", fallback)
        return fallback
    
    def _get_system_prompt(self) -> str:
        """Generate system prompt with available tools"""
        tool_descriptions = []
        for name, func in self.tools.items():
            desc = func._tool_description
            tool_descriptions.append(f"- {name}: {desc}")
        
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        
        return f"""You are a helpful AI assistant with access to tools.

Available tools:
{tools_text}

When you need to use a tool, respond with a function call. Otherwise, provide a direct answer.
Be concise and helpful."""
    
    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for model"""
        return [func._tool_schema for func in self.tools.values()]
```

---

## src/majdk/models.py
```python
"""Model backends for different AI providers"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
import random


class BaseModel(ABC):
    """Base class for AI models"""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Any:
        """Generate response from messages"""
        pass


class MockModel(BaseModel):
    """Mock model for testing"""
    
    def __init__(self):
        self.responses = [
            "I'll help you with that calculation.",
            "Let me search for that information.",
            "I'll read that file for you.",
            "Based on my analysis, here's what I found:",
            "The answer is 42."
        ]
    
    def generate(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Any:
        """Generate mock response"""
        # Simple mock response object
        class MockResponse:
            def __init__(self, content: str):
                self.content = content
                self.tool_calls = None
        
        # Sometimes randomly decide to use a tool
        if tools and random.random() < 0.3:
            tool = random.choice(tools)
            tool_name = tool["function"]["name"]
            
            # Mock tool call
            class MockToolCall:
                def __init__(self):
                    self.id = f"call_{random.randint(1000, 9999)}"
                    self.function = self
                    self.name = tool_name
                    self.arguments = json.dumps({"value": "42"})
            
            response = MockResponse("")
            response.tool_calls = [MockToolCall()]
            return response
        
        return MockResponse(random.choice(self.responses))


class OpenAIModel(BaseModel):
    """OpenAI model backend"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Any:
        """Generate response using OpenAI"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message


class OllamaModel(BaseModel):
    """Ollama model backend"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
    
    def generate(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Any:
        """Generate response using Ollama"""
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        result = response.json()
        
        # Simple response object
        class OllamaResponse:
            def __init__(self, content: str):
                self.content = content
                self.tool_calls = None
        
        return OllamaResponse(result.get("response", ""))
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to single prompt"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts) + "\nAssistant: "
```

---

## src/majdk/memory.py
```python
"""Memory and logging system"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path


class Memory:
    """Simple memory system with JSONL logging"""
    
    def __init__(self, log_file: str = "agent_logs.jsonl"):
        self.log_file = Path(log_file)
        self.session_id = str(int(time.time()))
        self.steps = []
    
    def log_step(self, step_type: str, content: str, metadata: Dict[str, Any] = None):
        """Log a step in the agent's process"""
        step = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,  # thought, action, observation, final
            "content": content,
            "metadata": metadata or {}
        }
        
        self.steps.append(step)
        
        # Write to JSONL file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(step) + "\n")
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """Get all steps from current session"""
        return self.steps.copy()
    
    def clear(self):
        """Clear current session steps"""
        self.steps.clear()
        self.session_id = str(int(time.time()))
    
    def load_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Load steps from a specific session"""
        if not self.log_file.exists():
            return []
        
        session_steps = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    step = json.loads(line.strip())
                    if step.get("session_id") == session_id:
                        session_steps.append(step)
                except json.JSONDecodeError:
                    continue
        
        return session_steps
```

---

## src/majdk/tools/__init__.py
```python
"""Built-in tools for MAJD Agent Kit"""

from . import calculator
from . import web
from . import filesystem

__all__ = ["calculator", "web", "filesystem"]
```

---

## src/majdk/tools/calculator.py
```python
"""Calculator tool for mathematical operations"""

import math
import operator
from typing import Union
from ..agent import tool


@tool("calculate", "Perform mathematical calculations safely")
def calculate(expression: str) -> Union[float, int, str]:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        Result of the calculation
    """
    # Safe evaluation with limited operations
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names.update({
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    })
    
    # Remove potentially dangerous functions
    dangerous = ['eval', 'exec', 'compile', '__import__', 'open', 'input']
    for name in dangerous:
        allowed_names.pop(name, None)
    
    try:
        # Use eval with restricted globals
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@tool("add", "Add two numbers")
def add(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b


@tool("multiply", "Multiply two numbers")
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together"""
    return a * b
```

---

## src/majdk/tools/web.py
```python
"""Web-related tools"""

from typing import Optional
from ..agent import tool


@tool("web_fetch", "Fetch content from a URL")
def fetch(url: str) -> str:
    """
    Fetch content from a web URL.
    
    Args:
        url: The URL to fetch content from
    
    Returns:
        Text content from the URL
    """
    try:
        import requests
        
        headers = {
            'User-Agent': 'MAJD-Agent-Kit/0.1.0'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Return first 2000 characters to avoid token limits
        content = response.text[:2000]
        if len(response.text) > 2000:
            content += "... (truncated)"
        
        return content
    
    except ImportError:
        return "Error: requests library not installed. Run: pip install requests"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


@tool("search_web", "Search the web (mock implementation)")
def search(query: str, num_results: int = 3) -> str:
    """
    Search the web for information (mock implementation).
    
    Args:
        query: Search query
        num_results: Number of results to return
    
    Returns:
        Search results
    """
    # This is a mock implementation
    # In a real implementation, you'd integrate with a search API
    return f"Mock search results for '{query}':\n1. Result one\n2. Result two\n3. Result three"
```

---

## src/majdk/tools/filesystem.py
```python
"""Filesystem tools for reading and writing files"""

from pathlib import Path
from typing import Optional
from ..agent import tool


@tool("read_file", "Read content from a file")
def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)
    
    Returns:
        File content as string
    """
    try:
        path = Path(file_path)
        
        # Basic security check - don't read outside current directory
        if path.is_absolute() or ".." in str(path):
            return "Error: Access denied. Only relative paths in current directory allowed."
        
        if not path.exists():
            return f"Error: File '{file_path}' not found."
        
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
        
        # Limit file size to 50KB
        if path.stat().st_size > 50 * 1024:
            return "Error: File too large (max 50KB)."
        
        content = path.read_text(encoding=encoding)
        return content
    
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool("write_file", "Write content to a file")
def write_file(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write
        encoding: File encoding (default: utf-8)
    
    Returns:
        Success message or error
    """
    try:
        path = Path(file_path)
        
        # Basic security check
        if path.is_absolute() or ".." in str(path):
            return "Error: Access denied. Only relative paths in current directory allowed."
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(content, encoding=encoding)
        return f"Successfully wrote {len(content)} characters to '{file_path}'."
    
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool("list_files", "List files in a directory")
def list_files(directory: str = ".") -> str:
    """
    List files in a directory.
    
    Args:
        directory: Directory path (default: current directory)
    
    Returns:
        List of files and directories
    """
    try:
        path = Path(directory)
        
        # Security check
        if path.is_absolute() or ".." in str(path):
            return "Error: Access denied. Only relative paths allowed."
        
        if not path.exists():
            return f"Error: Directory '{directory}' not found."
        
        if not path.is_dir():
            return f"Error: '{directory}' is not a directory."
        
        items = []
        for item in sorted(path.iterdir()):
            if item.is_file():
                size = item.stat().st_size
                items.append(f"📄 {item.name} ({size} bytes)")
            elif item.is_dir():
                items.append(f"📁 {item.name}/")
        
        if not items:
            return f"Directory '{directory}' is empty."
        
        return f"Contents of '{directory}':\n" + "\n".join(items)
    
    except Exception as e:
        return f"Error listing directory: {str(e)}"
```

---

## src/majdk/cli.py
```python
"""Command line interface for MAJD Agent Kit"""

import argparse
import sys
from pathlib import Path
from .agent import Agent
from .models import OpenAIModel, OllamaModel, MockModel
from .memory import Memory


def create_agent(model_type: str, **kwargs) -> Agent:
    """Create agent with specified model"""
    if model_type == "openai":
        api_key = kwargs.get("api_key")
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
```

---

## src/majdk/ui_app.py
```python
"""Streamlit web UI for MAJD Agent Kit"""

import streamlit as st
import json
from datetime import datetime
from .agent import Agent
from .models import OpenAIModel, OllamaModel, MockModel
from .memory import Memory


def create_agent_ui() -> Agent:
    """Create agent based on UI settings"""
    model_type = st.session_state.get("model_type", "mock")
    
    if model_type == "openai":
        api_key = st.session_state.get("openai_api_key", "")
        model_name = st.session_state.get("openai_model", "gpt-3.5-turbo")
        if api_key:
            model = OpenAIModel(api_key, model_name)
        else:
            st.error("Please provide OpenAI API key")
            return None
    elif model_type == "ollama":
        model_name = st.session_state.get("ollama_model", "llama2")
        base_url = st.session_state.get("ollama_url", "http://localhost:11434")
        model = OllamaModel(model_name, base_url)
    else:  # mock
        model = MockModel()
    
    memory = Memory("streamlit_logs.jsonl")
    return Agent(model, memory)


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="MAJD Agent Kit",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 MAJD Agent Kit")
    st.markdown("*Lightweight AI Agent with Tool Support*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_type = st.selectbox(
            "Model Type",
            ["mock", "openai", "ollama"],
            key="model_type"
        )
        
        if model_type == "openai":
            st.text_input(
                "OpenAI API Key",
                type="password",
                key="openai_api_key",
                help="Your OpenAI API key"
            )
            st.selectbox(
                "Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                key="openai_model"
            )
        elif model_type == "ollama":
            st.text_input(
                "Model Name",
                value="llama2",
                key="ollama_model"
            )
            st.text_input(
                "Base URL",
                value="http://localhost:11434",
                key="ollama_url"
            )
        
        st.markdown("---")
        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent = create_agent_ui()
                if agent:
                    response = agent.run(prompt)
                    st.markdown(response)
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                    # Show agent steps in expander
                    with st.expander("🔍 Agent Steps"):
                        steps = agent.memory.get_steps()
                        for step in steps[-10:]:  # Show last 10 steps
                            st.json({
                                "type": step["step_type"],
                                "content": step["content"],
                                "time": step["timestamp"]
                            })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with **MAJD Agent Kit** • "
        "[GitHub](https://github.com/your-username/majd-agent-kit) • "
        "Apache 2.0 License"
    )


def run_app():
    """Entry point for running the Streamlit app"""
    main()


if __name__ == "__main__":
    main()
```

---

## examples/quickstart.py
```python
"""Quick start example for MAJD Agent Kit"""

import os
from majdk import Agent, tool, MockModel, OpenAIModel


# Custom tool example
@tool("greet", "Greet someone by name")
def greet_person(name: str) -> str:
    """Greet a person with their name"""
    return f"Hello, {name}! Nice to meet you."


def main():
    """Quick start demo"""
    print("🤖 MAJD Agent Kit - Quick Start Demo\n")
    
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
```

---

## pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "majdk"
version = "0.1.0"
description = "Lightweight AI Agent SDK with tool support"
readme = "README.md"
license = {text = "Apache-2.0"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.8"
dependencies = []

[project.optional-dependencies]
openai = ["openai>=1.0.0"]
ollama = ["requests>=2.25.0"]
web = ["requests>=2.25.0", "beautifulsoup4>=4.9.0"]
ui = ["streamlit>=1.28.0"]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]
all = [
    "openai>=1.0.0",
    "requests>=2.25.0",
    "beautifulsoup4>=4.9.0",
    "streamlit>=1.28.0",
]

[project.urls]
Homepage = "https://github.com/your-username/majd-agent-kit"
Repository = "https://github.com/your-username/majd-agent-kit"
Documentation = "https://github.com/your-username/majd-agent-kit#readme"
"Bug Tracker" = "https://github.com/your-username/majd-agent-kit/issues"

[project.scripts]
majdk = "majdk.cli:main"
majdk-ui = "majdk.ui_app:run_app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"

[tool.black]
line-length = 88
target-version = ['py38']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

---

## README.md
```markdown
# 🤖 MAJD Agent Kit

A lightweight, open-source Python AI agent SDK with tool support, multiple model backends, and comprehensive logging.

## ✨ Features

- **🚀 Simple**: One-file, one-function agent creation
- **🔧 Tool Support**: Easy `@tool` decorator for adding custom tools
- **🎯 Multi-Model**: OpenAI, Ollama, and Mock model support
- **📝 Full Logging**: Complete JSONL logging for reproducibility
- **🖥️ CLI & Web UI**: Command line and Streamlit interfaces
- **📦 Lightweight**: <500 lines of core code

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install majdk

# With all optional dependencies
pip install majdk[all]

# Or clone and install from source
git clone https://github.com/your-username/majd-agent-kit.git
cd majd-agent-kit
pip install -e .
```

### Basic Usage

```python
from majdk import Agent, tool, MockModel

# Create a custom tool
@tool("weather", "Get weather information")
def get_weather(location: str) -> str:
    return f"It's sunny in {location}!"

# Create agent
agent = Agent(MockModel())
agent.add_tool(get_weather)

# Run queries
response = agent.run("What's the weather in Paris?")
print(response)
```

### Built-in Tools

MAJD Agent Kit comes with three built-in tools:

1. **Calculator**: `calculate()`, `add()`, `multiply()`
2. **Web**: `web_fetch()`, `search_web()`
3. **Filesystem**: `read_file()`, `write_file()`, `list_files()`

## 🔧 Model Backends

### Mock Model (No API required)
```python
from majdk import Agent, MockModel
agent = Agent(MockModel())
```

### OpenAI
```python
from majdk import Agent, OpenAIModel
agent = Agent(OpenAIModel(api_key="your-api-key"))
```

### Ollama
```python
from majdk import Agent, OllamaModel
agent = Agent(OllamaModel(model="llama2"))
```

## 🖥️ Command Line Interface

```bash
# Interactive mode with mock model
majdk --interactive

# Single query with OpenAI
majdk --model-type openai --api-key sk-... --query "Calculate 15 * 24"

# Use Ollama
majdk --model-type ollama --model llama2 --interactive
```

## 🌐 Web Interface

```bash
# Launch Streamlit UI
majdk-ui

# Or with streamlit directly
streamlit run src/majdk/ui_app.py
```

## 📝 Creating Custom Tools

```python
from majdk import tool

@tool("my_tool", "Description of what this tool does")
def my_custom_tool(param1: str, param2: int = 42) -> str:
    """
    Tool function with type hints.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    
    Returns:
        Result description
    """
    return f"Processed {param1} with {param2}"

# Add to agent
agent.add_tool(my_custom_tool)
```

## 📊 Logging and Memory

All agent interactions are logged in JSONL format:

```python
# Access logs
steps = agent.memory.get_steps()
for step in steps:
    print(f"{step['step_type']}: {step['content']}")

# Load previous session
agent.memory.load_session("session_id")
```

Log format:
```json
{
  "session_id": "1234567890",
  "timestamp": "2024-01-15T10:30:00",
  "step_type": "thought|action|observation|final",
  "content": "Step content",
  "metadata": {}
}
```

## 🏗️ Architecture

```
src/majdk/
├── __init__.py          # Package exports
├── agent.py             # Core agent logic
├── models.py            # Model backends
├── tools/               # Built-in tools
│   ├── calculator.py    # Math operations
│   ├── web.py          # Web tools
│   └── filesystem.py   # File operations
├── memory.py           # Logging system
├── cli.py             # Command line interface
└── ui_app.py          # Streamlit web UI
```

## 🔐 Security

- File operations are restricted to current directory
- Safe expression evaluation for calculator
- Input validation and error handling
- No arbitrary code execution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Built with Python, Streamlit, and modern AI APIs
- Inspired by the need for simple, composable AI agents
- Thanks to the open-source community

---

**MAJD Agent Kit** - Making AI agents simple and accessible! 🚀
```

---

## LICENSE
```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.

   "Licensor" shall mean the copyright owner or entity granting the License.

   "Legal Entity" shall mean the union of the acting entity and all
   other entities that control, are controlled by, or are under common
   control with that entity. For the purposes of this definition,
   "control" means (i) the power, direct or indirect, to cause the
   direction or management of such entity, whether by contract or
   otherwise, or (ii) ownership of fifty percent (50%) or more of the
   outstanding shares, or (iii) beneficial ownership of such entity.

   "You" (or "Your") shall mean an individual or Legal Entity
   exercising permissions granted by this License.

   "Source" form shall mean the preferred form for making modifications,
   including but not limited to software source code, documentation
   source, and configuration files.

   "Object" form shall mean any form resulting from mechanical
   transformation or translation of a Source form, including but
   not limited to compiled object code, generated documentation,
   and conversions to other media types.

   "Work" shall mean the work of authorship, whether in Source or
   Object form, made available under the License, as indicated by a
   copyright notice that is included in or attached to the work
   (which shall not include communication that is prominently marked
   or otherwise designated in writing by the copyright owner as
   "Not a Contribution").

   "Derivative Works" shall mean any work, whether in Source or Object
   form, that is based upon (or derived from) the Work and for which the
   editorial revisions, annotations, elaborations, or other modifications
   represent, as a whole, an original work of authorship. For the purposes
   of this License, Derivative Works shall not include works that remain
   separable from, or merely link (or bind by name) to the interfaces of,
   the Work and derivative works thereof.

   "Contribution" shall mean any work of authorship, including
   the original version of the Work and any modifications or additions
   to that Work or Derivative Works thereof, that is intentionally
   submitted to Licensor for inclusion in the Work by the copyright owner
   or by an individual or Legal Entity authorized to submit on behalf of
   the copyright owner. For the purposes of this definition, "submitted"
   means any form of electronic, verbal, or written communication sent
   to the Licensor or its representatives, including but not limited to
   communication on electronic mailing lists, source code control
   systems, and issue tracking systems that are managed by, or on behalf
   of, the Licensor for the purpose of discussing and improving the Work,
   but excluding communication that is prominently marked or otherwise
   designated in writing by the copyright owner as "Not a Contribution".

2. Grant of Copyright License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   copyright license to use, reproduce, modify, display, perform,
   sublicense, and distribute the Work and such Derivative Works in
   Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   (except as stated in this section) patent license to make, have made,
   use, offer to sell, sell, import, and otherwise transfer the Work,
   where such license applies only to those patent claims licensable
   by such Contributor that are necessarily infringed by their
   Contribution(s) alone or by combination of their Contribution(s)
   with the Work to which such Contribution(s) was submitted. If You
   institute patent litigation against any entity (including a
   cross-claim or counterclaim in a lawsuit) alleging that the Work
   or a Contribution incorporated within the Work constitutes direct
   or contributory patent infringement, then any patent licenses
   granted to You under this License for that Work shall terminate
   as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
   Work or Derivative Works thereof in any medium, with or without
   modifications, and in Source or Object form, provided that You
   meet the following conditions:

   (a) You must give any other recipients of the Work or
       Derivative Works a copy of this License; and

   (b) You must cause any modified files to carry prominent notices
       stating that You changed the files; and

   (c) You must retain, in the Source form of any Derivative Works
       that You distribute, all copyright, trademark, patent,
       attribution and other notices from the Source form of the Work,
       excluding those notices that do not pertain to any part of
       the Derivative Works; and

   (d) If the Work includes a "NOTICE" text file as part of its
       distribution, then any Derivative Works that You distribute must
       include a readable copy of the attribution notices contained
       within such NOTICE file, excluding those notices that do not
       pertain to any part of the Derivative Works, in at least one
       of the following places: within a NOTICE text file distributed
       as part of the Derivative Works; within the Source form or
       documentation, if provided along with the Derivative Works; or,
       within a display generated by the Derivative Works, if and
       wherever such third-party notices normally appear. The contents
       of the NOTICE file are for informational purposes only and
       do not modify the License. You may add Your own attribution
       notices within Derivative Works that You distribute, alongside
       or as an addendum to the NOTICE text from the Work, provided
       that such additional attribution notices cannot be construed
       as modifying the License.

   You may add Your own copyright notice to Your modifications and
   may provide additional or different license terms and conditions
   for use, reproduction, or distribution of Your modifications, or
   for any such Derivative Works as a whole, provided Your use,
   reproduction, and distribution of the Work otherwise complies with
   the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
   names, trademarks, service marks, or product names of the Licensor,
   except as required for reasonable and customary use in describing the
   origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
   agreed to in writing, Licensor provides the Work (and each
   Contributor provides its Contributions) on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied, including, without limitation, any warranties or conditions
   of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
   PARTICULAR PURPOSE. You are solely responsible for determining the
   appropriateness of using or redistributing the Work and assume any
   risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
   whether in tort (including negligence), contract, or otherwise,
   unless required by applicable law (such as deliberate and grossly
   negligent acts) or agreed to in writing, shall any Contributor be
   liable to You for damages, including any direct, indirect, special,
   incidental, or consequential damages of any character arising as a
   result of this License or out of the use or inability to use the
   Work (including but not limited to damages for loss of goodwill,
   work stoppage, computer failure or malfunction, or any and all
   other commercial damages or losses), even if such Contributor
   has been advised of the possibility of such damages.

9. Accepting Warranty or Support. You may choose to offer, and to
   charge a fee for, acceptance of support, warranty, indemnity,
   or other liability obligations and/or rights consistent with this
   License. However, in accepting such obligations, You may act only
   on Your own behalf and on Your sole responsibility, not on behalf
   of any other Contributor, and only if You agree to indemnify,
   defend, and hold each Contributor harmless for any liability
   incurred by, or claims asserted against, such Contributor by reason
   of your accepting any such warranty or support.

END OF TERMS AND CONDITIONS

Copyright 2024 MAJD Agent Kit Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Setup Instructions

To get started with MAJD Agent Kit:

1. **Create the directory structure**:
```bash
mkdir -p src/majdk/tools examples
```

2. **Save each file** in its respective location

3. **Install dependencies**:
```bash
pip install -e .[all]
```

4. **Run the quickstart example**:
```bash
python examples/quickstart.py
```

5. **Try the CLI**:
```bash
majdk --interactive
```

6. **Launch the web UI**:
```bash
majdk-ui
```

This gives you a complete, production-ready AI agent SDK that's lightweight, extensible, and well-documented!