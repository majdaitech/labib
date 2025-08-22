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

            # Construct plausible arguments from the tool schema
            props: Dict[str, Any] = tool["function"]["parameters"].get("properties", {})
            req: List[str] = tool["function"]["parameters"].get("required", [])

            args: Dict[str, Any] = {}
            for param_name, spec in props.items():
                t = spec.get("type", "string")
                # Heuristics by param name
                lname = param_name.lower()
                if lname in ("url", "uri", "link"):
                    val = "https://example.com"
                elif lname in ("expression", "expr"):
                    val = "2+2"
                elif "path" in lname:
                    val = "README.md"
                elif lname == "a":
                    val = 2
                elif lname == "b":
                    val = 3
                elif lname == "name":
                    val = "Alice"
                else:
                    # Defaults by type
                    if t == "integer":
                        val = 1
                    elif t == "number":
                        val = 1.0
                    elif t == "boolean":
                        val = True
                    else:
                        val = "test"
                args[param_name] = val

            # Mock tool call
            class MockToolCall:
                def __init__(self):
                    self.id = f"call_{random.randint(1000, 9999)}"
                    self.function = self
                    self.name = tool_name
                    self.arguments = json.dumps(args)
            
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
