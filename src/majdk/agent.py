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
            "name": getattr(func, "_tool_name", func.__name__),
            "description": getattr(func, "_tool_description", func.__doc__ or ""),
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
    
    def __init__(
        self,
        model: BaseModel,
        memory: Memory = None,
        max_iterations: int = 5,
        system_prompt: Optional[str] = None,
        load_builtin_tools: bool = True,
        tools: Optional[List[Callable]] = None,
    ):
        self.model = model
        self.memory = memory or Memory()
        self.max_iterations = max_iterations
        self.custom_system_prompt = system_prompt
        self.tools = {}
        self.external_tools: Dict[str, Dict[str, Any]] = {}
        if load_builtin_tools:
            self._load_builtin_tools()
        if tools:
            for func in tools:
                self.add_tool(func)
    
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
    
    def add_external_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register an external tool with explicit schema and a handler.
        
        parameters must follow the internal schema format used by the agent:
        {
          "type": "function",
          "function": {
            "name": name,
            "description": description,
            "parameters": { "type": "object", "properties": {...}, "required": [...] }
          }
        }
        """
        if not isinstance(parameters, dict) or parameters.get("type") != "function":
            # Build a proper schema wrapper if raw JSON Schema for params is provided
            parameters = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters if isinstance(parameters, dict) else {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        self.external_tools[name] = {"schema": parameters, "handler": handler, "description": description}
    
    def run(self, query: str, extra_messages: Optional[List[Dict[str, str]]] = None) -> str:
        """Main agent execution loop"""
        self.memory.log_step("thought", f"Starting task: {query}")
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
        ]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": query})
        
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
                elif tool_name in self.external_tools:
                    handler = self.external_tools[tool_name]["handler"]
                    try:
                        # Try kwargs; if it fails, pass dict
                        try:
                            result = handler(**tool_args)
                        except TypeError:
                            result = handler(tool_args)
                        self.memory.log_step("observation", f"Tool result: {result}")
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
        if self.custom_system_prompt:
            return self.custom_system_prompt
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
        schemas = [func._tool_schema for func in self.tools.values()]
        schemas.extend([entry["schema"] for entry in self.external_tools.values()])
        return schemas
