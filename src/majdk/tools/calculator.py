"""Calculator tool for mathematical operations"""

import math
import operator
from typing import Union
from ..agent import tool


@tool("calculator", "Perform mathematical calculations safely")
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
