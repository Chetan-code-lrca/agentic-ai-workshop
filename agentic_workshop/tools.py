"""Pure example tools suitable for deterministic workshop tests."""
from __future__ import annotations
from typing import Any
from agentic_workshop.core import Tool, ToolRegistry

def _number(arguments: dict[str, Any], name: str) -> int | float:
    value = arguments.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return value

def add(arguments: dict[str, Any]) -> int | float:
    return _number(arguments, "left") + _number(arguments, "right")

def multiply(arguments: dict[str, Any]) -> int | float:
    return _number(arguments, "left") * _number(arguments, "right")

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool("add", "Add two numbers", add))
    registry.register(Tool("multiply", "Multiply two numbers", multiply))
    return registry
