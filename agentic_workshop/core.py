"""Typed tool registry and bounded execution loop for workshop exercises."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable

class AgentError(Exception):
    """Base class for expected agent execution errors."""
class UnknownToolError(AgentError):
    """Raised when a plan references an unregistered tool."""
class StepLimitExceeded(AgentError):
    """Raised when a plan exceeds the configured number of tool calls."""
class ToolValidationError(AgentError):
    """Raised when tool input validation fails."""

@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    value: Any
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: Callable[[dict[str, Any]], Any]

class ToolRegistry:
    """Register named tools and execute only known entries."""
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
    def register(self, tool: Tool) -> None:
        if not tool.name.strip():
            raise ValueError("tool name must not be empty")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            raise UnknownToolError(f"unknown tool: {call.name}")
        try:
            value = tool.handler(call.arguments)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolValidationError(f"invalid arguments for {call.name}: {exc}") from exc
        return ToolResult(tool_name=tool.name, value=value)
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

class ToolAgent:
    """Execute an explicit plan with a fixed upper bound on tool calls."""
    def __init__(self, registry: ToolRegistry, max_steps: int = 5) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.registry = registry
        self.max_steps = max_steps
    def run(self, plan: Iterable[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for step, call in enumerate(plan, start=1):
            if step > self.max_steps:
                raise StepLimitExceeded(f"plan exceeded the {self.max_steps}-step limit")
            results.append(self.registry.execute(call))
        return results
