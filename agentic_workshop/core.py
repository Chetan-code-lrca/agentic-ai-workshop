"""Typed tool registry and bounded execution loop for workshop exercises."""
from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from agentic_workshop.policy import ApprovalManager, ApprovalRequired, AuditEvent, make_audit_event


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
    risk_level: str = "read"


class ToolRegistry:
    """Register named tools and execute only known entries."""

    RISK_LEVELS = {"read", "write", "destructive"}

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name.strip():
            raise ValueError("tool name must not be empty")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        if tool.risk_level not in self.RISK_LEVELS:
            raise ValueError("risk_level must be read, write, or destructive")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"unknown tool: {name}")
        return tool

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self.get(call.name)
        if not isinstance(call.arguments, dict):
            raise ToolValidationError("tool arguments must be a dictionary")
        try:
            handler_arguments = deepcopy(call.arguments)
        except Exception as exc:
            raise ToolValidationError(f"arguments for {call.name} could not be copied safely") from exc
        try:
            value = tool.handler(handler_arguments)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolValidationError(f"invalid arguments for {call.name}: {exc}") from exc
        return ToolResult(tool_name=tool.name, value=value)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


class ToolAgent:
    """Execute explicit plans; write/destructive calls require a one-time approval."""

    def __init__(
        self,
        registry: ToolRegistry,
        max_steps: int = 5,
        approval_manager: ApprovalManager | None = None,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.registry = registry
        self.max_steps = max_steps
        self.approval_manager = approval_manager
        self.audit_events: list[AuditEvent] = []

    def run(
        self,
        plan: Iterable[ToolCall],
        approvals: Mapping[int, str] | None = None,
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        trace_id = uuid.uuid4().hex
        approvals = approvals or {}
        for step, call in enumerate(plan, start=1):
            if step > self.max_steps:
                raise StepLimitExceeded(f"plan exceeded the {self.max_steps}-step limit")
            tool = self.registry.get(call.name)
            try:
                # ToolCall is frozen, but its nested argument dictionary is not.
                # Snapshot it before approval so later caller mutations cannot
                # change the operation that is ultimately executed or audited.
                approved_call = ToolCall(call.name, deepcopy(call.arguments))
            except Exception as exc:
                raise ToolValidationError("tool arguments could not be copied safely") from exc
            if tool.risk_level != "read":
                token = approvals.get(step)
                approval_call = ToolCall(approved_call.name, deepcopy(approved_call.arguments))
                approved = self.approval_manager is not None and self.approval_manager.consume(token, approval_call)
                if not approved:
                    self.audit_events.append(make_audit_event(trace_id, step, approved_call, tool.risk_level, "blocked"))
                    raise ApprovalRequired(f"tool '{approved_call.name}' requires a valid one-time approval for this exact call")
            result = self.registry.execute(approved_call)
            self.audit_events.append(make_audit_event(trace_id, step, approved_call, tool.risk_level, "executed"))
            results.append(result)
        return results
