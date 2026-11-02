"""One-time, expiring approvals for higher-risk tool calls."""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


class ApprovalRequired(Exception):
    """Raised when a write/destructive tool call lacks a valid approval."""


def _fingerprint(call: Any) -> str:
    try:
        canonical = json.dumps(
            {"tool": call.name, "arguments": call.arguments},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("tool arguments must be JSON-serializable for approval") from error
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ApprovalManager:
    """Issue single-use approvals bound to an exact tool name and argument set."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._approvals: dict[str, tuple[str, float]] = {}

    def issue(self, call: Any, ttl_seconds: float = 300.0) -> str:
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise ValueError("approval TTL must be a number")
        if not 1 <= ttl_seconds <= 3600:
            raise ValueError("approval TTL must be between 1 and 3600 seconds")
        token = secrets.token_urlsafe(32)
        self._approvals[token] = (_fingerprint(call), self._clock() + float(ttl_seconds))
        return token

    def consume(self, token: str | None, call: Any) -> bool:
        if not isinstance(token, str) or not token:
            return False
        record = self._approvals.pop(token, None)
        if record is None:
            return False
        expected_fingerprint, expires_at = record
        if self._clock() > expires_at:
            return False
        return secrets.compare_digest(expected_fingerprint, _fingerprint(call))


@dataclass(frozen=True)
class AuditEvent:
    trace_id: str
    step: int
    tool_name: str
    risk_level: str
    decision: str
    argument_sha256: str
    timestamp: str


def make_audit_event(trace_id: str, step: int, call: Any, risk_level: str, decision: str) -> AuditEvent:
    return AuditEvent(
        trace_id=trace_id,
        step=step,
        tool_name=call.name,
        risk_level=risk_level,
        decision=decision,
        argument_sha256=_fingerprint(call),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    )
