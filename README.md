# Agentic AI Workshop

A dependency-free demonstration of bounded tool execution with explicit risk levels and human approval gates.

## Requirements
- Python 3.10+
- No API keys or third-party runtime packages

## Run
```bash
python -m agentic_workshop.demo
python -m unittest discover -s tests -v
```

## Tool safety model

Each registered tool declares one risk level:

- `read`: runs normally.
- `write`: requires a valid approval token.
- `destructive`: requires a valid approval token.

An approval is short-lived, single-use, and bound to the exact tool name and JSON arguments. A token for one operation cannot authorize a different operation. `ToolAgent.run()` receives approvals as a mapping from the 1-based plan step to token.

```python
from agentic_workshop.core import Tool, ToolCall, ToolAgent
from agentic_workshop.policy import ApprovalManager

manager = ApprovalManager()
# Register your own write tool with risk_level="write".
# Issue approval only after a trusted UI/operator confirms the exact action:
call = ToolCall("write_note", {"text": "approved note"})
token = manager.issue(call, ttl_seconds=300)
agent = ToolAgent(registry, approval_manager=manager)
result = agent.run([call], approvals={1: token})
```

Do not ask the model to self-approve. The included examples do not register tools with external side effects.

## Audit trail

The agent records in-memory events for blocked higher-risk actions and successful execution. Events include trace ID, step, tool name, risk level, decision, timestamp, and an SHA-256 digest of arguments—not raw argument values. They are not persistent in this workshop version.

## Limits

This is an educational policy layer, not production authorization. Before production, add authenticated approver identity, persistent/tamper-evident audit storage, tenant isolation, policy versioning, replay protection across restarts, secret management, rate limits, and a threat model. Outstanding approvals are lost on restart.
