# Agentic AI Workshop

A dependency-free demonstration of bounded tool execution: validate a planned action, call a registered tool, and keep an observable result. It is deterministic and does not pretend to be an LLM planner, so the execution and safety checks are easy to study.

## Requirements
- Python 3.10+
- No API keys or third-party packages

## Run
```bash
python -m agentic_workshop.demo
python -m unittest discover -s tests -v
```

The demo adds two numbers and multiplies the result. Tool calls are explicit and reproducible.

## Workshop tasks
1. Read `ToolCall`, `ToolResult`, and `ToolRegistry`.
2. Add a pure function to `tools.py`, register it, and test it.
3. Test valid input, invalid input, and an unknown tool.
4. Keep a future planner separate from tool execution.
5. If connecting a model later, validate proposed tool names and arguments, cap steps, and never expose secrets to tool output.

This starter demonstrates tool execution and bounded steps, not autonomous reasoning or production security.
