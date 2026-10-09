"""Run a deterministic two-step tool-use example."""
from agentic_workshop.core import ToolAgent, ToolCall
from agentic_workshop.tools import build_registry

def main() -> None:
    agent = ToolAgent(build_registry(), max_steps=3)
    plan = [ToolCall("add", {"left": 2, "right": 3}), ToolCall("multiply", {"left": 5, "right": 4})]
    results = agent.run(plan)
    for index, result in enumerate(results, start=1):
        print(f"Step {index} — {result.tool_name}: {result.value:g}")
    print(f"Final result: {results[-1].value:g}")

if __name__ == "__main__":
    main()
