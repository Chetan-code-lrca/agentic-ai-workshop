"""Run a deterministic tool-use example with an observed intermediate result."""
from agentic_workshop.core import ToolAgent, ToolCall
from agentic_workshop.tools import build_registry

def main() -> None:
    agent = ToolAgent(build_registry(), max_steps=3)
    results = agent.run([ToolCall("add", {"left": 2, "right": 3})])
    observed_sum = results[-1].value
    # The caller observes the first result before constructing the next tool call.
    results.extend(agent.run([ToolCall("multiply", {"left": observed_sum, "right": 4})]))
    for index, result in enumerate(results, start=1):
        print(f"Step {index} — {result.tool_name}: {result.value:g}")
    print(f"Final result: {results[-1].value:g}")

if __name__ == "__main__":
    main()
