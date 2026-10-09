import unittest
from agentic_workshop.core import StepLimitExceeded, Tool, ToolAgent, ToolCall, ToolValidationError, UnknownToolError
from agentic_workshop.tools import build_registry

class ToolAgentTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry()
        self.agent = ToolAgent(self.registry, max_steps=2)
    def test_explicit_plan_executes_tools_in_order(self):
        results = self.agent.run([ToolCall("add", {"left": 2, "right": 3}), ToolCall("multiply", {"left": 5, "right": 4})])
        self.assertEqual([result.value for result in results], [5.0, 20.0])
        self.assertEqual([result.tool_name for result in results], ["add", "multiply"])
    def test_unknown_tools_are_rejected(self):
        with self.assertRaises(UnknownToolError):
            self.agent.run([ToolCall("shell", {"command": "echo unsafe"})])
    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ToolValidationError):
            self.agent.run([ToolCall("add", {"left": True, "right": 1})])

    def test_non_dictionary_arguments_are_rejected(self):
        with self.assertRaisesRegex(ToolValidationError, "must be a dictionary"):
            self.agent.run([ToolCall("add", None)])

    def test_large_integer_results_remain_exact(self):
        large_integer = 2**53 + 1
        result = self.agent.run([ToolCall("add", {"left": large_integer, "right": 0})])
        self.assertEqual(result[0].value, large_integer)
        self.assertIsInstance(result[0].value, int)
    def test_plan_cannot_exceed_step_limit(self):
        plan = [ToolCall("add", {"left": 1, "right": 1})] * 3
        with self.assertRaises(StepLimitExceeded):
            self.agent.run(plan)
    def test_registry_prevents_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.registry.register(Tool("add", "duplicate", lambda arguments: 0))

if __name__ == "__main__":
    unittest.main()
