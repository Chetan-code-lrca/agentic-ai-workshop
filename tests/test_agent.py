import unittest
from agentic_workshop.core import StepLimitExceeded, Tool, ToolAgent, ToolCall, ToolValidationError, UnknownToolError
from agentic_workshop.policy import ApprovalManager, ApprovalRequired
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


class ApprovalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.manager = ApprovalManager(clock=lambda: self.now[0])
        self.registry = build_registry()
        self.registry.register(Tool("write_note", "Write a note", lambda args: args["text"], "write"))
        self.registry.register(Tool("delete_note", "Delete a note", lambda args: True, "destructive"))
        self.agent = ToolAgent(self.registry, approval_manager=self.manager)

    def test_read_tool_needs_no_approval(self):
        result = self.agent.run([ToolCall("add", {"left": 1, "right": 2})])
        self.assertEqual(result[0].value, 3)

    def test_write_tool_is_blocked_without_approval(self):
        with self.assertRaises(ApprovalRequired):
            self.agent.run([ToolCall("write_note", {"text": "hello"})])
        self.assertEqual(self.agent.audit_events[-1].decision, "blocked")

    def test_exact_approval_allows_once(self):
        call = ToolCall("write_note", {"text": "hello"})
        token = self.manager.issue(call)
        result = self.agent.run([call], approvals={1: token})
        self.assertEqual(result[0].value, "hello")
        with self.assertRaises(ApprovalRequired):
            self.agent.run([call], approvals={1: token})

    def test_approval_is_bound_to_exact_arguments_and_tool(self):
        token = self.manager.issue(ToolCall("write_note", {"text": "hello"}))
        with self.assertRaises(ApprovalRequired):
            self.agent.run([ToolCall("write_note", {"text": "different"})], approvals={1: token})

    def test_expired_approval_is_rejected(self):
        call = ToolCall("delete_note", {"id": "note-1"})
        token = self.manager.issue(call, ttl_seconds=5)
        self.now[0] += 6
        with self.assertRaises(ApprovalRequired):
            self.agent.run([call], approvals={1: token})

    def test_destructive_tools_require_approval_and_audit_is_redacted(self):
        call = ToolCall("delete_note", {"id": "private-record"})
        token = self.manager.issue(call)
        self.agent.run([call], approvals={1: token})
        event = self.agent.audit_events[-1]
        self.assertEqual(event.risk_level, "destructive")
        self.assertEqual(event.decision, "executed")
        self.assertEqual(len(event.argument_sha256), 64)
        self.assertFalse(hasattr(event, "arguments"))
        self.assertTrue(event.trace_id)

    def test_rejects_invalid_risk_and_ttl(self):
        with self.assertRaisesRegex(ValueError, "risk_level"):
            self.registry.register(Tool("weird", "bad", lambda _args: None, "unknown"))
        with self.assertRaises(ValueError):
            self.manager.issue(ToolCall("write_note", {"text": "x"}), ttl_seconds=0)


if __name__ == "__main__":
    unittest.main()
