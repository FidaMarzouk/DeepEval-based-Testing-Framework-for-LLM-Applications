"""Agent evaluation helpers: expected tools, JSON tool calls, guards, metric choice."""

import json
from types import SimpleNamespace

from deepeval.test_case import LLMTestCase, ToolCall

from elyaeval.core.agentic import (
    RepeatedToolCallMetric,
    ToolCallBudgetMetric,
    agent_action_metrics_for_golden,
    expected_tools_from_golden,
    guard_metrics,
    has_expected_tools,
    tool_calls_from_json,
)


def test_tool_calls_from_json_accepts_common_shapes():
    calls = tool_calls_from_json([
        {"name": "a", "input_parameters": {"x": 1}, "output": 2},
        {"name": "b", "args": {"y": 2}},
        {"name": "c", "arguments": json.dumps({"z": 3})},
    ])
    assert [(c.name, c.input_parameters) for c in calls] == [("a", {"x": 1}), ("b", {"y": 2}), ("c", {"z": 3})]
    assert tool_calls_from_json(None) == []


def test_expected_tools_semantics():
    stated_none = SimpleNamespace(expected_tools=None, additional_metadata={"expected_tools": []})
    stated_one = SimpleNamespace(expected_tools=[ToolCall(name="calc")], additional_metadata={})
    unstated = SimpleNamespace(expected_tools=None, additional_metadata={})
    assert has_expected_tools(stated_none) and expected_tools_from_golden(stated_none) == []
    assert has_expected_tools(stated_one) and expected_tools_from_golden(stated_one)[0].name == "calc"
    assert not has_expected_tools(unstated)


def _case(*calls):
    return LLMTestCase(input="q", actual_output="a", tools_called=[ToolCall(name=n, input_parameters=p) for n, p in calls])


def test_budget_guard_from_tools_called():
    m = ToolCallBudgetMetric(max_tool_calls=2)
    assert m.measure(_case(("a", {}), ("b", {}))) == 1.0 and m.is_successful()
    assert m.measure(_case(("a", {}), ("b", {}), ("c", {}), ("d", {}))) == 0.5 and not m.is_successful()


def test_repeat_guard_detects_identical_calls_only():
    m = RepeatedToolCallMetric()
    assert m.measure(_case(("a", {"x": 1}), ("a", {"x": 2}))) == 1.0
    assert m.measure(_case(("a", {"x": 1}), ("a", {"x": 1}))) == 0.0 and "a" in m.reason


def test_guards_read_the_trace_when_present():
    case = _case()
    case._trace_dict = {
        "type": "agent", "name": "run_agent", "children": [
            {"type": "llm", "name": "call_model", "children": []},
            {"type": "tool", "name": "lookup", "input": {"q": "x"}, "children": []},
            {"type": "tool", "name": "lookup", "input": {"q": "x"}, "children": []},
        ],
    }
    assert RepeatedToolCallMetric().measure(case) == 0.0
    budget = ToolCallBudgetMetric(max_tool_calls=1)
    budget.measure(case)
    assert budget.reason == "2 tool call(s); budget 1"


def test_guard_budget_from_golden():
    golden = SimpleNamespace(additional_metadata={"max_tool_calls": 3})
    assert guard_metrics(golden)[0].max_tool_calls == 3


def test_action_metric_choice(fake_judge):
    names = lambda ms: [m.__name__ for m in ms]  # noqa: E731
    expects = SimpleNamespace(expected_tools=[ToolCall(name="calc")], additional_metadata={})
    no_tools = SimpleNamespace(expected_tools=[], additional_metadata={})
    unstated = SimpleNamespace(expected_tools=None, additional_metadata={})
    assert names(agent_action_metrics_for_golden(expects, model=fake_judge)) == [
        "Tool Correctness", "Argument Correctness", "Tool Call Budget", "Repeated Tool Calls"]
    assert names(agent_action_metrics_for_golden(no_tools, model=fake_judge)) == [
        "Tool Correctness", "Tool Call Budget", "Repeated Tool Calls"]
    assert names(agent_action_metrics_for_golden(unstated, model=fake_judge)) == ["Tool Call Budget", "Repeated Tool Calls"]


def test_zero_tool_budget_is_respected():
    """max_tool_calls: 0 means 'no tool allowed'   it must not fall back to the default."""
    budget = guard_metrics(SimpleNamespace(additional_metadata={"max_tool_calls": 0}))[0]
    assert budget.max_tool_calls == 0
    assert budget.measure(_case()) == 1.0 and budget.is_successful()
    assert budget.measure(_case(("a", {}))) == 0.0 and not budget.is_successful()
