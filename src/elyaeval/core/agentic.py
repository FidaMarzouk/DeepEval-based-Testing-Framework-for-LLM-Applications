"""
Agent evaluation helpers.

Three levels, matching how agents fail:

  action      which tools were called, with which arguments
              (ToolCorrectness vs the golden's expected tools, ArgumentCorrectness)
  trajectory  the whole run: was the task completed, efficiently, with a
              sensible plan that was followed (TaskCompletion, StepEfficiency,
              PlanQuality, PlanAdherence   trace-level, via TracedRunner)
  guards      deterministic, judge-free checks that catch runaway agents:
              too many tool calls (ToolCallBudgetMetric), the same call
              repeated (RepeatedToolCallMetric)

Goldens describe the expected behaviour (top-level fields or in
additional_metadata):

    "expected_tools": [{"name": "convert_energy",
                        "input_parameters": {"value": 250, "from_unit": "MWh", "to_unit": "GWh"}}]
    "expected_tools": []          # explicitly: the agent should call NO tool
    "max_tool_calls": 3           # optional per-golden budget for the guard

An agent served over HTTP returns its tool calls as JSON; tool_calls_from_json()
turns `[{"name", "input_parameters" | "args" | "arguments", "output"}]` into
DeepEval ToolCalls for AgenticRunResult.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from typing import Any, Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import ToolCall

DEFAULT_MAX_TOOL_CALLS = 8


def _meta(golden: Any) -> dict:
    return getattr(golden, "additional_metadata", None) or {}


def has_expected_tools(golden: Any) -> bool:
    """True if the golden states its expected tools (an empty list counts:
    it means 'call no tool')."""
    return getattr(golden, "expected_tools", None) is not None or "expected_tools" in _meta(golden)


def _to_tool_call(item: Any) -> ToolCall:
    if isinstance(item, ToolCall):
        return item
    params = item.get("input_parameters", item.get("args", item.get("arguments")))
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {"input": params}
    return ToolCall(name=item["name"], input_parameters=params, output=item.get("output"))


def expected_tools_from_golden(golden: Any) -> list[ToolCall]:
    raw = getattr(golden, "expected_tools", None)
    if raw is None:
        raw = _meta(golden).get("expected_tools") or []
    return [_to_tool_call(t) for t in raw]


def tool_calls_from_json(items: Optional[list[Any]]) -> list[ToolCall]:
    """An agent API's JSON tool calls -> DeepEval ToolCalls (order kept)."""
    return [_to_tool_call(t) for t in items or []]


# ---------------------------------------------------------------------------
# Deterministic guards (no judge, no cost)
# ---------------------------------------------------------------------------


def _tool_calls_of(test_case: Any) -> list[tuple[str, str]]:
    """(name, canonical-args) for every tool call: from the trace when the
    metric runs trace-level (TracedRunner), else from test_case.tools_called."""
    trace = getattr(test_case, "_trace_dict", None)
    if isinstance(trace, dict):
        calls: list[tuple[str, str]] = []

        def walk(span: dict) -> None:
            if str(span.get("type", "")).lower() == "tool":
                calls.append((str(span.get("name")), json.dumps(span.get("input"), sort_keys=True, default=str)))
            for child in span.get("children") or []:
                walk(child)

        walk(trace)
        return calls
    return [
        (tc.name, json.dumps(tc.input_parameters, sort_keys=True, default=str))
        for tc in (getattr(test_case, "tools_called", None) or [])
    ]


class _DeterministicMetric(BaseMetric):
    # free to recompute: never served from elyaeval's result cache
    elyaeval_cacheable = False

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.evaluation_model = "deterministic"
        self.evaluation_cost = None
        self.include_reason = True
        self.async_mode = False
        self.strict_mode = False
        self.verbose_mode = False
        self.requires_trace = True  # gets the trace in TracedRunner; falls back to tools_called

    async def a_measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


class ToolCallBudgetMetric(_DeterministicMetric):
    """Fails when the agent made more than `max_tool_calls` tool calls.
    Score: 1.0 within budget, else max_tool_calls / calls."""

    def __init__(self, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS):
        super().__init__(threshold=1.0)
        self.max_tool_calls = max_tool_calls

    def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        n = len(_tool_calls_of(test_case))
        self.score = 1.0 if n <= self.max_tool_calls else self.max_tool_calls / n  # n > budget >= 0, so n > 0
        self.success = n <= self.max_tool_calls
        self.reason = f"{n} tool call(s); budget {self.max_tool_calls}"
        return self.score

    @property
    def __name__(self) -> str:
        return "Tool Call Budget"


class RepeatedToolCallMetric(_DeterministicMetric):
    """Fails when the same tool is called with the same arguments more than
    `max_identical` times   the usual signature of an agent stuck in a loop."""

    def __init__(self, max_identical: int = 1):
        super().__init__(threshold=1.0)
        self.max_identical = max_identical

    def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        counts = Counter(_tool_calls_of(test_case))
        repeated = {name: n for (name, _), n in counts.items() if n > self.max_identical}
        self.score = 0.0 if repeated else 1.0
        self.success = not repeated
        self.reason = (
            f"identical call(s) repeated: {repeated}" if repeated else "no identical tool call repeated"
        )
        return self.score

    @property
    def __name__(self) -> str:
        return "Repeated Tool Calls"


def guard_metrics(golden: Any = None, max_tool_calls: Optional[int] = None) -> list[BaseMetric]:
    """Judge-free agent guards; the golden's `max_tool_calls` (top-level or
    metadata) overrides the default budget."""
    budget = max_tool_calls
    if budget is None:
        budget = _meta(golden).get("max_tool_calls")
    if budget is None:  # explicit 0 ("no tool calls allowed") must survive
        budget = DEFAULT_MAX_TOOL_CALLS
    return [ToolCallBudgetMetric(int(budget)), RepeatedToolCallMetric()]


def agent_action_metrics_for_golden(
    golden: Any,
    action_metrics: Optional[Callable[..., list[BaseMetric]]] = None,
    **kwargs: Any,
) -> list[BaseMetric]:
    """Action-level metrics for one golden: ToolCorrectness only when the
    golden states its expected tools (scoring against a missing list would
    be meaningless), ArgumentCorrectness when tools are expected, plus the
    deterministic guards."""
    from .metrics import build_metric

    metrics: list[BaseMetric] = []
    expected = expected_tools_from_golden(golden)
    if has_expected_tools(golden):
        metrics.append(build_metric("tool_correctness", **kwargs))
    if expected:
        metrics.append(build_metric("argument_correctness", **kwargs))
    if action_metrics is not None:
        metrics.extend(action_metrics(**kwargs))
    return metrics + guard_metrics(golden)
