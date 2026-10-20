"""
Typed return contracts for run_app(), one per task family.

Every generated test template needs run_app() to return a specific shape.
This module makes each shape a real, importable type instead,
so:
  - it's self-documenting (hover/go-to-definition instead of scrolling to
    a comment)
  - it's a single source of truth   change a field once here, not once per
    template that happens to mention it
  - mistakes (wrong order, missing field) surface immediately instead of
    failing later inside LLMTestCase construction

Add a new NamedTuple here whenever a new task family gets its own template.
"""

from typing import NamedTuple, Optional

from deepeval.test_case import ToolCall


class SUTUsage(NamedTuple):
    """What the system under test spent answering one golden — optional on
    every run result. Fill in what your app/provider reports (e.g. the
    `usage` block of an OpenAI/Anthropic/Groq response); leave the rest None.
    Cost can also be derived from tokens via [tool.elyaeval.pricing]. The
    generated templates measure latency_seconds themselves if you don't.
    Feeds budgets (elyaeval.core.budget) and the run's budget ledger."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost: Optional[float] = None
    latency_seconds: Optional[float] = None


class RAGRunResult(NamedTuple):
    """Return shape for rag_qa / retrieval / summarization / qa_correctness
    task types (elyaeval/templates/test_template*.py.tmpl)."""

    actual_output: str
    retrieval_context: list[str]
    usage: Optional[SUTUsage] = None


class AgenticRunResult(NamedTuple):
    """Return shape for agentic task types evaluated at the component
    (tool-calling) level   ToolCorrectnessMetric / ArgumentCorrectnessMetric
    (elyaeval/templates/test_template_agentic.py.tmpl).

    tools_called: the ToolCall objects your agent actually invoked while
        answering `input`, in order. Populate each with at least `name`;
        add `input_parameters` / `output` if you want ArgumentCorrectness
        or output-matching strictness to have something to check.
    available_tools: optional. Every ToolCall your agent COULD have called
        for this input   only needed if you want a metric to evaluate tool
        *selection* not just correctness of the ones it did call.
        Leave as None if you're not using that.

    Not for trajectory-level metrics (TaskCompletion, StepEfficiency,
    PlanQuality, PlanAdherence) those read a full @observe trace, not a
    single run_app() call, and need a different runner.
    """

    actual_output: str
    tools_called: list[ToolCall]
    available_tools: Optional[list[ToolCall]] = None
    usage: Optional[SUTUsage] = None
