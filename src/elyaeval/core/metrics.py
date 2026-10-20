"""
Metric presets, one per use case, built by FACTORIES rather than held as
module-level instances:

    from elyaeval.core.metrics import rag_metrics
    evaluate_golden(golden, test_case, rag_metrics(), csv_path=REPORT_CSV)

Why factories:
  - DeepEval metric objects are stateful (score/reason/cost are stored on
    the instance after measure()), so one shared instance reused across
    goldens/spans can leak results between them. A factory call per test
    gives every golden fresh instances.
  - Nothing is constructed at import time, so importing this module never
    requires a judge to be configured.
  - The judge model comes from elyaeval.core.judge.get_judge() — provider-
    agnostic (Ollama, Anthropic, OpenAI, Azure, Gemini, Bedrock, any
    OpenAI-compatible server, or your own class), chosen by config.

Thresholds: 0.7 unless overridden in [tool.elyaeval.thresholds] (keys are
the metric keys in METRIC_BUILDERS, plus `default`), or per call:

    rag_metrics(thresholds={"faithfulness": 0.8})

The old constant names (RAG_METRICS, RETRIEVAL_METRICS, ...) still resolve,
to a freshly built list, with a DeprecationWarning.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any, Optional

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ArgumentCorrectnessMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    SummarizationMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import SingleTurnParams

from .config import config_section
from .judge import get_judge

DEFAULT_THRESHOLD = 0.7


def _correctness(**kwargs: Any) -> GEval:
    return GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct and "
        "consistent with the expected output. Minor differences in "
        "phrasing are acceptable; differences in facts are not.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        **kwargs,
    )


def _abstention(**kwargs: Any) -> GEval:
    return GEval(
        name="Abstention",
        criteria="The question cannot be answered from the retrieval context. Determine whether the "
        "actual output clearly says the information is not available (or declines), WITHOUT "
        "inventing an answer or stating facts that are not in the retrieval context. Briefly "
        "mentioning what the available information does cover is acceptable.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        **kwargs,
    )


# metric key -> constructor. Keys are what [tool.elyaeval.thresholds] uses.
METRIC_BUILDERS: dict[str, Callable[..., BaseMetric]] = {
    "answer_relevancy": AnswerRelevancyMetric,
    "faithfulness": FaithfulnessMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_recall": ContextualRecallMetric,
    "contextual_relevancy": ContextualRelevancyMetric,
    "summarization": SummarizationMetric,
    "correctness": _correctness,
    "abstention": _abstention,
    "tool_correctness": ToolCorrectnessMetric,
    "argument_correctness": ArgumentCorrectnessMetric,
    "task_completion": TaskCompletionMetric,
    "step_efficiency": StepEfficiencyMetric,
    "plan_quality": PlanQualityMetric,
    "plan_adherence": PlanAdherenceMetric,
}

PRESETS: dict[str, tuple[str, ...]] = {
    # End-to-end RAG: generation quality + retriever quality. All five read
    # retrieval_context, which callers populate at test time.
    "rag_qa": (
        "answer_relevancy",
        "faithfulness",
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
    ),
    # Retriever in isolation: input + retrieval_context (+ expected_output).
    "retrieval": ("contextual_precision", "contextual_recall", "contextual_relevancy"),
    # Generator in isolation, given whatever retrieval_context it was handed.
    "generation": ("answer_relevancy", "faithfulness"),
    "summarization": ("summarization",),
    # Goldens the app SHOULD NOT answer (expected_behavior = "abstain"): scored
    # on declining honestly; answer-relevancy-style metrics would punish it.
    "abstention": ("abstention",),
    # Plain QA (no retrieval leg) scored against expected_output via G-Eval.
    "qa_correctness": ("correctness",),
    # Agentic action layer: one LLMTestCase with tools_called (+ expected_tools).
    "agentic_action": ("tool_correctness", "argument_correctness"),
    # Agentic trajectory layer: TRACE-level metrics — need a full @observe
    # trace via evals_iterator (TracedRunner), not evaluate_golden().
    "agentic_trajectory": ("task_completion", "step_efficiency", "plan_quality", "plan_adherence"),
}

_UNSET: Any = object()


def _threshold_for(key: str, overrides: Optional[dict[str, float]]) -> float:
    """Per-call overrides > [tool.elyaeval.thresholds] > DEFAULT_THRESHOLD,
    with a `default` key at either level applying to unlisted metrics."""
    merged = {**config_section("thresholds"), **(overrides or {})}
    return float(merged.get(key, merged.get("default", DEFAULT_THRESHOLD)))


def _accepts(builder: Callable[..., Any], name: str) -> bool:
    params = inspect.signature(builder).parameters
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def build_metric(
    key: str,
    *,
    model: Any = _UNSET,
    threshold: Optional[float] = None,
    thresholds: Optional[dict[str, float]] = None,
    **kwargs: Any,
) -> BaseMetric:
    """One fresh metric instance. `model` defaults to the configured judge
    (get_judge()); pass a DeepEvalBaseLLM or a model-name string to override,
    or None to use DeepEval's own default."""
    if key not in METRIC_BUILDERS:
        raise KeyError(f"Unknown metric '{key}'. Valid: {', '.join(sorted(METRIC_BUILDERS))}")
    builder = METRIC_BUILDERS[key]
    if model is _UNSET:
        model = get_judge()
    params: dict[str, Any] = {"threshold": threshold if threshold is not None else _threshold_for(key, thresholds)}
    if model is not None and _accepts(builder, "model"):
        params["model"] = model
    params.update(kwargs)
    return builder(**params)


def metrics_for(
    preset: str,
    *,
    model: Any = _UNSET,
    thresholds: Optional[dict[str, float]] = None,
) -> list[BaseMetric]:
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset '{preset}'. Valid: {', '.join(PRESETS)}")
    return [build_metric(k, model=model, thresholds=thresholds) for k in PRESETS[preset]]


def rag_metrics(**kwargs: Any) -> list[BaseMetric]:
    return metrics_for("rag_qa", **kwargs)


def retrieval_metrics(**kwargs: Any) -> list[BaseMetric]:
    return metrics_for("retrieval", **kwargs)


def generation_metrics(**kwargs: Any) -> list[BaseMetric]:
    return metrics_for("generation", **kwargs)


def summarization_metrics(**kwargs: Any) -> list[BaseMetric]:
    return metrics_for("summarization", **kwargs)


def qa_correctness_metrics(**kwargs: Any) -> list[BaseMetric]:
    return metrics_for("qa_correctness", **kwargs)


def abstention_metrics(**kwargs: Any) -> list[BaseMetric]:
    return metrics_for("abstention", **kwargs)


def metrics_for_golden(golden: Any, default: Callable[..., list[BaseMetric]], **kwargs: Any) -> list[BaseMetric]:
    """The golden's metrics: abstention_metrics() for goldens marked
    additional_metadata.expected_behavior = "abstain" (questions the app
    should decline), else default() (e.g. rag_metrics)."""
    meta = getattr(golden, "additional_metadata", None) or {}
    if meta.get("expected_behavior") == "abstain":
        return abstention_metrics(**kwargs)
    return default(**kwargs)


def agentic_action_metrics(**kwargs: Any) -> list[BaseMetric]:
    return metrics_for("agentic_action", **kwargs)


def agentic_trajectory_metrics(**kwargs: Any) -> list[BaseMetric]:
    return metrics_for("agentic_trajectory", **kwargs)


# task_type -> factory name, used by `elyaeval init` to fill in templates
TASK_TYPE_FACTORIES = {
    "rag_qa": "rag_metrics",
    "retrieval": "retrieval_metrics",
    "summarization": "summarization_metrics",
    "qa_correctness": "qa_correctness_metrics",
    "agentic_action": "agentic_action_metrics",
    "agentic_trajectory": "agentic_trajectory_metrics",
}

_DEPRECATED_CONSTANTS = {
    "RAG_METRICS": "rag_qa",
    "RETRIEVAL_METRICS": "retrieval",
    "GENERATION_METRICS": "generation",
    "SUMMARIZATION_METRICS": "summarization",
    "QA_CORRECTNESS_METRICS": "qa_correctness",
    "AGENTIC_ACTION_METRICS": "agentic_action",
    "AGENTIC_TRAJECTORY_METRICS": "agentic_trajectory",
}


def __getattr__(name: str) -> list[BaseMetric]:
    preset = _DEPRECATED_CONSTANTS.get(name)
    if preset is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    factory = f"{preset}_metrics" if preset != "rag_qa" else "rag_metrics"
    warnings.warn(
        f"{name} is deprecated: use elyaeval.core.metrics.{factory}() so each test gets "
        f"fresh metric instances and the configured judge.",
        DeprecationWarning,
        stacklevel=2,
    )
    return metrics_for(preset)
