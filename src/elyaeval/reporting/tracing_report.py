"""
Component-level (span-aware) evaluation + CSV reporting.

Sibling to report.py NOT a rewrite of it. report.py drives DeepEval's
end-to-end evaluate() over a single LLMTestCase per golden; this module
drives DeepEval's evals_iterator() over a single-golden EvaluationDataset,
so that per-span metrics attached via @observe(metrics=...) +
update_current_span(test_case=...) inside the SUT are actually collected.
That only works for an IN-PROCESS call into the SUT.

evals_iterator() is a generator: driving it to exhaustion is what runs the
golden through the SUT and produces its EvaluationResult

For each golden, DeepEval's iterator appends MULTIPLE TestResults to that
EvaluationResult:
  - one trace-level result (name=golden.name)
  - one result per span that had metrics attached, named after that span
    (test_result.name == the @observe'd function's name, e.g. "retrieve",
    "generate_answer")
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Union

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.metrics import BaseMetric

from ..core.dataset import golden_id as _stable_golden_id
from ..core.usage import METER, GoldenUsage, reported_judge_usage
from .results_csv import TRACED_FIELDS, TRACED_GROUP_KEYS, append_rows, current_repeat, metric_rows

TRACED_CSV_FIELDS = TRACED_FIELDS


def golden_id(golden: Golden, fallback_index: Optional[int] = None) -> str:
    """Stable id of `golden` (see elyaeval.core.dataset.golden_id).
    fallback_index is deprecated and ignored: ids no longer depend on row
    order, so reordering goldens doesn't re-key every CSV row."""
    return _stable_golden_id(golden)


def append_traced_csv_rows(csv_path: str | Path, rows: list[dict]) -> None:
    """Append component rows (with span_name) and register the CSV, grouped
    by (span_name, metric_name) — the same metric scored on two different
    spans must not be averaged together."""
    append_rows(csv_path, rows, TRACED_FIELDS, TRACED_GROUP_KEYS)


TRACE_SPAN_NAME = "trace"


def traced_test_result_to_csv_rows(golden_id: str, priority: str, test_result: TestResult) -> list[dict]:
    """One row per metric; span_name is the @observe'd function's name
    (test_result.name), or "trace" for the trace-level result (trace metrics
    such as TaskCompletion). DeepEval names that result after the golden /
    pytest test, which would make every golden its own group; it is the only
    result carrying an `index` (span results have none)."""
    span_name = TRACE_SPAN_NAME if getattr(test_result, "index", None) is not None else test_result.name
    return metric_rows(golden_id, priority, test_result, span_name=span_name)


SpanMetricSource = Union[Callable[[], list[BaseMetric]], list[BaseMetric]]


def _walk_spans(spans: list) -> list:
    out = []
    for span in spans or []:
        out.append(span)
        out.extend(_walk_spans(getattr(span, "children", None) or []))
    return out


def result_passed(test_result: TestResult) -> bool:
    """Whether every (non-flaky) metric on this result passed.

    Use this instead of `test_result.success` for span results: in DeepEval
    4.2.0, extract_span_test_results() sets success from
    `span_api.status == TraceSpanApiStatus.SUCCESS`, but BaseApiSpan stores
    status as a plain string (use_enum_values=True), so the comparison is
    always False and EVERY span result reports success=False regardless of
    its scores. Errored metrics carry success=False, so they still fail here.
    A result with no metrics (e.g. an unscored trace-level result) passes.
    """
    return all(md.success for md in (test_result.metrics_data or []) if not getattr(md, "flaky", False))


def failed_results(result: EvaluationResult) -> list[str]:
    """Names of the trace/span results in `result` that have a failing
    metric — what a component test should assert is empty."""
    return [tr.name for tr in result.test_results if not result_passed(tr)]


class TracedRunner:
    """
    Drives DeepEval's evals_iterator() for a SINGLE golden, wrapped around
    an in-process call into the SUT, and exposes the resulting
    EvaluationResult once that golden has run.

    Span-level metrics: pass `span_metrics`, mapping a span name (the
    @observe'd function's name, or its `name=`) to a metric factory or list:

        TracedRunner(golden, span_metrics={
            "retrieve": retrieval_metrics,
            "generate_answer": generation_metrics,
        }).run(call_sut)

    The metrics are attached by the TEST HARNESS after the app ran, so the
    app's own code only needs plain `@observe` spans + update_current_span(
    test_case=...)   it never imports a judge or metric. With
    fill_expected_output (default), spans that get metrics and have no
    expected_output receive the golden's, so reference-based metrics
    (ContextualPrecision/Recall) work without the app ever seeing the
    reference answer. Metrics declared on the app side (@observe(metrics=...))
    still work too.

    `metrics` means TRACE-level metrics (evals_iterator(metrics=...)), judged
    against the whole trace; leave None unless the app also calls
    update_current_trace(input=..., output=...).

    error_config defaults to ignore_errors=True (mirrors evaluate_golden in
    report.py) so one golden's SUT exception doesn't abort the whole
    parametrized suite it surfaces as an errored span/trace in the
    result instead of raising out of this call.
    """

    def __init__(
        self,
        golden: Golden,
        metrics: Optional[list[BaseMetric]] = None,
        display_config: Optional[DisplayConfig] = None,
        span_metrics: Optional[dict[str, SpanMetricSource]] = None,
        fill_expected_output: bool = True,
        cache_config: Optional[CacheConfig] = None,
        error_config: Optional[ErrorConfig] = None,
    ):
        self.golden = golden
        self.metrics = metrics
        self.span_metrics = span_metrics or {}
        self.fill_expected_output = fill_expected_output
        self.display_config = display_config or DisplayConfig(show_indicator=False, print_results=False)
        # write_cache off: see the CacheConfig note in report.evaluate_golden
        self.cache_config = cache_config or CacheConfig(write_cache=False, use_cache=False)
        self.error_config = error_config or ErrorConfig(ignore_errors=True)
        self.result: Optional[EvaluationResult] = None

        self._dataset = EvaluationDataset(goldens=[golden])
        self._iterator = self._dataset.evals_iterator(
            metrics=self.metrics,
            display_config=self.display_config,
            cache_config=self.cache_config,
            error_config=self.error_config,
            async_config=AsyncConfig(run_async=False),
        )

    def __iter__(self):
        return self

    def __next__(self) -> Golden:
        try:
            return next(self._iterator)
        except StopIteration as e:
            # e.value is the generator's `return EvaluationResult(...)`
            # the only place it's available; a plain for-loop discards it.
            self.result = e.value
            raise

    def _attach_span_metrics(self, golden: Golden) -> None:
        """Runs between the app call and DeepEval's scoring of the trace (the
        iterator body still has the golden's trace as the current one)."""
        if not self.span_metrics:
            return
        from deepeval.tracing.context import current_trace_context

        trace = current_trace_context.get()
        if trace is None:
            return
        found = set()
        for span in _walk_spans(trace.root_spans):
            source = self.span_metrics.get(span.name)
            if source is None:
                continue
            found.add(span.name)
            fresh = source() if callable(source) else list(source)
            span.metrics = list(span.metrics or []) + fresh
            if self.fill_expected_output and span.expected_output is None and golden.expected_output:
                span.expected_output = golden.expected_output
        missing = set(self.span_metrics) - found
        if missing:
            import warnings

            warnings.warn(
                f"span_metrics names {sorted(missing)} matched no span of this trace "
                f"(spans seen: {sorted(s.name for s in _walk_spans(trace.root_spans))})",
                stacklevel=3,
            )

    def run(self, call_sut: Callable[[Golden], None]) -> EvaluationResult:
        """Convenience for the common case: no extra logic needed around
        the SUT call itself   drive the single-golden iterator, call
        call_sut(golden) for its one iteration, and return the result."""
        started = time.perf_counter()
        for golden in self:
            call_sut(golden)
            self._attach_span_metrics(golden)
        assert self.result is not None, "evals_iterator exhausted without producing a result"
        # component mode runs the SUT and the span judges in one pass, so the
        # time recorded here is SUT + judging together (sut_seconds unknown)
        metrics_data = [md for tr in self.result.test_results for md in (tr.metrics_data or [])]
        judge_cost, judge_tokens = reported_judge_usage(metrics_data)
        METER.record_golden(GoldenUsage(
            golden_id=_stable_golden_id(self.golden),
            repeat=current_repeat(),
            priority=str((self.golden.additional_metadata or {}).get("priority", "")),
            n_metrics=len(metrics_data),
            judge_seconds=time.perf_counter() - started,
            judge_cost_reported=judge_cost,
            judge_tokens_reported=judge_tokens,
        ))
        return self.result
