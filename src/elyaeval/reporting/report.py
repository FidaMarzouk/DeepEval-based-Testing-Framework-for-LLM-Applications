"""
Per-golden evaluation + CSV reporting.

Replaces assert_test() (junit.xml only, no structured scores) with
evaluate(), called ONCE PER GOLDEN never batched across the whole
GOLDENS list, since a batched call would still gate correctly but every
CSV row would belong to one opaque run with no way to trace which golden
produced which score.

evaluate_golden() is a drop-in replacement for assert_test() inside a
parametrized pytest test: same one-test-per-golden shape, but returns a
structured TestResult (metrics_data: list[MetricData], each with
name/score/threshold/success/reason) instead of raising an AssertionError.
That result becomes the CSV row(s) one per (golden, metric) and is
also what the caller asserts on, so junit.xml keeps working unchanged.
"""

import time
from pathlib import Path
from typing import Optional

from deepeval.dataset import Golden
from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.types import TestResult
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run.api import MetricData

from ..core.cache import cache_key, get_result_cache, is_cacheable, judge_identity
from ..core.dataset import golden_id as _stable_golden_id
from ..core.run_contracts import SUTUsage
from ..core.usage import METER, GoldenUsage, reported_judge_usage
from .results_csv import E2E_FIELDS, E2E_GROUP_KEYS, append_rows, current_repeat, metric_rows, new_report_path

CSV_FIELDS = E2E_FIELDS

__all__ = [
    "CSV_FIELDS",
    "append_csv_rows",
    "evaluate_golden",
    "new_report_path",
    "test_result_to_csv_rows",
]


def append_csv_rows(csv_path: str | Path, rows: list[dict]) -> None:
    """Append e2e rows to csv_path (header on first write) and register it
    for HTML rendering at session end. See results_csv.append_rows."""
    append_rows(csv_path, rows, E2E_FIELDS, E2E_GROUP_KEYS)


def test_result_to_csv_rows(golden_id: str, priority: str, test_result: TestResult) -> list[dict]:
    """One row per metric in test_result.metrics_data."""
    return metric_rows(golden_id, priority, test_result)


test_result_to_csv_rows.__test__ = False  # type: ignore[attr-defined]  # not a pytest test


def evaluate_golden(
    golden: Golden,
    test_case: LLMTestCase,
    metrics: list[BaseMetric],
    csv_path: str | Path,
    golden_id: Optional[str] = None,
    fallback_index: Optional[int] = None,  # deprecated, ignored: ids are content-stable now
    sut_usage: Optional[SUTUsage] = None,
) -> TestResult:
    """
    Drop-in replacement for `assert_test(test_case, metrics)`:

        result = evaluate_golden(golden, test_case, RAG_METRICS, csv_path=REPORT_CSV)
        assert result.success, result.name

    Runs `metrics` against `test_case` via evaluate() with test_cases=[test_case]
    (a batch of exactly one), appends one CSV row per metric to csv_path, and
    returns the TestResult so the caller decides how to fail the pytest test.
    print_results/show_indicator are disabled.

    With the result cache enabled (elyaeval.core.cache), metrics already
    judged for this exact test case/judge/repeat are not re-run; their rows
    are marked cached=True with zero cost.

    `sut_usage` (tokens/cost/latency of the app's own call, see
    run_contracts.SUTUsage) goes into the run's usage ledger and budget,
    alongside the judge's measured usage.
    """
    gid = golden_id or _stable_golden_id(golden)
    priority = (golden.additional_metadata or {}).get("priority", "")

    cache = get_result_cache()
    keys: dict[int, str] = {}
    cached: dict[int, MetricData] = {}
    if cache is not None:
        judge = judge_identity()
        for i, metric in enumerate(metrics):
            if not is_cacheable(metric):
                continue
            keys[i] = cache_key(metric, test_case, judge, current_repeat())
            hit = cache.get(keys[i])
            if hit is not None:
                md = MetricData.model_validate(hit)
                cached[i] = md.model_copy(update={"evaluation_cost": 0.0, "input_tokens": 0, "output_tokens": 0})

    to_run = [i for i in range(len(metrics)) if i not in cached]
    test_result: Optional[TestResult] = None
    fresh: dict[str, MetricData] = {}
    judge_seconds = 0.0
    if to_run:
        started = time.perf_counter()
        result = evaluate(
            test_cases=[test_case],
            metrics=[metrics[i] for i in to_run],
            display_config=DisplayConfig(show_indicator=False, print_results=False),
            async_config=AsyncConfig(run_async=False),
            error_config=ErrorConfig(ignore_errors=True),
            # DeepEval's per-run cache file isn't used by elyaeval, and writing
            # it once per golden races on Windows without pywin32 (shared-lock
            # failure -> intermittent 'test_cases_lookup_map' crashes).
            cache_config=CacheConfig(write_cache=False, use_cache=False),
        )
        judge_seconds = time.perf_counter() - started
        test_result = result.test_results[0]
        fresh = {md.name: md for md in test_result.metrics_data or []}
        if cache is not None:
            for i in (i for i in to_run if i in keys):
                fresh_md = fresh.get(metrics[i].__name__)
                if fresh_md is not None and fresh_md.error is None:  # never cache errors/timeouts
                    cache.put(keys[i], fresh_md.model_dump(by_alias=True))

    cached_names: set[str] = set()
    if cached:
        # merge cached + fresh results back into metric order
        merged: list[MetricData] = []
        for i, metric in enumerate(metrics):
            if i in cached:
                merged.append(cached[i])
                cached_names.add(cached[i].name)
            elif metric.__name__ in fresh:
                merged.append(fresh.pop(metric.__name__))
        merged.extend(fresh.values())
        success = all(md.success for md in merged if not md.flaky)
        if test_result is None:
            test_result = TestResult(
                name="test_case_0",
                success=success,
                metrics_data=merged,
                conversational=False,
                index=0,
                multimodal=False,
                input=test_case.input,
                actual_output=test_case.actual_output,
                expected_output=test_case.expected_output,
                context=test_case.context,
                retrieval_context=[
                    rc if isinstance(rc, str) else getattr(rc, "context", str(rc))
                    for rc in test_case.retrieval_context or []
                ] or None,
                turns=None,
                metadata=None,
            )
        else:
            test_result.metrics_data = merged
            test_result.success = success
    assert test_result is not None

    rows = test_result_to_csv_rows(gid, priority, test_result)
    for row in rows:
        if row["metric_name"] in cached_names:  # nothing spent this run
            row.update(cached=True, evaluation_cost=0.0, input_tokens=0, output_tokens=0)
    append_csv_rows(csv_path, rows)

    judge_cost, judge_tokens = reported_judge_usage(test_result.metrics_data or [], frozenset(cached_names))
    usage = sut_usage or SUTUsage()
    METER.record_golden(GoldenUsage(
        golden_id=gid,
        repeat=current_repeat(),
        priority=str(priority),
        n_metrics=len(metrics),
        n_cached=len(cached_names),
        judge_seconds=judge_seconds,
        judge_cost_reported=judge_cost,
        judge_tokens_reported=judge_tokens,
        sut_seconds=usage.latency_seconds,
        sut_input_tokens=usage.input_tokens,
        sut_output_tokens=usage.output_tokens,
        sut_cost=usage.cost,
    ))
    return test_result
