"""CSV writing/reading, formula escaping, worker-part merging, comparison HTML."""

import csv

from elyaeval.reporting.html_report import compare_runs, read_csv_rows, render_comparison_html
from elyaeval.reporting.results_csv import (
    E2E_FIELDS,
    E2E_GROUP_KEYS,
    append_rows,
    escape_cell,
    final_path_for,
    merge_parts,
    new_report_path,
    unescape_cell,
)


def _row(gid, metric, score, **extra):
    return {"golden_id": gid, "priority": "P0", "input": "q", "metric_name": metric, "score": score,
            "threshold": 0.5, "success": score >= 0.5, **extra}


def test_formula_escaping_roundtrip():
    for value in ["=SUM(A1)", "+1", "-cmd", "@x", "'quoted", "plain", "", "''double"]:
        assert unescape_cell(escape_cell(value)) == value
    assert escape_cell("=1+1").startswith("'")
    assert escape_cell(-0.5) == -0.5  # numbers untouched


def test_append_and_read_roundtrip_with_untrusted_text(tmp_path):
    path = tmp_path / "results_x_20260101T000000Z.csv"
    append_rows(path, [_row("g1", "M", 0.9, reason="=HYPERLINK(\"evil\")")], E2E_FIELDS, E2E_GROUP_KEYS)
    append_rows(path, [_row("g2", "M", 0.1, reason="fine")], E2E_FIELDS, E2E_GROUP_KEYS)
    raw = path.read_text(encoding="utf-8")
    assert "'=HYPERLINK" in raw  # escaped on disk
    rows = read_csv_rows(path)
    assert [r["golden_id"] for r in rows] == ["g1", "g2"]
    assert rows[0]["reason"] == '=HYPERLINK("evil")'  # restored on read
    assert rows[1]["score"] == 0.1


def test_new_report_path_uses_shared_run_ts_and_worker_parts(tmp_path, monkeypatch):
    monkeypatch.setenv("ELYAEVAL_RUN_TS", "20260101T000000Z")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    assert new_report_path("rag_qa", tmp_path).name == "results_rag_qa_20260101T000000Z.csv"
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    part = new_report_path("rag_qa", tmp_path)
    assert part.name == "results_rag_qa_20260101T000000Z.gw1.part.csv"
    assert final_path_for(part).name == "results_rag_qa_20260101T000000Z.csv"


def test_merge_parts(tmp_path):
    parts = []
    for worker, gid in (("gw0", "a"), ("gw1", "b")):
        part = tmp_path / f"results_x_20260101T000000Z.{worker}.part.csv"
        append_rows(part, [_row(gid, "M", 1.0)], E2E_FIELDS, E2E_GROUP_KEYS)
        parts.append(part)
    final = merge_parts(parts)
    assert final.name == "results_x_20260101T000000Z.csv"
    assert not any(p.exists() for p in parts)
    with open(final, newline="", encoding="utf-8") as f:
        assert sorted(r["golden_id"] for r in csv.DictReader(f)) == ["a", "b"]


def test_comparison_html_shows_actual_tolerance(tmp_path):
    base = [_row("g", "M", 0.9)]
    cand = [_row("g", "M", 0.5)]
    comparisons = compare_runs(base, cand, tolerance=0.1)
    assert comparisons[0]["status"] == "regressed"
    html = render_comparison_html(comparisons, html_path=tmp_path / "r.html", tolerance=0.1).read_text(encoding="utf-8")
    assert "Beyond tolerance (0.1)" in html


def test_duplicate_metric_entries_are_collapsed():
    """DeepEval can report a timed-out metric twice; keep one row per metric."""
    from types import SimpleNamespace

    from elyaeval.reporting.results_csv import metric_rows

    def md(name, score, error=None):
        return SimpleNamespace(name=name, score=score, threshold=0.5, success=score is not None and score >= 0.5,
                               reason="", error=error, evaluation_model="j", evaluation_cost=None,
                               input_tokens=None, output_tokens=None)

    result = SimpleNamespace(input="q", metrics_data=[
        md("Relevancy", None, "Timed out after 900s"),
        md("Relevancy", None, "RetryError"),
        md("Faithfulness", None, "boom"),
        md("Faithfulness", 0.8),
    ])
    rows = metric_rows("g1", "P0", result)
    assert [(r["metric_name"], r["score"]) for r in rows] == [("Relevancy", None), ("Faithfulness", 0.8)]
    assert rows[0]["error"] == "Timed out after 900s"


# ---------------------------------------------------------------------------
# Per-metric judge usage, errored rows, per-golden usage
# ---------------------------------------------------------------------------


def _metered_metric(name: str, calls: int):
    """A deterministic metric that 'calls the judge' `calls` times, the way
    an instrumented judge records each call on the meter."""
    from conftest import ExactMatchMetric
    from elyaeval.core.usage import METER

    class Metered(ExactMatchMetric):
        def measure(self, test_case, *args, **kwargs):
            for _ in range(calls):
                METER.record_judge_call(0.5, prompt_tokens=100, output_tokens=20, ok=True)
            return super().measure(test_case)

        @property
        def __name__(self):
            return name

    return Metered()


def test_judge_usage_is_attributed_per_metric(tmp_path):
    from deepeval.dataset import Golden
    from deepeval.test_case import LLMTestCase

    from elyaeval.core.run_contracts import SUTUsage
    from elyaeval.core.usage import METER
    from elyaeval.reporting.report import evaluate_golden

    METER.reset()
    golden = Golden(input="q", expected_output="A", additional_metadata={"id": "g1"})
    tc = LLMTestCase(input="q", actual_output="A", expected_output="A")
    csv_path = tmp_path / "results.csv"
    result = evaluate_golden(
        golden, tc, [_metered_metric("Two Calls", 2), _metered_metric("One Call", 1)],
        csv_path=csv_path, sut_usage=SUTUsage(input_tokens=600, output_tokens=30, latency_seconds=4.0),
    )
    assert result.success
    rows = {r["metric_name"]: r for r in read_csv_rows(csv_path)}
    assert rows["Two Calls"]["judge_calls"] == 2 and rows["One Call"]["judge_calls"] == 1
    assert rows["Two Calls"]["input_tokens"] == 200 and rows["Two Calls"]["output_tokens"] == 40
    assert rows["Two Calls"]["tokens_estimated"] and rows["Two Calls"]["judge_seconds"] == 1.0
    usage = METER.ledger()[0]
    assert usage["sut_input_tokens"] == 600 and usage["sut_seconds"] == 4.0
    METER.reset()


def _write_rows(path, rows):
    from elyaeval.reporting.results_csv import E2E_FIELDS

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=E2E_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _report_row(golden, metric, score, success, error="", **extra):
    return {"golden_id": golden, "priority": "P0", "input": "q", "actual_output": "a", "metric_name": metric,
            "score": score, "threshold": 0.7, "success": success, "error": error, "evaluation_model": "llama3.1", **extra}


def test_errored_rows_are_counted_separately_from_low_scores():
    from elyaeval.reporting.html_report import metric_averages

    rows = [
        {"golden_id": "g1", "metric_name": "Recall", "score": None, "success": False, "error": "Invalid JSON"},
        {"golden_id": "g2", "metric_name": "Recall", "score": 0.83, "success": True, "error": ""},
    ]
    (s,) = metric_averages(rows)
    assert s["n_errors"] == 1 and s["golden_count"] == 2 and s["n_goldens"] == 1
    assert (s["pass_count"], s["pass_total"]) == (1, 2)


def test_html_report_shows_errors_estimates_and_per_golden_usage(tmp_path):
    import json

    from elyaeval.reporting.html_report import render_html_report

    csv_path = tmp_path / "results_rag_qa_20261020T220719Z.csv"
    _write_rows(csv_path, [
        _report_row("solar", "Faithfulness", 0.5, False, input_tokens=900, output_tokens=120,
             tokens_estimated=True, judge_calls=3, judge_seconds=12.5),
        _report_row("solar", "Contextual Recall", "", False, error="Invalid JSON: EOF while parsing"),
    ])
    (tmp_path / "budget_20261020T220719Z.json").write_text(json.dumps({"goldens": [
        {"golden_id": "solar", "repeat": 0, "judge_seconds": 80.0,
         "sut_seconds": 35.9, "sut_input_tokens": 627, "sut_output_tokens": 32, "sut_cost": None},
        {"golden_id": "other-suite-golden", "repeat": 0, "judge_seconds": 1.0},
    ]}), encoding="utf-8")
    html = render_html_report(csv_path).read_text(encoding="utf-8")
    assert "● error" in html and "1 error" in html and "1 errored" in html
    assert "≈900" in html and "12.5s · 3 calls" in html
    assert "Per-golden usage" in html and "35.9s" in html and "627" in html
    assert "other-suite-golden" not in html


def test_html_report_without_ledger_still_renders(tmp_path):
    from elyaeval.reporting.html_report import render_html_report

    csv_path = tmp_path / "results_rag_qa_20261020T220719Z.csv"
    _write_rows(csv_path, [_report_row("g", "Faithfulness", 0.9, True)])
    html = render_html_report(csv_path).read_text(encoding="utf-8")
    assert "Per-golden usage" not in html and "Faithfulness" in html
