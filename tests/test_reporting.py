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
