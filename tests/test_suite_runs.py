"""
End-to-end: run small generated-style suites in a subprocess, through
DeepEval's real evaluate()/evals_iterator() and elyaeval's pytest plugin,
with a deterministic metric (no LLM). Covers serial runs, pytest-xdist
runs (worker part files merged into one CSV) and component/span mode.
"""

import csv
import importlib.util
import json
import os

import pytest

GOLDENS = [
    {"id": "pass-1", "input": "pass one", "expected_output": "A"},
    {"id": "pass-2", "input": "pass two", "expected_output": "B"},
    {"id": "fail-1", "input": "fail one", "expected_output": "C"},
]

EXACT_MATCH_CONFTEST = '''
from deepeval.metrics import BaseMetric

class ExactMatchMetric(BaseMetric):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.evaluation_model = "exact-match"
        self.evaluation_cost = None
        self.include_reason = True
        self.async_mode = False
        self.strict_mode = False
        self.verbose_mode = False

    def measure(self, test_case, *args, **kwargs):
        self.score = 1.0 if test_case.actual_output == test_case.expected_output else 0.0
        self.reason = "exact match" if self.score else "mismatch"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return bool(self.success)

    @property
    def __name__(self):
        return "Exact Match"
'''

E2E_SUITE = '''
import pytest
from deepeval.test_case import LLMTestCase
from elyaeval import evaluate_golden, load_standard_dataset, new_report_path
from elyaeval.core.dataset import golden_id
from conftest import ExactMatchMetric

GOLDENS = load_standard_dataset(dataset_path="goldens.jsonl", ci_stage="nightly")
REPORT_CSV = new_report_path("qa")

@pytest.mark.parametrize("golden", GOLDENS, ids=[golden_id(g) for g in GOLDENS])
def test_qa(golden):
    actual = golden.expected_output if golden.input.startswith("pass") else "wrong"
    tc = LLMTestCase(input=golden.input, actual_output=actual, expected_output=golden.expected_output)
    result = evaluate_golden(golden, tc, [ExactMatchMetric()], csv_path=REPORT_CSV)
    assert result.success
'''

COMPONENT_SUITE = '''
import pytest
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span
from elyaeval import load_standard_dataset, new_report_path
from elyaeval.core.dataset import golden_id
from elyaeval.reporting.tracing_report import TracedRunner, append_traced_csv_rows, failed_results, traced_test_result_to_csv_rows
from conftest import ExactMatchMetric

GOLDENS = load_standard_dataset(dataset_path="goldens.jsonl", ci_stage="nightly")
REPORT_CSV = new_report_path("qa_component")

@observe(metrics=[ExactMatchMetric()])
def generate_answer(question, expected):
    answer = expected if question.startswith("pass") else "wrong"
    update_current_span(test_case=LLMTestCase(input=question, actual_output=answer, expected_output=expected))
    return answer

@observe()
def pipeline(question, expected):
    return generate_answer(question, expected)

@pytest.mark.parametrize("golden", GOLDENS, ids=[golden_id(g) for g in GOLDENS])
def test_qa_component(golden):
    result = TracedRunner(golden).run(lambda g: pipeline(g.input, g.expected_output))
    rows = []
    for tr in result.test_results:
        rows.extend(traced_test_result_to_csv_rows(golden_id(golden), "P0", tr))
    append_traced_csv_rows(REPORT_CSV, rows)
    assert not failed_results(result)
'''


@pytest.fixture
def suite_dir(pytester, monkeypatch):
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    monkeypatch.setenv("ERROR_REPORTING", "NO")
    monkeypatch.delenv("ELYAEVAL_RUN_TS", raising=False)
    rows = [
        {**g, "additional_metadata": {"task_type": "qa_correctness", "ci_stage": "per_commit", "priority": "P0"}}
        for g in GOLDENS
    ]
    (pytester.path / "goldens.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pytester.makeconftest(EXACT_MATCH_CONFTEST)
    return pytester


def _results(pytester, name):
    report = pytester.path / "report"
    csvs = sorted(report.glob(f"results_{name}_*.csv"))
    assert len(csvs) == 1, f"expected one merged CSV, got {[p.name for p in report.iterdir()]}"
    assert not list(report.glob("*.part.csv")), "worker part files were not merged"
    assert csvs[0].with_suffix(".html").exists(), "HTML report was not rendered"
    with open(csvs[0], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_e2e_suite_serial(suite_dir):
    suite_dir.makepyfile(test_suite=E2E_SUITE)
    result = suite_dir.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(["*elyaeval report*", "*Exact Match*"])
    rows = _results(suite_dir, "qa")
    assert sorted(r["golden_id"] for r in rows) == ["fail-1", "pass-1", "pass-2"]
    assert {r["golden_id"]: r["score"] for r in rows}["fail-1"] == "0.0"
    assert {r["golden_id"]: r["actual_output"] for r in rows} == {"pass-1": "A", "pass-2": "B", "fail-1": "wrong"}


@pytest.mark.skipif(importlib.util.find_spec("xdist") is None, reason="pytest-xdist not installed")
def test_e2e_suite_under_xdist_merges_worker_parts(suite_dir):
    suite_dir.makepyfile(test_suite=E2E_SUITE)
    result = suite_dir.runpytest_subprocess("-p", "no:cacheprovider", "-n", "2")
    result.assert_outcomes(passed=2, failed=1)
    rows = _results(suite_dir, "qa")
    assert sorted(r["golden_id"] for r in rows) == ["fail-1", "pass-1", "pass-2"]


def test_component_suite_scores_spans(suite_dir):
    """Also guards the DeepEval trace end_time issue an older consumer conftest
    used to monkeypatch: this exercises the same sync evals_iterator path."""
    suite_dir.makepyfile(test_suite=COMPONENT_SUITE)
    result = suite_dir.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=2, failed=1)
    rows = _results(suite_dir, "qa_component")
    span_rows = [r for r in rows if r["span_name"] == "generate_answer"]
    assert sorted(r["golden_id"] for r in span_rows) == ["fail-1", "pass-1", "pass-2"]


def test_repeats_and_metadata_sidecar(suite_dir):
    suite_dir.makepyfile(test_suite=E2E_SUITE)
    result = suite_dir.runpytest_subprocess("-p", "no:cacheprovider", "--elyaeval-repeats", "2")
    result.assert_outcomes(passed=4, failed=2)
    rows = _results(suite_dir, "qa")
    assert sorted((r["golden_id"], r["repeat"]) for r in rows if r["golden_id"] == "pass-1") == [
        ("pass-1", "0"),
        ("pass-1", "1"),
    ]
    (meta_path,) = (suite_dir.path / "report").glob("results_qa_*.meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["repeats"] == 2
    assert meta["judge"] == "deepeval:default"
    assert meta["datasets"][0]["path"] == "goldens.jsonl" and meta["dataset_sha256"]


def _write_run(path, scores, judge="ollama:llama3.1"):
    lines = ["golden_id,priority,repeat,input,metric_name,score,threshold,success,reason,error"]
    lines += [f"{g},P0,0,q,Faithfulness,{s},0.5,{s >= 0.5},,"for g, s in scores.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".meta.json").write_text(
        json.dumps({"judge": judge, "judge_temperature": 0.0}), encoding="utf-8"
    )
    return str(path)


def test_compare_cli_exit_codes(tmp_path):
    import subprocess
    import sys

    ids = [f"g{i}" for i in range(8)]
    base = _write_run(tmp_path / "base.csv", dict.fromkeys(ids, 0.9))
    same = _write_run(tmp_path / "same.csv", dict.fromkeys(ids, 0.9))
    worse = _write_run(tmp_path / "worse.csv", dict.fromkeys(ids, 0.5))
    other_judge = _write_run(tmp_path / "other.csv", dict.fromkeys(ids, 0.9), judge="anthropic:claude-sonnet-5-5")

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "elyaeval.cli", "compare", *args],
            capture_output=True, text=True, encoding="utf-8", cwd=tmp_path,
            # the child writes in the console's encoding otherwise (cp1252 on Windows)
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

    assert run("--baseline", base, "--candidate", same).returncode == 0
    regressed = run("--baseline", base, "--candidate", worse, "--html", str(tmp_path / "r.html"))
    assert regressed.returncode == 1, regressed.stdout + regressed.stderr
    assert "REGRESSED" in regressed.stdout and (tmp_path / "r.html").exists()
    mismatch = run("--baseline", base, "--candidate", other_judge)
    assert mismatch.returncode == 2 and "judge differs" in mismatch.stdout
    assert run("--baseline", base, "--candidate", other_judge, "--allow-metadata-mismatch").returncode == 0


SLOW_SUITE = '''
import time

import pytest
from deepeval.test_case import LLMTestCase
from elyaeval import evaluate_golden, load_standard_dataset, new_report_path
from elyaeval.core.dataset import golden_id
from conftest import ExactMatchMetric

GOLDENS = load_standard_dataset(dataset_path="goldens.jsonl", ci_stage="nightly")
REPORT_CSV = new_report_path("qa")


class SlowMetric(ExactMatchMetric):
    def measure(self, test_case, *args, **kwargs):
        time.sleep(0.4)
        return super().measure(test_case)


@pytest.mark.parametrize("golden", GOLDENS, ids=[golden_id(g) for g in GOLDENS])
def test_qa(golden):
    tc = LLMTestCase(input=golden.input, actual_output=golden.expected_output, expected_output=golden.expected_output)
    assert evaluate_golden(golden, tc, [SlowMetric()], csv_path=REPORT_CSV).success
'''


def test_time_budget_runs_p0_first_and_skips_the_rest(pytester, monkeypatch):
    import subprocess
    import sys

    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    monkeypatch.delenv("ELYAEVAL_RUN_TS", raising=False)
    priorities = ["P2", "P2", "P1", "P0", "P2", "P0"]  # file order != priority order
    rows = [
        {"id": f"{p.lower()}-{i}", "input": f"q{i}", "expected_output": "A",
         "additional_metadata": {"task_type": "qa_correctness", "ci_stage": "per_commit", "priority": p}}
        for i, p in enumerate(priorities)
    ]
    (pytester.path / "goldens.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pytester.makeconftest(EXACT_MATCH_CONFTEST)
    pytester.makepyfile(test_suite=SLOW_SUITE)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--elyaeval-budget-seconds", "1.5", "-v")
    outcomes = result.parseoutcomes()
    assert outcomes["passed"] >= 2 and outcomes["skipped"] >= 2, result.stdout.str()
    # -v lines look like "test_suite.py::test_qa[p0-3] PASSED"; keep run order
    verdicts = [
        (line.split("[", 1)[1].split("]", 1)[0], "PASSED" in line)
        for line in result.stdout.lines
        if "test_qa[" in line and ("PASSED" in line or "SKIPPED" in line)
    ]
    ran = [gid for gid, passed in verdicts if passed]
    assert {gid[:2] for gid in ran[:2]} == {"p0"}, f"P0 goldens must run first, got {verdicts}"
    result.stdout.fnmatch_lines(["*Budget EXHAUSTED*"])

    (ledger_path,) = (pytester.path / "report").glob("budget_*.json")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["budget"]["exhausted"] and ledger["budget"]["skipped"]
    assert all("p2" in node for node in ledger["budget"]["skipped"][:1])
    assert ledger["totals"]["goldens_evaluated"] == outcomes["passed"]
    (html,) = (pytester.path / "report").glob("results_qa_*.html")
    assert "Budget exhausted" in html.read_text(encoding="utf-8")

    gate = subprocess.run(
        [sys.executable, "-m", "elyaeval.cli", "budget", "--fail-on-exhausted"],
        capture_output=True, text=True, encoding="utf-8", cwd=pytester.path,
    )
    assert gate.returncode == 1 and "EXHAUSTED" in gate.stdout

    est = subprocess.run(
        [sys.executable, "-m", "elyaeval.cli", "estimate", "--task-type", "qa_correctness", "--ci-stage", "nightly",
         "--dataset", "goldens.jsonl", "--budget-seconds", "1"],
        capture_output=True, text=True, encoding="utf-8", cwd=pytester.path,
    )
    assert est.returncode == 1 and "history" in est.stdout and "Over budget" in est.stdout, est.stdout + est.stderr


HARNESS_COMPONENT_SUITE = '''
import pytest
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span
from elyaeval import load_standard_dataset, new_report_path
from elyaeval.core.dataset import golden_id
from elyaeval.reporting.tracing_report import TracedRunner, append_traced_csv_rows, failed_results, traced_test_result_to_csv_rows
from conftest import ExactMatchMetric

GOLDENS = load_standard_dataset(dataset_path="goldens.jsonl", ci_stage="nightly")
REPORT_CSV = new_report_path("qa_component")
ANSWERS = {"pass one": "A", "pass two": "B", "fail one": "wrong"}


# production-style app code: plain spans, no metrics, never sees expected_output
@observe()
def generate_answer(question):
    answer = ANSWERS[question]
    update_current_span(test_case=LLMTestCase(input=question, actual_output=answer))
    return answer


@observe()
def pipeline(question):
    return generate_answer(question)


@pytest.mark.parametrize("golden", GOLDENS, ids=[golden_id(g) for g in GOLDENS])
def test_qa_component(golden):
    result = TracedRunner(golden, span_metrics={"generate_answer": lambda: [ExactMatchMetric()]}).run(
        lambda g: pipeline(g.input)
    )
    rows = []
    for tr in result.test_results:
        rows.extend(traced_test_result_to_csv_rows(golden_id(golden), "P0", tr))
    append_traced_csv_rows(REPORT_CSV, rows)
    assert not failed_results(result)
'''


def test_component_metrics_attached_by_the_harness(suite_dir):
    """span_metrics + golden expected_output: the app declares no metrics and
    never receives the reference answer, yet its span is scored against it."""
    suite_dir.makepyfile(test_suite=HARNESS_COMPONENT_SUITE)
    result = suite_dir.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=2, failed=1)
    rows = _results(suite_dir, "qa_component")
    scores = {r["golden_id"]: r["score"] for r in rows if r["span_name"] == "generate_answer"}
    assert scores == {"pass-1": "1.0", "pass-2": "1.0", "fail-1": "0.0"}


AGENT_TRACE_SUITE = '''
import pytest
from deepeval.tracing import observe, update_current_trace
from elyaeval import load_standard_dataset, new_report_path
from elyaeval.core.agentic import guard_metrics
from elyaeval.core.dataset import golden_id
from elyaeval.reporting.tracing_report import TracedRunner, append_traced_csv_rows, failed_results, traced_test_result_to_csv_rows

GOLDENS = load_standard_dataset(dataset_path="goldens.jsonl", ci_stage="nightly")
REPORT_CSV = new_report_path("agent")


@observe(type="tool")
def lookup(query):
    return f"result for {query}"


@observe(type="agent")
def run_agent(question):
    # the "fail" golden loops: same tool call three times
    for _ in range(3 if question.startswith("fail") else 1):
        lookup(question)
    update_current_trace(input=question, output="done")
    return "done"


@pytest.mark.parametrize("golden", GOLDENS, ids=[golden_id(g) for g in GOLDENS])
def test_agent(golden):
    result = TracedRunner(golden, metrics=guard_metrics(golden)).run(lambda g: run_agent(g.input))
    rows = []
    for tr in result.test_results:
        rows.extend(traced_test_result_to_csv_rows(golden_id(golden), "P0", tr))
    append_traced_csv_rows(REPORT_CSV, rows)
    assert not failed_results(result)
'''


def test_agent_guards_on_real_traces(suite_dir):
    """Deterministic guards read tool spans from DeepEval's real trace dict."""
    suite_dir.makepyfile(test_suite=AGENT_TRACE_SUITE)
    result = suite_dir.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=2, failed=1)
    rows = _results(suite_dir, "agent")
    loops = {r["golden_id"]: r["score"] for r in rows if r["metric_name"] == "Repeated Tool Calls"}
    assert loops == {"pass-1": "1.0", "pass-2": "1.0", "fail-1": "0.0"}
    reasons = {r["golden_id"]: r["reason"] for r in rows if r["metric_name"] == "Tool Call Budget"}
    assert reasons["fail-1"].startswith("3 tool call(s)")
    # trace-level results get a stable span label, not the pytest test name
    assert {r["span_name"] for r in rows} == {"trace"}
