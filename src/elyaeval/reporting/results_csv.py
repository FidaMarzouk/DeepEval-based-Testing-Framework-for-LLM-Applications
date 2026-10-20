"""
The results-CSV format shared by e2e (report.py) and component
(tracing_report.py) runs: one row per (golden, [span,] metric).

Two schemas, identical except that component CSVs carry a span_name
column. Everything that writes or names a results CSV goes through here.

Parallel runs (pytest-xdist): every worker writes its own part file
(results_<name>_<run ts>.<worker>.part.csv); elyaeval's pytest plugin
merges the parts into the single results_<name>_<run ts>.csv on the
controller at session end. The run timestamp is shared by all workers via
ELYAEVAL_RUN_TS, which the plugin sets before workers start.
"""

from __future__ import annotations

import csv
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .html_report import register_report

E2E_FIELDS = [
    "golden_id",
    "priority",
    "repeat",
    "input",
    "actual_output",
    "metric_name",
    "score",
    "threshold",
    "success",
    "reason",
    "error",
    "evaluation_model",
    "evaluation_cost",
    "input_tokens",
    "output_tokens",
    "cached",
]
TRACED_FIELDS = E2E_FIELDS[:2] + ["span_name"] + E2E_FIELDS[2:]

E2E_GROUP_KEYS = ("metric_name",)
TRACED_GROUP_KEYS = ("span_name", "metric_name")

RUN_TS_ENV = "ELYAEVAL_RUN_TS"

# Which repeat of a golden is running (0-based). Set by the pytest plugin
# around each test when --elyaeval-repeats > 1; 0 otherwise.
_CURRENT_REPEAT: ContextVar[int] = ContextVar("elyaeval_repeat", default=0)


def current_repeat() -> int:
    return _CURRENT_REPEAT.get()


def set_current_repeat(index: int):
    """Returns a token for _CURRENT_REPEAT.reset()."""
    return _CURRENT_REPEAT.set(index)


def reset_current_repeat(token) -> None:
    _CURRENT_REPEAT.reset(token)
PART_SUFFIX = ".part.csv"

# Spreadsheet apps execute cells starting with these as formulas; LLM output
# (inputs, reasons, errors) is untrusted text, so it's escaped on write.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def run_timestamp() -> str:
    ts = os.environ.get(RUN_TS_ENV)
    if not ts:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.environ[RUN_TS_ENV] = ts  # same value for every new_report_path() this process
    return ts


def new_report_path(name: str, report_dir: str | Path = "report") -> Path:
    """report/results_<name>_<run timestamp>.csv — or, under an xdist
    worker, that worker's part file. Call ONCE at module level in a test
    file; every golden in the session appends to the same file."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"results_{name}_{run_timestamp()}"
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    return report_dir / (f"{stem}.{worker}{PART_SUFFIX}" if worker else f"{stem}.csv")


def final_path_for(part_path: Path) -> Path:
    """results_x_TS.gw0.part.csv -> results_x_TS.csv"""
    stem = part_path.name[: -len(PART_SUFFIX)].rsplit(".", 1)[0]
    return part_path.with_name(stem + ".csv")


def escape_cell(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in (*_FORMULA_TRIGGERS, "'"):
        return "'" + value
    return value


def unescape_cell(value: Any) -> Any:
    if isinstance(value, str) and value[:1] == "'":
        return value[1:]
    return value


def _dedupe_metrics(metrics_data: list[Any]) -> list[Any]:
    """One entry per metric name. DeepEval 4.2.0 can report a metric that hit
    the per-case timeout twice (a 'Timed out' entry and a RetryError entry),
    which would double-count it in averages and error rates. Keep the first
    entry with a score, else the first entry."""
    chosen: dict[str, Any] = {}
    for md in metrics_data:
        current = chosen.get(md.name)
        if current is None or (current.score is None and md.score is not None):
            chosen[md.name] = md
    return list(chosen.values())


def metric_rows(
    golden_id: str,
    priority: str,
    test_result: Any,
    span_name: Optional[str] = None,
) -> list[dict]:
    """One row per entry in test_result.metrics_data. A result with no
    metrics_data (e.g. an unscored trace-level result) produces no rows.

    evaluation_model/cost/input_tokens/output_tokens come straight off
    DeepEval's MetricData; a judge that doesn't report them leaves them
    None (blank in the CSV), never 0. DeepEval reports local judges (e.g.
    Ollama) as cost 0.0 with 0 tokens — that's "not reported", so it's
    written blank too (elyaeval meters such judges itself, see core.usage)."""
    rows = []
    for md in _dedupe_metrics(test_result.metrics_data or []):
        cost, in_tok, out_tok = md.evaluation_cost, getattr(md, "input_tokens", None), getattr(md, "output_tokens", None)
        if not cost and not in_tok and not out_tok:
            cost = in_tok = out_tok = None
        row = {
            "golden_id": golden_id,
            "priority": priority,
            "repeat": current_repeat(),
            "input": test_result.input,
            "actual_output": getattr(test_result, "actual_output", None),
            "metric_name": md.name,
            "score": md.score,
            "threshold": md.threshold,
            "success": md.success,
            "reason": md.reason,
            "error": md.error,
            "evaluation_model": md.evaluation_model,
            "evaluation_cost": cost,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cached": False,
        }
        if span_name is not None:
            row["span_name"] = span_name
        rows.append(row)
    return rows


def append_rows(
    csv_path: str | Path,
    rows: list[dict],
    fields: list[str],
    group_keys: tuple[str, ...],
) -> None:
    """Append rows (writing the header first if the file is new/empty) and
    register the CSV with the pytest plugin for HTML rendering at session
    end. Safe under xdist because each worker has its own part file."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows({k: escape_cell(v) for k, v in row.items()} for row in rows)
    register_report(path, group_keys=group_keys)


def merge_parts(part_paths: list[Path]) -> Path:
    """Concatenate worker part files (same header) into their final CSV and
    delete the parts. Returns the final path."""
    part_paths = sorted(part_paths)
    final = final_path_for(part_paths[0])
    header: Optional[list[str]] = None
    rows: list[dict] = []
    for part in part_paths:
        with open(part, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = header or list(reader.fieldnames or [])
            rows.extend(reader)
    with open(final, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header or E2E_FIELDS)
        writer.writeheader()
        writer.writerows(rows)  # already escaped by the workers
    for part in part_paths:
        part.unlink()
    return final
