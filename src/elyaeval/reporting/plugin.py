"""
Auto-registered pytest plugin, this is what makes HTML reports + per-metric
averages appear with NO changes to generated test files or CI YAML beyond
bumping the installed elyaeval version. Nothing in here needs to be
imported explicitly anywhere; pytest discovers and loads it automatically
for any environment that has elyaeval installed.

At the end of the pytest session, for every CSV that the results writers
(results_csv.append_rows) wrote to during THIS run (tracked via
html_report.register_report), this:
  1. renders an HTML sibling next to that CSV (same name, .html extension)
  2. prints a per-metric (or per-span+metric, for component CSVs) average
     table to the terminal, right after pytest's own summary

Only touches CSVs that were actually written this session, a stale
results_*.csv left over from a previous run in the same report/ directory
is not re-rendered or double-counted.

pytest-xdist: workers each write a part file and hand their registered
paths (and usage/metadata) back to the controller via `workeroutput`; the
controller merges the parts into one CSV per suite, then renders/prints as
above. Workers themselves render nothing.

Repeats: `--elyaeval-repeats K` (or ELYAEVAL_REPEATS=K) runs every test
of an elyaeval suite (a module defining GOLDENS) K times; rows carry a
`repeat` column, reports show per-metric 95% CIs and the run-to-run noise.

Run metadata: alongside every results CSV the plugin writes a
<name>.meta.json sidecar (judge, datasets, versions, git SHA, SUT tags)
that `elyaeval compare` uses to refuse apples-to-oranges comparisons.

Budgets (elyaeval.core.budget): goldens run P0-first; a guard checks the
budget before every golden of an elyaeval suite and skips the rest once it
is exhausted (or would be by the next golden); report/budget_<run>.json
records spend, limits, skips and a per-golden usage ledger. Under xdist
each worker enforces an equal share of the budget.

Must stay cheap to import (pytest loads it at startup everywhere): no
deepeval imports at module level.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import pytest

from ..core import budget as budget_mod
from ..core.usage import METER
from . import run_metadata
from .html_report import (
    clear_session_reports,
    print_metric_averages,
    read_csv_rows,
    register_report,
    render_html_report,
    session_reports,
)

_RUN_TS_ENV = "ELYAEVAL_RUN_TS"  # mirrors results_csv.RUN_TS_ENV (not imported: keeps this module light)
_PART_SUFFIX = ".part.csv"
_WORKER_KEY = "elyaeval_reports"
_WORKER_META_KEY = "elyaeval_meta"
_WORKER_USAGE_KEY = "elyaeval_usage"
_REPEAT_PARAM = "elyaeval_repeat"
_REPEATS_ENV = "ELYAEVAL_REPEATS"
_DEFAULT_REPORT_DIR = "report"


def pytest_addoption(parser) -> None:
    group = parser.getgroup("elyaeval")
    group.addoption(
        "--elyaeval-repeats",
        type=int,
        default=None,
        help="Run each golden of elyaeval suites K times to measure run-to-run noise "
        "(default: $ELYAEVAL_REPEATS or 1).",
    )
    group.addoption("--elyaeval-budget-usd", default=None, help="Max judge + SUT spend in USD for this run.")
    group.addoption("--elyaeval-budget-seconds", default=None, help="Max wall-clock seconds of evaluation.")
    group.addoption("--elyaeval-budget-tokens", default=None, help="Max judge + SUT tokens (reported, else estimated).")
    group.addoption("--elyaeval-budget-calls", default=None, help="Max judge LLM calls.")
    group.addoption(
        "--elyaeval-budget-on-exceed",
        default=None,
        choices=["stop", "warn"],
        help="stop (default): skip remaining goldens once the budget is exhausted; warn: only report.",
    )


def _repeats(config) -> int:
    value = config.getoption("--elyaeval-repeats", default=None)
    if value is None:
        value = int(os.environ.get(_REPEATS_ENV) or 1)
    return max(1, int(value))


def _is_worker(config) -> bool:
    return hasattr(config, "workerinput")


def _is_elyaeval_item(item) -> bool:
    return hasattr(getattr(item, "module", None), "GOLDENS")


def _item_priority(item) -> Any:
    callspec = getattr(item, "callspec", None)
    golden = callspec.params.get("golden") if callspec else None
    meta = getattr(golden, "additional_metadata", None) or {}
    return meta.get("priority")


def pytest_generate_tests(metafunc) -> None:
    repeats = _repeats(metafunc.config)
    if repeats > 1 and hasattr(metafunc.module, "GOLDENS"):
        metafunc.fixturenames.append(_REPEAT_PARAM)
        metafunc.parametrize(_REPEAT_PARAM, range(repeats), ids=[f"r{i}" for i in range(repeats)])


@pytest.fixture(autouse=True)
def _elyaeval_repeat_context(request):
    """Expose the current repeat index to the CSV writers (repeat column)."""
    from .results_csv import reset_current_repeat, set_current_repeat

    callspec = getattr(request.node, "callspec", None)
    token = set_current_repeat(callspec.params.get(_REPEAT_PARAM, 0) if callspec else 0)
    yield
    reset_current_repeat(token)


def _build_guard(config) -> Optional[budget_mod.BudgetGuard]:
    try:
        cfg = budget_mod.budget_config_from_settings(
            max_cost_usd=config.getoption("--elyaeval-budget-usd"),
            max_seconds=config.getoption("--elyaeval-budget-seconds"),
            max_tokens=config.getoption("--elyaeval-budget-tokens"),
            max_judge_calls=config.getoption("--elyaeval-budget-calls"),
            on_exceed=config.getoption("--elyaeval-budget-on-exceed"),
        )
        pricing = budget_mod.pricing_from_settings()
    except ValueError as exc:
        raise pytest.UsageError(f"elyaeval budget configuration: {exc}") from exc
    if _is_worker(config):
        # each xdist worker enforces an equal share of the run's budget
        workers = max(int(config.workerinput.get("workercount", 1)), 1)
        cfg = budget_mod.BudgetConfig(
            max_cost_usd=cfg.max_cost_usd / workers if cfg.max_cost_usd is not None else None,
            max_seconds=cfg.max_seconds,  # wall clock runs in parallel
            max_tokens=cfg.max_tokens // workers if cfg.max_tokens is not None else None,
            max_judge_calls=cfg.max_judge_calls // workers if cfg.max_judge_calls is not None else None,
            on_exceed=cfg.on_exceed,
            predictive=cfg.predictive,
            prioritize=cfg.prioritize,
            preflight=cfg.preflight,
        )
    return budget_mod.BudgetGuard(cfg, pricing)


def pytest_configure(config) -> None:
    # One timestamp for the whole run, inherited by xdist workers (they're
    # spawned after configure), so every worker's part file shares a stem.
    if not _is_worker(config) and not os.environ.get(_RUN_TS_ENV):
        from datetime import datetime, timezone

        os.environ[_RUN_TS_ENV] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        config._elyaeval_set_run_ts = True
    config._elyaeval_guard = _build_guard(config)


def pytest_sessionstart(session) -> None:
    clear_session_reports()
    run_metadata.clear()
    METER.reset()


def pytest_unconfigure(config) -> None:
    if getattr(config, "_elyaeval_set_run_ts", False):
        os.environ.pop(_RUN_TS_ENV, None)


def _writer(config):
    terminalreporter = config.pluginmanager.get_plugin("terminalreporter")

    def _write(line: str = "") -> None:
        if terminalreporter is not None:
            terminalreporter.write_line(line)
        else:
            print(line)

    return _write


def pytest_collection_modifyitems(session, config, items) -> None:
    guard: Optional[budget_mod.BudgetGuard] = getattr(config, "_elyaeval_guard", None)
    if guard is None:
        return
    elyaeval_items = [it for it in items if _is_elyaeval_item(it)]
    if not elyaeval_items:
        return
    if guard.config.prioritize:
        # non-elyaeval tests first (cheap, deterministic), then goldens P0 -> P3
        items.sort(key=lambda it: 0 if not _is_elyaeval_item(it) else 1 + budget_mod.priority_rank(_item_priority(it)))
    if guard.config.enabled and guard.config.preflight and not _is_worker(config):
        _preflight(config, guard, len(elyaeval_items))


def _preflight(config, guard: budget_mod.BudgetGuard, n_items: int) -> None:
    write = _writer(config)
    history = budget_mod.latest_ledger(_DEFAULT_REPORT_DIR)
    limits = ", ".join(f"{k}={v}" for k, v in guard.config.limits().items() if v is not None)
    if history is None:
        write(f"[elyaeval] budget {limits} (on_exceed={guard.config.on_exceed}); no previous run ledger in "
              f"{_DEFAULT_REPORT_DIR}/ to project from   enforcing as the run goes.")
        return
    totals = history["totals"]
    metrics_per_golden = max(round((totals.get("metric_evaluations") or 0) / max(totals.get("goldens_evaluated") or 1, 1)), 1)
    projection = budget_mod.project(n_items, metrics_per_golden, 1, budget_mod.profile_from_ledger(history))
    over = budget_mod.projection_over_budget(projection, guard.config)
    seconds = projection.get("seconds")
    cost = projection.get("cost_usd")
    duration = f", ~{seconds / 60:.1f} min" if seconds is not None else ""
    write(f"[elyaeval] budget {limits}; preflight from {Path(history['_path']).name}: {n_items} golden run(s){duration}")
    if cost is not None:
        write(f"[elyaeval]   projected cost ~${cost:.4f}")
    for line in over:
        write(f"[elyaeval]   ! {line}   lowest-priority goldens will be skipped" if guard.config.on_exceed == "stop"
              else f"[elyaeval]   ! {line}")


def pytest_runtest_setup(item) -> None:
    guard: Optional[budget_mod.BudgetGuard] = getattr(item.config, "_elyaeval_guard", None)
    if guard is None or not _is_elyaeval_item(item):
        return
    guard.start()
    reason = guard.check()
    if reason is not None:
        guard.record_skip(item.nodeid)
        pytest.skip(reason)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    """xdist controller: collect the CSV paths, metadata and usage a finished worker reported."""
    output = getattr(node, "workeroutput", {}) or {}
    for path, group_keys in output.get(_WORKER_KEY, []):
        register_report(path, group_keys=tuple(group_keys))
    run_metadata.merge_worker_payload(output.get(_WORKER_META_KEY, {}))
    usage = output.get(_WORKER_USAGE_KEY)
    if usage:
        METER.merge(usage["meter"])
        guard = getattr(node.config, "_elyaeval_guard", None)
        if guard is not None:
            guard.skipped.extend(usage.get("skipped", []))
            guard.warnings.extend(w for w in usage.get("warnings", []) if w not in guard.warnings)
            if usage.get("exhausted_reason") and guard.exhausted_reason is None:
                guard.exhausted_reason = usage["exhausted_reason"]
        # workers run in parallel: the run's wall-clock is the slowest worker's
        node.config._elyaeval_worker_elapsed = max(
            getattr(node.config, "_elyaeval_worker_elapsed", 0.0), usage.get("elapsed", 0.0)
        )


def _merge_worker_parts(reports: dict[Path, tuple[str, ...]]) -> dict[Path, tuple[str, ...]]:
    from .results_csv import final_path_for, merge_parts

    merged: dict[Path, tuple[str, ...]] = {}
    parts: dict[Path, list[Path]] = {}
    for path, keys in reports.items():
        if path.name.endswith(_PART_SUFFIX):
            parts.setdefault(final_path_for(path), []).append(path)
            merged[final_path_for(path)] = keys
        else:
            merged[path] = keys
    for part_paths in parts.values():
        existing = [p for p in part_paths if p.exists()]
        if existing:
            merge_parts(existing)
    return merged


def _budget_lines(status: dict[str, Any]) -> list[str]:
    spent = status["spent"]
    limits = status["limits"]

    def show(dim: str, value: Any) -> str:
        if value is None:
            return "n/a"
        return budget_mod._fmt(dim, value)

    parts = []
    for dim in ("seconds", "cost_usd", "tokens", "judge_calls"):
        limit = limits.get(dim)
        parts.append(f"{dim}={show(dim, spent.get(dim))}" + (f"/{show(dim, limit)}" if limit is not None else ""))
    lines = [f"Usage: {', '.join(parts)} (cost: judge {spent['judge_cost_source']}, SUT {spent['sut_cost_source']}; "
             f"tokens {spent['tokens_source']})"]
    if status["exhausted"]:
        lines.append(f"Budget EXHAUSTED: {status['exhausted_reason']}; {len(status['skipped'])} golden run(s) skipped")
    for warning in status["warnings"]:
        lines.append(f"Budget warning: {warning}")
    return lines


def pytest_sessionfinish(session, exitstatus) -> None:
    config = session.config
    reports = session_reports()
    guard: Optional[budget_mod.BudgetGuard] = getattr(config, "_elyaeval_guard", None)

    if _is_worker(config):
        config.workeroutput[_WORKER_KEY] = [(str(p), list(k)) for p, k in reports.items()]
        config.workeroutput[_WORKER_META_KEY] = run_metadata.worker_payload()
        config.workeroutput[_WORKER_USAGE_KEY] = {
            "meter": METER.worker_payload(),
            "skipped": guard.skipped if guard else [],
            "warnings": guard.warnings if guard else [],
            "exhausted_reason": guard.exhausted_reason if guard else None,
            "elapsed": guard.elapsed() if guard else 0.0,
        }
        return

    ran_goldens = bool(METER.goldens) or bool(guard and guard.skipped)
    if not reports and not ran_goldens:
        return

    reports = _merge_worker_parts(reports)
    write = _writer(config)

    status: Optional[dict[str, Any]] = None
    if guard is not None and ran_goldens:
        status = guard.status()
        worker_elapsed = getattr(config, "_elyaeval_worker_elapsed", None)
        if worker_elapsed is not None:  # xdist: the controller never ran a golden itself
            status["spent"]["seconds"] = worker_elapsed
        report_dir = next(iter(reports)).parent if reports else Path(_DEFAULT_REPORT_DIR)
        try:
            budget_mod.write_ledger(report_dir, os.environ.get(_RUN_TS_ENV, "run"), guard, status=status)
        except OSError as exc:
            write(f"[elyaeval] could not write budget ledger: {exc}")

    try:
        meta = run_metadata.build_metadata(repeats=_repeats(config))
    except Exception as exc:  # noqa: BLE001   reporting must never fail the run
        meta = None
        write(f"[elyaeval] could not build run metadata: {exc}")

    write("")
    write("=" * 20 + " elyaeval report " + "=" * 20)
    for csv_path, group_keys in reports.items():
        if not csv_path.exists():
            continue
        if meta is not None:
            run_metadata.write_sidecar(csv_path, meta)
        try:
            html_path = render_html_report(csv_path, group_keys=group_keys, budget=status)
        except Exception as exc:  # noqa: BLE001   reporting must never fail the run
            write(f"[elyaeval] could not render HTML for {csv_path}: {exc}")
            continue

        write(f"[elyaeval] {csv_path} -> {html_path.name}")
        rows = read_csv_rows(csv_path)
        print_metric_averages(rows, group_keys=group_keys, write=write)
    if status is not None:
        write("")
        for line in _budget_lines(status):
            write(f"[elyaeval] {line}")
    write("")
