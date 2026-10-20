"""
Shared HTML rendering + per-metric averaging, used by both report.py
(flat e2e CSVs) and tracing_report.py (span-aware component CSVs).

Score/threshold/success come back from csv.DictReader as strings   every
function below that touches them numerically goes through _coerce_row()
first so a blank score (e.g. an errored metric) doesn't crash averaging,
it's just excluded from that metric's mean.

Styling lives in report_theme.css, next to this file, rather than as an
inline string here that's used by BOTH renderers below (and any future
one) but is checked into the repo as a normal, diffable stylesheet
instead of a Python string. _load_css() reads it once per process and
inlines it into every report's <style> tag, so the *output* HTML stays a
single self-contained file (no external <link>, safe to open from a bare
blob-storage download) while the *source* stays one file per concern.
"""

from __future__ import annotations

import csv
import html as _html
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .compare import DEFAULT_REGRESSION_TOLERANCE, compare_runs, per_golden_scores
from .stats import bootstrap_ci, within_golden_sd

__all__ = [
    "DEFAULT_REGRESSION_TOLERANCE",
    "compare_runs",
    "metric_averages",
    "print_comparison",
    "print_metric_averages",
    "read_csv_rows",
    "render_comparison_html",
    "render_html_report",
    "total_cost",
    "total_tokens",
]

_THEME_CSS_PATH = Path(__file__).with_name("report_theme.css")


@lru_cache(maxsize=1)
def _load_css() -> str:
    """Read report_theme.css once per process and cache it. Falls back to
    a minimal inline stylesheet (rather than raising) if the file is ever
    missing next to this module   a report that renders in plain-but-
    readable CSS beats a broken pytest run over a stylesheet."""
    try:
        return _THEME_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return (
            "body{font-family:sans-serif;margin:2rem;}"
            "table{border-collapse:collapse;width:100%;}"
            "th,td{border:1px solid #ccc;padding:.4rem;text-align:left;}"
        )


# Populated by report.append_csv_rows / tracing_report.append_traced_csv_rows
# as they write every CSV path either module has EVER appended a row to
# during this process, paired with the group_keys its rows should be
# averaged by. plugin.py reads this at pytest_sessionfinish to know which
# CSVs to render HTML for and print averages for, without needing the
# generated test file itself to call anything new.

_SESSION_REPORTS: dict[Path, tuple[str, ...]] = {}


def register_report(csv_path: str | Path, group_keys: tuple[str, ...] = ("metric_name",)) -> None:
    _SESSION_REPORTS[Path(csv_path)] = group_keys


def clear_session_reports() -> None:
    """Forget every registered CSV — called by the plugin at session start
    so a second pytest session in the same process starts clean."""
    _SESSION_REPORTS.clear()


def session_reports() -> dict[Path, tuple[str, ...]]:
    """Snapshot of every CSV path registered so far in this process, e.g.
    for plugin.py's pytest_sessionfinish hook to iterate over."""
    return dict(_SESSION_REPORTS)


def read_csv_rows(csv_path: str | Path) -> list[dict]:
    """Read a report CSV back into a list of dicts, coercing score/threshold/
    evaluation_cost to float, input_tokens/output_tokens to int, and
    success to bool where possible. Rows are returned in file order, which is also the order the detail
    table renders in."""
    path = Path(csv_path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        # undo results_csv.escape_cell (spreadsheet formula-injection guard);
        # inlined rather than imported because results_csv imports this module
        for key, value in row.items():
            if isinstance(value, str) and value[:1] == "'":
                row[key] = value[1:]
        row["score"] = _to_float(row.get("score"))
        row["threshold"] = _to_float(row.get("threshold"))
        row["success"] = _to_bool(row.get("success"))
        row["evaluation_cost"] = _to_float(row.get("evaluation_cost"))
        row["input_tokens"] = _to_int(row.get("input_tokens"))
        row["output_tokens"] = _to_int(row.get("output_tokens"))
        row["repeat"] = _to_int(row.get("repeat")) or 0
        row["cached"] = bool(_to_bool(row.get("cached")))
    return rows


def _to_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    return str(value).strip().lower() == "true"


def metric_averages(rows: list[dict], group_keys: tuple[str, ...] = ("metric_name",)) -> list[dict]:
    """
      - n:            how many rows landed in this group (goldens tested)
      - n_scored:     how many of those had a numeric score
      - avg_score:    mean of the numeric scores, or None if n_scored == 0
      - min_score / max_score: same population
      - pass_count / pass_rate: from the `success` column (True/False),
                      independent of avg_score
      - threshold:    the threshold value, IF every row in the group shares
                      the same one (the normal case a metric's threshold
                      is fixed per suite); None if goldens in this group
                      used different thresholds, so a caller can't silently
                      display one that isn't actually representative
      - total_cost:   sum of evaluation_cost (USD) across every row in the
                      group that had a cost value. None (not 0) if NO row
                      in the group had a cost
      - cost_n:       how many rows actually contributed a cost value
                      lets a caller tell "$0.0031 from 4 metrics" apart
                      from "$0.0031 from 1 metric, other 3 unpriced"
      - evaluation_model: the judge model name,
      - total_input_tokens / total_output_tokens: sum of input_tokens/
                      output_tokens across every row in the group that had
                      a value. Same None-means-unknown convention as
                      total_cost (DeepEval only started reporting these at
                      all in 4.2, and even then only for judges whose
                      provider actually returns token usage, so an older
                      DeepEval version or an unpriced/local judge both
                      leave this None rather than 0).
      - tokens_n:     how many rows actually contributed a token count
                      same purpose as cost_n, for the same reason


    Returns groups in first-seen order (matches CSV/golden run order)
    """
    order: list[tuple] = []
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)

    summaries = []
    for key in order:
        group_rows = buckets[key]
        scores = [r["score"] for r in group_rows if r.get("score") is not None]
        successes = [r["success"] for r in group_rows if r.get("success") is not None]
        thresholds = {r["threshold"] for r in group_rows if r.get("threshold") is not None}
        costs = [r["evaluation_cost"] for r in group_rows if r.get("evaluation_cost") is not None]
        models = {r["evaluation_model"] for r in group_rows if r.get("evaluation_model")}
        input_tokens_vals = [r["input_tokens"] for r in group_rows if r.get("input_tokens") is not None]
        output_tokens_vals = [r["output_tokens"] for r in group_rows if r.get("output_tokens") is not None]

        summary = dict(zip(group_keys, key, strict=True))
        summary["n"] = len(group_rows)
        summary["n_scored"] = len(scores)
        summary["avg_score"] = (sum(scores) / len(scores)) if scores else None
        summary["min_score"] = min(scores) if scores else None
        summary["max_score"] = max(scores) if scores else None
        summary["pass_count"] = sum(1 for s in successes if s)
        summary["pass_total"] = len(successes)
        summary["pass_rate"] = (summary["pass_count"] / summary["pass_total"]) if successes else None
        summary["threshold"] = thresholds.pop() if len(thresholds) == 1 else None
        summary["total_cost"] = sum(costs) if costs else None
        summary["cost_n"] = len(costs)
        summary["evaluation_model"] = (
            models.pop() if len(models) == 1 else ("multiple" if len(models) > 1 else None)
        )
        summary["total_input_tokens"] = sum(input_tokens_vals) if input_tokens_vals else None
        summary["total_output_tokens"] = sum(output_tokens_vals) if output_tokens_vals else None
        summary["tokens_n"] = len(input_tokens_vals) or len(output_tokens_vals)
        # uncertainty: bootstrap CI over per-golden means (repeats averaged
        # first, so repeats don't masquerade as extra goldens), and the
        # within-golden SD across repeats = run-to-run noise floor
        by_golden = per_golden_scores(group_rows)
        golden_means = [sum(v) / len(v) for v in by_golden.values()]
        summary["n_goldens"] = len(by_golden)
        summary["ci_low"], summary["ci_high"] = bootstrap_ci(golden_means)
        summary["noise_sd"] = within_golden_sd(by_golden)
        summaries.append(summary)
    return summaries


def total_cost(rows: list[dict]) -> Optional[float]:
    """Sum of evaluation_cost (USD) across ALL rows, independent of
    grouping   the single "what did this whole run cost the judge model"
    number for the HTML header / terminal summary. None if no row in the
    run has a cost value at all (e.g. a local/unpriced judge model, same
    case DeepEval's own terminal summary shows as "token cost: None"
    rather than $0), not conflating "unknown" with "free"."""
    costs: list[float] = [r["evaluation_cost"] for r in rows if r.get("evaluation_cost") is not None]
    return sum(costs) if costs else None


def total_tokens(rows: list[dict]) -> tuple[Optional[int], Optional[int]]:
    """(total_input_tokens, total_output_tokens) across ALL rows,
    independent of grouping"""
    inputs: list[int] = [r["input_tokens"] for r in rows if r.get("input_tokens") is not None]
    outputs: list[int] = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
    return (sum(inputs) if inputs else None, sum(outputs) if outputs else None)


def overall_pass_rate(rows: list[dict]) -> Optional[float]:
    """Pass rate across ALL rows, independent of grouping   the single
    headline number for the report's KPI strip. None if no row has a
    `success` value at all."""
    successes = [r.get("success") for r in rows if r.get("success") is not None]
    return (sum(1 for s in successes if s) / len(successes)) if successes else None


def _esc(value) -> str:
    return _html.escape("" if value is None else str(value))


def _fmt_score(value: Optional[float]) -> str:
    return " " if value is None else f"{value:.2f}"


def _fmt_pct(value: Optional[float]) -> str:
    return " " if value is None else f"{value * 100:.0f}%"


def _fmt_cost(value: Optional[float]) -> str:
    """USD cost values from LLM judges are often fractions of a cent, so a
    flat 2-decimal format would show "$0.00" for almost everything that
    isn't a large batch. Uses more precision for small-but-nonzero values,
    2 decimals once the amount is large enough for that to be meaningful."""
    if value is None:
        return " "
    if value == 0:
        return "$0.00"
    if abs(value) < 0.01:
        return f"${value:.6f}"
    return f"${value:.4f}"


def _fmt_tokens(value: Optional[int]) -> str:
    """Thousands-separated integer, " " if None (unknown/unavailable
    see total_tokens()'s docstring), never "0" standing in for unknown."""
    return " " if value is None else f"{value:,}"


def _score_bar(value: Optional[float], threshold: Optional[float]) -> str:
    if value is None:
        return " "
    pct = max(0.0, min(1.0, value))
    low = threshold is not None and value < threshold
    return (
        f'<div class="bar-cell"><span class="num">{_fmt_score(value)}</span>'
        f'<div class="bar-track"><div class="bar-fill{" low" if low else ""}" '
        f'style="width:{pct * 100:.0f}%"></div></div></div>'
    )


def _pass_badge(success: Optional[bool]) -> str:
    if success is True:
        return '<span class="badge pass">● pass</span>'
    if success is False:
        return '<span class="badge fail">● fail</span>'
    return " "


def _brand_header(eyebrow: str, title_html: str, pills: list[str]) -> str:
    """Shared header block for every report type: the italic wordmark +
    pink dot (echoes the site's brand mark), an eyebrow label, the report
    title, and a row of meta pills (source file, row count, cost, ...)."""
    pills_html = "".join(pills)
    return f"""<div class="report-header">
  <div class="report-titleblock">
    <div class="brand"><span class="dot"></span>elyaeval</div>
    <h1>{title_html}</h1>
  </div>
  <div class="meta-pills">{pills_html}</div>
</div>"""


def _kpi_card(tag: str, color: str, value: str, caption: str) -> str:
    return f"""<div class="kpi-card {color}">
  <span class="kpi-tag"><span class="dot"></span>{_esc(tag)}</span>
  <div class="kpi-value">{value}</div>
  <div class="kpi-caption">{_esc(caption)}</div>
</div>"""


def _summary_table(rows: list[dict], group_keys: tuple[str, ...]) -> str:
    summaries = metric_averages(rows, group_keys=group_keys)
    if not summaries:
        return "<p><em>No scored metrics in this report.</em></p>"

    headers = [k.replace("_", " ").title() for k in group_keys] + [
        "Goldens",
        "Avg score",
        "95% CI",
        "Noise σ",
        "Min",
        "Max",
        "Pass rate",
        "Threshold",
        "Judge model",
        "Total cost",
        "Input tokens",
        "Output tokens",
    ]
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)

    body_rows = []
    for s in summaries:
        key_cells = "".join(f"<td>{_esc(s[k])}</td>" for k in group_keys)
        cost_note = f" ({s['cost_n']}/{s['n']} priced)" if s["cost_n"] and s["cost_n"] < s["n"] else ""
        tokens_note = f" ({s['tokens_n']}/{s['n']} reported)" if s["tokens_n"] and s["tokens_n"] < s["n"] else ""
        body_rows.append(
            "<tr>"
            f"{key_cells}"
            f'<td class="num">{s["n_goldens"] or s["n"]}</td>'
            f'<td>{_score_bar(s["avg_score"], s["threshold"])}</td>'
            f'<td class="num">{_fmt_ci(s["ci_low"], s["ci_high"])}</td>'
            f'<td class="num">{_fmt_score(s["noise_sd"])}</td>'
            f'<td class="num">{_fmt_score(s["min_score"])}</td>'
            f'<td class="num">{_fmt_score(s["max_score"])}</td>'
            f'<td class="num">{_fmt_pct(s["pass_rate"])} '
            f'<span class="reason">({s["pass_count"]}/{s["pass_total"]})</span></td>'
            f'<td class="num">{_fmt_score(s["threshold"])}</td>'
            f'<td>{_esc(s["evaluation_model"] or " ")}</td>'
            f'<td class="num">{_fmt_cost(s["total_cost"])}'
            f'<span class="reason">{cost_note}</span></td>'
            f'<td class="num">{_fmt_tokens(s["total_input_tokens"])}'
            f'<span class="reason">{tokens_note}</span></td>'
            f'<td class="num">{_fmt_tokens(s["total_output_tokens"])}</td>'
            "</tr>"
        )

    return (
        '<div class="table-wrap"><table class="summary-table">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _detail_table(rows: list[dict], extra_columns: tuple[str, ...] = ()) -> str:
    if not rows:
        return "<p><em>No rows in this report.</em></p>"

    base_cols = ["golden_id", "priority"] + list(extra_columns) + ["input", "metric_name"]
    headers = [c.replace("_", " ").title() for c in base_cols[:-1]] + [
        "Output",
        "Metric Name",
        "Score",
        "Threshold",
        "Result",
        "Judge model",
        "Cost",
        "In tokens",
        "Out tokens",
        "Reason / error",
    ]
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)

    body_rows = []
    for r in rows:
        cells = "".join(f"<td>{_esc(r.get(c, ''))}</td>" for c in base_cols[:-2])
        input_val = _esc(r.get("input", ""))
        output = str(r.get("actual_output") or "")
        output_val = _esc(output if len(output) <= 400 else output[:400] + "…")
        metric_val = _esc(r.get("metric_name", ""))
        reason = r.get("error") or r.get("reason") or ""
        reason_class = "error-cell" if r.get("error") else "reason"
        body_rows.append(
            "<tr>"
            f"{cells}"
            f'<td class="input">{input_val}</td>'
            f'<td class="input">{output_val}</td>'
            f"<td>{metric_val}</td>"
            f'<td class="num">{_fmt_score(r.get("score"))}</td>'
            f'<td class="num">{_fmt_score(r.get("threshold"))}</td>'
            f"<td>{_pass_badge(r.get('success'))}</td>"
            f'<td>{_esc(r.get("evaluation_model") or " ")}</td>'
            f'<td class="num">{_fmt_cost(r.get("evaluation_cost"))}</td>'
            f'<td class="num">{_fmt_tokens(r.get("input_tokens"))}</td>'
            f'<td class="num">{_fmt_tokens(r.get("output_tokens"))}</td>'
            f'<td class="{reason_class}">{_esc(reason)}</td>'
            "</tr>"
        )

    return (
        '<div class="table-wrap"><table class="detail-table">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _cost_card(run_cost: Optional[float], budget: Optional[dict]) -> str:
    spent = (budget or {}).get("spent") or {}
    if run_cost is None and spent.get("cost_usd") is not None:
        source = f"judge {spent.get('judge_cost_source')}, SUT {spent.get('sut_cost_source')}"
        return _kpi_card("Cost", "pink", _fmt_cost(spent["cost_usd"]), f"Run total ({source})")
    caption = "Total for this run" if run_cost is not None else "Not reported by the judge provider"
    return _kpi_card("Judge cost", "pink", _fmt_cost(run_cost), caption)


def _usage_card(run_input_tokens: Optional[int], run_output_tokens: Optional[int], budget: Optional[dict]) -> str:
    if not budget:
        return _kpi_card(
            "Tokens", "neutral", f"{_fmt_tokens(run_input_tokens)} / {_fmt_tokens(run_output_tokens)}", "Input / output"
        )
    spent, limits = budget.get("spent") or {}, budget.get("limits") or {}
    minutes = (spent.get("seconds") or 0) / 60
    value = f"{minutes:.1f} min"
    if limits.get("seconds"):
        value += f" / {limits['seconds'] / 60:.0f}"
    caption = f"{spent.get('judge_calls', 0):,} judge calls · {_fmt_tokens(spent.get('tokens'))} tokens ({spent.get('tokens_source')})"
    if budget.get("exhausted"):
        return _kpi_card("Budget exhausted", "pink", value, f"{len(budget.get('skipped') or [])} golden run(s) skipped · {caption}")
    tag = "Budget" if any(v is not None for v in limits.values()) else "Usage"
    return _kpi_card(tag, "neutral", value, caption)


def render_html_report(
    csv_path: str | Path,
    html_path: Optional[str | Path] = None,
    title: Optional[str] = None,
    group_keys: tuple[str, ...] = ("metric_name",),
    budget: Optional[dict] = None,
) -> Path:
    """
    Build a single self-contained HTML file (inline CSS, no external
    assets/CDN calls   safe to open straight from a blob-storage download
    with no server behind it) next to `csv_path`, showing:

      1. a KPI strip: goldens tested, overall pass rate, total judge cost,
         total tokens   the four numbers someone checking CI wants first;
      2. a per-metric (or per-`group_keys`) summary table: average score,
         min/max, pass rate, and the threshold, computed once across every
         golden that ran in this suite;
      3. the full per-golden/per-metric detail table the CSV already has,
         with pass/fail badges and a score bar per row.

    `html_path` defaults to the same filename as `csv_path` with a .html
    extension, so an uploaded CSV and its HTML sibling are always easy to
    pair up by name (report-results in tasks.yaml relies on exactly this).
    `group_keys` should be ("span_name", "metric_name") for a component
    CSV (tracing_report.py) so metrics repeated under different spans get
    their own summary row instead of being averaged together.
    """
    csv_path = Path(csv_path)
    out_path = Path(html_path) if html_path else csv_path.with_suffix(".html")
    rows = read_csv_rows(csv_path)

    extra_columns = tuple(k for k in group_keys if k != "metric_name")
    report_title = title or csv_path.stem
    run_cost = total_cost(rows)
    run_input_tokens, run_output_tokens = total_tokens(rows)
    run_pass_rate = overall_pass_rate(rows)
    n_goldens = len({r.get("golden_id") for r in rows if r.get("golden_id")}) or len(rows)

    header = _brand_header(
        eyebrow="Evaluation report",
        title_html=_esc(report_title),
        pills=[
            f'<span class="pill">Source <strong>{_esc(csv_path.name)}</strong></span>',
            f'<span class="pill">Rows <strong>{len(rows)}</strong></span>',
        ],
    )

    kpi_html = (
        '<div class="kpi-grid">'
        + _kpi_card("Goldens", "blue", str(n_goldens), "Tested this run")
        + _kpi_card(
            "Pass rate", "cyan",
            _fmt_pct(run_pass_rate),
            "Across every scored row",
        )
        + _cost_card(run_cost, budget)
        + _usage_card(run_input_tokens, run_output_tokens, budget)
        + "</div>"
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(report_title)}</title>
<style>{_load_css()}</style>
</head>
<body>
<div class="report-shell">
{header}
{kpi_html}
<h2 class="section-label">Per-metric summary</h2>
{_summary_table(rows, group_keys)}
<h2 class="section-label pink">Per-golden detail</h2>
{_detail_table(rows, extra_columns)}
<div class="report-footer">Generated by elyaeval &middot; {_esc(csv_path.name)}</div>
</div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def print_metric_averages(
    rows: list[dict],
    group_keys: tuple[str, ...] = ("metric_name",),
    write=print,
) -> None:
    """Plain-text version of the summary table. `write` defaults to the
    builtin print(); pass a pytest TerminalReporter's write_line (or any
    other str -> None callable) to route it through pytest's own output
    machinery instead   plugin.py does this so the summary shows up
    reliably even under output capturing that a bare print() at
    sessionfinish time could otherwise get swallowed by."""
    summaries = metric_averages(rows, group_keys=group_keys)
    if not summaries:
        return
    label_width = max(
        (len(" / ".join(str(s[k]) for k in group_keys)) for s in summaries), default=10
    )
    label_width = max(label_width, len("metric"))
    write(
        f"{'metric':<{label_width}}  {'avg':>6}  {'95% CI':>12}  {'noise':>5}  {'min':>6}  {'max':>6}  "
        f"{'pass':>10}  {'cost':>10}  {'in tok':>10}  {'out tok':>10}  n"
    )
    write("-" * (label_width + 95))
    for s in summaries:
        label = " / ".join(str(s[k]) for k in group_keys)
        pass_str = f"{s['pass_count']}/{s['pass_total']}" if s["pass_total"] else " "
        write(
            f"{label:<{label_width}}  {_fmt_score(s['avg_score']):>6}  "
            f"{_fmt_ci(s['ci_low'], s['ci_high']):>12}  {_fmt_score(s['noise_sd']):>5}  "
            f"{_fmt_score(s['min_score']):>6}  {_fmt_score(s['max_score']):>6}  "
            f"{pass_str:>10}  {_fmt_cost(s['total_cost']):>10}  "
            f"{_fmt_tokens(s['total_input_tokens']):>10}  {_fmt_tokens(s['total_output_tokens']):>10}  {s['n']}"
        )
    run_input_tokens, run_output_tokens = total_tokens(rows)
    write(f"\nTotal judge cost for this run: {_fmt_cost(total_cost(rows))}")
    write(f"Total tokens for this run: {_fmt_tokens(run_input_tokens)} in / {_fmt_tokens(run_output_tokens)} out")


# ---------------------------------------------------------------------------
# Regression comparison   the engine lives in compare.py (paired, statistical
# gate, error/pass-rate gates); re-exported here for backward compatibility.
# ---------------------------------------------------------------------------


def _fmt_ci(low: Optional[float], high: Optional[float], signed: bool = False) -> str:
    if low is None or high is None:
        return " "
    fmt = "{:+.3f}" if signed else "{:.2f}"
    return f"[{fmt.format(low)}, {fmt.format(high)}]"


def _fmt_p(p: Optional[float]) -> str:
    return " " if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


_MARKERS = {
    "regressed": "▼ REGRESSED",
    "improved": "▲ improved",
    "unchanged": "= unchanged",
    "new": "+ new",
    "removed": "- removed",
}


def print_comparison(
    comparisons: list[dict],
    group_keys: tuple[str, ...] = ("metric_name",),
    write=print,
    compatibility: Optional[dict[str, list[str]]] = None,
) -> None:
    """Plain-text regression table   what `elyaeval compare` prints."""
    compatibility = compatibility or {}
    for note in compatibility.get("errors", []):
        write(f"✗ incompatible runs: {note}")
    for note in compatibility.get("warnings", []):
        write(f"! {note}")
    for note in compatibility.get("sut_changes", []):
        write(f"~ SUT change: {note}")
    if any(compatibility.values()):
        write("")
    if not comparisons:
        write("No metrics in either run.")
        return
    label_width = max(
        (len(" / ".join(str(c[k]) for k in group_keys)) for c in comparisons), default=10
    )
    label_width = max(label_width, len("metric"))
    write(
        f"{'metric':<{label_width}}  {'baseline':>8}  {'candidate':>9}  {'delta':>7}  "
        f"{'95% CI of delta':>17}  {'p':>6}  {'pairs':>5}  {'err Δ':>6}  status"
    )
    write("-" * (label_width + 86))
    for c in comparisons:
        label = " / ".join(str(c[k]) for k in group_keys)
        delta_str = " " if c["delta"] is None else f"{c['delta']:+.3f}"
        err = c.get("error_rate_delta")
        err_str = " " if err is None else f"{err:+.0%}"
        ci = _fmt_ci(c.get("ci_low"), c.get("ci_high"), signed=True)
        write(
            f"{label:<{label_width}}  {_fmt_score(c['baseline_avg']):>8}  "
            f"{_fmt_score(c['candidate_avg']):>9}  {delta_str:>7}  "
            f"{ci:>17}  {_fmt_p(c.get('p_value')):>6}  "
            f"{c.get('n_paired', 0):>5}  {err_str:>6}  {_MARKERS[c['status']]}"
        )
        for reason in c.get("reasons", []):
            write(f"{'':<{label_width}}    -> {reason}")
    unpaired = [c for c in comparisons if c.get("only_baseline") or c.get("only_candidate")]
    if unpaired:
        first = unpaired[0]
        write(
            f"\nNote: some goldens have a score in only one run (not run, or errored: e.g. "
            f"{first.get('only_baseline', 0)} baseline-only, {first.get('only_candidate', 0)} candidate-only); "
            "only goldens scored in both runs are compared."
        )
    methods = sorted({c["method"] for c in comparisons if c.get("method") and c["method"] != "n/a"})
    if methods:
        write(f"Method: {'; '.join(methods)}")
    n_regressed = sum(1 for c in comparisons if c["status"] == "regressed")
    write(f"\n{n_regressed} metric(s) regressed." if n_regressed else "\nNo regressions.")


_STATUS_BADGE = {
    "regressed": '<span class="badge fail">▼ regressed</span>',
    "improved": '<span class="badge pass">▲ improved</span>',
    "unchanged": '<span class="badge neutral">= unchanged</span>',
    "new": '<span class="badge cyan">+ new</span>',
    "removed": '<span class="badge pink">− removed</span>',
}


def render_comparison_html(
    comparisons: list[dict],
    group_keys: tuple[str, ...] = ("metric_name",),
    html_path: Optional[str | Path] = None,
    title: str = "Regression comparison",
    tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
    compatibility: Optional[dict[str, list[str]]] = None,
) -> Optional[Path]:
    """HTML version of print_comparison. Returns None (writes nothing) if
    no html_path given   pass one explicitly, there's no CSV to infer a
    path from here since this compares two of them."""
    headers = [k.replace("_", " ").title() for k in group_keys] + [
        "Baseline avg", "Candidate avg", "Delta", "95% CI of delta", "p", "Pairs",
        "Baseline pass", "Candidate pass", "Error rate Δ", "Status",
    ]
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_rows = []
    for c in comparisons:
        key_cells = "".join(f"<td>{_esc(c[k])}</td>" for k in group_keys)
        delta_str = " " if c["delta"] is None else f"{c['delta']:+.3f}"
        err = c.get("error_rate_delta")
        err_str = " " if err is None else f"{err:+.0%}"
        reasons = "".join(f'<div class="reason">{_esc(r)}</div>' for r in c.get("reasons", []))
        body_rows.append(
            "<tr>"
            f"{key_cells}"
            f'<td class="num">{_fmt_score(c["baseline_avg"])}</td>'
            f'<td class="num">{_fmt_score(c["candidate_avg"])}</td>'
            f'<td class="num">{delta_str}</td>'
            f'<td class="num">{_fmt_ci(c.get("ci_low"), c.get("ci_high"), signed=True)}</td>'
            f'<td class="num">{_fmt_p(c.get("p_value"))}</td>'
            f'<td class="num">{c.get("n_paired", 0)}</td>'
            f'<td class="num">{_fmt_pct(c["baseline_pass_rate"])}</td>'
            f'<td class="num">{_fmt_pct(c["candidate_pass_rate"])}</td>'
            f'<td class="num">{err_str}</td>'
            f'<td>{_STATUS_BADGE[c["status"]]}{reasons}</td>'
            "</tr>"
        )
    table_html = (
        '<div class="table-wrap"><table class="detail-table">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )

    notes = []
    for css, key, prefix in (
        ("error-cell", "errors", "Incompatible runs: "),
        ("reason", "warnings", ""),
        ("reason", "sut_changes", "SUT change: "),
    ):
        notes += [f'<li class="{css}">{_esc(prefix + n)}</li>' for n in (compatibility or {}).get(key, [])]
    notes_html = f'<h2 class="section-label">Run metadata</h2><ul>{"".join(notes)}</ul>' if notes else ""

    n_regressed = sum(1 for c in comparisons if c["status"] == "regressed")
    n_improved = sum(1 for c in comparisons if c["status"] == "improved")
    methods = sorted({c["method"] for c in comparisons if c.get("method") and c["method"] != "n/a"})

    header = _brand_header(
        eyebrow="Regression comparison",
        title_html=_esc(title),
        pills=[
            f'<span class="pill">Metrics compared <strong>{len(comparisons)}</strong></span>',
            f'<span class="pill">Tolerance <strong>{tolerance:g}</strong></span>',
            *[f'<span class="pill">{_esc(m)}</span>' for m in methods],
        ],
    )

    verdict_kpi = (
        _kpi_card("Regressed", "pink", str(n_regressed), f"Beyond tolerance ({tolerance:g})")
        if n_regressed
        else _kpi_card("Regressed", "cyan", "0", "No regressions beyond tolerance")
    )
    kpi_html = (
        '<div class="kpi-grid">'
        + verdict_kpi
        + _kpi_card("Improved", "blue", str(n_improved), "Beyond tolerance")
        + _kpi_card("Compared", "neutral", str(len(comparisons)), "Metric / group rows")
        + "</div>"
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_load_css()}</style>
</head>
<body>
<div class="report-shell">
{header}
{kpi_html}
{notes_html}
<h2 class="section-label blue">Per-metric comparison</h2>
{table_html}
<div class="report-footer">Generated by elyaeval</div>
</div>
</body>
</html>
"""
    if html_path is None:
        return None
    out_path = Path(html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
