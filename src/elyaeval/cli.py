"""
`elyaeval init`   generates a filled-in pytest file for a chosen task type,
using the standard dataset + the matching metric preset.

`elyaeval init-ci`   generates a per-repo Tekton PipelineRun stub that wires
this repo into the shared `elyaeval` Tekton Pipeline.

`elyaeval ci`   render/apply the Tekton recipe shipped with this version.

`elyaeval judge`   shows which judge model the current config resolves to
(and, with --ping, checks it answers a schema-constrained request).
"""

import argparse
from importlib import resources
from pathlib import Path
from typing import Optional

from elyaeval.core.dataset import load_standard_dataset
from elyaeval.core.stages import STAGES
from elyaeval.reporting.html_report import (
    DEFAULT_REGRESSION_TOLERANCE,
    compare_runs,
    print_comparison,
    print_metric_averages,
    read_csv_rows,
    render_comparison_html,
    render_html_report,
)

# task_type -> metric factory name in elyaeval.core.metrics (kept in sync
# with metrics.TASK_TYPE_FACTORIES; duplicated so `elyaeval init` doesn't
# import deepeval metrics just to render a template)
_TASK_TYPE_TO_METRICS_FACTORY = {
    "rag_qa": "rag_metrics",
    "retrieval": "retrieval_metrics",
    "summarization": "summarization_metrics",
    "qa_correctness": "qa_correctness_metrics",
    "agentic_action": "agentic_action_metrics",
    "agentic_trajectory": "agentic_trajectory_metrics",
}


_CORPUS_MODE_TO_TEMPLATE = {
    "existing": "test_template.py.tmpl",
    "seeded": "test_template_seeded.py.tmpl",
}

# Task types with no retrieval leg at all   --corpus-mode doesn't apply
# (there's no corpus to point at "existing" vs "seed")
_TASK_TYPES_WITHOUT_CORPUS_MODE = {
    "agentic_action": "test_template_agentic.py.tmpl",
    "agentic_trajectory": "test_template_agentic_trajectory.py.tmpl",
}

# --eval-mode is orthogonal to --corpus-mode: it picks e2e (HTTP, one flat
# TestResult per golden, via report.py/evaluate_golden) vs component
# (in-process, span-aware, via tracing_report.py/TracedRunner). "e2e" is
# the existing behavior and stays the default. Component mode has exactly one template regardless of task
# type, it runs the SUT's traced entrypoint in-process rather than
# retrieving from a corpus, so --corpus-mode doesn't apply to it either
_COMPONENT_TEMPLATE = "test_template_component.py.tmpl"

_CI_TEMPLATE = "pipelinerun_template.yaml.tmpl"


def _load_template(filename: str) -> str:
    return (resources.files("elyaeval") / "templates" / filename).read_text()


def _template_filename(task_type: str, corpus_mode: str, eval_mode: str) -> str:
    if eval_mode == "component":
        return _COMPONENT_TEMPLATE
    if task_type in _TASK_TYPES_WITHOUT_CORPUS_MODE:
        return _TASK_TYPES_WITHOUT_CORPUS_MODE[task_type]
    return _CORPUS_MODE_TO_TEMPLATE[corpus_mode]


def _default_test_filename(task_type: str, eval_mode: str = "e2e") -> str:
    """The naming convention every generated test file follows, the single
    place that convention is defined, so init() and init_ci() can't drift
    out of sync on what a task type's default filename looks like."""
    suffix = "_component" if eval_mode == "component" else ""
    return f"test_elyaeval_{task_type}{suffix}.py"


def _load_ci_template() -> str:
    return (resources.files("elyaeval") / "templates" / _CI_TEMPLATE).read_text()


def init(
    task_type: str,
    ci_stage: str,
    output: str,
    corpus_mode: str = "existing",
    eval_mode: str = "e2e",
) -> Path:
    if task_type not in _TASK_TYPE_TO_METRICS_FACTORY:
        valid = ", ".join(_TASK_TYPE_TO_METRICS_FACTORY)
        raise SystemExit(f"Unknown --task-type '{task_type}'. Valid options: {valid}")

    if ci_stage not in STAGES:
        raise SystemExit(f"Unknown --ci-stage '{ci_stage}'. Valid options: {', '.join(STAGES)}")

    if eval_mode not in ("e2e", "component"):
        raise SystemExit(f"Unknown --eval-mode '{eval_mode}'. Valid options: e2e, component")

    if corpus_mode not in _CORPUS_MODE_TO_TEMPLATE:
        valid = ", ".join(_CORPUS_MODE_TO_TEMPLATE)
        raise SystemExit(f"Unknown --corpus-mode '{corpus_mode}'. Valid options: {valid}")

    if task_type in _TASK_TYPES_WITHOUT_CORPUS_MODE and corpus_mode != "existing":
        raise SystemExit(
            f"--corpus-mode is not applicable to task_type='{task_type}' "
            f"(no retrieval corpus involved)   omit --corpus-mode."
        )

    if eval_mode == "component" and task_type in _TASK_TYPES_WITHOUT_CORPUS_MODE:
        raise SystemExit(
            f"--eval-mode component doesn't apply to {task_type}: agent suites have their own "
            f"templates (agentic_action = tool calls, agentic_trajectory = whole trace, in-process)."
        )

    if eval_mode == "component" and corpus_mode != "existing":
        raise SystemExit(
            "--corpus-mode is not applicable to --eval-mode=component   component mode "
            "always runs your app's traced entrypoint in-process against whatever corpus "
            "it already has (seeding doesn't apply here). Omit --corpus-mode."
        )

    n_matched = len(load_standard_dataset(task_type=task_type, ci_stage=ci_stage))
    if n_matched == 0:
        print(
            f"Warning: 0 goldens in the standard dataset match "
            f"task_type='{task_type}' and ci_stage='{ci_stage}'. "
            f"The generated file will collect but every test will be skipped by pytest. "
            f"Check `elyaeval.core.dataset.load_standard_dataset` filters or pass a different --ci-stage."
        )

    if eval_mode == "component":
        print(
            "Note: generating in component mode; this suite calls your app's traced "
            "entrypoint IN-PROCESS (no HTTP) and scores each @observe'd span separately"
        )
    elif corpus_mode == "existing" and task_type not in _TASK_TYPES_WITHOUT_CORPUS_MODE:
        print(
            "Note: generating in existing-corpus mode, GOLDENS defaults to the shared "
            "standard dataset, which is a generic benchmark, not written against your "
            "app's corpus. Point GOLDENS at your own goldens file (see the comment in "
            "the generated file) for retrieval-dependent metrics to mean anything."
        )

    metrics_factory = _TASK_TYPE_TO_METRICS_FACTORY[task_type]
    template_filename = _template_filename(task_type, corpus_mode, eval_mode)
    rendered = _load_template(template_filename).format(
        task_type=task_type,
        ci_stage=ci_stage,
        metrics_factory=metrics_factory,
    )

    out_path = Path(output)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists, refusing to overwrite. Remove it or pass --output.")

    out_path.write_text(rendered)
    return out_path


def init_ci(
    repo_url: Optional[str],
    revision: str,
    blob_container: str,
    output: str,
    task_type: str = "rag_qa",
    test_file: Optional[str] = None,
    sut_path: str = ".",
    runner_image: Optional[str] = None,
    ci_stage: str = "",
    budget_seconds: str = "",
    timeout: str = "2h",
    namespace: str = "elyaeval",
    name_prefix: Optional[str] = None,
) -> Path:
    """Write this mission's PipelineRun for the shared `elyaeval` Pipeline.
    The runner image is pinned to the installed elyaeval version unless
    overridden, so the recipe, the image and the generated test file match."""
    from elyaeval import __version__
    from elyaeval.ci import default_runner_image

    if ci_stage and ci_stage not in STAGES:
        raise SystemExit(f"Unknown --ci-stage '{ci_stage}'. Valid options: {', '.join(STAGES)}")
    out_path = Path(output)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists, refusing to overwrite. Remove it or pass --output.")

    rendered = _load_ci_template().format(
        elyaeval_version=__version__,
        repo_url=repo_url or "${REPO_URL}",
        revision=revision,
        sut_path=sut_path,
        test_file=test_file or _default_test_filename(task_type),
        runner_image=runner_image or default_runner_image(),
        blob_container=blob_container,
        ci_stage=ci_stage,
        budget_seconds=budget_seconds,
        timeout=timeout,
        namespace=namespace,
        name_prefix=name_prefix or f"elyaeval-{blob_container}",
        output_name=out_path.name,
    )
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


def ci_command(action: str, output_dir: str, namespace: str, runner_image: Optional[str], dry_run: Optional[str]) -> int:
    from elyaeval import ci

    if action == "render":
        for path in ci.write_recipe(output_dir, runner_image):
            print(f"Wrote {path}")
        print(f"Runner image: {runner_image or ci.default_runner_image()}")
        return 0
    if action == "apply":
        print(f"Applying the elyaeval {ci.installed_version()} Pipeline/Tasks to namespace '{namespace}' "
              f"(runner image {runner_image or ci.default_runner_image()})")
        return ci.apply_recipe(namespace, runner_image, dry_run=dry_run)
    print(ci.dockerfile_text(), end="")
    return 0


def report(csv_path: str, html_path: Optional[str] = None, group_by: str = "metric_name") -> Path:
    """
    `elyaeval report --csv <path>` regenerate the HTML report + print
    per-metric averages for a CSV that already exists on disk, without
    re-running any tests. Meant for two cases the pytest-plugin path
    (elyaeval.reporting.plugin, runs automatically at session end) doesn't cover:

      - a CSV produced by an older elyaeval version, before HTML reports
        existed (e.g. one already sitting in blob storage from a past run)
      - regenerating on demand after downloading a CSV from CI locally

    --group-by defaults to "metric_name" (matches report.py's flat CSVs).
    Pass --group-by span_name,metric_name for a *_component.csv produced
    by tracing_report.py, so metrics repeated across spans get their own
    row instead of being averaged together.
    """
    src = Path(csv_path)
    if not src.exists():
        raise SystemExit(f"{src} does not exist.")

    keys = tuple(k.strip() for k in group_by.split(",") if k.strip())
    out_path = render_html_report(src, html_path=html_path, group_keys=keys)
    print(f"Wrote {out_path}")

    rows = read_csv_rows(src)
    print_metric_averages(rows, group_keys=keys)
    return out_path


def compare(
    baseline_csv: str,
    candidate_csv: str,
    tolerance: Optional[float] = None,
    group_by: str = "metric_name",
    html_path: Optional[str] = None,
    alpha: Optional[float] = None,
    min_paired: Optional[int] = None,
    gate: Optional[str] = None,
    max_error_rate_increase: Optional[float] = None,
    max_pass_rate_drop: Optional[float] = None,
    unpaired: bool = False,
    allow_metadata_mismatch: bool = False,
) -> int:
    """
    `elyaeval compare --baseline <csv> --candidate <csv>`   did this
    model/prompt/config change regress anything? Pairs goldens by id, gates
    on practical AND statistical significance (see elyaeval.reporting.compare),
    plus error-rate and (optionally) pass-rate gates. Gate defaults come from
    [tool.elyaeval.regression]; arguments here override them.

    Exit code: 0 nothing regressed, 1 something regressed, 2 the runs aren't
    comparable (different judge, per their .meta.json sidecars) unless
    allow_metadata_mismatch. Meant to be a CI step's exit code directly.
    """
    from elyaeval.reporting.compare import gate_config_from_settings
    from elyaeval.reporting.run_metadata import compatibility, read_sidecar

    baseline_path = Path(baseline_csv)
    candidate_path = Path(candidate_csv)
    if not baseline_path.exists():
        raise SystemExit(f"{baseline_path} does not exist.")
    if not candidate_path.exists():
        raise SystemExit(f"{candidate_path} does not exist.")

    gate_cfg = gate_config_from_settings(
        tolerance=tolerance,
        alpha=alpha,
        min_paired=min_paired,
        gate=gate,
        max_error_rate_increase=max_error_rate_increase,
        max_pass_rate_drop=max_pass_rate_drop,
        paired=False if unpaired else None,
    )
    compat = compatibility(read_sidecar(baseline_path), read_sidecar(candidate_path))

    keys = tuple(k.strip() for k in group_by.split(",") if k.strip())
    comparisons = compare_runs(read_csv_rows(baseline_path), read_csv_rows(candidate_path), group_keys=keys, gate=gate_cfg)

    print(f"Baseline:  {baseline_path}")
    print(f"Candidate: {candidate_path}")
    print(
        f"Gate:      {gate_cfg.gate}, tolerance ±{gate_cfg.tolerance:g}, alpha {gate_cfg.alpha:g}, "
        f"min pairs {gate_cfg.min_paired}, max error-rate increase {gate_cfg.max_error_rate_increase}, "
        f"max pass-rate drop {gate_cfg.max_pass_rate_drop}\n"
    )
    print_comparison(comparisons, group_keys=keys, compatibility=compat)

    if html_path:
        out = render_comparison_html(
            comparisons,
            group_keys=keys,
            html_path=html_path,
            title=f"{candidate_path.stem} vs {baseline_path.stem}",
            tolerance=gate_cfg.tolerance,
            compatibility=compat,
        )
        print(f"\nWrote {out}")

    if compat["errors"] and not allow_metadata_mismatch:
        print("\nRuns are not comparable (see above); pass --allow-metadata-mismatch to compare anyway.")
        return 2
    return 1 if any(c["status"] == "regressed" for c in comparisons) else 0


def calibrate(results_csv: str, labels_csv: Optional[str] = None, group_by: str = "metric_name") -> int:
    """`elyaeval calibrate`: noise-based tolerance recommendations (from a
    run with repeats) and judge-vs-human agreement (with --labels)."""
    from elyaeval.reporting.calibration import agreement_report, metric_key_for, noise_report, read_labels

    rows = read_csv_rows(results_csv)
    if not rows:
        raise SystemExit(f"{results_csv} has no rows.")
    keys = tuple(k.strip() for k in group_by.split(",") if k.strip())

    def fmt(v, spec=".3f"):
        return " " if v is None else format(v, spec)

    print("Run-to-run noise (needs --elyaeval-repeats >= 2)")
    noise = noise_report(rows, keys)
    width = max(len(" / ".join(str(r[k]) for k in keys)) for r in noise)
    print(f"  {'metric':<{width}}  {'goldens':>7}  {'repeats':>7}  {'noise sd':>8}  {'recommended tolerance':>21}")
    for r in noise:
        label = " / ".join(str(r[k]) for k in keys)
        print(f"  {label:<{width}}  {r['n_goldens']:>7}  {r['repeats']:>7}  {fmt(r['noise_sd']):>8}  {fmt(r['recommended_tolerance']):>21}")
    recs = {" / ".join(str(r[k]) for k in keys): r["recommended_tolerance"] for r in noise if r["recommended_tolerance"] is not None}
    if recs:
        print("\n  Suggested config (tolerances below these are within judge/SUT noise):")
        print("    [tool.elyaeval.regression]")
        print("    tolerances = { " + ", ".join(f'"{k}" = {v:.3f}' for k, v in recs.items()) + " }")
    else:
        print("  No golden was scored more than once: run with --elyaeval-repeats 3 to measure noise.")

    if labels_csv:
        labels = read_labels(labels_csv)
        agreement = agreement_report(rows, labels, keys)
        print("\nJudge vs human labels")
        if not agreement:
            print("  No overlap between labels and results (check golden_id / metric_name values).")
            return 1
        print(f"  {'metric':<{width}}  {'n':>4}  {'accuracy':>8}  {'kappa':>6}  {'judge pass':>10}  {'human pass':>10}  {'MAE':>5}  {'best thr':>8}")
        for r in agreement:
            label = " / ".join(str(r[k]) for k in keys)
            print(
                f"  {label:<{width}}  {r['n']:>4}  {fmt(r['accuracy'], '.0%'):>8}  {fmt(r['kappa'], '.2f'):>6}  "
                f"{fmt(r['judge_pass_rate'], '.0%'):>10}  {fmt(r['human_pass_rate'], '.0%'):>10}  "
                f"{fmt(r['mae'], '.2f'):>5}  {fmt(r['best_threshold'], '.2f'):>8}"
            )
        best = {metric_key_for(str(r.get("metric_name", ""))): r["best_threshold"] for r in agreement if r["best_threshold"] is not None}
        if best:
            print("\n  Thresholds that best match the human labels:")
            print("    [tool.elyaeval.thresholds]")
            for k, v in best.items():
                print(f"    {k} = {v:.2f}")
        low = [r for r in agreement if r["kappa"] is not None and r["kappa"] < 0.4]
        if low:
            print("\n  ! kappa < 0.4 (weak agreement) for: " + ", ".join(str(r.get("metric_name")) for r in low)
                  + "   don't gate on these with this judge.")
    return 0


def estimate(
    task_type: str,
    ci_stage: str,
    dataset_path: Optional[str] = None,
    repeats: int = 1,
    history: Optional[str] = None,
    budget_overrides: Optional[dict] = None,
) -> int:
    """`elyaeval estimate`: project a run's time/cost/tokens/judge calls
    BEFORE running it   from a previous run's budget ledger (most accurate:
    it reflects this judge's real speed/price) or, without history, from a
    rough token heuristic. Exit 1 if the projection exceeds the budget."""
    import json

    from elyaeval.core import budget as b
    from elyaeval.core.metrics import PRESETS

    goldens = load_standard_dataset(task_type=task_type, ci_stage=ci_stage, dataset_path=dataset_path)
    metric_keys = list(PRESETS[task_type])
    if history:
        ledger = json.loads(Path(history).read_text(encoding="utf-8"))
        ledger["_path"] = history
    else:
        ledger = b.latest_ledger("report")
    profile = b.profile_from_ledger(ledger) if ledger else b.heuristic_profile(goldens, metric_keys)
    projection = b.project(len(goldens), len(metric_keys), repeats, profile)
    cfg = b.budget_config_from_settings(**(budget_overrides or {}))

    source = f"history {ledger['_path']}" if ledger else "heuristic (no run history: tokens/calls only, rough)"
    print(f"Task type: {task_type}   stage: {ci_stage} (cumulative)   repeats: {repeats}")
    print(f"Goldens:   {len(goldens)}   metrics per golden: {len(metric_keys)} ({', '.join(metric_keys)})")
    print(f"Basis:     {source}\n")

    def show(dim, value):
        return "unknown" if value is None else b._fmt(dim, value)

    for dim, label in (("seconds", "time"), ("cost_usd", "cost"), ("tokens", "tokens"), ("judge_calls", "judge calls")):
        limit = cfg.limits().get(dim)
        print(f"  {label:<12} {show(dim, projection.get(dim)):>14}" + (f"   budget {show(dim, limit)}" if limit is not None else ""))
    over = b.projection_over_budget(projection, cfg)
    if over:
        print("\nOver budget:")
        for line in over:
            print(f"  ! {line}")
        print("  -> lower the stage/repeats, enable the result cache, or raise the budget.")
        return 1
    return 0


def budget_command(ledger_path: Optional[str] = None, fail_on_exhausted: bool = False) -> int:
    """`elyaeval budget`: summarize a run's budget ledger (default: the latest
    report/budget_*.json). With --fail-on-exhausted, exit 1 if the run hit
    its budget   what CI's check-budget step runs."""
    import json

    from elyaeval.core import budget as b

    if ledger_path:
        data = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
        data["_path"] = ledger_path
    else:
        data = b.latest_ledger("report")
    if not data:
        print("No budget ledger found (report/budget_*.json): nothing has been evaluated yet.")
        return 0
    status = data.get("budget") or {}
    spent, limits = status.get("spent") or {}, status.get("limits") or {}
    totals = data["totals"]
    print(f"Ledger:  {data['_path']}")
    print(f"Goldens: {totals.get('goldens_evaluated')} evaluated, {len(status.get('skipped') or [])} skipped by budget")
    for dim in ("seconds", "cost_usd", "tokens", "judge_calls"):
        value, limit = spent.get(dim), limits.get(dim)
        shown = "unknown" if value is None else b._fmt(dim, value)
        print(f"  {dim:<12} {shown:>14}" + (f"   budget {b._fmt(dim, limit)}" if limit is not None else ""))
    if spent:
        print(f"  cost source: judge {spent.get('judge_cost_source')}, SUT {spent.get('sut_cost_source')}; "
              f"tokens {spent.get('tokens_source')}")
    if status.get("exhausted"):
        print(f"\nBudget EXHAUSTED: {status.get('exhausted_reason')}")
        return 1 if fail_on_exhausted else 0
    return 0


def cache_command(action: str) -> int:
    from elyaeval.core.cache import ResultCache, cache_settings

    enabled, path = cache_settings()
    print(f"Cache:   {'enabled' if enabled else 'disabled'} ([tool.elyaeval.cache] / ELYAEVAL_CACHE)")
    print(f"Path:    {path}")
    if not path.exists():
        print("Entries: 0 (no cache file yet)")
        return 0
    cache = ResultCache(path)
    if action == "clear":
        print(f"Cleared {cache.clear()} entries.")
    else:
        print(f"Entries: {cache.count()}")
    return 0


def judge(ping: bool = False) -> int:
    """`elyaeval judge [--ping]`: print the resolved judge configuration
    (never the API key) and optionally make one tiny test call."""
    from elyaeval.core.judge import JudgeConfigError, ping_judge, resolve_judge_config

    try:
        cfg = resolve_judge_config()
    except JudgeConfigError as exc:
        print(f"Invalid judge configuration: {exc}")
        return 2
    print(f"Judge:       {cfg.describe()}")
    print(f"Provider:    {cfg.provider}")
    print(f"Model:       {cfg.model or '(provider default)'}")
    print(f"Base URL:    {cfg.base_url or '(provider default)'}")
    print(f"Temperature: {cfg.temperature}")
    if cfg.timeout_seconds is None:
        print("Timeout:     DeepEval default (180s per test case)")
    else:
        print(f"Timeout:     {'disabled' if cfg.timeout_seconds <= 0 else f'{cfg.timeout_seconds:g}s per test case'}")
    print(f"API key:     {'set' if cfg.api_key else '(provider env var / not needed)'}")
    if cfg.options:
        print(f"Options:     {dict(cfg.options)}")
    if not ping:
        return 0
    try:
        print(f"Ping:        {ping_judge()}")
    except Exception as exc:  # noqa: BLE001   report any provider failure readably
        print(f"Ping FAILED: {type(exc).__name__}: {exc}")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(prog="elyaeval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Generate a starter pytest file for a task type")
    p_init.add_argument(
        "--task-type",
        required=True,
        choices=list(_TASK_TYPE_TO_METRICS_FACTORY),
        help="Which use case to generate a test file for.",
    )
    p_init.add_argument(
        "--ci-stage",
        default="nightly",
        choices=list(STAGES),
        help=(
            "Stage whose goldens to run. Cumulative: nightly also includes per_commit and "
            "merge goldens. Overridable per run with ELYAEVAL_CI_STAGE. Default: nightly."
        ),
    )
    p_init.add_argument(
        "--output",
        default=None,
        help="Path to write the generated test file. Default: test_elyaeval_<task_type>.py",
    )
    p_init.add_argument(
        "--corpus-mode",
        choices=list(_CORPUS_MODE_TO_TEMPLATE),
        default="existing",
        help=(
            "'existing' (default): run_app() calls your app and it retrieves from "
            "whatever corpus it already has   nothing is inserted or torn down. Use "
            "this with your own goldens file written against your real corpus. "
            "'seeded': temporarily inserts each golden's context passages into your "
            "app's storage before the suite and removes them after   for running the "
            "shared standard dataset against an app whose corpus doesn't already "
            "cover it. Requires filling in _ingest_passage/_teardown_passages for "
            "your storage backend. Not applicable to task types with no retrieval "
            "leg (currently: agentic_action)   omit this flag for those."
        ),
    )
    p_init.add_argument(
        "--eval-mode",
        choices=["e2e", "component"],
        default="e2e",
        help=(
            "'e2e' (default): run_app() calls your app over HTTP and produces one flat "
            "TestResult per golden (report.py/evaluate_golden). "
            "'component': imports your app's traced entrypoint IN-PROCESS and scores "
            "each @observe'd span separately (tracing_report.py/TracedRunner)   meant "
            "to run alongside the e2e suite, not replace it. --corpus-mode does not "
            "apply to component mode   omit it."
        ),
    )

    p_init_ci = sub.add_parser(
        "init-ci",
        help="Generate this mission repo's PipelineRun for the shared elyaeval Tekton Pipeline",
    )
    p_init_ci.add_argument(
        "--repo-url",
        default=None,
        help="Public HTTPS URL of THIS repo. Default: left as ${REPO_URL}, filled in by envsubst at run time.",
    )
    p_init_ci.add_argument("--revision", default="main", help="Branch, tag or commit to evaluate. Default: main.")
    p_init_ci.add_argument(
        "--blob-container", required=True,
        help="Azure Blob container for this mission's results and baselines (created if missing), e.g. rag-demo.",
    )
    p_init_ci.add_argument(
        "--task-type", default="rag_qa",
        help="Only used to derive the default --test-file (test_elyaeval_<task_type>.py).",
    )
    p_init_ci.add_argument("--test-file", default=None, help="Test file path, relative to --sut-path.")
    p_init_ci.add_argument(
        "--sut-path", default=".",
        help="Folder in the repo holding the test file, goldens and elyaeval.toml. Default: repo root.",
    )
    p_init_ci.add_argument(
        "--runner-image", default=None,
        help="Runner image to pin. Default: $ELYAEVAL_RUNNER_IMAGE, [tool.elyaeval.ci] runner_image, "
             "or elyaeval-runner:<installed version>.",
    )
    p_init_ci.add_argument("--ci-stage", default="", help="Stage for CI runs (empty = the test file's own).")
    p_init_ci.add_argument("--budget-seconds", default="", help="Wall-clock budget for CI runs (empty = none/config).")
    p_init_ci.add_argument("--timeout", default="2h", help="Tekton pipeline timeout. Default: 2h (local judges are slow).")
    p_init_ci.add_argument("--namespace", default="elyaeval", help="Namespace shown in the instructions. Default: elyaeval.")
    p_init_ci.add_argument("--output", default="pipelinerun.yaml", help="Default: pipelinerun.yaml")

    p_ci = sub.add_parser("ci", help="The Tekton recipe shipped with this elyaeval version")
    p_ci.add_argument(
        "action", choices=["render", "apply", "dockerfile"],
        help="render: write pipeline.yaml + tasks.yaml; apply: kubectl apply them; dockerfile: print the runner Dockerfile",
    )
    p_ci.add_argument("--output-dir", default="ci", help="render: where to write the YAML. Default: ci/")
    p_ci.add_argument("-n", "--namespace", default="elyaeval", help="apply: target namespace. Default: elyaeval.")
    p_ci.add_argument("--runner-image", default=None, help="Runner image pinned as the Pipeline default.")
    p_ci.add_argument("--dry-run", choices=["client", "server"], default=None, help="apply: kubectl --dry-run mode.")

    p_report = sub.add_parser(
        "report",
        help="Regenerate the HTML report + per-metric averages for an existing results CSV",
    )
    p_report.add_argument("--csv", required=True, help="Path to a results_*.csv file (e2e or component).")
    p_report.add_argument(
        "--html",
        default=None,
        help="Output HTML path. Default: same name as --csv with a .html extension.",
    )
    p_report.add_argument(
        "--group-by",
        default="metric_name",
        help=(
            "Comma-separated columns to average by. Default: metric_name "
            "(matches results_<task_type>_<ts>.csv from report.py). Use "
            "span_name,metric_name for a *_component.csv from tracing_report.py."
        ),
    )

    p_compare = sub.add_parser(
        "compare",
        help="Diff two results CSVs' per-metric averages and flag regressions (no Confident AI needed)",
    )
    p_compare.add_argument("--baseline", required=True, help="Path to the known-good/previous results_*.csv.")
    p_compare.add_argument("--candidate", required=True, help="Path to the results_*.csv being checked.")
    p_compare.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=f"Mean paired score drop (0..1) that counts as a regression. Default: config or {DEFAULT_REGRESSION_TOLERANCE}.",
    )
    p_compare.add_argument("--alpha", type=float, default=None, help="Significance level of the statistical gate. Default: 0.05.")
    p_compare.add_argument("--min-paired", type=int, default=None, help="Below this many paired goldens the tolerance alone decides. Default: 5.")
    p_compare.add_argument("--gate", choices=["stat", "delta"], default=None, help="stat (default): tolerance AND significance; delta: tolerance only.")
    p_compare.add_argument(
        "--max-error-rate-increase", type=float, default=None,
        help="Fail if the share of errored/unscored rows rises by more than this. Default: 0.05.",
    )
    p_compare.add_argument("--max-pass-rate-drop", type=float, default=None, help="Fail if the pass rate drops by more than this. Default: off.")
    p_compare.add_argument("--unpaired", action="store_true", help="Compare all-goldens averages instead of paired goldens.")
    p_compare.add_argument(
        "--allow-metadata-mismatch", action="store_true",
        help="Compare even if the runs used different judges (exit code 2 otherwise).",
    )
    p_compare.add_argument(
        "--group-by",
        default="metric_name",
        help=(
            "Comma-separated columns to average by. Default: metric_name. Use "
            "span_name,metric_name for *_component.csv files."
        ),
    )
    p_compare.add_argument(
        "--html",
        default=None,
        help="Optional path to also write an HTML regression report.",
    )

    p_cal = sub.add_parser(
        "calibrate",
        help="Judge noise -> recommended regression tolerances; judge vs human labels -> agreement + thresholds",
    )
    p_cal.add_argument("--results", required=True, help="A results CSV, ideally from a run with --elyaeval-repeats >= 2.")
    p_cal.add_argument("--labels", default=None, help="Optional human labels CSV (golden_id, metric_name, human_pass[, human_score]).")
    p_cal.add_argument("--group-by", default="metric_name", help="span_name,metric_name for component CSVs.")

    budget_flags = (
        ("--budget-usd", "max_cost_usd"),
        ("--budget-seconds", "max_seconds"),
        ("--budget-tokens", "max_tokens"),
        ("--budget-calls", "max_judge_calls"),
    )
    p_est = sub.add_parser("estimate", help="Project a run's time/cost/tokens/judge calls before running it")
    p_est.add_argument("--task-type", required=True, choices=list(_TASK_TYPE_TO_METRICS_FACTORY))
    p_est.add_argument("--ci-stage", default="nightly", choices=list(STAGES))
    p_est.add_argument("--dataset", default=None, help="Goldens file (default: the bundled standard dataset).")
    p_est.add_argument("--repeats", type=int, default=1)
    p_est.add_argument("--history", default=None, help="A budget_*.json ledger to project from (default: latest in report/).")
    for flag, _ in budget_flags:
        p_est.add_argument(flag, default=None, help="Budget limit (overrides [tool.elyaeval.budget]).")

    p_budget = sub.add_parser("budget", help="Summarize a run's budget ledger (report/budget_*.json)")
    p_budget.add_argument("--ledger", default=None, help="Path to a budget_*.json (default: latest in report/).")
    p_budget.add_argument("--fail-on-exhausted", action="store_true", help="Exit 1 if the run exhausted its budget.")

    p_cache = sub.add_parser("cache", help="Inspect or clear the judge-result cache")
    p_cache.add_argument("action", choices=["stats", "clear"], nargs="?", default="stats")

    p_judge = sub.add_parser(
        "judge",
        help="Show the judge model the current config resolves to (provider-agnostic)",
    )
    p_judge.add_argument(
        "--ping",
        action="store_true",
        help="Also make one tiny schema-constrained call to check the judge works.",
    )

    args = parser.parse_args()

    if args.command == "init":
        output = args.output or _default_test_filename(args.task_type, args.eval_mode)
        out_path = init(args.task_type, args.ci_stage, output, args.corpus_mode, args.eval_mode)
        print(f"Wrote {out_path}")
        print("Next steps:")
        if args.eval_mode == "component":
            print(f"  1. Open {out_path} and fill in TODO 1 and TODO 2 "
                  f"(import your traced entrypoint + implement call_sut()).")
            print("  2. Point GOLDENS at your own goldens file (dataset_path=...).")
            print(f"  3. Run: deepeval test run {out_path}")
        elif args.task_type == "agentic_action":
            print(f"  1. Open {out_path} and implement run_app() (HTTP or in-process) returning an AgenticRunResult.")
            print("     Goldens state expected_tools (and optionally expected_output, max_tool_calls).")
            print("  2. Point GOLDENS at your own goldens file (dataset_path=...).")
            print(f"  3. Run: deepeval test run {out_path}")
        elif args.task_type == "agentic_trajectory":
            print(f"  1. Open {out_path}: import your traced agent entrypoint and implement call_sut().")
            print("     The agent needs @observe(type='agent'|'tool'|'llm') spans + update_current_trace().")
            print("  2. Point GOLDENS at your own goldens file (dataset_path=...).")
            print(f"  3. Run: deepeval test run {out_path}")
        elif args.corpus_mode == "existing":
            print(f"  1. Open {out_path} and fill in TODO 1 and TODO 2 (import + run_app()).")
            print("  2. Point GOLDENS at your own goldens file (dataset_path=...).")
            print(f"  3. Run: deepeval test run {out_path}")
        else:
            print(f"  1. Open {out_path} and fill in TODO 1, TODO 2, and TODO 3 (ingest/teardown).")
            print(f"  2. Run: deepeval test run {out_path}")

    elif args.command == "init-ci":
        out_path = init_ci(
            repo_url=args.repo_url,
            revision=args.revision,
            blob_container=args.blob_container,
            output=args.output,
            task_type=args.task_type,
            test_file=args.test_file,
            sut_path=args.sut_path,
            runner_image=args.runner_image,
            ci_stage=args.ci_stage,
            budget_seconds=args.budget_seconds,
            timeout=args.timeout,
            namespace=args.namespace,
        )
        print(f"Wrote {out_path}")
        print("Next steps:")
        print(f"  1. Commit {out_path} to this repo (with the test file, goldens and elyaeval.toml).")
        print(f"  2. Once per cluster: build/load the runner image and `elyaeval ci apply -n {args.namespace}`.")
        print("  3. Once per namespace: create the judge-model-credentials and blob-storage-credentials Secrets.")
        print("  4. Start the app, set $SUT_URL / $JUDGE_* (see the file header) and")
        print(f"     envsubst < {out_path.name} | kubectl create -n {args.namespace} -f -")

    elif args.command == "ci":
        raise SystemExit(ci_command(args.action, args.output_dir, args.namespace, args.runner_image, args.dry_run))

    elif args.command == "report":
        report(args.csv, html_path=args.html, group_by=args.group_by)

    elif args.command == "compare":
        exit_code = compare(
            args.baseline,
            args.candidate,
            tolerance=args.tolerance,
            group_by=args.group_by,
            html_path=args.html,
            alpha=args.alpha,
            min_paired=args.min_paired,
            gate=args.gate,
            max_error_rate_increase=args.max_error_rate_increase,
            max_pass_rate_drop=args.max_pass_rate_drop,
            unpaired=args.unpaired,
            allow_metadata_mismatch=args.allow_metadata_mismatch,
        )
        raise SystemExit(exit_code)

    elif args.command == "calibrate":
        raise SystemExit(calibrate(args.results, labels_csv=args.labels, group_by=args.group_by))

    elif args.command == "estimate":
        overrides = {key: getattr(args, flag[2:].replace("-", "_")) for flag, key in budget_flags}
        raise SystemExit(estimate(
            args.task_type, args.ci_stage, dataset_path=args.dataset, repeats=args.repeats,
            history=args.history, budget_overrides=overrides,
        ))

    elif args.command == "budget":
        raise SystemExit(budget_command(args.ledger, fail_on_exhausted=args.fail_on_exhausted))

    elif args.command == "cache":
        raise SystemExit(cache_command(args.action))

    elif args.command == "judge":
        raise SystemExit(judge(ping=args.ping))


if __name__ == "__main__":
    main()
