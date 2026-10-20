"""
Cross-run regression comparison.

For each metric (or span + metric) group present in either run:

  1. PAIR goldens by golden_id. Only goldens scored in both runs are
     compared, so a candidate that ran a different/smaller golden set can't
     look better or worse just because of what it ran. Goldens present in
     only one run are counted and reported.
  2. Per golden, average its scores (over repeats, if any), then take the
     paired deltas candidate - baseline.
  3. Decide:
       score regressed  if mean delta < -tolerance  AND  (gate="stat":)
                        the sign-flip permutation p-value < alpha.
                        With fewer than `min_paired` pairs, or gate="delta",
                        the tolerance alone decides (method says so).
       error regressed  if the share of errored/unscored rows rose by more
                        than max_error_rate_increase — errors are otherwise
                        invisible in averages (and a judge that crashes on
                        the hard cases would make the average go UP).
       pass regressed   if the pass rate dropped by more than
                        max_pass_rate_drop (off unless configured).
     Any of the three -> "regressed". "improved" mirrors the score test.

Gate settings come from [tool.elyaeval.regression] (CLI flags override):

    [tool.elyaeval.regression]
    tolerance = 0.02
    tolerances = { faithfulness = 0.05 }   # per metric_name or "span/metric"
    alpha = 0.05
    min_paired = 5
    gate = "stat"                          # or "delta"
    max_error_rate_increase = 0.05
    max_pass_rate_drop = 0.10              # unset = not gated
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

from .stats import bootstrap_ci, mean, paired_permutation_pvalue

DEFAULT_REGRESSION_TOLERANCE = 0.02


@dataclass(frozen=True)
class GateConfig:
    tolerance: float = DEFAULT_REGRESSION_TOLERANCE
    tolerances: dict[str, float] = field(default_factory=dict)
    alpha: float = 0.05
    min_paired: int = 5
    gate: str = "stat"
    max_error_rate_increase: Optional[float] = 0.05
    max_pass_rate_drop: Optional[float] = None
    paired: bool = True

    def tolerance_for(self, label: str, metric_name: str) -> float:
        return float(self.tolerances.get(label, self.tolerances.get(metric_name, self.tolerance)))


def gate_config_from_settings(**overrides: Any) -> GateConfig:
    """GateConfig from [tool.elyaeval.regression], then non-None overrides."""
    from ..core.config import config_section

    section = dict(config_section("regression"))
    known = set(GateConfig.__dataclass_fields__)
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in [tool.elyaeval.regression]: {sorted(unknown)}")
    if section.get("gate", "stat") not in ("stat", "delta"):
        raise ValueError("[tool.elyaeval.regression] gate must be 'stat' or 'delta'")
    cfg = GateConfig(**section)
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


def _is_error(row: dict) -> bool:
    return bool(row.get("error")) or row.get("score") is None


def _group(rows: list[dict], group_keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k, "") for k in group_keys), []).append(row)
    return groups


def per_golden_scores(rows: list[dict]) -> dict[str, list[float]]:
    """golden_id -> numeric scores (one per repeat) for one group's rows."""
    scores: dict[str, list[float]] = {}
    for row in rows:
        if row.get("score") is not None and not row.get("error"):
            scores.setdefault(str(row.get("golden_id")), []).append(float(row["score"]))
    return scores


def _pass_rate(rows: list[dict]) -> Optional[float]:
    flags = [r["success"] for r in rows if r.get("success") is not None]
    return sum(1 for f in flags if f) / len(flags) if flags else None


def _error_rate(rows: list[dict]) -> Optional[float]:
    return sum(1 for r in rows if _is_error(r)) / len(rows) if rows else None


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else a - b


def compare_runs(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    group_keys: tuple[str, ...] = ("metric_name",),
    tolerance: Optional[float] = None,
    gate: Optional[GateConfig] = None,
) -> list[dict]:
    """One dict per group, in first-seen order (baseline then candidate).
    `tolerance` is kept for backward compatibility and overrides gate's."""
    gate = gate or GateConfig()
    if tolerance is not None:
        gate = replace(gate, tolerance=tolerance)

    base_groups, cand_groups = _group(baseline_rows, group_keys), _group(candidate_rows, group_keys)
    order = list(dict.fromkeys([*base_groups, *cand_groups]))

    comparisons = []
    for key in order:
        b_rows, c_rows = base_groups.get(key), cand_groups.get(key)
        label = "/".join(str(k) for k in key)
        metric_name = str(dict(zip(group_keys, key, strict=True)).get("metric_name", label))
        tol = gate.tolerance_for(label, metric_name)
        out: dict[str, Any] = {
            **dict(zip(group_keys, key, strict=True)),
            "tolerance": tol,
            "reasons": [],
            "baseline_pass_rate": _pass_rate(b_rows) if b_rows else None,
            "candidate_pass_rate": _pass_rate(c_rows) if c_rows else None,
            "baseline_error_rate": _error_rate(b_rows) if b_rows else None,
            "candidate_error_rate": _error_rate(c_rows) if c_rows else None,
            "baseline_cost": _sum_cost(b_rows),
            "candidate_cost": _sum_cost(c_rows),
        }
        out["pass_rate_delta"] = _sub(out["candidate_pass_rate"], out["baseline_pass_rate"])
        out["error_rate_delta"] = _sub(out["candidate_error_rate"], out["baseline_error_rate"])

        if b_rows is None or c_rows is None:
            present = per_golden_scores(c_rows or b_rows or [])
            side_avg = mean([sum(v) / len(v) for v in present.values()])
            out.update(
                status="new" if b_rows is None else "removed",
                baseline_avg=side_avg if c_rows is None else None,
                candidate_avg=side_avg if b_rows is None else None,
                delta=None, ci_low=None, ci_high=None, p_value=None, n_paired=0,
                only_baseline=len(present) if c_rows is None else 0,
                only_candidate=len(present) if b_rows is None else 0,
                method="n/a",
            )
            comparisons.append(out)
            continue

        b_scores, c_scores = per_golden_scores(b_rows), per_golden_scores(c_rows)
        b_mean = {g: sum(v) / len(v) for g, v in b_scores.items()}
        c_mean = {g: sum(v) / len(v) for g, v in c_scores.items()}

        if gate.paired:
            shared = sorted(set(b_mean) & set(c_mean))
            deltas = [c_mean[g] - b_mean[g] for g in shared]
            b_avg = mean([b_mean[g] for g in shared])
            c_avg = mean([c_mean[g] for g in shared])
            n_paired = len(shared)
            only_b, only_c = len(set(b_mean) - set(c_mean)), len(set(c_mean) - set(b_mean))
        else:
            deltas = []
            b_avg, c_avg = mean(list(b_mean.values())), mean(list(c_mean.values()))
            n_paired, only_b, only_c = 0, 0, 0

        delta = None if b_avg is None or c_avg is None else round(c_avg - b_avg, 9)
        ci_low, ci_high = bootstrap_ci(deltas) if deltas else (None, None)
        p_value = paired_permutation_pvalue(deltas) if deltas else None

        use_stat = gate.gate == "stat" and gate.paired and n_paired >= gate.min_paired
        method = "paired permutation" if use_stat else (
            f"delta only (n={n_paired} < {gate.min_paired})" if gate.paired and gate.gate == "stat" else "delta only"
        )

        score_status = "unchanged"
        if delta is not None:
            significant = (p_value is not None and p_value < gate.alpha) if use_stat else True
            if delta < -tol and significant:
                score_status = "regressed"
            elif delta > tol and significant:
                score_status = "improved"
        if score_status == "regressed":
            out["reasons"].append(f"score {delta:+.3f} < -{tol:g}" + (f" (p={p_value:.3f})" if use_stat else ""))

        err_delta = out["error_rate_delta"]
        if gate.max_error_rate_increase is not None and err_delta is not None and err_delta > gate.max_error_rate_increase + 1e-12:
            out["reasons"].append(f"error rate +{err_delta:.0%} > {gate.max_error_rate_increase:.0%}")
        pr_delta = out["pass_rate_delta"]
        if gate.max_pass_rate_drop is not None and pr_delta is not None and pr_delta < -gate.max_pass_rate_drop - 1e-12:
            out["reasons"].append(f"pass rate {pr_delta:+.0%} < -{gate.max_pass_rate_drop:.0%}")

        status = "regressed" if out["reasons"] else score_status
        out.update(
            status=status,
            baseline_avg=b_avg,
            candidate_avg=c_avg,
            delta=delta,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_paired=n_paired,
            only_baseline=only_b,
            only_candidate=only_c,
            method=method,
        )
        comparisons.append(out)
    return comparisons


def _sum_cost(rows: Optional[list[dict]]) -> Optional[float]:
    costs = [r["evaluation_cost"] for r in rows or [] if r.get("evaluation_cost") is not None]
    return sum(costs) if costs else None
