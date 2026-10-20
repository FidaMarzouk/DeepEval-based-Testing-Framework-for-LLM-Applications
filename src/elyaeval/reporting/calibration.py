"""
Judge calibration: how much can this judge's scores be trusted, and what
tolerances/thresholds follow from that? Backs `elyaeval calibrate`.

1. Noise (needs a run with --elyaeval-repeats >= 2): per metric, the
   within-golden SD of repeated scores, and the regression tolerance that
   noise alone would rarely exceed for a suite of this size:
       recommended_tolerance = 1.96 * sd * sqrt(2 / (n_goldens * repeats))
   Gating tighter than that means CI fails on judge noise, not regressions.

2. Agreement (needs human labels): per metric, how often the judge's
   pass/fail agrees with a human's (accuracy, Cohen's kappa), and which
   threshold would best match the humans (max balanced accuracy over a
   0.05-step grid). A labels CSV has columns:
       golden_id, metric_name, human_pass   [, span_name] [, human_score]
   human_pass is true/false/1/0/pass/fail; human_score (0..1) is optional
   and adds a mean-absolute-error column.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

from .compare import per_golden_scores
from .stats import cohen_kappa, recommended_tolerance, within_golden_sd

_TRUE = {"1", "true", "yes", "y", "pass", "passed"}
_FALSE = {"0", "false", "no", "n", "fail", "failed"}


def _groups(rows: list[dict], group_keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(tuple(r.get(k, "") for k in group_keys), []).append(r)
    return out


def noise_report(rows: list[dict], group_keys: tuple[str, ...] = ("metric_name",)) -> list[dict[str, Any]]:
    report = []
    for key, group in _groups(rows, group_keys).items():
        by_golden = per_golden_scores(group)
        repeats = max((len(v) for v in by_golden.values()), default=0)
        sd = within_golden_sd(by_golden)
        report.append({
            **dict(zip(group_keys, key, strict=True)),
            "n_goldens": len(by_golden),
            "repeats": repeats,
            "noise_sd": sd,
            "recommended_tolerance": (
                recommended_tolerance(sd, len(by_golden), repeats) if sd is not None else None
            ),
        })
    return report


def _parse_bool(value: Any) -> Optional[bool]:
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def read_labels(path: str | Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = {"golden_id", "metric_name", "human_pass"} - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"labels file {path} is missing column(s): {sorted(missing)}")
    labels = []
    for line_no, row in enumerate(rows, start=2):
        human_pass = _parse_bool(row["human_pass"])
        if human_pass is None:
            raise ValueError(f"{path}:{line_no}: human_pass must be true/false/1/0/pass/fail, got {row['human_pass']!r}")
        score = row.get("human_score")
        labels.append({**row, "human_pass": human_pass, "human_score": float(score) if score not in (None, "") else None})
    return labels


def _balanced_accuracy(pred: list[bool], truth: list[bool]) -> Optional[float]:
    pos = [p for p, t in zip(pred, truth, strict=True) if t]
    neg = [p for p, t in zip(pred, truth, strict=True) if not t]
    if not pos or not neg:
        return None
    return (sum(pos) / len(pos) + sum(1 for p in neg if not p) / len(neg)) / 2


def agreement_report(
    rows: list[dict],
    labels: list[dict],
    group_keys: tuple[str, ...] = ("metric_name",),
) -> list[dict[str, Any]]:
    labels_by_key: dict[tuple, dict[str, dict]] = {}
    for lab in labels:
        labels_by_key.setdefault(tuple(lab.get(k, "") for k in group_keys), {})[str(lab["golden_id"])] = lab

    report = []
    for key, group in _groups(rows, group_keys).items():
        group_labels = labels_by_key.get(key)
        if not group_labels:
            continue
        thresholds = {r["threshold"] for r in group if r.get("threshold") is not None}
        threshold = thresholds.pop() if len(thresholds) == 1 else None
        scores = {g: sum(v) / len(v) for g, v in per_golden_scores(group).items()}
        shared = sorted(set(scores) & set(group_labels))
        if not shared:
            continue
        judge_scores = [scores[g] for g in shared]
        human = [bool(group_labels[g]["human_pass"]) for g in shared]
        judge_pass = [s >= threshold for s in judge_scores] if threshold is not None else []

        best_t, best_ba = None, None
        for step in range(1, 20):
            t = step / 20
            ba = _balanced_accuracy([s >= t for s in judge_scores], human)
            if ba is not None and (best_ba is None or ba > best_ba + 1e-12):
                best_t, best_ba = t, ba

        human_scores = [(scores[g], group_labels[g]["human_score"]) for g in shared if group_labels[g]["human_score"] is not None]
        report.append({
            **dict(zip(group_keys, key, strict=True)),
            "n": len(shared),
            "threshold": threshold,
            "accuracy": (sum(1 for j, h in zip(judge_pass, human, strict=True) if j == h) / len(shared)) if judge_pass else None,
            "kappa": cohen_kappa(judge_pass, human) if judge_pass else None,
            "judge_pass_rate": (sum(judge_pass) / len(judge_pass)) if judge_pass else None,
            "human_pass_rate": sum(human) / len(human),
            "mae": (sum(abs(j - h) for j, h in human_scores) / len(human_scores)) if human_scores else None,
            "best_threshold": best_t,
            "best_balanced_accuracy": best_ba,
        })
    return report


def metric_key_for(metric_name: str) -> str:
    """'Contextual Precision' -> 'contextual_precision' (threshold config key)."""
    return metric_name.strip().lower().replace(" ", "_").replace("-", "_")
