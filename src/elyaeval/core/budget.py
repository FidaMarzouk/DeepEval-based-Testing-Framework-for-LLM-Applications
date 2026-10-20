"""
Cost budgeting for evaluation runs.

A budget caps what one run may consume, in any combination of:

    max_cost_usd     judge + SUT spend (reported by the provider, or priced
                     from usage via [tool.elyaeval.pricing])
    max_seconds      wall-clock time of the evaluation (the dependable unit
                     for local judges, whose providers report no cost)
    max_tokens       judge + SUT tokens (reported, else estimated)
    max_judge_calls  number of judge LLM calls

Configure in [tool.elyaeval.budget], override per run with
ELYAEVAL_BUDGET_USD / _SECONDS / _TOKENS / _CALLS / _ON_EXCEED or the
--elyaeval-budget-* pytest options.

Enforcement (pytest plugin):
  - prioritize: goldens run P0 first, so a budget that runs out cuts the
    least important goldens;
  - before each golden, the guard checks the budget; with predictive=true
    it also stops when the *next* golden would likely overrun it (based on
    the average so far) — one golden can take minutes on a local judge;
  - on_exceed="stop" skips the remaining goldens (reported as skipped,
    with the reason), on_exceed="warn" only reports;
  - report/budget_<run>.json records limits, spend, what was skipped and a
    per-golden ledger; CI's check-budget step fails the run on exhaustion.

Before a run, `elyaeval estimate` (and the plugin's preflight line) project
time/cost/tokens from the previous run's ledger, or from a rough heuristic
when there's no history.

Pricing for providers that don't report cost (local/self-hosted judges):

    [tool.elyaeval.pricing]
    judge_input_per_mtok = 0.0     # USD per million prompt tokens
    judge_output_per_mtok = 0.0
    judge_per_hour = 0.40          # e.g. GPU-hour cost of a self-hosted judge
    sut_input_per_mtok = 0.15      # the SUT's own model, if it reports tokens
    sut_output_per_mtok = 0.60
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional

from .config import config_section
from .usage import METER, UsageMeter, estimate_tokens

BUDGET_ENV = {
    "max_cost_usd": "ELYAEVAL_BUDGET_USD",
    "max_seconds": "ELYAEVAL_BUDGET_SECONDS",
    "max_tokens": "ELYAEVAL_BUDGET_TOKENS",
    "max_judge_calls": "ELYAEVAL_BUDGET_CALLS",
    "on_exceed": "ELYAEVAL_BUDGET_ON_EXCEED",
}
BUDGET_FILE_GLOB = "budget_*.json"
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

# Rough judge calls per metric evaluation (DeepEval's multi-step metrics:
# extract -> verdicts -> reason). Contextual metrics also scale with the
# number of retrieved chunks; used only when there's no run history.
_CALLS_PER_METRIC = {
    "faithfulness": 4,
    "answer_relevancy": 3,
    "contextual_precision": 2,
    "contextual_recall": 2,
    "contextual_relevancy": 2,
    "summarization": 5,
    "correctness": 2,
    "tool_correctness": 1,
    "argument_correctness": 2,
}
_PROMPT_TEMPLATE_TOKENS = 700
_OUTPUT_TOKENS_PER_CALL = 150


@dataclass(frozen=True)
class BudgetConfig:
    max_cost_usd: Optional[float] = None
    max_seconds: Optional[float] = None
    max_tokens: Optional[int] = None
    max_judge_calls: Optional[int] = None
    on_exceed: str = "stop"
    predictive: bool = True
    prioritize: bool = True
    preflight: bool = True

    @property
    def enabled(self) -> bool:
        return any(v is not None for v in (self.max_cost_usd, self.max_seconds, self.max_tokens, self.max_judge_calls))

    def limits(self) -> dict[str, Any]:
        return {
            "cost_usd": self.max_cost_usd,
            "seconds": self.max_seconds,
            "tokens": self.max_tokens,
            "judge_calls": self.max_judge_calls,
        }


@dataclass(frozen=True)
class Pricing:
    judge_input_per_mtok: Optional[float] = None
    judge_output_per_mtok: Optional[float] = None
    judge_per_hour: Optional[float] = None
    sut_input_per_mtok: Optional[float] = None
    sut_output_per_mtok: Optional[float] = None


def _coerce(name: str, value: Any) -> Any:
    if value in (None, ""):
        return None
    if name == "on_exceed":
        value = str(value).lower()
        if value not in ("stop", "warn"):
            raise ValueError("budget on_exceed must be 'stop' or 'warn'")
        return value
    if name in ("predictive", "prioritize", "preflight"):
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
    if name in ("max_tokens", "max_judge_calls"):
        return int(float(value))
    return float(value)


def budget_config_from_settings(**overrides: Any) -> BudgetConfig:
    section = dict(config_section("budget"))
    known = {f.name for f in fields(BudgetConfig)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in [tool.elyaeval.budget]: {sorted(unknown)}")
    values = {k: _coerce(k, v) for k, v in section.items()}
    for key, env in BUDGET_ENV.items():
        if os.environ.get(env):
            values[key] = _coerce(key, os.environ[env])
    for key, value in overrides.items():
        if value is not None:
            values[key] = _coerce(key, value)
    return BudgetConfig(**{k: v for k, v in values.items() if v is not None})


def pricing_from_settings() -> Pricing:
    section = dict(config_section("pricing"))
    known = {f.name for f in fields(Pricing)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in [tool.elyaeval.pricing]: {sorted(unknown)}")
    return Pricing(**{k: float(v) for k, v in section.items()})


def _per_mtok(tokens: Optional[int], rate: Optional[float]) -> Optional[float]:
    return None if tokens is None or rate is None else tokens / 1e6 * rate


def compute_spend(totals: dict[str, Any], pricing: Pricing, elapsed_seconds: float) -> dict[str, Any]:
    """Turn meter totals into spend per budget dimension, recording where
    each cost figure came from (reported / priced / unknown)."""
    judge_cost, judge_source = totals.get("judge_cost_reported"), "reported"
    if judge_cost is None:
        token_cost = [
            c for c in (
                _per_mtok(totals.get("judge_prompt_tokens_estimated"), pricing.judge_input_per_mtok),
                _per_mtok(totals.get("judge_output_tokens_estimated"), pricing.judge_output_per_mtok),
            ) if c is not None
        ]
        hour_cost = (totals.get("judge_seconds", 0.0) / 3600 * pricing.judge_per_hour) if pricing.judge_per_hour is not None else None
        if token_cost or hour_cost is not None:
            judge_cost, judge_source = sum(token_cost) + (hour_cost or 0.0), "priced"
        else:
            judge_source = "unknown"

    sut_cost, sut_source = totals.get("sut_cost"), "reported"
    if sut_cost is None:
        priced = [c for c in (
            _per_mtok(totals.get("sut_input_tokens"), pricing.sut_input_per_mtok),
            _per_mtok(totals.get("sut_output_tokens"), pricing.sut_output_per_mtok),
        ) if c is not None]
        sut_cost, sut_source = (sum(priced), "priced") if priced else (None, "unknown")

    judge_tokens = totals.get("judge_tokens_reported") or (
        totals.get("judge_prompt_tokens_estimated", 0) + totals.get("judge_output_tokens_estimated", 0)
    )
    sut_tokens = (totals.get("sut_input_tokens") or 0) + (totals.get("sut_output_tokens") or 0)
    known_costs = [c for c in (judge_cost, sut_cost) if c is not None]
    return {
        "cost_usd": sum(known_costs) if known_costs else None,
        "judge_cost_usd": judge_cost,
        "judge_cost_source": judge_source,
        "sut_cost_usd": sut_cost,
        "sut_cost_source": sut_source,
        "seconds": elapsed_seconds,
        "tokens": judge_tokens + sut_tokens,
        "tokens_source": "reported" if totals.get("judge_tokens_reported") else "estimated",
        "judge_calls": totals.get("judge_calls", 0),
        "goldens_evaluated": totals.get("goldens_evaluated", 0),
    }


class BudgetGuard:
    def __init__(
        self,
        config: BudgetConfig,
        pricing: Optional[Pricing] = None,
        meter: UsageMeter = METER,
        clock=time.monotonic,
    ):
        self.config = config
        self.pricing = pricing or Pricing()
        self.meter = meter
        self._clock = clock
        self._started: Optional[float] = None
        self.skipped: list[str] = []
        self.exhausted_reason: Optional[str] = None
        self.warnings: list[str] = []

    def start(self) -> None:
        if self._started is None:
            self._started = self._clock()

    def elapsed(self) -> float:
        return 0.0 if self._started is None else self._clock() - self._started

    def spend(self) -> dict[str, Any]:
        return compute_spend(self.meter.totals(), self.pricing, self.elapsed())

    def _over(self, spend: dict[str, Any], projected: bool) -> list[str]:
        limits = self.config.limits()
        n = spend["goldens_evaluated"]
        reasons = []
        for dim, limit in limits.items():
            value = spend[dim]
            if limit is None or value is None:
                continue
            if value >= limit:
                reasons.append(f"{dim} {_fmt(dim, value)} reached the budget of {_fmt(dim, limit)}")
            elif projected and n:
                nxt = value + value / n
                if nxt > limit:
                    reasons.append(
                        f"{dim} {_fmt(dim, value)} + ~{_fmt(dim, value / n)} for the next golden would exceed "
                        f"the budget of {_fmt(dim, limit)}"
                    )
        return reasons

    def check(self) -> Optional[str]:
        """Reason to skip the next golden, or None to run it."""
        if not self.config.enabled:
            return None
        if self.exhausted_reason is not None and self.config.on_exceed == "stop":
            return self.exhausted_reason
        reasons = self._over(self.spend(), projected=self.config.predictive)
        if not reasons:
            return None
        reason = "; ".join(reasons)
        if self.config.on_exceed == "warn":
            if reason not in self.warnings:
                self.warnings.append(reason)
            return None
        self.exhausted_reason = f"elyaeval budget exhausted: {reason}"
        return self.exhausted_reason

    def record_skip(self, node_id: str) -> None:
        self.skipped.append(node_id)

    def status(self) -> dict[str, Any]:
        spend = self.spend()
        over = self._over(spend, projected=False)
        return {
            "limits": self.config.limits(),
            "on_exceed": self.config.on_exceed,
            "predictive": self.config.predictive,
            "spent": spend,
            "exhausted": bool(self.exhausted_reason) or (self.config.enabled and bool(over)),
            "exhausted_reason": self.exhausted_reason or ("; ".join(over) if over else None),
            "warnings": self.warnings,
            "skipped": self.skipped,
        }


def _fmt(dim: str, value: float) -> str:
    if dim == "cost_usd":
        return f"${value:.4f}"
    if dim == "seconds":
        return f"{value / 60:.1f} min" if value >= 120 else f"{value:.0f}s"
    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# Ledger + projection
# ---------------------------------------------------------------------------


def write_ledger(
    report_dir: str | Path,
    run_ts: str,
    guard: Optional[BudgetGuard],
    meter: UsageMeter = METER,
    status: Optional[dict[str, Any]] = None,
) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    if status is None and guard is not None:
        status = guard.status()
    payload = {
        "schema_version": 1,
        "run_ts": run_ts,
        "totals": meter.totals(),
        "budget": status,
        "pricing": asdict(guard.pricing) if guard is not None else None,
        "goldens": meter.ledger(),
    }
    path = report_dir / f"budget_{run_ts}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def latest_ledger(report_dir: str | Path = "report", exclude: Optional[str] = None) -> Optional[dict[str, Any]]:
    candidates = sorted(glob.glob(str(Path(report_dir) / BUDGET_FILE_GLOB)))
    for path in reversed(candidates):
        if exclude and Path(path).name == exclude:
            continue
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("totals", {}).get("goldens_evaluated"):
            data["_path"] = path
            return data
    return None


def profile_from_ledger(ledger: dict[str, Any]) -> dict[str, Optional[float]]:
    """Per-golden and per-metric-evaluation averages from a past run."""
    t = ledger["totals"]
    goldens = max(t.get("goldens_evaluated") or 0, 1)
    fresh_evals = max((t.get("metric_evaluations") or 0) - (t.get("cached_metric_evaluations") or 0), 1)
    spent = (ledger.get("budget") or {}).get("spent") or {}
    judge_tokens = t.get("judge_tokens_reported") or (
        (t.get("judge_prompt_tokens_estimated") or 0) + (t.get("judge_output_tokens_estimated") or 0)
    )
    def per(value: Optional[float], count: int) -> Optional[float]:
        return None if value is None else value / count

    return {
        "judge_seconds_per_metric": (t.get("judge_seconds") or 0.0) / fresh_evals,
        "judge_calls_per_metric": per(t.get("judge_calls") or None, fresh_evals),
        "judge_tokens_per_metric": per(judge_tokens or None, fresh_evals),
        "judge_cost_per_metric": per(spent.get("judge_cost_usd"), fresh_evals),
        "sut_seconds_per_golden": per(t.get("sut_seconds"), goldens),
        "sut_cost_per_golden": per(spent.get("sut_cost_usd"), goldens),
    }


def heuristic_profile(goldens: list[Any], metric_keys: list[str]) -> dict[str, Optional[float]]:
    """No history: rough token/call counts from golden text sizes. Time and
    cost stay unknown (they depend entirely on the judge's speed/price)."""
    if not goldens or not metric_keys:
        return {"judge_calls_per_metric": None, "judge_tokens_per_metric": None}
    text_tokens = sum(
        estimate_tokens(g.input) + estimate_tokens(g.expected_output) + sum(estimate_tokens(c) for c in (g.context or []))
        for g in goldens
    ) / len(goldens)
    calls = sum(_CALLS_PER_METRIC.get(k, 3) for k in metric_keys) / len(metric_keys)
    return {
        "judge_calls_per_metric": calls,
        "judge_tokens_per_metric": calls * (_PROMPT_TEMPLATE_TOKENS + text_tokens + _OUTPUT_TOKENS_PER_CALL),
        "judge_seconds_per_metric": None,
        "judge_cost_per_metric": None,
        "sut_seconds_per_golden": None,
        "sut_cost_per_golden": None,
    }


def project(n_goldens: int, metrics_per_golden: int, repeats: int, profile: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    runs = n_goldens * max(repeats, 1)
    evals = runs * metrics_per_golden

    def scale(key: str, count: int) -> Optional[float]:
        value = profile.get(key)
        return None if value is None else value * count

    judge_seconds = scale("judge_seconds_per_metric", evals)
    sut_seconds = scale("sut_seconds_per_golden", runs)
    judge_cost = scale("judge_cost_per_metric", evals)
    sut_cost = scale("sut_cost_per_golden", runs)
    costs = [c for c in (judge_cost, sut_cost) if c is not None]
    return {
        "goldens": n_goldens,
        "golden_runs": runs,
        "metric_evaluations": evals,
        "judge_calls": scale("judge_calls_per_metric", evals),
        "tokens": scale("judge_tokens_per_metric", evals),
        "seconds": (judge_seconds or 0.0) + (sut_seconds or 0.0) if judge_seconds is not None else None,
        "cost_usd": sum(costs) if costs else None,
    }


def projection_over_budget(projection: dict[str, Optional[float]], config: BudgetConfig) -> list[str]:
    out = []
    for dim, limit in config.limits().items():
        value = projection.get(dim)
        if limit is not None and value is not None and value > limit:
            out.append(f"projected {dim} {_fmt(dim, value)} > budget {_fmt(dim, limit)}")
    return out


def priority_rank(priority: Any) -> int:
    return PRIORITY_ORDER.get(str(priority or "").upper(), len(PRIORITY_ORDER))


def with_overrides(config: BudgetConfig, **overrides: Any) -> BudgetConfig:
    return replace(config, **{k: _coerce(k, v) for k, v in overrides.items() if v is not None})
