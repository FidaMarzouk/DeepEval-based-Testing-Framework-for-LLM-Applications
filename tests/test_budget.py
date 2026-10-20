"""Budget config, spend accounting, guard, ledger and projections."""

import json

import pytest

from elyaeval.core import budget as b
from elyaeval.core.usage import GoldenUsage, UsageMeter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_budget_config_precedence(isolated_config, monkeypatch):
    from elyaeval.core import config

    assert not b.budget_config_from_settings().enabled
    config.load_config.cache_clear()
    (isolated_config / "pyproject.toml").write_text(
        '[tool.elyaeval.budget]\nmax_seconds = 600\nmax_cost_usd = 1.5\non_exceed = "warn"\n', encoding="utf-8"
    )
    cfg = b.budget_config_from_settings()
    assert (cfg.max_seconds, cfg.max_cost_usd, cfg.on_exceed) == (600, 1.5, "warn") and cfg.enabled
    monkeypatch.setenv("ELYAEVAL_BUDGET_SECONDS", "120")
    assert b.budget_config_from_settings().max_seconds == 120
    assert b.budget_config_from_settings(max_seconds="30").max_seconds == 30


def test_budget_config_rejects_bad_values(isolated_config):
    from elyaeval.core import config

    with pytest.raises(ValueError, match="on_exceed"):
        b.budget_config_from_settings(on_exceed="explode")
    config.load_config.cache_clear()
    (isolated_config / "pyproject.toml").write_text("[tool.elyaeval.budget]\nmax_secs = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_secs"):
        b.budget_config_from_settings()


def _totals(**kw):
    base = {"judge_calls": 10, "judge_seconds": 1800.0, "judge_prompt_tokens_estimated": 900_000,
            "judge_output_tokens_estimated": 100_000, "goldens_evaluated": 3}
    base.update(kw)
    return base


def test_spend_uses_reported_cost_when_real():
    spend = b.compute_spend(_totals(judge_cost_reported=0.25, judge_tokens_reported=5000), b.Pricing(), 60.0)
    assert spend["judge_cost_usd"] == 0.25 and spend["judge_cost_source"] == "reported"
    assert spend["tokens"] == 5000 and spend["tokens_source"] == "reported"


def test_spend_prices_unreported_local_judge():
    by_tokens = b.compute_spend(_totals(), b.Pricing(judge_input_per_mtok=1.0, judge_output_per_mtok=2.0), 60.0)
    assert by_tokens["judge_cost_usd"] == pytest.approx(0.9 + 0.2) and by_tokens["judge_cost_source"] == "priced"
    by_hour = b.compute_spend(_totals(), b.Pricing(judge_per_hour=0.5), 60.0)
    assert by_hour["judge_cost_usd"] == pytest.approx(0.25)  # 1800 s of judging at $0.50/h
    unknown = b.compute_spend(_totals(), b.Pricing(), 60.0)
    assert unknown["cost_usd"] is None and unknown["judge_cost_source"] == "unknown"
    assert unknown["tokens"] == 1_000_000 and unknown["tokens_source"] == "estimated"


def test_spend_includes_sut():
    spend = b.compute_spend(
        _totals(judge_cost_reported=0.1, sut_input_tokens=1_000_000, sut_output_tokens=0),
        b.Pricing(sut_input_per_mtok=0.5),
        1.0,
    )
    assert spend["sut_cost_usd"] == 0.5 and spend["cost_usd"] == pytest.approx(0.6)


def _guard(clock, **cfg):
    meter = UsageMeter()
    return b.BudgetGuard(b.BudgetConfig(**cfg), meter=meter, clock=clock), meter


def test_guard_stops_when_time_budget_is_reached():
    clock = FakeClock()
    guard, meter = _guard(clock, max_seconds=100, predictive=False)
    guard.start()
    assert guard.check() is None
    meter.record_golden(GoldenUsage("g1"))
    clock.now = 100.0
    reason = guard.check()
    assert reason and "seconds" in reason
    assert guard.check() == reason  # stays exhausted
    assert guard.status()["exhausted"]


def test_guard_predicts_the_next_golden():
    clock = FakeClock()
    guard, meter = _guard(clock, max_seconds=100)
    guard.start()
    meter.record_golden(GoldenUsage("g1"))
    clock.now = 40.0
    assert guard.check() is None          # 40 + 40 = 80 fits
    meter.record_golden(GoldenUsage("g2"))
    clock.now = 80.0
    assert "next golden" in guard.check()  # 80 + 40 > 100


def test_guard_warn_mode_never_skips():
    clock = FakeClock()
    guard, meter = _guard(clock, max_judge_calls=5, on_exceed="warn")
    guard.start()
    meter.judge_calls = 10
    meter.record_golden(GoldenUsage("g1"))
    assert guard.check() is None
    assert guard.warnings and guard.status()["exhausted"]


def test_disabled_guard_never_skips():
    guard, meter = _guard(FakeClock())
    meter.judge_calls = 10**6
    assert guard.check() is None


def test_ledger_roundtrip_and_projection(tmp_path):
    clock = FakeClock()
    guard, meter = _guard(clock, max_seconds=3600)
    guard.start()
    for i in range(4):
        meter.record_golden(GoldenUsage(f"g{i}", n_metrics=5, judge_seconds=300.0, sut_seconds=2.0))
    meter.judge_calls = 60
    clock.now = 1210.0
    path = b.write_ledger(tmp_path, "20260101T000000Z", guard, meter=meter)
    assert path.name == "budget_20260101T000000Z.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["totals"]["goldens_evaluated"] == 4 and len(data["goldens"]) == 4

    latest = b.latest_ledger(tmp_path)
    profile = b.profile_from_ledger(latest)
    assert profile["judge_seconds_per_metric"] == pytest.approx(60.0)
    assert profile["judge_calls_per_metric"] == pytest.approx(3.0)
    projection = b.project(10, 5, 2, profile)
    assert projection["golden_runs"] == 20 and projection["metric_evaluations"] == 100
    assert projection["seconds"] == pytest.approx(100 * 60 + 20 * 2)
    over = b.projection_over_budget(projection, b.BudgetConfig(max_seconds=3600))
    assert over and "seconds" in over[0]


def test_heuristic_profile_without_history():
    from deepeval.dataset import Golden

    goldens = [Golden(input="q" * 400, expected_output="a" * 400, context=["c" * 4000])]
    profile = b.heuristic_profile(goldens, ["faithfulness", "contextual_recall"])
    assert profile["judge_calls_per_metric"] == 3
    assert profile["judge_tokens_per_metric"] > 3 * 1000
    assert profile["judge_seconds_per_metric"] is None


def test_priority_rank():
    assert b.priority_rank("P0") < b.priority_rank("p1") < b.priority_rank("P2") < b.priority_rank(None)
