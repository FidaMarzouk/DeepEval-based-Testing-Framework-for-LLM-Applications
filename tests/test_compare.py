"""Regression gate: pairing, statistical gate, error/pass-rate gates, config."""

import pytest

from elyaeval.reporting.compare import GateConfig, compare_runs, gate_config_from_settings


def _rows(scores: dict[str, float], metric="Faithfulness", errors=(), threshold=0.5, repeat=0):
    rows = []
    for gid, score in scores.items():
        err = gid in errors
        rows.append({
            "golden_id": gid, "metric_name": metric, "repeat": repeat,
            "score": None if err else score, "threshold": threshold,
            "success": False if err else score >= threshold,
            "error": "boom" if err else "", "evaluation_cost": None,
        })
    return rows


def _one(base, cand, **gate):
    (c,) = compare_runs(base, cand, gate=GateConfig(**gate))
    return c


IDS = [f"g{i}" for i in range(10)]


def test_consistent_drop_is_a_significant_regression():
    base = _rows(dict.fromkeys(IDS, 0.9))
    cand = _rows(dict.fromkeys(IDS, 0.6))
    c = _one(base, cand)
    assert c["status"] == "regressed"
    assert c["n_paired"] == 10 and c["method"] == "paired permutation"
    assert c["p_value"] < 0.01
    assert c["ci_high"] < 0


def test_noisy_mean_drop_is_not_significant():
    base = _rows(dict.fromkeys(IDS, 0.5))
    # mean delta -0.04 (beyond tolerance 0.02) but signs alternate: noise
    cand = _rows({g: 0.5 + (0.3 if i % 2 else -0.38) for i, g in enumerate(IDS)})
    c = _one(base, cand)
    assert c["delta"] < -0.02
    assert c["status"] == "unchanged"
    assert c["p_value"] > 0.05


def test_delta_gate_ignores_significance():
    base = _rows(dict.fromkeys(IDS, 0.5))
    cand = _rows({g: 0.5 + (0.3 if i % 2 else -0.38) for i, g in enumerate(IDS)})
    assert _one(base, cand, gate="delta")["status"] == "regressed"


def test_few_pairs_fall_back_to_tolerance_only():
    c = _one(_rows({"a": 0.9, "b": 0.9}), _rows({"a": 0.5, "b": 0.5}))
    assert c["status"] == "regressed"
    assert c["method"].startswith("delta only (n=2")


def test_only_paired_goldens_are_compared():
    base = _rows({"easy": 1.0, "hard": 0.2, **dict.fromkeys(IDS, 0.7)})
    cand = _rows({"easy": 1.0, **dict.fromkeys(IDS, 0.7)})  # dropped the hard golden
    c = _one(base, cand)
    assert c["n_paired"] == 11 and c["only_baseline"] == 1
    assert c["delta"] == 0 and c["status"] == "unchanged"
    # unpaired averages would have shown a fake improvement
    assert _one(base, cand, paired=False)["status"] == "improved"


def test_error_rate_increase_regresses_even_if_scores_hold():
    base = _rows(dict.fromkeys(IDS, 0.8))
    cand = _rows(dict.fromkeys(IDS, 0.8), errors={"g0", "g1"})
    c = _one(base, cand)
    assert c["status"] == "regressed"
    assert any("error rate" in r for r in c["reasons"])
    assert _one(base, cand, max_error_rate_increase=None)["status"] == "unchanged"


def test_pass_rate_gate_is_opt_in():
    base = _rows(dict.fromkeys(IDS, 0.55))
    cand = _rows(dict.fromkeys(IDS, 0.45))  # crosses the 0.5 threshold, delta -0.10
    assert _one(base, cand, tolerance=0.2)["status"] == "unchanged"
    c = _one(base, cand, tolerance=0.2, max_pass_rate_drop=0.1)
    assert c["status"] == "regressed" and any("pass rate" in r for r in c["reasons"])


def test_repeats_are_averaged_per_golden_before_pairing():
    base = _rows(dict.fromkeys(IDS, 0.8), repeat=0) + _rows(dict.fromkeys(IDS, 0.6), repeat=1)
    cand = _rows(dict.fromkeys(IDS, 0.7))
    c = _one(base, cand)
    assert c["n_paired"] == 10
    assert c["baseline_avg"] == pytest.approx(0.7)
    assert c["status"] == "unchanged"


def test_per_metric_tolerances():
    base = _rows(dict.fromkeys(IDS, 0.9), metric="Faithfulness") + _rows(dict.fromkeys(IDS, 0.9), metric="Answer Relevancy")
    cand = _rows(dict.fromkeys(IDS, 0.85), metric="Faithfulness") + _rows(dict.fromkeys(IDS, 0.85), metric="Answer Relevancy")
    out = {c["metric_name"]: c for c in compare_runs(base, cand, gate=GateConfig(tolerances={"Faithfulness": 0.1}))}
    assert out["Faithfulness"]["status"] == "unchanged"
    assert out["Answer Relevancy"]["status"] == "regressed"


def test_new_and_removed_groups():
    base = _rows({"a": 0.5}, metric="Old")
    cand = _rows({"a": 0.5}, metric="New")
    statuses = {c["metric_name"]: c["status"] for c in compare_runs(base, cand)}
    assert statuses == {"Old": "removed", "New": "new"}


def test_gate_config_from_pyproject(isolated_config):
    (isolated_config / "pyproject.toml").write_text(
        "[tool.elyaeval.regression]\ntolerance = 0.05\nalpha = 0.1\nmax_pass_rate_drop = 0.2\n"
        'tolerances = { Faithfulness = 0.08 }\n',
        encoding="utf-8",
    )
    cfg = gate_config_from_settings(alpha=0.01)
    assert (cfg.tolerance, cfg.alpha, cfg.max_pass_rate_drop) == (0.05, 0.01, 0.2)
    assert cfg.tolerance_for("Faithfulness", "Faithfulness") == 0.08


def test_gate_config_rejects_unknown_keys(isolated_config):
    (isolated_config / "pyproject.toml").write_text("[tool.elyaeval.regression]\ntolerence = 0.05\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tolerence"):
        gate_config_from_settings()
