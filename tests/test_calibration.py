"""Calibration: noise-based tolerances and judge-vs-human agreement."""

import pytest

from elyaeval.reporting.calibration import agreement_report, metric_key_for, noise_report, read_labels


def _row(gid, score, repeat=0, metric="Faithfulness", threshold=0.5):
    return {"golden_id": gid, "metric_name": metric, "score": score, "threshold": threshold,
            "success": score >= threshold, "error": "", "repeat": repeat}


def test_noise_report_from_repeats():
    rows = [_row("a", 0.4, 0), _row("a", 0.6, 1), _row("b", 0.8, 0), _row("b", 0.8, 1)]
    (r,) = noise_report(rows)
    assert r["n_goldens"] == 2 and r["repeats"] == 2
    assert r["noise_sd"] == pytest.approx(0.1)
    assert r["recommended_tolerance"] > 0


def test_noise_report_without_repeats():
    (r,) = noise_report([_row("a", 0.4), _row("b", 0.6)])
    assert r["noise_sd"] is None and r["recommended_tolerance"] is None


def test_agreement_and_best_threshold():
    # judge scores separate cleanly at ~0.65, but the configured threshold is 0.5
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.55, "f": 0.2}
    human = {"a": True, "b": True, "c": True, "d": False, "e": False, "f": False}
    rows = [_row(g, s) for g, s in scores.items()]
    labels = [{"golden_id": g, "metric_name": "Faithfulness", "human_pass": h, "human_score": None} for g, h in human.items()]
    (r,) = agreement_report(rows, labels)
    assert r["n"] == 6
    assert r["accuracy"] == pytest.approx(4 / 6)  # d and e wrongly pass at 0.5
    assert r["best_threshold"] == pytest.approx(0.65) and r["best_balanced_accuracy"] == 1.0


def test_read_labels_validates(tmp_path):
    good = tmp_path / "labels.csv"
    good.write_text("golden_id,metric_name,human_pass,human_score\ng1,Faithfulness,pass,0.9\ng2,Faithfulness,0,\n", encoding="utf-8")
    labels = read_labels(good)
    assert [lab["human_pass"] for lab in labels] == [True, False]
    assert labels[0]["human_score"] == 0.9 and labels[1]["human_score"] is None

    bad = tmp_path / "bad.csv"
    bad.write_text("golden_id,metric_name,human_pass\ng1,F,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maybe"):
        read_labels(bad)
    missing = tmp_path / "missing.csv"
    missing.write_text("golden_id,label\ng1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        read_labels(missing)


def test_metric_key_for():
    assert metric_key_for("Contextual Precision") == "contextual_precision"
