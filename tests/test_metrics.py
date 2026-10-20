"""Metric factories: fresh instances, configured judge, configurable thresholds."""

import pytest

from elyaeval.core import metrics
from elyaeval.core.metrics import PRESETS, build_metric, metrics_for, rag_metrics


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_builds_with_a_judge(preset, fake_judge):
    built = metrics_for(preset, model=fake_judge)
    assert len(built) == len(PRESETS[preset])


def test_factories_return_fresh_instances(fake_judge):
    a, b = rag_metrics(model=fake_judge), rag_metrics(model=fake_judge)
    assert all(x is not y for x, y in zip(a, b, strict=True))


def test_configured_judge_is_used_by_default(fake_judge, monkeypatch):
    monkeypatch.setattr(metrics, "get_judge", lambda: fake_judge)
    (metric,) = metrics_for("summarization")
    assert metric.model is fake_judge


def test_threshold_precedence(isolated_config, fake_judge):
    (isolated_config / "pyproject.toml").write_text(
        "[tool.elyaeval.thresholds]\ndefault = 0.6\nfaithfulness = 0.8\n", encoding="utf-8"
    )
    by_key = dict(zip(PRESETS["generation"], metrics_for("generation", model=fake_judge), strict=True))
    assert by_key["faithfulness"].threshold == 0.8
    assert by_key["answer_relevancy"].threshold == 0.6
    overridden = build_metric("faithfulness", model=fake_judge, thresholds={"faithfulness": 0.9})
    assert overridden.threshold == 0.9
    assert build_metric("faithfulness", model=fake_judge, threshold=0.5).threshold == 0.5


def test_default_threshold_without_config(fake_judge):
    assert build_metric("faithfulness", model=fake_judge).threshold == metrics.DEFAULT_THRESHOLD


def test_unknown_metric_and_preset():
    with pytest.raises(KeyError):
        build_metric("vibes")
    with pytest.raises(KeyError):
        metrics_for("vibes")


def test_deprecated_constants_still_resolve(fake_judge, monkeypatch):
    monkeypatch.setattr(metrics, "get_judge", lambda: fake_judge)
    with pytest.warns(DeprecationWarning, match="rag_metrics"):
        from elyaeval.core.metrics import RAG_METRICS
    assert len(RAG_METRICS) == len(PRESETS["rag_qa"])


def test_importing_metrics_constructs_nothing():
    """Module import must not need a judge (no metric objects at import time)."""
    import subprocess
    import sys

    code = (
        "import elyaeval.core.metrics as m, sys; "
        "sys.exit(0 if not any(isinstance(v, list) for k, v in vars(m).items() if k.isupper() and k.endswith('_METRICS')) else 1)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_metrics_for_golden_routes_abstentions(fake_judge):
    from types import SimpleNamespace

    from elyaeval.core.metrics import metrics_for_golden

    abstain = SimpleNamespace(additional_metadata={"expected_behavior": "abstain"})
    answer = SimpleNamespace(additional_metadata={"expected_behavior": "answer"})
    assert [m.__name__ for m in metrics_for_golden(abstain, rag_metrics, model=fake_judge)] == ["Abstention [GEval]"]
    assert len(metrics_for_golden(answer, rag_metrics, model=fake_judge)) == len(PRESETS["rag_qa"])
