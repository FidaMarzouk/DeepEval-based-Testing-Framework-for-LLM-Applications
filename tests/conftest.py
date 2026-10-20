"""
Shared fixtures for elyaeval's self-tests. Nothing here calls an LLM:

  - FakeJudge: a DeepEvalBaseLLM that satisfies DeepEval's model checks
    (so metric presets can be constructed against it) and fails loudly if
    anything actually tries to generate with it.
  - ExactMatchMetric: a deterministic BaseMetric, used to drive DeepEval's
    real evaluate()/evals_iterator() code paths end to end offline.
"""

from __future__ import annotations

import pytest
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM

pytest_plugins = ["pytester"]


class FakeJudge(DeepEvalBaseLLM):
    def __init__(self, name: str = "fake-judge"):
        self._name = name
        super().__init__(name)

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, **kwargs):
        raise AssertionError("FakeJudge must never be called in offline tests")

    async def a_generate(self, prompt, schema=None, **kwargs):
        raise AssertionError("FakeJudge must never be called in offline tests")

    def get_model_name(self):
        return self._name


class ExactMatchMetric(BaseMetric):
    """score 1.0 if actual_output == expected_output else 0.0."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.evaluation_model = "exact-match"
        self.evaluation_cost = None
        self.include_reason = True
        self.async_mode = False
        self.strict_mode = False
        self.verbose_mode = False

    def measure(self, test_case, *args, **kwargs) -> float:
        self.score = 1.0 if test_case.actual_output == test_case.expected_output else 0.0
        self.reason = "exact match" if self.score else "mismatch"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Exact Match"


@pytest.fixture
def fake_judge() -> FakeJudge:
    return FakeJudge()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Every test runs in its own empty directory with no ELYAEVAL_* env and
    fresh config/judge caches, so a developer's real pyproject or env vars
    can't leak into results."""
    import os

    from elyaeval.core import config, judge

    for key in list(os.environ):
        if key.startswith("ELYAEVAL_") or key in ("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "DEEPEVAL_DISABLE_TIMEOUTS"):
            monkeypatch.delenv(key)
    # apply_timeout() writes these; register them so monkeypatch restores them
    monkeypatch.setenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "")
    monkeypatch.delenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE")
    monkeypatch.setenv("DEEPEVAL_DISABLE_TIMEOUTS", "")
    monkeypatch.delenv("DEEPEVAL_DISABLE_TIMEOUTS")
    monkeypatch.chdir(tmp_path)
    config.load_config.cache_clear()
    judge._build.cache_clear()
    yield tmp_path
    config.load_config.cache_clear()
    judge._build.cache_clear()
