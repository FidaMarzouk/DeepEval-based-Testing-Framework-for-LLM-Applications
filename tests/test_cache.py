"""Judge-result cache: keys, storage, and evaluate_golden integration."""

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase

from conftest import ExactMatchMetric
from elyaeval.core.cache import ResultCache, cache_key, cache_settings, get_result_cache
from elyaeval.reporting.html_report import read_csv_rows
from elyaeval.reporting.report import evaluate_golden


class CountingMetric(ExactMatchMetric):
    calls = 0

    def measure(self, test_case, *args, **kwargs):
        type(self).calls += 1
        return super().measure(test_case)


class FailingMetric(ExactMatchMetric):
    calls = 0

    def measure(self, test_case, *args, **kwargs):
        type(self).calls += 1
        raise RuntimeError("judge timed out")

    @property
    def __name__(self):
        return "Flaky Judge"


def _tc(actual="A"):
    return LLMTestCase(input="q", actual_output=actual, expected_output="A")


def test_cache_key_sensitivity():
    m = ExactMatchMetric()
    base = cache_key(m, _tc(), "ollama:llama3.1@0.0", 0)
    assert base == cache_key(ExactMatchMetric(), _tc(), "ollama:llama3.1@0.0", 0)
    assert base != cache_key(m, _tc("B"), "ollama:llama3.1@0.0", 0)          # output changed
    assert base != cache_key(m, _tc(), "anthropic:claude@0.0", 0)            # judge changed
    assert base != cache_key(m, _tc(), "ollama:llama3.1@0.0", 1)             # other repeat
    assert base != cache_key(ExactMatchMetric(threshold=0.9), _tc(), "ollama:llama3.1@0.0", 0)


def test_result_cache_storage(tmp_path):
    cache = ResultCache(tmp_path / "c.sqlite")
    assert cache.get("k") is None
    cache.put("k", {"name": "M", "score": 1.0})
    assert cache.get("k") == {"name": "M", "score": 1.0}
    assert cache.count() == 1 and cache.clear() == 1 and cache.count() == 0


def test_cache_settings_env_and_config(isolated_config, monkeypatch):
    assert cache_settings()[0] is False
    (isolated_config / "pyproject.toml").write_text('[tool.elyaeval.cache]\nenabled = true\npath = "x/c.sqlite"\n', encoding="utf-8")
    from elyaeval.core import config

    config.load_config.cache_clear()
    assert cache_settings() == (True, __import__("pathlib").Path("x/c.sqlite"))
    monkeypatch.setenv("ELYAEVAL_CACHE", "0")
    assert get_result_cache() is None


def test_second_run_is_served_from_cache(isolated_config, monkeypatch):
    monkeypatch.setenv("ELYAEVAL_CACHE", "1")
    golden = Golden(input="q", expected_output="A", additional_metadata={"id": "g1"})
    CountingMetric.calls = 0
    csv_path = isolated_config / "results.csv"

    first = evaluate_golden(golden, _tc(), [CountingMetric()], csv_path=csv_path)
    second = evaluate_golden(golden, _tc(), [CountingMetric()], csv_path=csv_path)
    assert CountingMetric.calls == 1
    assert first.success and second.success
    rows = read_csv_rows(csv_path)
    assert [r["cached"] for r in rows] == [False, True]
    assert rows[1]["score"] == 1.0 and rows[1]["evaluation_cost"] == 0.0

    evaluate_golden(golden, _tc("B"), [CountingMetric()], csv_path=csv_path)  # different output
    assert CountingMetric.calls == 2


def test_errors_are_never_cached(isolated_config, monkeypatch):
    monkeypatch.setenv("ELYAEVAL_CACHE", "1")
    golden = Golden(input="q", expected_output="A", additional_metadata={"id": "g1"})
    FailingMetric.calls = 0
    for _ in range(2):
        result = evaluate_golden(golden, _tc(), [FailingMetric()], csv_path=isolated_config / "r.csv")
        assert not result.success
    assert FailingMetric.calls == 2


def test_partial_hits_merge_in_metric_order(isolated_config, monkeypatch):
    monkeypatch.setenv("ELYAEVAL_CACHE", "1")
    golden = Golden(input="q", expected_output="A", additional_metadata={"id": "g1"})
    CountingMetric.calls = 0
    evaluate_golden(golden, _tc(), [CountingMetric()], csv_path=isolated_config / "r.csv")

    class Second(ExactMatchMetric):
        @property
        def __name__(self):
            return "Second"

    result = evaluate_golden(golden, _tc(), [CountingMetric(), Second()], csv_path=isolated_config / "r.csv")
    assert [md.name for md in result.metrics_data] == ["Exact Match", "Second"]
    assert CountingMetric.calls == 1


def test_deterministic_guards_are_never_cached(isolated_config, monkeypatch):
    from elyaeval.core.agentic import ToolCallBudgetMetric
    from elyaeval.core.cache import is_cacheable

    assert not is_cacheable(ToolCallBudgetMetric(0)) and is_cacheable(ExactMatchMetric())
    monkeypatch.setenv("ELYAEVAL_CACHE", "1")
    golden = Golden(input="q", expected_output="A", additional_metadata={"id": "g1"})
    tc = LLMTestCase(input="q", actual_output="A", expected_output="A",
                     tools_called=[__import__("deepeval.test_case", fromlist=["ToolCall"]).ToolCall(name="t")])
    first = evaluate_golden(golden, tc, [ToolCallBudgetMetric(8)], csv_path=isolated_config / "r.csv")
    second = evaluate_golden(golden, tc, [ToolCallBudgetMetric(0)], csv_path=isolated_config / "r.csv")
    assert first.success and not second.success  # the stricter budget is applied, not a cached pass
    assert [r["cached"] for r in read_csv_rows(isolated_config / "r.csv")] == [False, False]
