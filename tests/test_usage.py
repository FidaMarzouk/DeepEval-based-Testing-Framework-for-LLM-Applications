"""Usage metering, judge instrumentation and rate limiting."""

from types import SimpleNamespace

import pytest

from elyaeval.core.usage import RateLimiter, UsageMeter, estimate_tokens, instrument_judge, reported_judge_usage


class EchoJudge:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, schema=None):
        self.calls += 1
        if prompt == "boom":
            raise RuntimeError("provider error")
        return ("x" * 40, 0.0)

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


def test_estimate_tokens():
    assert estimate_tokens(None) == 0
    assert estimate_tokens("a" * 400) == 101
    assert estimate_tokens(SimpleNamespace(model_dump_json=lambda: "b" * 40)) == 11


def test_instrumented_judge_is_metered_and_keeps_its_class():
    meter = UsageMeter()
    judge = instrument_judge(EchoJudge(), meter=meter)
    assert type(judge) is EchoJudge
    judge.generate("p" * 400)
    with pytest.raises(RuntimeError):
        judge.generate("boom")
    assert meter.judge_calls == 2 and meter.judge_call_errors == 1
    assert meter.judge_prompt_tokens_estimated == 101 + 2
    assert meter.judge_output_tokens_estimated == 11
    assert instrument_judge(judge, meter=meter) is judge  # idempotent


def test_instrumented_async_generate():
    import asyncio

    meter = UsageMeter()
    judge = instrument_judge(EchoJudge(), meter=meter)
    asyncio.run(judge.a_generate("hello"))
    assert meter.judge_calls == 2  # a_generate -> instrumented generate as well


def test_ollama_judge_stays_native_after_instrumentation():
    from deepeval.metrics.utils import initialize_model

    from elyaeval.core.judge import get_judge

    judge = get_judge(provider="ollama", model="llama3.1")
    assert judge._elyaeval_instrumented
    _, native = initialize_model(judge)
    assert native, "instrumentation must not change DeepEval's native-model detection (cost accounting)"


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_rate_limiter_calls_per_minute():
    clock = FakeClock()
    limiter = RateLimiter(max_calls_per_minute=2, max_tokens_per_minute=None, clock=clock)
    limiter._events.append((0.0, 0))
    limiter._events.append((10.0, 0))
    clock.now = 20.0
    assert limiter._wait_time(0) == pytest.approx(40.0)  # first event expires at t=60
    clock.now = 61.0
    assert limiter._wait_time(0) == 0.0


def test_rate_limiter_tokens_per_minute():
    clock = FakeClock()
    limiter = RateLimiter(max_calls_per_minute=None, max_tokens_per_minute=1000, clock=clock)
    limiter._events.extend([(0.0, 600), (30.0, 300)])
    clock.now = 40.0
    assert limiter._wait_time(50) == 0.0           # 950 fits
    assert limiter._wait_time(200) == pytest.approx(20.0)   # needs the t=0 event to expire
    assert limiter._wait_time(5000) == pytest.approx(50.0)  # can never fit: wait for the window to clear


def test_reported_judge_usage_treats_zero_zero_as_unknown():
    def md(name, cost, tin, tout):
        return SimpleNamespace(name=name, evaluation_cost=cost, input_tokens=tin, output_tokens=tout)

    assert reported_judge_usage([md("A", 0.0, 0, 0), md("B", 0.0, 0, 0)]) == (None, None)  # e.g. Ollama
    assert reported_judge_usage([md("A", 0.002, 300, 50), md("B", 0.001, 100, 20)]) == (pytest.approx(0.003), 470)
    assert reported_judge_usage([md("A", 0.002, 300, 50)], frozenset({"A"})) == (None, None)  # cached
