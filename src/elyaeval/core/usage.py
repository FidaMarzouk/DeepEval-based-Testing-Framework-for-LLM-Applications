"""
Usage metering: what a run actually consumed, independent of the provider.

DeepEval's own usage numbers are only as good as the provider integration:
hosted APIs report cost and tokens, but e.g. the Ollama judge reports cost
0.0 and 0 tokens. So elyaeval measures for itself:

  - every judge call made through a judge built by elyaeval.core.judge is
    instrumented (instrument_judge): call count, wall time, and token
    estimates from prompt/response size (~4 characters per token);
  - every golden evaluation records its judge cost/tokens as reported by
    DeepEval (when real), the judging wall time, and the SUT's usage
    (tokens/cost/latency) if the test passes it in;
  - optional rate limiting (max calls / estimated tokens per minute) for
    rate-limited free tiers, applied at the same instrumentation point.

The meter is process-global (one per pytest worker); the pytest plugin
merges xdist workers' ledgers and writes report/budget_<run>.json.
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: Any) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = text.model_dump_json() if hasattr(text, "model_dump_json") else str(text)
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass
class GoldenUsage:
    golden_id: str
    repeat: int = 0
    priority: str = ""
    n_metrics: int = 0
    n_cached: int = 0
    judge_seconds: float = 0.0
    judge_cost_reported: Optional[float] = None
    judge_tokens_reported: Optional[int] = None
    sut_seconds: Optional[float] = None
    sut_input_tokens: Optional[int] = None
    sut_output_tokens: Optional[int] = None
    sut_cost: Optional[float] = None


@dataclass
class UsageMeter:
    judge_calls: int = 0
    judge_call_seconds: float = 0.0
    judge_prompt_tokens_estimated: int = 0
    judge_output_tokens_estimated: int = 0
    judge_call_errors: int = 0
    goldens: list[GoldenUsage] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- recording ---------------------------------------------------------
    def record_judge_call(self, seconds: float, prompt_tokens: int, output_tokens: int, ok: bool) -> None:
        with self._lock:
            self.judge_calls += 1
            self.judge_call_seconds += seconds
            self.judge_prompt_tokens_estimated += prompt_tokens
            self.judge_output_tokens_estimated += output_tokens
            if not ok:
                self.judge_call_errors += 1

    def record_golden(self, usage: GoldenUsage) -> None:
        with self._lock:
            self.goldens.append(usage)

    # -- reading -----------------------------------------------------------
    def totals(self) -> dict[str, Any]:
        with self._lock:
            goldens = list(self.goldens)
            out: dict[str, Any] = {
                "judge_calls": self.judge_calls,
                "judge_call_seconds": self.judge_call_seconds,
                "judge_prompt_tokens_estimated": self.judge_prompt_tokens_estimated,
                "judge_output_tokens_estimated": self.judge_output_tokens_estimated,
                "judge_call_errors": self.judge_call_errors,
            }
        reported_costs = [g.judge_cost_reported for g in goldens if g.judge_cost_reported is not None]
        reported_tokens = [g.judge_tokens_reported for g in goldens if g.judge_tokens_reported is not None]
        sut_costs = [g.sut_cost for g in goldens if g.sut_cost is not None]
        out.update(
            goldens_evaluated=len(goldens),
            metric_evaluations=sum(g.n_metrics for g in goldens),
            cached_metric_evaluations=sum(g.n_cached for g in goldens),
            judge_seconds=sum(g.judge_seconds for g in goldens),
            judge_cost_reported=sum(reported_costs) if reported_costs else None,
            judge_tokens_reported=sum(reported_tokens) if reported_tokens else None,
            sut_seconds=sum(g.sut_seconds or 0.0 for g in goldens) if any(g.sut_seconds is not None for g in goldens) else None,
            sut_input_tokens=_sum_opt(g.sut_input_tokens for g in goldens),
            sut_output_tokens=_sum_opt(g.sut_output_tokens for g in goldens),
            sut_cost=sum(sut_costs) if sut_costs else None,
        )
        return out

    def ledger(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(g) for g in self.goldens]

    # -- xdist -------------------------------------------------------------
    def worker_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "judge_calls": self.judge_calls,
                "judge_call_seconds": self.judge_call_seconds,
                "judge_prompt_tokens_estimated": self.judge_prompt_tokens_estimated,
                "judge_output_tokens_estimated": self.judge_output_tokens_estimated,
                "judge_call_errors": self.judge_call_errors,
                "goldens": [asdict(g) for g in self.goldens],
            }

    def merge(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.judge_calls += payload.get("judge_calls", 0)
            self.judge_call_seconds += payload.get("judge_call_seconds", 0.0)
            self.judge_prompt_tokens_estimated += payload.get("judge_prompt_tokens_estimated", 0)
            self.judge_output_tokens_estimated += payload.get("judge_output_tokens_estimated", 0)
            self.judge_call_errors += payload.get("judge_call_errors", 0)
            self.goldens.extend(GoldenUsage(**g) for g in payload.get("goldens", []))

    def reset(self) -> None:
        with self._lock:
            self.judge_calls = 0
            self.judge_call_seconds = 0.0
            self.judge_prompt_tokens_estimated = 0
            self.judge_output_tokens_estimated = 0
            self.judge_call_errors = 0
            self.goldens.clear()


def _sum_opt(values) -> Optional[int]:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def reported_judge_usage(metrics_data: list[Any], cached_names: frozenset[str] = frozenset()) -> tuple[Optional[float], Optional[int]]:
    """(cost, tokens) DeepEval reported for freshly judged metrics, or None
    for each when nothing real was reported. A cost of 0.0 with 0 tokens is
    what local/unpriced judges report (e.g. Ollama) — that's "unknown", not
    "free", so it maps to None."""
    fresh = [md for md in metrics_data if md.name not in cached_names]
    costs = [md.evaluation_cost for md in fresh if md.evaluation_cost is not None]
    tokens = [(md.input_tokens or 0) + (md.output_tokens or 0) for md in fresh
              if md.input_tokens is not None or md.output_tokens is not None]
    total_tokens = sum(tokens) if tokens else None
    total_cost = sum(costs) if costs else None
    if not total_tokens and not total_cost:
        return None, None
    return total_cost, total_tokens


METER = UsageMeter()


class RateLimiter:
    """Sliding 60 s window over calls and (estimated) tokens. Thread-safe;
    blocks (or awaits) until the next call fits."""

    def __init__(self, max_calls_per_minute: Optional[int], max_tokens_per_minute: Optional[int], clock=time.monotonic):
        self.max_calls = max_calls_per_minute
        self.max_tokens = max_tokens_per_minute
        self._clock = clock
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _wait_time(self, tokens: int) -> float:
        now = self._clock()
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()
        waits = [0.0]
        if self.max_calls is not None and len(self._events) >= self.max_calls:
            waits.append(60.0 - (now - self._events[len(self._events) - self.max_calls][0]))
        if self.max_tokens is not None:
            used = sum(t for _, t in self._events)
            if used + tokens > self.max_tokens and self._events:
                # wait until enough old events expire to fit this call
                freed, i = 0, 0
                while i < len(self._events) and used - freed + tokens > self.max_tokens:
                    freed += self._events[i][1]
                    i += 1
                waits.append(60.0 - (now - self._events[i - 1][0]))
        return max(waits)

    def acquire(self, tokens: int = 0) -> float:
        waited = 0.0
        while True:
            with self._lock:
                wait = self._wait_time(tokens)
                if wait <= 0:
                    self._events.append((self._clock(), tokens))
                    return waited
            time.sleep(wait)
            waited += wait

    async def a_acquire(self, tokens: int = 0) -> float:
        waited = 0.0
        while True:
            with self._lock:
                wait = self._wait_time(tokens)
                if wait <= 0:
                    self._events.append((self._clock(), tokens))
                    return waited
            await asyncio.sleep(wait)
            waited += wait


_INSTRUMENTED_METHODS = ("generate", "generate_raw_response")
_ASYNC_INSTRUMENTED_METHODS = ("a_generate", "a_generate_raw_response")


def _result_payload(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) and result else result


def instrument_judge(judge: Any, limiter: Optional[RateLimiter] = None, meter: UsageMeter = METER) -> Any:
    """Shadow the judge INSTANCE's generate methods with metered (and
    optionally rate-limited) wrappers. The class is untouched, so DeepEval's
    native-model detection — and with it its cost accounting — is unchanged."""
    if getattr(judge, "_elyaeval_instrumented", False):
        return judge

    def wrap_sync(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
            prompt_tokens = estimate_tokens(prompt)
            if limiter is not None:
                limiter.acquire(prompt_tokens)
            start = time.perf_counter()
            try:
                result = fn(prompt, *args, **kwargs)
            except BaseException:
                meter.record_judge_call(time.perf_counter() - start, prompt_tokens, 0, ok=False)
                raise
            meter.record_judge_call(time.perf_counter() - start, prompt_tokens, estimate_tokens(_result_payload(result)), ok=True)
            return result

        return wrapper

    def wrap_async(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
            prompt_tokens = estimate_tokens(prompt)
            if limiter is not None:
                await limiter.a_acquire(prompt_tokens)
            start = time.perf_counter()
            try:
                result = await fn(prompt, *args, **kwargs)
            except BaseException:
                meter.record_judge_call(time.perf_counter() - start, prompt_tokens, 0, ok=False)
                raise
            meter.record_judge_call(time.perf_counter() - start, prompt_tokens, estimate_tokens(_result_payload(result)), ok=True)
            return result

        return wrapper

    for name in _INSTRUMENTED_METHODS:
        if callable(getattr(judge, name, None)):
            object.__setattr__(judge, name, wrap_sync(getattr(judge, name)))
    for name in _ASYNC_INSTRUMENTED_METHODS:
        if callable(getattr(judge, name, None)):
            object.__setattr__(judge, name, wrap_async(getattr(judge, name)))
    object.__setattr__(judge, "_elyaeval_instrumented", True)
    return judge
