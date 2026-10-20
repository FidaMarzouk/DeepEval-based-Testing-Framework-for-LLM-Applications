"""
Judge-result cache: skip re-judging (metric, test case) pairs that were
already scored, by the same judge, with the same metric configuration.

The expensive part of a suite is the judge. Re-running a suite after an
unrelated change, or re-rendering a report, shouldn't re-pay for — or, on a
local GPU, re-wait for — identical judgements. The key covers everything
that can change a verdict:

  - the judge (provider:model, temperature) and the metric's evaluation
    model name;
  - the metric class, name, threshold, strictness and rubric-like settings
    (G-Eval criteria/steps/params, ...);
  - the full test case (input, actual/expected output, contexts, tools);
  - the repeat index, so --elyaeval-repeats still measures run-to-run noise;
  - the DeepEval version (its metric prompts change between releases).

Only error-free results are stored, so a timeout is retried next run.
Cached rows are written with cost/tokens 0 (nothing was spent this run)
and cached=True.

Off by default. Enable per project or per run:

    [tool.elyaeval.cache]
    enabled = true
    path = ".elyaeval_cache/results.sqlite"     # default

    ELYAEVAL_CACHE=1 pytest ...      # or =0 to force off

SQLite (stdlib) in WAL mode: safe for concurrent pytest-xdist workers.
Applies to e2e evaluation (evaluate_golden); component/span metrics run
inside DeepEval's tracing and are not cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

from .config import config_section

CACHE_VERSION = 1
DEFAULT_CACHE_PATH = ".elyaeval_cache/results.sqlite"
CACHE_ENV = "ELYAEVAL_CACHE"
CACHE_PATH_ENV = "ELYAEVAL_CACHE_PATH"

# metric attributes that change what the judge is asked (when present)
_METRIC_SETTINGS = (
    "threshold",
    "strict_mode",
    "include_reason",
    "criteria",
    "evaluation_steps",
    "evaluation_params",
    "rubric",
    "assessment_questions",
    "truths_extraction_limit",
    "penalize_ambiguous_claims",
    "top_logprobs",
    "n",
    "max_tool_calls",
    "max_identical",
)
_TEST_CASE_FIELDS = (
    "input",
    "actual_output",
    "expected_output",
    "context",
    "retrieval_context",
    "tools_called",
    "expected_tools",
)


class ResultCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS results ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, key: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM results WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO results (key, value, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, default=str), datetime.now(timezone.utc).isoformat()),
            )

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM results").fetchone()[0])

    def clear(self) -> int:
        with self._connect() as conn:
            n = int(conn.execute("SELECT COUNT(*) FROM results").fetchone()[0])
            conn.execute("DELETE FROM results")
        return n


def cache_settings() -> tuple[bool, Path]:
    section = config_section("cache")
    enabled = bool(section.get("enabled", False))
    env = os.environ.get(CACHE_ENV)
    if env not in (None, ""):
        enabled = env.strip().lower() in ("1", "true", "yes", "on")
    path = Path(os.environ.get(CACHE_PATH_ENV) or section.get("path") or DEFAULT_CACHE_PATH)
    return enabled, path


_CACHES: dict[str, ResultCache] = {}


def get_result_cache() -> Optional[ResultCache]:
    enabled, path = cache_settings()
    if not enabled:
        return None
    key = str(path.resolve())
    if key not in _CACHES:
        _CACHES[key] = ResultCache(path)
    return _CACHES[key]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):  # enums
        return value.value
    return value


def judge_identity() -> str:
    """Secret-free identity of the configured judge (provider:model@temp)."""
    from .judge import JudgeConfigError, resolve_judge_config

    try:
        cfg = resolve_judge_config()
    except JudgeConfigError:
        return "invalid-config"
    return f"{cfg.describe()}@{cfg.temperature}|{cfg.base_url or ''}"


def is_cacheable(metric: Any) -> bool:
    """Metrics can opt out with `elyaeval_cacheable = False`   e.g.
    deterministic checks, which are free to recompute and whose settings a
    generic cache key can't fully know."""
    return bool(getattr(metric, "elyaeval_cacheable", True))


def cache_key(metric: Any, test_case: Any, judge: str, repeat: int = 0) -> str:
    try:
        deepeval_version = version("deepeval")
    except PackageNotFoundError:
        deepeval_version = None
    payload = {
        "v": CACHE_VERSION,
        "deepeval": deepeval_version,
        "judge": judge,
        "evaluation_model": getattr(metric, "evaluation_model", None),
        "metric_class": f"{type(metric).__module__}.{type(metric).__qualname__}",
        "metric_name": getattr(metric, "__name__", type(metric).__name__),
        "metric_settings": {a: _jsonable(getattr(metric, a)) for a in _METRIC_SETTINGS if hasattr(metric, a)},
        "test_case": {f: _jsonable(getattr(test_case, f, None)) for f in _TEST_CASE_FIELDS},
        "repeat": repeat,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
