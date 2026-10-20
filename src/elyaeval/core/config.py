"""
Project-level elyaeval configuration.

Read from (first match wins, searching from the current directory upward):
  1. elyaeval.toml            — whole file is the elyaeval config
  2. pyproject.toml           — the [tool.elyaeval] table

Example (pyproject.toml of the system under test):

    [tool.elyaeval.judge]
    provider = "ollama"          # see elyaeval.core.judge.PROVIDERS
    model = "llama3.1"
    base_url = "http://localhost:11434"
    temperature = 0.0

    [tool.elyaeval.thresholds]
    default = 0.7
    faithfulness = 0.8           # keys are metric keys, see elyaeval.core.metrics

Secrets (API keys) never belong in these files — set them as environment
variables (ELYAEVAL_JUDGE_API_KEY, or the provider's own variable such as
ANTHROPIC_API_KEY / OPENAI_API_KEY, which DeepEval reads natively).
"""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

CONFIG_PATH_ENV = "ELYAEVAL_CONFIG"


def _find_config_file(start: Path) -> tuple[Path, bool] | None:
    """Returns (path, is_pyproject) for the nearest config file, or None."""
    for directory in (start, *start.parents):
        dedicated = directory / "elyaeval.toml"
        if dedicated.is_file():
            return dedicated, False
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            with open(pyproject, "rb") as f:
                if "elyaeval" in tomllib.load(f).get("tool", {}):
                    return pyproject, True
    return None


@cache
def _load(start: str) -> dict[str, Any]:
    explicit = os.environ.get(CONFIG_PATH_ENV)
    if explicit:
        path = Path(explicit)
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return data.get("tool", {}).get("elyaeval", data) if path.name == "pyproject.toml" else data

    found = _find_config_file(Path(start).resolve())
    if found is None:
        return {}
    path, is_pyproject = found
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data["tool"]["elyaeval"] if is_pyproject else data


def load_config(start: str | Path | None = None) -> dict[str, Any]:
    """The elyaeval config table ({} if none). Cached per start directory;
    call load_config.cache_clear() after changing files/env in tests."""
    return _load(str(start or Path.cwd()))


load_config.cache_clear = _load.cache_clear  # type: ignore[attr-defined]


def config_section(name: str, start: str | Path | None = None) -> dict[str, Any]:
    section = load_config(start).get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"[tool.elyaeval.{name}] must be a table, got {type(section).__name__}")
    return section
