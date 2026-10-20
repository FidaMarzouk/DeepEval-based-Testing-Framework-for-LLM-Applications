"""
Run metadata: what produced a results CSV.

Every results CSV gets a sidecar `<name>.meta.json` written by the pytest
plugin at session end, recording the judge, the dataset(s) and their
content hashes, framework versions, git SHA, CI stage, repeats, and any
SUT tags (model, prompt version, ...). `elyaeval compare` reads both sides'
sidecars and:

  - REFUSES to compare runs scored by different judges (or judge
    temperatures): a score from llama3.1 and one from a hosted model are
    not on the same scale, so the "regression" would be meaningless;
  - WARNS when datasets/versions/stage/repeats differ (paired comparison
    already restricts itself to goldens present in both runs);
  - REPORTS SUT tags that changed — that's usually the point of the run
    (new prompt, new model), not a reason to refuse.

SUT tags come from record_run_metadata(...) in a test file, or from
ELYAEVAL_META_<KEY> environment variables (handy in CI):

    ELYAEVAL_META_SUT_MODEL=llama-3.3-70b ELYAEVAL_META_PROMPT_VERSION=v7 pytest ...

Must stay import-light (the pytest plugin imports it at startup).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1
META_ENV_PREFIX = "ELYAEVAL_META_"

# keys whose mismatch makes a comparison meaningless -> refuse
STRICT_KEYS = ("judge", "judge_temperature")
# keys whose mismatch deserves a warning, not a refusal
WARN_KEYS = ("dataset_sha256", "deepeval_version", "elyaeval_version", "ci_stage", "repeats")

_USER_META: dict[str, Any] = {}
_DATASETS: dict[str, dict[str, Any]] = {}


def record_run_metadata(**tags: Any) -> None:
    """Attach SUT tags (sut_model=..., prompt_version=..., ...) to this run's
    metadata. Call at module level in a test file."""
    _USER_META.update({k: v for k, v in tags.items() if v is not None})


def register_dataset(path: str | Path, task_type: Optional[str], ci_stage: Optional[str], n_goldens: int) -> None:
    """Called by load_standard_dataset() for every goldens file it loads."""
    p = Path(path)
    try:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        digest = None
    _DATASETS[str(p)] = {
        "path": p.name,
        "sha256": digest,
        "task_type": task_type,
        "ci_stage": ci_stage,
        "n_goldens": n_goldens,
    }


def clear() -> None:
    _USER_META.clear()
    _DATASETS.clear()


def worker_payload() -> dict[str, Any]:
    """What an xdist worker sends back to the controller."""
    return {"user": dict(_USER_META), "datasets": dict(_DATASETS)}


def merge_worker_payload(payload: dict[str, Any]) -> None:
    _USER_META.update(payload.get("user", {}))
    _DATASETS.update(payload.get("datasets", {}))


def _pkg_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_sha() -> Optional[str]:
    for var in ("ELYAEVAL_GIT_SHA", "GIT_COMMIT", "GITHUB_SHA", "CI_COMMIT_SHA"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _env_tags() -> dict[str, str]:
    return {
        k[len(META_ENV_PREFIX):].lower(): v
        for k, v in os.environ.items()
        if k.startswith(META_ENV_PREFIX) and v
    }


def build_metadata(repeats: int = 1) -> dict[str, Any]:
    from ..core.judge import JudgeConfigError, resolve_judge_config
    from ..core.stages import STAGE_ENV

    try:
        judge_cfg = resolve_judge_config()
        judge, judge_temperature = judge_cfg.describe(), judge_cfg.temperature
    except JudgeConfigError:
        judge, judge_temperature = "invalid-config", None

    datasets = sorted(_DATASETS.values(), key=lambda d: d["path"])
    combined = hashlib.sha256(
        "".join(f"{d['path']}:{d['sha256']}" for d in datasets).encode("utf-8")
    ).hexdigest() if datasets else None
    stages = sorted({d["ci_stage"] for d in datasets if d.get("ci_stage")})

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": judge,
        "judge_temperature": judge_temperature,
        "dataset_sha256": combined,
        "datasets": datasets,
        "ci_stage": os.environ.get(STAGE_ENV) or (stages[0] if len(stages) == 1 else ",".join(stages) or None),
        "repeats": repeats,
        "elyaeval_version": _pkg_version("elyaeval"),
        "deepeval_version": _pkg_version("deepeval"),
        "python_version": platform.python_version(),
        "git_sha": _git_sha(),
        "sut": {**_env_tags(), **_USER_META},
    }


def sidecar_path(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix(".meta.json")


def write_sidecar(csv_path: str | Path, meta: dict[str, Any]) -> Path:
    path = sidecar_path(csv_path)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_sidecar(csv_path: str | Path) -> Optional[dict[str, Any]]:
    path = sidecar_path(csv_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compatibility(baseline: Optional[dict], candidate: Optional[dict]) -> dict[str, list[str]]:
    """{'errors': [...], 'warnings': [...], 'sut_changes': [...]}. Missing
    sidecars (e.g. CSVs not produced by the elyaeval pytest plugin) produce a warning only."""
    result: dict[str, list[str]] = {"errors": [], "warnings": [], "sut_changes": []}
    if baseline is None or candidate is None:
        side = "baseline" if baseline is None else "candidate"
        result["warnings"].append(f"no run metadata for the {side} (no .meta.json next to the CSV)   compatibility not checked")
        return result
    for key in STRICT_KEYS:
        if baseline.get(key) != candidate.get(key):
            result["errors"].append(f"{key} differs: baseline={baseline.get(key)!r} candidate={candidate.get(key)!r}")
    for key in WARN_KEYS:
        if baseline.get(key) != candidate.get(key):
            result["warnings"].append(f"{key} differs: baseline={baseline.get(key)!r} candidate={candidate.get(key)!r}")
    b_sut, c_sut = baseline.get("sut") or {}, candidate.get("sut") or {}
    for key in sorted(set(b_sut) | set(c_sut)):
        if b_sut.get(key) != c_sut.get(key):
            result["sut_changes"].append(f"{key}: {b_sut.get(key)!r} -> {c_sut.get(key)!r}")
    return result
