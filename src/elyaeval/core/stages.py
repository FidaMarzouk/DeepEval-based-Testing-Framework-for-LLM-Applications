"""
CI stages, ordered cheapest/most-frequent first. Stages are CUMULATIVE: a
golden tagged `per_commit` also runs at merge, nightly and release, so a
nightly run is a superset of a per-commit run rather than a disjoint slice.

    per_commit  ⊂  merge  ⊂  nightly  ⊂  release

The stage a suite runs at is normally baked into the generated test file
by `elyaeval init --ci-stage`, but can be overridden per run without editing
the file via the ELYAEVAL_CI_STAGE environment variable (resolve_stage), so
the same test file serves a PR pipeline and a nightly cron.
"""

from __future__ import annotations

import os

STAGES: tuple[str, ...] = ("per_commit", "merge", "nightly", "release")
STAGE_ENV = "ELYAEVAL_CI_STAGE"


def validate_stage(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"Unknown ci_stage '{stage}'. Valid: {', '.join(STAGES)}")
    return stage


def stages_up_to(stage: str) -> tuple[str, ...]:
    """Every stage that runs when `stage` runs, e.g. 'merge' ->
    ('per_commit', 'merge')."""
    return STAGES[: STAGES.index(validate_stage(stage)) + 1]


def resolve_stage(default: str) -> str:
    """ELYAEVAL_CI_STAGE if set, else `default` (validated either way)."""
    return validate_stage(os.environ.get(STAGE_ENV) or default)
