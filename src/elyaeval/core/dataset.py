"""
Loads the standard dataset (goldens.jsonl, versioned and shipped inside this
package and returns deepeval Golden objects, filtered by task_type and/or ci_stage.

Pass your own `dataset_path` to evaluate against a goldens file written
for YOUR app's real corpus instead of the bundled generic one   this is
the recommended input for the default `elyaeval init` template.

Same JSONL shape as the standard dataset; `context` is optional and not
required by any task_type   no metric preset in metrics.py (RAG_METRICS,
RETRIEVAL_METRICS, GENERATION_METRICS) reads it. Leave it out.

GOLDEN_SCHEMA below is the single source of truth for what a row must
contain, per task_type.

Golden ids: every loaded golden gets a stable id in
additional_metadata["id"] — the row's own top-level "id" if it has one
(recommended for hand-curated files), otherwise a content hash of
task_type + input. Ids never depend on row order, so inserting or
reordering goldens doesn't re-key every CSV row and break cross-run
regression comparison. Duplicate ids are a validation error.
"""

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Optional

from deepeval.dataset import Golden
from deepeval.test_case import ToolCall

from .stages import STAGES, stages_up_to

_PACKAGE_DATASET = "standard_dataset/goldens.jsonl"

# Every row needs these regardless of task_type   enforced on EVERY row in
# the file, even ones that will be filtered out by task_type/ci_stage,
# because a row that's missing these can never be matched by ANY filter

BASELINE_REQUIRED_FIELDS = ["input"]

# Optional additional_metadata.expected_behavior: "answer" (default) or
# "abstain" for questions the app should decline (e.g. not covered by its
# knowledge base); see elyaeval.core.metrics.metrics_for_golden.
EXPECTED_BEHAVIORS = ("answer", "abstain")
BASELINE_REQUIRED_METADATA_FIELDS = ["task_type", "ci_stage"]

# Per-task-type requirements, layered on top of the baseline above.
# Keys here are the only valid `task_type` values.

GOLDEN_SCHEMA = {
    "rag_qa": {
        "required_fields": ["expected_output"],
        "why": (
            "expected_output is what's actually scored   read by "
            "ContextualPrecision/Recall (against live retrieval_context) "
            "and by Faithfulness's reference check. "
        ),
    },
    "retrieval": {
        "required_fields": ["expected_output"],
        "why": (
            "Same as rag_qa: expected_output is read by "
            "ContextualPrecision/Recall (against live retrieval_context); "
            "ContextualRelevancy needs neither expected_output nor context, "
            "just input + retrieval_context. "
        ),
    },
    "summarization": {
        "required_fields": [],
        "why": "SummarizationMetric is reference-free, only input + your app's actual_output are scored.",
    },
    "qa_correctness": {
        "required_fields": ["expected_output"],
        "why": "The Correctness G-Eval rubric explicitly compares actual_output against expected_output.",
    },
    "agentic_action": {
        "required_fields": [],
        "required_metadata_fields": [],
        "why": (
            "ToolCorrectness/ArgumentCorrectness score run_app()'s actual "
            "tools_called. expected_tools (top-level or in additional_metadata) "
            "is opt-in ground truth; an explicit [] means 'call no tool'   "
            "see elyaeval.core.agentic."
        ),
    },
    "agentic_trajectory": {
        "required_fields": [],
        "required_metadata_fields": [],
        "why": (
            "TaskCompletion/StepEfficiency/PlanQuality/PlanAdherence judge the "
            "agent's whole trace; expected_output / expected_tools are optional "
            "context for the reader and the action-level metrics."
        ),
    },
}


class GoldenValidationError(ValueError):
    """Raised once per load_standard_dataset() call, listing every problem
    found across the whole file   not just the first one   so a goldens
    file with several bad rows can be fixed in one pass instead of
    one failed run at a time."""

    def __init__(self, path: Path, problems: list[str]):
        self.path = path
        self.problems = problems
        header = f"{len(problems)} problem(s) in goldens file {path}:"
        super().__init__("\n".join([header] + [f"  - {p}" for p in problems]))


def _default_dataset_path() -> Path:
    return Path(str(resources.files("elyaeval") / _PACKAGE_DATASET))


def _validate_row(line_no: int, row: dict) -> list[str]:
    """Returns a list of problem strings for this one row (empty if none).
    Never raises   the caller collects problems across every row before
    deciding whether to raise, so validation is whole-file-upfront."""
    problems = []

    for field in BASELINE_REQUIRED_FIELDS:
        if not row.get(field):
            problems.append(f"line {line_no}: missing required field '{field}'")

    meta = row.get("additional_metadata")
    if not isinstance(meta, dict):
        problems.append(
            f"line {line_no}: 'additional_metadata' is missing or not an object "
            f"(needs at least: {BASELINE_REQUIRED_METADATA_FIELDS})"
        )
        return problems  # nothing below is checkable without metadata

    for field in BASELINE_REQUIRED_METADATA_FIELDS:
        if not meta.get(field):
            problems.append(f"line {line_no}: missing required additional_metadata.'{field}'")

    row_stage = meta.get("ci_stage")
    if row_stage and row_stage not in STAGES:
        problems.append(
            f"line {line_no}: unknown additional_metadata.ci_stage '{row_stage}'   "
            f"expected one of {list(STAGES)}"
        )

    for where, tools in (("expected_tools", row.get("expected_tools")),
                         ("additional_metadata.expected_tools", meta.get("expected_tools"))):
        if tools is None:
            continue
        if not isinstance(tools, list) or not all(isinstance(t, dict) and t.get("name") for t in tools):
            problems.append(
                f"line {line_no}: {where} must be a list of objects with a 'name' "
                f"(and optional 'input_parameters')"
            )

    behavior = meta.get("expected_behavior")
    if behavior is not None and behavior not in EXPECTED_BEHAVIORS:
        problems.append(
            f"line {line_no}: additional_metadata.expected_behavior must be one of "
            f"{list(EXPECTED_BEHAVIORS)}, got {behavior!r}"
        )

    if "id" in row and (not isinstance(row["id"], str) or not row["id"].strip()):
        problems.append(f"line {line_no}: 'id' must be a non-empty string if present")

    row_task_type = meta.get("task_type")
    if row_task_type is not None and row_task_type not in GOLDEN_SCHEMA:
        problems.append(
            f"line {line_no}: unknown additional_metadata.task_type '{row_task_type}' "
            f"  expected one of {sorted(GOLDEN_SCHEMA)}"
        )
        return problems  # can't check task-specific fields against an unknown schema

    if row_task_type in GOLDEN_SCHEMA:
        schema = GOLDEN_SCHEMA[row_task_type]
        for field in schema.get("required_fields", []):
            if not row.get(field):
                problems.append(
                    f"line {line_no}: task_type='{row_task_type}' requires '{field}' "
                    f"({schema['why']})"
                )
        for field in schema.get("required_metadata_fields", []):
            if not meta.get(field):
                problems.append(
                    f"line {line_no}: task_type='{row_task_type}' requires "
                    f"additional_metadata.'{field}' ({schema['why']})"
                )

    return problems


def _expected_tools(row: dict) -> Optional[list[ToolCall]]:
    """Top-level expected_tools, else additional_metadata.expected_tools;
    None when the golden states none ([] is kept: 'call no tool')."""
    raw = row.get("expected_tools")
    if raw is None:
        raw = (row.get("additional_metadata") or {}).get("expected_tools")
    if raw is None:
        return None
    return [ToolCall(name=t["name"], input_parameters=t.get("input_parameters"), output=t.get("output")) for t in raw]


def compute_golden_id(row: dict) -> str:
    """The row's explicit id, else a short content hash of task_type + input."""
    if isinstance(row.get("id"), str) and row["id"].strip():
        return row["id"].strip()
    meta = row.get("additional_metadata") or {}
    key = f"{meta.get('task_type', '')}\n{row.get('input', '')}"
    digest = hashlib.sha1(key.encode("utf-8"))
    return digest.hexdigest()[:12]


def golden_id(golden: Golden) -> str:
    """Stable id of a golden loaded by load_standard_dataset()."""
    meta = golden.additional_metadata or {}
    if meta.get("id"):
        return str(meta["id"])
    return compute_golden_id({"input": golden.input, "additional_metadata": meta})


def load_standard_dataset(
    task_type: Optional[str] = None,
    ci_stage: Optional[str] = None,
    dataset_path: Optional[str] = None,
    exact_stage: bool = False,
) -> list[Golden]:
    """
    Parameters
    ----------
    task_type: filter to e.g. "rag_qa", "summarization", "qa_correctness".
               None returns all task types mixed
    ci_stage:  "per_commit", "merge", "nightly" or "release". Cumulative by
               default: "nightly" returns per_commit + merge + nightly
               goldens (see elyaeval.core.stages). None returns all.
    exact_stage: only return goldens tagged with exactly `ci_stage`.
    dataset_path: override the package-bundled dataset (e.g. to point at a
               pinned version, or a path fetched via the version tag.
               Defaults to the dataset shipped inside elyaeval.

    Validates the WHOLE file upfront against GOLDEN_SCHEMA before returning
    anything   every row is checked (regardless of task_type/ci_stage
    filters, since a row that fails baseline checks would otherwise just
    vanish from every filter silently)
    """
    path = Path(dataset_path) if dataset_path else _default_dataset_path()

    parsed_rows: list[tuple[int, dict]] = []
    problems: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {line_no}: invalid JSON ({e})")
                continue
            problems.extend(_validate_row(line_no, row))
            parsed_rows.append((line_no, row))

    first_seen: dict[str, int] = {}
    for line_no, row in parsed_rows:
        gid = compute_golden_id(row)
        if gid in first_seen:
            problems.append(
                f"line {line_no}: duplicate golden id '{gid}' (first used on line "
                f"{first_seen[gid]})   give one of them an explicit unique 'id', or "
                f"remove the duplicate"
            )
        else:
            first_seen[gid] = line_no

    if problems:
        raise GoldenValidationError(path, problems)

    if ci_stage is None:
        allowed_stages = None
    else:
        allowed_stages = (ci_stage,) if exact_stage else stages_up_to(ci_stage)

    goldens: list[Golden] = []
    for _, row in parsed_rows:
        meta = {**row.get("additional_metadata", {}), "id": compute_golden_id(row)}
        if task_type is not None and meta.get("task_type") != task_type:
            continue
        if allowed_stages is not None and meta.get("ci_stage") not in allowed_stages:
            continue

        goldens.append(
            Golden(  # type: ignore[call-arg]  # pydantic defaults invisible to mypy
                input=row["input"],
                expected_output=row.get("expected_output"),
                context=row.get("context"),
                # retrieval_context deliberately NOT passed through
                # see module docstring.
                additional_metadata=meta,
                comments=row.get("comments"),
                # provenance (origin dataset/id, ...) travels with the golden
                custom_column_key_values=row.get("custom_column_key_values"),
                expected_tools=_expected_tools(row),
            )
        )

    # recorded in the run's .meta.json so comparisons know which data ran
    from ..reporting.run_metadata import register_dataset

    register_dataset(path, task_type, ci_stage, len(goldens))
    return goldens
