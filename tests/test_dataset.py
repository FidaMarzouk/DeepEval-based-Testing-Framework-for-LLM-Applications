"""Goldens loading: validation, cumulative stages, stable ids, provenance."""

import json

import pytest

from elyaeval.core.dataset import GoldenValidationError, compute_golden_id, golden_id, load_standard_dataset
from elyaeval.core.stages import STAGES, resolve_stage, stages_up_to


def _row(input_, stage="per_commit", task="qa_correctness", **extra):
    row = {
        "input": input_,
        "expected_output": "x",
        "additional_metadata": {"task_type": task, "ci_stage": stage, "priority": "P0"},
    }
    row.update(extra)
    return row


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def test_stages_are_cumulative():
    assert stages_up_to("per_commit") == ("per_commit",)
    assert stages_up_to("nightly") == ("per_commit", "merge", "nightly")
    assert stages_up_to("release") == STAGES


def test_resolve_stage_env_override(monkeypatch):
    assert resolve_stage("nightly") == "nightly"
    monkeypatch.setenv("ELYAEVAL_CI_STAGE", "per_commit")
    assert resolve_stage("nightly") == "per_commit"
    monkeypatch.setenv("ELYAEVAL_CI_STAGE", "weekly")
    with pytest.raises(ValueError, match="weekly"):
        resolve_stage("nightly")


def test_load_filters_cumulatively_and_exactly(tmp_path):
    path = _write(tmp_path / "g.jsonl", [_row(f"q{i}", stage) for i, stage in enumerate(STAGES)])
    assert len(load_standard_dataset(ci_stage="per_commit", dataset_path=path)) == 1
    assert len(load_standard_dataset(ci_stage="nightly", dataset_path=path)) == 3
    assert len(load_standard_dataset(ci_stage="release", dataset_path=path)) == 4
    assert len(load_standard_dataset(ci_stage="nightly", dataset_path=path, exact_stage=True)) == 1
    assert len(load_standard_dataset(dataset_path=path)) == 4


def test_ids_are_explicit_or_content_hashed_and_order_independent(tmp_path):
    rows = [_row("first", id="my-id"), _row("second")]
    a = load_standard_dataset(dataset_path=_write(tmp_path / "a.jsonl", rows))
    b = load_standard_dataset(dataset_path=_write(tmp_path / "b.jsonl", list(reversed(rows))))
    ids_a = {g.input: golden_id(g) for g in a}
    ids_b = {g.input: golden_id(g) for g in b}
    assert ids_a == ids_b
    assert ids_a["first"] == "my-id"
    assert ids_a["second"] == compute_golden_id(rows[1]) and len(ids_a["second"]) == 12


def test_validation_collects_every_problem(tmp_path):
    rows = [
        {"input": "no metadata"},
        _row("bad stage", stage="weekly"),
        _row("bad task", task="poetry"),
        _row("dup"),
        _row("dup"),
        _row("bad id", id=""),
    ]
    path = tmp_path / "bad.jsonl"
    _write(path, rows)
    with path.open("a", encoding="utf-8") as f:
        f.write("{not json\n")
    with pytest.raises(GoldenValidationError) as exc:
        load_standard_dataset(dataset_path=str(path))
    text = str(exc.value)
    for fragment in ("additional_metadata", "weekly", "poetry", "duplicate golden id", "'id' must be", "invalid JSON"):
        assert fragment in text, fragment


def test_rag_qa_requires_expected_output(tmp_path):
    row = _row("q", task="rag_qa")
    del row["expected_output"]
    with pytest.raises(GoldenValidationError, match="expected_output"):
        load_standard_dataset(dataset_path=_write(tmp_path / "g.jsonl", [row]))


def test_provenance_is_kept(tmp_path):
    row = _row("q", custom_column_key_values={"origin_dataset": "ragbench"})
    (golden,) = load_standard_dataset(dataset_path=_write(tmp_path / "g.jsonl", [row]))
    assert golden.custom_column_key_values == {"origin_dataset": "ragbench"}


@pytest.mark.parametrize("task_type", ["rag_qa", "retrieval", "summarization", "qa_correctness", "agentic_action", "agentic_trajectory"])
@pytest.mark.parametrize("stage", STAGES)
def test_standard_dataset_covers_every_task_type_and_stage(task_type, stage):
    goldens = load_standard_dataset(task_type=task_type, ci_stage=stage)
    assert goldens, f"no {task_type} goldens at stage {stage}"
    assert all(g.additional_metadata.get("id") for g in goldens)


def test_expected_behavior_is_validated(tmp_path):
    good = _row("q1")
    good["additional_metadata"]["expected_behavior"] = "abstain"
    bad = _row("q2")
    bad["additional_metadata"]["expected_behavior"] = "refuse"
    with pytest.raises(GoldenValidationError, match="expected_behavior"):
        load_standard_dataset(dataset_path=_write(tmp_path / "g.jsonl", [good, bad]))
    assert load_standard_dataset(dataset_path=_write(tmp_path / "ok.jsonl", [good]))


def test_expected_tools_are_validated_and_loaded(tmp_path):
    good = _row("q1", task="agentic_action", expected_tools=[{"name": "calc", "input_parameters": {"x": 1}}])
    none_expected = _row("q2", task="agentic_action", expected_tools=[])
    bad = _row("q3", task="agentic_action", expected_tools=[{"input_parameters": {}}])
    with pytest.raises(GoldenValidationError, match="expected_tools"):
        load_standard_dataset(dataset_path=_write(tmp_path / "bad.jsonl", [good, bad]))
    g1, g2 = load_standard_dataset(dataset_path=_write(tmp_path / "ok.jsonl", [good, none_expected]))
    assert [t.name for t in g1.expected_tools] == ["calc"] and g1.expected_tools[0].input_parameters == {"x": 1}
    assert g2.expected_tools == []
