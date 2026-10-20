"""Run metadata sidecars and comparison compatibility."""

import json

from elyaeval.core.dataset import load_standard_dataset
from elyaeval.reporting import run_metadata


def test_build_metadata_records_judge_dataset_and_tags(isolated_config, monkeypatch):
    run_metadata.clear()
    monkeypatch.setenv("ELYAEVAL_JUDGE_PROVIDER", "ollama")
    monkeypatch.setenv("ELYAEVAL_JUDGE_MODEL", "llama3.1")
    monkeypatch.setenv("ELYAEVAL_META_SUT_MODEL", "gpt-oss-120b")
    load_standard_dataset(task_type="qa_correctness", ci_stage="merge")
    run_metadata.record_run_metadata(prompt_version="v7")

    meta = run_metadata.build_metadata(repeats=3)
    assert meta["judge"] == "ollama:llama3.1"
    assert meta["repeats"] == 3
    assert meta["dataset_sha256"] and meta["datasets"][0]["n_goldens"] == 3
    assert meta["sut"] == {"sut_model": "gpt-oss-120b", "prompt_version": "v7"}
    assert meta["ci_stage"] == "merge"
    run_metadata.clear()


def test_sidecar_roundtrip(tmp_path):
    csv_path = tmp_path / "results_x_20260101T000000Z.csv"
    run_metadata.write_sidecar(csv_path, {"judge": "ollama:llama3.1"})
    assert run_metadata.sidecar_path(csv_path).name == "results_x_20260101T000000Z.meta.json"
    assert run_metadata.read_sidecar(csv_path) == {"judge": "ollama:llama3.1"}
    assert run_metadata.read_sidecar(tmp_path / "missing.csv") is None


def test_compatibility_levels():
    base = {"judge": "ollama:llama3.1", "judge_temperature": 0.0, "dataset_sha256": "a", "sut": {"prompt": "v1"}}
    same = json.loads(json.dumps(base))
    assert run_metadata.compatibility(base, same) == {"errors": [], "warnings": [], "sut_changes": []}

    other = {**base, "judge": "anthropic:claude-sonnet-5-5", "dataset_sha256": "b", "sut": {"prompt": "v2"}}
    result = run_metadata.compatibility(base, other)
    assert any("judge differs" in e for e in result["errors"])
    assert any("dataset_sha256" in w for w in result["warnings"])
    assert result["sut_changes"] == ["prompt: 'v1' -> 'v2'"]


def test_missing_sidecar_only_warns():
    result = run_metadata.compatibility(None, {"judge": "x"})
    assert not result["errors"] and result["warnings"]
