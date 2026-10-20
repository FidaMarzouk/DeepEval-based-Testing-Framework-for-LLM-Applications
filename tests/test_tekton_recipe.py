"""
Static checks of the Tekton recipe (no cluster needed): YAML parses, every
Pipeline -> Task param is declared and wired, required params are passed,
and the Python embedded in step scripts compiles. Catches the mistakes that
otherwise only surface as a failed PipelineRun.
"""

import re

import pytest

yaml = pytest.importorskip("yaml")

@pytest.fixture(scope="module")
def recipe():
    """The recipe as `elyaeval ci render` produces it from the INSTALLED package."""
    from elyaeval.ci import render_recipe

    rendered = render_recipe(runner_image="registry.example/elyaeval-runner:9.9.9")
    tasks = {d["metadata"]["name"]: d for d in yaml.safe_load_all(rendered["tasks.yaml"]) if d}
    pipeline = yaml.safe_load(rendered["pipeline.yaml"])
    return tasks, pipeline


def test_pipeline_params_are_wired(recipe):
    tasks, pipeline = recipe
    pipeline_params = {p["name"] for p in pipeline["spec"]["params"]}
    for ref in pipeline["spec"]["tasks"] + pipeline["spec"]["finally"]:
        task = tasks[ref["taskRef"]["name"]]
        declared = {p["name"]: p for p in task["spec"].get("params", [])}
        passed = {p["name"]: p["value"] for p in ref.get("params", [])}
        assert set(passed) <= set(declared), f"{ref['name']} passes undeclared {set(passed) - set(declared)}"
        missing = {n for n, p in declared.items() if "default" not in p} - set(passed)
        assert not missing, f"{ref['name']} is missing required params {missing}"
        for value in passed.values():
            for used in re.findall(r"\$\(params\.([\w-]+)\)", str(value)):
                assert used in pipeline_params, f"{ref['name']} references unknown pipeline param {used}"


def test_task_scripts_only_use_declared_params(recipe):
    tasks, _ = recipe
    for name, task in tasks.items():
        declared = {p["name"] for p in task["spec"].get("params", [])}
        for step in task["spec"]["steps"]:
            text = (step.get("script") or "") + str(step.get("env") or "") + str(step.get("envFrom") or "")
            used = set(re.findall(r"\$\(params\.([\w-]+)\)", text))
            assert used <= declared, f"{name}/{step['name']} uses undeclared params {used - declared}"


def test_embedded_python_compiles(recipe):
    tasks, _ = recipe
    found = 0
    for name, task in tasks.items():
        for step in task["spec"]["steps"]:
            heredocs = re.findall(r"<<'(PYEOF|EOF)'[^\n]*\n(.*?)\n\s*\1\s*(?:\n|$)", step.get("script") or "", re.S)
            for block in heredocs:
                source = block[1]
                indent = min(len(line) - len(line.lstrip()) for line in source.splitlines() if line.strip())
                dedented = "\n".join(line[indent:] for line in source.splitlines())
                compile(dedented, f"{name}/{step['name']}", "exec")
                found += 1
    assert found >= 2  # check-results + check-regression compare


def test_recipe_is_generic_and_pinned(recipe):
    tasks, pipeline = recipe
    assert pipeline["metadata"]["name"] == "elyaeval"
    assert all(name.startswith("elyaeval-") for name in tasks), "task names must not collide with other teams'"
    params = {p["name"]: p for p in pipeline["spec"]["params"]}
    assert params["runner-image"]["default"] == "registry.example/elyaeval-runner:9.9.9"
    assert params["sut-path"]["default"] == "."
    assert "__ELYAEVAL_RUNNER_IMAGE__" not in str(pipeline)


def test_python_steps_run_in_the_runner_image(recipe):
    tasks, _ = recipe
    for name, task in tasks.items():
        for step in task["spec"]["steps"]:
            script = step.get("script") or ""
            runs_python = re.search(r"^\s*(python3?|pip|pytest|elyaeval)\s", script, re.M)
            if runs_python:
                assert step["image"] == "$(params.runner-image)", f"{name}/{step['name']} must use the runner image"


def test_failed_gates_block_baseline_promotion(recipe):
    tasks, pipeline = recipe
    regular = {t["name"] for t in pipeline["spec"]["tasks"]}
    assert {"check-results", "check-budget"} <= regular, "gates must be regular tasks to count in $(tasks.status)"
    regression = next(t for t in pipeline["spec"]["finally"] if t["name"] == "check-regression")
    passed = {p["name"]: p["value"] for p in regression["params"]}
    assert passed["pipeline-status"] == "$(tasks.status)"
    promote = next(s for s in tasks["elyaeval-check-regression"]["spec"]["steps"] if s["name"] == "promote-baseline")
    assert "Succeeded|Completed" in promote["script"] and "pipeline-status" in promote["script"]


def test_fetch_source_supports_commits_and_cleans_workspace(recipe):
    tasks, _ = recipe
    script = tasks["elyaeval-fetch-source"]["spec"]["steps"][0]["script"]
    assert "git fetch" in script and "FETCH_HEAD" in script and "--branch" not in script
    assert "rm -rf" in script
