"""
Packaging smoke tests: run against the INSTALLED package (src/ layout means
`import elyaeval` can only resolve to the installed copy), so they catch the
class of bug where the source tree works but the built wheel doesn't —
broken entry points, data files missing from package-data, templates that
no longer render. No LLM / judge calls anywhere in here.
"""

import py_compile
import subprocess
import sys
from importlib import resources
from importlib.metadata import entry_points

import pytest

from elyaeval import cli

TEMPLATES = [
    "test_template.py.tmpl",
    "test_template_seeded.py.tmpl",
    "test_template_agentic.py.tmpl",
    "test_template_component.py.tmpl",
    "test_template_agentic_trajectory.py.tmpl",
    "pipelinerun_template.yaml.tmpl",
]


def _entry_point(group: str, name: str):
    matches = [ep for ep in entry_points(group=group) if ep.name == name]
    assert matches, f"no '{name}' entry point registered in group '{group}'"
    return matches[0]


def test_console_script_entry_point_resolves():
    assert callable(_entry_point("console_scripts", "elyaeval").load())


def test_pytest_plugin_entry_point_resolves():
    plugin = _entry_point("pytest11", "elyaeval").load()
    assert hasattr(plugin, "pytest_sessionfinish")


@pytest.mark.parametrize("filename", TEMPLATES)
def test_template_is_packaged(filename):
    assert (resources.files("elyaeval") / "templates" / filename).is_file()


def test_standard_dataset_is_packaged():
    assert (resources.files("elyaeval") / "standard_dataset" / "goldens.jsonl").is_file()


@pytest.mark.parametrize("path", ["recipes/tekton/pipeline.yaml", "recipes/tekton/tasks.yaml", "recipes/docker/runner.Dockerfile"])
def test_ci_recipe_is_packaged(path):
    assert (resources.files("elyaeval") / path).is_file()


def test_ci_render_pins_runner_image(tmp_path, monkeypatch):
    from elyaeval import ci

    monkeypatch.delenv("ELYAEVAL_RUNNER_IMAGE", raising=False)
    paths = ci.write_recipe(tmp_path)
    pipeline = (tmp_path / "pipeline.yaml").read_text(encoding="utf-8")
    assert [p.name for p in paths] == ["pipeline.yaml", "tasks.yaml"]
    assert f"elyaeval-runner:{ci.installed_version().replace('+', '-')}" in pipeline
    monkeypatch.setenv("ELYAEVAL_RUNNER_IMAGE", "ghcr.io/acme/elyaeval-runner:1.2.3")
    assert "ghcr.io/acme/elyaeval-runner:1.2.3" in ci.render_recipe()["pipeline.yaml"]


def test_init_ci_pins_image_and_paths(tmp_path):
    out = tmp_path / "pipelinerun.yaml"
    cli.init_ci(None, "dev", "rag-demo", str(out), test_file="test_x.py", sut_path="app",
                runner_image="ghcr.io/acme/elyaeval-runner:1.2.3", ci_stage="per_commit", budget_seconds="900")
    text = out.read_text(encoding="utf-8")
    for expected in ("${REPO_URL}", "ghcr.io/acme/elyaeval-runner:1.2.3", 'value: "app"', "test_x.py",
                     'value: "per_commit"', 'value: "900"', "pipelineRef:"):
        assert expected in text, expected
    assert "    name: elyaeval\n" in text


def test_report_theme_is_packaged():
    assert (resources.files("elyaeval") / "reporting" / "report_theme.css").is_file()


def test_cli_help_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "elyaeval.cli", "--help"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    for command in ("init", "init-ci", "report", "compare"):
        assert command in proc.stdout


def test_plugin_import_does_not_load_deepeval():
    """pytest loads the plugin at startup in every environment with elyaeval
    installed; it must stay cheap and not depend on judge configuration."""
    code = (
        "import sys, elyaeval.reporting.plugin; "
        "sys.exit(1 if 'deepeval' in sys.modules else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, "importing the pytest plugin pulled in deepeval"


@pytest.mark.parametrize(
    "task_type,corpus_mode,eval_mode",
    [
        ("rag_qa", "existing", "e2e"),
        ("rag_qa", "seeded", "e2e"),
        ("rag_qa", "existing", "component"),
        ("retrieval", "existing", "e2e"),
        ("summarization", "existing", "e2e"),
        ("qa_correctness", "existing", "e2e"),
        ("agentic_action", "existing", "e2e"),
        ("agentic_trajectory", "existing", "e2e"),
    ],
)
def test_init_renders_valid_python(tmp_path, task_type, corpus_mode, eval_mode):
    out = tmp_path / "generated_test.py"
    cli.init(task_type, "nightly", str(out), corpus_mode=corpus_mode, eval_mode=eval_mode)
    py_compile.compile(str(out), doraise=True)


def test_init_ci_renders(tmp_path):
    out = tmp_path / "pipelinerun.yaml"
    cli.init_ci("https://example.com/repo.git", "main", "results", str(out))
    text = out.read_text()
    assert "https://example.com/repo.git" in text
    assert "{" not in text.replace("${", "")  # no unrendered {placeholders}


def test_html_report_inlines_packaged_theme(tmp_path):
    from elyaeval.reporting.html_report import render_html_report

    csv_path = tmp_path / "results_rag_qa_20260101T000000Z.csv"
    csv_path.write_text(
        "golden_id,priority,input,metric_name,score,threshold,success,reason,error,"
        "evaluation_model,evaluation_cost,input_tokens,output_tokens\n"
        "P0_0,P0,q,Faithfulness,0.9,0.7,True,ok,,llama3.1,,,\n",
        encoding="utf-8",
    )
    html = render_html_report(csv_path).read_text(encoding="utf-8")
    # .report-shell only exists in report_theme.css, not in the inline fallback
    assert ".report-shell" in html
