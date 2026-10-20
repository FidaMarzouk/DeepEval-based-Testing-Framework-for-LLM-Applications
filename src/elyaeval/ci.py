"""
The CI side of elyaeval: the Tekton recipe and the runner-image Dockerfile
ship INSIDE the package, so the recipe a cluster runs always matches the
elyaeval version that generated a mission's test files and PipelineRun.

    elyaeval ci render --output-dir ci/     # write pipeline.yaml + tasks.yaml
    elyaeval ci apply -n elyaeval           # kubectl apply them to a cluster
    elyaeval ci dockerfile                  # print the runner-image Dockerfile

The runner image every Python step uses is, in order of precedence:
--runner-image, ELYAEVAL_RUNNER_IMAGE, [tool.elyaeval.ci] runner_image, or
`elyaeval-runner:<installed version>` (a local tag: `kind load docker-image`
it into a kind cluster, or push it to a registry such as ghcr.io and set the
full name).
"""

from __future__ import annotations

import os
import subprocess
from importlib import resources
from pathlib import Path
from typing import Optional

from .core.config import config_section

RUNNER_IMAGE_ENV = "ELYAEVAL_RUNNER_IMAGE"
RUNNER_IMAGE_REPO = "elyaeval-runner"
_PLACEHOLDER = "__ELYAEVAL_RUNNER_IMAGE__"
_RECIPE_FILES = ("pipeline.yaml", "tasks.yaml")


def installed_version() -> str:
    from . import __version__

    return __version__


def default_runner_image() -> str:
    explicit = os.environ.get(RUNNER_IMAGE_ENV) or config_section("ci").get("runner_image")
    if explicit:
        return str(explicit)
    # docker tags can't contain '+' (local version labels)
    return f"{RUNNER_IMAGE_REPO}:{installed_version().replace('+', '-')}"


def _recipe_text(name: str) -> str:
    return (resources.files("elyaeval") / "recipes" / "tekton" / name).read_text(encoding="utf-8")


def render_recipe(runner_image: Optional[str] = None) -> dict[str, str]:
    """{filename: yaml} with the runner image pinned as the Pipeline default."""
    image = runner_image or default_runner_image()
    return {name: _recipe_text(name).replace(_PLACEHOLDER, image) for name in _RECIPE_FILES}


def write_recipe(output_dir: str | Path, runner_image: Optional[str] = None) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in render_recipe(runner_image).items():
        path = out / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def apply_recipe(
    namespace: str,
    runner_image: Optional[str] = None,
    kubectl: str = "kubectl",
    dry_run: Optional[str] = None,
) -> int:
    """kubectl apply the rendered recipe. dry_run: None, "client" or "server"."""
    rendered = render_recipe(runner_image)
    manifest = "\n---\n".join(rendered[name] for name in _RECIPE_FILES)
    cmd = [kubectl, "apply", "-n", namespace, "-f", "-"]
    if dry_run:
        cmd.append(f"--dry-run={dry_run}")
    # explicit UTF-8: the recipe contains non-ASCII (✓/✗ in step logs) and the
    # Windows console code page (cp1252) can't encode it
    return subprocess.run(cmd, input=manifest, text=True, encoding="utf-8", check=False).returncode


def dockerfile_text() -> str:
    return (resources.files("elyaeval") / "recipes" / "docker" / "runner.Dockerfile").read_text(encoding="utf-8")
