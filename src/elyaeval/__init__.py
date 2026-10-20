"""
elyaeval: DeepEval-based testing framework for LLM applications.

Public names are resolved lazily (PEP 562) so that importing a submodule —
in particular `elyaeval.reporting.plugin`, which pytest loads at startup
in every environment that has elyaeval installed — doesn't pull in
DeepEval's evaluation stack until a test file actually uses it.

Note: metric presets (RAG_METRICS, etc.) are deliberately NOT re-exported
here. Constructing a DeepEval metric object initializes the judge model
immediately (raises if no model/API key is configured) fine for a
generated test file that's about to run evaluation, but not for `elyaeval
init` or anything else that only needs the dataset loader. Import metric
presets from elyaeval.core.metrics directly where they're actually used.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("elyaeval")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

_EXPORTS = {
    "load_standard_dataset": "elyaeval.core.dataset",
    "make_golden_context_fixture": "elyaeval.core.fixtures",
    "evaluate_golden": "elyaeval.reporting.report",
    "new_report_path": "elyaeval.reporting.report",
    "render_html_report": "elyaeval.reporting.html_report",
    "metric_averages": "elyaeval.reporting.html_report",
    "read_csv_rows": "elyaeval.reporting.html_report",
    "total_cost": "elyaeval.reporting.html_report",
    "total_tokens": "elyaeval.reporting.html_report",
    "compare_runs": "elyaeval.reporting.html_report",
    "render_comparison_html": "elyaeval.reporting.html_report",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'elyaeval' has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    globals()[name] = value  # cache so __getattr__ only runs once per name
    return value


def __dir__():
    return sorted(__all__)
