# elyaeval

A DeepEval-based testing framework and **CI gate** for LLM applications: unit (component),
integration (end-to-end) and regression testing for RAG apps, plain LLM tasks and tool-calling agents,
with a provider-agnostic judge, statistical regression gating, cost budgets and a Tekton CI recipe
that ships inside the package. Built to be dropped into any Elyadata LLM mission.

It gives you:
1. **`elyaeval init`**   generates a pytest suite for your app (RAG, retrieval, summarization, QA,
   agent actions, agent trajectories), scored with DeepEval metric presets.
2. **A provider-agnostic judge**   Ollama, OpenAI, Anthropic, Azure, Gemini, Bedrock, any
   OpenAI-compatible server or your own model, chosen by config.
3. **Reports and gates**   CSV + HTML per run, paired statistical regression comparison against a
   baseline, judge calibration, result caching, cost/time/token budgets.
4. **CI**   `elyaeval ci apply` installs the Tekton Pipeline on a cluster; `elyaeval init-ci`
   generates a mission's `PipelineRun`.

**Case studies** (separate repos that use elyaeval like any mission would):
[RAG-Demo](https://github.com/FidaMarzouk/RAG-Demo) (LlamaIndex RAG app) and
[Agent-Demo](https://github.com/FidaMarzouk/Agent-Demo) (LangGraph tool-calling agent).

## Repository layout

```
.
├── pyproject.toml               # package metadata, `elyaeval` CLI + pytest plugin entry points
├── src/elyaeval/                # the package (src layout: `import elyaeval` only resolves to the installed copy)
│   ├── cli.py                   # the `elyaeval` command
│   ├── ci.py                    # `elyaeval ci render|apply|dockerfile`
│   ├── core/
│   │   ├── judge.py             # provider-agnostic judge selection, timeouts, rate limits
│   │   ├── config.py            # reads [tool.elyaeval] / elyaeval.toml
│   │   ├── dataset.py           # loads + validates goldens files
│   │   ├── stages.py            # cumulative CI stages
│   │   ├── metrics.py           # metric presets (factories) + thresholds
│   │   ├── agentic.py           # agent helpers: expected tools, tool-call guards
│   │   ├── run_contracts.py     # what run_app() returns (RAGRunResult, AgenticRunResult, SUTUsage)
│   │   ├── usage.py / budget.py # usage metering, cost budgets
│   │   ├── cache.py             # judge-result cache
│   │   └── fixtures.py          # optional fixture for --corpus-mode seeded
│   ├── reporting/               # CSV/HTML reports, regression comparison, statistics, run metadata,
│   │                            # calibration, component tracing, the pytest plugin
│   ├── templates/               # files `elyaeval init` / `init-ci` generate into a mission repo
│   ├── standard_dataset/        # bundled generic goldens
│   └── recipes/                 # Tekton Pipeline + Tasks + CI guide, runner-image Dockerfile
├── tests/                       # elyaeval's own test suite (see below)
├── docker-compose.yml           # Azurite (Azure Blob emulator) for running the CI recipe locally
└── CHANGELOG.md
```

`tests/` tests **elyaeval itself**, offline: no LLM or judge is ever called (a fake judge and a
deterministic metric drive DeepEval's real evaluation code). One file per module   e.g.
`test_judge.py` (judge config), `test_compare.py` (regression gate), `test_budget.py`,
`test_tekton_recipe.py` (the CI YAML wires up and its embedded Python compiles), and
`test_suite_runs.py`, which runs small generated-style suites end to end in a subprocess. They are not
the `test_elyaeval_*.py` files a *mission* gets from `elyaeval init`. Run them with
`pip install -e ".[dev]" && pytest`.

## 0. Install

```bash
# in a mission repo: pinned to a release tag of this repo
pip install "elyaeval[ollama] @ git+https://github.com/<owner>/<repo>.git@v0.1.2"

# framework development, from a checkout of this repo
pip install -e ".[dev,ollama]"
```

Judge-provider SDKs are extras: `ollama`, `anthropic`, `gemini`, `bedrock`, `litellm` (OpenAI-based
providers need none).

## 1. Configure a judge model (once)

elyaeval is **provider-agnostic**: every metric preset gets its judge from the same config, so a
suite switches between a local model and a hosted API without code changes. Configure it in the
SUT repo's `pyproject.toml` (or a dedicated `elyaeval.toml` with the same tables, minus the
`tool.elyaeval.` prefix):

```toml
[tool.elyaeval.judge]
provider = "ollama"            # see the table below
model = "llama3.1"
# base_url = "http://localhost:11434"
# temperature = 0.0
# timeout_seconds = 900        # per metric evaluation; 0 disables DeepEval's timeouts
# max_output_tokens = 2048     # cap each judge reply (local default 2048; 0 = no cap)
# options = { generation_kwargs = { num_ctx = 8192 } }   # provider-specific kwargs
# max_calls_per_minute = 30    # pace rate-limited providers (see Cost budgeting)
```

Any key can be overridden per run with environment variables   `ELYAEVAL_JUDGE_PROVIDER`,
`ELYAEVAL_JUDGE_MODEL`, `ELYAEVAL_JUDGE_BASE_URL`, `ELYAEVAL_JUDGE_TEMPERATURE`,
`ELYAEVAL_JUDGE_TIMEOUT_SECONDS`, `ELYAEVAL_JUDGE_MAX_OUTPUT_TOKENS`, `ELYAEVAL_JUDGE_CLASS`   which
is how CI selects the judge.

| `provider` | Backend | Extra to install |
|---|---|---|
| `ollama` | Ollama native API, **schema-constrained JSON** (recommended for local models) | `elyaeval[ollama]` |
| `openai_compatible` (alias `local`) | any OpenAI-compatible server: vLLM, LM Studio, Groq, Ollama `/v1` | – |
| `openai` / `azure_openai` | OpenAI / Azure OpenAI (`options = { deployment_name, api_version }`) | – |
| `anthropic` | Anthropic Claude models | `elyaeval[anthropic]` |
| `gemini` | Google Gemini / Vertex AI (`options = { project, location, use_vertexai }`) | `elyaeval[gemini]` |
| `bedrock` | Amazon Bedrock (`options = { region }`) | `elyaeval[bedrock]` |
| `litellm` | anything LiteLLM routes to | `elyaeval[litellm]` |
| `deepseek` / `openrouter` / `grok` | those hosted APIs | – |
| `custom` | your own `DeepEvalBaseLLM` subclass: `class = "my_pkg.judges:Gateway"` | – |
| *(unset)* `deepeval` | default: whatever DeepEval itself is configured with (`deepeval set-*`) | – |

**API keys never go in config files** (it's rejected): set `ELYAEVAL_JUDGE_API_KEY`, the
provider's own variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...), or name a variable with
`api_key_env = "MY_TEAM_KEY"`.

Check what a config resolves to (the key is never printed) and that the judge answers:

```bash
elyaeval judge          # resolved provider/model/base URL/timeout
elyaeval judge --ping   # + one tiny schema-constrained call
```.

### Thresholds

Every metric defaults to a 0.7 threshold. Override per metric (keys are the metric keys in
`elyaeval.core.metrics.METRIC_BUILDERS`) or for all of them:

```toml
[tool.elyaeval.thresholds]
default = 0.7
faithfulness = 0.8
contextual_relevancy = 0.6
```

## 2. Generate a test file

```bash
elyaeval init --task-type rag_qa
```

| Flag | Options | Default | Notes |
|---|---|---|---|
| `--task-type` | `rag_qa`, `retrieval`, `summarization`, `qa_correctness`, `agentic_action` | required | picks the metric preset + template |
| `--ci-stage` | `per_commit`, `merge`, `nightly`, `release` | `nightly` | **cumulative**: `nightly` runs per_commit + merge + nightly goldens. Override per run with `ELYAEVAL_CI_STAGE` (e.g. `per_commit` in PR CI) without editing the file |
| `--eval-mode` | `e2e`, `component` | `e2e` | `e2e` scores the final output; `component` scores each traced span in-process (needs `@observe(metrics=...)` on the SUT side) |
| `--corpus-mode` | `existing`, `seeded` | `existing` | `seeded` temporarily inserts each golden's `context` into your storage before the run (fill in `_ingest_passage`/`_teardown_passages`) |

If a flag combination matches 0 goldens, `init` warns immediately instead of generating a silently-empty suite.

## 3. Fill in the TODOs

Open the generated file and search for `TODO`:
1. Import your app's/agent's entrypoint.
2. Implement `run_app()` (or `call_sut()` for `--eval-mode component`) to call it and return the
   typed result (`RAGRunResult` / `AgenticRunResult`, see `elyaeval.core.run_contracts`).
3. `--corpus-mode seeded` only: implement `_ingest_passage`/`_teardown_passages`.
   `agentic_action` only: adjust `_expected_tools_from_golden()` if needed.

## 4. Run it

```bash
deepeval test run test_elyaeval_rag_qa.py
```

Each golden runs through its own `evaluate()` call, so pass/fail and scores stay traceable per golden.
Metrics come from factories (`rag_metrics()`, `retrieval_metrics()`, ...) called per golden, so every
golden gets fresh metric instances scored by the configured judge. (The old `RAG_METRICS`-style
constants still work but are deprecated.)

**Parallel runs:** with `pip install "elyaeval[xdist]"`, `pytest -n 4 test_elyaeval_rag_qa.py`
works: each worker writes its own part file and elyaeval's plugin merges them into one CSV + HTML
report at the end. Useful with hosted judges; with a single local GPU, parallelism mostly queues.

**Component mode:** assert with `failed_results(result)` (the generated template does), not
`TestResult.success`   DeepEval 4.2.0 reports `success=False` for every span result regardless of
its scores (`elyaeval.reporting.tracing_report.result_passed` documents why).

## Reading the results

Written per golden as the suite runs, no extra config needed:
- **`junit.xml`**   plain pass/fail, what CI gates on.
- **`report/results_<task_type>_<timestamp>[_component].csv`**   one row per `(golden, [span,]
  metric, repeat)`: score, threshold, success, reason, error, judge, cost, input/output tokens,
  `cached`.
- **`report/results_..._<timestamp>[_component].html`**   same data, self-contained HTML with a
  per-metric summary (average, 95% CI, noise σ, pass rate, cost) + per-golden detail table.
  Auto-generated at session end by elyaeval's pytest plugin (registered automatically once
  `elyaeval` is installed).
- **`report/budget_<timestamp>.json`**   usage ledger and budget status for the run; see
  [Cost budgeting](#cost-budgeting).
- **`report/results_..._<timestamp>[_component].meta.json`**   what produced the run (judge,
  datasets, versions, git SHA, SUT tags); see [Run metadata](#run-metadata).

The plugin also prints the per-metric average table to the terminal right after pytest's summary.

To regenerate a report from an existing CSV (e.g. one pulled from blob storage):
```bash
elyaeval report --csv report/results_rag_qa_20260825T130000Z.csv
```

## Goldens file schema

One JSON object per line. Every row needs:

| Field | Notes |
|---|---|
| `id` | *optional but recommended*   stable, unique id. Without it elyaeval uses a content hash of `task_type` + `input`. Ids key every CSV row and the regression comparison, so they must not depend on row order. Duplicates are rejected. |
| `input` | the question/prompt |
| `additional_metadata.task_type` | one of the 5 task types above |
| `additional_metadata.ci_stage` | `per_commit` / `merge` / `nightly` / `release` |

Per task type, on top of that baseline:

| task_type | Also needs | Why |
|---|---|---|
| `rag_qa` | `expected_output` | drives ContextualPrecision/Recall + Faithfulness |
| `retrieval` | `expected_output` | drives ContextualPrecision/Recall |
| `summarization` | *(none)* | reference-free metric |
| `qa_correctness` | `expected_output` | scored against `actual_output` |
| `agentic_action` | *(none, `expected_tools` optional)* | see the agentic template |

`context` is only needed for `--corpus-mode seeded`. `retrieval_context` is never pre-filled   only
your own retriever knows what it fetched at test time.

`elyaeval/core/dataset.py`'s `GOLDEN_SCHEMA` is the single source of truth; `load_standard_dataset()`
validates the whole file upfront and raises one error listing every problem found, not just the first.

## Regression testing

LLM-judge scores are noisy and suites are small, so "the average dropped by 0.03" is often just
noise. `elyaeval compare` is built for that:

```bash
elyaeval compare --baseline report/results_rag_qa_20260820T090000Z.csv \
                 --candidate report/results_rag_qa_20260826T090000Z.csv \
                 --html report/regression.html
```

For each metric (or span + metric) it:

1. **pairs goldens by id**: only goldens scored in both runs are compared, so a candidate that ran
   a different or smaller golden set can't look better or worse just because of what it ran;
2. averages repeats per golden, then computes the paired deltas, a **95% bootstrap CI** of the mean
   delta and a **sign-flip permutation p-value**;
3. flags a **regression** when the mean drop exceeds the tolerance **and** is statistically
   significant (`p < alpha`). With fewer than `min_paired` pairs, or `--gate delta`, the tolerance
   alone decides (the output says which method was used);
4. separately flags a rise in the **error rate** (errored/unscored rows   invisible in averages,
   and a judge that crashes on the hard cases would otherwise make the average go *up*), and, if
   configured, a drop in the **pass rate**.

Gate settings live in config; CLI flags override them:

```toml
[tool.elyaeval.regression]
tolerance = 0.02                        # mean paired score drop (0..1)
tolerances = { Faithfulness = 0.05 }    # per metric_name or "span/metric"
alpha = 0.05
min_paired = 5
gate = "stat"                           # or "delta"
max_error_rate_increase = 0.05
max_pass_rate_drop = 0.10               # unset = not gated
```

Exit code: `0` nothing regressed, `1` something regressed, `2` the runs aren't comparable. CI gates
on it directly.

### Run metadata

Next to every results CSV the pytest plugin writes `<name>.meta.json`: judge, judge temperature,
datasets (+ SHA-256), elyaeval/DeepEval versions, git SHA, CI stage, repeats and SUT tags.
`compare` **refuses** (exit `2`) to compare runs scored by different judges   their scores aren't
on the same scale   unless `--allow-metadata-mismatch`; it **warns** about dataset/version/stage
differences and **reports** SUT changes, which are usually the point of the comparison. Tag runs
with `ELYAEVAL_META_<KEY>` env vars (`ELYAEVAL_META_PROMPT_VERSION=v7`) or, in a test file,
`record_run_metadata(prompt_version="v7")` from `elyaeval.reporting.run_metadata`.

### Repeats and noise

```bash
pytest test_elyaeval_rag_qa.py --elyaeval-repeats 3      # or ELYAEVAL_REPEATS=3
```

runs every golden 3 times (rows get a `repeat` column). Reports then show per-metric **95% CIs**
and the **noise σ** (spread of repeated scores of the same golden), and `compare` averages
repeats before pairing. Use repeats at nightly/release; with a local judge they multiply run time.

### Calibrating the judge

```bash
elyaeval calibrate --results report/results_rag_qa_<ts>.csv            # needs repeats >= 2
elyaeval calibrate --results report/results_rag_qa_<ts>.csv --labels labels.csv
```

- **Noise -> tolerances:** per metric, the tolerance that judge/SUT noise alone would rarely exceed
  for a suite of this size (`1.96 · σ · √(2 / (goldens · repeats))`), printed as a ready-to-paste
  `[tool.elyaeval.regression] tolerances` snippet. Gating tighter than that fails CI on noise.
- **Agreement with humans:** label a sample (`golden_id, metric_name, human_pass[, human_score]`)
  and get per-metric accuracy, Cohen's κ, pass rates, MAE, and the threshold that best matches the
  humans (as a `[tool.elyaeval.thresholds]` snippet). κ < 0.4 is flagged: don't gate on that
  metric with that judge.

### Result cache

Re-running a suite after an unrelated change shouldn't re-pay for   or, on a local GPU, re-wait
for   identical judgements:

```toml
[tool.elyaeval.cache]
enabled = true                                  # or ELYAEVAL_CACHE=1 per run
path = ".elyaeval_cache/results.sqlite"         # default
```

The key covers the judge, the metric and its settings, the full test case, the repeat index and
the DeepEval version, so any change re-judges. Only error-free results are stored (timeouts are
retried). Cached rows are written with `cached=True` and zero cost. `elyaeval cache [stats|clear]`.
Applies to e2e suites (`evaluate_golden`); component/span metrics run inside DeepEval's tracing.

## Agent evaluation

Two task types, at two levels:

| | `agentic_action` | `agentic_trajectory` |
|---|---|---|
| Runs the agent | over HTTP or in-process (`run_app()` -> `AgenticRunResult`) | in-process, under DeepEval tracing |
| Scores | Tool Correctness vs `expected_tools`, Argument Correctness, answer Correctness | Task Completion, Step Efficiency, Plan Quality, Plan Adherence over the whole trace |
| Plus (both, no judge) | Tool Call Budget (≤ `max_tool_calls`), Repeated Tool Calls (loops) | same |

Goldens state `expected_tools` as `[{"name": ..., "input_parameters": {...}}]` (top-level or in
`additional_metadata`); **`[]` means "call no tool"** (greetings, out-of-scope requests); leaving it
out skips Tool Correctness. `max_tool_calls` sets the per-golden budget. An agent API returns its
calls as JSON; `elyaeval.core.agentic.tool_calls_from_json` reads `{name, input_parameters | args |
arguments, output}`. For trajectory evaluation the agent only needs DeepEval spans:
`@observe(type="agent")` on the entrypoint (+ `update_current_trace(input=..., output=...)`),
`@observe(type="tool")` on tools and `@observe(type="llm")` on model calls   no judge or metrics in
the agent's code. See [Agent-Demo](https://github.com/FidaMarzouk/Agent-Demo) for a complete LangGraph example.

## Cost budgeting

Every run can be capped   in USD, wall-clock time, tokens and/or judge calls   and every run gets a
usage ledger, whatever the provider.

```toml
[tool.elyaeval.budget]
max_seconds = 1800        # wall clock: the dependable unit with a local judge
max_cost_usd = 2.00       # judge + SUT spend (reported, or priced below)
max_tokens = 2_000_000    # judge + SUT tokens (reported, else estimated)
max_judge_calls = 500
on_exceed = "stop"        # stop: skip remaining goldens | warn: only report
predictive = true         # also stop when the NEXT golden would overrun
prioritize = true         # run P0 goldens first, so a budget cuts P2s
```

Override per run with `--elyaeval-budget-seconds/-usd/-tokens/-calls/-on-exceed` or
`ELYAEVAL_BUDGET_SECONDS` / `_USD` / `_TOKENS` / `_CALLS` / `_ON_EXCEED`.

**How it's enforced.** Goldens run P0 first. Before each golden the guard checks the spend so far
(and, with `predictive`, whether the next golden, at the average cost so far, would overrun):
once the budget is exhausted the remaining goldens are **skipped with the reason**, and the run's
report says so. Under pytest-xdist each worker gets an equal share.

**How usage is measured.** Provider-reported cost/tokens when they're real. Local judges (e.g.
Ollama) report cost 0 and 0 tokens, which elyaeval treats as *unknown*, not free: every judge
elyaeval builds is instrumented to count calls and time and to estimate tokens from prompt/response
size. To get dollars for such judges   or for a SUT that only reports tokens   add pricing:

```toml
[tool.elyaeval.pricing]
judge_per_hour = 0.40          # e.g. what the judge's GPU costs per hour
judge_input_per_mtok = 0.15    # or per-token pricing
judge_output_per_mtok = 0.60
sut_input_per_mtok = 0.59
sut_output_per_mtok = 0.79
```

The SUT's own spend comes from `run_app()`: return `RAGRunResult(..., usage=SUTUsage(input_tokens=...,
output_tokens=..., cost=...))`; the generated templates measure its latency automatically.

**Rate limits** (free tiers such as Groq): pace the judge in `[tool.elyaeval.judge]`:

```toml
max_calls_per_minute = 30
max_tokens_per_minute = 6000   # estimated from prompt size
```

**Before a run**   project it from the last run's ledger (or a rough heuristic without history):

```bash
elyaeval estimate --task-type rag_qa --ci-stage nightly --repeats 3 --budget-seconds 1800
```

exits `1` if the projection is over budget. pytest prints the same projection as a preflight line
when a budget is set.

**After a run**   `report/budget_<run>.json` holds limits, spend (with where each cost came from:
reported / priced / unknown), skipped goldens and a per-golden ledger (judge time, cost, tokens,
SUT tokens/cost/latency). The HTML report shows a Budget/Usage card and a **per-golden usage**
table (the app's latency/tokens/cost next to the judge's time/calls/tokens). Every CSV row carries
its own metric's `judge_calls`, `judge_seconds` and tokens; when the provider reports no tokens
(e.g. Ollama), elyaeval fills in its own estimate (`tokens_estimated`, shown as `≈` in the
report). Metrics that errored (timeout, unparsable judge reply) are shown as **error**, separately
from low scores; they still count as not passed.

```bash
elyaeval budget                      # summarize the latest ledger
elyaeval budget --fail-on-exhausted  # exit 1 if the run hit its budget (CI's check-budget step)
```

## CI/CD (Tekton)

elyaeval is meant to run as a **CI gate**. The Tekton Pipeline/Tasks and the runner-image
Dockerfile ship inside this package, version-matched:

```bash
# once per cluster
docker build -f src/elyaeval/recipes/docker/runner.Dockerfile -t elyaeval-runner:0.1.2 .
kind load docker-image elyaeval-runner:0.1.2 --name elyaeval     # or push to a registry
elyaeval ci apply -n elyaeval

# once per mission repo
elyaeval init-ci --blob-container <mission> --ci-stage per_commit --budget-seconds 1800

# per run (app already running)
envsubst < pipelinerun.yaml | kubectl create -n elyaeval -f -
```

The run gates on metric thresholds, the evaluation budget and regressions against a baseline
stored in Azure Blob Storage, and uploads every report there. The full guide   cluster setup,
runner image, Azurite, secrets, parameters, baselines, troubleshooting   is
[`src/elyaeval/recipes/tekton/README.md`](src/elyaeval/recipes/tekton/README.md).
