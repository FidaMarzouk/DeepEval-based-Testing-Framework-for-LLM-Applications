# elyaeval CI guide (Tekton)

elyaeval runs as a **CI gate**: one shared Tekton `Pipeline` evaluates any mission's LLM app, and each
mission only commits a small `PipelineRun` describing *what* to evaluate. The Pipeline/Tasks and the
runner-image Dockerfile ship **inside the elyaeval package**, so the recipe on a cluster always matches
the elyaeval version that generated a mission's files.

```
fetch-source ─► setup-env ─► run-suite ─┬─► check-results   metric thresholds (junit)
 (mission repo   (opt-in     (pytest in  └─► check-budget    evaluation budget
  @ branch/tag/   extras)     runner
  commit)                     image)
finally:  report-results    upload CSV/HTML/metadata/ledger to Azure Blob (always, even on failure)
          check-regression  compare against the stored baseline; promote only if every gate passed
```

The **app under test must already be running** and reachable from the cluster (`sut-url`); the
pipeline does not build or deploy it. Each mission brings **its own judge** (provider, model, key).
Results and baselines go to **Azure Blob Storage** (Azurite for local clusters).

---

## 1. One-time cluster setup

### 1.1 A cluster with Tekton

Any Kubernetes cluster with Tekton Pipelines serving the `tekton.dev/v1` API (tested on v1.15.1). Locally, with Docker Desktop:

```bash
kind create cluster --name elyaeval
kubectl apply -f https://storage.googleapis.com/tekton-releases/pipeline/latest/release.yaml
kubectl create namespace elyaeval
```

### 1.2 The runner image

Every Python step runs in `elyaeval-runner:<version>`: elyaeval + DeepEval + the Ollama and
Anthropic SDKs, preinstalled. Build it from the root of an elyaeval checkout:

```bash
docker build -f src/elyaeval/recipes/docker/runner.Dockerfile -t elyaeval-runner:0.1.2 .
```

Then make it available to the cluster, either:

- **kind** (local): `kind load docker-image elyaeval-runner:0.1.2 --name elyaeval`
- **a registry** (shared clusters), e.g. GitHub Container Registry:
  ```bash
  docker tag elyaeval-runner:0.1.2 ghcr.io/<you>/elyaeval-runner:0.1.2
  docker push ghcr.io/<you>/elyaeval-runner:0.1.2
  ```
  and use that full name everywhere below (`--runner-image`, or `ELYAEVAL_RUNNER_IMAGE`).

Other judge SDKs: `--build-arg ELYAEVAL_EXTRAS="ollama,anthropic,gemini"`. Without an elyaeval
checkout, `elyaeval ci dockerfile > runner.Dockerfile` plus
`--build-arg ELYAEVAL_SPEC="elyaeval[ollama,anthropic] @ git+https://github.com/<owner>/<repo>.git@v0.1.2"`
builds the same image from a git tag.

### 1.3 The Pipeline and Tasks

```bash
pip install "elyaeval @ git+https://github.com/<owner>/<repo>.git@v0.1.2"
elyaeval ci apply -n elyaeval                      # runner image defaults to elyaeval-runner:<this version>
elyaeval ci apply -n elyaeval --runner-image ghcr.io/<you>/elyaeval-runner:0.1.2
elyaeval ci render --output-dir ci/                # or just write the YAML, to review / kubectl apply yourself
```

This installs Pipeline `elyaeval` and Tasks `elyaeval-*` (prefixed, so they can share a namespace).
Re-run it when you upgrade elyaeval.

### 1.4 Results storage (Azure Blob / Azurite)

Real Azure: any Storage Account; use its connection string below. Locally: Azurite, the Azure Storage
emulator (`docker compose up -d` at the root of the elyaeval repo starts it on ports
10000-10002). From inside kind, Azurite's well-known development connection string is:

```
DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://host.docker.internal:10000/devstoreaccount1;
```

Each mission's container is created automatically on its first run.

### 1.5 Secrets (once per namespace)

```bash
# Azure Blob / Azurite
kubectl create secret generic blob-storage-credentials -n elyaeval \
  --from-literal=connection-string="<connection string>"

# the judge's API key   any placeholder for local judges (Ollama ignores it)
kubectl create secret generic judge-model-credentials -n elyaeval \
  --from-literal=api-key="<key or 'ollama'>"
```

Different missions with different judge keys: create one Secret per mission and set
`judge-api-key-secret-name` in that mission's PipelineRun.

---

## 2. Onboarding a mission

In the mission's repo (public GitHub repos clone without credentials):

```bash
pip install "elyaeval[ollama] @ git+https://github.com/<owner>/<repo>.git@v0.1.2"
elyaeval init --task-type rag_qa                    # the test file; fill in run_app() (calls APP_URL)
# + goldens file, elyaeval.toml (judge, thresholds, budget, regression gate)
elyaeval init-ci --blob-container <mission-name> --ci-stage per_commit --budget-seconds 1800
git add test_elyaeval_*.py goldens*.jsonl elyaeval.toml pipelinerun.yaml && git commit && git push
```

`init-ci` pins the runner image to the installed elyaeval version. Useful flags: `--sut-path`
(test file in a sub-folder), `--repo-url` (otherwise `${REPO_URL}` at run time), `--revision`,
`--runner-image`, `--timeout`.

## 3. Running the gate

1. Start the app (and a local judge if you use one) so the cluster can reach them   with Docker
   Desktop / kind: `http://host.docker.internal:<port>`.
2. From the mission repo:
   ```powershell
   $env:REPO_URL         = "https://github.com/<you>/<mission>.git"
   $env:SUT_URL          = "http://host.docker.internal:8000"
   $env:JUDGE_PROVIDER   = "ollama"                               # or anthropic, openai, ...
   $env:JUDGE_MODEL_NAME = "llama3.1"
   $env:JUDGE_BASE_URL   = "http://host.docker.internal:11434"    # "" for hosted providers
   envsubst < pipelinerun.yaml | kubectl create -n elyaeval -f -
   ```
3. Follow it: `kubectl get pipelineruns -n elyaeval -w`, logs with
   `tkn pipelinerun logs -f -n elyaeval <name>` (or the Tekton Dashboard, or
   `kubectl logs -n elyaeval -l tekton.dev/pipelineRun=<name> --all-containers`).

### What fails the run

| Gate | Fails when |
|---|---|
| `check-results` | any golden misses a metric threshold, errors, or nothing was collected |
| `check-budget` | the run exhausted its budget (`budget-*` params / `[tool.elyaeval.budget]`); budget-skipped goldens are pytest *skips*, so this is what surfaces them |
| `check-regression` | the scores regressed against the stored baseline (paired, statistically significant drop, or more errors), or the baseline was scored by a different judge |

`report-results` uploads, per results CSV, `<run-id>_<suffix>.csv/.html/.meta.json` and the run's
`<run-id>_budget.json` to the mission's container   also for failing runs.

### Baselines

There is no baseline until you create one: run once with `promote-baseline: "true"`. The run's CSV
(+ its `.meta.json`) becomes `baseline_<suffix>.csv` in the container   **only if every gate
passed** (thresholds, budget, regression). Set it back to `"false"` afterwards, so ordinary runs never
move the baseline. Re-promote whenever you change the judge (runs scored by a different judge are
refused as incomparable) or after an intentional quality change.

### Pipeline parameters

Render the Pipeline (`elyaeval ci render`) to see every parameter with its description. The ones a
mission usually touches:

| Param | Default | Meaning |
|---|---|---|
| `repo-url`, `revision` |   / `main` | mission repo and branch / tag / commit |
| `sut-path`, `test-file` | `.` /   | folder with the test file + its goldens/config; the test file |
| `sut-url` |   | the running app (`APP_URL` in tests) |
| `runner-image` | pinned by `init-ci` | elyaeval version that evaluates |
| `judge-provider`, `judge-model-name`, `judge-model-base-url` |   | the mission's judge |
| `judge-api-key-secret-name` | `judge-model-credentials` | Secret holding the judge key |
| `ci-stage`, `repeats` | file's / `1` | which goldens; repeated runs for noise |
| `budget-seconds`, `budget-usd`, `budget-tokens`, `budget-calls`, `budget-on-exceed` | config | evaluation budget |
| `regression-tolerance`, `regression-alpha`, `max-error-rate-increase`, `max-pass-rate-drop`, `allow-metadata-mismatch` | config | regression gate |
| `promote-baseline` | `false` | make this run the baseline (only if every gate passes) |
| `blob-container` |   | the mission's results container |

Component suites (which import the app in-process) also need `sut-requirements-file` and
`sut-env-secret-name` (a Secret with the app's own environment).

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| `ErrImageNeverPull` / `ImagePullBackOff` on runner steps | load (`kind load docker-image`) or push the runner image; tag must match `runner-image` |
| `elyaeval judge` fails in run-suite | wrong `judge-*` params / Secret; for Ollama use the native URL (`:11434`, no `/v1`) |
| connection refused to the app or Ollama | they must listen on `0.0.0.0` and be reachable at `host.docker.internal` from the cluster |
| `no tests collected` | wrong `test-file` / `sut-path`, or no goldens at that `ci-stage` |
| budget exhausted on every run | a local judge needs minutes per golden: raise `budget-seconds`, lower the stage, or use a faster judge |
| regression gate exits with "different judge" | promote a new baseline with the new judge |
| pipeline timeout | raise `--timeout` in `init-ci` (default 2h) |
