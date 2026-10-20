# elyaeval runner image: what every Python step of the elyaeval Tekton
# recipe runs in. elyaeval + DeepEval + judge-provider SDKs are installed at
# build time, so a pipeline run installs nothing (fast, reproducible) and a
# mission pins exactly which elyaeval version evaluates it.
#
# Build from the root of an elyaeval source checkout:
#   docker build -f src/elyaeval/recipes/docker/runner.Dockerfile -t elyaeval-runner:<version> .
#
# Or from anywhere, installing elyaeval from a git tag instead of the context
# (`elyaeval ci dockerfile > runner.Dockerfile` writes this file out):
#   docker build -f runner.Dockerfile -t elyaeval-runner:<version> \
#     --build-arg ELYAEVAL_SPEC="elyaeval[ollama,anthropic] @ git+https://github.com/<owner>/<repo>.git@v<version>" .
#
# ELYAEVAL_EXTRAS selects judge-provider SDKs (ollama, anthropic, gemini,
# bedrock, litellm; OpenAI-based providers need none).
FROM python:3.11-slim

ARG ELYAEVAL_SPEC=""
ARG ELYAEVAL_EXTRAS="ollama,anthropic"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    ERROR_REPORTING=NO

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY . /tmp/elyaeval-src
RUN if [ -n "$ELYAEVAL_SPEC" ]; then \
      pip install --no-cache-dir "$ELYAEVAL_SPEC"; \
    else \
      pip install --no-cache-dir "/tmp/elyaeval-src[${ELYAEVAL_EXTRAS}]"; \
    fi \
 && rm -rf /tmp/elyaeval-src \
 && elyaeval --help >/dev/null

LABEL org.opencontainers.image.title="elyaeval-runner" \
      org.opencontainers.image.description="elyaeval + DeepEval runner for the elyaeval Tekton recipe"
