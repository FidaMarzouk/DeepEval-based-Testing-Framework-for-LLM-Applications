"""
Provider-agnostic judge-model selection.

Every metric preset in elyaeval.core.metrics takes its judge from
get_judge(), so switching a whole suite between a local Ollama model, an
Anthropic/OpenAI model, Azure, Gemini, Bedrock, any OpenAI-compatible server
(vLLM, LM Studio, Groq, ...) or your own DeepEvalBaseLLM subclass is a config
change, never a code change.

Resolution order, highest priority first:
  1. keyword arguments to get_judge() / resolve_judge_config()
  2. environment variables   ELYAEVAL_JUDGE_PROVIDER, ELYAEVAL_JUDGE_MODEL,
                              ELYAEVAL_JUDGE_BASE_URL, ELYAEVAL_JUDGE_API_KEY,
                              ELYAEVAL_JUDGE_TEMPERATURE, ELYAEVAL_JUDGE_CLASS
  3. [tool.elyaeval.judge] in pyproject.toml (or elyaeval.toml)
  4. provider "deepeval": defer to whatever DeepEval itself is configured
     with (`deepeval set-ollama`, USE_LOCAL_MODEL, OPENAI_API_KEY, ...) —
     i.e. exactly the pre-elyaeval-config behaviour.

API keys: ELYAEVAL_JUDGE_API_KEY, or the env var named by `api_key_env` in
the config, or — if neither is set — the provider's own variable
(ANTHROPIC_API_KEY, OPENAI_API_KEY, ...), which DeepEval reads natively.
Keys are never read from config files and never included in describe().

Timeouts: DeepEval caps each test case at 180s by default — fine for hosted
APIs, far too short for a local model on a laptop GPU. `timeout_seconds`
(env ELYAEVAL_JUDGE_TIMEOUT_SECONDS) sets that per-case budget (one metric
per case: evaluate_golden runs metrics one evaluate() call at a time); 0 disables DeepEval's timeouts. Unset, local providers (ollama,
openai_compatible) get LOCAL_TIMEOUT_SECONDS and hosted providers keep
DeepEval's default. An explicit DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE /
DEEPEVAL_DISABLE_TIMEOUTS in the environment always wins.

Reply length: a small local judge can get stuck repeating itself inside
schema-constrained JSON and generate for many minutes. `max_output_tokens`
(env ELYAEVAL_JUDGE_MAX_OUTPUT_TOKENS) caps every reply; local providers
default to LOCAL_MAX_OUTPUT_TOKENS, 0 disables the cap. A runaway reply then
fails fast as unparsable JSON instead of eating the test case's timeout. A
cap set directly in `options.generation_kwargs` takes precedence.

Rate limits (free tiers such as Groq): `max_calls_per_minute` and
`max_tokens_per_minute` (estimated from prompt size) pace judge calls with a
sliding 60 s window. Every judge built here is also metered (calls, time,
estimated tokens) for budgets and reports, see elyaeval.core.usage.

Provider-specific constructor arguments (Azure deployment_name/api_version,
Gemini project/location, Bedrock region, cost_per_input_token, ...) go in an
`options` sub-table:

    [tool.elyaeval.judge]
    provider = "azure_openai"
    model = "gpt-4.1"
    base_url = "https://my-resource.openai.azure.com/"
    options = { deployment_name = "judge", api_version = "2024-10-21" }
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .config import config_section

if TYPE_CHECKING:
    from deepeval.models import DeepEvalBaseLLM

# provider name -> DeepEval model class name in deepeval.models.
# "deepeval" (None) means: don't pass a model, let DeepEval use its default.
PROVIDERS: dict[str, Optional[str]] = {
    "deepeval": None,
    "ollama": "OllamaModel",
    "openai": "OpenAIModel",
    "anthropic": "AnthropicModel",
    "azure_openai": "AzureOpenAIModel",
    "gemini": "GeminiModel",
    "bedrock": "AmazonBedrockModel",
    "litellm": "LiteLLMModel",
    "openai_compatible": "LocalModel",
    "deepseek": "DeepSeekModel",
    "openrouter": "OpenRouterModel",
    "grok": "GrokModel",
    "custom": None,  # import path given by `class` / ELYAEVAL_JUDGE_CLASS
}
PROVIDER_ALIASES = {"local": "openai_compatible", "azure": "azure_openai", "claude": "anthropic"}

# pip extra that installs a provider's SDK (DeepEval imports these lazily)
PROVIDER_EXTRAS = {
    "ollama": "ollama",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "bedrock": "bedrock",
    "litellm": "litellm",
}

LOCAL_PROVIDERS = frozenset({"ollama", "openai_compatible"})
LOCAL_TIMEOUT_SECONDS = 900.0
LOCAL_MAX_OUTPUT_TOKENS = 2048

# generation_kwargs key that caps a reply's length, per provider
_MAX_OUTPUT_KEY = {"ollama": "num_predict", "openai_compatible": "max_tokens"}

_ENV = {
    "provider": "ELYAEVAL_JUDGE_PROVIDER",
    "timeout_seconds": "ELYAEVAL_JUDGE_TIMEOUT_SECONDS",
    "max_output_tokens": "ELYAEVAL_JUDGE_MAX_OUTPUT_TOKENS",
    "max_calls_per_minute": "ELYAEVAL_JUDGE_MAX_CALLS_PER_MINUTE",
    "max_tokens_per_minute": "ELYAEVAL_JUDGE_MAX_TOKENS_PER_MINUTE",
    "model": "ELYAEVAL_JUDGE_MODEL",
    "base_url": "ELYAEVAL_JUDGE_BASE_URL",
    "temperature": "ELYAEVAL_JUDGE_TEMPERATURE",
    "class_path": "ELYAEVAL_JUDGE_CLASS",
}
API_KEY_ENV = "ELYAEVAL_JUDGE_API_KEY"


class JudgeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class JudgeConfig:
    provider: str = "deepeval"
    model: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = 0.0
    class_path: Optional[str] = None
    timeout_seconds: Optional[float] = None
    max_output_tokens: Optional[int] = None
    max_calls_per_minute: Optional[int] = None
    max_tokens_per_minute: Optional[int] = None
    options: tuple[tuple[str, Any], ...] = ()
    api_key: Optional[str] = field(default=None, repr=False, compare=True)

    def describe(self) -> str:
        """Secret-free identifier, e.g. 'ollama:llama3.1' — used in reports
        and (Phase 2) baseline metadata."""
        if self.provider == "deepeval":
            return "deepeval:default"
        if self.provider == "custom":
            return f"custom:{self.class_path}" + (f":{self.model}" if self.model else "")
        return f"{self.provider}:{self.model or 'default'}"


def _normalize_provider(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    key = PROVIDER_ALIASES.get(key, key)
    if key not in PROVIDERS:
        raise JudgeConfigError(
            f"Unknown judge provider '{name}'. Valid: {', '.join(sorted(PROVIDERS))} "
            f"(aliases: {', '.join(sorted(PROVIDER_ALIASES))})"
        )
    return key


def resolve_judge_config(**overrides: Any) -> JudgeConfig:
    file_cfg = config_section("judge")
    unknown = set(file_cfg) - {
        "provider", "model", "base_url", "temperature", "class", "options", "api_key_env", "timeout_seconds",
        "max_output_tokens", "max_calls_per_minute", "max_tokens_per_minute",
    }
    if "api_key" in file_cfg:
        raise JudgeConfigError(
            "Don't put api_key in a config file — set ELYAEVAL_JUDGE_API_KEY or the "
            "provider's own environment variable, or name one with api_key_env."
        )
    if unknown:
        raise JudgeConfigError(f"Unknown key(s) in [tool.elyaeval.judge]: {sorted(unknown)}")

    def pick(name: str, file_key: Optional[str] = None) -> Any:
        if overrides.get(name) is not None:
            return overrides[name]
        env_value = os.environ.get(_ENV[name]) if name in _ENV else None
        if env_value not in (None, ""):
            return env_value
        return file_cfg.get(file_key or name)

    provider = _normalize_provider(str(pick("provider") or "deepeval"))
    temperature = pick("temperature")
    temperature = 0.0 if temperature is None else float(temperature)
    timeout = pick("timeout_seconds")
    if timeout is None and provider in LOCAL_PROVIDERS:
        timeout = LOCAL_TIMEOUT_SECONDS
    max_output = _opt_int(pick("max_output_tokens"))
    if max_output is None and provider in LOCAL_PROVIDERS:
        max_output = LOCAL_MAX_OUTPUT_TOKENS

    api_key = overrides.get("api_key") or os.environ.get(API_KEY_ENV) or None
    if api_key is None and file_cfg.get("api_key_env"):
        api_key = os.environ.get(file_cfg["api_key_env"]) or None

    options = dict(file_cfg.get("options", {}))
    options.update(overrides.get("options") or {})
    cap_key = _MAX_OUTPUT_KEY.get(provider)
    if cap_key and max_output and max_output > 0:
        # an explicit generation_kwargs cap in `options` wins
        generation_kwargs = dict(options.get("generation_kwargs") or {})
        generation_kwargs.setdefault(cap_key, max_output)
        options["generation_kwargs"] = generation_kwargs

    cfg = JudgeConfig(
        provider=provider,
        model=pick("model"),
        base_url=pick("base_url"),
        temperature=temperature,
        class_path=pick("class_path", "class"),
        timeout_seconds=None if timeout is None else float(timeout),
        max_output_tokens=max_output,
        max_calls_per_minute=_opt_int(pick("max_calls_per_minute")),
        max_tokens_per_minute=_opt_int(pick("max_tokens_per_minute")),
        options=tuple(sorted(options.items())),
        api_key=api_key,
    )
    if provider == "custom" and not cfg.class_path:
        raise JudgeConfigError("provider 'custom' needs `class = \"package.module:ClassName\"`")
    return cfg


def _opt_int(value: Any) -> Optional[int]:
    return None if value in (None, "") else int(value)


def _import_class(path: str) -> type:
    module_name, _, attr = path.replace(":", ".").rpartition(".")
    if not module_name:
        raise JudgeConfigError(f"class path '{path}' must look like 'package.module:ClassName'")
    return getattr(importlib.import_module(module_name), attr)


def _filtered_kwargs(cls: type, candidates: dict[str, Any]) -> dict[str, Any]:
    """Pass only the arguments this provider's constructor actually accepts
    (by name), so one config shape works for every provider. Options are
    always passed through — they're explicitly provider-specific."""
    params = inspect.signature(cls).parameters
    return {k: v for k, v in candidates.items() if v is not None and k in params}


_JUDGE_CACHE: dict[str, Optional[DeepEvalBaseLLM]] = {}


def _build(cfg: JudgeConfig) -> Optional[DeepEvalBaseLLM]:
    """One shared instance per distinct config (keyed on a canonical JSON
    form, since options may contain nested dicts)."""
    key = json.dumps(asdict(cfg), sort_keys=True, default=str)
    if key not in _JUDGE_CACHE:
        _JUDGE_CACHE[key] = _construct(cfg)
    return _JUDGE_CACHE[key]


_build.cache_clear = _JUDGE_CACHE.clear  # type: ignore[attr-defined]


def _construct(cfg: JudgeConfig) -> Optional[DeepEvalBaseLLM]:
    if cfg.provider == "deepeval":
        return None

    if cfg.provider == "custom":
        cls = _import_class(cfg.class_path or "")
    else:
        import deepeval.models

        cls = getattr(deepeval.models, PROVIDERS[cfg.provider] or "")

    common = {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "temperature": cfg.temperature,
        "api_key": cfg.api_key,
    }
    kwargs = {**_filtered_kwargs(cls, common), **dict(cfg.options)}
    try:
        judge = cls(**kwargs)
    except Exception as exc:
        # DeepEval wraps a missing provider SDK (ImportError) in DeepEvalError
        missing_sdk = isinstance(exc, ImportError) or isinstance(exc.__cause__, ImportError)
        if not missing_sdk:
            raise
        extra = PROVIDER_EXTRAS.get(cfg.provider)
        hint = f' Install it with: pip install "elyaeval[{extra}]"' if extra else ""
        raise JudgeConfigError(f"Judge provider '{cfg.provider}' is missing its SDK ({exc}).{hint}") from exc

    from .usage import RateLimiter, instrument_judge

    limiter = None
    if cfg.max_calls_per_minute or cfg.max_tokens_per_minute:
        limiter = RateLimiter(cfg.max_calls_per_minute, cfg.max_tokens_per_minute)
    return instrument_judge(judge, limiter)


def apply_timeout(cfg: JudgeConfig) -> None:
    """Translate cfg.timeout_seconds into DeepEval's timeout settings (read
    from the environment, re-read whenever it changes). Never overrides a
    DEEPEVAL_* timeout the user set explicitly."""
    if cfg.timeout_seconds is None:
        return
    if "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE" in os.environ or "DEEPEVAL_DISABLE_TIMEOUTS" in os.environ:
        return
    if cfg.timeout_seconds <= 0:
        os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
    else:
        os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = f"{cfg.timeout_seconds:g}"


def get_judge(**overrides: Any) -> Optional[DeepEvalBaseLLM]:
    """The judge model every metric preset uses: a DeepEvalBaseLLM instance,
    or None to let DeepEval use its own default. One instance is shared per
    distinct configuration (clients are reused across metrics/goldens).
    Also applies the judge's timeout policy (see apply_timeout)."""
    cfg = resolve_judge_config(**overrides)
    apply_timeout(cfg)
    return _build(cfg)


def describe_judge(**overrides: Any) -> str:
    return resolve_judge_config(**overrides).describe()


def ping_judge(**overrides: Any) -> str:
    """One tiny schema-constrained call — used by `elyaeval judge --ping` to
    check connectivity, credentials and structured-output support before
    burning a whole suite on a misconfigured judge."""
    from pydantic import BaseModel

    class _Pong(BaseModel):
        verdict: str
        score: float

    judge = get_judge(**overrides)
    if judge is None:
        from deepeval.metrics.utils import initialize_model

        judge, _ = initialize_model(None)
    prompt = (
        'Return JSON with keys "verdict" (the word "yes") and "score" (the number 1). '
        "No other text."
    )
    result: Any = judge.generate(prompt, schema=_Pong)
    # DeepEval models return either the parsed schema or (parsed, cost)
    parsed: Any = result[0] if isinstance(result, tuple) else result
    if isinstance(parsed, str):
        parsed = _Pong.model_validate_json(parsed)
    return f"{judge.get_model_name()} -> {parsed.model_dump()}"
