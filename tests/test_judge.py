"""Judge selection: config precedence, provider mapping, secret handling."""

import importlib.util

import pytest

from elyaeval.core.judge import (
    JudgeConfigError,
    describe_judge,
    get_judge,
    resolve_judge_config,
)


def _write_pyproject(path, body: str) -> None:
    (path / "pyproject.toml").write_text(body, encoding="utf-8")


def test_default_defers_to_deepeval():
    cfg = resolve_judge_config()
    assert cfg.provider == "deepeval"
    assert get_judge() is None
    assert describe_judge() == "deepeval:default"


def test_pyproject_config_is_read(isolated_config):
    _write_pyproject(
        isolated_config,
        '[tool.elyaeval.judge]\nprovider = "ollama"\nmodel = "llama3.1"\n'
        'base_url = "http://localhost:11434"\n',
    )
    cfg = resolve_judge_config()
    assert (cfg.provider, cfg.model, cfg.base_url) == ("ollama", "llama3.1", "http://localhost:11434")


def test_dedicated_elyaeval_toml_is_read(isolated_config):
    (isolated_config / "elyaeval.toml").write_text(
        '[judge]\nprovider = "anthropic"\nmodel = "claude-sonnet-5-5"\n', encoding="utf-8"
    )
    assert describe_judge() == "anthropic:claude-sonnet-5-5"


def test_config_found_from_subdirectory(isolated_config, monkeypatch):
    _write_pyproject(isolated_config, '[tool.elyaeval.judge]\nprovider = "openai"\nmodel = "gpt-4.1"\n')
    sub = isolated_config / "tests" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert describe_judge() == "openai:gpt-4.1"


def test_env_overrides_file_and_kwargs_override_env(isolated_config, monkeypatch):
    _write_pyproject(isolated_config, '[tool.elyaeval.judge]\nprovider = "ollama"\nmodel = "llama3.1"\n')
    monkeypatch.setenv("ELYAEVAL_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("ELYAEVAL_JUDGE_MODEL", "gpt-4.1-mini")
    assert describe_judge() == "openai:gpt-4.1-mini"
    assert describe_judge(provider="anthropic", model="claude-haiku-5-5") == "anthropic:claude-haiku-5-5"


@pytest.mark.parametrize("alias,canonical", [("local", "openai_compatible"), ("Azure", "azure_openai"), ("claude", "anthropic")])
def test_provider_aliases(alias, canonical):
    assert resolve_judge_config(provider=alias).provider == canonical


def test_unknown_provider_is_rejected():
    with pytest.raises(JudgeConfigError, match="Unknown judge provider"):
        resolve_judge_config(provider="not-a-provider")


def test_api_key_in_config_file_is_rejected(isolated_config):
    _write_pyproject(isolated_config, '[tool.elyaeval.judge]\nprovider = "openai"\napi_key = "sk-oops"\n')
    with pytest.raises(JudgeConfigError, match="api_key"):
        resolve_judge_config()


def test_unknown_config_key_is_rejected(isolated_config):
    _write_pyproject(isolated_config, '[tool.elyaeval.judge]\nprovider = "openai"\nmodle = "typo"\n')
    with pytest.raises(JudgeConfigError, match="modle"):
        resolve_judge_config()


def test_api_key_from_env_never_described(monkeypatch):
    monkeypatch.setenv("ELYAEVAL_JUDGE_API_KEY", "sk-secret-123")
    cfg = resolve_judge_config(provider="openai", model="gpt-4.1")
    assert cfg.api_key == "sk-secret-123"
    assert "sk-secret" not in cfg.describe()
    assert "sk-secret" not in repr(cfg)


def test_api_key_env_indirection(isolated_config, monkeypatch):
    _write_pyproject(
        isolated_config,
        '[tool.elyaeval.judge]\nprovider = "openai"\nmodel = "gpt-4.1"\napi_key_env = "MY_TEAM_KEY"\n',
    )
    monkeypatch.setenv("MY_TEAM_KEY", "sk-team")
    assert resolve_judge_config().api_key == "sk-team"


def test_ollama_builds_native_model():
    from deepeval.models import OllamaModel

    judge = get_judge(provider="ollama", model="llama3.1", base_url="http://localhost:11434")
    assert isinstance(judge, OllamaModel)
    assert judge.get_model_name().startswith("llama3.1")


def test_openai_compatible_builds_local_model():
    from deepeval.models import LocalModel

    judge = get_judge(provider="local", model="llama3.1", base_url="http://localhost:11434/v1", api_key="ollama")
    assert isinstance(judge, LocalModel)


def test_openai_builds_openai_model():
    from deepeval.models import OpenAIModel

    judge = get_judge(provider="openai", model="gpt-4.1", api_key="sk-test")
    assert isinstance(judge, OpenAIModel)


def test_judge_instance_is_shared_per_config():
    a = get_judge(provider="ollama", model="llama3.1")
    b = get_judge(provider="ollama", model="llama3.1")
    c = get_judge(provider="ollama", model="llama3.2")
    assert a is b and a is not c


def test_options_are_passed_through():
    judge = get_judge(provider="ollama", model="llama3.1", options={"generation_kwargs": {"num_ctx": 8192}})
    assert judge.generation_kwargs == {"num_ctx": 8192, "num_predict": 2048}


def test_local_judges_cap_reply_length_by_default():
    assert get_judge(provider="ollama", model="llama3.1").generation_kwargs == {"num_predict": 2048}
    local = get_judge(provider="openai_compatible", model="m", base_url="http://localhost:1234/v1", api_key="x")
    assert local.generation_kwargs == {"max_tokens": 2048}
    assert resolve_judge_config(provider="openai", model="gpt-4.1").max_output_tokens is None


def test_reply_cap_is_configurable(monkeypatch):
    assert get_judge(provider="ollama", model="llama3.1", max_output_tokens=512).generation_kwargs == {"num_predict": 512}
    assert get_judge(provider="ollama", model="llama3.1", max_output_tokens=0).generation_kwargs == {}
    # a cap set directly in options wins over the default
    judge = get_judge(provider="ollama", model="llama3.1", options={"generation_kwargs": {"num_predict": 100}})
    assert judge.generation_kwargs == {"num_predict": 100}
    monkeypatch.setenv("ELYAEVAL_JUDGE_MAX_OUTPUT_TOKENS", "300")
    assert get_judge(provider="ollama", model="llama3.1").generation_kwargs == {"num_predict": 300}


def test_custom_class_provider(isolated_config, monkeypatch):
    (isolated_config / "my_judges.py").write_text(
        "from deepeval.models import DeepEvalBaseLLM\n"
        "class Gateway(DeepEvalBaseLLM):\n"
        "    def __init__(self, model=None, temperature=0.0):\n"
        "        self.temperature = temperature\n"
        "        super().__init__(model)\n"
        "    def load_model(self): return self\n"
        "    def generate(self, prompt, schema=None): return ''\n"
        "    async def a_generate(self, prompt, schema=None): return ''\n"
        "    def get_model_name(self): return 'gateway'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(isolated_config))
    judge = get_judge(provider="custom", class_path="my_judges:Gateway", model="internal-llm")
    assert type(judge).__name__ == "Gateway"
    assert describe_judge(provider="custom", class_path="my_judges:Gateway", model="m") == "custom:my_judges:Gateway:m"


def test_custom_provider_requires_class():
    with pytest.raises(JudgeConfigError, match="class"):
        resolve_judge_config(provider="custom")


@pytest.mark.skipif(importlib.util.find_spec("anthropic") is not None, reason="anthropic SDK installed")
def test_missing_provider_sdk_gives_install_hint():
    with pytest.raises(JudgeConfigError, match=r"elyaeval\[anthropic\]"):
        get_judge(provider="anthropic", model="claude-sonnet-5-5", api_key="sk-test")


def test_local_providers_get_a_longer_default_timeout():
    assert resolve_judge_config(provider="ollama", model="llama3.1").timeout_seconds == 900
    assert resolve_judge_config(provider="local", model="m").timeout_seconds == 900
    assert resolve_judge_config(provider="anthropic", model="m").timeout_seconds is None


def test_timeout_applied_to_deepeval(monkeypatch):
    import os

    get_judge(provider="ollama", model="llama3.1", timeout_seconds=1200)
    assert os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] == "1200"


def test_timeout_zero_disables_deepeval_timeouts():
    import os

    get_judge(provider="ollama", model="llama3.1", timeout_seconds=0)
    assert os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] == "1"


def test_explicit_deepeval_timeout_env_wins(monkeypatch):
    import os

    monkeypatch.setenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "42")
    get_judge(provider="ollama", model="llama3.1", timeout_seconds=1200)
    assert os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] == "42"


def test_timeout_from_config_file(isolated_config):
    _write_pyproject(
        isolated_config, '[tool.elyaeval.judge]\nprovider = "anthropic"\nmodel = "m"\ntimeout_seconds = 300\n'
    )
    assert resolve_judge_config().timeout_seconds == 300
