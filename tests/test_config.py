import importlib
import sys
from types import SimpleNamespace

import pytest


class FakeSecrets(dict):
    def __getattr__(self, name):
        return self[name]


@pytest.fixture(autouse=True)
def clear_reloaded_config():
    yield
    sys.modules.pop("config", None)


def reload_config(monkeypatch, secrets):
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets=FakeSecrets(secrets)))
    sys.modules.pop("config", None)
    return importlib.import_module("config")


def test_streamlit_secrets_override_local_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "local-groq-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    config = reload_config(
        monkeypatch,
        {
            "GROQ_API_KEY": "cloud-groq-key",
            "LLM_PROVIDER": "groq",
        },
    )

    assert config.GROQ_API_KEY == "cloud-groq-key"
    assert config.LLM_PROVIDER == "groq"


def test_streamlit_secret_sections_are_flattened_for_runtime_env(monkeypatch):
    config = reload_config(
        monkeypatch,
        {
            "llm": {
                "OPENAI_API_KEY": "cloud-openai-key",
                "LLM_MAX_TOKENS": 900,
            },
        },
    )

    assert config.OPENAI_API_KEY == "cloud-openai-key"
    assert config.LLM_MAX_TOKENS == 900
