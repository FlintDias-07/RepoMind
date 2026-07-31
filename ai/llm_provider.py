"""Provider abstraction for Groq and OpenAI completion APIs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import sleep
from typing import Dict, Optional, Sequence, Type

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TEMPERATURE,
)
from utils import get_logger

from .prompt_builder import PromptPackage


@dataclass(frozen=True)
class ProviderSettings:
    provider_name: str
    model: str
    api_key: str
    temperature: float = TEMPERATURE
    max_tokens: int = LLM_MAX_TOKENS
    timeout_seconds: float = LLM_TIMEOUT_SECONDS
    max_retries: int = LLM_MAX_RETRIES


@dataclass(frozen=True)
class LLMResponse:
    content: str
    provider_name: str
    model: str
    finish_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Common provider contract for the AI pipeline."""

    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def generate(self, prompt: PromptPackage) -> LLMResponse:
        raise NotImplementedError


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        self._client = None

    def generate(self, prompt: PromptPackage) -> LLMResponse:
        client = self._ensure_client()
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    messages=[
                        {"role": "system", "content": prompt.system_prompt},
                        {"role": "user", "content": prompt.user_prompt},
                    ],
                )
                message = response.choices[0].message.content if response.choices else ""
                usage = {}
                if getattr(response, "usage", None):
                    usage = {
                        "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
                    }
                return LLMResponse(
                    content=message or "",
                    provider_name="openai",
                    model=self.settings.model,
                    finish_reason=str(response.choices[0].finish_reason if response.choices else ""),
                    usage=usage,
                )
            except Exception as exc:
                last_error = exc
                self.logger.warning("OpenAI request attempt %s failed: %s", attempt + 1, exc)
                if attempt < self.settings.max_retries:
                    sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"OpenAI generation failed: {last_error}") from last_error

    def _ensure_client(self):
        if self._client is None:
            if not self.settings.api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.api_key, timeout=self.settings.timeout_seconds)
        return self._client


class GroqProvider(BaseLLMProvider):
    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        self._client = None

    def generate(self, prompt: PromptPackage) -> LLMResponse:
        client = self._ensure_client()
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    messages=[
                        {"role": "system", "content": prompt.system_prompt},
                        {"role": "user", "content": prompt.user_prompt},
                    ],
                )
                message = response.choices[0].message.content if response.choices else ""
                usage = {}
                if getattr(response, "usage", None):
                    usage = {
                        "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
                    }
                return LLMResponse(
                    content=message or "",
                    provider_name="groq",
                    model=self.settings.model,
                    finish_reason=str(response.choices[0].finish_reason if response.choices else ""),
                    usage=usage,
                )
            except Exception as exc:
                last_error = exc
                self.logger.warning("Groq request attempt %s failed: %s", attempt + 1, exc)
                if attempt < self.settings.max_retries:
                    sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"Groq generation failed: {last_error}") from last_error

    def _ensure_client(self):
        if self._client is None:
            if not self.settings.api_key:
                raise RuntimeError("GROQ_API_KEY is not configured")
            from groq import Groq

            self._client = Groq(api_key=self.settings.api_key, timeout=self.settings.timeout_seconds)
        return self._client


class LLMProviderFactory:
    """Select provider implementations without coupling the pipeline to SDKs."""

    def __init__(self, registry: Optional[Dict[str, Type[BaseLLMProvider]]] = None):
        self.registry = registry or {
            "openai": OpenAIProvider,
            "groq": GroqProvider,
        }

    def create(self, provider_name: str | None = None, overrides: Optional[Dict[str, object]] = None) -> BaseLLMProvider:
        selected = (provider_name or LLM_PROVIDER or "groq").strip().lower()
        if selected not in self.registry:
            raise ValueError(f"Unsupported LLM provider: {selected}")
        settings = self._settings_for(selected, overrides or {})
        return self.registry[selected](settings)

    def _settings_for(self, provider_name: str, overrides: Dict[str, object]) -> ProviderSettings:
        if provider_name == "openai":
            model = str(overrides.get("model") or OPENAI_MODEL or "gpt-4o-mini")
            api_key = str(overrides.get("api_key") or OPENAI_API_KEY or "")
        else:
            model = str(overrides.get("model") or GROQ_MODEL or "mixtral-8x7b-32768")
            api_key = str(overrides.get("api_key") or GROQ_API_KEY or "")
        return ProviderSettings(
            provider_name=provider_name,
            model=model,
            api_key=api_key,
            temperature=float(overrides.get("temperature", TEMPERATURE)),
            max_tokens=int(overrides.get("max_tokens", LLM_MAX_TOKENS)),
            timeout_seconds=float(overrides.get("timeout_seconds", LLM_TIMEOUT_SECONDS)),
            max_retries=int(overrides.get("max_retries", LLM_MAX_RETRIES)),
        )
