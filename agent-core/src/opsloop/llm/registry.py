"""Provider registry and the presets the settings menu offers.

The settings menu lets you add any endpoint by hand - kind, base URL, key,
model - so nothing here is a restriction. The presets exist only to save
typing for the endpoints people actually use.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .base import ConnectionTest, LLMProvider, ProviderConfig, ProviderKind
from .providers import AnthropicProvider, GoogleProvider, OpenAICompatibleProvider

_IMPLEMENTATIONS: dict[ProviderKind, type[LLMProvider]] = {
    ProviderKind.OPENAI_COMPATIBLE: OpenAICompatibleProvider,
    ProviderKind.ANTHROPIC: AnthropicProvider,
    ProviderKind.GOOGLE: GoogleProvider,
}


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    kind: ProviderKind
    base_url: str
    suggested_models: tuple[str, ...]
    needs_key: bool = True
    hint: str = ""


PRESETS: tuple[Preset, ...] = (
    Preset(
        "anthropic",
        "Anthropic (Claude)",
        ProviderKind.ANTHROPIC,
        "https://api.anthropic.com/v1",
        ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"),
        hint="Strongest tool use. Billed per token.",
    ),
    Preset(
        "openai",
        "OpenAI",
        ProviderKind.OPENAI_COMPATIBLE,
        "https://api.openai.com/v1",
        ("gpt-4o", "gpt-4o-mini", "o4-mini"),
    ),
    Preset(
        "azure_openai",
        "Azure OpenAI",
        ProviderKind.OPENAI_COMPATIBLE,
        "https://<resource>.openai.azure.com/openai/deployments/<deployment>",
        (),
        hint=(
            "Base URL must include the deployment path. Add query parameter "
            "api-version (e.g. 2024-10-21) under Advanced."
        ),
    ),
    Preset(
        "google",
        "Google Gemini",
        ProviderKind.GOOGLE,
        "https://generativelanguage.googleapis.com/v1beta",
        ("gemini-2.0-flash", "gemini-1.5-pro"),
    ),
    Preset(
        "ollama",
        "Ollama (local)",
        ProviderKind.OPENAI_COMPATIBLE,
        "http://localhost:11434/v1",
        ("llama3.1:8b", "qwen2.5:14b", "mistral-nemo"),
        needs_key=False,
        hint="Zero cost, fully offline. Pick a model that supports tool calling.",
    ),
    Preset(
        "lmstudio",
        "LM Studio (local)",
        ProviderKind.OPENAI_COMPATIBLE,
        "http://localhost:1234/v1",
        (),
        needs_key=False,
    ),
    Preset(
        "vllm",
        "vLLM / self-hosted",
        ProviderKind.OPENAI_COMPATIBLE,
        "http://localhost:8000/v1",
        (),
        needs_key=False,
    ),
    Preset(
        "openrouter",
        "OpenRouter",
        ProviderKind.OPENAI_COMPATIBLE,
        "https://openrouter.ai/api/v1",
        ("anthropic/claude-sonnet-4.5", "google/gemini-2.0-flash-001"),
        hint="One key, many models - convenient for the model-comparison study.",
    ),
    Preset(
        "groq",
        "Groq",
        ProviderKind.OPENAI_COMPATIBLE,
        "https://api.groq.com/openai/v1",
        ("llama-3.3-70b-versatile",),
    ),
    Preset(
        "together",
        "Together AI",
        ProviderKind.OPENAI_COMPATIBLE,
        "https://api.together.xyz/v1",
        (),
    ),
    Preset(
        "deepseek",
        "DeepSeek",
        ProviderKind.OPENAI_COMPATIBLE,
        "https://api.deepseek.com/v1",
        ("deepseek-chat",),
    ),
    Preset(
        "custom",
        "Custom OpenAI-compatible endpoint",
        ProviderKind.OPENAI_COMPATIBLE,
        "",
        (),
        needs_key=False,
        hint="Any server exposing POST /chat/completions.",
    ),
)

PRESETS_BY_KEY = {p.key: p for p in PRESETS}


def build_provider(config: ProviderConfig) -> LLMProvider:
    impl = _IMPLEMENTATIONS.get(config.kind)
    if impl is None:
        raise ValueError(f"No implementation registered for provider kind {config.kind!r}")
    return impl(config)


class ProviderRegistry:
    """Holds live provider clients, keyed by config id.

    Clients are cached because each owns an HTTP connection pool; rebuilding
    one per request would defeat keep-alive on every call.
    """

    def __init__(self) -> None:
        self._clients: dict[str, LLMProvider] = {}
        self._configs: dict[str, ProviderConfig] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, config: ProviderConfig) -> None:
        async with self._lock:
            existing = self._clients.pop(config.id, None)
            if existing is not None:
                await existing.close()
            self._configs[config.id] = config
            if config.enabled:
                self._clients[config.id] = build_provider(config)

    async def remove(self, provider_id: str) -> None:
        async with self._lock:
            self._configs.pop(provider_id, None)
            client = self._clients.pop(provider_id, None)
            if client is not None:
                await client.close()

    def get(self, provider_id: str) -> LLMProvider:
        client = self._clients.get(provider_id)
        if client is None:
            known = ", ".join(sorted(self._configs)) or "none configured"
            raise KeyError(
                f"Provider {provider_id!r} is not available (known: {known}). "
                "Enable it in Settings."
            )
        return client

    def config(self, provider_id: str) -> ProviderConfig | None:
        return self._configs.get(provider_id)

    def list_configs(self) -> list[ProviderConfig]:
        return list(self._configs.values())

    async def test(self, provider_id: str) -> ConnectionTest:
        cfg = self._configs.get(provider_id)
        if cfg is None:
            return ConnectionTest(ok=False, error=f"Unknown provider {provider_id!r}")
        # Test even a disabled provider, so the user can verify before enabling.
        client = self._clients.get(provider_id) or build_provider(cfg)
        try:
            return await client.test_connection()
        finally:
            if provider_id not in self._clients:
                await client.close()

    async def close_all(self) -> None:
        async with self._lock:
            for client in self._clients.values():
                await client.close()
            self._clients.clear()
