"""Where the active LLM provider comes from, and where the settings menu saves it.

Precedence, highest first:

  1. `opsloop-settings.json`  - written by the /llm menu at runtime
  2. environment / `.env`      - the bootstrap configuration

The file wins so a change made from the chat menu survives a restart without
anyone editing `.env`. It holds an API key, so it is gitignored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from .base import ProviderConfig, ProviderKind
from .registry import PRESETS_BY_KEY

_KIND_BY_NAME = {"anthropic": ProviderKind.ANTHROPIC, "google": ProviderKind.GOOGLE}


def config_from_values(
    *, provider: str, base_url: str = "", api_key: str = "", model: str = ""
) -> ProviderConfig:
    """Build a config from loose user input, filling gaps from presets.

    `provider` may be a preset key (groq, openai, ollama, ...) or any label.
    An unknown label with a base URL is simply a custom OpenAI-compatible
    endpoint - the settings menu is not limited to the presets.
    """
    key = provider.strip().lower()
    preset = PRESETS_BY_KEY.get(key)
    kind = preset.kind if preset else _KIND_BY_NAME.get(key, ProviderKind.OPENAI_COMPATIBLE)
    url = (base_url or (preset.base_url if preset else "")).strip()
    if not url:
        raise ValueError(
            f"No base URL for provider {provider!r}. Give a base URL, or use a preset: "
            + ", ".join(sorted(PRESETS_BY_KEY))
        )
    if "<" in url:
        raise ValueError(f"Base URL still contains a placeholder: {url}")
    chosen = model.strip() or (preset.suggested_models[0] if preset and preset.suggested_models else "")
    if not chosen:
        raise ValueError("A model name is required for this provider.")
    return ProviderConfig(
        id=key or "custom",
        label=preset.label if preset else provider.strip(),
        kind=kind,
        base_url=url,
        api_key=SecretStr(api_key.strip()),
        model=chosen,
    )


def load(path: Path) -> ProviderConfig | None:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8")).get("llm") or {}
            if data.get("provider"):
                return config_from_values(
                    provider=data["provider"],
                    base_url=data.get("base_url", ""),
                    api_key=data.get("api_key", ""),
                    model=data.get("model", ""),
                )
        except (ValueError, OSError, json.JSONDecodeError):
            pass  # a corrupt settings file must not stop the agent; fall back to env

    provider = os.environ.get("OPSLOOP_LLM_PROVIDER", "").strip()
    if not provider:
        return None
    return config_from_values(
        provider=provider,
        base_url=os.environ.get("OPSLOOP_LLM_BASE_URL", ""),
        api_key=os.environ.get("OPSLOOP_LLM_API_KEY", ""),
        model=os.environ.get("OPSLOOP_LLM_MODEL", ""),
    )


def save(path: Path, config: ProviderConfig) -> None:
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    existing["llm"] = {
        "provider": config.id,
        "base_url": config.base_url,
        "api_key": config.api_key.get_secret_value(),
        "model": config.model,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)  # best effort; a no-op on Windows
    except OSError:
        pass
    tmp.replace(path)  # atomic: a crash mid-write cannot leave half a key
