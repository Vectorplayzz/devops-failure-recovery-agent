"""Provider-neutral LLM contract.

Everything above this module speaks only in these types. Swapping Claude for a
local Ollama model, or for an OpenAI-compatible endpoint behind a custom base
URL, must not require a change anywhere else in the codebase - that is what
makes comparative evaluation (same incidents, several models, honest numbers)
cheap to produce.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator

Role = Literal["system", "user", "assistant", "tool"]


# --------------------------------------------------------------------------
# Wire types
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set when role == "tool"
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> ChatMessage:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str, name: str | None = None) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


class ToolSpec(BaseModel):
    """A tool offered to the model. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider_id: str = ""
    finish_reason: str = ""
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# --------------------------------------------------------------------------
# Configuration - what the settings UI edits
# --------------------------------------------------------------------------


class ProviderKind(str, Enum):
    """Wire dialect, not vendor.

    OPENAI_COMPATIBLE covers OpenAI, Azure OpenAI, Ollama, LM Studio, vLLM,
    llama.cpp, OpenRouter, Groq, Together, DeepSeek, Mistral and anything else
    exposing /chat/completions - which is the point of the custom base URL.
    """

    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ProviderConfig(BaseModel):
    """One configured endpoint. Several may coexist; one is active."""

    id: str
    label: str = ""
    kind: ProviderKind = ProviderKind.OPENAI_COMPATIBLE
    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = SecretStr("")
    model: str = ""

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: int = Field(default=120, gt=0)
    supports_tools: bool = True

    # Azure OpenAI and some gateways need these.
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_query: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    enabled: bool = True
    verify_ssl: bool = True
    notes: str = ""

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def display(self) -> str:
        return self.label or f"{self.kind.value}:{self.model}"

    def redacted(self) -> dict[str, Any]:
        """Safe to render in the settings UI or log."""
        d = self.model_dump(mode="json")
        key = self.api_key.get_secret_value()
        d["api_key"] = f"...{key[-4:]}" if len(key) >= 4 else ("set" if key else "")
        d["api_key_set"] = bool(key)
        return d


class ProviderError(RuntimeError):
    """Transport or protocol failure talking to a provider."""

    def __init__(self, provider_id: str, message: str, *, status: int | None = None):
        self.provider_id = provider_id
        self.status = status
        super().__init__(f"[{provider_id}] {message}")


@dataclass
class ConnectionTest:
    ok: bool
    latency_ms: int = 0
    detail: str = ""
    model_echo: str = ""
    error: str = ""


# --------------------------------------------------------------------------
# Provider interface
# --------------------------------------------------------------------------


class LLMProvider(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def id(self) -> str:
        return self.config.id

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Single turn. Tool-call loops live in the agent, not here."""

    @abstractmethod
    async def close(self) -> None: ...

    async def test_connection(self) -> ConnectionTest:
        """Cheapest possible round trip, for the settings UI Test button."""
        started = time.perf_counter()
        try:
            resp = await self.chat(
                [ChatMessage.user("Reply with the single word: ok")],
                max_tokens=16,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
            return ConnectionTest(
                ok=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )
        ms = int((time.perf_counter() - started) * 1000)
        return ConnectionTest(
            ok=True,
            latency_ms=ms,
            detail=resp.text.strip()[:120],
            model_echo=resp.model,
        )
