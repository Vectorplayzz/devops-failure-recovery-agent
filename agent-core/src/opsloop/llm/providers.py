"""Concrete providers.

Three wire dialects cover essentially the whole market:

  openai_compatible -> OpenAI, Azure OpenAI, Ollama, LM Studio, vLLM,
                       llama.cpp, OpenRouter, Groq, Together, DeepSeek,
                       Mistral, and any self-hosted /chat/completions endpoint
  anthropic         -> Claude (/v1/messages)
  google            -> Gemini (generateContent)

Adding a fourth means implementing `LLMProvider` and registering it in
`registry.py`; nothing else in the codebase changes.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from .base import (
    ChatMessage,
    ConnectionTest,
    LLMProvider,
    LLMResponse,
    ProviderConfig,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)


def _client(config: ProviderConfig, headers: dict[str, str]) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=config.base_url,
        headers={**headers, **config.extra_headers},
        timeout=httpx.Timeout(config.timeout_seconds),
        verify=config.verify_ssl,
        params=config.extra_query or None,
    )


async def _post_json(
    client: httpx.AsyncClient, provider_id: str, path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    try:
        resp = await client.post(path, json=payload)
    except httpx.RequestError as exc:
        raise ProviderError(provider_id, f"cannot reach endpoint: {exc}") from exc

    if resp.status_code >= 400:
        body = resp.text[:600]
        raise ProviderError(
            provider_id,
            f"HTTP {resp.status_code}: {body}",
            status=resp.status_code,
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise ProviderError(
            provider_id, f"non-JSON response: {resp.text[:300]}"
        ) from exc


# --------------------------------------------------------------------------
# OpenAI-compatible
# --------------------------------------------------------------------------


class OpenAICompatibleProvider(LLMProvider):
    """Anything that speaks POST {base_url}/chat/completions."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        headers = {"Content-Type": "application/json"}
        key = config.api_key.get_secret_value()
        if key:
            # Local runtimes (Ollama, LM Studio) ignore this; harmless to send.
            headers["Authorization"] = f"Bearer {key}"
            headers.setdefault("api-key", key)  # Azure OpenAI dialect
        self._http = _client(config, headers)

    async def close(self) -> None:
        await self._http.aclose()

    @staticmethod
    def _encode_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": m.content,
                    }
                )
                continue
            entry: dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
                # OpenAI rejects a null content alongside tool_calls on some
                # gateways and requires it on others; empty string is accepted
                # by both.
                entry["content"] = m.content or ""
            out.append(entry)
        return out

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        cfg = self.config
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": self._encode_messages(messages),
            "temperature": cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or cfg.max_tokens,
            **cfg.extra_body,
        }
        if tools and cfg.supports_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format

        started = time.perf_counter()
        data = await _post_json(self._http, cfg.id, "/chat/completions", payload)
        latency = int((time.perf_counter() - started) * 1000)

        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(cfg.id, f"response contained no choices: {data}")
        message = choices[0].get("message") or {}

        calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                # A malformed tool call is a model error, not a crash. Surface
                # it to the agent loop so it can re-prompt.
                args = {"__malformed_arguments__": raw_args}
            calls.append(
                ToolCall(id=tc.get("id") or uuid.uuid4().hex, name=fn.get("name", ""), arguments=args)
            )

        u = data.get("usage") or {}
        return LLMResponse(
            text=message.get("content") or "",
            tool_calls=calls,
            usage=Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            ),
            model=data.get("model", cfg.model),
            provider_id=cfg.id,
            finish_reason=choices[0].get("finish_reason", ""),
            latency_ms=latency,
            raw=data,
        )

    async def test_connection(self) -> ConnectionTest:
        """Prefer /models - it is free and proves auth without burning tokens."""
        started = time.perf_counter()
        try:
            resp = await self._http.get("/models")
            if resp.status_code < 400:
                ms = int((time.perf_counter() - started) * 1000)
                try:
                    ids = [m.get("id", "") for m in (resp.json().get("data") or [])]
                except ValueError:
                    ids = []
                found = self.config.model in ids if ids else None
                detail = f"{len(ids)} model(s) available"
                if found is False:
                    detail += f"; WARNING: '{self.config.model}' not in the list"
                return ConnectionTest(
                    ok=True, latency_ms=ms, detail=detail, model_echo=self.config.model
                )
        except httpx.RequestError:
            pass  # endpoint may not implement /models; fall through to a real call
        return await super().test_connection()


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Claude via POST {base_url}/messages."""

    API_VERSION = "2023-06-01"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.API_VERSION,
        }
        key = config.api_key.get_secret_value()
        if key:
            headers["x-api-key"] = key
        self._http = _client(config, headers)

    async def close(self) -> None:
        await self._http.aclose()

    @staticmethod
    def _encode(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
        """Anthropic takes `system` out of band and uses content blocks."""
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue

            if m.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
                # Consecutive tool results belong in one user turn.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
                continue

            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                    for tc in m.tool_calls
                )
                out.append({"role": "assistant", "content": blocks})
                continue

            out.append({"role": m.role, "content": m.content or ""})

        return "\n\n".join(p for p in system_parts if p), out

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        cfg = self.config
        system, encoded = self._encode(messages)
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": encoded,
            "max_tokens": max_tokens or cfg.max_tokens,
            "temperature": cfg.temperature if temperature is None else temperature,
            **cfg.extra_body,
        }
        if system:
            payload["system"] = system
        if tools and cfg.supports_tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        started = time.perf_counter()
        data = await _post_json(self._http, cfg.id, "/messages", payload)
        latency = int((time.perf_counter() - started) * 1000)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.get("id", uuid.uuid4().hex),
                        name=block.get("name", ""),
                        arguments=block.get("input") or {},
                    )
                )

        u = data.get("usage") or {}
        pt, ct = u.get("input_tokens", 0), u.get("output_tokens", 0)
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
            model=data.get("model", cfg.model),
            provider_id=cfg.id,
            finish_reason=data.get("stop_reason", ""),
            latency_ms=latency,
            raw=data,
        )


# --------------------------------------------------------------------------
# Google Gemini
# --------------------------------------------------------------------------


class GoogleProvider(LLMProvider):
    """Gemini via POST {base_url}/models/{model}:generateContent."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        headers = {"Content-Type": "application/json"}
        key = config.api_key.get_secret_value()
        if key:
            headers["x-goog-api-key"] = key
        self._http = _client(config, headers)

    async def close(self) -> None:
        await self._http.aclose()

    @staticmethod
    def _encode(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            if m.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": m.name or "tool",
                                    "response": {"result": m.content},
                                }
                            }
                        ],
                    }
                )
                continue
            role = "model" if m.role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"text": m.content})
            for tc in m.tool_calls:
                parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
            contents.append({"role": role, "parts": parts or [{"text": ""}]})
        return "\n\n".join(p for p in system_parts if p), contents

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        cfg = self.config
        system, contents = self._encode(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": cfg.temperature if temperature is None else temperature,
                "maxOutputTokens": max_tokens or cfg.max_tokens,
            },
            **cfg.extra_body,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools and cfg.supports_tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in tools
                    ]
                }
            ]

        started = time.perf_counter()
        data = await _post_json(
            self._http, cfg.id, f"/models/{cfg.model}:generateContent", payload
        )
        latency = int((time.perf_counter() - started) * 1000)

        candidates = data.get("candidates") or []
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        finish = ""
        if candidates:
            finish = candidates[0].get("finishReason", "")
            for part in (candidates[0].get("content") or {}).get("parts") or []:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    calls.append(
                        ToolCall(
                            id=uuid.uuid4().hex,
                            name=fc.get("name", ""),
                            arguments=fc.get("args") or {},
                        )
                    )

        u = data.get("usageMetadata") or {}
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=Usage(
                prompt_tokens=u.get("promptTokenCount", 0),
                completion_tokens=u.get("candidatesTokenCount", 0),
                total_tokens=u.get("totalTokenCount", 0),
            ),
            model=cfg.model,
            provider_id=cfg.id,
            finish_reason=finish,
            latency_ms=latency,
            raw=data,
        )
