"""The LLM reasoning loop, against a scripted fake model.

No network, no API key, no quota. The fake plays back a fixed sequence of
model turns, so each test pins down exactly what the harness does with a given
model behaviour - including the behaviours we must not trust: invented
citations, actions outside the catalogue, a model that rates its own proposal
as low-risk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from opsloop.agent.reasoner import SYSTEM_PROMPT, Reasoner, ReasonerError, format_answer
from opsloop.agent.tools import ToolBox
from opsloop.domain.models import Incident, RiskLevel, Signal
from opsloop.llm import settings_store
from opsloop.llm.base import (
    ChatMessage,
    ConnectionTest,
    LLMProvider,
    LLMResponse,
    ProviderConfig,
    ProviderError,
    ToolCall,
)
from opsloop.policy.engine import PolicyEngine
from opsloop.remediate.executors import ExecutorRegistry, ExecutorSpec

from test_triage_and_agent import HEAP_LOGS, INJECTION_LOGS, FakeDocker


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class ScriptedModel(LLMProvider):
    """Plays back turns. A turn is text, a list of tool calls, or an exception."""

    def __init__(self, turns: list[Any]) -> None:
        super().__init__(ProviderConfig(id="fake", label="Fake", model="fake-model",
                                        base_url="http://fake"))
        self.turns = list(turns)
        self.seen: list[list[ChatMessage]] = []

    async def chat(self, messages: list[ChatMessage], **_: Any) -> LLMResponse:
        self.seen.append(list(messages))
        turn = self.turns.pop(0) if self.turns else "done"
        if isinstance(turn, Exception):
            raise turn
        if isinstance(turn, list):
            return LLMResponse(text="", tool_calls=turn, model="fake-model")
        return LLMResponse(text=turn, model="fake-model")

    async def close(self) -> None:
        return None


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(id=f"c_{name}_{len(args)}", name=name, arguments=args)


class FakeStore:
    def recent(self, n: int = 10) -> list[Incident]:
        return []

    def get(self, _: str) -> None:
        return None


def registry() -> ExecutorRegistry:
    reg = ExecutorRegistry()

    async def ok(_: dict[str, Any]) -> tuple[bool, str, str]:
        return True, "ok", ""

    reg.register(ExecutorSpec(id="docker.start", description="start", risk=RiskLevel.LOW,
                              handler=ok, required_params=("container",)))
    reg.register(ExecutorSpec(id="docker.stop", description="stop", risk=RiskLevel.MEDIUM,
                              handler=ok, required_params=("container",)))
    reg.register(ExecutorSpec(id="demo.apply_code_fix", description="fix", risk=RiskLevel.MEDIUM,
                              handler=ok, required_params=("service",)))
    reg.register(ExecutorSpec(id="demo.revert_code_fix", description="revert", risk=RiskLevel.MEDIUM,
                              handler=ok, required_params=("service", "mode")))
    return reg


def reasoner(turns: list[Any], docker: FakeDocker | None = None) -> tuple[Reasoner, ScriptedModel]:
    model = ScriptedModel(turns)
    tb = ToolBox(docker=docker or FakeDocker(), store=FakeStore(), executors=registry())
    return Reasoner(model, tb, policy=PolicyEngine(), max_rounds=4), model


def incident() -> Incident:
    inc = Incident(title="orders-api down", service="orders-api")
    inc.signals.append(Signal(detector="t", title="exited", raw={"container": "demo-orders-api"}))
    return inc


def evidence_ids_in(model: ScriptedModel) -> list[str]:
    """Ids the harness actually showed the model, in tool results."""
    import re

    ids: list[str] = []
    for m in model.seen[-1]:
        if m.role == "tool":
            ids += re.findall(r"\[evidence (ev_[0-9a-f]{12})\]", m.content)
    return ids


# --------------------------------------------------------------------------
# Conversational
# --------------------------------------------------------------------------


class TestAsk:
    async def test_answer_cites_real_evidence(self) -> None:
        docker = FakeDocker()
        docker.break_with(oom=True, logs=HEAP_LOGS)
        r, model = reasoner([[call("container_logs", container="demo-orders-api")], "PLACEHOLDER"], docker)

        # Let the first round run, then script an answer citing what was shown.
        async def run() -> Any:
            orig = model.chat

            async def chat(messages: list[ChatMessage], **kw: Any) -> LLMResponse:
                if len(model.seen) == 1:
                    model.seen.append(list(messages))
                    ids = evidence_ids_in(model)
                    return LLMResponse(text=f"Heap grows every request [{ids[0]}].", model="fake-model")
                return await orig(messages, **kw)

            model.chat = chat  # type: ignore[method-assign]
            return await r.ask("why is orders-api down?")

        answer = await run()
        assert answer.cited and not answer.fabricated
        assert answer.evidence[0].id == answer.cited[0]
        assert answer.tool_calls == ['container_logs({"container": "demo-orders-api"})']

    async def test_invented_citation_is_flagged(self) -> None:
        r, _ = reasoner(["It is DNS [ev_000000000000]."])
        answer = await r.ask("what is wrong?")
        assert answer.fabricated == ["ev_000000000000"]
        text = format_answer(answer)
        assert "never collected" in text
        assert "No telemetry was consulted" in text

    async def test_telemetry_never_enters_the_system_prompt(self) -> None:
        """The AIOpsDoom rule: log text arrives as a tool result, fenced."""
        docker = FakeDocker()
        docker.break_with(logs=INJECTION_LOGS)
        r, model = reasoner([[call("container_logs", container="demo-orders-api")], "noted"], docker)
        await r.ask("anything odd in the logs?")
        final = model.seen[-1]
        system = [m for m in final if m.role == "system"]
        assert len(system) == 1 and system[0].content == SYSTEM_PROMPT
        tool = next(m for m in final if m.role == "tool")
        assert "<untrusted_telemetry>" in tool.content
        assert "WARNING: sanitiser flagged" in tool.content
        assert "[[defanged:" in tool.content
        assert "[[defanged: [[defanged:" not in tool.content  # not sanitised twice

    async def test_rate_limit_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import opsloop.agent.reasoner as mod

        async def no_sleep(_: float) -> None:
            return None

        monkeypatch.setattr(mod.asyncio, "sleep", no_sleep)
        r, model = reasoner([ProviderError("fake", "slow down", status=429), "fine"])
        answer = await r.ask("status?")
        assert answer.text == "fine"
        assert len(model.seen) == 2

    async def test_hard_provider_error_surfaces(self) -> None:
        r, _ = reasoner([ProviderError("fake", "bad key", status=401)])
        with pytest.raises(ReasonerError, match="bad key"):
            await r.ask("status?")


# --------------------------------------------------------------------------
# Structured diagnosis
# --------------------------------------------------------------------------


class _DiagScript:
    """Round 1: gather evidence. Round 2: submit, citing what was gathered."""

    def __init__(self, build: Any) -> None:
        self.build = build

    async def run(self, docker: FakeDocker | None = None) -> Any:
        docker = docker or FakeDocker()
        docker.break_with(oom=True, logs=HEAP_LOGS)
        r, model = reasoner([[call("list_services"), call("container_logs", container="demo-orders-api")]], docker)
        orig = model.chat

        async def chat(messages: list[ChatMessage], **kw: Any) -> LLMResponse:
            if len(model.seen) == 1:
                model.seen.append(list(messages))
                ids = evidence_ids_in(model)
                return LLMResponse(text="", tool_calls=[call("submit_diagnosis", **self.build(ids))],
                                   model="fake-model")
            return await orig(messages, **kw)

        model.chat = chat  # type: ignore[method-assign]
        return await r.diagnose(incident(), "demo-orders-api")


class TestDiagnose:
    async def test_grounded_diagnosis_with_catalogued_action(self) -> None:
        res = await _DiagScript(lambda ids: {
            "hypotheses": [{"statement": "OOM leak", "mechanism": "heap grows",
                            "confidence": 0.8, "evidence_ids": ids}],
            "actions": [{"intent": "fix", "executor": "demo.apply_code_fix",
                         "params": {"service": "orders-api"},
                         "rollback_executor": "demo.revert_code_fix",
                         "rollback_params": {"service": "orders-api", "mode": "memleak"},
                         "evidence_ids": ids[:1]}],
        }).run()
        assert not res.diagnosis.abstained
        assert res.diagnosis.leading.statement == "OOM leak"
        assert res.diagnosis.tool_call_count == 2
        assert len(res.actions) == 1 and res.rejected_actions == []

    async def test_risk_comes_from_the_registry_not_the_model(self) -> None:
        res = await _DiagScript(lambda ids: {
            "hypotheses": [{"statement": "x", "mechanism": "y", "confidence": 0.8, "evidence_ids": ids}],
            "actions": [{"intent": "fix", "executor": "demo.apply_code_fix",
                         "params": {"service": "orders-api"}, "risk": "safe",
                         "rollback_executor": "demo.revert_code_fix",
                         "rollback_params": {"service": "orders-api", "mode": "memleak"},
                         "evidence_ids": ids}],
        }).run()
        assert res.actions[0].risk is RiskLevel.MEDIUM

    @pytest.mark.parametrize(
        "action,why",
        [
            ({"executor": "bash.run", "params": {"cmd": "rm -rf /"}, "rollback_executor": "noop",
              "rollback_params": {"reason": "x"}}, "Unknown executor"),
            ({"executor": "db.drop", "params": {}, "rollback_executor": "noop",
              "rollback_params": {"reason": "x"}}, "prohibited"),
            ({"executor": "docker.start", "params": {}, "rollback_executor": "docker.stop",
              "rollback_params": {"container": "c"}}, "missing params"),
            ({"executor": "docker.start", "params": {"container": "c"},
              "rollback_executor": "docker.stop", "rollback_params": {}}, "rollback docker.stop missing"),
        ],
    )
    async def test_invalid_actions_are_refused_with_a_reason(self, action: dict, why: str) -> None:
        res = await _DiagScript(lambda ids: {
            "hypotheses": [{"statement": "x", "mechanism": "y", "confidence": 0.8, "evidence_ids": ids}],
            "actions": [{"intent": "do it", "evidence_ids": ids, **action}],
        }).run()
        assert res.actions == []
        assert why in res.rejected_actions[0]

    async def test_action_citing_nothing_real_is_refused(self) -> None:
        res = await _DiagScript(lambda ids: {
            "hypotheses": [{"statement": "x", "mechanism": "y", "confidence": 0.8, "evidence_ids": ids}],
            "actions": [{"intent": "fix", "executor": "docker.start", "params": {"container": "c"},
                         "rollback_executor": "docker.stop", "rollback_params": {"container": "c"},
                         "evidence_ids": ["ev_ffffffffffff"]}],
        }).run()
        assert res.actions == [] and "cites no evidence" in res.rejected_actions[0]

    async def test_fabricated_citations_force_abstention(self) -> None:
        res = await _DiagScript(lambda ids: {
            "hypotheses": [{"statement": "DNS", "mechanism": "?", "confidence": 0.99,
                            "evidence_ids": ["ev_aaaaaaaaaaaa"]}],
        }).run()
        assert res.diagnosis.abstained
        assert res.actions == []

    async def test_no_tool_calls_forces_abstention(self) -> None:
        """Cloud-OpsBench's 32%: a confident diagnosis with no investigation."""
        r, _ = reasoner([[call("submit_diagnosis", hypotheses=[
            {"statement": "database down", "mechanism": "guess", "confidence": 0.95, "evidence_ids": []}])]])
        res = await r.diagnose(incident(), "demo-orders-api")
        assert res.diagnosis.abstained
        assert "without collecting" in res.diagnosis.abstain_reason

    async def test_model_that_never_submits_abstains(self) -> None:
        r, _ = reasoner(["I think it is fine."])
        res = await r.diagnose(incident(), "demo-orders-api")
        assert res.diagnosis.abstained
        assert "did not submit" in res.diagnosis.abstain_reason


class TestToolSafety:
    async def test_unknown_tool_is_an_error_not_a_crash(self) -> None:
        tb = ToolBox(docker=FakeDocker(), store=FakeStore())
        result = await tb.call("run_shell", {"cmd": "id"})
        assert result.text.startswith("ERROR: unknown tool")

    def test_no_url_taking_tool_exists(self) -> None:
        """HTTP tools take an allowlisted service name - no request forgery."""
        tb = ToolBox(docker=FakeDocker(), store=FakeStore(),
                     service_endpoints={"orders-api": "http://localhost:18081"})
        for spec in tb.specs(include_diagnosis=True):
            props = spec.parameters.get("properties", {})
            assert "url" not in props, spec.name
        health = next(s for s in tb.specs() if s.name == "service_health")
        assert health.parameters["properties"]["service"]["enum"] == ["orders-api"]

    async def test_unlisted_service_is_refused(self) -> None:
        tb = ToolBox(docker=FakeDocker(), store=FakeStore(),
                     service_endpoints={"orders-api": "http://localhost:18081"})
        result = await tb.call("service_health", {"service": "169.254.169.254"})
        assert "unknown service" in result.text

    def test_no_tool_changes_anything(self) -> None:
        """Every tool is a read. The only write path is a proposal a human approves."""
        tb = ToolBox(docker=FakeDocker(), store=FakeStore())
        names = {s.name for s in tb.specs(include_diagnosis=True)}
        assert names <= {"list_services", "container_logs", "container_stats", "list_incidents",
                         "incident_detail", "service_health", "service_metrics", "host_discover",
                         "host_disk_usage", "host_logs", "submit_diagnosis"}


# --------------------------------------------------------------------------
# Settings menu
# --------------------------------------------------------------------------


class TestSettingsStore:
    def test_preset_fills_url_and_model(self) -> None:
        c = settings_store.config_from_values(provider="groq", api_key="k")
        assert c.base_url == "https://api.groq.com/openai/v1"
        assert c.model == "openai/gpt-oss-120b"

    def test_any_custom_openai_compatible_endpoint(self) -> None:
        c = settings_store.config_from_values(
            provider="my-vllm", base_url="http://10.0.0.5:8000/v1/", model="qwen3")
        assert c.base_url == "http://10.0.0.5:8000/v1" and c.label == "my-vllm"

    def test_placeholder_url_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="placeholder"):
            settings_store.config_from_values(provider="azure_openai", model="m")

    def test_unknown_provider_without_url_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="No base URL"):
            settings_store.config_from_values(provider="mystery", model="m")

    def test_file_overrides_env_and_roundtrips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPSLOOP_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPSLOOP_LLM_MODEL", "gpt-4o")
        path = tmp_path / "s.json"
        assert settings_store.load(path).model == "gpt-4o"  # env when no file
        saved = settings_store.config_from_values(provider="groq", api_key="secret", model="qwen/qwen3.8-27b")
        settings_store.save(path, saved)
        loaded = settings_store.load(path)
        assert loaded.id == "groq" and loaded.model == "qwen/qwen3.8-27b"
        assert loaded.api_key.get_secret_value() == "secret"

    def test_corrupt_file_falls_back_to_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPSLOOP_LLM_PROVIDER", "groq")
        path = tmp_path / "s.json"
        path.write_text("{not json", encoding="utf-8")
        assert settings_store.load(path).id == "groq"


class TestConfigureFromMenu:
    def _agent(self, tmp_path: Path) -> Any:
        from test_triage_and_agent import make_agent

        agent, *_ = make_agent()
        agent.settings_path = tmp_path / "s.json"
        return agent

    def _cmd(self, admin: bool = True, **args: str) -> Any:
        from opsloop.chat.base import ChatCommand, ChatUser

        return ChatCommand(name="llm_set", args=args, surface="test", channel_id="c",
                           user=ChatUser(id="1", display_name="op", is_admin=admin))

    async def test_non_admin_cannot_change_provider(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        reply = await agent.handle_command(self._cmd(admin=False, provider="groq"))
        assert "Only administrators" in reply
        assert agent.reasoner is None

    async def test_failed_connection_test_keeps_the_working_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import opsloop.app as app_mod

        agent = self._agent(tmp_path)
        good = settings_store.config_from_values(provider="groq", api_key="good")
        agent._install_llm(good)

        class Broken(ScriptedModel):
            async def test_connection(self) -> ConnectionTest:
                return ConnectionTest(ok=False, error="401 invalid key")

        monkeypatch.setattr(app_mod, "build_provider", lambda c: Broken([]))
        reply = await agent.handle_command(self._cmd(provider="openai", api_key="typo", model="gpt-4o"))
        assert "connection test failed" in reply.title
        assert agent.llm_config.id == "groq"  # unchanged
        assert not agent.settings_path.exists()  # nothing saved

    async def test_successful_change_is_installed_and_saved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import opsloop.app as app_mod

        agent = self._agent(tmp_path)

        class Works(ScriptedModel):
            async def test_connection(self) -> ConnectionTest:
                return ConnectionTest(ok=True, latency_ms=12, detail="ok")

        monkeypatch.setattr(app_mod, "build_provider", lambda c: Works([]))
        reply = await agent.handle_command(
            self._cmd(provider="groq", api_key="gsk_test_1234", model="openai/gpt-oss-20b"))
        assert reply.title == "LLM provider updated"
        assert agent.reasoner is not None
        saved = json.loads(agent.settings_path.read_text())
        assert saved["llm"]["model"] == "openai/gpt-oss-20b"

    async def test_blank_key_keeps_the_current_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import opsloop.app as app_mod

        agent = self._agent(tmp_path)
        agent._install_llm(settings_store.config_from_values(provider="groq", api_key="gsk_keep_me"))

        class Works(ScriptedModel):
            async def test_connection(self) -> ConnectionTest:
                return ConnectionTest(ok=True)

        monkeypatch.setattr(app_mod, "build_provider", lambda c: Works([]))
        await agent.handle_command(self._cmd(provider="groq", api_key="", model="qwen/qwen3.8-27b"))
        assert agent.llm_config.api_key.get_secret_value() == "gsk_keep_me"
        assert agent.llm_config.model == "qwen/qwen3.8-27b"

    def test_menu_defaults_never_contain_the_key(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._install_llm(settings_store.config_from_values(provider="groq", api_key="gsk_secret"))
        defaults = agent._form_defaults("llm")
        assert "gsk_secret" not in json.dumps(defaults)
        assert "gsk_secret" not in agent._llm_summary()
