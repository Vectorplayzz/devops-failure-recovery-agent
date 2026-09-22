"""OpsLoop runtime - where the pieces meet.

    python -m opsloop            # Discord if OPSLOOP_DISCORD_TOKEN is set, else console

The loop this file runs:

    monitor ──► open incident ──► triage (grounded) ──► approval cards
                                                            │  human taps Approve
                                                            ▼
            incident card ◄── verify / auto-rollback ◄── execute ◄── policy re-check

Everything with an opinion lives elsewhere: grounding in `domain/`, the gate in
`policy/`, the loop in `verify/`, rendering in `chat/`. This module sequences
them and owns the only long-running tasks.

What it deliberately does NOT know
----------------------------------
The demo scenarios' ground truth. `/inject` needs to know which fault mode to
set on which service, and that is all it is given here. The answer key -
true root cause, correct fix, decoy fix - stays in the fault injector, so the
agent is diagnosing from telemetry and nothing else.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx

from .chat.base import (
    ApprovalEvent,
    ApprovalOutcome,
    Card,
    ChatCommand,
    ChatQuestion,
    ChatSurface,
    Field,
    MessageRef,
    Tone,
    approval_card,
    diagnosis_card,
    incident_card,
    truncate,
    verification_card,
)
from .domain.models import (
    ActionSpec,
    ApprovalDecision,
    Incident,
    IncidentState,
    ProposedAction,
    Signal,
    _now,
)
from .policy.engine import AutonomyMode, PolicyConfig, PolicyEngine
from .remediate.executors import (
    ExecutorRegistry,
    register_demo_executors,
    register_docker_executors,
)
from .store.incidents import IncidentStore
from .telemetry.base import AdapterRegistry, LogQuery, TimeRange, utcnow
from .telemetry.docker_adapter import DockerAdapter
from .triage.rules import MODEL_ID, RuleTriage, classify_log_burst, service_of
from .verify.loop import Verifier, plan_for_container_service

log = logging.getLogger("opsloop")

# Fault modes only - no ground truth. See the module docstring.
DEMO_SCENARIOS: dict[str, dict[str, str]] = {
    "oom": {"orders-api": "memleak"},
    "db-pool": {"orders-api": "pool_exhaust"},
    "bad-deploy": {"orders-api": "error500"},
    "dep-down": {"payments-api": "down"},
    "dep-slow": {"payments-api": "slow"},
    "flaky-dep": {"payments-api": "flaky"},
    "log-injection": {"orders-api": "log_injection"},
}
DEMO_ENDPOINTS = {"orders-api": "http://localhost:18081", "payments-api": "http://localhost:18082"}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader; real environment variables always win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    discord_token: str = ""
    discord_guild_id: str = ""
    discord_channel_id: str = ""
    discord_admin_role: str = ""
    discord_message_chat: bool = True
    demo_enabled: bool = False
    monitor_interval_seconds: int = 15
    error_burst_threshold: int = 5
    verify_window_seconds: int = 70
    verify_poll_seconds: int = 7
    verify_required_passes: int = 3
    autonomy: AutonomyMode = AutonomyMode.APPROVE_THEN_EXECUTE
    llm_configured: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        return cls(
            discord_token=e("OPSLOOP_DISCORD_TOKEN", ""),
            discord_guild_id=e("OPSLOOP_DISCORD_GUILD_ID", ""),
            discord_channel_id=e("OPSLOOP_DISCORD_CHANNEL_ID", ""),
            discord_admin_role=e("OPSLOOP_DISCORD_ADMIN_ROLE", ""),
            discord_message_chat=_flag("OPSLOOP_DISCORD_MESSAGE_CHAT", True),
            demo_enabled=_flag("OPSLOOP_DEMO_ENABLED", False),
            monitor_interval_seconds=int(e("OPSLOOP_MONITOR_INTERVAL", "15")),
            error_burst_threshold=int(e("OPSLOOP_ERROR_BURST", "5")),
            verify_window_seconds=int(e("OPSLOOP_VERIFY_WINDOW", "70")),
            verify_poll_seconds=int(e("OPSLOOP_VERIFY_POLL", "7")),
            verify_required_passes=int(e("OPSLOOP_VERIFY_PASSES", "3")),
            llm_configured=bool(e("OPSLOOP_LLM_PROVIDER", "")),
        )


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


@dataclass
class _Posted:
    ref: MessageRef
    incident_id: str
    action: ProposedAction


class Agent:
    def __init__(
        self,
        settings: Settings,
        surface: ChatSurface,
        *,
        docker_adapter: Any = None,
        executors: ExecutorRegistry | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.settings = settings
        self.surface = surface
        self.store = IncidentStore()

        self.docker = docker_adapter or DockerAdapter("docker")
        self.adapters = AdapterRegistry()
        self.adapters.add(self.docker)

        if executors is None:
            executors = ExecutorRegistry()
            register_docker_executors(executors)
            if settings.demo_enabled:
                register_demo_executors(executors)
        self.executors = executors

        self.policy = PolicyEngine(
            PolicyConfig(autonomy=settings.autonomy, max_actions_per_incident=6)
        )
        self.verifier = verifier or Verifier(self.adapters, self.executors)
        self.triage = RuleTriage(self.docker, demo_executors=settings.demo_enabled)

        self._busy: set[str] = set()  # incidents with a remediation in flight
        self._approval_cards: dict[str, _Posted] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._monitor: asyncio.Task[Any] | None = None

        surface.on_command(self.handle_command)
        surface.on_approval(self.handle_approval)
        surface.on_question(self.handle_question)

    # -- lifecycle ---------------------------------------------------------

    async def start(self, *, monitor: bool = True) -> None:
        await self.surface.start()
        await self.surface.post(await self._status_card(title="OpsLoop is online"))
        if monitor:
            self._monitor = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        if self._monitor is not None:
            self._monitor.cancel()
        for t in list(self._tasks):
            t.cancel()
        await self.surface.stop()
        await self.verifier.close()
        await self.adapters.close_all()

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # -- monitoring ----------------------------------------------------------

    async def _monitor_loop(self) -> None:
        while True:
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the monitor must outlive a bad scan
                log.exception("monitor scan failed")
            await asyncio.sleep(self.settings.monitor_interval_seconds)

    async def scan_once(self) -> list[Incident]:
        """One pass over every monitored service. Returns incidents opened."""
        opened: list[Incident] = []
        for state in await self.docker.inventory():
            if state.labels.get("opsloop.role") != "service":
                continue
            service = service_of(state.name)
            if self.store.open_for(service) is not None:
                continue  # one live incident per service

            reasons: list[str] = []
            if state.is_suspicious:
                reasons.append(state.summarise())

            window = TimeRange.last(1)
            quiet = self.store.quiet_since(service)
            if quiet is not None and quiet > window.start:
                window = TimeRange(start=quiet, end=utcnow())
            burst = classify_log_burst(
                await self.docker.fetch_logs(
                    LogQuery(target=state.name, time_range=window, limit=200)
                )
            )
            if burst["errors"] >= self.settings.error_burst_threshold:
                reasons.append(f"{burst['errors']} error lines in the last minute")
            if burst["tainted"]:
                reasons.append("adversarial content detected in logs")

            if reasons:
                opened.append(await self.open_incident(state.name, "; ".join(reasons)))
        return opened

    async def open_incident(self, container: str, reason: str) -> Incident:
        service = service_of(container)
        incident = Incident(
            title=f"{service}: {truncate(reason, 180)}",
            service=service,
            environment="production",
            state=IncidentState.TRIAGING,
        )
        incident.signals.append(
            Signal(detector="monitor", title=reason, service=service, raw={"container": container})
        )
        incident.log("agent", "detected", reason)
        self.store.add(incident)
        await self._triage(incident, container)
        return incident

    async def _triage(self, incident: Incident, container: str) -> None:
        result = await self.triage.investigate(incident, container)
        incident.diagnosis = result.diagnosis
        incident.title = result.title or incident.title
        incident.severity = result.severity
        incident.proposed_actions = result.actions
        incident.log("agent", "diagnosed", result.diagnosis.model_id,
                     abstained=result.diagnosis.abstained)

        if result.diagnosis.abstained or not result.actions:
            incident.state = IncidentState.ESCALATED
            incident.log(
                "agent",
                "escalated",
                "Security finding - no action proposed."
                if result.security_finding
                else (result.diagnosis.abstain_reason or "No safe action identified."),
            )
        else:
            incident.state = IncidentState.AWAITING_APPROVAL

        await self.surface.post(incident_card(incident))

        for action in result.actions:
            decision = self.policy.evaluate(action, incident=incident)
            if decision.blocked:
                await self.surface.post(
                    Card(
                        title=f"Blocked by policy: {action.intent}",
                        body=decision.explain(),
                        tone=Tone.NEUTRAL,
                        footer=incident.id,
                    )
                )
                continue
            incident.log("agent", "approval_requested", action.intent, action_id=action.id)
            ref = await self.surface.post(approval_card(incident, action, decision))
            self._approval_cards[action.id] = _Posted(ref, incident.id, action)

    # -- approvals -----------------------------------------------------------

    async def handle_approval(self, event: ApprovalEvent) -> ApprovalOutcome:
        """Decide what a click means - and whether its card should survive it.

        Refusals come in two kinds and the difference matters. TRANSIENT ones
        (another fix in flight, clicker not an approver) keep the card live,
        because the right person must still be able to press it later.
        PERMANENT ones (incident closed, action already decided, card from an
        earlier run) retire it, because it can never work again. Treating
        every refusal as permanent strands the incident with no button for the
        fix it needs - which is exactly what happened the first time this ran
        live: a refused click on the real fix deleted its only button.
        """
        keep = lambda msg: ApprovalOutcome(msg, retire_card=False)  # noqa: E731
        dead = lambda msg: ApprovalOutcome(msg, retire_card=True, label="NOT ACTIONABLE")  # noqa: E731

        incident = self.store.get(event.incident_id)
        if incident is None:
            return dead("This card is from an earlier run of the agent. Nothing was run.")
        action = incident.action(event.action_id)
        if action is None:
            return dead("Unknown action. Nothing was run.")

        # Transient: who may change production is not who can see the channel.
        if not event.user.is_admin:
            role = self.settings.discord_admin_role or "a server administrator"
            return keep(f"Only {role} can approve production changes. Nothing was run.")

        # Transient: one remediation at a time per incident.
        if incident.id in self._busy or incident.state in {
            IncidentState.REMEDIATING, IncidentState.VERIFYING,
        }:
            return keep(
                "Another remediation for this incident is being executed or verified. "
                "This card stays live - press it again once that finishes."
            )

        # Permanent.
        if incident.state in {IncidentState.RESOLVED, IncidentState.CLOSED_MANUALLY}:
            return dead(f"Incident is already {incident.state.value}. Nothing was run.")
        if incident.state is IncidentState.ESCALATED:
            return dead("Incident is escalated to a human. Nothing was run.")
        if incident.approval_for(action.id) is not None:
            return dead("This action was already decided. Nothing was run again.")

        decision = ApprovalDecision.APPROVED if event.approved else ApprovalDecision.REJECTED
        approval = self.policy.record_decision(
            incident,
            action.id,
            decision=decision,
            user_id=f"{event.surface}:{event.user.id}",
            user_name=event.user.display_name,
            channel=event.surface,
        )
        if approval.decision is ApprovalDecision.EXPIRED:
            return ApprovalOutcome(
                "This approval request expired - the system may have changed since "
                "it was posted. Nothing was run.",
                retire_card=True,
                label="EXPIRED",
            )
        if not event.approved:
            return ApprovalOutcome("Rejected. Nothing was run.", retire_card=True, label="REJECTED")

        allowed, why = self.policy.may_execute(incident, action)
        if not allowed:
            return ApprovalOutcome(f"Not run: {why}", retire_card=True, label="BLOCKED")

        self._busy.add(incident.id)
        self._spawn(self._remediate(incident, action))
        return ApprovalOutcome(
            "Approved. Executing now; verification results will follow in the channel.",
            retire_card=True,
            label="APPROVED",
        )

    def _plan_for(self, incident: Incident) -> Any:
        container = (incident.signals[0].raw.get("container") if incident.signals else "") or (
            f"demo-{incident.service}"
        )
        health_url = ""
        if self.settings.demo_enabled and incident.service in DEMO_ENDPOINTS:
            health_url = f"{DEMO_ENDPOINTS[incident.service]}/health"
        return plan_for_container_service(
            container=container,
            health_url=health_url,
            window_seconds=self.settings.verify_window_seconds,
            poll_interval_seconds=self.settings.verify_poll_seconds,
            required_consecutive_passes=self.settings.verify_required_passes,
        )

    async def _remediate(self, incident: Incident, action: ProposedAction) -> None:
        try:
            incident.state = IncidentState.REMEDIATING
            incident.closed_at = None
            plan = self._plan_for(incident)

            # Baseline BEFORE the change - without it REGRESSED is undetectable.
            await self.verifier.capture_baseline(plan)

            result = await self.executors.execute(action.forward, action_id=action.id)
            incident.executions.append(result)
            incident.log(
                "agent",
                "executed" if result.succeeded else "execution_failed",
                result.stdout or result.error or result.stderr,
                action_id=action.id,
            )
            if not result.succeeded:
                incident.state = IncidentState.ESCALATED
                await self.surface.post(
                    Card(
                        title=f"Execution failed: {action.intent}",
                        body=truncate(result.error or result.stderr or "unknown error", 1500),
                        tone=Tone.CRITICAL,
                        footer=incident.id,
                    )
                )
                return

            await self.surface.post(
                Card(
                    title=f"Executed: {action.intent}",
                    body=truncate(result.stdout, 1500),
                    fields=[
                        Field(
                            "Now verifying",
                            f"{len(plan.checks)} checks, every {plan.poll_interval_seconds}s "
                            f"for {plan.window_seconds}s. Health must hold to the end of the "
                            f"window - one green poll is not a recovery.",
                        )
                    ],
                    tone=Tone.INFO,
                    footer=incident.id,
                )
            )

            report = await self.verifier.verify_and_maybe_rollback(incident, action, plan)
            await self.surface.post(verification_card(incident, report))
            await self.surface.post(incident_card(incident))

            if incident.state is IncidentState.RESOLVED:
                self.store.mark_quiet(incident.service, _now())
                await self._retire_cards(incident, "Incident resolved - no longer actionable.")
        except Exception as exc:  # noqa: BLE001
            log.exception("remediation failed")
            incident.state = IncidentState.ESCALATED
            incident.log("agent", "escalated", f"Remediation crashed: {exc}")
            await self.surface.post(
                Card(
                    title="Remediation crashed - escalating",
                    body=f"`{exc}`\nProduction may be in an unknown state.",
                    tone=Tone.CRITICAL,
                    footer=incident.id,
                )
            )
        finally:
            self._busy.discard(incident.id)

    async def _retire_cards(self, incident: Incident, note: str) -> None:
        """Remove buttons from every undecided card for this incident."""
        for action_id, posted in list(self._approval_cards.items()):
            if posted.incident_id != incident.id:
                continue
            del self._approval_cards[action_id]
            if incident.approval_for(action_id) is not None:
                continue  # the surface already retired this one on click
            card = approval_card(incident, posted.action)
            card.buttons = []
            card.title = f"No longer needed: {posted.action.intent}"
            card.tone = Tone.NEUTRAL
            card.fields.append(Field("Status", note))
            try:
                await self.surface.update(posted.ref, card)
            except Exception:  # noqa: BLE001 - cosmetic; never fail a resolution over it
                log.exception("could not retire card for %s", action_id)

    # -- commands ------------------------------------------------------------

    async def handle_command(self, cmd: ChatCommand) -> Card | str:
        name = cmd.name
        if name == "status":
            return await self._status_card()
        if name == "incidents":
            return self._incidents_card()
        if name == "incident":
            incident = self.store.get(cmd.args.get("incident_id", "").strip())
            if incident is None:
                return f"No incident `{cmd.args.get('incident_id')}`. Try `/incidents`."
            if incident.diagnosis is not None:
                await self.surface.post(
                    diagnosis_card(incident, incident.diagnosis), channel_id=cmd.channel_id
                )
            return incident_card(incident)
        if name == "diagnose":
            return await self._rediagnose(cmd.args.get("incident_id", "").strip(), cmd)
        if name == "discover":
            return await self._discover_card()
        if name == "settings":
            return self._settings_card()
        if name == "inject":
            return await self._inject(cmd.args.get("scenario", "").strip(), cmd)
        if name == "clear":
            return await self._clear(cmd)
        return f"Unknown command `/{name}`."

    async def _status_card(self, title: str = "Status") -> Card:
        health = await self.docker.health()
        states = [s for s in await self.docker.inventory()]
        lines = []
        for s in sorted(states, key=lambda x: x.name):
            mark = "[!!]" if s.is_suspicious else "[ok]"
            lines.append(f"`{mark}` {s.summarise()}")
        open_ = self.store.open()
        fields = [
            Field("Docker", health.detail if health.ok else f"UNREACHABLE: {health.error}"),
            Field("Monitored", "\n".join(lines) or "_no labelled containers_"),
            Field(
                "Open incidents",
                "\n".join(f"`{i.id}` {i.state.value} - {truncate(i.title, 90)}" for i in open_)
                or "none",
            ),
            Field(
                "Reasoning",
                MODEL_ID if not self.settings.llm_configured else "LLM configured",
                inline=True,
            ),
            Field("Autonomy", self.policy.config.autonomy.value, inline=True),
        ]
        tone = Tone.CRITICAL if open_ else Tone.SUCCESS
        return Card(title=title, fields=fields, tone=tone,
                    footer=f"monitor every {self.settings.monitor_interval_seconds}s")

    def _incidents_card(self) -> Card:
        recent = self.store.recent(10)
        body = "\n".join(
            f"`{i.id}` **{i.state.value}** {i.severity.value.upper()} - {truncate(i.title, 90)}"
            for i in recent
        )
        return Card(title="Recent incidents", body=body or "No incidents yet.", tone=Tone.INFO)

    async def _rediagnose(self, incident_id: str, cmd: ChatCommand) -> Card | str:
        incident = self.store.get(incident_id)
        if incident is None:
            return f"No incident `{incident_id}`."
        if incident.id in self._busy:
            return "A remediation is in flight; diagnosis is frozen until it finishes."
        container = incident.signals[0].raw.get("container", f"demo-{incident.service}")
        result = await self.triage.investigate(incident, container)
        incident.diagnosis = result.diagnosis
        incident.log("user:" + cmd.user.display_name, "rediagnosed")
        return diagnosis_card(incident, result.diagnosis)

    async def _discover_card(self) -> Card:
        states = await self.docker.inventory()
        lines = [f"`{s.name}` {s.status} {s.image}" for s in states]
        return Card(
            title="Discovery",
            fields=[
                Field("Local Docker", "\n".join(lines) or "nothing labelled for monitoring"),
                Field("Remote hosts", "No SSH hosts configured yet."),
            ],
            tone=Tone.INFO,
        )

    def _settings_card(self) -> Card:
        s = self.settings
        return Card(
            title="Settings",
            fields=[
                Field("Reasoning", "LLM configured" if s.llm_configured else
                      f"{MODEL_ID}. No LLM provider configured yet - rule-based triage "
                      "handles known failure shapes and abstains on the rest."),
                Field("Autonomy", s.autonomy.value, inline=True),
                Field("Approvers", s.discord_admin_role or "server administrators", inline=True),
                Field("Verification",
                      f"{s.verify_window_seconds}s window, poll {s.verify_poll_seconds}s, "
                      f"{s.verify_required_passes} consecutive passes", inline=True),
                Field("Demo stack", "enabled" if s.demo_enabled else "disabled", inline=True),
                Field("Executors", truncate(
                    ", ".join(sorted(e.id for e in self.executors.all())), 1000)),
            ],
            tone=Tone.NEUTRAL,
        )

    async def _inject(self, scenario: str, cmd: ChatCommand) -> str:
        if not self.settings.demo_enabled:
            return "Demo commands are disabled (OPSLOOP_DEMO_ENABLED=false)."
        if not cmd.user.is_admin:
            return "Only administrators can inject faults."
        faults = DEMO_SCENARIOS.get(scenario)
        if faults is None:
            return f"Unknown scenario `{scenario}`. Options: {', '.join(DEMO_SCENARIOS)}"
        done = []
        async with httpx.AsyncClient(timeout=10) as http:
            for service, mode in faults.items():
                try:
                    r = await http.post(f"{DEMO_ENDPOINTS[service]}/admin/fault",
                                        json={"mode": mode, "intensity": 1.0})
                    r.raise_for_status()
                    done.append(f"{service} -> {mode}")
                except httpx.HTTPError as exc:
                    return (f"Could not reach {service} to inject `{mode}` ({exc.__class__.__name__}). "
                            "Is the demo stack up, or is the service already down? Try `/clear`.")
        return (f"Injected **{scenario}**: {', '.join(done)}. The monitor scans every "
                f"{self.settings.monitor_interval_seconds}s - watch this channel.")

    async def _clear(self, cmd: ChatCommand) -> str:
        if not self.settings.demo_enabled:
            return "Demo commands are disabled."
        if not cmd.user.is_admin:
            return "Only administrators can clear faults."
        notes = []
        for service in DEMO_ENDPOINTS:
            result = await self.executors.execute(
                ActionSpec(executor="demo.apply_code_fix", params={"service": service},
                           target=service),
                action_id="operator-clear",
            )
            notes.append(result.stdout if result.succeeded else f"{service}: {result.error}")
        closed = 0
        for incident in self.store.open():
            if incident.id in self._busy:
                continue
            incident.state = IncidentState.CLOSED_MANUALLY
            incident.closed_at = _now()
            incident.log(f"user:{cmd.user.display_name}", "closed", "demo reset via /clear")
            self.store.mark_quiet(incident.service, _now())
            await self._retire_cards(incident, "Closed by /clear.")
            closed += 1
        return ("Faults cleared.\n" + "\n".join(f"- {n}" for n in notes)
                + f"\nClosed {closed} open incident(s).")

    # -- questions -------------------------------------------------------------

    async def handle_question(self, q: ChatQuestion) -> Card | str:
        open_ = self.store.open()
        summary = (
            "\n".join(f"- `{i.id}` {i.state.value}: {truncate(i.title, 100)}" for i in open_)
            or "- nothing open"
        )
        return (
            "Free-text reasoning needs an LLM provider, and none is configured yet - "
            "that is the next piece being built. Rule-based triage is running and "
            "watching production in the meantime.\n\n"
            f"**Open incidents**\n{summary}\n\n"
            "Try `/status`, `/incidents` or `/incident <id>`."
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_surface(settings: Settings) -> ChatSurface:
    if settings.discord_token:
        from .chat.discord_bot import DiscordSurface

        return DiscordSurface(
            settings.discord_token,
            default_channel_id=settings.discord_channel_id,
            guild_id=settings.discord_guild_id,
            admin_role=settings.discord_admin_role,
            enable_message_chat=settings.discord_message_chat,
            scenario_choices=list(DEMO_SCENARIOS) if settings.demo_enabled else None,
        )
    from .chat.console import ConsoleSurface

    return ConsoleSurface()


async def run() -> None:
    settings = Settings.from_env()
    agent = Agent(settings, build_surface(settings))
    await agent.start()
    log.info("OpsLoop running on %s", agent.surface.name)
    try:
        await asyncio.Event().wait()
    finally:
        await agent.stop()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    _load_dotenv(Path.cwd() / ".env")
    _load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    logging.basicConfig(
        level=os.environ.get("OPSLOOP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("discord").setLevel(logging.WARNING)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
