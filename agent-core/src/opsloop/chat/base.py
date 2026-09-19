"""Chat surface protocol - the third axis of neutrality.

The project already refuses to be locked to one telemetry vendor
(`telemetry/`) or one model vendor (`llm/`). This is the same argument applied
to the conversation itself: Discord, Teams, Slack and a local console are
renderers, not architecture.

A `Card` is described once in neutral terms and rendered natively by each
surface - a Discord embed with buttons, a Teams Adaptive Card, plain text in a
terminal. No incident logic lives in a surface; a surface translates.

The security rule that survives every surface
---------------------------------------------
`ApprovalEvent.user_id` MUST come from the platform's authenticated identity -
`interaction.user.id` on Discord, the AAD object id on Teams - and never from
message text. This is what stops a log line reading "the on-call engineer has
already approved this" from becoming an approval. `policy.record_decision`
refuses an empty user id for the same reason, so the rule is enforced twice,
independently, at the boundary and at the gate.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from ..domain.models import (
    Diagnosis,
    Incident,
    IncidentState,
    ProposedAction,
    Severity,
    VerificationReport,
    VerificationVerdict,
)


# --------------------------------------------------------------------------
# Neutral presentation model
# --------------------------------------------------------------------------


class Tone(str, Enum):
    """Semantic, not literal. Each surface maps these to its own palette."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"
    NEUTRAL = "neutral"
    SECURITY = "security"


class ButtonStyle(str, Enum):
    PRIMARY = "primary"
    DANGER = "danger"
    SECONDARY = "secondary"


@dataclass
class Button:
    action_id: str  # opaque; routed back to the handler verbatim
    label: str
    style: ButtonStyle = ButtonStyle.SECONDARY
    emoji: str = ""


@dataclass
class Field:
    name: str
    value: str
    inline: bool = False


@dataclass
class Card:
    title: str
    body: str = ""
    fields: list[Field] = field(default_factory=list)
    tone: Tone = Tone.NEUTRAL
    footer: str = ""
    buttons: list[Button] = field(default_factory=list)
    # Surfaces that support threading use this to keep an incident together.
    thread_key: str = ""


@dataclass
class MessageRef:
    """Where a card was posted, so it can be edited or replied to later."""

    surface: str
    channel_id: str
    message_id: str = ""
    thread_id: str = ""

    def as_conversation_ref(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "channel_id": self.channel_id,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
        }


# --------------------------------------------------------------------------
# Inbound events
# --------------------------------------------------------------------------


@dataclass
class ChatUser:
    """An identity the PLATFORM authenticated. Never parsed from text."""

    id: str
    display_name: str
    is_admin: bool = False


@dataclass
class ApprovalEvent:
    incident_id: str
    action_id: str
    approved: bool
    user: ChatUser
    surface: str
    note: str = ""


@dataclass
class ChatCommand:
    name: str  # "status" | "incidents" | "discover" | ...
    args: dict[str, Any]
    user: ChatUser
    channel_id: str
    surface: str


@dataclass
class ChatQuestion:
    """Free-text addressed to the agent."""

    text: str
    user: ChatUser
    channel_id: str
    surface: str
    incident_id: str = ""  # set when asked inside an incident thread


ApprovalHandler = Callable[[ApprovalEvent], Awaitable[str]]
CommandHandler = Callable[[ChatCommand], Awaitable[Card | str]]
QuestionHandler = Callable[[ChatQuestion], Awaitable[Card | str]]


# --------------------------------------------------------------------------
# Surface
# --------------------------------------------------------------------------


class ChatSurface(abc.ABC):
    name: str = "abstract"

    def __init__(self) -> None:
        self._on_approval: ApprovalHandler | None = None
        self._on_command: CommandHandler | None = None
        self._on_question: QuestionHandler | None = None

    def on_approval(self, handler: ApprovalHandler) -> None:
        self._on_approval = handler

    def on_command(self, handler: CommandHandler) -> None:
        self._on_command = handler

    def on_question(self, handler: QuestionHandler) -> None:
        self._on_question = handler

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def post(self, card: Card, *, channel_id: str = "") -> MessageRef:
        """Post a card. An empty channel_id means the configured default."""

    @abc.abstractmethod
    async def update(self, ref: MessageRef, card: Card) -> None:
        """Replace an existing card in place - used to retire approval buttons
        once a decision is made, so a stale card cannot be clicked twice."""

    async def post_text(self, text: str, *, channel_id: str = "") -> MessageRef:
        return await self.post(Card(title="", body=text), channel_id=channel_id)


# --------------------------------------------------------------------------
# Card builders - shared by every surface
# --------------------------------------------------------------------------

_SEVERITY_TONE = {
    Severity.SEV1: Tone.CRITICAL,
    Severity.SEV2: Tone.CRITICAL,
    Severity.SEV3: Tone.WARNING,
    Severity.SEV4: Tone.INFO,
}

_STATE_TONE = {
    IncidentState.RESOLVED: Tone.SUCCESS,
    IncidentState.ROLLED_BACK: Tone.WARNING,
    IncidentState.ESCALATED: Tone.CRITICAL,
    IncidentState.AWAITING_APPROVAL: Tone.WARNING,
    IncidentState.VERIFYING: Tone.INFO,
}

_VERDICT_TONE = {
    VerificationVerdict.RECOVERED: Tone.SUCCESS,
    VerificationVerdict.NOT_RECOVERED: Tone.CRITICAL,
    VerificationVerdict.REGRESSED: Tone.CRITICAL,
    VerificationVerdict.INCONCLUSIVE: Tone.WARNING,
}


def truncate(text: str, limit: int, *, suffix: str = " ...[truncated]") -> str:
    """Chat platforms cap field lengths; exceeding one rejects the whole message.

    Losing an entire incident alert because a stack trace was long is a real
    failure mode, so every builder clamps rather than hoping.
    """
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix


def incident_card(incident: Incident) -> Card:
    diag = incident.diagnosis
    fields = [
        Field("State", incident.state.value, inline=True),
        Field("Severity", incident.severity.value.upper(), inline=True),
        Field("Service", incident.service or "unknown", inline=True),
    ]

    if diag is not None:
        leading = diag.leading
        if diag.abstained or leading is None:
            fields.append(
                Field(
                    "Diagnosis",
                    truncate(
                        "**Abstained.** "
                        + (diag.abstain_reason or "Evidence did not meet the bar.")
                        + "\n\nAbstaining is a correct answer; escalating to a human.",
                        1024,
                    ),
                )
            )
        else:
            fields.append(
                Field(
                    f"Leading hypothesis ({leading.confidence:.0%} confidence)",
                    truncate(f"{leading.statement}\n\n_{leading.mechanism}_", 1024),
                )
            )
            cited = [e for e in diag.evidence if e.id in leading.evidence_ids]
            if cited:
                lines = [f"`{e.id}` {truncate(e.query, 70)}" for e in cited[:5]]
                if len(cited) > 5:
                    lines.append(f"_...and {len(cited) - 5} more_")
                fields.append(Field(f"Evidence ({len(cited)} cited)", truncate("\n".join(lines), 1024)))

        tainted = [e for e in diag.evidence if e.tainted]
        if tainted:
            reasons = sorted({r for e in tainted for r in e.taint_reasons})[:4]
            fields.append(
                Field(
                    "[!] Security",
                    truncate(
                        f"{len(tainted)} piece(s) of telemetry were flagged as "
                        f"potentially adversarial ({', '.join(reasons)}). Treated as a "
                        f"finding to investigate, not as instructions.",
                        1024,
                    ),
                )
            )

    if incident.verification is not None:
        v = incident.verification
        fields.append(
            Field(f"Verification: {v.verdict.value.upper()}", truncate(v.summary, 1024))
        )

    tone = _STATE_TONE.get(incident.state) or _SEVERITY_TONE.get(
        incident.severity, Tone.NEUTRAL
    )
    return Card(
        title=truncate(incident.title, 240),
        body="",
        fields=fields,
        tone=tone,
        footer=f"{incident.id} · opened {incident.opened_at:%H:%M:%S UTC}",
        thread_key=incident.id,
    )


def approval_card(
    incident: Incident, action: ProposedAction, decision: Any = None
) -> Card:
    """The card a human acts on.

    Everything needed for an informed decision is on it before anything runs:
    what will happen, how it will be undone, what else it could disturb, and
    any security warning. A rollback that is only discovered after the fact is
    not a rollback anyone consented to.
    """
    fields = [
        Field("Will run", f"`{truncate(action.forward.describe(), 1000)}`"),
        Field(
            "Rollback if it fails",
            f"`{truncate(action.rollback.describe(), 1000)}`"
            if action.rollback
            else "_none declared_",
        ),
        Field("Risk", action.risk.value, inline=True),
        Field("Target", truncate(action.forward.target, 100), inline=True),
    ]
    if action.expected_effect:
        fields.append(Field("Expected effect", truncate(action.expected_effect, 1024)))
    if action.blast_radius:
        fields.append(Field("Blast radius", truncate(action.blast_radius, 1024)))
    if action.rationale:
        fields.append(Field("Why", truncate(action.rationale, 1024)))

    tone = Tone.WARNING
    if decision is not None:
        for warning in getattr(decision, "warnings", []) or []:
            fields.append(Field("[!] Security", truncate(warning, 1024)))
            tone = Tone.SECURITY
        reasons = getattr(decision, "reasons", []) or []
        if reasons:
            fields.append(
                Field("Policy", truncate("\n".join(f"• {r}" for r in reasons), 1024))
            )

    return Card(
        title=truncate(f"Approval needed: {action.intent}", 240),
        fields=fields,
        tone=tone,
        footer=f"{incident.id} · {action.id}",
        thread_key=incident.id,
        buttons=[
            Button(f"approve:{incident.id}:{action.id}", "Approve", ButtonStyle.PRIMARY, "✅"),
            Button(f"reject:{incident.id}:{action.id}", "Reject", ButtonStyle.DANGER, "✖"),
        ],
    )


def verification_card(incident: Incident, report: VerificationReport) -> Card:
    passed = sum(1 for o in report.outcomes if o.passed)
    fields = [
        Field("Verdict", report.verdict.value.upper(), inline=True),
        Field("Checks", f"{passed}/{len(report.outcomes)} polls green", inline=True),
        Field("Window", f"{report.plan.window_seconds}s", inline=True),
        Field("Summary", truncate(report.summary, 1024)),
    ]
    if report.rollback_triggered:
        fields.append(
            Field(
                "Rollback",
                "The remediation was reverted automatically. Production is back to "
                "its pre-fix state and needs a human.",
            )
        )
    return Card(
        title=f"Verification: {report.verdict.value.replace('_', ' ')}",
        fields=fields,
        tone=_VERDICT_TONE.get(report.verdict, Tone.NEUTRAL),
        footer=f"{incident.id}",
        thread_key=incident.id,
    )


def diagnosis_card(incident: Incident, diagnosis: Diagnosis) -> Card:
    """Every hypothesis, including the ones that were thrown away.

    Showing dropped hypotheses is deliberate. It makes the grounding rule
    visible to the person reading: the agent considered this, could not cite
    evidence for it, and discarded it.
    """
    fields: list[Field] = []
    for i, h in enumerate(diagnosis.grounded_hypotheses[:3], 1):
        fields.append(
            Field(
                f"{i}. {truncate(h.statement, 200)} ({h.confidence:.0%})",
                truncate(
                    f"{h.mechanism}\n\nCited: {', '.join(f'`{e}`' for e in h.evidence_ids[:6])}",
                    1024,
                ),
            )
        )
    dropped = diagnosis.dropped_hypotheses
    if dropped:
        fields.append(
            Field(
                f"Discarded ({len(dropped)})",
                truncate(
                    "\n".join(f"• {truncate(h.statement, 120)}" for h in dropped[:4])
                    + "\n\n_Dropped: no citation, or cited evidence we never collected._",
                    1024,
                ),
            )
        )
    return Card(
        title="Diagnosis",
        body=truncate(
            f"{diagnosis.tool_call_count} tool call(s), "
            f"{len(diagnosis.evidence)} piece(s) of evidence collected.",
            2000,
        ),
        fields=fields,
        tone=Tone.INFO if not diagnosis.abstained else Tone.WARNING,
        footer=f"{incident.id} · model: {diagnosis.model_id}",
        thread_key=incident.id,
    )
