"""Core domain contracts for OpsLoop.

Two invariants are enforced here rather than left to prompt wording, because a
prompt is not a safety mechanism:

1. EVIDENCE GROUNDING - a Hypothesis with no evidence references is not a
   hypothesis, it is a guess. `Diagnosis.grounded_hypotheses` filters them out
   and `Diagnosis.must_abstain` reports when nothing survives.
   (Cloud-OpsBench 2026: a diagnosis was asserted with zero tool calls in 32%
   of cases; ORCA-bench: ~40% implausible-cause rate on weaker models.)

2. REVERSIBILITY - a ProposedAction that changes state must carry its own
   inverse, or it cannot enter the approval flow. Without an inverse there is
   no rollback, and without rollback there is no closed loop.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Severity and lifecycle
# --------------------------------------------------------------------------


class Severity(str, Enum):
    SEV1 = "sev1"  # total outage / data at risk
    SEV2 = "sev2"  # major degradation, user-visible
    SEV3 = "sev3"  # partial degradation, contained
    SEV4 = "sev4"  # cosmetic or informational


class IncidentState(str, Enum):
    DETECTED = "detected"
    TRIAGING = "triaging"
    DIAGNOSED = "diagnosed"
    AWAITING_APPROVAL = "awaiting_approval"
    REMEDIATING = "remediating"
    VERIFYING = "verifying"
    RESOLVED = "resolved"
    ROLLED_BACK = "rolled_back"
    ESCALATED = "escalated"  # agent abstained or ran out of options
    CLOSED_MANUALLY = "closed_manually"


TERMINAL_STATES = {
    IncidentState.RESOLVED,
    IncidentState.ROLLED_BACK,
    IncidentState.ESCALATED,
    IncidentState.CLOSED_MANUALLY,
}


class RiskLevel(str, Enum):
    """Decides whether an action may ever run unattended (see policy/)."""

    SAFE = "safe"  # read-only or trivially reversible
    LOW = "low"  # restart a stateless process
    MEDIUM = "medium"  # config change, scale, cache flush
    HIGH = "high"  # deploy rollback, schema touch, data movement
    PROHIBITED = "prohibited"  # never automated, always human-only


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


class TrustLevel(str, Enum):
    """How far the *text* of this evidence can be trusted.

    Telemetry is attacker-reachable: anyone who can cause a log line can put
    words in front of the model (AIOpsDoom, RSAC 2025 - 90% success over 180
    trials). Numbers from a metrics store are hard to weaponise; free-text log
    bodies are not.
    """

    TRUSTED = "trusted"  # our own probes, exit codes, metric values
    UNTRUSTED_TEXT = "untrusted"  # log bodies, exception messages, HTTP bodies


class EvidenceRef(BaseModel):
    """One concrete, re-fetchable observation. The unit of proof."""

    id: str = Field(default_factory=lambda: _new_id("ev"))
    source_kind: str  # "loki" | "docker" | "ssh" | "prometheus" | "probe"
    source_name: str  # which configured adapter produced it
    query: str  # the exact query, so a human can re-run it
    observed_at: datetime
    excerpt: str  # SANITISED text, ready for a prompt
    raw_sha256: str  # fingerprint of the original, for audit
    trust: TrustLevel = TrustLevel.UNTRUSTED_TEXT
    tainted: bool = False  # sanitiser found injection-shaped content
    taint_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def citation(self) -> str:
        return f"[{self.id} | {self.source_name} | {self.observed_at.isoformat()}]"


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------


class Hypothesis(BaseModel):
    """A candidate root cause. Worthless without citations."""

    id: str = Field(default_factory=lambda: _new_id("hyp"))
    statement: str  # "Orders API is OOM-killed under checkout load"
    mechanism: str  # how the cause produces the observed symptom
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return len(self.evidence_ids) > 0


class Diagnosis(BaseModel):
    incident_id: str
    produced_at: datetime = Field(default_factory=_now)
    model_id: str  # which provider/model produced this
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None
    tool_call_count: int = 0  # zero tool calls + a confident claim = speculation

    @property
    def evidence_index(self) -> dict[str, EvidenceRef]:
        return {e.id: e for e in self.evidence}

    @property
    def grounded_hypotheses(self) -> list[Hypothesis]:
        """Hypotheses whose citations all resolve to evidence we really collected.

        A fabricated citation is treated exactly like a missing one.
        """
        index = self.evidence_index
        kept = [
            h
            for h in self.hypotheses
            if h.is_grounded and all(eid in index for eid in h.evidence_ids)
        ]
        return sorted(kept, key=lambda h: h.confidence, reverse=True)

    @property
    def dropped_hypotheses(self) -> list[Hypothesis]:
        kept = {h.id for h in self.grounded_hypotheses}
        return [h for h in self.hypotheses if h.id not in kept]

    @property
    def leading(self) -> Hypothesis | None:
        grounded = self.grounded_hypotheses
        return grounded[0] if grounded else None

    def must_abstain(self, min_confidence: float = 0.45) -> tuple[bool, str | None]:
        """Abstaining is a correct answer. Guessing is not."""
        if self.tool_call_count == 0:
            return True, "Model reached a conclusion without collecting any telemetry."
        grounded = self.grounded_hypotheses
        if not grounded:
            return True, "No hypothesis cited evidence that we actually collected."
        if grounded[0].confidence < min_confidence:
            return True, (
                f"Leading hypothesis confidence {grounded[0].confidence:.2f} is below "
                f"the {min_confidence:.2f} threshold."
            )
        return False, None


# --------------------------------------------------------------------------
# Remediation
# --------------------------------------------------------------------------


class ActionSpec(BaseModel):
    """A single executable step, expressed so the executor - not the model -
    decides what actually runs."""

    executor: str  # registered executor id, e.g. "docker.restart"
    params: dict[str, Any] = Field(default_factory=dict)
    target: str  # human-readable: "orders-api on vps-1"

    def describe(self) -> str:
        arg = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{self.executor}({arg}) -> {self.target}"


class ProposedAction(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("act"))
    intent: str  # "Restart the OOM-killed orders-api container"
    forward: ActionSpec
    rollback: ActionSpec | None = None  # required unless read_only
    read_only: bool = False
    risk: RiskLevel = RiskLevel.MEDIUM
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    expected_effect: str = ""  # what should change if this worked
    blast_radius: str = ""  # what else this could disturb

    @model_validator(mode="after")
    def _require_inverse(self) -> ProposedAction:
        if not self.read_only and self.rollback is None:
            raise ValueError(
                f"Action {self.id!r} mutates state but declares no rollback. "
                "Irreversible actions cannot enter the approval flow."
            )
        return self


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Approval(BaseModel):
    action_id: str
    decision: ApprovalDecision
    decided_by: str  # Teams AAD object id, or local user
    decided_by_name: str = ""
    decided_at: datetime = Field(default_factory=_now)
    note: str = ""
    channel: str = "teams"  # where the approval was given


class ExecutionResult(BaseModel):
    action_id: str
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    succeeded: bool = False
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    was_rollback: bool = False


# --------------------------------------------------------------------------
# Verification - Difference 1, the part nobody else guarantees
# --------------------------------------------------------------------------


class HealthCheck(BaseModel):
    """One falsifiable statement about the system being healthy again."""

    id: str = Field(default_factory=lambda: _new_id("chk"))
    name: str
    kind: Literal["http", "metric", "log_absence", "process", "command"]
    params: dict[str, Any] = Field(default_factory=dict)
    # A check must state its own pass condition; "looks fine" is not a condition.
    success_criteria: str


class CheckOutcome(BaseModel):
    check_id: str
    at: datetime = Field(default_factory=_now)
    passed: bool
    observed: str
    detail: str = ""


class VerificationPlan(BaseModel):
    checks: list[HealthCheck] = Field(default_factory=list)
    window_seconds: int = 300  # how long health must hold, not merely touch
    poll_interval_seconds: int = 15
    required_consecutive_passes: int = 3  # one green poll is a coincidence
    baseline: dict[str, Any] = Field(default_factory=dict)  # pre-fix readings
    #: Hard ceiling on polls, independent of timing. Bounds the work when a
    #: check is far faster than its interval, and makes tests deterministic.
    max_polls: int | None = None

    @model_validator(mode="after")
    def _reject_unbounded_polling(self) -> VerificationPlan:
        """A zero interval with no ceiling is a busy loop aimed at production.

        Verification runs against a system that is already unwell. Polling it
        as fast as the event loop allows would add load at the worst possible
        moment - a health check must never become part of the incident. Such a
        plan is refused rather than quietly clamped, so the misconfiguration is
        visible instead of merely survivable.
        """
        if self.poll_interval_seconds <= 0 and self.max_polls is None:
            raise ValueError(
                "poll_interval_seconds <= 0 requires max_polls to be set; "
                "otherwise verification would poll the target in a busy loop."
            )
        return self


class VerificationVerdict(str, Enum):
    RECOVERED = "recovered"
    NOT_RECOVERED = "not_recovered"
    REGRESSED = "regressed"  # made it worse - roll back immediately
    INCONCLUSIVE = "inconclusive"


class VerificationReport(BaseModel):
    plan: VerificationPlan
    outcomes: list[CheckOutcome] = Field(default_factory=list)
    verdict: VerificationVerdict = VerificationVerdict.INCONCLUSIVE
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    summary: str = ""
    rollback_triggered: bool = False


# --------------------------------------------------------------------------
# Incident
# --------------------------------------------------------------------------


class TimelineEntry(BaseModel):
    at: datetime = Field(default_factory=_now)
    actor: str  # "agent" | "user:<name>" | "system"
    event: str
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class Signal(BaseModel):
    """What woke the agent up."""

    id: str = Field(default_factory=lambda: _new_id("sig"))
    detector: str  # which monitor fired
    title: str
    observed_at: datetime = Field(default_factory=_now)
    severity: Severity = Severity.SEV3
    service: str = ""
    environment: str = "production"
    raw: dict[str, Any] = Field(default_factory=dict)


class Incident(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("inc"))
    title: str
    state: IncidentState = IncidentState.DETECTED
    severity: Severity = Severity.SEV3
    service: str = ""
    environment: str = "production"
    opened_at: datetime = Field(default_factory=_now)
    closed_at: datetime | None = None

    signals: list[Signal] = Field(default_factory=list)
    diagnosis: Diagnosis | None = None
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    approvals: list[Approval] = Field(default_factory=list)
    executions: list[ExecutionResult] = Field(default_factory=list)
    verification: VerificationReport | None = None

    timeline: list[TimelineEntry] = Field(default_factory=list)
    conversation_ref: dict[str, Any] = Field(default_factory=dict)  # Teams thread

    def log(self, actor: str, event: str, detail: str = "", **data: Any) -> None:
        self.timeline.append(
            TimelineEntry(actor=actor, event=event, detail=detail, data=data)
        )

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def action(self, action_id: str) -> ProposedAction | None:
        return next((a for a in self.proposed_actions if a.id == action_id), None)

    def approval_for(self, action_id: str) -> Approval | None:
        return next((a for a in self.approvals if a.action_id == action_id), None)
