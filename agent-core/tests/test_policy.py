"""Policy gate tests.

The claim being defended: no text arriving through telemetry can cause an
action to execute. These tests are the evidence for it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from opsloop.domain.models import (
    ActionSpec,
    Approval,
    ApprovalDecision,
    Diagnosis,
    EvidenceRef,
    Hypothesis,
    Incident,
    ProposedAction,
    RiskLevel,
    TimelineEntry,
)
from opsloop.policy.engine import (
    AutonomyMode,
    Disposition,
    PolicyConfig,
    PolicyEngine,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def evidence(ev_id: str = "ev_1", *, tainted: bool = False) -> EvidenceRef:
    return EvidenceRef(
        id=ev_id,
        source_kind="docker",
        source_name="demo",
        query="docker inspect orders-api",
        observed_at=NOW,
        excerpt="OOMKilled=true exit=137",
        raw_sha256="0" * 64,
        tainted=tainted,
        taint_reasons=["fake_approval", "instruction_override"] if tainted else [],
    )


def diagnosis(*evs: EvidenceRef, confidence: float = 0.8) -> Diagnosis:
    return Diagnosis(
        incident_id="inc_1",
        model_id="test",
        tool_call_count=3,
        evidence=list(evs),
        hypotheses=[
            Hypothesis(
                statement="orders-api OOM-killed",
                mechanism="heap exceeds the cgroup limit",
                confidence=confidence,
                evidence_ids=[e.id for e in evs],
            )
        ],
    )


def incident(diag: Diagnosis | None = None, *, environment: str = "production") -> Incident:
    inc = Incident(title="orders-api down", environment=environment)
    inc.diagnosis = diag
    return inc


def action(
    *,
    executor: str = "docker.restart",
    risk: RiskLevel = RiskLevel.LOW,
    evidence_ids: list[str] | None = None,
    read_only: bool = False,
) -> ProposedAction:
    return ProposedAction(
        intent="Restart orders-api",
        forward=ActionSpec(executor=executor, params={"container": "x"}, target="x"),
        rollback=None
        if read_only
        else ActionSpec(executor="noop", params={"reason": "restart is not undoable"}, target="x"),
        read_only=read_only,
        risk=risk,
        evidence_ids=evidence_ids if evidence_ids is not None else ["ev_1"],
    )


class TestBlocking:
    def test_no_diagnosis_blocks(self) -> None:
        d = PolicyEngine().evaluate(action(), incident=incident(None))
        assert d.blocked and "No diagnosis" in d.reasons[0]

    def test_abstention_authorises_nothing(self) -> None:
        diag = diagnosis(evidence())
        diag.abstained = True
        diag.abstain_reason = "evidence too thin"
        d = PolicyEngine().evaluate(action(), incident=incident(diag))
        assert d.blocked and "abstained" in d.reasons[0]

    def test_low_confidence_diagnosis_blocks_action(self) -> None:
        d = PolicyEngine().evaluate(
            action(), incident=incident(diagnosis(evidence(), confidence=0.2))
        )
        assert d.blocked

    def test_zero_tool_calls_blocks_action(self) -> None:
        diag = diagnosis(evidence())
        diag.tool_call_count = 0
        d = PolicyEngine().evaluate(action(), incident=incident(diag))
        assert d.blocked

    def test_action_without_evidence_blocks(self) -> None:
        d = PolicyEngine().evaluate(
            action(evidence_ids=[]), incident=incident(diagnosis(evidence()))
        )
        assert d.blocked and "cites no evidence" in d.reasons[0]

    def test_fabricated_citation_blocks(self) -> None:
        d = PolicyEngine().evaluate(
            action(evidence_ids=["ev_never_collected"]),
            incident=incident(diagnosis(evidence())),
        )
        assert d.blocked and "never collected" in d.reasons[0]

    def test_prohibited_executor_blocks_even_when_approved(self) -> None:
        inc = incident(diagnosis(evidence()))
        a = action(executor="db.drop", risk=RiskLevel.HIGH)
        inc.proposed_actions.append(a)
        inc.approvals.append(
            Approval(
                action_id=a.id,
                decision=ApprovalDecision.APPROVED,
                decided_by="aad|real-human",
            )
        )
        engine = PolicyEngine()
        assert engine.evaluate(a, incident=inc).blocked
        allowed, why = engine.may_execute(inc, a)
        assert allowed is False and "prohibited" in why.lower()

    def test_unknown_environment_blocks(self) -> None:
        d = PolicyEngine().evaluate(
            action(), incident=incident(diagnosis(evidence()), environment="customer-laptop")
        )
        assert d.blocked

    def test_action_limit_blocks_runaway_agent(self) -> None:
        inc = incident(diagnosis(evidence()))
        engine = PolicyEngine(PolicyConfig(max_actions_per_incident=2))
        for _ in range(3):
            inc.proposed_actions.append(action())
        assert engine.evaluate(inc.proposed_actions[-1], incident=inc).blocked


class TestAutonomyModes:
    def test_read_only_never_needs_approval(self) -> None:
        d = PolicyEngine().evaluate(
            action(read_only=True), incident=incident(diagnosis(evidence()))
        )
        assert d.disposition is Disposition.AUTO_EXECUTE

    def test_approve_then_execute_always_asks(self) -> None:
        engine = PolicyEngine(PolicyConfig(autonomy=AutonomyMode.APPROVE_THEN_EXECUTE))
        d = engine.evaluate(
            action(risk=RiskLevel.SAFE), incident=incident(diagnosis(evidence()))
        )
        assert d.needs_human

    def test_tiered_auto_executes_below_threshold(self) -> None:
        engine = PolicyEngine(
            PolicyConfig(
                autonomy=AutonomyMode.TIERED, auto_execute_max_risk=RiskLevel.LOW
            )
        )
        d = engine.evaluate(
            action(risk=RiskLevel.LOW), incident=incident(diagnosis(evidence()))
        )
        assert d.disposition is Disposition.AUTO_EXECUTE

    def test_tiered_asks_above_threshold(self) -> None:
        engine = PolicyEngine(
            PolicyConfig(
                autonomy=AutonomyMode.TIERED, auto_execute_max_risk=RiskLevel.LOW
            )
        )
        d = engine.evaluate(
            action(risk=RiskLevel.HIGH), incident=incident(diagnosis(evidence()))
        )
        assert d.needs_human


class TestInjectionResistance:
    """The AIOpsDoom rules."""

    def test_tainted_evidence_forbids_automatic_execution(self) -> None:
        """A successful injection must become a prompt, never an action."""
        engine = PolicyEngine(
            PolicyConfig(
                autonomy=AutonomyMode.TIERED, auto_execute_max_risk=RiskLevel.HIGH
            )
        )
        ev = evidence(tainted=True)
        d = engine.evaluate(action(risk=RiskLevel.LOW), incident=incident(diagnosis(ev)))
        assert d.disposition is Disposition.REQUIRE_APPROVAL
        assert d.tainted_evidence_ids == ["ev_1"]
        assert any("SECURITY" in w for w in d.warnings)

    def test_taint_warning_names_the_reasons(self) -> None:
        engine = PolicyEngine()
        d = engine.evaluate(
            action(), incident=incident(diagnosis(evidence(tainted=True)))
        )
        warning = " ".join(d.warnings)
        assert "fake_approval" in warning or "instruction_override" in warning
        assert "not as a reason to act" in warning

    def test_approval_requires_an_authenticated_user(self) -> None:
        """A log line claiming 'the operator approved this' is not an approval."""
        engine = PolicyEngine()
        inc = incident(diagnosis(evidence()))
        with pytest.raises(ValueError, match="authenticated user id"):
            engine.record_decision(
                inc, "act_1", decision=ApprovalDecision.APPROVED, user_id=""
            )

    def test_unapproved_action_cannot_execute(self) -> None:
        engine = PolicyEngine()
        inc = incident(diagnosis(evidence()))
        a = action()
        inc.proposed_actions.append(a)
        allowed, why = engine.may_execute(inc, a)
        assert allowed is False and "no approval" in why


class TestApprovalLifecycle:
    def test_rejection_prevents_execution(self) -> None:
        engine = PolicyEngine()
        inc = incident(diagnosis(evidence()))
        a = action()
        inc.proposed_actions.append(a)
        engine.record_decision(
            inc,
            a.id,
            decision=ApprovalDecision.REJECTED,
            user_id="aad|1",
            user_name="oncall-engineer",
        )
        allowed, why = engine.may_execute(inc, a)
        assert allowed is False and "rejected by oncall-engineer" in why

    def test_approval_expires(self) -> None:
        """A decision about the system as it was 20 minutes ago is stale."""
        engine = PolicyEngine(PolicyConfig(approval_timeout_seconds=600))
        inc = incident(diagnosis(evidence()))
        a = action()
        inc.proposed_actions.append(a)
        inc.timeline.append(
            TimelineEntry(
                at=datetime.now(timezone.utc) - timedelta(seconds=1200),
                actor="agent",
                event="approval_requested",
                data={"action_id": a.id},
            )
        )
        assert engine.is_expired(inc, a.id) is True
        approval = engine.record_decision(
            inc, a.id, decision=ApprovalDecision.APPROVED, user_id="aad|1"
        )
        assert approval.decision is ApprovalDecision.EXPIRED
        allowed, why = engine.may_execute(inc, a)
        assert allowed is False and "expired" in why

    def test_fresh_approval_permits_execution(self) -> None:
        engine = PolicyEngine()
        inc = incident(diagnosis(evidence()))
        a = action()
        inc.proposed_actions.append(a)
        inc.log("agent", "approval_requested", action_id=a.id)
        engine.record_decision(
            inc, a.id, decision=ApprovalDecision.APPROVED, user_id="aad|1", user_name="sre-lead"
        )
        allowed, why = engine.may_execute(inc, a)
        assert allowed is True and "sre-lead" in why

    def test_decision_is_recorded_on_the_timeline(self) -> None:
        engine = PolicyEngine()
        inc = incident(diagnosis(evidence()))
        a = action()
        engine.record_decision(
            inc, a.id, decision=ApprovalDecision.APPROVED, user_id="aad|7", user_name="platform-eng"
        )
        entry = inc.timeline[-1]
        assert entry.event == "approval_approved"
        assert entry.actor == "user:platform-eng"
        assert entry.data["action_id"] == a.id
