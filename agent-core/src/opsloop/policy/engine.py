"""Policy engine - the gate between a proposal and production.

This is the component that makes the security argument hold. The sanitiser
raises the cost of a prompt-injection attack and makes a successful one
visible, but it does not guarantee anything; AIOpsDoom beat classifiers far
better resourced than ours. The actual guarantee is here: a state-changing
action reaches production only after a human taps Approve, and no text
arriving through telemetry can produce that approval.

Rules, in the order they are applied:

  1. An abstaining diagnosis authorises nothing.
  2. PROHIBITED executors are blocked outright, whatever anyone approves.
  3. An action citing no evidence is blocked - acting on nothing is worse
     than acting late.
  4. If any cited evidence is TAINTED, automatic execution is forbidden and
     the approval card carries a security warning. This is the rule that
     turns a successful injection into a visible prompt rather than an action.
  5. Under RECOMMEND_ONLY nothing executes, ever.
  6. Under APPROVE_THEN_EXECUTE every mutation needs a human.
  7. Under TIERED, actions at or below `auto_execute_max_risk` may run
     unattended - subject to every rule above.
  8. An incident may propose at most `max_actions_per_incident` actions, so a
     confused agent cannot thrash production in a loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum

from ..domain.models import (
    Approval,
    ApprovalDecision,
    Diagnosis,
    EvidenceRef,
    Incident,
    ProposedAction,
    RiskLevel,
    _now,
)


class AutonomyMode(str, Enum):
    RECOMMEND_ONLY = "recommend_only"
    APPROVE_THEN_EXECUTE = "approve_then_execute"
    TIERED = "tiered"


class Disposition(str, Enum):
    AUTO_EXECUTE = "auto_execute"
    REQUIRE_APPROVAL = "require_approval"
    BLOCKED = "blocked"


_RISK_ORDER = {
    RiskLevel.SAFE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.PROHIBITED: 99,
}


@dataclass
class PolicyConfig:
    autonomy: AutonomyMode = AutonomyMode.APPROVE_THEN_EXECUTE
    auto_execute_max_risk: RiskLevel = RiskLevel.LOW  # only used when TIERED
    approval_timeout_seconds: int = 900  # 15 minutes
    max_actions_per_incident: int = 5
    block_on_tainted_evidence: bool = True
    require_evidence_for_actions: bool = True
    min_leading_confidence: float = 0.45
    prohibited_executors: set[str] = field(
        default_factory=lambda: {
            # Irrecoverable or out of scope for an automated agent, regardless
            # of who approves them.
            "db.drop",
            "db.truncate",
            "fs.rm_rf",
            "host.shutdown",
            "host.reboot",
            "secrets.rotate",
            "iam.grant",
        }
    )
    # Environments the agent may act on at all.
    allowed_environments: set[str] = field(default_factory=lambda: {"production", "staging"})


@dataclass
class PolicyDecision:
    disposition: Disposition
    risk: RiskLevel
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tainted_evidence_ids: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.disposition is Disposition.BLOCKED

    @property
    def needs_human(self) -> bool:
        return self.disposition is Disposition.REQUIRE_APPROVAL

    def explain(self) -> str:
        head = {
            Disposition.AUTO_EXECUTE: "Will execute automatically",
            Disposition.REQUIRE_APPROVAL: "Requires human approval",
            Disposition.BLOCKED: "Blocked by policy",
        }[self.disposition]
        lines = [f"{head} (risk: {self.risk.value})"]
        lines += [f"  - {r}" for r in self.reasons]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


class PolicyEngine:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    # -- main entry point --------------------------------------------------

    def evaluate(
        self,
        action: ProposedAction,
        *,
        incident: Incident,
        diagnosis: Diagnosis | None = None,
    ) -> PolicyDecision:
        cfg = self.config
        reasons: list[str] = []
        warnings: list[str] = []

        # 0. Read-only actions are not gated. They change nothing, and making
        #    the agent ask permission to look at a log would make it useless.
        if action.read_only:
            return PolicyDecision(
                disposition=Disposition.AUTO_EXECUTE,
                risk=RiskLevel.SAFE,
                reasons=["Read-only action; changes no state."],
            )

        # 1. An abstaining diagnosis authorises nothing.
        diagnosis = diagnosis or incident.diagnosis
        if diagnosis is None:
            return PolicyDecision(
                Disposition.BLOCKED,
                action.risk,
                ["No diagnosis exists for this incident."],
            )
        if diagnosis.abstained:
            return PolicyDecision(
                Disposition.BLOCKED,
                action.risk,
                [f"Agent abstained from diagnosis: {diagnosis.abstain_reason}"],
            )
        must_abstain, abstain_reason = diagnosis.must_abstain(cfg.min_leading_confidence)
        if must_abstain:
            return PolicyDecision(
                Disposition.BLOCKED,
                action.risk,
                [f"Diagnosis does not meet the evidence bar: {abstain_reason}"],
            )

        # 2. Environment scope.
        if incident.environment not in cfg.allowed_environments:
            return PolicyDecision(
                Disposition.BLOCKED,
                action.risk,
                [
                    f"Environment {incident.environment!r} is not in the allowed set "
                    f"({', '.join(sorted(cfg.allowed_environments))})."
                ],
            )

        # 3. Prohibited executors - not overridable by approval.
        if action.forward.executor in cfg.prohibited_executors:
            return PolicyDecision(
                Disposition.BLOCKED,
                RiskLevel.PROHIBITED,
                [
                    f"Executor {action.forward.executor!r} is on the prohibited list "
                    "and cannot be run by the agent under any approval."
                ],
            )
        if action.risk is RiskLevel.PROHIBITED:
            return PolicyDecision(
                Disposition.BLOCKED,
                RiskLevel.PROHIBITED,
                ["Action is classified PROHIBITED."],
            )

        # 4. Reversibility. The model validator already enforces this, so
        #    reaching here means something constructed an action by another
        #    route - worth failing loudly rather than trusting it.
        if action.rollback is None:
            return PolicyDecision(
                Disposition.BLOCKED,
                action.risk,
                ["Action mutates state but carries no rollback."],
            )

        # 5. Evidence.
        if cfg.require_evidence_for_actions and not action.evidence_ids:
            return PolicyDecision(
                Disposition.BLOCKED,
                action.risk,
                ["Action cites no evidence. Acting on nothing is worse than acting late."],
            )

        index = diagnosis.evidence_index
        unknown = [eid for eid in action.evidence_ids if eid not in index]
        if unknown:
            return PolicyDecision(
                Disposition.BLOCKED,
                action.risk,
                [
                    "Action cites evidence that was never collected: "
                    + ", ".join(unknown)
                    + ". A fabricated citation is treated as no citation."
                ],
            )

        # 6. Rate limit.
        mutations = [a for a in incident.proposed_actions if not a.read_only]
        if len(mutations) > cfg.max_actions_per_incident:
            return PolicyDecision(
                Disposition.BLOCKED,
                action.risk,
                [
                    f"Incident already has {len(mutations)} proposed actions "
                    f"(limit {cfg.max_actions_per_incident}). Escalating instead."
                ],
            )

        # 7. Taint. A successful injection becomes a visible prompt, not an act.
        tainted: list[EvidenceRef] = [
            index[eid] for eid in action.evidence_ids if index[eid].tainted
        ]
        if tainted:
            detail = "; ".join(
                f"{e.id} ({', '.join(sorted(set(e.taint_reasons))[:3])})" for e in tainted
            )
            warnings.append(
                f"SECURITY: this action rests on telemetry the sanitiser flagged as "
                f"potentially adversarial - {detail}. Treat the content as a finding "
                f"to investigate, not as a reason to act."
            )

        # 8. Autonomy mode.
        if cfg.autonomy is AutonomyMode.RECOMMEND_ONLY:
            reasons.append("Autonomy is RECOMMEND_ONLY; a human performs the fix.")
            return PolicyDecision(
                Disposition.REQUIRE_APPROVAL, action.risk, reasons, warnings,
                [e.id for e in tainted],
            )

        if cfg.autonomy is AutonomyMode.APPROVE_THEN_EXECUTE:
            reasons.append("Every state-changing action requires a human approval.")
            return PolicyDecision(
                Disposition.REQUIRE_APPROVAL, action.risk, reasons, warnings,
                [e.id for e in tainted],
            )

        # TIERED
        if tainted and cfg.block_on_tainted_evidence:
            reasons.append(
                "Automatic execution is disabled because the supporting evidence "
                "is tainted."
            )
            return PolicyDecision(
                Disposition.REQUIRE_APPROVAL, action.risk, reasons, warnings,
                [e.id for e in tainted],
            )

        if _RISK_ORDER[action.risk] <= _RISK_ORDER[cfg.auto_execute_max_risk]:
            reasons.append(
                f"Risk {action.risk.value} is at or below the automatic threshold "
                f"({cfg.auto_execute_max_risk.value})."
            )
            return PolicyDecision(
                Disposition.AUTO_EXECUTE, action.risk, reasons, warnings, []
            )

        reasons.append(
            f"Risk {action.risk.value} exceeds the automatic threshold "
            f"({cfg.auto_execute_max_risk.value})."
        )
        return PolicyDecision(
            Disposition.REQUIRE_APPROVAL, action.risk, reasons, warnings,
            [e.id for e in tainted],
        )

    # -- approvals ---------------------------------------------------------

    def approval_deadline_seconds(self) -> int:
        return self.config.approval_timeout_seconds

    def is_expired(self, incident: Incident, action_id: str) -> bool:
        """An approval request that nobody answered must not execute later.

        Approving a container restart is a decision about the system as it was
        when the card was posted. Twenty minutes on, that system may be
        different and the action may no longer be appropriate.
        """
        entry = next(
            (
                e
                for e in reversed(incident.timeline)
                if e.event == "approval_requested" and e.data.get("action_id") == action_id
            ),
            None,
        )
        if entry is None:
            return False
        age = _now() - entry.at
        return age > timedelta(seconds=self.config.approval_timeout_seconds)

    def record_decision(
        self,
        incident: Incident,
        action_id: str,
        *,
        decision: ApprovalDecision,
        user_id: str,
        user_name: str = "",
        note: str = "",
        channel: str = "teams",
    ) -> Approval:
        """Record an approval, and enforce that it came from a human.

        `user_id` is supplied by the Teams gateway from the authenticated
        activity, never parsed out of message text - which is what keeps a log
        line claiming "the on-call engineer approved this" from becoming an
        approval.
        """
        if not user_id:
            raise ValueError(
                "An approval requires an authenticated user id. Approvals cannot "
                "originate from message content or telemetry."
            )
        if self.is_expired(incident, action_id) and decision is ApprovalDecision.APPROVED:
            decision = ApprovalDecision.EXPIRED
            note = (note + " " if note else "") + (
                f"[approval window of {self.config.approval_timeout_seconds}s had elapsed]"
            )

        approval = Approval(
            action_id=action_id,
            decision=decision,
            decided_by=user_id,
            decided_by_name=user_name,
            note=note,
            channel=channel,
        )
        incident.approvals.append(approval)
        incident.log(
            actor=f"user:{user_name or user_id}",
            event=f"approval_{decision.value}",
            detail=note,
            action_id=action_id,
            channel=channel,
        )
        return approval

    def may_execute(self, incident: Incident, action: ProposedAction) -> tuple[bool, str]:
        """Final check immediately before execution.

        Deliberately separate from `evaluate`: state can change between the
        card being posted and the button being pressed, and this is the last
        point at which we can refuse.
        """
        if action.read_only:
            return True, "read-only"

        decision = self.evaluate(action, incident=incident)
        if decision.blocked:
            return False, decision.reasons[0] if decision.reasons else "blocked by policy"

        if decision.disposition is Disposition.AUTO_EXECUTE:
            return True, "auto-execute permitted by policy"

        approval = incident.approval_for(action.id)
        if approval is None:
            return False, "no approval recorded for this action"
        if approval.decision is ApprovalDecision.REJECTED:
            return False, f"rejected by {approval.decided_by_name or approval.decided_by}"
        if approval.decision is ApprovalDecision.EXPIRED:
            return False, "approval expired before execution"
        if approval.decision is not ApprovalDecision.APPROVED:
            return False, f"approval state is {approval.decision.value}"

        return True, f"approved by {approval.decided_by_name or approval.decided_by}"
