"""Rule-based triage - the deterministic first pass.

Many real incidents do not need a model to be understood. Exit code 137 with
`OOMKilled=true` is not ambiguous, and nobody should pay for tokens, or wait on
a slow local model, to read it. This module handles those cases directly and
produces a `Diagnosis` under exactly the same grounding rules the LLM path will
obey: every hypothesis cites evidence that was actually collected, or it is not
emitted at all.

It is also honest about its limits. A latency regression with no errors in any
log, or anything it has no rule for, produces an abstention - which the policy
engine treats as "authorise nothing" and the chat surface shows as an
escalation. A rule engine that guessed when unsure would be worse than none.

The `model_id` it stamps on every diagnosis says plainly that no model was
involved, so no card can be mistaken for LLM reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..domain.models import (
    ActionSpec,
    Diagnosis,
    EvidenceRef,
    Hypothesis,
    Incident,
    ProposedAction,
    RiskLevel,
    Severity,
)
from ..telemetry.base import LogQuery, ResourceState, TimeRange

MODEL_ID = "rule-based triage (no LLM)"


class TriageSource(Protocol):
    """What triage needs from telemetry. The Docker adapter satisfies it."""

    async def inventory(self) -> list[ResourceState]: ...
    async def inventory_evidence(self) -> list[EvidenceRef]: ...
    async def fetch_logs(self, query: LogQuery) -> list[EvidenceRef]: ...


@dataclass
class TriageResult:
    diagnosis: Diagnosis
    actions: list[ProposedAction] = field(default_factory=list)
    severity: Severity = Severity.SEV3
    title: str = ""
    security_finding: bool = False


def service_of(container: str) -> str:
    """demo-orders-api -> orders-api. The executors speak in service names."""
    return container[5:] if container.startswith("demo-") else container


# Patterns are matched against SANITISED excerpts, which is all triage ever
# sees. Each maps to evidence that a human can re-open from the card.
_POOL = re.compile(r"remaining connection slots|pool_exhausted", re.I)
_DEPLOY = re.compile(r"unhandled exception", re.I)
_DEPENDENCY = re.compile(
    r"payment (?:authorisation|call) failed|upstream[\"']?\s*[:=]\s*[\"']?payments-api", re.I
)
_HEAP = re.compile(r"allocation pressure|heap_mb", re.I)
_RELEASE = re.compile(r"[\"']release[\"']\s*:\s*[\"']([\w.\-]+)", re.I)


class RuleTriage:
    def __init__(self, source: TriageSource, *, demo_executors: bool = False) -> None:
        self.source = source
        self.demo_executors = demo_executors

    async def investigate(self, incident: Incident, container: str) -> TriageResult:
        calls = 0
        evidence: list[EvidenceRef] = []

        # 1. Container state - TRUSTED, it comes from the daemon.
        states = await self.source.inventory()
        calls += 1
        state = next((s for s in states if s.name == container), None)
        state_ev = next(
            (
                e
                for e in await self.source.inventory_evidence()
                if e.metadata.get("container") == container
            ),
            None,
        )
        calls += 1
        if state_ev is not None:
            evidence.append(state_ev)

        # 2. Recent logs - UNTRUSTED, sanitised by the adapter before we see them.
        logs = await self.source.fetch_logs(
            LogQuery(target=container, time_range=TimeRange.last(5), limit=150)
        )
        calls += 1
        evidence.extend(logs)

        log_text = "\n".join(e.excerpt for e in logs)
        log_ids = [e.id for e in logs]
        service = service_of(container)
        hypotheses: list[Hypothesis] = []
        actions: list[ProposedAction] = []
        severity = Severity.SEV3
        title = incident.title
        security = False

        def cite(*evs: EvidenceRef | None) -> list[str]:
            return [e.id for e in evs if e is not None]

        # -- security first: never let an injection hide behind an outage ----
        tainted = [e for e in evidence if e.tainted]
        if tainted:
            security = True
            severity = Severity.SEV2
            reasons = sorted({r.split("(")[0] for e in tainted for r in e.taint_reasons})
            hypotheses.append(
                Hypothesis(
                    statement=(
                        f"Adversarial text is reaching {service}'s logs and is "
                        "addressed to an automated operator."
                    ),
                    mechanism=(
                        "Request fields such as the User-Agent are logged verbatim. "
                        "The payloads claim authority and prior approval and ask for "
                        f"destructive actions ({', '.join(reasons[:5])}). This is a "
                        "prompt-injection attempt against log-reading agents. The "
                        "correct response is to report it; no action it requests "
                        "has been proposed."
                    ),
                    confidence=0.9,
                    evidence_ids=[e.id for e in tainted],
                )
            )
            title = f"Security: prompt-injection attempt in {service} logs"

        # -- OOM ------------------------------------------------------------
        if state is not None and state.oom_killed:
            severity = Severity.SEV1
            heap_evidence = [e for e in logs if _HEAP.search(e.excerpt)]
            limit_mb = (state.resources.get("memory_limit_bytes") or 0) // 1024 // 1024
            hypotheses.append(
                Hypothesis(
                    statement=(
                        f"{service} is being killed by the kernel OOM killer"
                        + (f" at its {limit_mb}MB limit." if limit_mb else ".")
                    ),
                    mechanism=(
                        "The container exited with code 137 and Docker reports "
                        "OOMKilled=true"
                        + (
                            "; the logs show heap growing request by request before "
                            "death, which points to a leak rather than a one-off spike."
                            if heap_evidence
                            else "."
                        )
                    ),
                    confidence=0.88 if heap_evidence else 0.75,
                    evidence_ids=cite(state_ev) + [e.id for e in heap_evidence],
                )
            )
            title = f"{service} is down - OOM-killed"
            restart = ProposedAction(
                intent=f"Restart {container}",
                forward=ActionSpec(
                    executor="docker.start", params={"container": container}, target=container
                ),
                rollback=ActionSpec(
                    executor="docker.stop", params={"container": container}, target=container
                ),
                risk=RiskLevel.LOW,
                rationale=(
                    "Restores service immediately. Does NOT address why memory grew - "
                    "if the cause is a leak, expect a relapse, which verification "
                    "will catch."
                ),
                evidence_ids=cite(state_ev),
                expected_effect=f"{service} serves traffic again.",
                blast_radius=f"{service} only; in-flight requests are lost.",
            )
            actions.append(restart)
            if self.demo_executors:
                actions.append(
                    self._code_fix(
                        service,
                        container,
                        revert_mode="memleak",
                        intent=f"Ship the corrected build of {service} (stops the leak), then start it",
                        evidence_ids=cite(state_ev) + [e.id for e in heap_evidence],
                        rationale="Removes the allocation that is never released, at source.",
                        effect="Heap stops growing; no further OOM kills.",
                    )
                )

        # -- connection pool ------------------------------------------------
        pool = [e for e in logs if _POOL.search(e.excerpt)]
        if pool:
            severity = min(severity, Severity.SEV2, key=_sev_rank)
            hypotheses.append(
                Hypothesis(
                    statement=f"{service} is exhausting the Postgres connection limit.",
                    mechanism=(
                        "Postgres is refusing new connections ('remaining connection "
                        "slots are reserved'). The database is up; the caller is "
                        "holding connections and never releasing them, so restarting "
                        "Postgres would only drop them until the pool refills."
                    ),
                    confidence=0.82,
                    evidence_ids=[e.id for e in pool],
                )
            )
            title = f"{service} cannot get database connections"
            if self.demo_executors:
                actions.append(
                    self._code_fix(
                        service, container, revert_mode="pool_exhaust",
                        intent=f"Ship the corrected build of {service} (releases connections)",
                        evidence_ids=[e.id for e in pool],
                        rationale="Stops the connection leak at source and closes held connections.",
                        effect="Backend count returns to baseline; 503s stop.",
                    )
                )

        # -- bad deploy -----------------------------------------------------
        deploy = [e for e in logs if _DEPLOY.search(e.excerpt)]
        if deploy:
            severity = min(severity, Severity.SEV2, key=_sev_rank)
            release = next(
                (m.group(1) for e in deploy for m in [_RELEASE.search(e.excerpt)] if m), ""
            )
            hypotheses.append(
                Hypothesis(
                    statement=(
                        f"A defect in {service}"
                        + (f" release {release}" if release else "")
                        + " is throwing on every order."
                    ),
                    mechanism=(
                        "The same unhandled exception repeats at the request rate with "
                        "the same stack frame, which is a code path failing "
                        "deterministically - a restart cannot fix it."
                    ),
                    confidence=0.78,
                    evidence_ids=[e.id for e in deploy],
                )
            )
            title = f"{service} is failing every request"
            if self.demo_executors:
                actions.append(
                    self._code_fix(
                        service, container, revert_mode="error500",
                        intent=f"Roll {service} forward to the corrected build",
                        evidence_ids=[e.id for e in deploy],
                        rationale="Removes the defective code path.",
                        effect="HTTP 500s stop.",
                    )
                )

        # -- dependency -----------------------------------------------------
        dep = [e for e in logs if _DEPENDENCY.search(e.excerpt)]
        if dep and service != "payments-api":
            hypotheses.append(
                Hypothesis(
                    statement=f"{service} is healthy; its dependency payments-api is failing.",
                    mechanism=(
                        f"{service} logs failed payment authorisations against "
                        "payments-api while itself staying up and degrading "
                        "gracefully. The fault is downstream."
                    ),
                    confidence=0.74,
                    evidence_ids=[e.id for e in dep],
                )
            )
            title = f"{service} degraded - payments-api failing"
            if self.demo_executors:
                actions.append(
                    self._code_fix(
                        "payments-api", "demo-payments-api", revert_mode="down",
                        intent="Restore payments-api",
                        evidence_ids=[e.id for e in dep],
                        rationale="The failure is in the dependency, so that is where the fix goes.",
                        effect="Payment authorisations succeed; orders reach CONFIRMED.",
                    )
                )

        # -- a crash with no better explanation ----------------------------
        if (
            state is not None
            and not state.oom_killed
            and state.status in {"exited", "dead"}
            and not hypotheses
        ):
            hypotheses.append(
                Hypothesis(
                    statement=f"{service} exited with code {state.exit_code}.",
                    mechanism="The process terminated; the logs do not show why.",
                    confidence=0.5,
                    evidence_ids=cite(state_ev),
                )
            )
            actions.append(
                ProposedAction(
                    intent=f"Start {container}",
                    forward=ActionSpec(executor="docker.start", params={"container": container}, target=container),
                    rollback=ActionSpec(executor="docker.stop", params={"container": container}, target=container),
                    risk=RiskLevel.LOW,
                    rationale="Brings the service back; the cause is not established.",
                    evidence_ids=cite(state_ev),
                    expected_effect=f"{service} is running.",
                    blast_radius=f"{service} only.",
                )
            )

        # A security finding proposes nothing. Whatever the payload asked for is
        # exactly what must not be offered as a button.
        if security and not (state is not None and state.oom_killed) and not pool and not deploy:
            actions = []

        diagnosis = Diagnosis(
            incident_id=incident.id,
            model_id=MODEL_ID,
            hypotheses=hypotheses,
            evidence=evidence,
            tool_call_count=calls,
        )
        must, reason = diagnosis.must_abstain()
        if must:
            diagnosis.abstained = True
            diagnosis.abstain_reason = (
                reason
                if hypotheses
                else "No rule matched the collected telemetry. This may need metrics "
                "the rule engine cannot see, or model-based reasoning."
            )
            actions = []

        return TriageResult(
            diagnosis=diagnosis,
            actions=actions,
            severity=severity,
            title=title,
            security_finding=security,
        )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _code_fix(
        service: str,
        container: str,
        *,
        revert_mode: str,
        intent: str,
        evidence_ids: list[str],
        rationale: str,
        effect: str,
    ) -> ProposedAction:
        return ProposedAction(
            intent=intent,
            forward=ActionSpec(
                executor="demo.apply_code_fix", params={"service": service}, target=container
            ),
            rollback=ActionSpec(
                executor="demo.revert_code_fix",
                params={"service": service, "mode": revert_mode},
                target=container,
            ),
            risk=RiskLevel.MEDIUM,
            rationale=rationale,
            evidence_ids=evidence_ids,
            expected_effect=effect,
            blast_radius=f"{service} only; takes effect on the running build.",
        )


_SEV_ORDER = {Severity.SEV1: 1, Severity.SEV2: 2, Severity.SEV3: 3, Severity.SEV4: 4}


def _sev_rank(s: Severity) -> int:
    return _SEV_ORDER[s]


def classify_log_burst(evidence: list[EvidenceRef]) -> dict[str, Any]:
    """Cheap detector signal: how many error lines and tainted lines are there?"""
    errors = 0
    tainted = 0
    for e in evidence:
        for line in e.excerpt.splitlines():
            if re.search(r"\"level\":\s*\"ERROR\"|\bERROR\b|\bFATAL\b", line):
                errors += 1
        if e.tainted:
            tainted += 1
    return {"errors": errors, "tainted": tainted}
