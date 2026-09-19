"""End-to-end proof of the closed loop.

Runs the same incident twice against the real demo stack:

  ROUND A  the DECOY fix - restart the container.
           Every error clears instantly. A system that checks "did the errors
           stop?" declares victory here. The leak is untouched, so the service
           OOMs again within seconds. Expected verdict: NOT_RECOVERED,
           followed by an automatic rollback.

  ROUND B  the CORRECT fix - remove the defect, then restart.
           Expected verdict: RECOVERED.

Round A is the claim. Every surveyed vendor stops at "action taken" and would
report Round A as a success.

The diagnosis is hand-constructed here so the loop can be proven independently
of any model. That separation is deliberate - it shows the guarantee comes
from the verification machinery, not from the LLM being clever.

    cd agent-core
    .venv/Scripts/python.exe scripts/closed_loop_demo.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from opsloop.domain.models import (  # noqa: E402
    ActionSpec,
    Diagnosis,
    Hypothesis,
    Incident,
    IncidentState,
    ProposedAction,
    RiskLevel,
    Severity,
    Signal,
    VerificationVerdict,
)
from opsloop.policy.engine import (  # noqa: E402
    ApprovalDecision,
    AutonomyMode,
    PolicyConfig,
    PolicyEngine,
)
from opsloop.remediate.executors import (  # noqa: E402
    ExecutorRegistry,
    register_demo_executors,
    register_docker_executors,
)
from opsloop.telemetry.base import AdapterRegistry, LogQuery, TimeRange  # noqa: E402
from opsloop.telemetry.docker_adapter import DockerAdapter  # noqa: E402
from opsloop.verify.loop import Verifier, plan_for_container_service  # noqa: E402

ORDERS = "http://localhost:18081"
CONTAINER = "demo-orders-api"

# The leak is ~12MB/request and loadgen runs at 2 rps, so a 256MB limit is
# reached in roughly 11 seconds. The window must be comfortably longer than
# that or the relapse happens after we stop looking - which is exactly the
# mistake this whole module exists to prevent.
WINDOW_SECONDS = 70
POLL_SECONDS = 7


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(f"  {title}")
        print("=" * 78)


async def set_fault(mode: str) -> None:
    async with httpx.AsyncClient(timeout=10) as http:
        await http.post(f"{ORDERS}/admin/fault", json={"mode": mode, "intensity": 1.0})


async def wait_for_oom(adapter: DockerAdapter, timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for state in await adapter.inventory():
            if state.name == CONTAINER and state.oom_killed:
                print(f"  [OK] {state.summarise()}")
                return True
        await asyncio.sleep(3)
    return False


async def build_incident(adapter: DockerAdapter) -> Incident:
    """Collect real evidence and attach a diagnosis that cites it."""
    incident = Incident(
        title="orders-api is down - repeated OOM kills",
        severity=Severity.SEV1,
        service="orders-api",
        environment="production",
        state=IncidentState.TRIAGING,
    )
    incident.signals.append(
        Signal(
            detector="docker-inventory",
            title="demo-orders-api exited with code 137",
            service="orders-api",
            severity=Severity.SEV1,
        )
    )

    evidence = await adapter.inventory_evidence()
    evidence = [e for e in evidence if e.metadata.get("container") == CONTAINER]
    logs = await adapter.fetch_logs(
        LogQuery(target=CONTAINER, time_range=TimeRange.last(10), limit=60)
    )
    evidence.extend(logs)

    print(f"  collected {len(evidence)} piece(s) of evidence:")
    for e in evidence:
        flag = "  [TAINTED]" if e.tainted else ""
        print(f"    {e.id}  {e.source_kind:<8} {e.query[:58]}{flag}")

    diagnosis = Diagnosis(
        incident_id=incident.id,
        model_id="hand-constructed (loop proof, no model involved)",
        evidence=evidence,
        tool_call_count=len(evidence),
        hypotheses=[
            Hypothesis(
                statement=(
                    "orders-api leaks heap on every request and is OOM-killed at "
                    "its 256MB container limit."
                ),
                mechanism=(
                    "Each order allocates ~12MB that is never released. Under "
                    "continuous load the process crosses the cgroup limit and the "
                    "kernel OOM killer reaps it, so the container exits with 137."
                ),
                confidence=0.86,
                evidence_ids=[e.id for e in evidence],
            )
        ],
    )
    incident.diagnosis = diagnosis
    incident.state = IncidentState.DIAGNOSED

    must, reason = diagnosis.must_abstain()
    print(f"\n  evidence-grounding gate: abstain={must} ({reason or 'passed'})")
    return incident


async def run_round(
    *,
    label: str,
    action: ProposedAction,
    incident: Incident,
    policy: PolicyEngine,
    executors: ExecutorRegistry,
    verifier: Verifier,
    expect: VerificationVerdict,
) -> bool:
    rule(label)

    # 1. Policy
    decision = policy.evaluate(action, incident=incident)
    print("POLICY\n" + decision.explain())
    if decision.blocked:
        print("  -> blocked; nothing executed.")
        return False

    # 2. Approval - supplied by an authenticated user, never by telemetry.
    if decision.needs_human:
        incident.state = IncidentState.AWAITING_APPROVAL
        incident.log(actor="agent", event="approval_requested", action_id=action.id)
        print(f"\nAPPROVAL CARD -> Teams")
        print(f"  intent    {action.intent}")
        print(f"  forward   {action.forward.describe()}")
        print(f"  rollback  {action.rollback.describe() if action.rollback else '(none)'}")
        print(f"  risk      {action.risk.value}")
        print(f"  expected  {action.expected_effect}")
        policy.record_decision(
            incident,
            action.id,
            decision=ApprovalDecision.APPROVED,
            user_id="aad|demo-operator-001",
            user_name="oncall-engineer",
            channel="local-demo",
        )
        print("  <- Approved by oncall-engineer")

    allowed, why = policy.may_execute(incident, action)
    print(f"\n  pre-execution check: {allowed} ({why})")
    if not allowed:
        return False

    # 3. Baseline BEFORE the fix - without it, REGRESSED is undetectable.
    plan = plan_for_container_service(
        container=CONTAINER,
        health_url=f"{ORDERS}/health",
        window_seconds=WINDOW_SECONDS,
        poll_interval_seconds=POLL_SECONDS,
        required_consecutive_passes=3,
    )
    baseline = await verifier.capture_baseline(plan)
    print(
        f"\nBASELINE  passing={baseline['passing_at_baseline']}  "
        f"failing={baseline['failing_at_baseline']}"
    )

    # 4. Execute
    incident.state = IncidentState.REMEDIATING
    print(f"\nEXECUTING {action.forward.describe()}")
    result = await executors.execute(action.forward, action_id=action.id)
    incident.executions.append(result)
    print(f"  succeeded={result.succeeded}  {result.stdout or result.error}")
    if not result.succeeded:
        return False

    # 5. Verify
    print(f"\nVERIFYING  window={WINDOW_SECONDS}s poll={POLL_SECONDS}s "
          f"required_consecutive_passes=3")

    def on_poll(n: int, round_: object) -> None:
        marks = "".join("." if o.passed else "X" for o in round_.outcomes)  # type: ignore[attr-defined]
        status = "ALL PASS" if round_.all_passed else "fail"  # type: ignore[attr-defined]
        failed = ""
        if not round_.all_passed:  # type: ignore[attr-defined]
            names = {c.id: c.name for c in plan.checks}
            failed = "  <- " + "; ".join(
                names.get(o.check_id, o.check_id)
                for o in round_.outcomes  # type: ignore[attr-defined]
                if not o.passed
            )
        print(f"  poll {n:>2}  [{marks}]  {status}{failed}")

    report = await verifier.verify_and_maybe_rollback(
        incident, action, plan, on_poll=on_poll
    )

    rule()
    print(f"VERDICT   {report.verdict.value.upper()}")
    print(f"SUMMARY   {report.summary}")
    print(f"INCIDENT  state={incident.state.value}  rollback_triggered="
          f"{report.rollback_triggered}")

    ok = report.verdict is expect
    print(f"\nEXPECTED  {expect.value}  ->  {'PASS' if ok else 'FAIL'}")
    return ok


async def main() -> int:
    adapters = AdapterRegistry()
    docker_adapter = DockerAdapter("demo-stack")
    adapters.add(docker_adapter)

    executors = ExecutorRegistry()
    register_docker_executors(executors)
    register_demo_executors(executors)

    verifier = Verifier(adapters, executors)
    policy = PolicyEngine(
        PolicyConfig(autonomy=AutonomyMode.APPROVE_THEN_EXECUTE, max_actions_per_incident=6)
    )

    results: dict[str, bool] = {}
    try:
        rule("SETUP - injecting the memory leak into production")
        await set_fault("none")
        async with httpx.AsyncClient(timeout=10) as http:
            await http.post(f"{ORDERS}/admin/fault", json={"mode": "memleak", "intensity": 1.0})
        print("  fault mode -> memleak (persisted in Postgres; survives a restart)")
        print("  waiting for the kernel OOM killer ...")
        if not await wait_for_oom(docker_adapter):
            print("  [!!] no OOM within 90s - is the stack running with loadgen?")
            return 1

        rule("TRIAGE - collecting evidence")
        incident = await build_incident(docker_adapter)

        # ---------------- ROUND A: the decoy ----------------
        decoy = ProposedAction(
            intent="Restart the OOM-killed orders-api container",
            forward=ActionSpec(
                executor="docker.start", params={"container": CONTAINER}, target=CONTAINER
            ),
            rollback=ActionSpec(
                executor="docker.stop", params={"container": CONTAINER}, target=CONTAINER
            ),
            risk=RiskLevel.LOW,
            rationale="The container is exited; starting it restores service immediately.",
            evidence_ids=[e.id for e in incident.diagnosis.evidence],
            expected_effect="orders-api serves traffic again and errors stop.",
            blast_radius="orders-api only; in-flight requests are lost.",
        )
        incident.proposed_actions.append(decoy)
        results["A (decoy: restart only)"] = await run_round(
            label="ROUND A - DECOY FIX: restart the container and nothing else",
            action=decoy,
            incident=incident,
            policy=policy,
            executors=executors,
            verifier=verifier,
            expect=VerificationVerdict.NOT_RECOVERED,
        )

        # ---------------- ROUND B: the real fix ----------------
        print("\n  (resetting for round B)")
        incident.state = IncidentState.DIAGNOSED
        incident.closed_at = None

        real = ProposedAction(
            intent="Ship the corrected build that stops the leak, then restart",
            forward=ActionSpec(
                executor="demo.apply_code_fix",
                params={"service": "orders-api"},
                target=CONTAINER,
            ),
            rollback=ActionSpec(
                executor="demo.revert_code_fix",
                params={"service": "orders-api", "mode": "memleak"},
                target=CONTAINER,
            ),
            risk=RiskLevel.MEDIUM,
            rationale="Removes the allocation that is never released, at source.",
            evidence_ids=[e.id for e in incident.diagnosis.evidence],
            expected_effect="Heap stops growing; no further OOM kills.",
            blast_radius="orders-api only; requires a restart to take effect.",
        )
        incident.proposed_actions.append(real)

        # The container is stopped after round A's rollback; bring it back so
        # the corrected build is actually running.
        state = next(
            (s for s in await docker_adapter.inventory() if s.name == CONTAINER), None
        )
        if state and state.status != "running":
            print(f"  container is {state.status}; starting it for round B")
            await executors.execute(
                ActionSpec(
                    executor="docker.start", params={"container": CONTAINER}, target=CONTAINER
                ),
                action_id="setup",
            )
            await asyncio.sleep(6)

        results["B (correct: fix at source)"] = await run_round(
            label="ROUND B - CORRECT FIX: remove the defect at source",
            action=real,
            incident=incident,
            policy=policy,
            executors=executors,
            verifier=verifier,
            expect=VerificationVerdict.RECOVERED,
        )

        # ---------------- summary ----------------
        rule("RESULT")
        for name, ok in results.items():
            print(f"  {'PASS' if ok else 'FAIL'}  round {name}")
        print()
        if all(results.values()):
            print("  The loop distinguished a fix that works from one that only looks")
            print("  like it works. No surveyed vendor documents a guarantee of")
            print("  that distinction.")
        print()
        return 0 if all(results.values()) else 1

    finally:
        await set_fault("none")
        await verifier.close()
        await adapters.close_all()
        print("  (cleanup: faults cleared)")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
