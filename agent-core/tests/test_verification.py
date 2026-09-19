"""Verification verdict tests - the closed loop's decision logic.

`closed_loop_demo.py` proves the loop against the real stack, but it takes
minutes and needs Docker. These tests pin the same logic deterministically by
scripting each poll's outcome, so a regression in the verdict rules is caught
in under a second.

The case that matters most is `test_relapse_after_green_is_not_recovered`:
green, then red. Every surveyed vendor's "self-healing" stops at the green.
"""

from __future__ import annotations

from typing import Any

import pytest

from opsloop.domain.models import (
    ActionSpec,
    CheckOutcome,
    HealthCheck,
    Incident,
    IncidentState,
    ProposedAction,
    VerificationPlan,
    VerificationVerdict,
)
from opsloop.remediate.executors import ExecutorRegistry
from opsloop.telemetry.base import AdapterRegistry
from opsloop.verify.loop import PollRound, Verifier, plan_for_container_service

CHECK_A = HealthCheck(id="chk_a", name="process up", kind="process", params={}, success_criteria="running")
CHECK_B = HealthCheck(id="chk_b", name="http 200", kind="http", params={}, success_criteria="200")


class ScriptedVerifier(Verifier):
    """A Verifier whose polls follow a script instead of touching the world.

    `script` is one entry per poll: a bool (all checks pass/fail) or an
    explicit per-check mapping.
    """

    def __init__(self, script: list[Any]) -> None:
        super().__init__(AdapterRegistry(), ExecutorRegistry())
        self.script = list(script)
        self.calls = 0

    async def run_checks(self, plan: VerificationPlan) -> PollRound:
        index = min(self.calls, len(self.script) - 1)
        step = self.script[index]
        self.calls += 1
        round_ = PollRound(at=float(self.calls))
        for check in plan.checks:
            passed = step[check.id] if isinstance(step, dict) else bool(step)
            round_.outcomes.append(
                CheckOutcome(
                    check_id=check.id,
                    passed=passed,
                    observed="scripted",
                )
            )
        return round_


def plan(**kwargs: Any) -> VerificationPlan:
    """A plan bounded by max_polls rather than by wall-clock time.

    Every test below fixes the exact number of polls, so verdicts are
    deterministic and the suite stays fast. A plan with a zero interval and no
    ceiling is refused outright - see test_unbounded_polling_is_refused.
    """
    return VerificationPlan(
        checks=[CHECK_A, CHECK_B],
        window_seconds=kwargs.pop("window_seconds", 60),
        poll_interval_seconds=kwargs.pop("poll_interval_seconds", 0),
        required_consecutive_passes=kwargs.pop("required_consecutive_passes", 2),
        max_polls=kwargs.pop("max_polls", 4),
        **kwargs,
    )


class TestVerdicts:
    async def test_sustained_health_is_recovered(self) -> None:
        v = ScriptedVerifier([True])
        report = await v.verify(plan(max_polls=4))
        assert report.verdict is VerificationVerdict.RECOVERED
        assert "held continuously" in report.summary
        await v.close()

    async def test_never_healthy_is_not_recovered(self) -> None:
        v = ScriptedVerifier([False])
        report = await v.verify(plan(max_polls=4))
        assert report.verdict is VerificationVerdict.NOT_RECOVERED
        assert "never reached a healthy state" in report.summary
        await v.close()

    async def test_relapse_after_green_is_not_recovered(self) -> None:
        """THE decisive case: healthy, then broken again.

        A one-shot check sees the green poll and reports success. This is the
        decoy-remediation signature, and catching it is the whole point.
        """
        v = ScriptedVerifier([True, True, False, False, False, False])
        report = await v.verify(plan(max_polls=6))
        assert report.verdict is VerificationVerdict.NOT_RECOVERED
        assert "relapsed" in report.summary
        assert "suppresses the symptom" in report.summary
        await v.close()

    async def test_single_green_poll_is_not_enough(self) -> None:
        """One green poll is a coincidence, not a recovery."""
        # Exactly three polls: fail, fail, pass. The single green poll at the
        # end is a one-poll streak against a requirement of three.
        p = plan(required_consecutive_passes=3, max_polls=3)
        v = ScriptedVerifier([False, False, True])
        report = await v.verify(p)
        assert report.verdict is VerificationVerdict.NOT_RECOVERED
        assert "consecutive poll" in report.summary
        await v.close()

    async def test_regression_against_baseline_is_detected(self) -> None:
        """A check that passed BEFORE the fix and fails after means we caused harm."""
        p = plan(max_polls=4)
        p.baseline = {"passing_at_baseline": ["chk_a", "chk_b"]}
        v = ScriptedVerifier([{"chk_a": False, "chk_b": True}])
        report = await v.verify(p)
        assert report.verdict is VerificationVerdict.REGRESSED
        assert "made things worse" in report.summary
        await v.close()

    async def test_regression_does_not_wait_out_the_window(self) -> None:
        """Harm must be undone in seconds, not after the full window."""
        p = plan(window_seconds=60, max_polls=50)
        p.baseline = {"passing_at_baseline": ["chk_a"]}
        v = ScriptedVerifier([False])
        report = await v.verify(p)
        assert report.verdict is VerificationVerdict.REGRESSED
        assert v.calls == 1  # returned on the very first poll
        await v.close()

    async def test_preexisting_failure_is_not_a_regression(self) -> None:
        """Still broken is not the same as broken by us."""
        p = plan(max_polls=3)
        p.baseline = {"passing_at_baseline": []}  # everything already failing
        v = ScriptedVerifier([False])
        report = await v.verify(p)
        assert report.verdict is VerificationVerdict.NOT_RECOVERED
        await v.close()

    async def test_no_checks_is_inconclusive_not_success(self) -> None:
        """An unmeasurable fix is not a verified fix."""
        v = ScriptedVerifier([True])
        report = await v.verify(VerificationPlan(checks=[], max_polls=1))
        assert report.verdict is VerificationVerdict.INCONCLUSIVE
        assert "could not be measured" in report.summary
        await v.close()


class TestRollback:
    def _action(self, *, with_rollback: bool = True) -> ProposedAction:
        return ProposedAction(
            intent="Restart orders-api",
            forward=ActionSpec(executor="noop", params={"reason": "forward"}, target="x"),
            rollback=ActionSpec(executor="noop", params={"reason": "inverse"}, target="x")
            if with_rollback
            else None,
            read_only=not with_rollback,
            evidence_ids=["ev_1"],
        )

    async def test_failed_verification_triggers_rollback(self) -> None:
        v = ScriptedVerifier([False])
        inc = Incident(title="t")
        action = self._action()
        report = await v.verify_and_maybe_rollback(inc, action, plan(max_polls=2))
        assert report.verdict is VerificationVerdict.NOT_RECOVERED
        assert report.rollback_triggered is True
        assert inc.state is IncidentState.ROLLED_BACK
        assert any(e.was_rollback for e in inc.executions)
        await v.close()

    async def test_successful_verification_resolves_without_rollback(self) -> None:
        v = ScriptedVerifier([True])
        inc = Incident(title="t")
        report = await v.verify_and_maybe_rollback(inc, self._action(), plan())
        assert report.verdict is VerificationVerdict.RECOVERED
        assert report.rollback_triggered is False
        assert inc.state is IncidentState.RESOLVED
        assert inc.closed_at is not None
        await v.close()

    async def test_inconclusive_escalates_rather_than_claiming_success(self) -> None:
        v = ScriptedVerifier([True])
        inc = Incident(title="t")
        report = await v.verify_and_maybe_rollback(
            inc, self._action(), VerificationPlan(checks=[], max_polls=1)
        )
        assert report.verdict is VerificationVerdict.INCONCLUSIVE
        assert inc.state is IncidentState.ESCALATED
        await v.close()

    async def test_missing_baseline_is_recorded_as_a_warning(self) -> None:
        v = ScriptedVerifier([True])
        inc = Incident(title="t")
        await v.verify_and_maybe_rollback(inc, self._action(), plan())
        assert any(e.event == "verification_warning" for e in inc.timeline)
        await v.close()


class TestPlanSafety:
    def test_unbounded_polling_is_refused(self) -> None:
        """A health check must never become part of the incident.

        Verification runs against a system that is already unwell. A zero
        interval with no ceiling would poll it as fast as the event loop
        allows, adding load at the worst possible moment.
        """
        with pytest.raises(ValueError, match="busy loop"):
            VerificationPlan(checks=[CHECK_A], poll_interval_seconds=0, max_polls=None)

    def test_normal_plan_needs_no_ceiling(self) -> None:
        p = VerificationPlan(checks=[CHECK_A], poll_interval_seconds=15)
        assert p.max_polls is None

    async def test_max_polls_bounds_the_loop(self) -> None:
        v = ScriptedVerifier([True])
        await v.verify(plan(window_seconds=3600, max_polls=5))
        assert v.calls == 5
        await v.close()


class TestPlanBuilder:
    def test_default_plan_covers_four_failure_modes(self) -> None:
        p = plan_for_container_service(
            container="demo-orders-api",
            health_url="http://localhost:18081/health",
            prometheus_url="http://localhost:19090",
            latency_query="orders_latency_ms",
        )
        kinds = {c.kind for c in p.checks}
        assert kinds == {"process", "http", "log_absence", "metric"}

    def test_metric_check_omitted_without_prometheus(self) -> None:
        p = plan_for_container_service(
            container="x", health_url="http://localhost/health"
        )
        assert "metric" not in {c.kind for c in p.checks}

    def test_process_check_forbids_oom_and_caps_restarts(self) -> None:
        p = plan_for_container_service(container="x", health_url="http://localhost/h")
        process = next(c for c in p.checks if c.kind == "process")
        assert process.params["forbid_oom"] is True
        assert process.params["max_restarts"] == 3

    def test_every_check_states_its_own_pass_condition(self) -> None:
        """'Looks fine' is not a success criterion."""
        p = plan_for_container_service(container="x", health_url="http://localhost/h")
        assert all(c.success_criteria.strip() for c in p.checks)
