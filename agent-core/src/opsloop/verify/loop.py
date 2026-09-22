"""Verification loop - closing the loop on a remediation.

Every surveyed commercial product stops at "action taken". This module answers
the next question: *did it actually work?* - and rolls back when the answer is
no.

The rule that does the work
---------------------------
A fix is RECOVERED only if health, once achieved, HOLDS CONTINUOUSLY to the end
of the verification window. Not "the errors stopped". Not "one poll was green".

That distinction is the whole point. Restarting an OOM-killed container clears
every error instantly and relapses about thirty seconds later. Restarting
Postgres to clear a leaked connection pool works until the client refills it.
Both look like successes to a system that checks once, immediately. Requiring
health to hold across a window is what separates a fix from a symptom
suppressant - and `decoy_remediation` in the demo scenarios exists precisely so
this can be measured rather than asserted.

Four verdicts
-------------
  RECOVERED     health achieved and held to the end of the window
  NOT_RECOVERED never became healthy, or relapsed after becoming healthy
  REGRESSED     a check that passed BEFORE the fix now fails - we made it
                worse. Does not wait for the window; rolls back immediately.
  INCONCLUSIVE  nothing could be measured. Escalates to a human rather than
                claiming success, because an unmeasurable fix is not a fix.

Why a baseline is captured first
--------------------------------
Without a pre-fix reading there is no way to tell "still broken" from "broken
differently because of what we just did". REGRESSED is only detectable against
a baseline, and it is the verdict that matters most - a fix that makes things
worse must be undone in seconds, not after a five-minute window elapses.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..domain.models import (
    CheckOutcome,
    ExecutionResult,
    HealthCheck,
    Incident,
    IncidentState,
    ProposedAction,
    VerificationPlan,
    VerificationReport,
    VerificationVerdict,
    _now,
)
from ..remediate.executors import ExecutorRegistry
from ..telemetry.base import AdapterRegistry, LogQuery, TimeRange, utcnow

log = logging.getLogger("opsloop.verify")

def _bounded_range(window_minutes: int, not_before: datetime | None) -> TimeRange:
    """The last `window_minutes`, but never earlier than `not_before`.

    Without the bound, a log check run straight after a fix reads the errors
    that caused the incident - still inside the window - and fails a fix that
    worked. That would roll back correct remediations for every failure that
    logs at ERROR level.
    """
    tr = TimeRange.last(window_minutes)
    if not_before is not None and not_before > tr.start:
        tr = TimeRange(start=not_before, end=utcnow())
    return tr


_OPERATORS = {
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
}


@dataclass
class CheckContext:
    adapters: AdapterRegistry
    http: httpx.AsyncClient
    timeout: float = 10.0
    #: Ignore telemetry older than this. Set to the moment verification began,
    #: so evidence of the failure that was just fixed cannot fail the check
    #: that is meant to confirm the fix.
    not_before: datetime | None = None


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


async def _check_http(check: HealthCheck, ctx: CheckContext) -> CheckOutcome:
    p = check.params
    url = p["url"]
    expect_status = int(p.get("expect_status", 200))
    max_latency_ms = p.get("max_latency_ms")

    started = time.perf_counter()
    try:
        r = await ctx.http.get(url, timeout=p.get("timeout", ctx.timeout))
    except httpx.HTTPError as exc:
        return CheckOutcome(
            check_id=check.id,
            passed=False,
            observed="unreachable",
            detail=f"{exc.__class__.__name__}: {exc}",
        )
    elapsed_ms = (time.perf_counter() - started) * 1000

    failures: list[str] = []
    if r.status_code != expect_status:
        failures.append(f"status {r.status_code} != {expect_status}")
    if max_latency_ms is not None and elapsed_ms > float(max_latency_ms):
        failures.append(f"latency {elapsed_ms:.0f}ms > {max_latency_ms}ms")

    json_path = p.get("expect_json_path")
    observed_value: Any = None
    if json_path:
        try:
            body = r.json()
            observed_value = body
            for part in str(json_path).split("."):
                observed_value = observed_value[part] if part else observed_value
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(f"json path {json_path!r} not readable ({exc.__class__.__name__})")
        else:
            expected = p.get("expect_json_value")
            if expected is not None and observed_value != expected:
                failures.append(f"{json_path}={observed_value!r} != {expected!r}")

    return CheckOutcome(
        check_id=check.id,
        passed=not failures,
        observed=(
            f"HTTP {r.status_code} in {elapsed_ms:.0f}ms"
            + (f", {json_path}={observed_value!r}" if json_path else "")
        ),
        detail="; ".join(failures),
    )


async def _check_metric(check: HealthCheck, ctx: CheckContext) -> CheckOutcome:
    """Instant PromQL query compared against a threshold."""
    p = check.params
    url = p["prometheus_url"].rstrip("/")
    query = p["query"]
    op = _OPERATORS.get(p.get("operator", "lt"))
    threshold = float(p["threshold"])
    if op is None:
        return CheckOutcome(
            check_id=check.id,
            passed=False,
            observed="invalid",
            detail=f"unknown operator {p.get('operator')!r}",
        )

    try:
        r = await ctx.http.get(
            f"{url}/api/v1/query", params={"query": query}, timeout=ctx.timeout
        )
        r.raise_for_status()
        result = (r.json().get("data") or {}).get("result") or []
    except (httpx.HTTPError, ValueError) as exc:
        return CheckOutcome(
            check_id=check.id, passed=False, observed="query failed", detail=str(exc)
        )

    if not result:
        # No series is not the same as a good value. A metric that vanished
        # because the process died must never read as healthy.
        return CheckOutcome(
            check_id=check.id,
            passed=False,
            observed="no data",
            detail=f"query returned no series: {query}",
        )

    try:
        value = float(result[0]["value"][1])
    except (KeyError, IndexError, ValueError) as exc:
        return CheckOutcome(
            check_id=check.id, passed=False, observed="unparseable", detail=str(exc)
        )

    passed = bool(op(value, threshold))
    return CheckOutcome(
        check_id=check.id,
        passed=passed,
        observed=f"{value:g}",
        detail="" if passed else f"{value:g} fails {p.get('operator', 'lt')} {threshold:g}",
    )


async def _check_log_absence(check: HealthCheck, ctx: CheckContext) -> CheckOutcome:
    """Pass when a symptom pattern is ABSENT from recent logs.

    The most direct statement of "the thing that was happening has stopped".
    """
    p = check.params
    pattern = re.compile(p["pattern"], re.IGNORECASE)
    window_minutes = int(p.get("window_minutes", 2))
    max_occurrences = int(p.get("max_occurrences", 0))

    adapter_name = p.get("adapter")
    adapters = (
        [ctx.adapters.get(adapter_name)] if adapter_name else ctx.adapters.with_logs()
    )

    total = 0
    samples: list[str] = []
    for adapter in adapters:
        try:
            evidence = await adapter.fetch_logs(
                LogQuery(
                    target=p.get("target", ""),
                    time_range=_bounded_range(window_minutes, ctx.not_before),
                    limit=int(p.get("limit", 300)),
                )
            )
        except Exception as exc:  # noqa: BLE001
            return CheckOutcome(
                check_id=check.id,
                passed=False,
                observed="log query failed",
                detail=f"{adapter.name}: {exc}",
            )
        for ev in evidence:
            for line in ev.excerpt.splitlines():
                if pattern.search(line):
                    total += 1
                    if len(samples) < 3:
                        samples.append(line.strip()[:160])

    passed = total <= max_occurrences
    return CheckOutcome(
        check_id=check.id,
        passed=passed,
        observed=f"{total} match(es) in the last {window_minutes}m",
        detail="" if passed else " | ".join(samples),
    )


async def _check_process(check: HealthCheck, ctx: CheckContext) -> CheckOutcome:
    """Is the thing running, and did it stay running?

    `forbid_oom` and the restart-count ceiling matter: a container that is
    'running' because it just restarted for the fifth time is not healthy, and
    a status check alone would call it so.
    """
    p = check.params
    name = p["name"]
    expect_status = p.get("expect_status", "running")
    forbid_oom = bool(p.get("forbid_oom", True))
    max_restarts = p.get("max_restarts")

    adapter_name = p.get("adapter")
    adapters = (
        [ctx.adapters.get(adapter_name)] if adapter_name else ctx.adapters.with_inventory()
    )

    for adapter in adapters:
        try:
            states = await adapter.inventory()
        except Exception as exc:  # noqa: BLE001
            return CheckOutcome(
                check_id=check.id,
                passed=False,
                observed="inventory failed",
                detail=f"{adapter.name}: {exc}",
            )
        for state in states:
            if state.name != name:
                continue
            failures: list[str] = []
            if state.status != expect_status:
                failures.append(f"status {state.status!r} != {expect_status!r}")
            if forbid_oom and state.oom_killed:
                failures.append("OOMKilled=true")
            if state.healthy is False:
                failures.append("healthcheck reports unhealthy")
            if max_restarts is not None and state.restart_count > int(max_restarts):
                failures.append(f"restarts {state.restart_count} > {max_restarts}")
            return CheckOutcome(
                check_id=check.id,
                passed=not failures,
                observed=state.summarise(),
                detail="; ".join(failures),
            )

    return CheckOutcome(
        check_id=check.id,
        passed=False,
        observed="not found",
        detail=f"no resource named {name!r} in any inventory",
    )


async def _check_command(check: HealthCheck, ctx: CheckContext) -> CheckOutcome:
    """Read-only command on a remote host via an SSH adapter."""
    p = check.params
    adapter = ctx.adapters.get(p["adapter"])
    if not hasattr(adapter, "run"):
        return CheckOutcome(
            check_id=check.id,
            passed=False,
            observed="unsupported",
            detail=f"adapter {adapter.name!r} cannot run commands",
        )
    expect_exit = int(p.get("expect_exit_code", 0))
    contains = p.get("stdout_contains")
    try:
        result = await adapter.run(p["command"])
    except Exception as exc:  # noqa: BLE001
        return CheckOutcome(
            check_id=check.id, passed=False, observed="command failed", detail=str(exc)
        )
    failures: list[str] = []
    if result.exit_code != expect_exit:
        failures.append(f"exit {result.exit_code} != {expect_exit}")
    if contains and contains.lower() not in result.stdout.lower():
        failures.append(f"stdout does not contain {contains!r}")
    return CheckOutcome(
        check_id=check.id,
        passed=not failures,
        observed=f"exit {result.exit_code}, {len(result.stdout)} bytes",
        detail="; ".join(failures) or result.stdout.strip()[:160],
    )


_CHECKERS = {
    "http": _check_http,
    "metric": _check_metric,
    "log_absence": _check_log_absence,
    "process": _check_process,
    "command": _check_command,
}


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


@dataclass
class PollRound:
    at: float
    outcomes: list[CheckOutcome] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return bool(self.outcomes) and all(o.passed for o in self.outcomes)

    @property
    def failed_ids(self) -> list[str]:
        return [o.check_id for o in self.outcomes if not o.passed]


class Verifier:
    def __init__(
        self,
        adapters: AdapterRegistry,
        executors: ExecutorRegistry,
        *,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.adapters = adapters
        self.executors = executors
        self._http = http or httpx.AsyncClient(timeout=15.0)
        self._owns_http = http is None

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    # -- one round ---------------------------------------------------------

    async def run_checks(
        self, plan: VerificationPlan, *, not_before: datetime | None = None
    ) -> PollRound:
        ctx = CheckContext(adapters=self.adapters, http=self._http, not_before=not_before)
        round_ = PollRound(at=time.time())

        async def run_one(check: HealthCheck) -> CheckOutcome:
            checker = _CHECKERS.get(check.kind)
            if checker is None:
                return CheckOutcome(
                    check_id=check.id,
                    passed=False,
                    observed="unsupported",
                    detail=f"no checker for kind {check.kind!r}",
                )
            try:
                return await checker(check, ctx)
            except Exception as exc:  # noqa: BLE001 - a broken check must not
                # abort verification; it fails, loudly, and the loop continues.
                return CheckOutcome(
                    check_id=check.id,
                    passed=False,
                    observed="check errored",
                    detail=f"{exc.__class__.__name__}: {exc}",
                )

        round_.outcomes = list(
            await asyncio.gather(*(run_one(c) for c in plan.checks))
        )
        return round_

    # -- baseline ----------------------------------------------------------

    async def capture_baseline(self, plan: VerificationPlan) -> dict[str, Any]:
        """Read every check BEFORE remediation.

        This is what makes REGRESSED detectable. A check that failed at
        baseline and still fails means we did not help; a check that PASSED at
        baseline and now fails means we caused harm, which is a different and
        more urgent situation.
        """
        round_ = await self.run_checks(plan)
        baseline = {
            "captured_at": _now().isoformat(),
            "checks": {
                o.check_id: {"passed": o.passed, "observed": o.observed} for o in round_.outcomes
            },
            "passing_at_baseline": [o.check_id for o in round_.outcomes if o.passed],
            "failing_at_baseline": [o.check_id for o in round_.outcomes if not o.passed],
        }
        plan.baseline = baseline
        return baseline

    # -- full verification -------------------------------------------------

    async def verify(
        self,
        plan: VerificationPlan,
        *,
        incident: Incident | None = None,
        on_poll: Any = None,
    ) -> VerificationReport:
        report = VerificationReport(plan=plan)

        if not plan.checks:
            report.verdict = VerificationVerdict.INCONCLUSIVE
            report.summary = (
                "No health checks were defined, so recovery could not be measured. "
                "An unmeasurable fix is not a verified fix."
            )
            report.finished_at = _now()
            return report

        passed_at_baseline = set(plan.baseline.get("passing_at_baseline", []))
        verification_began = utcnow()
        deadline = time.time() + plan.window_seconds
        rounds: list[PollRound] = []
        first_green_index: int | None = None
        relapsed = False

        while time.time() < deadline:
            if plan.max_polls is not None and len(rounds) >= plan.max_polls:
                break
            round_ = await self.run_checks(plan, not_before=verification_began)
            rounds.append(round_)
            report.outcomes.extend(round_.outcomes)

            if on_poll is not None:
                try:
                    maybe = on_poll(len(rounds), round_)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                except Exception:  # noqa: BLE001 - a reporting callback must
                    log.exception("verification on_poll callback failed")

            # -- regression: stop immediately, do not wait out the window ---
            newly_broken = [
                o.check_id
                for o in round_.outcomes
                if not o.passed and o.check_id in passed_at_baseline
            ]
            if newly_broken:
                report.verdict = VerificationVerdict.REGRESSED
                names = {c.id: c.name for c in plan.checks}
                report.summary = (
                    "REGRESSED: "
                    + ", ".join(names.get(i, i) for i in newly_broken)
                    + " passed before the fix and fails now. The remediation made "
                    "things worse; rolling back without waiting out the window."
                )
                report.finished_at = _now()
                return report

            if round_.all_passed:
                if first_green_index is None:
                    first_green_index = len(rounds) - 1
            elif first_green_index is not None:
                # Was healthy, is not any more. This is the decoy-fix signature.
                relapsed = True
                first_green_index = None

            remaining = deadline - time.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(plan.poll_interval_seconds, remaining))

        report.finished_at = _now()
        names = {c.id: c.name for c in plan.checks}

        if first_green_index is None:
            failing = rounds[-1].failed_ids if rounds else []
            report.verdict = VerificationVerdict.NOT_RECOVERED
            if relapsed:
                report.summary = (
                    f"NOT RECOVERED: the system became healthy and then relapsed within "
                    f"the {plan.window_seconds}s window. This is the signature of a fix "
                    f"that suppresses the symptom without addressing the cause. "
                    f"Still failing: {', '.join(names.get(i, i) for i in failing) or 'unknown'}."
                )
            else:
                report.summary = (
                    f"NOT RECOVERED: the system never reached a healthy state during "
                    f"the {plan.window_seconds}s window. Still failing: "
                    f"{', '.join(names.get(i, i) for i in failing) or 'unknown'}."
                )
            return report

        green_streak = len(rounds) - first_green_index
        if green_streak < plan.required_consecutive_passes:
            report.verdict = VerificationVerdict.NOT_RECOVERED
            report.summary = (
                f"NOT RECOVERED: health held for only {green_streak} consecutive "
                f"poll(s); {plan.required_consecutive_passes} are required. One green "
                f"poll is a coincidence, not a recovery."
            )
            return report

        held_seconds = int(rounds[-1].at - rounds[first_green_index].at)
        report.verdict = VerificationVerdict.RECOVERED
        report.summary = (
            f"RECOVERED: all {len(plan.checks)} checks passed on {green_streak} "
            f"consecutive polls and health held continuously for {held_seconds}s "
            f"to the end of the {plan.window_seconds}s window."
        )
        if relapsed:
            report.summary += (
                " Note: an earlier relapse occurred during this window; the fix "
                "stabilised but this incident is worth a human review."
            )
        return report

    # -- rollback ----------------------------------------------------------

    async def rollback(
        self, action: ProposedAction, incident: Incident
    ) -> ExecutionResult | None:
        """Undo a failed remediation.

        Runs WITHOUT asking for approval, on purpose. The human already
        approved entering this state; returning production to where it was is
        the conservative act, and an unattended rollback beats waiting for
        someone to notice a card at 3am. The inverse was declared and shown on
        the approval card before anything ran, so nothing here is a surprise.
        """
        if action.rollback is None:
            incident.log(
                actor="agent",
                event="rollback_unavailable",
                detail=(
                    f"Action {action.id} declared no inverse; cannot roll back "
                    "automatically. Escalating."
                ),
                action_id=action.id,
            )
            return None

        incident.log(
            actor="agent",
            event="rollback_started",
            detail=action.rollback.describe(),
            action_id=action.id,
        )
        result = await self.executors.execute(
            action.rollback, action_id=action.id, is_rollback=True
        )
        incident.executions.append(result)
        incident.log(
            actor="agent",
            event="rollback_succeeded" if result.succeeded else "rollback_failed",
            detail=result.stdout or result.error or result.stderr,
            action_id=action.id,
        )
        if not result.succeeded:
            incident.state = IncidentState.ESCALATED
            incident.log(
                actor="agent",
                event="escalated",
                detail=(
                    "Rollback failed. Production is in an unknown state and needs a "
                    "human immediately."
                ),
            )
        return result

    # -- orchestration -----------------------------------------------------

    async def verify_and_maybe_rollback(
        self,
        incident: Incident,
        action: ProposedAction,
        plan: VerificationPlan,
        *,
        on_poll: Any = None,
    ) -> VerificationReport:
        """The closed loop, start to finish.

        Assumes `capture_baseline` ran before the remediation - if it did not,
        REGRESSED cannot be detected and the report says so rather than
        quietly downgrading to a weaker check.
        """
        incident.state = IncidentState.VERIFYING
        incident.log(
            actor="agent",
            event="verification_started",
            detail=(
                f"{len(plan.checks)} check(s), window {plan.window_seconds}s, "
                f"poll {plan.poll_interval_seconds}s, "
                f"{plan.required_consecutive_passes} consecutive passes required"
            ),
            action_id=action.id,
        )

        if not plan.baseline:
            incident.log(
                actor="agent",
                event="verification_warning",
                detail=(
                    "No baseline was captured before remediation; REGRESSED cannot "
                    "be distinguished from a pre-existing failure."
                ),
            )

        report = await self.verify(plan, incident=incident, on_poll=on_poll)
        incident.verification = report
        incident.log(
            actor="agent",
            event="verification_finished",
            detail=report.summary,
            verdict=report.verdict.value,
        )

        if report.verdict is VerificationVerdict.RECOVERED:
            incident.state = IncidentState.RESOLVED
            incident.closed_at = _now()
            return report

        if report.verdict is VerificationVerdict.INCONCLUSIVE:
            incident.state = IncidentState.ESCALATED
            incident.log(
                actor="agent",
                event="escalated",
                detail=(
                    "Recovery could not be measured. Escalating rather than claiming "
                    "success - an unverified fix is not a fix."
                ),
            )
            return report

        # NOT_RECOVERED or REGRESSED -> undo.
        report.rollback_triggered = True
        rollback_result = await self.rollback(action, incident)
        if rollback_result is not None and rollback_result.succeeded:
            incident.state = IncidentState.ROLLED_BACK
            incident.closed_at = _now()
            incident.log(
                actor="agent",
                event="rolled_back",
                detail=(
                    f"{report.verdict.value}: remediation reverted. "
                    "Production is back to its pre-fix state and needs a human."
                ),
            )
        elif rollback_result is None:
            incident.state = IncidentState.ESCALATED
        return report


# --------------------------------------------------------------------------
# Plan builders
# --------------------------------------------------------------------------


def plan_for_container_service(
    *,
    container: str,
    health_url: str,
    error_pattern: str = r"\b(ERROR|CRITICAL|FATAL|Traceback)\b",
    window_seconds: int = 180,
    poll_interval_seconds: int = 15,
    required_consecutive_passes: int = 3,
    prometheus_url: str = "",
    latency_query: str = "",
    latency_threshold_ms: float = 2000,
    adapter: str = "",
) -> VerificationPlan:
    """A sensible default plan for a containerised HTTP service.

    Four checks, because any one alone is fooled by a different failure mode:

      process     catches a dead or crash-looping container, which an HTTP
                  check on a load-balanced endpoint can miss
      http        catches a process that is up but not serving
      log_absence catches a service that answers 200 while erroring internally
      metric      catches a service that is up, quiet and far too slow -
                  the failure with no errors anywhere
    """
    checks = [
        HealthCheck(
            name=f"{container} is running and was not OOM-killed",
            kind="process",
            params={
                "name": container,
                "expect_status": "running",
                "forbid_oom": True,
                "max_restarts": 3,
                **({"adapter": adapter} if adapter else {}),
            },
            success_criteria="Container status is 'running', OOMKilled is false, restarts <= 3",
        ),
        HealthCheck(
            name=f"No errors in {container} logs",
            kind="log_absence",
            params={
                "pattern": error_pattern,
                "target": container,
                "window_minutes": 2,
                "max_occurrences": 0,
                **({"adapter": adapter} if adapter else {}),
            },
            success_criteria="Zero error-level log lines in the last 2 minutes",
        ),
    ]

    # A service with no known endpoint still gets process and log checks; an
    # HTTP check against a guessed URL would fail forever and read as a relapse.
    if health_url:
        checks.insert(
            1,
            HealthCheck(
                name=f"{container} health endpoint returns 200",
                kind="http",
                params={"url": health_url, "expect_status": 200, "max_latency_ms": 3000},
                success_criteria=f"GET {health_url} returns 200 within 3000ms",
            ),
        )

    if prometheus_url and latency_query:
        checks.append(
            HealthCheck(
                name=f"{container} p99 latency within SLO",
                kind="metric",
                params={
                    "prometheus_url": prometheus_url,
                    "query": latency_query,
                    "operator": "lt",
                    "threshold": latency_threshold_ms,
                },
                success_criteria=f"p99 latency < {latency_threshold_ms:g}ms",
            )
        )

    return VerificationPlan(
        checks=checks,
        window_seconds=window_seconds,
        poll_interval_seconds=poll_interval_seconds,
        required_consecutive_passes=required_consecutive_passes,
    )
