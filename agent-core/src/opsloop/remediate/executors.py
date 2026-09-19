"""Remediation executors - the only place production changes.

Every executor is registered here with a fixed id, a parameter schema and a
risk level. The model chooses an id from this closed set and supplies
parameters; it never supplies a command string. That is what makes the
approval gate meaningful - if a model could emit `bash -c ...`, everything in
policy/ would be decoration.

A note on the reversibility invariant
-------------------------------------
`ProposedAction` refuses to construct without a rollback. Applied naively that
is wrong: restarting an already-running container cannot be "un-restarted",
and demanding a fake inverse would be worse than demanding none, because it
would look like a rollback exists when it does not.

The resolution is the `noop` executor, which takes a mandatory `reason`. The
invariant is not "every action is undoable" - some genuinely are not. It is
"every action has had its reversal thought about, and the answer is recorded
where a human can read it before approving". `noop` makes that reasoning
explicit and auditable instead of silently absent.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..domain.models import ActionSpec, ExecutionResult, RiskLevel, _now

try:  # pragma: no cover
    import docker
    from docker.errors import APIError, DockerException, NotFound
except ImportError:  # pragma: no cover
    docker = None  # type: ignore[assignment]
    APIError = DockerException = NotFound = Exception  # type: ignore[misc,assignment]


Handler = Callable[[dict[str, Any]], Awaitable[tuple[bool, str, str]]]


@dataclass
class ExecutorSpec:
    id: str
    description: str
    risk: RiskLevel
    handler: Handler
    required_params: tuple[str, ...] = ()
    read_only: bool = False
    # What undoing this looks like, shown on the approval card.
    reversal: str = ""
    docs: str = ""


class ExecutorError(RuntimeError):
    pass


class ExecutorRegistry:
    """All actions the agent is capable of performing, and nothing else."""

    def __init__(self) -> None:
        self._executors: dict[str, ExecutorSpec] = {}
        self.register(
            ExecutorSpec(
                id="noop",
                description="Do nothing; record why no reversal is possible or needed.",
                risk=RiskLevel.SAFE,
                handler=self._noop,
                required_params=("reason",),
                read_only=True,
                reversal="n/a",
            )
        )

    # -- registry ----------------------------------------------------------

    def register(self, spec: ExecutorSpec) -> None:
        if spec.id in self._executors:
            raise ValueError(f"Executor {spec.id!r} is already registered")
        self._executors[spec.id] = spec

    def get(self, executor_id: str) -> ExecutorSpec:
        try:
            return self._executors[executor_id]
        except KeyError:
            raise ExecutorError(
                f"Unknown executor {executor_id!r}. Available: "
                f"{', '.join(sorted(self._executors))}"
            ) from None

    def all(self) -> list[ExecutorSpec]:
        return list(self._executors.values())

    def catalogue(self) -> str:
        """The menu the model is allowed to choose from."""
        lines = []
        for spec in sorted(self._executors.values(), key=lambda s: s.id):
            params = ", ".join(spec.required_params) or "-"
            lines.append(
                f"{spec.id}(risk={spec.risk.value}) params[{params}] - {spec.description}"
                + (f" Reversal: {spec.reversal}" if spec.reversal else "")
            )
        return "\n".join(lines)

    def risk_of(self, executor_id: str) -> RiskLevel:
        return self.get(executor_id).risk

    # -- execution ---------------------------------------------------------

    async def execute(
        self, spec: ActionSpec, *, action_id: str, is_rollback: bool = False
    ) -> ExecutionResult:
        started = _now()
        result = ExecutionResult(
            action_id=action_id, started_at=started, was_rollback=is_rollback
        )
        try:
            executor = self.get(spec.executor)
        except ExecutorError as exc:
            result.error = str(exc)
            result.finished_at = _now()
            return result

        missing = [p for p in executor.required_params if p not in spec.params]
        if missing:
            result.error = (
                f"Executor {spec.executor!r} is missing required parameter(s): "
                f"{', '.join(missing)}"
            )
            result.finished_at = _now()
            return result

        try:
            ok, stdout, stderr = await executor.handler(spec.params)
            result.succeeded = ok
            result.stdout = stdout
            result.stderr = stderr
            result.exit_code = 0 if ok else 1
        except Exception as exc:  # noqa: BLE001 - an executor must never crash the loop
            result.succeeded = False
            result.error = f"{exc.__class__.__name__}: {exc}"
        result.finished_at = _now()
        return result

    # -- built-in ----------------------------------------------------------

    @staticmethod
    async def _noop(params: dict[str, Any]) -> tuple[bool, str, str]:
        return True, f"no action taken: {params['reason']}", ""


# --------------------------------------------------------------------------
# Docker executors
# --------------------------------------------------------------------------


def register_docker_executors(registry: ExecutorRegistry, client: Any = None) -> None:
    if docker is None:
        raise ExecutorError("the 'docker' package is not installed")
    client = client or docker.from_env()

    def _get(name: str) -> Any:
        try:
            return client.containers.get(name)
        except NotFound as exc:
            raise ExecutorError(f"no container named {name!r}") from exc

    async def restart(params: dict[str, Any]) -> tuple[bool, str, str]:
        name = params["container"]
        timeout = int(params.get("timeout", 10))

        def _do() -> str:
            c = _get(name)
            before = c.attrs.get("State", {}).get("Status", "unknown")
            c.restart(timeout=timeout)
            c.reload()
            after = c.attrs.get("State", {}).get("Status", "unknown")
            return f"{name}: {before} -> {after}"

        out = await asyncio.to_thread(_do)
        return True, out, ""

    async def start(params: dict[str, Any]) -> tuple[bool, str, str]:
        name = params["container"]

        def _do() -> str:
            c = _get(name)
            c.start()
            c.reload()
            return f"{name}: started, status={c.attrs.get('State', {}).get('Status')}"

        return True, await asyncio.to_thread(_do), ""

    async def stop(params: dict[str, Any]) -> tuple[bool, str, str]:
        name = params["container"]
        timeout = int(params.get("timeout", 10))

        def _do() -> str:
            c = _get(name)
            c.stop(timeout=timeout)
            return f"{name}: stopped"

        return True, await asyncio.to_thread(_do), ""

    async def update_memory(params: dict[str, Any]) -> tuple[bool, str, str]:
        name = params["container"]
        mb = int(params["memory_mb"])
        if mb < 64:
            raise ExecutorError(f"memory limit {mb}MB is below the 64MB floor")

        def _do() -> str:
            c = _get(name)
            old = c.attrs.get("HostConfig", {}).get("Memory", 0)
            # memswap must move with memory, or the daemon rejects the update.
            c.update(mem_limit=f"{mb}m", memswap_limit=f"{mb}m")
            c.reload()
            new = c.attrs.get("HostConfig", {}).get("Memory", 0)
            return (
                f"{name}: memory limit {old // 1024 // 1024}MB -> {new // 1024 // 1024}MB"
            )

        return True, await asyncio.to_thread(_do), ""

    registry.register(
        ExecutorSpec(
            id="docker.restart",
            description="Restart a container.",
            risk=RiskLevel.LOW,
            handler=restart,
            required_params=("container",),
            reversal=(
                "None - a restart cannot be un-performed. Pair with noop and state "
                "why that is acceptable."
            ),
        )
    )
    registry.register(
        ExecutorSpec(
            id="docker.start",
            description="Start a stopped container.",
            risk=RiskLevel.LOW,
            handler=start,
            required_params=("container",),
            reversal="docker.stop on the same container.",
        )
    )
    registry.register(
        ExecutorSpec(
            id="docker.stop",
            description="Stop a running container.",
            risk=RiskLevel.MEDIUM,
            handler=stop,
            required_params=("container",),
            reversal="docker.start on the same container.",
        )
    )
    registry.register(
        ExecutorSpec(
            id="docker.update_memory",
            description="Change a container's memory limit without recreating it.",
            risk=RiskLevel.MEDIUM,
            handler=update_memory,
            required_params=("container", "memory_mb"),
            reversal="docker.update_memory back to the previous limit.",
            docs=(
                "Relieves an OOM kill but does not fix a leak - it postpones it. "
                "Verification over a window is what tells the two apart."
            ),
        )
    )


# --------------------------------------------------------------------------
# SSH executors
# --------------------------------------------------------------------------


def register_ssh_executors(registry: ExecutorRegistry, adapter: Any, prefix: str = "ssh") -> None:
    """Register systemd controls against one SSH adapter.

    Unit names are shell-quoted. The model chooses an executor id and a unit
    name; it cannot smuggle a command through either.
    """

    async def systemctl(verb: str, unit: str) -> tuple[bool, str, str]:
        safe_unit = shlex.quote(unit if unit.endswith(".service") else f"{unit}.service")
        result = await adapter.run(f"systemctl {verb} {safe_unit}")
        status = await adapter.run(f"systemctl is-active {safe_unit}")
        return (
            result.exit_code == 0,
            f"systemctl {verb} {unit} -> exit {result.exit_code}; "
            f"is-active={status.stdout.strip()}",
            result.stderr.strip(),
        )

    async def restart(params: dict[str, Any]) -> tuple[bool, str, str]:
        return await systemctl("restart", params["unit"])

    async def start(params: dict[str, Any]) -> tuple[bool, str, str]:
        return await systemctl("start", params["unit"])

    async def stop(params: dict[str, Any]) -> tuple[bool, str, str]:
        return await systemctl("stop", params["unit"])

    registry.register(
        ExecutorSpec(
            id=f"{prefix}.systemctl_restart",
            description="Restart a systemd unit on the remote host.",
            risk=RiskLevel.LOW,
            handler=restart,
            required_params=("unit",),
            reversal="None - pair with noop and justify.",
        )
    )
    registry.register(
        ExecutorSpec(
            id=f"{prefix}.systemctl_start",
            description="Start a systemd unit on the remote host.",
            risk=RiskLevel.LOW,
            handler=start,
            required_params=("unit",),
            reversal=f"{prefix}.systemctl_stop on the same unit.",
        )
    )
    registry.register(
        ExecutorSpec(
            id=f"{prefix}.systemctl_stop",
            description="Stop a systemd unit on the remote host.",
            risk=RiskLevel.MEDIUM,
            handler=stop,
            required_params=("unit",),
            reversal=f"{prefix}.systemctl_start on the same unit.",
        )
    )


# --------------------------------------------------------------------------
# Demo-stack executor
# --------------------------------------------------------------------------


def register_demo_executors(registry: ExecutorRegistry) -> None:
    """Stands in for a code-level fix in the demo environment.

    In a real system the fix for a memory leak is shipping a patched build.
    The demo stack has no deploy pipeline, so clearing the injected fault plays
    that role: it frees the leaked memory and closes leaked connections, which
    is what a corrected build would achieve.

    Registered only for the demo stack. It is called out explicitly rather than
    disguised as a generic executor, because anyone reading a demo is entitled
    to know which parts are real - the OOM kill is real, and this is the
    stand-in.
    """
    import httpx

    ENDPOINTS = {
        "orders-api": "http://localhost:18081",
        "payments-api": "http://localhost:18082",
    }

    async def clear_fault(params: dict[str, Any]) -> tuple[bool, str, str]:
        service = params["service"]
        base = ENDPOINTS.get(service)
        if base is None:
            raise ExecutorError(
                f"unknown demo service {service!r}; known: {', '.join(ENDPOINTS)}"
            )
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.post(f"{base}/admin/fault", json={"mode": "none", "intensity": 1.0})
            r.raise_for_status()
            body = r.json()
        return True, f"{service}: fault {body.get('previous')} -> {body.get('mode')}", ""

    async def set_fault(params: dict[str, Any]) -> tuple[bool, str, str]:
        service = params["service"]
        mode = params["mode"]
        base = ENDPOINTS.get(service)
        if base is None:
            raise ExecutorError(f"unknown demo service {service!r}")
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.post(f"{base}/admin/fault", json={"mode": mode, "intensity": 1.0})
            r.raise_for_status()
            body = r.json()
        return True, f"{service}: fault {body.get('previous')} -> {body.get('mode')}", ""

    registry.register(
        ExecutorSpec(
            id="demo.apply_code_fix",
            description=(
                "Apply the corrected build for a demo service (stops the leak / "
                "removes the defect at source)."
            ),
            risk=RiskLevel.MEDIUM,
            handler=clear_fault,
            required_params=("service",),
            reversal="demo.revert_code_fix re-introduces the defect.",
        )
    )
    registry.register(
        ExecutorSpec(
            id="demo.revert_code_fix",
            description="Re-introduce a defect in a demo service (rollback path).",
            risk=RiskLevel.MEDIUM,
            handler=set_fault,
            required_params=("service", "mode"),
            reversal="demo.apply_code_fix removes it again.",
        )
    )
