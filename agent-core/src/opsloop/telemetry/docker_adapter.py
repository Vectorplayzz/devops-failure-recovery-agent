"""Docker adapter - container state and logs.

The most important thing this adapter produces is not logs, it is
`State.OOMKilled` and `State.ExitCode`. Those come from the Docker daemon, not
from application output, so an attacker who controls what the application logs
cannot forge them. They are marked TRUSTED for exactly that reason, and the
policy engine weighs them accordingly.

The docker SDK is synchronous, so every call is pushed to a worker thread.
Blocking the event loop here would stall the Teams gateway during an incident,
which is the worst possible moment to be unresponsive.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from ..domain.models import EvidenceRef
from .base import (
    AdapterError,
    AdapterHealth,
    LogQuery,
    MetricQuery,
    ResourceState,
    TelemetryAdapter,
)

try:  # pragma: no cover - import guard
    import docker
    from docker.errors import DockerException, NotFound
except ImportError:  # pragma: no cover
    docker = None  # type: ignore[assignment]
    DockerException = NotFound = Exception  # type: ignore[misc,assignment]


def _parse_docker_time(value: str | None) -> datetime | None:
    """Docker returns RFC3339 with nanosecond precision, which
    datetime.fromisoformat rejects on Python < 3.11 and still dislikes with a
    trailing Z on some versions. Normalise both."""
    if not value or value.startswith("0001-01-01"):
        return None
    text = value.replace("Z", "+00:00")
    if "." in text:
        head, _, tail = text.partition(".")
        frac = tail
        offset = ""
        for sign in ("+", "-"):
            if sign in tail:
                frac, _, off = tail.partition(sign)
                offset = sign + off
                break
        text = f"{head}.{frac[:6]}{offset}"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


class DockerAdapter(TelemetryAdapter):
    kind = "docker"

    def __init__(
        self,
        name: str = "docker",
        *,
        base_url: str | None = None,
        label_filter: dict[str, str] | None = None,
        sanitiser: Any = None,
    ) -> None:
        super().__init__(name, sanitiser)
        if docker is None:
            raise AdapterError(name, "the 'docker' package is not installed")
        try:
            self._client = (
                docker.DockerClient(base_url=base_url)
                if base_url
                else docker.from_env()
            )
        except DockerException as exc:
            raise AdapterError(name, f"cannot connect to the Docker daemon: {exc}") from exc
        # Restricting to labelled containers keeps the agent's view scoped to
        # what it is meant to operate on, rather than everything on the host.
        self.label_filter = label_filter or {"opsloop.environment": "production"}

    @property
    def supports_inventory(self) -> bool:
        return True

    @property
    def supports_metrics(self) -> bool:
        return True

    # -- health ------------------------------------------------------------

    async def health(self) -> AdapterHealth:
        started = time.perf_counter()
        try:
            info = await asyncio.to_thread(self._client.version)
            n = len(await asyncio.to_thread(self._list_containers))
            return AdapterHealth(
                ok=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
                detail=f"docker {info.get('Version', '?')}, {n} monitored container(s)",
            )
        except Exception as exc:  # noqa: BLE001
            return AdapterHealth(
                ok=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )

    # -- inventory ---------------------------------------------------------

    def _list_containers(self) -> list[Any]:
        filters = {"label": [f"{k}={v}" for k, v in self.label_filter.items()]}
        return self._client.containers.list(all=True, filters=filters)

    async def inventory(self) -> list[ResourceState]:
        containers = await asyncio.to_thread(self._list_containers)
        out: list[ResourceState] = []
        for c in containers:
            attrs = c.attrs or {}
            state = attrs.get("State") or {}
            config = attrs.get("Config") or {}
            host = attrs.get("HostConfig") or {}
            health = (state.get("Health") or {}).get("Status")

            out.append(
                ResourceState(
                    name=c.name,
                    kind="container",
                    status=state.get("Status", "unknown"),
                    healthy=None if health is None else health == "healthy",
                    started_at=_parse_docker_time(state.get("StartedAt")),
                    restart_count=attrs.get("RestartCount", 0),
                    exit_code=state.get("ExitCode"),
                    oom_killed=bool(state.get("OOMKilled")),
                    image=(config.get("Image") or ""),
                    labels=config.get("Labels") or {},
                    resources={
                        "memory_limit_bytes": host.get("Memory") or 0,
                        "nano_cpus": host.get("NanoCpus") or 0,
                        "restart_policy": (host.get("RestartPolicy") or {}).get("Name", ""),
                    },
                    raw={"State": state},
                )
            )
        return out

    async def inventory_evidence(self) -> list[EvidenceRef]:
        """Container state as citable evidence.

        TRUSTED: these values come from the daemon. An attacker who can write
        log lines cannot write an exit code.
        """
        states = await self.inventory()
        evidence: list[EvidenceRef] = []
        for s in states:
            body = json.dumps(
                {
                    "container": s.name,
                    "status": s.status,
                    "health": s.healthy,
                    "exit_code": s.exit_code,
                    "oom_killed": s.oom_killed,
                    "restart_count": s.restart_count,
                    "image": s.image,
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                    "memory_limit_bytes": s.resources.get("memory_limit_bytes"),
                    "restart_policy": s.resources.get("restart_policy"),
                },
                indent=2,
            )
            evidence.append(
                self.make_trusted_evidence(
                    raw=body,
                    query=f"docker inspect {s.name} --format '{{{{json .State}}}}'",
                    container=s.name,
                    suspicious=s.is_suspicious,
                    summary=s.summarise(),
                )
            )
        return evidence

    # -- logs --------------------------------------------------------------

    def _fetch_container_logs(self, container_name: str, since: datetime, tail: int) -> str:
        try:
            container = self._client.containers.get(container_name)
        except NotFound as exc:
            raise AdapterError(self.name, f"no container named {container_name!r}") from exc
        raw = container.logs(
            since=since,
            tail=tail,
            timestamps=False,
            stdout=True,
            stderr=True,
        )
        return raw.decode("utf-8", errors="replace")

    async def fetch_logs(self, query: LogQuery) -> list[EvidenceRef]:
        if query.target:
            targets = [query.target]
        else:
            targets = [s.name for s in await self.inventory()]

        evidence: list[EvidenceRef] = []
        for target in targets:
            try:
                text = await asyncio.to_thread(
                    self._fetch_container_logs,
                    target,
                    query.time_range.start,
                    query.limit,
                )
            except AdapterError:
                continue  # a container that vanished mid-incident is not an error

            lines = [ln for ln in text.splitlines() if ln.strip()]
            if query.level:
                want = query.level.upper()
                lines = [ln for ln in lines if want in ln.upper()]
            if query.contains:
                needle = query.contains.lower()
                lines = [ln for ln in lines if needle in ln.lower()]
            if not lines:
                continue
            lines = lines[-query.limit :]

            filters = []
            if query.level:
                filters.append(f"level={query.level}")
            if query.contains:
                filters.append(f"contains={query.contains!r}")
            suffix = f"  # {' '.join(filters)}" if filters else ""

            evidence.append(
                self.make_evidence(
                    raw="\n".join(lines),
                    query=(
                        f"docker logs {target} --since "
                        f"{query.time_range.start.isoformat()} --tail {query.limit}{suffix}"
                    ),
                    container=target,
                    line_count=len(lines),
                    time_range=query.time_range.describe(),
                )
            )
        return evidence

    # -- metrics -----------------------------------------------------------

    def _stats(self, container_name: str) -> dict[str, Any]:
        container = self._client.containers.get(container_name)
        return container.stats(stream=False)

    async def fetch_metric(self, query: MetricQuery) -> list[EvidenceRef]:
        """Instantaneous CPU and memory. Docker keeps no history, so this is a
        point sample - `window` questions belong to Prometheus."""
        targets = (
            [query.target] if query.target else [s.name for s in await self.inventory()]
        )
        evidence: list[EvidenceRef] = []
        for target in targets:
            try:
                stats = await asyncio.to_thread(self._stats, target)
            except (AdapterError, NotFound, DockerException):
                continue

            mem = stats.get("memory_stats") or {}
            usage = mem.get("usage", 0)
            limit = mem.get("limit", 0) or 1
            pct = usage / limit * 100

            cpu = stats.get("cpu_stats") or {}
            pre = stats.get("precpu_stats") or {}
            cpu_pct = 0.0
            try:
                delta = cpu["cpu_usage"]["total_usage"] - pre["cpu_usage"]["total_usage"]
                sys_delta = cpu["system_cpu_usage"] - pre["system_cpu_usage"]
                if sys_delta > 0:
                    cpu_pct = delta / sys_delta * (cpu.get("online_cpus") or 1) * 100
            except (KeyError, TypeError):
                pass

            body = json.dumps(
                {
                    "container": target,
                    "memory_usage_mb": round(usage / 1024 / 1024, 1),
                    "memory_limit_mb": round(limit / 1024 / 1024, 1),
                    "memory_percent": round(pct, 1),
                    "cpu_percent": round(cpu_pct, 1),
                    # The number that predicts an imminent OOM kill.
                    "approaching_memory_limit": pct > 85,
                },
                indent=2,
            )
            evidence.append(
                self.make_trusted_evidence(
                    raw=body,
                    query=f"docker stats {target} --no-stream",
                    container=target,
                    memory_percent=round(pct, 1),
                )
            )
        return evidence

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)
