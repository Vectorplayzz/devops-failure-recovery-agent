"""Loki adapter - log aggregation over LogQL.

Present for two reasons. It is what a real deployment uses once there is more
than one host, and it proves the adapter protocol is not shaped around Docker. Docker returns a blob of
container stdout; Loki returns label-indexed streams queried in a different
language entirely. Both arrive at the agent as `EvidenceRef`.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx

from ..domain.models import EvidenceRef
from .base import (
    AdapterError,
    AdapterHealth,
    LogQuery,
    TelemetryAdapter,
)


def _to_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


def _from_ns(ns: str) -> datetime:
    return datetime.fromtimestamp(int(ns) / 1_000_000_000, tz=timezone.utc)


class LokiAdapter(TelemetryAdapter):
    kind = "loki"

    def __init__(
        self,
        name: str = "loki",
        *,
        base_url: str = "http://localhost:3100",
        label: str = "container",
        tenant: str = "",
        timeout: int = 20,
        sanitiser: Any = None,
    ) -> None:
        super().__init__(name, sanitiser)
        self.base_url = base_url.rstrip("/")
        # Which label identifies a target. Promtail in the demo stack sets
        # `container`, `service`, `role` and `tier`; a different deployment may
        # use `app` or `job`, so it is configurable rather than assumed.
        self.label = label
        headers = {"X-Scope-OrgID": tenant} if tenant else {}
        self._http = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, headers=headers
        )

    async def health(self) -> AdapterHealth:
        started = time.perf_counter()
        try:
            r = await self._http.get("/ready")
            ms = int((time.perf_counter() - started) * 1000)
            if r.status_code != 200:
                return AdapterHealth(
                    ok=False, latency_ms=ms, error=f"HTTP {r.status_code}: {r.text[:200]}"
                )
            labels = await self._http.get("/loki/api/v1/labels")
            names = (labels.json().get("data") or []) if labels.status_code == 200 else []
            return AdapterHealth(
                ok=True, latency_ms=ms, detail=f"ready, {len(names)} label(s) indexed"
            )
        except httpx.HTTPError as exc:
            return AdapterHealth(
                ok=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )

    def build_logql(self, query: LogQuery) -> str:
        """Assemble LogQL from a structured query.

        Built here rather than accepted from the model: a selector is a small
        closed grammar, and letting a model emit raw LogQL would mean telemetry
        content could influence what gets queried.
        """
        selector = (
            f'{{{self.label}=~".+"}}' if not query.target
            else f'{{{self.label}="{query.target}"}}'
        )
        expr = selector
        if query.level:
            expr += f' | json | level=~"(?i){query.level}"'
        if query.contains:
            escaped = query.contains.replace("\\", "\\\\").replace('"', '\\"')
            expr += f' |= "{escaped}"'
        return expr

    async def fetch_logs(self, query: LogQuery) -> list[EvidenceRef]:
        expr = self.build_logql(query)
        params = {
            "query": expr,
            "start": str(_to_ns(query.time_range.start)),
            "end": str(_to_ns(query.time_range.end)),
            "limit": str(query.limit),
            "direction": "backward",  # newest first; the tail matters most
        }
        try:
            r = await self._http.get("/loki/api/v1/query_range", params=params)
        except httpx.HTTPError as exc:
            raise AdapterError(self.name, f"query failed: {exc}") from exc
        if r.status_code != 200:
            raise AdapterError(self.name, f"HTTP {r.status_code}: {r.text[:300]}")

        result = (r.json().get("data") or {}).get("result") or []
        evidence: list[EvidenceRef] = []

        for stream in result:
            labels = stream.get("stream") or {}
            values = stream.get("values") or []
            if not values:
                continue
            # Loki returned these newest-first; read them oldest-first.
            ordered = sorted(values, key=lambda v: int(v[0]))
            lines = []
            for ns, line in ordered:
                lines.append(f"{_from_ns(ns).isoformat()} {line}")

            target = labels.get(self.label) or labels.get("service") or "unknown"
            evidence.append(
                self.make_evidence(
                    raw="\n".join(lines),
                    query=f"logcli query '{expr}' --since {query.time_range.describe()}",
                    observed_at=_from_ns(ordered[-1][0]),
                    target=target,
                    labels=labels,
                    line_count=len(lines),
                )
            )
        return evidence

    async def close(self) -> None:
        await self._http.aclose()
