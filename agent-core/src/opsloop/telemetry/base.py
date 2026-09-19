"""Telemetry adapter protocol - the vendor-neutrality claim, in code.

Every source of truth about production sits behind this one interface. The
agent never learns whether it is reading Loki, Docker, journald over SSH or
Elasticsearch; it asks for logs, metrics, an inventory or a probe, and gets
`EvidenceRef` objects back.

Three rules hold for every adapter:

  1. Adapters return EVIDENCE, never prose. Anything an adapter produces is
     re-fetchable via the `query` recorded on it, so a human can check the
     agent's work.
  2. Adapters SANITISE before returning. Raw telemetry never reaches a caller,
     so there is no code path where unsanitised text can reach a prompt by
     accident. This is why sanitisation lives here and not in the agent.
  3. Adapters are READ-ONLY. Changing production is the job of `remediate/`,
     which is gated on human approval. An adapter that could mutate state
     would route around that gate.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..domain.models import EvidenceRef, TrustLevel
from ..security.sanitizer import TelemetrySanitiser


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Query types
# --------------------------------------------------------------------------


@dataclass
class TimeRange:
    start: datetime
    end: datetime

    @classmethod
    def last(cls, minutes: int = 15) -> TimeRange:
        end = utcnow()
        return cls(start=end - timedelta(minutes=minutes), end=end)

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()

    def describe(self) -> str:
        return f"{self.start.isoformat()}..{self.end.isoformat()}"


@dataclass
class LogQuery:
    target: str = ""  # service or container; "" means every target
    contains: str = ""  # substring filter
    level: str = ""  # ERROR | WARNING | INFO
    time_range: TimeRange = field(default_factory=lambda: TimeRange.last(15))
    limit: int = 200


@dataclass
class MetricQuery:
    name: str
    target: str = ""
    time_range: TimeRange = field(default_factory=lambda: TimeRange.last(15))
    step_seconds: int = 30


@dataclass
class ResourceState:
    """One monitored thing, as the adapter currently sees it."""

    name: str
    kind: str  # "container" | "service" | "process" | "host"
    status: str  # "running" | "exited" | "failed" | "unknown"
    healthy: bool | None = None
    started_at: datetime | None = None
    restart_count: int = 0
    exit_code: int | None = None
    oom_killed: bool = False
    image: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)  # cpu/mem/limits
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_suspicious(self) -> bool:
        """Cheap triage before any model is involved - most real incidents are
        visible right here, and an LLM is not needed to notice exit code 137."""
        return (
            self.oom_killed
            or self.status in {"exited", "dead", "failed"}
            or self.restart_count > 2
            or self.healthy is False
            or (self.exit_code not in (None, 0))
        )

    def summarise(self) -> str:
        bits = [f"{self.kind} {self.name}: {self.status}"]
        if self.healthy is not None:
            bits.append(f"health={'ok' if self.healthy else 'UNHEALTHY'}")
        if self.exit_code not in (None, 0):
            bits.append(f"exit={self.exit_code}")
        if self.oom_killed:
            bits.append("OOM_KILLED=true")
        if self.restart_count:
            bits.append(f"restarts={self.restart_count}")
        return " ".join(bits)


class AdapterError(RuntimeError):
    def __init__(self, adapter: str, message: str) -> None:
        self.adapter = adapter
        super().__init__(f"[{adapter}] {message}")


@dataclass
class AdapterHealth:
    ok: bool
    detail: str = ""
    error: str = ""
    latency_ms: int = 0


# --------------------------------------------------------------------------
# Adapter base
# --------------------------------------------------------------------------


class TelemetryAdapter(abc.ABC):
    """Read-only window onto one telemetry source."""

    #: Stable identifier used in EvidenceRef.source_kind
    kind: str = "abstract"

    def __init__(self, name: str, sanitiser: TelemetrySanitiser | None = None) -> None:
        self.name = name
        self.sanitiser = sanitiser or TelemetrySanitiser()

    # -- capabilities ------------------------------------------------------

    @property
    def supports_logs(self) -> bool:
        return True

    @property
    def supports_metrics(self) -> bool:
        return False

    @property
    def supports_inventory(self) -> bool:
        return False

    # -- required ----------------------------------------------------------

    @abc.abstractmethod
    async def health(self) -> AdapterHealth:
        """Can we reach this source at all? Shown in Settings."""

    @abc.abstractmethod
    async def fetch_logs(self, query: LogQuery) -> list[EvidenceRef]: ...

    # -- optional ----------------------------------------------------------

    async def fetch_metric(self, query: MetricQuery) -> list[EvidenceRef]:
        raise AdapterError(self.name, f"{self.kind} adapter does not provide metrics")

    async def inventory(self) -> list[ResourceState]:
        raise AdapterError(self.name, f"{self.kind} adapter does not provide an inventory")

    async def close(self) -> None:
        return None

    # -- helper ------------------------------------------------------------

    def make_evidence(
        self,
        *,
        raw: str,
        query: str,
        observed_at: datetime | None = None,
        trust: TrustLevel = TrustLevel.UNTRUSTED_TEXT,
        **metadata: Any,
    ) -> EvidenceRef:
        """The ONLY way an adapter should produce evidence.

        Sanitisation happens here so that no adapter can forget it. Taint
        findings are carried onto the evidence, where `policy/` can refuse to
        auto-approve actions that rest on tainted data.
        """
        clean = self.sanitiser.sanitise(raw)
        return EvidenceRef(
            source_kind=self.kind,
            source_name=self.name,
            query=query,
            observed_at=observed_at or utcnow(),
            excerpt=clean.text,
            raw_sha256=clean.raw_sha256,
            trust=trust,
            tainted=clean.tainted,
            taint_reasons=clean.reasons,
            metadata=metadata,
        )

    def make_trusted_evidence(
        self, *, raw: str, query: str, observed_at: datetime | None = None, **metadata: Any
    ) -> EvidenceRef:
        """For values we generated ourselves - exit codes, metric numbers, our
        own probe results. An attacker cannot write prose into these.

        Still sanitised: `make_evidence` handles size bounding and secret
        redaction, which apply to trusted sources too.
        """
        return self.make_evidence(
            raw=raw,
            query=query,
            observed_at=observed_at,
            trust=TrustLevel.TRUSTED,
            **metadata,
        )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class AdapterRegistry:
    """All configured telemetry sources.

    The agent fans a question out across every adapter that can answer it,
    which is what lets one incident cite Docker state and Loki logs and an SSH
    probe in a single diagnosis - the cross-source correlation that
    single-vendor tools cannot do over someone else's telemetry.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, TelemetryAdapter] = {}

    def add(self, adapter: TelemetryAdapter) -> None:
        self._adapters[adapter.name] = adapter

    async def remove(self, name: str) -> None:
        adapter = self._adapters.pop(name, None)
        if adapter is not None:
            await adapter.close()

    def get(self, name: str) -> TelemetryAdapter:
        try:
            return self._adapters[name]
        except KeyError:
            known = ", ".join(sorted(self._adapters)) or "none configured"
            raise KeyError(f"No telemetry adapter named {name!r} (known: {known})") from None

    def all(self) -> list[TelemetryAdapter]:
        return list(self._adapters.values())

    def with_logs(self) -> list[TelemetryAdapter]:
        return [a for a in self._adapters.values() if a.supports_logs]

    def with_metrics(self) -> list[TelemetryAdapter]:
        return [a for a in self._adapters.values() if a.supports_metrics]

    def with_inventory(self) -> list[TelemetryAdapter]:
        return [a for a in self._adapters.values() if a.supports_inventory]

    async def close_all(self) -> None:
        for adapter in self._adapters.values():
            await adapter.close()
        self._adapters.clear()
