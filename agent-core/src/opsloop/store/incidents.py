"""In-memory incident store.

Deliberately simple for now: incidents live for the life of the process. That
is enough for a running agent and a demo, and it keeps the persistence decision
- SQLite, Postgres, something else - separate from everything that reads and
writes incidents, which only ever talk to this interface.

Deduplication lives here because it is a property of the incident set, not of
any one detector: one service should have one live incident, however many
signals keep firing about it.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime

from ..domain.models import Incident, IncidentState

# States in which an incident still needs attention. ROLLED_BACK and
# ESCALATED are terminal for the agent but NOT resolved for a human: the
# service is still broken and someone has to act. Opening a fresh incident for
# the same service in those states would bury the one with the history.
_NEEDS_ATTENTION = {
    IncidentState.DETECTED,
    IncidentState.TRIAGING,
    IncidentState.DIAGNOSED,
    IncidentState.AWAITING_APPROVAL,
    IncidentState.REMEDIATING,
    IncidentState.VERIFYING,
    IncidentState.ROLLED_BACK,
    IncidentState.ESCALATED,
}


class IncidentStore:
    def __init__(self, max_incidents: int = 500) -> None:
        self._items: OrderedDict[str, Incident] = OrderedDict()
        self._max = max_incidents
        # When each service last had an incident closed. Detectors ignore
        # telemetry older than this, so the errors that caused a resolved
        # incident cannot immediately open a new one.
        self._quiet_since: dict[str, datetime] = {}

    def add(self, incident: Incident) -> Incident:
        self._items[incident.id] = incident
        while len(self._items) > self._max:
            self._items.popitem(last=False)
        return incident

    def get(self, incident_id: str) -> Incident | None:
        return self._items.get(incident_id)

    def recent(self, limit: int = 10) -> list[Incident]:
        return list(reversed(self._items.values()))[:limit]

    def open(self) -> list[Incident]:
        return [i for i in self._items.values() if i.state in _NEEDS_ATTENTION]

    def open_for(self, service: str) -> Incident | None:
        return next((i for i in reversed(self._items.values())
                     if i.service == service and i.state in _NEEDS_ATTENTION), None)

    def mark_quiet(self, service: str, at: datetime) -> None:
        self._quiet_since[service] = at

    def quiet_since(self, service: str) -> datetime | None:
        return self._quiet_since.get(service)
