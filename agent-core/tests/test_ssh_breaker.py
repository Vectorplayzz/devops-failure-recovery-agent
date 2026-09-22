"""The SSH circuit breaker.

Regression for a real incident during development: with the key refused, one
host inventory made ~13 login attempts in two seconds - one per probe - and
the server began resetting connections mid-handshake. That is the signature
fail2ban bans on (default: 5 failures in 10 minutes), and a monitor retrying
every minute would have kept the owner locked out of their own server.
"""

from __future__ import annotations

from typing import Any

import pytest

from opsloop.telemetry.base import AdapterError
from opsloop.telemetry.ssh_adapter import SSHAdapter, SSHConfig


def adapter(monkeypatch: pytest.MonkeyPatch, *, fail: bool = True) -> tuple[SSHAdapter, list[int]]:
    attempts: list[int] = []
    a = SSHAdapter("vps", SSHConfig(host="example.invalid", username="u"))

    def connect() -> Any:
        attempts.append(1)
        if fail:
            raise OSError("Authentication failed")
        return _FakeClient()

    monkeypatch.setattr(a, "_connect_sync", connect)
    return a, attempts


class _FakeClient:
    class _T:
        def is_active(self) -> bool:
            return True

    def get_transport(self) -> Any:
        return self._T()

    def close(self) -> None:
        return None


class TestBreaker:
    async def test_inventory_makes_one_attempt_not_one_per_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a, attempts = adapter(monkeypatch)
        inv = await a.discover()
        assert len(attempts) == 1
        assert inv.hostname == "" and len(inv.errors) == 1

    async def test_repeated_calls_do_not_retry_during_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a, attempts = adapter(monkeypatch)
        for _ in range(10):
            await a.discover()
            await a.health()
        assert len(attempts) == 1  # everything after the first is refused locally

    async def test_error_explains_the_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        a, _ = adapter(monkeypatch)
        with pytest.raises(AdapterError):
            await a._ensure_connected()
        with pytest.raises(AdapterError, match="not retrying SSH"):
            await a._ensure_connected()

    async def test_backoff_doubles_and_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import opsloop.telemetry.ssh_adapter as mod

        clock = [1000.0]
        monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
        a, attempts = adapter(monkeypatch)
        waits = []
        for _ in range(8):
            with pytest.raises(AdapterError):
                await a._ensure_connected()
            waits.append(a._retry_after - clock[0])
            clock[0] = a._retry_after + 0.1  # jump just past the backoff
        assert waits[:4] == pytest.approx([15, 30, 60, 120])
        assert max(waits) == pytest.approx(SSHAdapter.BACKOFF_MAX_SECONDS)
        assert len(attempts) == 8

    async def test_success_resets_the_breaker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import opsloop.telemetry.ssh_adapter as mod

        clock = [1000.0]
        monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
        a, _ = adapter(monkeypatch)
        with pytest.raises(AdapterError):
            await a._ensure_connected()

        monkeypatch.setattr(a, "_connect_sync", lambda: _FakeClient())
        clock[0] = a._retry_after + 1
        await a._ensure_connected()
        assert a._failures == 0 and a._retry_after == 0.0
