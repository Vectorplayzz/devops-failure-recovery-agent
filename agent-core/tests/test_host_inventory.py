"""Host readiness tests.

The fixture below is a real reading taken from a small production VPS that
turned out to be at 100% disk. That case is why `can_run_demo_stack` checks
disk before memory: the host had 5.7GB of RAM free and was still completely
unable to run anything.
"""

from __future__ import annotations

from opsloop.telemetry.ssh_adapter import HostInventory


def vps(**overrides: object) -> HostInventory:
    """A real reading from a small VPS: plenty of RAM, no disk at all."""
    inv = HostInventory(
        hostname="vps-1",
        os_name="Ubuntu 24.04.4 LTS",
        cpu_count=2,
        memory_total_mb=7940,
        memory_available_mb=5697,
        disk=[{"mount": "/", "size": "96G", "used": "96G", "use_percent": "100%"}],
        docker_present=True,
    )
    for key, value in overrides.items():
        setattr(inv, key, value)
    return inv


class TestDiskGate:
    def test_full_disk_blocks_despite_plenty_of_ram(self) -> None:
        """The case that caught this: 5.7GB RAM free, and still unusable."""
        ok, why = vps().can_run_demo_stack()
        assert ok is False
        assert "100% full" in why
        assert "active incident" in why

    def test_nearly_full_disk_blocks(self) -> None:
        ok, why = vps(
            disk=[{"mount": "/", "size": "96G", "used": "90G", "use_percent": "94%"}]
        ).can_run_demo_stack()
        assert ok is False
        assert "6% of the root filesystem is free" in why

    def test_healthy_disk_and_ram_passes(self) -> None:
        ok, why = vps(
            disk=[{"mount": "/", "size": "96G", "used": "40G", "use_percent": "42%"}]
        ).can_run_demo_stack()
        assert ok is True
        assert "full profile" in why

    def test_free_percent_is_computed_from_root_only(self) -> None:
        inv = vps(
            disk=[
                {"mount": "/boot", "size": "1G", "used": "1G", "use_percent": "99%"},
                {"mount": "/", "size": "96G", "used": "48G", "use_percent": "50%"},
            ]
        )
        assert inv.root_disk_free_percent() == 50

    def test_unreadable_disk_does_not_crash_the_gate(self) -> None:
        inv = vps(disk=[])
        assert inv.root_disk_free_percent() is None
        ok, _ = inv.can_run_demo_stack()
        assert ok is True  # falls back to the memory check


class TestOtherGates:
    def test_missing_docker_blocks(self) -> None:
        ok, why = vps(
            disk=[{"mount": "/", "size": "96G", "used": "10G", "use_percent": "10%"}],
            docker_present=False,
        ).can_run_demo_stack()
        assert ok is False and "Docker is not installed" in why

    def test_low_memory_recommends_the_core_stack_only(self) -> None:
        ok, why = vps(
            disk=[{"mount": "/", "size": "96G", "used": "10G", "use_percent": "10%"}],
            memory_available_mb=900,
        ).can_run_demo_stack()
        assert ok is True
        assert "skip the observability profile" in why

    def test_very_low_memory_blocks(self) -> None:
        ok, why = vps(
            disk=[{"mount": "/", "size": "96G", "used": "10G", "use_percent": "10%"}],
            memory_available_mb=200,
        ).can_run_demo_stack()
        assert ok is False and "RAM available" in why


class TestSummary:
    def test_summary_surfaces_failed_units(self) -> None:
        text = vps(failed_units=["myapp.service"]).summarise()
        assert "FAILED" in text and "myapp.service" in text

    def test_summary_includes_disk_pressure(self) -> None:
        assert "100%" in vps().summarise()
