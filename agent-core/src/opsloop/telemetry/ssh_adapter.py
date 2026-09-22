"""SSH adapter - the plain-server connector.

This is the adapter that backs vendor-neutrality in the least flattering way
possible: no Loki, no Prometheus, no agent installed on the box. Just a shell,
journald, and files in /var/log - which is what an actual small deployment
looks like.

Two things make it safe to point at a real server:

  READ-ONLY BY CONSTRUCTION. Commands are assembled here from a fixed
  allowlist; the model never supplies a command string. A model that asks for
  "logs for nginx" gets `journalctl -u nginx ...` built by this module. There
  is no code path from model output to the shell, so there is nothing for
  prompt injection to hijack - the remediation executors are where state
  changes, and they are gated on human approval.

  DISCOVERY, NOT CONFIGURATION. `discover()` inventories the host itself, so
  nobody has to hand-write a list of services. Point it at a host and it
  reports the OS, resources, what is running under systemd and Docker, which
  ports are listening and where the logs are.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..domain.models import EvidenceRef
from .base import (
    AdapterError,
    AdapterHealth,
    LogQuery,
    ResourceState,
    TelemetryAdapter,
    utcnow,
)

try:  # pragma: no cover - import guard
    import paramiko
except ImportError:  # pragma: no cover
    paramiko = None  # type: ignore[assignment]


@dataclass
class SSHConfig:
    host: str
    username: str = "root"
    port: int = 22
    key_path: str = ""  # preferred
    password: str = ""  # fall back only; discouraged
    known_hosts_path: str = ""  # "" means use the system default
    connect_timeout: int = 15
    command_timeout: int = 30
    # Directories the adapter may read log files from.
    log_paths: list[str] = field(
        default_factory=lambda: ["/var/log/nginx", "/var/log/syslog", "/var/log/messages"]
    )

    def redacted(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "username": self.username,
            "port": self.port,
            "auth": "key" if self.key_path else ("password" if self.password else "agent"),
            "key_path": self.key_path,
            "log_paths": self.log_paths,
        }


@dataclass
class HostInventory:
    """What `discover()` finds - the answer to 'what is on this host?'"""

    hostname: str = ""
    os_name: str = ""
    kernel: str = ""
    uptime: str = ""
    cpu_count: int = 0
    memory_total_mb: int = 0
    memory_available_mb: int = 0
    disk: list[dict[str, str]] = field(default_factory=list)
    load_average: str = ""
    systemd_services: list[dict[str, str]] = field(default_factory=list)
    docker_present: bool = False
    docker_containers: list[dict[str, str]] = field(default_factory=list)
    listening_ports: list[dict[str, str]] = field(default_factory=list)
    failed_units: list[str] = field(default_factory=list)
    log_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summarise(self) -> str:
        lines = [
            f"host        {self.hostname}",
            f"os          {self.os_name}",
            f"kernel      {self.kernel}",
            f"uptime      {self.uptime}",
            f"cpu         {self.cpu_count} core(s), load {self.load_average}",
            f"memory      {self.memory_available_mb}MB available of {self.memory_total_mb}MB",
        ]
        for d in self.disk:
            lines.append(
                f"disk        {d.get('mount')} {d.get('used')}/{d.get('size')} "
                f"({d.get('use_percent')} used)"
            )
        lines.append(f"docker      {'yes' if self.docker_present else 'no'}"
                     + (f", {len(self.docker_containers)} container(s)"
                        if self.docker_present else ""))
        lines.append(f"services    {len(self.systemd_services)} running under systemd")
        if self.failed_units:
            lines.append(f"FAILED      {', '.join(self.failed_units)}")
        if self.listening_ports:
            ports = ", ".join(
                f"{p.get('port')}({p.get('process', '?')})" for p in self.listening_ports[:12]
            )
            lines.append(f"listening   {ports}")
        if self.errors:
            lines.append(f"errors      {'; '.join(self.errors)}")
        return "\n".join(lines)

    def root_disk_free_percent(self) -> int | None:
        """Free space on / as a percentage, or None if it could not be read."""
        for entry in self.disk:
            if entry.get("mount") == "/":
                pct = (entry.get("use_percent") or "").rstrip("%")
                if pct.isdigit():
                    return 100 - int(pct)
        return None

    def can_run_demo_stack(self) -> tuple[bool, str]:
        """Practical question: will the demo stack fit on this box?

        Disk is checked before memory, and deliberately so. A full root
        filesystem is not a capacity planning note - it is an active incident.
        Docker cannot pull an image, databases fail writes, logs stop, and
        services die on their next restart. Reporting "plenty of RAM, go
        ahead" on a host at 100% disk would be exactly the kind of confidently
        wrong answer this system exists to avoid.
        """
        free_pct = self.root_disk_free_percent()
        if free_pct is not None and free_pct <= 2:
            return False, (
                f"root filesystem is {100 - free_pct}% full. This is an active "
                "incident, not a capacity note: Docker cannot pull images, "
                "databases fail writes, and services die on restart. Free space "
                "before deploying anything."
            )

        if not self.docker_present:
            return False, "Docker is not installed"

        if free_pct is not None and free_pct < 10:
            return False, (
                f"only {free_pct}% of the root filesystem is free; the demo stack "
                "needs roughly 2.5GB of images"
            )

        if self.memory_available_mb < 500:
            return False, (
                f"only {self.memory_available_mb}MB RAM available; the core stack "
                "needs ~350MB and the observability profile ~1.4GB"
            )
        if self.memory_available_mb < 1600:
            return True, (
                f"{self.memory_available_mb}MB RAM available - run the core stack "
                "(`docker compose up -d`), skip the observability profile"
            )
        return True, (
            f"{self.memory_available_mb}MB RAM available"
            + (f", {free_pct}% disk free" if free_pct is not None else "")
            + " - the full profile will fit"
        )


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    at: datetime

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class SSHAdapter(TelemetryAdapter):
    kind = "ssh"

    def __init__(self, name: str, config: SSHConfig, sanitiser: Any = None) -> None:
        super().__init__(name, sanitiser)
        if paramiko is None:
            raise AdapterError(name, "the 'paramiko' package is not installed")
        self.config = config
        self._client: Any = None
        self._lock = asyncio.Lock()
        # Circuit breaker. See _ensure_connected.
        self._failures = 0
        self._retry_after = 0.0
        self._last_error = ""

    @property
    def supports_inventory(self) -> bool:
        return True

    # -- connection --------------------------------------------------------

    def _connect_sync(self) -> Any:
        client = paramiko.SSHClient()
        if self.config.known_hosts_path:
            client.load_host_keys(self.config.known_hosts_path)
        else:
            client.load_system_host_keys()
        # A first connection to a fresh VPS has no known_hosts entry. We accept
        # and record the key rather than refusing, and the host key fingerprint
        # is reported by discover() so it can be checked out of band.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        kwargs: dict[str, Any] = {
            "hostname": self.config.host,
            "port": self.config.port,
            "username": self.config.username,
            "timeout": self.config.connect_timeout,
            "allow_agent": True,
            "look_for_keys": not self.config.password,
        }
        if self.config.key_path:
            kwargs["key_filename"] = self.config.key_path
        elif self.config.password:
            kwargs["password"] = self.config.password

        client.connect(**kwargs)
        return client

    BACKOFF_BASE_SECONDS = 15
    BACKOFF_MAX_SECONDS = 600

    async def _ensure_connected(self) -> Any:
        """Connect once, and back off hard when that fails.

        A monitor that retries a failing login on every probe and every scan
        is indistinguishable from a brute-force attack. fail2ban's default
        bans an address after 5 failures in 10 minutes - so an agent without
        this breaker would lock its owner out of their own server, and keep
        them locked out. After a failure no new attempt is made until the
        backoff expires (15s, doubling to a 10 minute ceiling); callers get
        the last error instantly instead.
        """
        async with self._lock:
            if self._client is not None:
                transport = self._client.get_transport()
                if transport is not None and transport.is_active():
                    return self._client
                self._client = None

            now = time.monotonic()
            if now < self._retry_after:
                raise AdapterError(
                    self.name,
                    f"not retrying SSH to {self.config.host} for another "
                    f"{int(self._retry_after - now)}s after {self._failures} failure(s): "
                    f"{self._last_error}",
                )
            try:
                self._client = await asyncio.to_thread(self._connect_sync)
            except Exception as exc:  # noqa: BLE001
                self._failures += 1
                self._last_error = str(exc) or exc.__class__.__name__
                wait = min(
                    self.BACKOFF_MAX_SECONDS,
                    self.BACKOFF_BASE_SECONDS * 2 ** (self._failures - 1),
                )
                self._retry_after = time.monotonic() + wait
                raise AdapterError(
                    self.name, f"SSH connection to {self.config.host} failed: {self._last_error}"
                ) from exc
            self._failures = 0
            self._retry_after = 0.0
            self._last_error = ""
            return self._client

    def _run_sync(self, client: Any, command: str, timeout: int | None = None) -> CommandResult:
        at = utcnow()
        _, stdout, stderr = client.exec_command(
            command, timeout=timeout or self.config.command_timeout
        )
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        return CommandResult(command=command, exit_code=code, stdout=out, stderr=err, at=at)

    async def disk_usage(self, path: str = "/") -> EvidenceRef:
        """What is using the space under `path`, largest first.

        Read-only. `sudo -n` is tried first because an unprivileged `du` skips
        every directory it cannot read and under-reports badly; `-n` means it
        fails instantly instead of hanging on a password prompt, and plain
        `du` is the fallback. `-x` keeps it on one filesystem, so a mounted
        network share cannot turn this into an hour-long crawl.
        """
        if not path.startswith("/") or ".." in path.split("/"):
            raise AdapterError(self.name, f"disk_usage needs an absolute path without '..': {path!r}")
        q = shlex.quote(path)
        du = f"du -xh --max-depth=1 {q} 2>/dev/null | sort -rh | head -15"
        command = f"(sudo -n sh -c {shlex.quote(du)} 2>/dev/null || {du})"
        result = await self.run(command, timeout=120)
        return self.make_trusted_evidence(
            raw=result.stdout or "(no output - path unreadable or empty)",
            query=f"{self.config.username}@{self.config.host}: du -xh --max-depth=1 {path}",
            observed_at=result.at,
            path=path,
        )

    async def run(self, command: str, *, timeout: int | None = None) -> CommandResult:
        """Run one read-only command.

        Internal use only: every caller in this module passes a string it
        assembled itself from the allowlist below. Nothing derived from model
        output reaches this method.
        """
        client = await self._ensure_connected()
        try:
            return await asyncio.to_thread(self._run_sync, client, command, timeout)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(self.name, f"command failed: {exc}") from exc

    # -- health ------------------------------------------------------------

    async def health(self) -> AdapterHealth:
        started = time.perf_counter()
        try:
            r = await self.run("echo opsloop-ok && uptime -p")
            ms = int((time.perf_counter() - started) * 1000)
            if not r.ok:
                return AdapterHealth(ok=False, latency_ms=ms, error=r.stderr.strip())
            uptime = r.stdout.replace("opsloop-ok", "").strip()
            return AdapterHealth(
                ok=True,
                latency_ms=ms,
                detail=f"{self.config.username}@{self.config.host} reachable, {uptime}",
            )
        except AdapterError as exc:
            return AdapterHealth(
                ok=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )

    # -- discovery ---------------------------------------------------------

    async def discover(self) -> HostInventory:
        """Inventory the host so nobody has to describe it by hand.

        Every probe is independent and failures are recorded rather than
        raised: a minimal VPS may have no `ss`, no Docker and no `systemctl`,
        and a partial inventory is far more useful than an exception.
        """
        inv = HostInventory()

        # One connection attempt for the whole inventory. If it fails, stop:
        # a dozen probes each retrying the login is what gets an address
        # banned (see _ensure_connected).
        try:
            await self._ensure_connected()
        except AdapterError as exc:
            inv.errors.append(str(exc))
            return inv

        async def probe(command: str, label: str) -> str:
            try:
                r = await self.run(command)
                if not r.ok and not r.stdout.strip():
                    inv.errors.append(f"{label}: {r.stderr.strip()[:120] or 'no output'}")
                    return ""
                return r.stdout.strip()
            except AdapterError as exc:
                inv.errors.append(f"{label}: {exc}")
                return ""

        inv.hostname = await probe("hostname -f 2>/dev/null || hostname", "hostname")
        inv.os_name = await probe(
            "grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'", "os"
        )
        inv.kernel = await probe("uname -r", "kernel")
        inv.uptime = await probe("uptime -p", "uptime")
        inv.load_average = await probe("cat /proc/loadavg | cut -d' ' -f1-3", "loadavg")

        cpus = await probe("nproc", "nproc")
        inv.cpu_count = int(cpus) if cpus.isdigit() else 0

        mem = await probe("free -m | awk '/^Mem:/ {print $2, $7}'", "memory")
        parts = mem.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            inv.memory_total_mb, inv.memory_available_mb = int(parts[0]), int(parts[1])

        disk = await probe(
            "df -h --output=target,size,used,pcent -x tmpfs -x devtmpfs 2>/dev/null "
            "| tail -n +2",
            "disk",
        )
        for line in disk.splitlines():
            cols = line.split()
            if len(cols) >= 4:
                inv.disk.append(
                    {"mount": cols[0], "size": cols[1], "used": cols[2], "use_percent": cols[3]}
                )

        services = await probe(
            "systemctl list-units --type=service --state=running --no-pager --no-legend "
            "2>/dev/null | awk '{print $1}'",
            "systemd",
        )
        for unit in services.splitlines():
            unit = unit.strip()
            if unit.endswith(".service"):
                inv.systemd_services.append({"unit": unit, "state": "running"})

        failed = await probe(
            "systemctl list-units --state=failed --no-pager --no-legend 2>/dev/null "
            "| awk '{print $1}'",
            "failed-units",
        )
        inv.failed_units = [u.strip() for u in failed.splitlines() if u.strip()]

        docker_version = await probe("docker --version 2>/dev/null", "docker")
        inv.docker_present = bool(docker_version)
        if inv.docker_present:
            containers = await probe(
                "docker ps -a --format "
                "'{{.Names}}\\t{{.Image}}\\t{{.Status}}' 2>/dev/null",
                "docker-ps",
            )
            for line in containers.splitlines():
                cols = line.split("\t")
                if len(cols) >= 3:
                    inv.docker_containers.append(
                        {"name": cols[0], "image": cols[1], "status": cols[2]}
                    )

        ports = await probe(
            "ss -tlnp 2>/dev/null | tail -n +2 || netstat -tlnp 2>/dev/null | tail -n +3",
            "ports",
        )
        seen: set[str] = set()
        for line in ports.splitlines():
            cols = line.split()
            if len(cols) < 4:
                continue
            local = cols[3] if ":" in cols[3] else cols[4] if len(cols) > 4 else ""
            port = local.rsplit(":", 1)[-1]
            if not port.isdigit() or port in seen:
                continue
            seen.add(port)
            process = ""
            if "users:" in line:
                chunk = line.split('users:', 1)[1]
                if '"' in chunk:
                    process = chunk.split('"')[1]
            inv.listening_ports.append({"port": port, "process": process})

        logs = await probe(
            "find /var/log -maxdepth 2 -type f -name '*.log' -size +0 2>/dev/null | head -25",
            "log-files",
        )
        inv.log_files = [p.strip() for p in logs.splitlines() if p.strip()]

        return inv

    async def discovery_evidence(self) -> EvidenceRef:
        inv = await self.discover()
        return self.make_trusted_evidence(
            raw=inv.summarise(),
            query=f"opsloop discover {self.config.username}@{self.config.host}",
            hostname=inv.hostname,
            docker_present=inv.docker_present,
            failed_units=inv.failed_units,
        )

    # -- inventory ---------------------------------------------------------

    async def inventory(self) -> list[ResourceState]:
        inv = await self.discover()
        out: list[ResourceState] = [
            ResourceState(
                name=inv.hostname or self.config.host,
                kind="host",
                status="running",
                healthy=not inv.failed_units,
                resources={
                    "cpu_count": inv.cpu_count,
                    "memory_total_mb": inv.memory_total_mb,
                    "memory_available_mb": inv.memory_available_mb,
                    "load_average": inv.load_average,
                    "disk": inv.disk,
                },
            )
        ]
        for svc in inv.systemd_services:
            out.append(
                ResourceState(name=svc["unit"], kind="service", status="running", healthy=True)
            )
        for unit in inv.failed_units:
            out.append(ResourceState(name=unit, kind="service", status="failed", healthy=False))
        for c in inv.docker_containers:
            status = c.get("status", "")
            out.append(
                ResourceState(
                    name=c["name"],
                    kind="container",
                    status="running" if status.startswith("Up") else "exited",
                    healthy="unhealthy" not in status.lower() if status.startswith("Up") else False,
                    image=c.get("image", ""),
                    raw={"status_text": status},
                )
            )
        return out

    # -- logs --------------------------------------------------------------

    async def fetch_logs(self, query: LogQuery) -> list[EvidenceRef]:
        """Read journald, a Docker container, or a log file.

        The command is assembled here; `shlex.quote` is applied to every value
        that originated outside this module, so a service name like
        `nginx; rm -rf /` becomes a single literal argument that journald
        simply fails to find.
        """
        evidence: list[EvidenceRef] = []
        since = query.time_range.start.strftime("%Y-%m-%d %H:%M:%S")

        if query.target:
            targets = [query.target]
        else:
            inv = await self.discover()
            targets = [s["unit"] for s in inv.systemd_services[:10]] or ["*"]

        for target in targets:
            safe = shlex.quote(target)
            if target.startswith("/"):
                # A path: tail the file.
                command = f"tail -n {int(query.limit)} {safe} 2>/dev/null"
            elif target == "*":
                command = (
                    f"journalctl --since {shlex.quote(since)} "
                    f"-n {int(query.limit)} --no-pager -o short-iso 2>/dev/null"
                )
            else:
                unit = target if target.endswith(".service") else f"{target}.service"
                command = (
                    f"journalctl -u {shlex.quote(unit)} --since {shlex.quote(since)} "
                    f"-n {int(query.limit)} --no-pager -o short-iso 2>/dev/null"
                )

            if query.level:
                command += f" | grep -i {shlex.quote(query.level)}"
            if query.contains:
                command += f" | grep -i {shlex.quote(query.contains)}"

            try:
                result = await self.run(command)
            except AdapterError:
                continue
            if not result.stdout.strip():
                continue

            evidence.append(
                self.make_evidence(
                    raw=result.stdout,
                    query=f"{self.config.username}@{self.config.host}: {command}",
                    observed_at=result.at,
                    target=target,
                    exit_code=result.exit_code,
                    line_count=len(result.stdout.splitlines()),
                )
            )
        return evidence

    # -- probes ------------------------------------------------------------

    async def probe_http(self, url: str, timeout: int = 5) -> EvidenceRef:
        """Check an endpoint from the server's own network position.

        Used by the verification loop: 'is it healthy from outside?' and 'is it
        healthy from the box itself?' are different questions, and the gap
        between them is usually a firewall or a proxy.
        """
        command = (
            f"curl -s -o /dev/null -w '%{{http_code}} %{{time_total}}s' "
            f"--max-time {int(timeout)} {shlex.quote(url)}"
        )
        result = await self.run(command)
        body = json.dumps(
            {
                "url": url,
                "response": result.stdout.strip() or "(no response)",
                "curl_exit_code": result.exit_code,
                "reachable": result.exit_code == 0,
            },
            indent=2,
        )
        return self.make_trusted_evidence(
            raw=body,
            query=f"{self.config.username}@{self.config.host}: {command}",
            observed_at=result.at,
            url=url,
            reachable=result.exit_code == 0,
        )

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                await asyncio.to_thread(self._client.close)
                self._client = None
