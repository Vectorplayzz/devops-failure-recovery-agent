"""Fault injector CLI.

    python inject.py list                 # catalogue with difficulty and symptoms
    python inject.py show oom             # full ground truth for one scenario
    python inject.py inject oom           # break production
    python inject.py inject oom --wait    # break it, then wait for the symptom
    python inject.py status               # what is currently broken
    python inject.py clear                # restore everything

Ground truth is deliberately reachable from the CLI (`show`) but never sent to
the agent. The agent gets telemetry; the grader gets the answer key.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import httpx

from scenarios import SCENARIOS, Scenario, get

# The Windows console defaults to cp1252, which cannot encode box-drawing or
# tick characters; printing one raises UnicodeEncodeError mid-run. Ask for
# UTF-8 where the stream supports it and fall back silently where it does not,
# and keep all markers in this file ASCII regardless.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

ORDERS = "http://localhost:18081"
PAYMENTS = "http://localhost:18082"
ORDERS_CONTAINER = "demo-orders-api"


def _post_fault(base: str, mode: str, intensity: float) -> dict | None:
    try:
        r = httpx.post(
            f"{base}/admin/fault", json={"mode": mode, "intensity": intensity}, timeout=5
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        print(f"  [!!] {base} unreachable: {exc}", file=sys.stderr)
        return None


def _container_state(name: str) -> dict:
    """Read the container's true state. This is the evidence the agent reads
    for the OOM scenario, so the injector reads it the same way."""
    try:
        out = subprocess.run(
            ["docker", "inspect", "--format", "{{json .State}}", name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return {"error": out.stderr.strip() or "container not found"}
        return json.loads(out.stdout.strip())
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_list(_: argparse.Namespace) -> int:
    print(f"\n{len(SCENARIOS)} scenarios\n")
    width = max(len(s.key) for s in SCENARIOS)
    for s in SCENARIOS:
        print(f"  {s.key:<{width}}  [{s.difficulty:^6}]  {s.title}")
        print(f"  {'':<{width}}             {s.symptom[:88]}")
        if s.decoy_remediation:
            print(f"  {'':<{width}}             decoy: {s.decoy_remediation[:78]}")
        print()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    s = get(args.scenario)
    print(f"\n{s.title}  [{s.key}]  difficulty={s.difficulty}\n")
    print(f"  symptom            {s.symptom}")
    print(f"  misleading signal  {s.misleading_signal}")
    print(f"\n  --- GROUND TRUTH (never shown to the agent) ---")
    print(f"  root cause         {s.true_root_cause}")
    print(f"  tags               {', '.join(s.true_cause_tags)}")
    print(f"  correct fix        {s.correct_remediation}")
    if s.decoy_remediation:
        print(f"  decoy fix          {s.decoy_remediation}")
        if s.decoy_relapse_seconds:
            print(f"  decoy relapses in  {s.decoy_relapse_seconds}s")
    print("\n  expected evidence")
    for e in s.expected_evidence:
        print(f"    - {e}")
    if s.notes:
        print(f"\n  notes              {s.notes}")
    print()
    return 0


def _apply(s: Scenario) -> None:
    if s.orders_fault != "none":
        r = _post_fault(ORDERS, s.orders_fault, s.intensity)
        if r:
            print(f"  orders-api    {r['previous']} -> {r['mode']}")
    if s.payments_fault != "none":
        r = _post_fault(PAYMENTS, s.payments_fault, s.intensity)
        if r:
            print(f"  payments-api  {r['previous']} -> {r['mode']}")


def cmd_inject(args: argparse.Namespace) -> int:
    s = get(args.scenario)
    print(f"\nInjecting: {s.title}  [{s.key}]\n")
    _apply(s)
    print(f"\n  expected symptom: {s.symptom}")

    if not args.wait:
        print(f"\n  Give it ~{s.warmup_seconds}s to manifest, then ask the agent.\n")
        return 0

    print(f"\n  Waiting up to {s.warmup_seconds}s for the symptom ...")
    deadline = time.time() + s.warmup_seconds
    while time.time() < deadline:
        time.sleep(3)
        state = _container_state(ORDERS_CONTAINER)
        if state.get("OOMKilled"):
            print(
                f"  [OK] orders-api OOM-killed "
                f"(exit {state.get('ExitCode')}, status {state.get('Status')})"
            )
            return 0
        try:
            r = httpx.get(f"{ORDERS}/health/deep", timeout=4)
            if r.status_code == 503:
                problems = r.json().get("problems", [])
                print(f"  [OK] deep health failing: {'; '.join(problems)}")
                return 0
        except httpx.HTTPError:
            print("  [OK] orders-api not responding")
            return 0
        print(f"    ... {int(deadline - time.time())}s remaining")

    print("  [!!] symptom did not appear in time - is loadgen running?")
    return 1


def cmd_status(_: argparse.Namespace) -> int:
    print()
    for name, base in (("orders-api", ORDERS), ("payments-api", PAYMENTS)):
        try:
            fault = httpx.get(f"{base}/admin/fault", timeout=4).json()
            print(f"  {name:<14} fault={fault.get('mode')} "
                  f"intensity={fault.get('intensity')} "
                  f"for={fault.get('active_for_seconds', '-')}s")
        except httpx.HTTPError as exc:
            print(f"  {name:<14} UNREACHABLE ({exc.__class__.__name__})")

    for name, base in (("orders-api", ORDERS), ("payments-api", PAYMENTS)):
        try:
            r = httpx.get(f"{base}/health/deep" if name == "orders-api" else f"{base}/health",
                          timeout=4)
            body = r.json()
            print(f"  {name:<14} health={body.get('status')} http={r.status_code}"
                  + (f" problems={body.get('problems')}" if body.get("problems") else ""))
        except httpx.HTTPError:
            print(f"  {name:<14} health=UNREACHABLE")

    state = _container_state(ORDERS_CONTAINER)
    if "error" not in state:
        print(f"  container       status={state.get('Status')} "
              f"exit={state.get('ExitCode')} oom_killed={state.get('OOMKilled')} "
              f"restarts={state.get('Restarting')}")
    print()
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    print("\nClearing all faults\n")
    for name, base in (("orders-api", ORDERS), ("payments-api", PAYMENTS)):
        r = _post_fault(base, "none", 1.0)
        if r:
            print(f"  {name:<14} {r['previous']} -> none")

    # Clearing the fault releases leaked memory and connections, but a process
    # already OOM-killed is gone and must be brought back.
    state = _container_state(ORDERS_CONTAINER)
    if state.get("Status") in {"exited", "dead"} or args.restart:
        print(f"  orders-api is {state.get('Status')} - restarting it")
        subprocess.run(["docker", "start", ORDERS_CONTAINER], capture_output=True)
        for _ in range(20):
            time.sleep(1)
            try:
                if httpx.get(f"{ORDERS}/health", timeout=3).status_code == 200:
                    print("  orders-api healthy again")
                    break
            except httpx.HTTPError:
                continue
        else:
            print("  [!!] orders-api did not come back - check `docker compose logs orders-api`")
    print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="inject", description="Break the demo production stack on purpose."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all scenarios").set_defaults(fn=cmd_list)

    show = sub.add_parser("show", help="print one scenario's ground truth")
    show.add_argument("scenario")
    show.set_defaults(fn=cmd_show)

    inj = sub.add_parser("inject", help="inject a scenario")
    inj.add_argument("scenario")
    inj.add_argument("--wait", action="store_true", help="block until the symptom appears")
    inj.set_defaults(fn=cmd_inject)

    sub.add_parser("status", help="show current fault and health state").set_defaults(
        fn=cmd_status
    )

    clr = sub.add_parser("clear", help="clear faults and restore health")
    clr.add_argument("--restart", action="store_true", help="restart orders-api regardless")
    clr.set_defaults(fn=cmd_clear)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
