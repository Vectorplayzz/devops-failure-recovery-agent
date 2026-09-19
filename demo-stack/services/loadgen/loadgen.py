"""Continuous background traffic.

Without live traffic a fault is invisible: no requests means no errors, no
latency samples and nothing in the logs. The load generator is what makes an
injected fault actually *manifest*, and what lets the verification loop observe
genuine recovery rather than the silence of an idle system.

It also tolerates failure quietly - a load generator that crashes when the
target returns 503 would take the demo down with it.
"""

from __future__ import annotations

import os
import random
import sys
import time

import httpx

TARGET = os.getenv("TARGET_URL", "http://orders-api:8080")
RPS = float(os.getenv("RPS", "2"))
SKUS = ["SKU-1000", "SKU-2110", "SKU-3300", "SKU-4821"]

# A slice of ordinary, harmless user agents. The injection payloads come from
# the app's own fault mode, not from here - the point is that an attacker needs
# only to make the app log something, not to control this generator.
AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) Safari/17.2",
    "checkout-mobile/3.8.1 (Android 14)",
    "Datadog/Synthetics",
]


def main() -> None:
    interval = 1.0 / RPS if RPS > 0 else 0.5
    sent = ok = failed = 0
    started = time.time()
    print(f"loadgen -> {TARGET} at {RPS} rps", flush=True)

    with httpx.Client(timeout=10.0) as client:
        while True:
            try:
                r = client.post(
                    f"{TARGET}/orders",
                    json={"sku": random.choice(SKUS), "qty": random.randint(1, 3)},
                    headers={"user-agent": random.choice(AGENTS)},
                )
                sent += 1
                if r.status_code == 200:
                    ok += 1
                else:
                    failed += 1
            except Exception:  # noqa: BLE001 - the target being down is the point
                sent += 1
                failed += 1

            if sent % 25 == 0:
                rate = failed / sent if sent else 0
                print(
                    f"[loadgen] sent={sent} ok={ok} failed={failed} "
                    f"error_rate={rate:.1%} uptime={time.time() - started:.0f}s",
                    flush=True,
                )
            time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
