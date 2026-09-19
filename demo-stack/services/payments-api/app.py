"""payments-api - the downstream dependency.

Exists so that failures can *propagate*. A single-service demo lets an agent
guess the cause correctly by luck; with a dependency in the path, the agent has
to distinguish "orders-api is broken" from "orders-api is fine and its
dependency is broken", which is where naive diagnosis falls over.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

SERVICE = "payments-api"
VERSION = os.getenv("APP_VERSION", "2.0.1")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": SERVICE,
            "version": VERSION,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)  # type: ignore[attr-defined]
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
log = logging.getLogger(SERVICE)
log.setLevel(logging.INFO)
log.addHandler(handler)
log.propagate = False


def emit(level: int, msg: str, **fields: Any) -> None:
    log.log(level, msg, extra={"extra_fields": fields})


VALID_MODES = {"none", "slow", "down", "flaky"}

_state = {"mode": "none", "intensity": 1.0}
_counters = {"authorised": 0, "declined": 0, "calls": 0}
_lock = threading.Lock()

app = FastAPI(title=SERVICE, version=VERSION)


@app.get("/health")
def health() -> Response:
    mode = _state["mode"]
    if mode == "down":
        emit(logging.ERROR, "health check failing", mode=mode)
        return Response(
            content=json.dumps({"status": "down", "service": SERVICE}),
            status_code=503,
            media_type="application/json",
        )
    return Response(
        content=json.dumps({"status": "ok", "service": SERVICE, "version": VERSION}),
        media_type="application/json",
    )


class AuthIn(BaseModel):
    amount: int


@app.post("/authorise")
def authorise(body: AuthIn) -> dict[str, Any]:
    with _lock:
        _counters["calls"] += 1
        n = _counters["calls"]
    mode = _state["mode"]

    if mode == "down":
        emit(logging.ERROR, "authorisation refused, processor unreachable",
             upstream="acquirer-gateway", error="ConnectionRefusedError")
        raise HTTPException(503, "payment processor unreachable")

    if mode == "slow":
        time.sleep(min(4.0 * _state["intensity"], 10.0))

    if mode == "flaky" and n % 3 == 0:
        emit(logging.ERROR, "transient authorisation failure",
             error="GatewayTimeout", attempt=n)
        raise HTTPException(502, "acquirer gateway timeout")

    with _lock:
        _counters["authorised"] += 1
    emit(logging.INFO, "payment authorised", amount=body.amount, txn=n)
    return {"authorised": True, "amount": body.amount, "txn": n}


@app.get("/metrics")
def metrics() -> Response:
    with _lock:
        c = dict(_counters)
    body = "\n".join(
        [
            "# TYPE payments_calls_total counter",
            f"payments_calls_total {c['calls']}",
            "# TYPE payments_authorised_total counter",
            f"payments_authorised_total {c['authorised']}",
            "",
        ]
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")


class FaultIn(BaseModel):
    mode: str
    intensity: float = 1.0


@app.post("/admin/fault")
def set_fault(f: FaultIn) -> dict[str, Any]:
    if f.mode not in VALID_MODES:
        raise HTTPException(400, f"unknown mode {f.mode!r}; valid: {sorted(VALID_MODES)}")
    previous = _state["mode"]
    _state["mode"], _state["intensity"] = f.mode, f.intensity
    emit(logging.WARNING, "fault mode changed", previous=previous, mode=f.mode)
    return {"mode": f.mode, "previous": previous}


@app.get("/admin/fault")
def get_fault() -> dict[str, Any]:
    return dict(_state)
