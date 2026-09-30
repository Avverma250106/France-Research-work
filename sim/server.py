"""FastAPI server: runs the simulation and streams it to the browser.

Two channels on one socket, matching the engine's two clocks:
  - "positions", ~10 Hz, so vehicles animate smoothly
  - "hour", once per simulated hour, carrying predictions and error metrics

Run:  python3 -m uvicorn sim.server:app --port 8008
"""

import asyncio
import contextlib
import json
import logging
import traceback
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import Engine, SIM_HOUR_WALL_SECONDS
from .model_service import ModelService

log = logging.getLogger("sim")
STATIC = Path(__file__).parent / "static"
ART = Path("artifacts")
TICK = 0.1            # wall seconds between position frames
MAX_TICK_ERRORS = 20  # consecutive failures before the loop gives up

state = {"engine": None, "model": None, "clients": set(), "task": None,
         "status": "starting", "error": None, "tick_errors": 0}


def _boot():
    model = ModelService()
    wh = ws = None
    if (ART / "warmup_hourly.parquet").exists():
        wh = pd.read_parquet(ART / "warmup_hourly.parquet")
        ws = pd.read_parquet(ART / "warmup_sessions.parquet")
    return model, Engine(model, wh, ws)


async def _broadcast(payload):
    """Send to every client, dropping any that have gone away."""
    msg = json.dumps(payload)
    for ws in list(state["clients"]):
        try:
            await ws.send_text(msg)
        except Exception:
            state["clients"].discard(ws)


async def _loop():
    """Advance the simulation forever.

    A failure in one tick must not kill the loop - otherwise the page silently
    freezes with no indication of why. Errors are counted, logged and surfaced
    to the browser; the loop only gives up after MAX_TICK_ERRORS in a row.
    """
    while True:
        try:
            engine = state["engine"]
            hour_frame = engine.advance(TICK)
            state["tick_errors"] = 0
            state["status"] = "running"
            if state["clients"]:
                await _broadcast(engine.positions())
                if hour_frame is not None:
                    await _broadcast(hour_frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state["tick_errors"] += 1
            state["error"] = f"{type(exc).__name__}: {exc}"
            log.error("simulation tick failed (%d/%d)\n%s",
                      state["tick_errors"], MAX_TICK_ERRORS, traceback.format_exc())
            if state["tick_errors"] >= MAX_TICK_ERRORS:
                state["status"] = "failed"
                await _broadcast({"type": "error", "message": state["error"]})
                log.error("giving up after %d consecutive failures", MAX_TICK_ERRORS)
                return
            await asyncio.sleep(1.0)
            continue
        await asyncio.sleep(TICK)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"], state["engine"] = _boot()
        state["status"] = "running"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        log.error("startup failed: %s", state["error"])
        yield
        return
    state["task"] = asyncio.create_task(_loop())
    try:
        yield
    finally:
        task = state["task"]
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


app = FastAPI(title="IRVE load simulator", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico")
async def favicon():
    # Avoids a 404 per page load cluttering the server log.
    return Response(status_code=204)


@app.get("/health")
async def health():
    engine = state["engine"]
    return JSONResponse({
        "status": state["status"],
        "error": state["error"],
        "clients": len(state["clients"]),
        "sim_hours": engine.sim_hours if engine else 0,
        "sim_time": str(engine.now) if engine else None,
        "vehicles": len(engine.vehicles) if engine else 0,
        "charging": len(engine.active) if engine else 0,
        "data_source": state["model"].data_source if state["model"] else None,
    })


@app.get("/api/meta")
async def meta():
    if state["engine"] is None:
        return JSONResponse({"error": state["error"] or "not started"}, status_code=503)
    engine, model = state["engine"], state["model"]
    m = model.meta
    return JSONResponse({
        "data_source": model.data_source,
        "sites": engine.sites,
        "stations": engine.stations,
        "n_features": m.get("n_features"),
        "offline_metrics": m.get("metrics", {}),
        "n_sessions_trained": m.get("n_sessions"),
        "sim_hour_wall_seconds": SIM_HOUR_WALL_SECONDS,
        "routed": engine.routed,
    })


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    engine = state["engine"]
    if engine is None:
        await ws.send_text(json.dumps({
            "type": "error",
            "message": state["error"] or "simulation did not start",
        }))
        await ws.close()
        return

    state["clients"].add(ws)
    try:
        await ws.send_text(json.dumps({
            "type": "hello",
            "sites": engine.sites,
            "stations": engine.stations,
            "data_source": state["model"].data_source,
            "sim_hour_wall_seconds": SIM_HOUR_WALL_SECONDS,
            "routed": engine.routed,
        }))
        if engine.last_frame:
            await ws.send_text(json.dumps(engine.last_frame))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.debug("websocket closed unexpectedly", exc_info=True)
    finally:
        state["clients"].discard(ws)
