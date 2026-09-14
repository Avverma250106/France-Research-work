"""FastAPI server: runs the simulation and streams it to the browser.

Two channels on one socket, matching the engine's two clocks:
  - "positions", ~10 Hz, so vehicles animate smoothly
  - "hour", once per simulated hour, carrying predictions and error metrics

Run:  python3 -m uvicorn sim.server:app --port 8008
"""

import asyncio
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import Engine, SIM_HOUR_WALL_SECONDS
from .model_service import ModelService

STATIC = Path(__file__).parent / "static"
ART = Path("artifacts")
TICK = 0.1  # wall seconds between position frames

app = FastAPI(title="IRVE load simulator")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

state = {"engine": None, "model": None, "clients": set(), "task": None}


def _boot():
    model = ModelService()
    wh = ws = None
    if (ART / "warmup_hourly.parquet").exists():
        wh = pd.read_parquet(ART / "warmup_hourly.parquet")
        ws = pd.read_parquet(ART / "warmup_sessions.parquet")
    return model, Engine(model, wh, ws)


async def _broadcast(payload):
    dead = []
    msg = json.dumps(payload)
    for ws in list(state["clients"]):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state["clients"].discard(ws)


async def _loop():
    engine = state["engine"]
    while True:
        hour_frame = engine.advance(TICK)
        if state["clients"]:
            await _broadcast(engine.positions())
            if hour_frame is not None:
                await _broadcast(hour_frame)
        await asyncio.sleep(TICK)


@app.on_event("startup")
async def startup():
    state["model"], state["engine"] = _boot()
    state["task"] = asyncio.create_task(_loop())


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
async def meta():
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
    })


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state["clients"].add(ws)
    engine = state["engine"]
    try:
        await ws.send_text(json.dumps({
            "type": "hello",
            "sites": engine.sites,
            "stations": engine.stations,
            "data_source": state["model"].data_source,
            "sim_hour_wall_seconds": SIM_HOUR_WALL_SECONDS,
        }))
        if engine.last_frame:
            await ws.send_text(json.dumps(engine.last_frame))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state["clients"].discard(ws)
