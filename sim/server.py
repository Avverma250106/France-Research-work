"""FastAPI server: runs the simulation and streams it to the browser.

Two channels on one socket, matching the engine's two clocks:
  - "positions", ~10 Hz, so vehicles animate smoothly
  - "hour", once per simulated hour, carrying predictions and error metrics

Run:  python3 -m uvicorn sim.server:app --port 8008
"""

import asyncio
import contextlib
import inspect
import json
import logging
import os
import traceback
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import Engine, SIM_HOUR_WALL_SECONDS
from .model_service import ModelService
from .station_provider import DATASET_CONFIG
from marl.config import MAX_STATIONS

log = logging.getLogger("sim")
STATIC = Path(__file__).parent / "static"
ART = Path("artifacts")
TICK = 0.1            # wall seconds between position frames
MAX_TICK_ERRORS = 20  # consecutive failures before the loop gives up

state = {"engine": None, "model": None, "clients": set(), "task": None,
         "status": "starting", "error": None, "tick_errors": 0,
         "allocation_strategy": "proportional"}
location_switch_lock = asyncio.Lock()


def _engine_for(model, dataset, station_mode="osm"):
    wh = ws = None
    if station_mode == "synthetic" and (ART / "warmup_hourly.parquet").exists():
        wh = pd.read_parquet(ART / "warmup_hourly.parquet")
        ws = pd.read_parquet(ART / "warmup_sessions.parquet")
    station_cache = os.environ.get("SIM_STATION_CACHE")
    refresh_stations = os.environ.get("SIM_STATION_REFRESH", "").strip().lower()
    if refresh_stations not in {"", "0", "1", "false", "true", "no", "yes"}:
        raise ValueError("SIM_STATION_REFRESH must be a boolean value.")
    max_stations = int(os.environ.get(
        "MAX_STATIONS",
        os.environ.get("SIM_STATION_MAX", str(MAX_STATIONS)),
    ))
    if max_stations != MAX_STATIONS:
        raise ValueError(
            f"The saved MAPPO checkpoint is fixed to {MAX_STATIONS} agents; "
            f"MAX_STATIONS must equal {MAX_STATIONS}."
        )
    return Engine(
        model,
        wh,
        ws,
        station_mode=station_mode,
        dataset=dataset,
        station_cache=station_cache,
        refresh_stations=refresh_stations in {"1", "true", "yes"},
        max_stations=max_stations,
    )


def _boot():
    model = ModelService()
    station_mode = os.environ.get("SIM_STATION_MODE", "osm").strip().lower()
    dataset = os.environ.get("SIM_DATASET", "paris").strip().lower()
    return model, _engine_for(model, dataset, station_mode)


async def _broadcast(payload):
    """Send to every client, dropping any that have gone away."""
    msg = json.dumps(payload)
    for ws in list(state["clients"]):
        try:
            await ws.send_text(msg)
        except Exception:
            state["clients"].discard(ws)


def _select_strategy(engine):
    return getattr(engine, "allocation_strategy", "proportional")


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
        "geography": engine.geography,
        "dataset": engine.dataset,
        "station_mode": engine.station_mode,
        "station_count": len(engine.stations),
        "station_ids": [station["id"] for station in engine.stations],
        "agent_ids": list(engine.mappo.station_ids),
        "station_source": engine.geography["station_source"],
        "demand_source": engine.geography["demand_source"],
        "controller": engine.allocation_strategy,
        "allocation_strategy": engine.allocation_strategy,
        "n_features": m.get("n_features"),
        "offline_metrics": m.get("metrics", {}),
        "n_sessions_trained": m.get("n_sessions"),
        "sim_hour_wall_seconds": SIM_HOUR_WALL_SECONDS,
        "routed": engine.routed,
    })


@app.get("/api/station-datasets")
async def station_datasets(mode: str | None = None):
    engine = state["engine"]
    mode = str(
        mode
        or (engine.station_mode if engine else os.environ.get("SIM_STATION_MODE", "osm"))
    ).strip().lower()
    if mode not in {"osm", "synthetic"}:
        raise HTTPException(status_code=422, detail="mode must be 'osm' or 'synthetic'.")
    return JSONResponse({
        "selected": engine.dataset if engine else os.environ.get("SIM_DATASET", "paris"),
        "mode": engine.station_mode if engine else mode,
        "datasets": [
            {
                "id": dataset,
                "label": config["label"],
                "available": (
                    config["place_query"] is not None
                    if mode == "osm"
                    else dataset == "paris"
                ),
                "note": config["location_note"],
                "unavailable_reason": (
                    config["location_note"]
                    if mode == "osm" and config["place_query"] is None
                    else (
                        "Synthetic geography is configured only for the Paris "
                        "offline demo; use REAL mode for this city."
                        if mode == "synthetic" and dataset != "paris"
                        else None
                    )
                ),
            }
            for dataset, config in DATASET_CONFIG.items()
        ],
    })


@app.post("/api/allocation-strategy")
async def select_allocation_strategy(payload: dict):
    engine = state["engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail="The simulation engine is not available.")
    strategy = str(payload.get("strategy", "")).strip().lower()
    if strategy not in {"proportional", "mappo"}:
        raise HTTPException(
            status_code=422,
            detail="Strategy must be either 'proportional' or 'mappo'.",
        )
    engine.set_allocation_strategy(strategy)
    msg = {
        "type": "strategy",
        "strategy": engine.allocation_strategy,
        "controller": engine.allocation_strategy,
        "allocation_strategy": engine.allocation_strategy,
    }
    result = _broadcast(msg)
    if inspect.isawaitable(result):
        await result
    return JSONResponse({
        "strategy": engine.allocation_strategy,
        "controller": engine.allocation_strategy,
        "allocation_strategy": engine.allocation_strategy,
        "available": ["proportional", "mappo"],
    })


@app.post("/api/station-dataset")
async def select_station_dataset(payload: dict):
    dataset = str(payload.get("dataset", "")).strip().lower()
    mode = str(payload.get("mode", "osm")).strip().lower()
    if dataset not in DATASET_CONFIG:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown station dataset {dataset!r}.",
        )
    if mode not in {"osm", "synthetic"}:
        raise HTTPException(status_code=422, detail="mode must be 'osm' or 'synthetic'.")
    if mode == "osm" and DATASET_CONFIG[dataset]["place_query"] is None:
        raise HTTPException(
            status_code=422,
            detail=DATASET_CONFIG[dataset]["location_note"],
        )
    if mode == "synthetic" and dataset != "paris":
        raise HTTPException(
            status_code=422,
            detail=(
                "Synthetic station geography is configured only for the Paris "
                "offline demo; select REAL mode for other dataset locations."
            ),
        )
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="The model is not available.")
    async with location_switch_lock:
        try:
            new_engine = await asyncio.to_thread(
                _engine_for, state["model"], dataset, mode
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        state["engine"] = new_engine
        state["status"] = "running"
        state["error"] = None
        hello = {
            "type": "hello",
            "sites": new_engine.sites,
            "stations": new_engine.stations,
            "geography": new_engine.geography,
            "data_source": state["model"].data_source,
            "station_count": len(new_engine.stations),
            "station_ids": [station["id"] for station in new_engine.stations],
            "agent_ids": list(new_engine.mappo.station_ids),
            "sim_hour_wall_seconds": SIM_HOUR_WALL_SECONDS,
            "routed": new_engine.routed,
            "strategy": _select_strategy(new_engine),
            "controller": _select_strategy(new_engine),
            "allocation_strategy": _select_strategy(new_engine),
        }
        await _broadcast(hello)
    return JSONResponse({
        "dataset": dataset,
        "station_mode": mode,
        "geography": new_engine.geography,
        "station_count": len(new_engine.stations),
        "station_ids": [station["id"] for station in new_engine.stations],
        "agent_ids": list(new_engine.mappo.station_ids),
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
            "geography": engine.geography,
            "data_source": state["model"].data_source,
            "station_count": len(engine.stations),
            "station_ids": [station["id"] for station in engine.stations],
            "agent_ids": list(engine.mappo.station_ids),
            "sim_hour_wall_seconds": SIM_HOUR_WALL_SECONDS,
            "routed": engine.routed,
            "strategy": _select_strategy(engine),
            "controller": _select_strategy(engine),
            "allocation_strategy": _select_strategy(engine),
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
