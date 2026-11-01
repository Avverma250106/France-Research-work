"""Simulation engine: vehicles on a fast clock, the model on an hourly clock.

Vehicles move continuously so the map animates smoothly. The model fires only
when a simulated hour closes, because that is the resolution it was trained at.
The simulation is the source of ground truth - it knows exactly how much energy
each session drew - so predicted-vs-actual is genuine rather than replayed.

Energy is allocated to hours with the same proportional rule as
session_to_hourly.split_session_energy. If it were allocated any other way,
predicted and actual would not be measuring the same quantity.
"""

import math
from collections import deque

import numpy as np
import pandas as pd

from marl.baseline import proportional_allocation
from marl.config import MAX_STATIONS

from . import stations as st
from . import synth
from .demand_profiles import HistoricalDemandProfile
# KEEP_HOURS is re-exported: the engine owns the ring buffer, so callers
# (and test_sim_forecast) reasonably look for the window length here.
from .features_rt import KEEP_HOURS, FeatureBuffer
from .mappo_runtime import MAPPOController
from .roads import RoadNetwork
from .station_provider import DATASET_CONFIG

SIM_HOUR_WALL_SECONDS = 8.0        # 1 simulated hour per 8 wall seconds
METRIC_WINDOW = 72                  # rolling window, in simulated hours



class Session:
    __slots__ = ("entity_id", "station_id", "connect", "end", "energy_kwh", "departed")

    def __init__(self, entity_id, station_id, connect, end, energy_kwh):
        self.departed = False
        self.entity_id = entity_id
        self.station_id = station_id
        self.connect = connect
        self.end = end
        self.energy_kwh = energy_kwh

    def allocated(self, h0, h1):
        """kWh falling in [h0, h1), proportional to overlap - matches training."""
        total_min = (self.end - self.connect).total_seconds() / 60
        if total_min <= 0:
            return 0.0
        s = max(self.connect, h0)
        e = min(self.end, h1)
        overlap = (e - s).total_seconds() / 60
        if overlap <= 0:
            return 0.0
        return self.energy_kwh * (overlap / total_min)


class Vehicle:
    __slots__ = ("id", "path", "depart", "arrive", "station", "entity_id", "session", "kind")

    def __init__(self, vid, path, depart, arrive, station, entity_id, session, kind="in"):
        self.kind = kind
        self.id = vid
        self.path = path
        self.depart = depart
        self.arrive = arrive
        self.station = station
        self.entity_id = entity_id
        self.session = session

    def position(self, now):
        frac = (now - self.depart).total_seconds() / max(
            1.0, (self.arrive - self.depart).total_seconds())
        frac = min(max(frac, 0.0), 1.0)
        n = len(self.path) - 1
        seg = min(int(frac * n), n - 1)
        local = frac * n - seg
        (x0, y0), (x1, y1) = self.path[seg], self.path[seg + 1]
        return [x0 + (x1 - x0) * local, y0 + (y1 - y0) * local]


def _street_path(rng, origin, dest):
    """A few axis-aligned legs with jitter, so movement reads as streets."""
    ox, oy = origin
    dx, dy = dest
    mid_x = ox + (dx - ox) * rng.uniform(0.45, 0.75)
    mid_y = oy + (dy - oy) * rng.uniform(0.25, 0.55)
    j = 0.0035
    return [
        [ox, oy],
        [mid_x + rng.normal(0, j), oy + rng.normal(0, j)],
        [mid_x + rng.normal(0, j), mid_y + rng.normal(0, j)],
        [dx + rng.normal(0, j), mid_y + rng.normal(0, j)],
        [dx, dy],
    ]


class Engine:
    def __init__(
        self,
        model,
        warmup_hourly=None,
        warmup_sessions=None,
        seed=5,
        *,
        station_mode="synthetic",
        dataset="paris",
        station_cache=None,
        refresh_stations=False,
        max_stations=MAX_STATIONS,
    ):
        self.rng = np.random.default_rng(seed)
        self.model = model
        self.station_mode = station_mode
        self.dataset = str(dataset).strip().lower()
        self.demand_profile = None
        expanding_state = None
        if station_mode == "osm":
            if max_stations != MAX_STATIONS:
                raise ValueError(
                    f"The existing MAPPO checkpoint requires exactly "
                    f"{MAX_STATIONS} stations."
                )
            self.sites, self.stations, self.geography = st.build_osm(
                self.dataset,
                cache_path=station_cache,
                refresh=refresh_stations,
                max_stations=max_stations,
            )
            dataset_config = DATASET_CONFIG[self.dataset]
            self.timezone = dataset_config["timezone"]
            source_dataset = self.dataset
            region = dataset_config["region"]
            self.demand_profile = HistoricalDemandProfile.load(self.dataset)
            warmup_hourly, warmup_sessions, expanding_state = (
                self.demand_profile.warmup()
            )
            self.geography.update({
                "station_source": "openstreetmap",
                "demand_source": "historical_profile",
                "demand_profile_entity": self.demand_profile.entity_id,
                "controller": "MAPPO",
                "station_mode": "osm",
            })
            coordinates = [(s["lon"], s["lat"]) for s in self.stations]
            self.centre = (
                sum(point[0] for point in coordinates) / len(coordinates),
                sum(point[1] for point in coordinates) / len(coordinates),
            )
        elif station_mode == "synthetic":
            if self.dataset != "paris":
                raise ValueError(
                    "Synthetic geography is available only for the Paris demo."
                )
            self.sites, all_stations = st.build()
            self.stations = [
                next(
                    station for station in all_stations
                    if station["entity_id"] == site["entity_id"]
                )
                for site in self.sites
            ]
            if len(self.stations) != MAX_STATIONS:
                raise ValueError(
                    "The existing MAPPO checkpoint requires exactly "
                    f"{MAX_STATIONS} synthetic demo stations."
                )
            self.timezone = "Europe/Paris"
            source_dataset = "paris"
            region = "France"
            self.centre = st.CENTRE
            self.geography = {
                "dataset": "paris",
                "dataset_label": "Paris, France",
                "provider": "synthetic",
                "attribution": None,
                "cache_used": False,
                "location_note": "Station coordinates and demand are synthetic.",
                "station_source": "synthetic",
                "demand_source": "synthetic_profile",
                "controller": "MAPPO",
                "station_mode": "synthetic",
            }
        else:
            raise ValueError("station_mode must be 'synthetic' or 'osm'.")
        self.entities = [s["entity_id"] for s in self.sites]
        self.by_entity = {}
        self.station_xy = {}
        for s in self.stations:
            self.by_entity.setdefault(s["entity_id"], []).append(s)
            self.station_xy[s["id"]] = (s["lon"], s["lat"])
        self.mappo = MAPPOController(self.stations)
        if tuple(self.mappo.station_ids) != tuple(s["id"] for s in self.stations):
            raise ValueError("MAPPO agent IDs must exactly match map station IDs.")
        self.allocation_strategy = "proportional"
        self._pending_allocation_action = None
        self._pending_allocation_forecast = None
        self._pending_strategy_metrics = None
        self._strategy_history = []

        self.buffer = FeatureBuffer(
            self.entities, source_dataset=source_dataset, region=region
        )
        if expanding_state is None:
            expanding_state = model.meta.get("expanding_state")
        if warmup_hourly is not None and len(warmup_hourly):
            self.buffer.warm(warmup_hourly, warmup_sessions,
                             prior=expanding_state)

        last = self.buffer.last_hour()
        self.now = (last + pd.Timedelta(hours=1)) if last is not None \
            else pd.Timestamp("2025-01-06 00:00", tz="UTC")
        self.hour_start = self.now.floor("h")
        self.initial_mappo_frame = None
        if station_mode == "osm" and last is not None:
            initial_current = self._distribute_profile_demand({
                entity: self.buffer.hourly[entity][-1][1]
                for entity in self.entities
            })
            initial_forecast = self._forecast_next_station_demand()
            self.initial_mappo_frame = self.mappo.advance(
                initial_current, initial_forecast
            )
            self.initial_mappo_frame["forecast_hour"] = self.now.isoformat()

        # Real street routing when the cached road graph is available; the
        # approximated path is kept as a fallback so the simulation still runs
        # without it (build it with: python3 -m sim.roads).
        self.road = RoadNetwork.load() if self.dataset == "paris" else None
        self.routed = self.road is not None
        # A fixed pool of origins keeps the router's cache hot: the same
        # (origin, station) pairs recur instead of every trip being unique.
        self.origins = [
            [self.centre[0] + r * math.cos(a), self.centre[1] + r * math.sin(a) * 0.7]
            for a, r in ((i * 2 * math.pi / 24, 0.030 + 0.055 * ((i % 3) / 2.0))
                         for i in range(24))
        ]
        self.vehicles = {}
        self.pending = deque()
        self.active = []            # sessions still charging
        self.hour_sessions = []     # sessions that connected in the current hour
        self.vid = 0

        # one entry per site per hour, so the cap is scaled by site count
        self.history = deque(maxlen=METRIC_WINDOW * len(self.entities))
        self.per_site = {e: deque(maxlen=METRIC_WINDOW) for e in self.entities}
        self.last_actual = {
            e: (self.buffer.hourly[e][-1][1] if self.buffer.hourly[e] else 0.0)
            for e in self.entities
        }
        self.sim_hours = 0
        self.frame = 0
        self.last_frame = None
        self._schedule_hour(self.hour_start)

    def set_allocation_strategy(self, strategy):
        strategy_name = str(strategy).strip().lower()
        if strategy_name not in {"proportional", "mappo"}:
            raise ValueError("allocation strategy must be 'proportional' or 'mappo'.")
        self.allocation_strategy = strategy_name
        self._pending_allocation_action = None
        self._pending_allocation_forecast = None
        self._pending_strategy_metrics = None
        return self.allocation_strategy

    def _allocation_action(self, actual_by_station, forecast_by_station):
        capacities = self.mappo.capacities
        renewable_available = self.mappo.environment.renewable_available
        grid_capacity = self.mappo.environment.grid_capacity
        current_grid_load = self.mappo.environment.current_grid_load
        if self.allocation_strategy == "proportional":
            return proportional_allocation(
                forecast_by_station,
                station_capacities=capacities,
                renewable_available=renewable_available,
                grid_capacity=grid_capacity,
                current_grid_load=current_grid_load,
            )
        if self.allocation_strategy == "mappo":
            return self.mappo.plan_action(actual_by_station, forecast_by_station)
        raise ValueError(f"Unsupported allocation strategy: {self.allocation_strategy!r}")

    def _execute_strategy_step(self, actual_by_station, forecast_by_station):
        if self._pending_allocation_action is None:
            self.mappo.environment.reset(
                actual_by_station,
                forecast_by_station,
                forecast_source="model",
            )
            action = self._allocation_action(actual_by_station, forecast_by_station)
            self._pending_allocation_action = action
            self._pending_allocation_forecast = dict(forecast_by_station)
            self._pending_strategy_metrics = {
                "strategy": self.allocation_strategy,
                "actual_energy_served": 0.0,
                "actual_unserved": 0.0,
                "renewable_used": 0.0,
                "grid_used": 0.0,
                "renewable_utilization": 0.0,
                "global_reward": 0.0,
            }
            return action, self._pending_strategy_metrics

        rewards, _, info = self.mappo.environment.step(
            self._pending_allocation_action,
            self._pending_allocation_forecast,
            lambda: dict(actual_by_station),
            forecast_source="model",
        )
        action = self._allocation_action(actual_by_station, forecast_by_station)
        self._pending_allocation_action = action
        self._pending_allocation_forecast = dict(forecast_by_station)
        self._pending_strategy_metrics = {
            "strategy": self.allocation_strategy,
            "actual_energy_served": float(info["actual_energy_served"]),
            "actual_unserved": float(info["actual_unserved"]),
            "renewable_used": float(info["renewable_used"]),
            "grid_used": float(info["grid_used"]),
            "renewable_utilization": (
                float(info["renewable_used"]) / float(info["renewable_available"])
                if float(info["renewable_available"]) > 0 else 0.0
            ),
            "global_reward": float(sum(rewards.values())),
            "reward_components": info.get("reward_components", {}),
            "allocations": info.get("allocations", {}),
            "forecast": dict(forecast_by_station),
            "actual_demand": dict(actual_by_station),
        }
        return action, self._pending_strategy_metrics

    # ---------- arrivals ----------

    def _schedule_hour(self, hour):
        local = hour.tz_convert(self.timezone)
        for site in self.sites:
            e = site["entity_id"]
            if self.demand_profile is not None:
                lam = self.demand_profile.arrival_rate(
                    local.hour, local.weekday() >= 5
                )
            else:
                lam = synth.arrival_rate(
                    e, local.hour, local.weekday() >= 5,
                    dayofyear=local.dayofyear,
                )
            n = self.rng.poisson(lam)
            for _ in range(n):
                arrive = hour + pd.Timedelta(minutes=float(self.rng.uniform(0, 60)))
                travel = float(self.rng.uniform(20, 50))
                self.pending.append((arrive - pd.Timedelta(minutes=travel), arrive, e))

    def _distribute_profile_demand(self, profile_demand):
        capacities = self.mappo.capacities
        station_demand = {}
        for entity in self.entities:
            group = [
                station for station in self.stations
                if station["entity_id"] == entity
            ]
            total = max(0.0, float(profile_demand.get(entity, 0.0)))
            capacity_total = sum(capacities[station["id"]] for station in group)
            for station in group:
                station_demand[station["id"]] = (
                    total * capacities[station["id"]] / capacity_total
                    if capacity_total > 0
                    else total / len(group)
                )
        if set(station_demand) != set(self.mappo.station_ids):
            raise ValueError("Demand station IDs do not match the MAPPO agents.")
        return station_demand

    def _forecast_next_station_demand(self):
        next_rows = self.buffer.feature_rows_for_next_hour()
        if next_rows is None or len(next_rows) != len(self.entities):
            raise ValueError(
                "The leakage-safe feature buffer did not produce one next-hour "
                "forecast row per dataset profile."
            )
        prediction, _ = self.model.predict(next_rows)
        if len(prediction) != len(next_rows):
            raise ValueError("The model returned a mismatched next-hour forecast.")
        by_entity = {
            str(entity): float(value)
            for entity, value in zip(next_rows["entity_id"], prediction)
        }
        return self._distribute_profile_demand(by_entity)

    def _outer_point(self):
        return list(self.origins[self.rng.integers(len(self.origins))])

    def _path(self, origin, dest):
        """Street route when the road graph is loaded, approximation otherwise."""
        if self.road is not None:
            r = self.road.route(tuple(origin), tuple(dest))
            if r and len(r) > 2:
                return r
        return _street_path(self.rng, list(origin), list(dest))

    def _spawn_departure(self, session):
        """A finished session drives away, so the map reflects the whole fleet
        rather than only the vehicles arriving."""
        xy = self.station_xy.get(session.station_id)
        if xy is None:
            return
        travel = float(self.rng.uniform(20, 50))
        self.vid += 1
        v = Vehicle(f"o{self.vid}",
                    self._path(list(xy), self._outer_point()),
                    self.now, self.now + pd.Timedelta(minutes=travel),
                    session.station_id, session.entity_id, None, kind="out")
        self.vehicles[v.id] = v

    def _spawn(self, depart, arrive, entity):
        station = self.by_entity[entity][self.rng.integers(len(self.by_entity[entity]))]
        origin = self._outer_point()
        dest = [station["lon"], station["lat"]]

        if self.demand_profile is not None:
            local = arrive.tz_convert(self.timezone)
            dur_h, energy = self.demand_profile.sample_session(
                self.rng, local.hour, local.weekday() >= 5
            )
        else:
            dur_h, energy = synth.sample_session(self.rng, entity)
        session = Session(entity, station["id"], arrive,
                          arrive + pd.Timedelta(hours=dur_h), energy)

        self.vid += 1
        v = Vehicle(f"v{self.vid}", self._path(origin, dest),
                    depart, arrive, station["id"], entity, session)
        self.vehicles[v.id] = v

    # ---------- clock ----------

    def advance(self, wall_dt):
        sim_dt = pd.Timedelta(seconds=wall_dt * 3600.0 / SIM_HOUR_WALL_SECONDS)
        target = self.now + sim_dt
        closed = None
        while self.hour_start + pd.Timedelta(hours=1) <= target:
            self.now = self.hour_start + pd.Timedelta(hours=1)
            self._drain()
            closed = self._close_hour()
        self.now = target
        self._drain()
        return closed

    def _drain(self):
        while self.pending and self.pending[0][0] <= self.now:
            depart, arrive, entity = self.pending.popleft()
            self._spawn(depart, arrive, entity)
        done = [vid for vid, v in self.vehicles.items() if v.arrive <= self.now]
        for vid in done:
            v = self.vehicles.pop(vid)
            if v.session is not None:          # inbound: plug in
                self.active.append(v.session)
                self.hour_sessions.append(v.session)
        for s in self.active:                  # finished charging: drive away
            if not s.departed and s.end <= self.now:
                s.departed = True
                self._spawn_departure(s)

    def _close_hour(self):
        h0, h1 = self.hour_start, self.hour_start + pd.Timedelta(hours=1)

        actual, started = {}, {}
        actual_by_station = {station["id"]: 0.0 for station in self.stations}
        for e in self.entities:
            actual[e] = 0.0
            started[e] = []
        for s in self.active:
            if s.end > h0 and s.connect < h1:
                energy = s.allocated(h0, h1)
                actual[s.entity_id] += energy
                actual_by_station[s.station_id] += energy
        for s in self.hour_sessions:
            started[s.entity_id].append(s.connect)

        for e in self.entities:
            self.buffer.append(e, h0, actual[e], started[e])

        rows = self.buffer.feature_rows()
        preds, probs = {}, {}
        if rows is not None and len(rows):
            p, pr = self.model.predict(rows)
            for i, e in enumerate(rows["entity_id"]):
                preds[e] = float(p[i])
                probs[e] = float(pr[i])

        forecast_by_station = self._forecast_next_station_demand()
        strategy_action, strategy_step = self._execute_strategy_step(
            actual_by_station,
            forecast_by_station,
        )
        mappo_frame = {
            "station_count": len(self.mappo.station_ids),
            "station_ids": list(self.mappo.station_ids),
            "agent_ids": list(self.mappo.station_ids),
            "forecast_demand": dict(forecast_by_station),
            "current_demand": dict(actual_by_station),
            "capacity_kwh": dict(self.mappo.capacities),
            "capacity_source": dict(self.mappo.capacity_sources),
            "renewable_availability_kwh": self.mappo.environment.renewable_available,
            "grid_capacity_kwh": self.mappo.environment.grid_capacity,
            "next_hour_plan": strategy_action,
            "planned_renewable_request": {
                station: allocation["renewable"]
                for station, allocation in strategy_action.items()
            },
            "planned_grid_request": {
                station: allocation["grid"]
                for station, allocation in strategy_action.items()
            },
            "renewable_allocation": {
                station: strategy_step.get("allocations", {}).get(station, {}).get("renewable", 0.0)
                for station in self.mappo.station_ids
            },
            "grid_allocation": {
                station: strategy_step.get("allocations", {}).get(station, {}).get("grid", 0.0)
                for station in self.mappo.station_ids
            },
            "renewable_utilization": strategy_step.get("renewable_utilization"),
            "completed_step": strategy_step,
            "controller": self.allocation_strategy,
            "strategy": self.allocation_strategy,
            "forecast_hour": h1.isoformat(),
        }

        site_frames = []
        for site in self.sites:
            e = site["entity_id"]
            a = actual[e]
            pred = preds.get(e, 0.0)
            pers = self.last_actual[e]
            self.per_site[e].append((pred, a, pers))
            self.history.append((pred, a, pers))
            site_frames.append({
                "entity_id": e, "label": site["label"],
                "lon": site["lon"], "lat": site["lat"],
                "kwh_actual": round(a, 3),
                "kwh_pred": round(pred, 3),
                "kwh_persistence": round(pers, 3),
                "p_charge": round(probs.get(e, 0.0), 4),
                "abs_err": round(abs(pred - a), 3),
                "sessions_started": len(started[e]),
                "site_metrics": _metrics(self.per_site[e]),
            })
            self.last_actual[e] = a

        self.active = [s for s in self.active if s.end > h1]
        self.hour_sessions = []
        self.hour_start = h1
        self.sim_hours += 1
        self._schedule_hour(h1)

        step = len(self.entities)
        recent = list(self.history)[-48 * step:]
        series = []
        for i in range(0, len(recent) - step + 1, step):
            chunk = recent[i:i + step]
            series.append({
                "p": round(sum(c[0] for c in chunk), 2),
                "a": round(sum(c[1] for c in chunk), 2),
                "q": round(sum(c[2] for c in chunk), 2),
            })

        self.last_frame = {
            "type": "hour",
            "series": series,
            "sim_hour": h0.isoformat(),
            "sim_hours_elapsed": self.sim_hours,
            "sites": site_frames,
            "overall": {**_metrics(self.history),
                        "hours": len(self.history) // len(self.entities)},
            "n_active_sessions": len(self.active),
            "station_source": self.geography["station_source"],
            "demand_source": self.geography["demand_source"],
            "controller": self.allocation_strategy,
            "strategy": self.allocation_strategy,
            "allocation": {
                "strategy": self.allocation_strategy,
                "action": strategy_action,
                "metrics": strategy_step,
            },
            "mappo": mappo_frame,
        }
        return self.last_frame

    # ---------- fast channel ----------

    def positions(self):
        # Route lines change only when a vehicle spawns or finishes, so they go
        # out at 1 Hz while positions stream at 10 Hz. Sending both every frame
        # is ~128 KB/s of geometry that nobody can see change.
        self.frame += 1
        send_routes = (self.frame % 10 == 1)

        moving, routes = [], []
        for v in self.vehicles.values():
            lon, lat = v.position(self.now)
            moving.append({"id": v.id, "lon": round(lon, 6), "lat": round(lat, 6),
                           "e": v.entity_id, "k": v.kind})
            if send_routes:
                routes.append(_thin(v.path))

        charging = []
        for s in self.active:
            xy = self.station_xy.get(s.station_id)
            if xy and s.connect <= self.now < s.end:
                charging.append({"lon": round(xy[0], 6), "lat": round(xy[1], 6),
                                 "e": s.entity_id})
        frame = {"type": "positions", "sim_time": self.now.isoformat(),
                 "vehicles": moving, "charging": charging,
                 "n_charging": len(charging)}
        if send_routes:
            frame["routes"] = routes
        return frame


def _thin(path, max_pts=36):
    """Reduce a route to at most max_pts points for transmission."""
    n = len(path)
    if n <= max_pts:
        return [[round(x, 6), round(y, 6)] for x, y in path]
    step = n / float(max_pts - 1)
    idx = sorted({int(i * step) for i in range(max_pts - 1)} | {n - 1})
    return [[round(path[i][0], 6), round(path[i][1], 6)] for i in idx]


def _metrics(hist):
    if not hist:
        return {"n": 0, "mae": 0.0, "rmse": 0.0, "wape": 0.0,
                "persistence_mae": 0.0, "persistence_rmse": 0.0}
    p = np.array([h[0] for h in hist])
    a = np.array([h[1] for h in hist])
    q = np.array([h[2] for h in hist])
    denom = np.abs(a).sum()
    return {
        "n": len(hist),
        "mae": round(float(np.abs(p - a).mean()), 4),
        "rmse": round(float(np.sqrt(((p - a) ** 2).mean())), 4),
        "wape": round(float(np.abs(p - a).sum() / denom * 100) if denom > 0 else 0.0, 2),
        "persistence_mae": round(float(np.abs(q - a).mean()), 4),
        "persistence_rmse": round(float(np.sqrt(((q - a) ** 2).mean())), 4),
    }
