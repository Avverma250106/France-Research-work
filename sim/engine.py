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

from . import stations as st
from . import synth
from .features_rt import FeatureBuffer

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
    def __init__(self, model, warmup_hourly=None, warmup_sessions=None, seed=5):
        self.rng = np.random.default_rng(seed)
        self.model = model
        self.sites, self.stations = st.build()
        self.entities = [s["entity_id"] for s in self.sites]
        self.by_entity = {}
        self.station_xy = {}
        for s in self.stations:
            self.by_entity.setdefault(s["entity_id"], []).append(s)
            self.station_xy[s["id"]] = (s["lon"], s["lat"])

        self.buffer = FeatureBuffer(self.entities)
        if warmup_hourly is not None and len(warmup_hourly):
            self.buffer.warm(warmup_hourly, warmup_sessions,
                             prior=model.meta.get("expanding_state"))

        last = self.buffer.last_hour()
        self.now = (last + pd.Timedelta(hours=1)) if last is not None \
            else pd.Timestamp("2025-01-06 00:00", tz="UTC")
        self.hour_start = self.now.floor("h")

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
        self.last_frame = None
        self._schedule_hour(self.hour_start)

    # ---------- arrivals ----------

    def _schedule_hour(self, hour):
        local = hour.tz_convert("Europe/Paris")
        for site in self.sites:
            e = site["entity_id"]
            lam = synth.arrival_rate(e, local.hour, local.weekday() >= 5,
                                     dayofyear=local.dayofyear)
            n = self.rng.poisson(lam)
            for _ in range(n):
                arrive = hour + pd.Timedelta(minutes=float(self.rng.uniform(0, 60)))
                travel = float(self.rng.uniform(20, 50))
                self.pending.append((arrive - pd.Timedelta(minutes=travel), arrive, e))

    def _outer_point(self):
        ang = self.rng.uniform(0, 2 * math.pi)
        rad = self.rng.uniform(0.03, 0.085)
        return [st.CENTRE[0] + rad * math.cos(ang),
                st.CENTRE[1] + rad * math.sin(ang) * 0.7]

    def _spawn_departure(self, session):
        """A finished session drives away, so the map reflects the whole fleet
        rather than only the vehicles arriving."""
        xy = self.station_xy.get(session.station_id)
        if xy is None:
            return
        travel = float(self.rng.uniform(20, 50))
        self.vid += 1
        v = Vehicle(f"o{self.vid}",
                    _street_path(self.rng, list(xy), self._outer_point()),
                    self.now, self.now + pd.Timedelta(minutes=travel),
                    session.station_id, session.entity_id, None, kind="out")
        self.vehicles[v.id] = v

    def _spawn(self, depart, arrive, entity):
        station = self.by_entity[entity][self.rng.integers(len(self.by_entity[entity]))]
        origin = self._outer_point()
        dest = [station["lon"], station["lat"]]

        dur_h, energy = synth.sample_session(self.rng, entity)
        session = Session(entity, station["id"], arrive,
                          arrive + pd.Timedelta(hours=dur_h), energy)

        self.vid += 1
        v = Vehicle(f"v{self.vid}", _street_path(self.rng, origin, dest),
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
        for e in self.entities:
            actual[e] = 0.0
            started[e] = []
        for s in self.active:
            if s.end > h0 and s.connect < h1:
                actual[s.entity_id] += s.allocated(h0, h1)
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
        }
        return self.last_frame

    # ---------- fast channel ----------

    def positions(self):
        moving, routes = [], []
        for v in self.vehicles.values():
            lon, lat = v.position(self.now)
            moving.append({"id": v.id, "lon": round(lon, 6), "lat": round(lat, 6),
                           "e": v.entity_id, "k": v.kind})
            routes.append([[round(x, 6), round(y, 6)] for x, y in v.path])

        charging = []
        for s in self.active:
            xy = self.station_xy.get(s.station_id)
            if xy and s.connect <= self.now < s.end:
                charging.append({"lon": round(xy[0], 6), "lat": round(xy[1], 6),
                                 "e": s.entity_id})
        return {"type": "positions", "sim_time": self.now.isoformat(),
                "vehicles": moving, "routes": routes, "charging": charging,
                "n_charging": len(charging)}


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
