"""Synthetic EV charging sessions in the harmonized_sessions schema.

Used only when the real corpus is absent (see README.md for the Kaggle link).
Produces the same columns the real pipeline emits, so train_for_sim.py runs
identically on either source and the swap is a one-line change.

Nothing here is calibrated against the real data. Metrics from a model trained
on this are a demonstration that the plumbing works, not a research result.
"""

import numpy as np
import pandas as pd

SCHEMA = [
    "source_dataset", "entity_id", "site", "region",
    "connect_time_utc", "charge_end_time_utc", "energy_kwh",
]

# kind -> (sessions/day weekday, weekend multiplier, duration median h, sigma, kW)
KINDS = {
    "workplace":   (14.0, 0.18, 4.2, 0.45, 7.4),
    "public":      (22.0, 0.85, 1.6, 0.55, 22.0),
    "residential": (9.0, 1.10, 7.5, 0.35, 3.7),
    "rapid":       (30.0, 0.95, 0.55, 0.45, 50.0),
}

# Diurnal arrival shape per kind, 24 weights (local time).
SHAPES = {
    "workplace": [.2,.1,.1,.1,.1,.3,1.2,3.6,6.5,5.2,3.0,2.0,
                  1.8,1.6,1.5,1.6,1.9,1.7,1.0,.7,.5,.4,.3,.2],
    "public": [.4,.3,.2,.2,.2,.4,.9,1.8,2.6,3.0,3.2,3.4,
               3.6,3.4,3.2,3.2,3.4,3.8,3.9,3.2,2.4,1.8,1.2,.7],
    "residential": [.6,.4,.3,.2,.2,.3,.7,1.1,1.0,.8,.7,.7,
                    .8,.8,.9,1.1,1.9,3.6,4.6,4.2,3.4,2.6,1.8,1.1],
    "rapid": [.5,.3,.2,.2,.3,.6,1.4,2.6,3.0,2.8,2.8,3.0,
              3.2,3.0,2.9,3.0,3.2,3.4,3.0,2.2,1.6,1.2,.9,.6],
}

# The entities the model is trained on. The six paris_* sites are what the map
# renders; the rest exist so the one-hot column space matches the real corpus.
ENTITIES = [
    # (source_dataset, entity_id, region, kind, scale)
    ("paris", "paris_bercy",        "France",     "public",      3.50),
    ("paris", "paris_republique",   "France",     "public",      2.90),
    ("paris", "paris_montparnasse", "France",     "rapid",       2.50),
    ("paris", "paris_la_defense",   "France",     "workplace",   4.40),
    ("paris", "paris_batignolles",  "France",     "residential", 2.20),
    ("paris", "paris_ivry",         "France",     "public",      2.00),
    ("sap",   "sap",                "France",     "workplace",   1.10),
    ("acn",   "caltech",            "California", "workplace",   1.20),
    ("acn",   "jpl",                "California", "workplace",   0.95),
    ("acn",   "office001",          "California", "workplace",   0.35),
    ("palo_alto", "palo_alto",      "California", "public",      1.15),
    ("boulder",   "boulder",        "Colorado",   "public",      0.55),
    ("dundee",    "dundee",         "UK",         "public",      0.75),
    ("perth",     "perth",          "Australia",  "public",      0.85),
]


PARIS_SITES = [e[1] for e in ENTITIES if e[0] == "paris"]

ENTITY_KIND = {e[1]: e[3] for e in ENTITIES}
ENTITY_SCALE = {e[1]: e[4] for e in ENTITIES}
ENTITY_REGION = {e[1]: e[2] for e in ENTITIES}

_NORM_SHAPE = {k: np.asarray(v, dtype=float) / np.sum(v) for k, v in SHAPES.items()}


def rated_kw(entity):
    return KINDS[ENTITY_KIND[entity]][4]


def arrival_rate(entity, local_hour, is_weekend, dayofyear=None):
    """Expected sessions connecting in one hour. The live simulation uses this
    so its arrival regime matches the corpus the model was trained on - without
    that, the ring buffer's lag features describe a different world than the
    hours being predicted, and the error metrics measure the discontinuity."""
    kind = ENTITY_KIND[entity]
    per_day, weekend_mult, _, _, _ = KINDS[kind]
    lam = per_day * ENTITY_SCALE[entity] * _NORM_SHAPE[kind][local_hour]
    if is_weekend:
        lam *= weekend_mult
    if dayofyear is not None:
        lam *= 1.0 + 0.18 * np.sin(2 * np.pi * dayofyear / 365.25)
    return float(lam)


def sample_session(rng, entity):
    """(duration_hours, energy_kwh) drawn the same way generate() draws them."""
    kind = ENTITY_KIND[entity]
    _, _, dur_med, dur_sigma, kw = KINDS[kind]
    dur_h = float(np.clip(rng.lognormal(np.log(dur_med), dur_sigma), 0.15, 24.0))
    active_h = min(dur_h, rng.uniform(0.6, 1.0) * np.sqrt(dur_h) * 1.6)
    energy = float(max(0.05, active_h * kw * rng.uniform(0.55, 0.95)))
    return dur_h, energy


def generate(days=365, end="2024-12-31", seed=7):
    """Generate sessions for every entity. Returns one DataFrame."""
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end, tz="UTC").floor("h")
    start_ts = end_ts - pd.Timedelta(days=days)
    hours = pd.date_range(start_ts, end_ts, freq="h")

    rows = []
    for source, entity, region, kind, scale in ENTITIES:
        per_day, weekend_mult, dur_med, dur_sigma, kw = KINDS[kind]
        shape = _NORM_SHAPE[kind]

        local = hours.tz_convert({
            "France": "Europe/Paris", "California": "America/Los_Angeles",
            "Colorado": "America/Denver", "UK": "Europe/London",
            "Australia": "Australia/Perth",
        }[region])

        lam = per_day * scale * shape[local.hour]
        lam = lam * np.where(local.weekday >= 5, weekend_mult, 1.0)
        # mild seasonal drift so rolling features have something to track
        lam = lam * (1.0 + 0.18 * np.sin(2 * np.pi * local.dayofyear / 365.25))

        counts = rng.poisson(lam)
        total = int(counts.sum())
        if total == 0:
            continue

        starts = np.repeat(hours.to_numpy(), counts)
        offset = rng.uniform(0, 60, total)
        connect = pd.to_datetime(starts, utc=True) + pd.to_timedelta(offset, unit="m")

        dur_h = np.clip(rng.lognormal(np.log(dur_med), dur_sigma, total), 0.15, 24.0)
        end_time = connect + pd.to_timedelta(dur_h * 60, unit="m")

        # Energy: charging for part of the connection, tapering with duration.
        # Long connections are mostly idle-plugged, which is what the real
        # workplace datasets look like.
        active_h = np.minimum(dur_h, rng.uniform(0.6, 1.0, total) * np.sqrt(dur_h) * 1.6)
        energy = active_h * kw * rng.uniform(0.55, 0.95, total)
        energy = np.round(np.clip(energy, 0.05, None), 4)

        rows.append(pd.DataFrame({
            "source_dataset": source,
            "entity_id": entity,
            "site": entity,
            "region": region,
            "connect_time_utc": connect,
            "charge_end_time_utc": end_time,
            "energy_kwh": energy,
        }))

    df = pd.concat(rows, ignore_index=True)
    return df.sort_values(["source_dataset", "entity_id", "connect_time_utc"]).reset_index(drop=True)


if __name__ == "__main__":
    d = generate(days=120)
    print(d.shape)
    print(d.groupby("entity_id").size())
