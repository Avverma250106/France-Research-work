"""Charging-station geometry for synthetic and OpenStreetMap-backed runs.

Synthetic mode preserves the original deterministic demo geometry. OSM mode
uses real charge-point coordinates but retains the simulation's existing
demand-profile granularity; it does not claim station-specific historical
demand.
"""

import numpy as np

from marl.config import MAX_STATIONS

from . import synth
from .station_provider import DATASET_CONFIG, load_station_catalog

# entity_id -> (label, centre lon/lat, n stations, rated kW, spread in degrees)
SITES = {
    "paris_bercy":        ("Bercy",         2.3800, 48.8400, 14, 22.0, 0.013),
    "paris_republique":   ("Republique",    2.3636, 48.8674, 12, 22.0, 0.012),
    "paris_montparnasse": ("Montparnasse",  2.3220, 48.8422, 10, 50.0, 0.011),
    "paris_la_defense":   ("La Defense",    2.2380, 48.8920, 16, 7.4,  0.014),
    "paris_batignolles":  ("Batignolles",   2.3190, 48.8880, 9, 3.7,  0.011),
    "paris_ivry":         ("Ivry",          2.3880, 48.8130, 10, 22.0, 0.013),
}

CENTRE = (2.3400, 48.8600)


def build(seed=11, *, mode="synthetic", dataset="paris", **provider_options):
    """Return (sites, stations), preserving synthetic mode as an explicit option."""
    if mode == "osm":
        return build_osm(dataset=dataset, **provider_options)[:2]
    if mode != "synthetic":
        raise ValueError("mode must be 'synthetic' or 'osm'.")
    rng = np.random.default_rng(seed)
    sites, stations = [], []
    sid = 0
    for entity, (label, lon, lat, n, kw, spread) in SITES.items():
        sites.append({
            "entity_id": entity, "label": label,
            "lon": lon, "lat": lat, "n_stations": n,
            "rated_kw": float(synth.rated_kw(entity)),
        })
        for k in range(n):
            stations.append({
                "id": f"st{sid:03d}",
                "entity_id": entity,
                "label": f"{label} {k + 1:02d}",
                "lon": float(lon + rng.normal(0, spread)),
                "lat": float(lat + rng.normal(0, spread * 0.7)),
                "rated_kw": float(synth.rated_kw(entity)),
                "n_points": int(rng.integers(2, 7)),
            })
            sid += 1
    return sites, stations


def build_osm(dataset="paris", **provider_options):
    """Return six OSM stations linked to their dataset-level historical profile."""
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unknown OSM dataset {dataset!r}.")
    provider_options.setdefault("max_stations", MAX_STATIONS)
    provider_options.setdefault("require_minimum", True)
    catalog = load_station_catalog(dataset, **provider_options)
    stations = list(catalog["stations"])
    if len(stations) != MAX_STATIONS:
        raise ValueError(
            f"The existing MAPPO checkpoint requires exactly {MAX_STATIONS} "
            f"stations; the {dataset} OSM catalog selected {len(stations)}."
        )
    profile_entity = DATASET_CONFIG[dataset].get("profile_entity_id")
    if not profile_entity:
        raise ValueError(
            f"No historical demand profile entity is configured for {dataset}."
        )
    for station in stations:
        station["entity_id"] = profile_entity
        station["profile_entity_id"] = profile_entity
        station["profile_source"] = "dataset-level historical profile"
    locations = [(station["lon"], station["lat"]) for station in stations]
    sites = [{
        "entity_id": profile_entity,
        "label": f"{DATASET_CONFIG[dataset]['label']} historical profile",
        "lon": sum(point[0] for point in locations) / len(locations),
        "lat": sum(point[1] for point in locations) / len(locations),
        "n_stations": len(stations),
        "profile_source": "historical dataset aggregate, not station-specific",
    }]
    return sites, stations, {
        key: value for key, value in catalog.items() if key != "stations"
    }
