"""Synthetic charging station geometry for Ile-de-France.

Coordinates are plausible, not real. The national IRVE register
(data.gouv.fr, schema etalab/schema-irve-statique) is the drop-in replacement
when real geometry is wanted: it carries lat/lon, rated power and operator for
every charge point in France.

Stations are grouped into SITES. The model predicts per site (entity_id),
because that is the granularity it was trained at - every non-ACN entity in the
corpus is a whole site, not an individual charge point. Stations exist to make
the map legible and to generate arrivals; predictions and error metrics are
reported per site.
"""

import numpy as np

from . import synth

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


def build(seed=11):
    """Return (sites, stations). Stations carry the site they belong to."""
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
