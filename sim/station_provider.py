"""OpenStreetMap charging-station catalog with an explicit local cache."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.error as urllib_error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_FALLBACK_URLS = (
    OVERPASS_URL,
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "France-Research-work-EV-simulation/1.0 (research)"
CACHE_DIRECTORY = Path("artifacts/station_cache")
ATTRIBUTION = "© OpenStreetMap contributors"
_last_geocode_request = 0.0
_last_overpass_request = 0.0
OVERPASS_RETRIES = 3
OVERPASS_BACKOFF_BASE_SECONDS = 2.0
OVERPASS_BACKOFF_MAX_SECONDS = 60.0


class _OverpassRateLimited(RuntimeError):
    pass

# These are named places derived from session metadata, not hand-entered
# coordinates. Nominatim resolves each place to current OSM bounds at fetch time.
DATASET_CONFIG = {
    "paris": {
        "label": "Paris, France",
        "region": "France",
        "timezone": "Europe/Paris",
        "place_query": "Paris, France",
        "country_code": "fr",
        "profile_file": "paris_sessions.csv",
        "profile_entity_id": "paris",
        "location_note": "Paris-area stations; demand remains site-profile level.",
    },
    "boulder": {
        "label": "Boulder, Colorado, USA",
        "region": "Colorado",
        "timezone": "America/Denver",
        "place_query": "Boulder, Colorado, United States",
        "country_code": "us",
        "profile_file": "boulder_2021_sessions.csv",
        "profile_entity_id": "boulder",
        "location_note": "Boulder-area stations; demand remains site-profile level.",
    },
    "palo_alto": {
        "label": "Palo Alto, California, USA",
        "region": "California",
        "timezone": "America/Los_Angeles",
        "place_query": "Palo Alto, California, United States",
        "country_code": "us",
        "profile_file": "palo_alto_sessions.csv",
        "profile_entity_id": "palo_alto",
        "location_note": "Palo Alto-area stations; demand remains site-profile level.",
    },
    "dundee": {
        "label": "Dundee, Scotland, UK",
        "region": "UK",
        "timezone": "Europe/London",
        "place_query": "Dundee, Scotland, United Kingdom",
        "country_code": "gb",
        "profile_file": "dundee_sessions.csv",
        "profile_entity_id": "dundee",
        "location_note": "Dundee-area stations; demand remains site-profile level.",
    },
    "perth": {
        "label": "Perth & Kinross, Scotland, UK",
        "region": "UK",
        "timezone": "Europe/London",
        "place_query": "Perth, Perth and Kinross, Scotland, United Kingdom",
        "country_code": "gb",
        "profile_file": "perth_sessions.csv",
        "profile_entity_id": "perth",
        "location_note": (
            "Perth & Kinross, SCOTLAND - not Perth, Western Australia. The "
            "upstream open-data corpus lists this dataset as Perth & Kinross "
            "(UK); region drives timezone and holiday calendar, so the "
            "distinction changes every temporal feature."
        ),
    },
    "acn": {
        "label": "ACN sites (California)",
        "region": "California",
        "timezone": None,
        "place_query": None,
        "country_code": None,
        "location_note": (
            "ACN contains multiple named campuses and an unresolved office entity; "
            "the session files do not provide verified coordinates."
        ),
    },
    "sap": {
        "label": "SAP workplace (France)",
        "region": "France",
        "timezone": None,
        "place_query": None,
        "country_code": None,
        "location_note": (
            "The SAP session metadata identifies France but does not identify "
            "the workplace address or coordinates."
        ),
    },
}

_POWER_VALUE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(kw|w)?\s*$", re.IGNORECASE)
_POWER_TAGS = (
    "charging_station:output",
    "socket:output",
    "socket:type2:output",
    "socket:ccs:output",
    "socket:chademo:output",
    "output",
)


def _validate_bbox(bbox):
    south, west, north, east = map(float, bbox)
    if not all(map(math.isfinite, (south, west, north, east))):
        raise ValueError("Station query bounds must be finite.")
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError(f"Invalid station query bounds: {bbox!r}.")
    return south, west, north, east


def parse_power_kw(tags):
    """Return the largest explicitly tagged output in kW, or None."""
    values = []
    for key in _POWER_TAGS:
        raw = tags.get(key)
        if not raw:
            continue
        for part in str(raw).split(";"):
            match = _POWER_VALUE.match(part)
            if not match:
                continue
            value = float(match.group(1).replace(",", "."))
            unit = (match.group(2) or "").lower()
            if unit == "w":
                value /= 1000.0
            if 0 < value <= 1000:
                values.append(value)
    return max(values) if values else None


def _in_bbox(lat, lon, bbox):
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


def normalize_overpass_elements(elements, bbox):
    """Normalize, validate, and deduplicate Overpass elements."""
    bounds = _validate_bbox(bbox)
    normalized = {}
    seen_coordinates = set()
    elements = sorted(
        elements,
        key=lambda element: (
            str(element.get("type", "")),
            str(element.get("id", "")),
        ),
    )
    for element in elements:
        osm_type = str(element.get("type", ""))
        osm_id = element.get("id")
        if osm_type not in {"node", "way", "relation"} or osm_id is None:
            continue
        center = element.get("center") or {}
        lat = element.get("lat", center.get("lat"))
        lon = element.get("lon", center.get("lon"))
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        if not _in_bbox(lat, lon, bounds):
            continue

        tags = element.get("tags") or {}
        if tags.get("amenity") != "charging_station":
            continue
        coordinate_key = (round(lat, 6), round(lon, 6))
        if coordinate_key in seen_coordinates:
            continue
        seen_coordinates.add(coordinate_key)

        power_keys = [
            key for key in _POWER_TAGS
            if tags.get(key) and parse_power_kw({key: tags[key]}) is not None
        ]
        power = parse_power_kw(tags)
        capacity = tags.get("capacity")
        parsed_capacity = 0
        try:
            parsed_capacity = int(capacity) if capacity is not None else 0
            n_points = parsed_capacity if parsed_capacity > 0 else 1
        except (TypeError, ValueError):
            n_points = 1
        station_id = f"osm_{osm_type}_{osm_id}"
        normalized[station_id] = {
            "id": station_id,
            "osm_type": osm_type,
            "osm_id": int(osm_id),
            "label": str(tags.get("name") or f"OSM charger {osm_id}"),
            "lon": lon,
            "lat": lat,
            "rated_kw": power,
            "capacity_source": power_keys[0] if power_keys else "unknown",
            "n_points": n_points,
            "n_points_source": (
                "capacity tag"
                if capacity is not None and parsed_capacity > 0
                else "default 1"
            ),
            "access": tags.get("access", "not tagged"),
            "operator": tags.get("operator"),
            "tags": tags,
        }
    return sorted(normalized.values(), key=lambda station: station["id"])


def _distance_squared(a, b):
    # Good enough for within-city ordering and avoids a geospatial dependency.
    mean_lat = math.radians((a["lat"] + b["lat"]) / 2)
    dx = (a["lon"] - b["lon"]) * math.cos(mean_lat)
    dy = a["lat"] - b["lat"]
    return dx * dx + dy * dy


def select_spatially_distributed(stations, limit, bbox):
    """Use deterministic farthest-point sampling rather than first-N selection."""
    if limit <= 0:
        raise ValueError("Station selection limit must be positive.")
    if len(stations) <= limit:
        return list(stations)
    south, west, north, east = _validate_bbox(bbox)
    center = {"lat": (south + north) / 2, "lon": (west + east) / 2}
    candidates = sorted(stations, key=lambda station: station["id"])
    first = min(
        candidates,
        key=lambda station: (_distance_squared(station, center), station["id"]),
    )
    selected = [first]
    remaining = [station for station in candidates if station["id"] != first["id"]]
    while len(selected) < limit:
        next_station = max(
            remaining,
            key=lambda station: (
                min(_distance_squared(station, chosen) for chosen in selected),
                station["id"],
            ),
        )
        selected.append(next_station)
        remaining.remove(next_station)
    return selected


def _fetch_overpass(bbox, timeout=60):
    south, west, north, east = _validate_bbox(bbox)
    lat_edges = list(_tile_edges(south, north, 0.04))
    lon_edges = list(_tile_edges(west, east, 0.05))
    tiles = [
        (lat_edges[i], lon_edges[j], lat_edges[i + 1], lon_edges[j + 1])
        for i in range(len(lat_edges) - 1)
        for j in range(len(lon_edges) - 1)
    ]
    elements = []
    for tile in tiles:
        elements.extend(_fetch_overpass_tile(tile, timeout=timeout))
    return elements


def _fetch_overpass_tile(tile, *, timeout, depth=0):
    global _last_overpass_request
    south, west, north, east = tile
    query = (
        "[out:json][timeout:50];"
        f'nwr["amenity"="charging_station"]'
        f"({south},{west},{north},{east});out center tags;"
    )
    failures = []
    for endpoint in OVERPASS_FALLBACK_URLS:
        for attempt in range(OVERPASS_RETRIES):
            delay = 1.05 - (time.monotonic() - _last_overpass_request)
            if delay > 0:
                time.sleep(delay)
            request = urllib.request.Request(
                endpoint,
                data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
                headers={
                    "User-Agent": USER_AGENT,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            try:
                _last_overpass_request = time.monotonic()
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.load(response)
                elements = payload.get("elements")
                if not isinstance(elements, list):
                    raise ValueError(
                        "Overpass returned a response without an elements list."
                    )
                return elements
            except urllib_error.HTTPError as exc:
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        server_delay = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        server_delay = 0.0
                    backoff = min(
                        max(
                            server_delay,
                            OVERPASS_BACKOFF_BASE_SECONDS * (2 ** attempt),
                        ),
                        OVERPASS_BACKOFF_MAX_SECONDS,
                    )
                    failures.append(f"{endpoint}: {exc}; retrying in {backoff:g}s")
                    if attempt + 1 < OVERPASS_RETRIES:
                        time.sleep(backoff)
                        continue
                    raise _OverpassRateLimited(
                        f"Overpass rate limit persisted at {endpoint} after "
                        f"{OVERPASS_RETRIES} attempts; retry later. "
                        "No partial station cache was written."
                    ) from exc
                failures.append(f"{endpoint}: {exc}")
                break
            except Exception as exc:
                failures.append(f"{endpoint}: {exc}")
                break

    if depth < 2:
        mid_lat = (south + north) / 2
        mid_lon = (west + east) / 2
        smaller_tiles = (
            (south, west, mid_lat, mid_lon),
            (south, mid_lon, mid_lat, east),
            (mid_lat, west, north, mid_lon),
            (mid_lat, mid_lon, north, east),
        )
        elements = []
        for smaller_tile in smaller_tiles:
            try:
                elements.extend(
                    _fetch_overpass_tile(
                        smaller_tile, timeout=timeout, depth=depth + 1
                    )
                )
            except _OverpassRateLimited:
                raise
        return elements
    raise RuntimeError(
        f"Overpass failed for tile {tile} after subdivision: " + "; ".join(failures)
    )


def _tile_edges(start, end, maximum_width):
    count = max(1, math.ceil((end - start) / maximum_width))
    width = (end - start) / count
    return [start + index * width for index in range(count)] + [end]


def _resolve_location(config, timeout=30):
    """Resolve a metadata-backed place name to current OSM bounding coordinates."""
    global _last_geocode_request
    delay = 1.0 - (time.monotonic() - _last_geocode_request)
    if delay > 0:
        time.sleep(delay)
    params = urllib.parse.urlencode({
        "q": config["place_query"],
        "format": "jsonv2",
        "addressdetails": 1,
        "featuretype": "city",
        "limit": 5,
    })
    request = urllib.request.Request(
        f"{NOMINATIM_URL}?{params}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            results = json.load(response)
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve {config['place_query']!r} with OpenStreetMap "
            f"Nominatim: {exc}"
        ) from exc
    finally:
        _last_geocode_request = time.monotonic()

    if not isinstance(results, list):
        raise ValueError("Nominatim returned an invalid place-search response.")
    expected_country = config["country_code"]
    matches = [
        result for result in results
        if (result.get("address") or {}).get("country_code") == expected_country
    ]
    if not matches:
        raise ValueError(
            f"Nominatim could not resolve {config['place_query']!r} "
            f"within country code {expected_country!r}."
        )
    place = matches[0]
    try:
        # Nominatim order: south, north, west, east.
        raw_bounds = place["boundingbox"]
        bbox = _validate_bbox((
            raw_bounds[0], raw_bounds[2], raw_bounds[1], raw_bounds[3]
        ))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Nominatim place result for {config['place_query']!r} has no "
            "valid bounding box."
        ) from exc
    return {
        "query": config["place_query"],
        "display_name": str(place.get("display_name") or config["label"]),
        "country_code": expected_country,
        "osm_type": place.get("osm_type"),
        "osm_id": place.get("osm_id"),
        "bbox": list(bbox),
    }


def load_station_catalog(
    dataset="paris",
    *,
    cache_path=None,
    refresh=False,
    max_stations=120,
    require_minimum=False,
    fetcher=None,
    geocoder=None,
):
    """Load an OSM station catalog from cache, fetching only when required."""
    key = str(dataset).strip().lower()
    if key not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown station dataset {dataset!r}; choose from {sorted(DATASET_CONFIG)}."
        )
    if (
        not isinstance(max_stations, int)
        or isinstance(max_stations, bool)
        or max_stations <= 0
    ):
        raise ValueError("max_stations must be a positive integer.")
    config = DATASET_CONFIG[key]
    if config["place_query"] is None:
        raise ValueError(
            f"OSM station geography is not configured for {key}: "
            f"{config['location_note']} Add a verified location query before enabling it."
        )
    if cache_path:
        path = Path(cache_path)
        if path.is_dir() or path.suffix.lower() != ".json":
            path = path / f"{key}.json"
    else:
        path = CACHE_DIRECTORY / f"{key}.json"
    cache_used = path.is_file() and not refresh
    if cache_used:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("dataset") != key:
                raise ValueError("Cached station catalog belongs to another dataset.")
            if payload.get("location_query") != config["place_query"]:
                raise ValueError("Cached station catalog has a different location query.")
            bbox = _validate_bbox(payload.get("bbox", ()))
            location = payload.get("location")
            if not isinstance(location, dict):
                raise ValueError("Cached station catalog has no resolved OSM location.")
            stations = payload.get("stations")
            if not isinstance(stations, list) or not stations:
                raise ValueError("Cached station catalog has no station records.")
            validated = normalize_overpass_elements(
                [
                    {
                        "type": station["osm_type"],
                        "id": station["osm_id"],
                        "lat": station["lat"],
                        "lon": station["lon"],
                        "tags": station.get("tags", {"amenity": "charging_station"}),
                    }
                    for station in stations
                ],
                bbox,
            )
            if not validated:
                raise ValueError("Cached station catalog contains no valid stations.")
            # Preserve normalized cache fields (including capacity provenance).
            by_id = {station["id"]: station for station in stations}
            stations = [
                {**station, **by_id.get(station["id"], {})}
                for station in validated
            ]
            fetched_at = payload.get("fetched_at_utc")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Could not read cached station catalog {path}: {exc}") from exc
    else:
        try:
            location = (geocoder or _resolve_location)(config)
            bbox = _validate_bbox(location["bbox"])
            elements = (fetcher or _fetch_overpass)(bbox)
        except Exception as exc:
            if path.exists():
                raise RuntimeError(
                    f"Could not refresh OSM station data for {key}; existing cache "
                    f"{path} was not used because refresh was requested: {exc}"
                ) from exc
            raise RuntimeError(
                f"Could not resolve/fetch OSM station data for {key}, and no cache exists at "
                f"{path}. Check network access or provide a valid local cache. "
                f"Underlying error: {exc}"
            ) from exc
        stations = normalize_overpass_elements(elements, bbox)
        if not stations:
            raise ValueError(
                f"Overpass returned no valid charging stations in the configured "
                f"{key} bounds {bbox}."
            )
        if require_minimum and len(stations) < max_stations:
            raise ValueError(
                f"Overpass returned {len(stations)} valid stations for {key}; "
                f"at least {max_stations} are required. No cache was written."
            )
        fetched_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": 1,
            "dataset": key,
            "location_query": config["place_query"],
            "location": location,
            "bbox": list(bbox),
            "fetched_at_utc": fetched_at,
            "attribution": ATTRIBUTION,
            "source_url": OVERPASS_URL,
            "stations": stations,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise OSError(f"Could not cache OSM station catalog at {path}: {exc}") from exc

    if require_minimum and len(stations) < max_stations:
        raise ValueError(
            f"Station catalog for {key} contains {len(stations)} records; "
            f"at least {max_stations} are required."
        )
    selected = select_spatially_distributed(stations, max_stations, bbox)
    return {
        "dataset": key,
        "dataset_label": config["label"],
        "location": location,
        "region": config["region"],
        "timezone": config["timezone"],
        "bbox": list(bbox),
        "location_source": NOMINATIM_URL,
        "provider": "OpenStreetMap Overpass",
        "attribution": ATTRIBUTION,
        "source_url": OVERPASS_URL,
        "cache_path": str(path),
        "cache_used": cache_used,
        "fetched_at_utc": fetched_at,
        "available_station_count": len(stations),
        "selected_station_count": len(selected),
        "station_source": "openstreetmap",
        "demand_source": "historical_profile",
        "location_note": config["location_note"],
        "stations": selected,
    }
