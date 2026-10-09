import asyncio
import json
import urllib.error

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from sim import stations
from sim import engine as sim_engine
from sim import server as sim_server
from sim import station_provider
from sim.station_provider import (
    load_station_catalog,
    normalize_overpass_elements,
    parse_power_kw,
    select_spatially_distributed,
)

BBOX = (48.80, 2.20, 48.92, 2.46)


def _geocoder(_config):
    return {
        "query": "Paris, France",
        "display_name": "Paris, France",
        "country_code": "fr",
        "osm_type": "relation",
        "osm_id": 1,
        "bbox": list(BBOX),
    }


def _elements():
    return [
        {
            "type": "node",
            "id": 1,
            "lat": 48.81,
            "lon": 2.21,
            "tags": {
                "amenity": "charging_station",
                "name": "Slow charger",
                "socket:type2:output": "22 kW",
                "capacity": "2",
            },
        },
        {
            "type": "node",
            "id": 2,
            "lat": 48.91,
            "lon": 2.45,
            "tags": {
                "amenity": "charging_station",
                "name": "Fast charger",
                "socket:ccs:output": "50 kW",
            },
        },
        {
            "type": "way",
            "id": 3,
            "center": {"lat": 48.86, "lon": 2.33},
            "tags": {"amenity": "charging_station"},
        },
        {
            "type": "node",
            "id": 4,
            "lat": 48.86,
            "lon": 2.33,
            "tags": {"amenity": "charging_station"},
        },
        {
            "type": "node",
            "id": 5,
            "lat": 49.2,
            "lon": 2.33,
            "tags": {"amenity": "charging_station"},
        },
        {
            "type": "node",
            "id": 6,
            "lat": 48.87,
            "lon": 2.35,
            "tags": {"amenity": "parking"},
        },
    ]


def test_parse_power_only_accepts_explicit_wattage_units():
    assert parse_power_kw({"socket:output": "22000 W"}) == 22.0
    assert parse_power_kw({"output": "11 kW; 7.4 kW"}) == 11.0
    assert parse_power_kw({"output": "230 V"}) is None


def test_normalization_validates_bounds_and_deduplicates_same_coordinates():
    records = normalize_overpass_elements(_elements(), BBOX)

    assert [station["id"] for station in records] == [
        "osm_node_1",
        "osm_node_2",
        "osm_node_4",
    ]
    assert records[0]["rated_kw"] == 22.0
    assert records[0]["capacity_source"] == "socket:type2:output"
    assert records[0]["n_points"] == 2
    assert records[2]["rated_kw"] is None


def test_station_selection_is_spatially_distributed_and_deterministic():
    normalized = normalize_overpass_elements(
        _elements(), BBOX
    )
    first = select_spatially_distributed(
        normalized, 2, BBOX
    )
    second = select_spatially_distributed(
        list(reversed(normalized)), 2, BBOX
    )

    assert [station["id"] for station in first] == [
        station["id"] for station in second
    ]
    selected = {station["id"] for station in first}
    assert "osm_node_4" in selected
    assert len(selected & {"osm_node_1", "osm_node_2"}) == 1


def test_cache_is_used_without_network_and_preserves_capacity_tags(tmp_path):
    cache = tmp_path / "paris.json"
    calls = []

    def fetcher(_bbox):
        calls.append(True)
        return _elements()

    initial = load_station_catalog(
        "paris",
        cache_path=cache,
        max_stations=10,
        fetcher=fetcher,
        geocoder=_geocoder,
    )
    cached = load_station_catalog(
        "paris",
        cache_path=cache,
        max_stations=10,
        fetcher=lambda _bbox: pytest.fail("cache should avoid network"),
    )

    assert calls == [True]
    assert initial["cache_used"] is False
    assert cached["cache_used"] is True
    assert cached["fetched_at_utc"] == initial["fetched_at_utc"]
    assert cached["stations"][0]["rated_kw"] == 22.0
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["attribution"] == "© OpenStreetMap contributors"


def test_missing_cache_and_network_returns_actionable_error(tmp_path):
    with pytest.raises(RuntimeError, match="no cache exists"):
        load_station_catalog(
            "paris",
            cache_path=tmp_path / "missing.json",
            fetcher=lambda _bbox: (_ for _ in ()).throw(OSError("offline")),
            geocoder=_geocoder,
        )


def test_unverified_acn_and_sap_locations_are_not_guessed():
    for dataset in ("acn", "sap"):
        with pytest.raises(ValueError, match="not configured"):
            load_station_catalog(dataset, cache_path="unused.json")


def test_overpass_rate_limit_retries_before_failing(monkeypatch):
    calls = []

    def rate_limited(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests", {}, None
        )

    monkeypatch.setattr(station_provider.urllib.request, "urlopen", rate_limited)
    monkeypatch.setattr(station_provider, "_last_overpass_request", 0.0)
    monkeypatch.setattr(station_provider.time, "sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="rate limit"):
        station_provider._fetch_overpass_tile(
            (48.80, 2.20, 48.81, 2.21), timeout=1
        )

    assert len(calls) == station_provider.OVERPASS_RETRIES


def test_insufficient_catalog_is_not_cached(tmp_path):
    cache = tmp_path / "paris.json"
    with pytest.raises(ValueError, match="No cache was written"):
        load_station_catalog(
            "paris",
            cache_path=cache,
            max_stations=6,
            require_minimum=True,
            fetcher=lambda _bbox: _elements(),
            geocoder=_geocoder,
        )
    assert not cache.exists()


def test_dataset_selector_rejects_acn_and_sap_without_verified_locations():
    for dataset in ("acn", "sap"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                sim_server.select_station_dataset({"dataset": dataset})
            )
        assert getattr(exc.value, "status_code", None) == 422
        assert "coordinates" in str(exc.value.detail)


def test_dataset_metadata_maps_to_the_matching_city_and_profile():
    expected = {
        "paris": ("Paris, France", "paris"),
        "boulder": ("Boulder, Colorado, United States", "boulder"),
        "palo_alto": ("Palo Alto, California, United States", "palo_alto"),
        "dundee": ("Dundee, Scotland, United Kingdom", "dundee"),
        "perth": ("Perth, Perth and Kinross, Scotland, United Kingdom", "perth"),
    }
    for dataset, (place_query, entity_id) in expected.items():
        config = station_provider.DATASET_CONFIG[dataset]
        assert config["place_query"] == place_query
        assert config["profile_entity_id"] == entity_id
        assert config["profile_file"]
    assert station_provider.DATASET_CONFIG["acn"]["place_query"] is None
    assert station_provider.DATASET_CONFIG["sap"]["place_query"] is None


def test_dataset_options_mark_real_and_synthetic_availability():
    real = asyncio.run(sim_server.station_datasets("osm"))
    synthetic = asyncio.run(sim_server.station_datasets("synthetic"))
    real_options = {item["id"]: item["available"] for item in json.loads(real.body)["datasets"]}
    synthetic_options = {
        item["id"]: item["available"] for item in json.loads(synthetic.body)["datasets"]
    }

    assert all(real_options[name] for name in ("paris", "boulder", "palo_alto", "dundee", "perth"))
    assert not real_options["acn"]
    assert not real_options["sap"]
    assert synthetic_options == {
        name: name == "paris" for name in station_provider.DATASET_CONFIG
    }


def test_dataset_selector_replaces_engine_only_after_new_engine_is_ready(monkeypatch):
    model = type("Model", (), {"data_source": "real"})()
    existing = type("Engine", (), {"dataset": "paris"})()
    replacement = type(
        "Engine",
        (),
        {
            "dataset": "boulder",
            "sites": [],
            "stations": [],
            "geography": {"dataset": "boulder"},
            "routed": False,
            "mappo": type("Controller", (), {"station_ids": ()})(),
        },
    )()
    monkeypatch.setitem(sim_server.state, "model", model)
    monkeypatch.setitem(sim_server.state, "engine", existing)
    monkeypatch.setitem(sim_server.state, "clients", set())
    monkeypatch.setattr(
        sim_server, "_engine_for", lambda *_args: replacement
    )

    response = asyncio.run(
        sim_server.select_station_dataset({"dataset": "boulder"})
    )

    assert sim_server.state["engine"] is replacement
    assert json.loads(response.body)["dataset"] == "boulder"


def test_real_geometry_uses_dataset_level_profile(monkeypatch):
    catalog = {
        "dataset": "paris",
        "dataset_label": "Paris, France",
        "provider": "OpenStreetMap Overpass",
        "timezone": "Europe/Paris",
        "stations": [
            {
                "id": f"osm_node_{index}",
                "label": f"Charger {index}",
                "lon": 2.25 + index * 0.025,
                "lat": 48.82 + index * 0.018,
                "rated_kw": None,
                "n_points": 1,
                "capacity_source": "unknown",
                "n_points_source": "default 1",
                "access": "not tagged",
            }
            for index in range(6)
        ],
    }
    monkeypatch.setattr(stations, "load_station_catalog", lambda *_args, **_kw: catalog)

    sites, points, metadata = stations.build_osm("paris")

    assert len(sites) == 1
    assert len(points) == 6
    assert {site["entity_id"] for site in sites} == {"paris"}
    assert {point["profile_entity_id"] for point in points} == {"paris"}
    assert all(point["id"].startswith("osm_node_") for point in points)
    assert metadata["provider"] == "OpenStreetMap Overpass"


@pytest.mark.parametrize(
    ("dataset", "profile_entity"),
    [
        ("paris", "paris"),
        ("boulder", "boulder"),
        ("palo_alto", "palo_alto"),
        ("dundee", "dundee"),
        ("perth", "perth"),
    ],
)
def test_osm_catalog_stations_use_their_dataset_profile(
    monkeypatch, dataset, profile_entity
):
    catalog = {
        "dataset": dataset,
        "stations": [
            {
                "id": f"osm_{dataset}_{index}",
                "label": f"Charger {index}",
                "lon": float(index),
                "lat": float(index),
                "rated_kw": 7.4,
                "n_points": 1,
            }
            for index in range(6)
        ],
    }
    monkeypatch.setattr(stations, "load_station_catalog", lambda *_args, **_kw: catalog)

    sites, selected, _ = stations.build_osm(dataset)

    assert len(selected) == 6
    assert {station["id"] for station in selected} == {
        f"osm_{dataset}_{index}" for index in range(6)
    }
    assert {station["entity_id"] for station in selected} == {profile_entity}
    assert sites[0]["entity_id"] == profile_entity


def test_engine_uses_osm_points_without_changing_forecast_feature_boundary(monkeypatch):
    catalog = {
        "dataset": "paris",
        "dataset_label": "Paris, France",
        "provider": "OpenStreetMap Overpass",
        "region": "France",
        "timezone": "Europe/Paris",
        "stations": [
            {
                "id": f"osm_node_{index}",
                "label": f"Charger {index}",
                "lon": 2.30 + index * 0.01,
                "lat": 48.84 + index * 0.01,
                "rated_kw": 22.0,
                "n_points": 2,
                "capacity_source": "socket:type2:output",
                "n_points_source": "capacity tag",
                "access": "yes",
                "operator": "Example",
            }
            for index in range(6)
        ],
    }
    monkeypatch.setattr(
        stations, "load_station_catalog", lambda *_args, **_kwargs: catalog
    )
    monkeypatch.setattr(sim_engine.RoadNetwork, "load", lambda: None)

    class RecordingModel:
        meta = {"expanding_state": {}}

        def predict(self, rows):
            self.rows = rows.copy()
            return [0.0] * len(rows), [0.0] * len(rows)

    model = RecordingModel()
    engine = sim_engine.Engine(model, station_mode="osm", dataset="paris")
    engine._close_hour()

    assert {station["id"] for station in engine.stations} == {
        f"osm_node_{index}" for index in range(6)
    }
    assert set(engine.mappo.station_ids) == {station["id"] for station in engine.stations}
    assert engine.initial_mappo_frame["forecast_hour"] == engine.now.isoformat()
    assert engine.buffer.source_dataset == "paris"
    assert engine.buffer.region == "France"
    assert "target" not in model.rows.columns
    assert "sessions_started" not in model.rows.columns
    assert "sessions_started_lag24h" in model.rows.columns


def test_real_mode_does_not_fall_back_to_synthetic_stations(monkeypatch):
    monkeypatch.setattr(
        stations,
        "build_osm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("OSM catalog unavailable")
        ),
    )
    monkeypatch.setattr(
        stations,
        "build",
        lambda *_args, **_kwargs: pytest.fail(
            "real mode must not fall back to synthetic geometry"
        ),
    )
    model = type("Model", (), {"meta": {}})()

    with pytest.raises(RuntimeError, match="OSM catalog unavailable"):
        sim_engine.Engine(model, station_mode="osm", dataset="paris")


def test_historical_profiles_follow_dataset_metadata():
    from sim.demand_profiles import HistoricalDemandProfile

    expected = {
        "paris": "paris",
        "boulder": "boulder",
        "palo_alto": "palo_alto",
        "dundee": "dundee",
        "perth": "perth",
    }
    for dataset, entity_id in expected.items():
        profile = HistoricalDemandProfile.load(dataset)
        assert profile.entity_id == entity_id
        assert profile.sessions["source_dataset"].eq(dataset).all()
        assert profile.arrival_rate(12, False) >= 0
        duration, energy = profile.sample_session(
            np.random.default_rng(42), 12, False
        )
        assert duration > 0
        assert energy > 0


def test_next_hour_forecast_features_use_only_closed_hours():
    import pipeline as pipeline
    from sim.demand_profiles import HistoricalDemandProfile
    from sim.features_rt import FeatureBuffer

    profile = HistoricalDemandProfile.load("paris")
    hourly, sessions, prior = profile.warmup()
    buffer = FeatureBuffer(
        [profile.entity_id],
        source_dataset="paris",
        region=profile.region,
    )
    buffer.warm(hourly, sessions, prior=prior)
    origin = buffer.last_hour()
    rows = buffer.feature_rows_for_next_hour()
    row = rows.iloc[0]
    target_hour = origin + pd.Timedelta(hours=1)
    lag_hour = target_hour - pd.Timedelta(hours=24)
    expected_lag = int(
        profile.sessions["connect_time_utc"].dt.floor("h").eq(lag_hour).sum()
    )

    assert pd.Timestamp(row["hour"]) == target_hour
    assert row[pipeline.SESSION_COUNT_FEATURE] == expected_lag
    assert "sessions_started" not in rows.columns
    pipeline.validate_forecast_feature_leakage(
        pipeline.design_matrix(rows, columns=None)
    )


def test_saved_mappo_checkpoint_runs_with_catalog_station_ids():
    from sim.mappo_runtime import MAPPOController
    from marl.environment import OBSERVATION_FIELDS
    from marl.mappo import observations_to_array

    stations = [
        {
            "id": f"osm_node_{index}",
            "rated_kw": 22.0,
            "n_points": 1,
        }
        for index in range(6)
    ]
    controller = MAPPOController(stations)
    station_ids = controller.station_ids
    demand = {station_id: 1.0 for station_id in station_ids}
    forecast = {station_id: 1.2 for station_id in station_ids}

    first = controller.advance(demand, forecast)
    observations = controller.environment.observe(forecast)
    observation_array = observations_to_array(
        observations, station_ids, OBSERVATION_FIELDS
    )
    second = controller.advance(demand, forecast)

    assert observation_array.shape == (6, 6)
    assert first["station_ids"] == list(station_ids)
    assert first["agent_ids"] == list(station_ids)
    assert set(first["next_hour_plan"]) == set(station_ids)
    assert set(second["completed_step"]["allocations"]) == set(station_ids)
    assert set(second["renewable_allocation"]) == set(station_ids)
    assert set(second["grid_allocation"]) == set(station_ids)
    assert all(
        set(action) == {"renewable", "grid"}
        for action in first["next_hour_plan"].values()
    )
