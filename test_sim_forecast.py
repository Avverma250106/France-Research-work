import numpy as np
import pandas as pd
import pytest

from baselines import evaluate_baselines, persistence_forecast
import pipeline as P
from sim import stations
from sim.engine import (
    Engine,
    KEEP_HOURS,
    METRIC_WINDOW,
    _align_forecast_pairs,
    _metrics,
    _validate_warmup_history,
)


class CountingModel:
    target_mode = "forecast"

    def __init__(self, entities, prior):
        self.meta = {"expanding_state": prior, "entities": entities}
        self.calls = 0
        self.entities = entities
        self.last_rows = None

    def predict(self, rows):
        self.calls += 1
        self.last_rows = rows.copy()
        return np.full(len(rows), self.calls, dtype=float), np.ones(len(rows), dtype=float)


def _history():
    sites, _ = stations.build()
    entities = [site["entity_id"] for site in sites]
    hours = pd.date_range("2024-12-15 00:00", periods=KEEP_HOURS, freq="h", tz="UTC")
    hourly = pd.DataFrame(
        [
            {
                "entity_id": entity,
                "hour": hour,
                "kwh_allocated": float(i % 11 + 1),
            }
            for entity in entities
            for i, hour in enumerate(hours)
        ]
    )
    sessions = pd.DataFrame({
        "entity_id": pd.Series(dtype=str),
        "connect_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
    })
    prior = {
        entity: [
            float(group["kwh_allocated"].sum()),
            int(len(group)),
        ]
        for entity, group in hourly.groupby("entity_id")
    }
    return entities, hourly, sessions, prior


def test_forecast_and_actual_samples_must_match_entity_and_target_time():
    target_time = pd.Timestamp("2025-01-01 01:00", tz="UTC")
    forecasts = [
        {
            "entity_id": "site-a",
            "prediction_time": target_time - pd.Timedelta(hours=1),
            "target_time": target_time,
            "model_prediction": 12.0,
            "persistence_prediction": 10.0,
        },
        {
            "entity_id": "site-b",
            "prediction_time": target_time - pd.Timedelta(hours=1),
            "target_time": target_time,
            "model_prediction": 22.0,
            "persistence_prediction": 20.0,
        },
    ]

    with pytest.raises(ValueError, match="sample mismatch"):
        _align_forecast_pairs(forecasts, {"site-a": 11.0}, target_time)

    with pytest.raises(ValueError, match="expected"):
        _align_forecast_pairs(forecasts, {"site-a": 11.0, "site-b": 21.0},
                              target_time + pd.Timedelta(hours=1))


def test_model_and_persistence_metrics_use_identical_site_hour_pairs():
    target_time = pd.Timestamp("2025-01-01 01:00", tz="UTC")
    forecasts = [
        {
            "entity_id": entity,
            "prediction_time": target_time - pd.Timedelta(hours=1),
            "target_time": target_time,
            "model_prediction": prediction,
            "persistence_prediction": prediction - 1,
        }
        for entity, prediction in (("site-a", 12.0), ("site-b", 22.0))
    ]
    records = _align_forecast_pairs(
        forecasts, {"site-a": 11.0, "site-b": 20.0}, target_time
    )
    metrics = _metrics(records)

    assert {(r["entity_id"], r["target_time"]) for r in records} == {
        (r["entity_id"], r["target_time"]) for r in forecasts
    }
    assert metrics["valid_pairs"] == len(records) == 2
    assert metrics["hours"] == 1
    assert metrics["window_hours"] == METRIC_WINDOW
    assert metrics["mae"] == 1.5
    assert metrics["persistence_mae"] == 0.5
    assert metrics["delta_mae_pct"] == 200.0
    assert all("model_absolute_error" in r and "persistence_absolute_error" in r
               for r in records)


def test_engine_preloads_history_and_makes_a_new_forecast_each_hour():
    entities, hourly, sessions, prior = _history()
    _validate_warmup_history(hourly, entities)
    model = CountingModel(entities, prior)
    engine = Engine(model, hourly, sessions)

    assert model.calls == 1
    first = engine._close_hour()
    second = engine._close_hour()

    assert model.calls == 3
    assert model.last_rows["target"].isna().all()
    assert first["overall"]["valid_pairs"] == len(entities)
    assert first["overall"]["hours"] == 1
    assert second["overall"]["valid_pairs"] == 2 * len(entities)
    assert second["overall"]["hours"] == 2
    assert all(
        record["prediction_time"] + pd.Timedelta(hours=1) == record["target_time"]
        for record in engine.history
    )
    target_times = sorted({record["target_time"] for record in engine.history})
    assert len(target_times) == 2
    assert target_times[0] == hourly["hour"].max() + pd.Timedelta(hours=1)
    assert {
        record["model_prediction"]
        for record in engine.history
        if record["target_time"] == target_times[1]
    } == {2.0}


def test_warmup_rejects_missing_simulator_entities():
    entities, hourly, _, _ = _history()
    incomplete = hourly[hourly["entity_id"] == entities[0]]

    with pytest.raises(ValueError, match="missing"):
        _validate_warmup_history(incomplete, entities)


def test_warmup_rejects_a_gap_in_required_history():
    entities, hourly, _, _ = _history()
    incomplete = hourly[hourly["entity_id"] == entities[0]].copy()
    row_index = incomplete.index[100]
    incomplete.loc[row_index, "hour"] += pd.Timedelta(minutes=30)

    with pytest.raises(ValueError, match="not contiguous"):
        _validate_warmup_history(incomplete, entities)


def test_engine_rejects_model_without_training_entities():
    entities, hourly, sessions, prior = _history()
    model = CountingModel(entities[:-1], prior)

    with pytest.raises(ValueError, match="not trained for simulated entities"):
        Engine(model, hourly, sessions)


def test_forecast_persistence_is_current_load_at_feature_time():
    frame = pd.DataFrame({
        "entity_id": ["site-a", "site-a"],
        "hour": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
        "kwh_allocated": [10.0, 15.0],
        "target": [15.0, 20.0],
    })

    np.testing.assert_allclose(persistence_forecast(frame, target_col="target"), [10.0, 15.0])


def test_forecast_targets_stay_attached_to_entities_after_row_sorting():
    hours = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    hourly = pd.DataFrame([
        {"source_dataset": "paris", "entity_id": "site-b", "site": "site-b", "region": "France", "hour": hours[0], "kwh_allocated": 10.0},
        {"source_dataset": "paris", "entity_id": "site-a", "site": "site-a", "region": "France", "hour": hours[0], "kwh_allocated": 1.0},
        {"source_dataset": "paris", "entity_id": "site-b", "site": "site-b", "region": "France", "hour": hours[1], "kwh_allocated": 20.0},
        {"source_dataset": "paris", "entity_id": "site-a", "site": "site-a", "region": "France", "hour": hours[1], "kwh_allocated": 2.0},
    ])
    sessions = pd.DataFrame(columns=["source_dataset", "entity_id", "connect_time_utc"])

    features = P.build_feature_table(hourly, sessions, target_mode="forecast")
    targets = features.set_index(["entity_id", "hour"])["target"]

    assert targets.loc[("site-a", hours[0])] == 2.0
    assert targets.loc[("site-b", hours[0])] == 20.0
    assert pd.isna(targets.loc[("site-a", hours[1])])
    assert pd.isna(targets.loc[("site-b", hours[1])])


def test_offline_baseline_keeps_predictions_aligned_after_sorting_shuffled_rows():
    frame = pd.DataFrame([
        {"entity_id": "site-a", "hour": "2025-01-01T01:00Z", "kwh_allocated": 20.0, "target": 25.0, "hour_of_day": 1, "day_of_week": 2, "feature": 0.0},
        {"entity_id": "site-b", "hour": "2025-01-01T00:00Z", "kwh_allocated": 100.0, "target": 110.0, "hour_of_day": 0, "day_of_week": 2, "feature": 0.0},
        {"entity_id": "site-a", "hour": "2025-01-01T00:00Z", "kwh_allocated": 10.0, "target": 20.0, "hour_of_day": 0, "day_of_week": 2, "feature": 0.0},
        {"entity_id": "site-b", "hour": "2025-01-01T01:00Z", "kwh_allocated": 110.0, "target": 111.0, "hour_of_day": 1, "day_of_week": 2, "feature": 0.0},
    ])
    frame["hour"] = pd.to_datetime(frame["hour"], utc=True)

    result = evaluate_baselines(
        frame, target_col="target", entity_col="entity_id", feature_columns=["feature"]
    )
    persistence = result["by_entity"]["persistence"]
    by_entity = {row["entity_id"]: row["mae"] for row in persistence}

    assert by_entity == {"site-a": 7.5, "site-b": 5.5}
