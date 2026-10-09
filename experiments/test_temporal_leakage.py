"""Automated assertions for past-only demand and session-arrival features."""

import numpy as np
import pandas as pd
import pytest

import pipeline as P
from experiments.common import (
    add_same_hour_demand_features,
    add_session_history_features,
    temporal_splits,
)


def _fixture():
    hours = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "source_dataset": "test",
        "entity_id": "site-a",
        "hour": hours,
        "hour_of_day": hours.hour,
        "kwh_allocated": np.arange(len(hours), dtype=float) % 17,
    })
    sessions = pd.DataFrame({
        "source_dataset": "test",
        "entity_id": "site-a",
        "connect_time_utc": [
            hour + pd.Timedelta(minutes=15)
            for index, hour in enumerate(hours)
            for _ in range(index % 4)
        ],
    })
    return frame, sessions


def test_session_arrival_lags_are_exactly_24_and_168_hours():
    frame, sessions = _fixture()
    features = add_session_history_features(frame, sessions)
    sessions_at_hour = (
        sessions.assign(hour=sessions["connect_time_utc"].dt.floor("h"))
        .groupby("hour")
        .size()
        .reindex(frame["hour"], fill_value=0)
        .to_numpy()
    )
    np.testing.assert_array_equal(
        features["sessions_started_lag24h"].iloc[24:].to_numpy(),
        sessions_at_hour[:-24],
    )
    np.testing.assert_array_equal(
        features["sessions_started_lag168h"].iloc[168:].to_numpy(),
        sessions_at_hour[:-168],
    )


@pytest.mark.parametrize(
    "feature_builder",
    [add_session_history_features, add_same_hour_demand_features],
)
def test_feature_values_at_forecast_origin_ignore_target_and_future_changes(feature_builder):
    frame, sessions = _fixture()
    original = feature_builder(frame, sessions) if feature_builder is add_session_history_features \
        else feature_builder(frame)

    changed_frame = frame.copy()
    changed_frame.loc[250:, "kwh_allocated"] += 100000
    changed_sessions = sessions[
        sessions["connect_time_utc"] < frame.loc[250, "hour"]
    ].copy()
    changed = feature_builder(changed_frame, changed_sessions) \
        if feature_builder is add_session_history_features \
        else feature_builder(changed_frame)

    row = 249
    columns = [
        column for column in original.columns
        if column not in {"hour", "kwh_allocated", "_session_count"}
    ]
    pd.testing.assert_series_equal(
        original.loc[row, columns],
        changed.loc[row, columns],
        check_names=False,
        check_dtype=False,
    )


def test_session_rollups_exclude_current_hour_and_future_session_arrivals():
    frame, sessions = _fixture()
    original = add_session_history_features(frame, sessions)
    modified = sessions.copy()
    target_hour = frame.loc[250, "hour"]
    added = pd.DataFrame({
        "source_dataset": ["test"] * 6,
        "entity_id": ["site-a"] * 6,
        "connect_time_utc": [target_hour + pd.Timedelta(minutes=5 * i) for i in range(6)],
    })
    modified = pd.concat([modified, added], ignore_index=True)
    changed = add_session_history_features(frame, modified)
    historical = [
        column for column in original.columns
        if column.startswith("sessions_started_roll_")
    ]
    pd.testing.assert_series_equal(
        original.loc[250, historical],
        changed.loc[250, historical],
        check_names=False,
        check_dtype=False,
    )
    assert (
        original.loc[251, historical].to_numpy()
        != changed.loc[251, historical].to_numpy()
    ).any()


def test_production_features_ignore_target_hour_and_future_demand_or_arrivals():
    frame, sessions = _fixture()
    hourly = frame.assign(
        site="site-a",
        region="France",
    )
    original = P.build_feature_table(hourly, sessions)

    target_hour = frame.loc[250, "hour"]
    changed_hourly = hourly.copy()
    changed_hourly.loc[changed_hourly["hour"] >= target_hour, "kwh_allocated"] += 100000
    changed_sessions = sessions[
        sessions["connect_time_utc"] < target_hour
    ].copy()
    changed = P.build_feature_table(changed_hourly, changed_sessions)

    excluded = {
        "source_dataset",
        "entity_id",
        "site",
        "region",
        "hour",
        "kwh_allocated",
    }
    features = [column for column in original.columns if column not in excluded]
    pd.testing.assert_series_equal(
        original.loc[250, features],
        changed.loc[250, features],
        check_names=False,
        check_dtype=False,
    )


def test_feature_guard_rejects_future_and_contemporaneous_session_columns():
    with pytest.raises(ValueError, match="future information"):
        P.validate_forecast_feature_leakage(
            pd.DataFrame({"future_total_energy": [1.0], "target": [1.0]})
        )
    with pytest.raises(ValueError, match="future information"):
        P.validate_forecast_feature_leakage(
            pd.DataFrame({"sessions_started": [1]})
        )
    with pytest.raises(ValueError, match="future information"):
        P.validate_forecast_feature_leakage(
            pd.DataFrame({"sessions_started_current": [1]})
        )
    with pytest.raises(ValueError, match="future information"):
        P.validate_forecast_feature_leakage(
            pd.DataFrame({"current_hour_energy": [1.0]})
        )


def test_design_matrix_never_exposes_a_target_column():
    frame = pd.DataFrame({
        "hour": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
        "source_dataset": ["test", "test"],
        "region": ["France", "France"],
        "entity_id": ["site-a", "site-a"],
        "site": ["site-a", "site-a"],
        "kwh_allocated": [4.0, 5.0],
        "target": [40.0, 50.0],
        "safe_lag": [3.0, 4.0],
    })
    assert "target" not in P.design_matrix(frame).columns


def test_temporal_splits_follow_row_cutoffs_without_splitting_an_hour():
    hours = pd.date_range("2025-01-01", periods=40, freq="h", tz="UTC")
    frame = pd.DataFrame([
        {"hour": hour, "entity_id": entity, "kwh_allocated": float(index)}
        for index, hour in enumerate(hours)
        for entity in ("site-a", "site-b", "site-c")
    ])
    train, validation, test = temporal_splits(frame)
    ordered = frame.sort_values(["hour", "entity_id"]).reset_index(drop=True)

    assert test["hour"].min() == ordered.iloc[int(len(ordered) * 0.85)]["hour"]
    assert validation["hour"].min() == train["hour"].max() + pd.Timedelta(hours=1)
    assert test["hour"].min() == validation["hour"].max() + pd.Timedelta(hours=1)
    assert not set(train["hour"]) & set(validation["hour"])
    assert not set(validation["hour"]) & set(test["hour"])
