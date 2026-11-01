from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from marl.config import DEMO_STATIONS, MARLConfig
from marl.environment import EnergyManagementEnv, OBSERVATION_FIELDS
from marl.hurdle_experiment import (
    HurdleForecastAdapter,
    _collect_episode,
    evaluate_hurdle_baseline,
    evaluate_hurdle_policy,
    generate_session_scenario,
    station_entity_mapping,
)
from marl.mappo import RolloutBuffer


class RecordingHurdleModel:
    target_mode = "forecast"
    data_source = "synthetic"

    def __init__(self, station_to_entity, event_log=None):
        self.entities = tuple(station_to_entity.values())
        self.meta = {
            "target_mode": self.target_mode,
            "data_source": self.data_source,
            "entities": list(self.entities),
            "expanding_state": {
                entity: [336.0 * 10, 336] for entity in self.entities
            },
        }
        self.station_to_entity = station_to_entity
        self.event_log = event_log
        self.prediction_hours = []

    def predict(self, rows):
        if self.event_log is not None:
            self.event_log.append("forecast")
        assert "actual_demand_next" not in rows.columns
        assert rows["target"].isna().all()
        hour_values = pd.to_datetime(rows["hour"], utc=True).unique()
        assert len(hour_values) == 1
        self.prediction_hours.append(pd.Timestamp(hour_values[0]))
        return (
            np.asarray([
                8.0 + list(self.entities).index(entity)
                for entity in rows["entity_id"]
            ]),
            np.full(len(rows), 0.8),
        )


def fixture_adapter(tmp_path, event_log=None):
    station_ids = tuple(DEMO_STATIONS)
    mapping = station_entity_mapping(station_ids)
    start = pd.Timestamp("2024-12-01T00:00:00Z")
    hours = pd.date_range(end=start, periods=336, freq="h")
    hourly_rows = []
    prior = {}
    for entity in mapping.values():
        for hour in hours:
            hourly_rows.append({
                "source_dataset": "paris",
                "entity_id": entity,
                "site": entity,
                "region": "France",
                "hour": hour,
                "kwh_allocated": 10.0,
            })
        prior[entity] = [3360.0, 336]
    hourly_path = tmp_path / "warmup_hourly.parquet"
    pd.DataFrame(hourly_rows).to_parquet(hourly_path, index=False)
    session_path = tmp_path / "warmup_sessions.parquet"
    pd.DataFrame({
        "source_dataset": pd.Series(dtype=str),
        "entity_id": pd.Series(dtype=str),
        "site": pd.Series(dtype=str),
        "region": pd.Series(dtype=str),
        "connect_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "charge_end_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "energy_kwh": pd.Series(dtype=float),
    }).to_parquet(session_path, index=False)
    model = RecordingHurdleModel(mapping, event_log)
    model.meta["expanding_state"] = prior
    adapter = HurdleForecastAdapter(
        station_ids,
        model=model,
        artifact_directory=tmp_path,
        warmup_hourly_path=hourly_path,
        warmup_sessions_path=session_path,
    )
    return adapter, model


def environment_config():
    return MARLConfig(
        station_ids=tuple(DEMO_STATIONS),
        station_capacities={
            station: spec["capacity"] for station, spec in DEMO_STATIONS.items()
        },
        renewable_availability=30.0,
        grid_capacity=120.0,
        normalize_observations=True,
    )


def make_scenario(adapter, *, horizon=2):
    adapter.reset()
    return generate_session_scenario(
        adapter,
        horizon,
        np.random.default_rng(777),
        renewable_available=30.0,
        grid_capacity=120.0,
    )


def test_station_to_entity_mapping_is_exact_and_rejects_unknown_stations():
    mapping = station_entity_mapping(tuple(DEMO_STATIONS))
    assert mapping == {
        "Bercy": "paris_bercy",
        "Republique": "paris_republique",
        "Montparnasse": "paris_montparnasse",
        "La Defense": "paris_la_defense",
        "Batignolles": "paris_batignolles",
        "Ivry": "paris_ivry",
    }
    with pytest.raises(ValueError, match="exactly the configured demo stations"):
        station_entity_mapping(("Unknown",))


def test_adapter_initializes_336_hour_feature_buffer_from_saved_warmup(tmp_path):
    adapter, _ = fixture_adapter(tmp_path)
    assert adapter.current_hour == pd.Timestamp("2024-12-01T00:00:00Z")
    assert set(adapter.current_demand) == set(DEMO_STATIONS)
    assert all(value == 10.0 for value in adapter.current_demand.values())
    assert all(
        len(adapter.buffer.hourly[entity]) == 336
        for entity in adapter.station_to_entity.values()
    )
    assert adapter.model.meta["expanding_state"]


def test_adapter_returns_six_finite_t_plus_one_forecasts_and_enforces_alignment(tmp_path):
    adapter, model = fixture_adapter(tmp_path)
    current_hour = adapter.current_hour
    forecast = adapter.forecast(adapter.current_demand)

    assert set(forecast) == set(DEMO_STATIONS)
    assert len(forecast) == 6
    assert all(np.isfinite(value) for value in forecast.values())
    assert model.prediction_hours == [current_hour]
    assert adapter._pending_target_hour == current_hour + pd.Timedelta(hours=1)
    with pytest.raises(RuntimeError, match="Advance the feature buffer"):
        adapter.forecast(adapter.current_demand)


def test_adapter_only_advances_after_actual_t_plus_one_is_revealed(tmp_path):
    adapter, model = fixture_adapter(tmp_path)
    adapter.forecast(adapter.current_demand)
    actual = {station: float(index + 20) for index, station in enumerate(DEMO_STATIONS)}
    target_hour = adapter.current_hour + pd.Timedelta(hours=1)
    starts = {
        station: (target_hour + pd.Timedelta(minutes=5),)
        for station in DEMO_STATIONS
    }

    with pytest.raises(RuntimeError, match="Advance the feature buffer"):
        adapter.forecast(adapter.current_demand)
    with pytest.raises(ValueError, match="latest demand already available"):
        adapter.forecast(actual)

    adapter.advance(actual, starts, hour=target_hour)
    next_forecast = adapter.forecast(actual)

    assert set(next_forecast) == set(DEMO_STATIONS)
    assert model.prediction_hours == [
        target_hour - pd.Timedelta(hours=1),
        target_hour,
    ]
    for station, entity in adapter.station_to_entity.items():
        latest_hour, latest_demand = adapter.buffer.hourly[entity][-1]
        assert latest_hour == target_hour
        assert latest_demand == actual[station]
        assert starts[station][0] in adapter.buffer.sessions[entity]


def test_session_scenario_is_reproducible_and_uses_sim_synth_output(tmp_path):
    adapter, _ = fixture_adapter(tmp_path)
    first = generate_session_scenario(
        adapter,
        24,
        np.random.default_rng(23),
        renewable_available=30.0,
        grid_capacity=120.0,
    )
    second = generate_session_scenario(
        adapter,
        24,
        np.random.default_rng(23),
        renewable_available=30.0,
        grid_capacity=120.0,
    )

    assert first.actual_demand_steps == second.actual_demand_steps
    assert first.session_starts_steps == second.session_starts_steps
    assert first.start_hour == adapter.current_hour
    assert len(first.actual_demand_steps) == 24
    assert all(set(step) == set(DEMO_STATIONS) for step in first.actual_demand_steps)
    assert all(
        set(step) == set(DEMO_STATIONS) for step in first.session_starts_steps
    )
    assert sum(
        len(starts)
        for step in first.session_starts_steps
        for starts in step.values()
    ) > 0


def test_training_loop_orders_forecast_actor_actual_then_feature_advance(tmp_path, monkeypatch):
    events = []
    adapter, _ = fixture_adapter(tmp_path, event_log=events)
    scenario = make_scenario(adapter, horizon=2)

    @dataclass
    class DummyAgentConfig:
        reward_mode: str = "individual"

    class DummyAgent:
        num_agents = len(DEMO_STATIONS)
        config = DummyAgentConfig()

        def act(self, local):
            events.append("actor")
            assert local.shape == (6, len(OBSERVATION_FIELDS))
            return (
                np.zeros((6, 2), dtype=np.float32),
                np.zeros((6, 2), dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
            )

    original_advance = adapter.advance

    def record_advance(*args, **kwargs):
        events.append("advance")
        return original_advance(*args, **kwargs)

    monkeypatch.setattr(adapter, "advance", record_advance)

    class ActualRecordingEnvironment(EnergyManagementEnv):
        def step(self, action, forecast, actual_demand_next, **kwargs):
            def reveal_actual():
                events.append("actual")
                return actual_demand_next()

            return super().step(action, forecast, reveal_actual, **kwargs)

    env = ActualRecordingEnvironment(environment_config())
    buffer = RolloutBuffer(2, 6, len(OBSERVATION_FIELDS), 2)
    _collect_episode(env, DummyAgent(), adapter, scenario, buffer)

    assert events == [
        "forecast", "actor", "actual", "advance",
        "forecast", "actor", "actual", "advance",
    ]


def test_hurdle_evaluation_replays_identical_forecasts_to_baseline(tmp_path, monkeypatch):
    adapter, model = fixture_adapter(tmp_path)
    scenario = make_scenario(adapter, horizon=3)
    agents = []

    class RecordingEnvironment(EnergyManagementEnv):
        def __init__(self, config):
            super().__init__(config)
            self.reset_conditions = None
            self.step_forecasts = []
            self.actual_steps = []
            agents.append(self)

        def reset(
            self, current_demand, forecast, *,
            renewable_available=None, grid_capacity=None,
            current_grid_load=0.0, forecast_source="model",
        ):
            self.reset_conditions = (
                dict(current_demand),
                dict(forecast),
                renewable_available,
                grid_capacity,
                current_grid_load,
            )
            return super().reset(
                current_demand,
                forecast,
                renewable_available=renewable_available,
                grid_capacity=grid_capacity,
                current_grid_load=current_grid_load,
                forecast_source=forecast_source,
            )

        def step(self, action, forecast, actual_demand_next, **kwargs):
            self.step_forecasts.append(dict(forecast))
            rewards, done, info = super().step(
                action, forecast, actual_demand_next, **kwargs
            )
            self.actual_steps.append(dict(info["actual_demand_next"]))
            return rewards, done, info

    import marl.hurdle_experiment as experiment

    monkeypatch.setattr(
        experiment, "_new_environment", lambda config: RecordingEnvironment(config)
    )
    env_config = environment_config()
    class DeterministicAgent:
        num_agents = len(DEMO_STATIONS)

        def act_local(self, _index, _observation, *, deterministic):
            assert deterministic is True
            assert 0 <= _index < self.num_agents
            assert _observation.shape == (len(OBSERVATION_FIELDS),)
            return np.zeros(2, dtype=np.float32)

    policy_metrics, forecast_paths = evaluate_hurdle_policy(
        env_config,
        DeterministicAgent(),
        [scenario],
        lambda: HurdleForecastAdapter(
            tuple(DEMO_STATIONS),
            model=model,
            artifact_directory=tmp_path,
            warmup_hourly_path=tmp_path / "warmup_hourly.parquet",
            warmup_sessions_path=tmp_path / "warmup_sessions.parquet",
        ),
    )
    policy_env = agents[-1]
    baseline_metrics = evaluate_hurdle_baseline(
        env_config, [scenario], forecast_paths
    )
    baseline_env = agents[-1]

    assert len(forecast_paths[0]) == scenario.horizon
    assert policy_env.step_forecasts == baseline_env.step_forecasts
    assert policy_env.actual_steps == baseline_env.actual_steps
    assert policy_env.reset_conditions == baseline_env.reset_conditions
    assert policy_env.reset_conditions[0] == baseline_env.reset_conditions[0]
    assert policy_env.reset_conditions[2:] == baseline_env.reset_conditions[2:]
    assert policy_metrics.keys() == baseline_metrics.keys()
    assert model.prediction_hours


def test_scenario_future_actuals_are_not_a_forecast_adapter_input(tmp_path):
    adapter, _ = fixture_adapter(tmp_path)
    scenario = make_scenario(adapter, horizon=2)
    forecast = adapter.forecast(scenario.initial_demand)
    assert set(forecast) == set(DEMO_STATIONS)
    assert "actual_demand_steps" not in forecast
    assert "session_starts_steps" not in forecast
    assert adapter.current_hour == scenario.start_hour
