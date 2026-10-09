"""Causal synthetic-session scenarios and existing hurdle-model MAPPO runner."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from session_to_hourly import sessions_to_hourly
from sim import synth
from sim.features_rt import FeatureBuffer
from sim.model_service import ART as MODEL_ARTIFACT_DIRECTORY
from sim.model_service import ModelService

from .baseline import proportional_allocation
from .config import DEMO_STATIONS, MARLConfig
from .environment import OBSERVATION_FIELDS, EnergyManagementEnv
from .mappo import (
    MAPPO,
    MAPPOConfig,
    RolloutBuffer,
    fractions_to_environment_action,
    observations_to_array,
    seed_everything,
)
from .train_mappo import TrainingRunConfig, _mean_metrics, _new_environment

LOGGER = logging.getLogger("marl.hurdle_experiment")
SESSION_COLUMNS = (
    "source_dataset",
    "entity_id",
    "site",
    "region",
    "connect_time_utc",
    "charge_end_time_utc",
    "energy_kwh",
)


def station_entity_mapping(
    station_ids: Sequence[str],
) -> dict[str, str]:
    """Resolve MARL station labels to the trained Paris model entities."""
    if set(station_ids) != set(DEMO_STATIONS):
        raise ValueError(
            "The existing hurdle artifacts support exactly the configured demo "
            f"stations; received {sorted(station_ids)}."
        )
    return {
        station_id: str(DEMO_STATIONS[station_id]["entity_id"])
        for station_id in station_ids
    }


class HurdleForecastAdapter:
    """Stateful causal adapter around the existing ModelService artifacts."""

    source_name = "model"

    def __init__(
        self,
        station_ids: Sequence[str],
        *,
        model: ModelService | None = None,
        artifact_directory: str | Path = MODEL_ARTIFACT_DIRECTORY,
        warmup_hourly_path: str | Path | None = None,
        warmup_sessions_path: str | Path | None = None,
    ):
        self.station_ids = tuple(station_ids)
        self.station_to_entity = station_entity_mapping(self.station_ids)
        self.entity_to_station = {
            entity_id: station_id
            for station_id, entity_id in self.station_to_entity.items()
        }
        self.model = model or ModelService(artifact_directory)
        self.artifact_directory = Path(artifact_directory)
        if self.model.target_mode != "forecast":
            raise ValueError("The saved hurdle model must use target_mode='forecast'.")
        if self.model.data_source != "synthetic":
            raise ValueError(
                "This session-compatible experiment requires the existing "
                "synthetic-trained hurdle artifacts."
            )
        missing = set(self.station_to_entity.values()) - set(
            map(str, self.model.meta.get("entities", ()))
        )
        if missing:
            raise ValueError(
                f"The saved model does not contain required entities: {sorted(missing)}."
            )

        self.warmup_hourly_path = Path(
            warmup_hourly_path or self.artifact_directory / "warmup_hourly.parquet"
        )
        self.warmup_sessions_path = Path(
            warmup_sessions_path or self.artifact_directory / "warmup_sessions.parquet"
        )
        missing_files = [
            str(path)
            for path in (self.warmup_hourly_path, self.warmup_sessions_path)
            if not path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Missing hurdle warm-up data: {missing_files}."
            )
        self.reset()

    def reset(self) -> Mapping[str, float]:
        """Reload the saved causal warm-up state and return demand at its last hour."""
        hourly = pd.read_parquet(self.warmup_hourly_path)
        sessions = pd.read_parquet(self.warmup_sessions_path)
        self.warmup_sessions = sessions.copy()
        self.buffer = FeatureBuffer(list(self.station_to_entity.values()))
        self.buffer.warm(
            hourly,
            sessions,
            prior=self.model.meta.get("expanding_state"),
        )
        self.current_hour = self.buffer.last_hour()
        if self.current_hour is None:
            raise ValueError("Hurdle warm-up contains no hourly observations.")
        latest: dict[str, float] = {}
        for station_id, entity_id in self.station_to_entity.items():
            if not self.buffer.hourly[entity_id]:
                raise ValueError(f"No warm-up history for model entity {entity_id}.")
            entity_hour, value = self.buffer.hourly[entity_id][-1]
            if pd.Timestamp(entity_hour) != pd.Timestamp(self.current_hour):
                raise ValueError("Station warm-up histories do not end at the same hour.")
            latest[station_id] = float(value)
        self.current_demand = latest
        self._pending_target_hour: pd.Timestamp | None = None
        return dict(latest)

    def forecast(
        self, current_demand: Mapping[str, float]
    ) -> Mapping[str, float]:
        """Forecast t+1 from the warmed/advanced feature state through t."""
        self._validate_station_mapping(current_demand, "current_demand")
        if dict(current_demand) != self.current_demand:
            raise ValueError(
                "Forecast input must match the latest demand already available "
                "in the feature buffer."
            )
        if self._pending_target_hour is not None:
            raise RuntimeError(
                "Advance the feature buffer with the realized hour before forecasting again."
            )
        rows = self.buffer.feature_rows(target_mode="forecast")
        if rows is None or len(rows) != len(self.station_ids):
            raise ValueError("Could not produce exactly one feature row per station.")
        rows = rows.sort_values("entity_id").reset_index(drop=True)
        expected_entities = set(self.station_to_entity.values())
        if set(rows["entity_id"].astype(str)) != expected_entities:
            raise ValueError("Feature rows do not match the configured station entities.")
        times = pd.to_datetime(rows["hour"], utc=True).unique()
        if len(times) != 1:
            raise ValueError("Feature rows must share a single prediction timestamp.")
        prediction_hour = pd.Timestamp(times[0])
        if prediction_hour != pd.Timestamp(self.current_hour):
            raise ValueError("Feature timestamp does not match the current state hour.")

        predictions, _ = self.model.predict(rows)
        if len(predictions) != len(rows):
            raise ValueError("The hurdle model returned a mismatched prediction count.")
        by_entity = {
            str(entity_id): float(prediction)
            for entity_id, prediction in zip(rows["entity_id"], predictions)
        }
        forecast = {
            station_id: by_entity[entity_id]
            for station_id, entity_id in self.station_to_entity.items()
        }
        if not np.isfinite(list(forecast.values())).all():
            raise ValueError("Hurdle model returned a non-finite demand prediction.")
        self._pending_target_hour = prediction_hour + pd.Timedelta(hours=1)
        return forecast

    def advance(
        self,
        actual_demand: Mapping[str, float],
        session_starts: Mapping[str, Sequence[pd.Timestamp]],
        *,
        hour: pd.Timestamp,
    ) -> None:
        """Append one realized hour after env.step() has revealed its actuals."""
        if self._pending_target_hour is None:
            raise RuntimeError("A forecast must be issued before advancing feature state.")
        target_hour = pd.Timestamp(hour)
        if target_hour.tzinfo is None:
            raise ValueError("Feature timestamps must be timezone-aware.")
        target_hour = target_hour.tz_convert("UTC")
        if target_hour != self._pending_target_hour:
            raise ValueError(
                f"Expected realized hour {self._pending_target_hour}, got {target_hour}."
            )
        self._validate_station_mapping(actual_demand, "actual_demand")
        self._validate_station_mapping(session_starts, "session_starts")
        for station_id, entity_id in self.station_to_entity.items():
            starts = [
                pd.Timestamp(value).tz_convert("UTC")
                for value in session_starts[station_id]
            ]
            if any(value.floor("h") != target_hour for value in starts):
                raise ValueError(
                    f"Session starts for {station_id} must belong to {target_hour}."
                )
            value = float(actual_demand[station_id])
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"Actual demand for {station_id} must be finite and non-negative."
                )
            self.buffer.append(entity_id, target_hour, value, starts)
        self.current_hour = target_hour
        self.current_demand = {
            station_id: float(actual_demand[station_id])
            for station_id in self.station_ids
        }
        self._pending_target_hour = None

    def _validate_station_mapping(self, values, name):
        if set(values) != set(self.station_ids):
            missing = sorted(set(self.station_ids) - set(values))
            extra = sorted(set(values) - set(self.station_ids))
            raise ValueError(
                f"{name} station IDs mismatch; missing={missing}, extra={extra}."
            )


@dataclass(frozen=True)
class HurdleEpisodeScenario:
    """A generated future path kept separate from actor observations."""

    initial_demand: Mapping[str, float]
    start_hour: pd.Timestamp
    actual_demand_steps: tuple[Mapping[str, float], ...]
    session_starts_steps: tuple[Mapping[str, tuple[pd.Timestamp, ...]], ...]
    renewable_available: float
    grid_capacity: float
    current_grid_load: float = 0.0

    @property
    def horizon(self) -> int:
        return len(self.actual_demand_steps)

    def __post_init__(self):
        if not self.actual_demand_steps:
            raise ValueError("A Hurdle episode scenario must contain at least one step.")
        if len(self.actual_demand_steps) != len(self.session_starts_steps):
            raise ValueError("Actual-demand and session-start paths must have equal length.")


def generate_session_scenario(
    adapter: HurdleForecastAdapter,
    horizon: int,
    rng: np.random.Generator,
    *,
    renewable_available: float,
    grid_capacity: float,
    current_grid_load: float = 0.0,
) -> HurdleEpisodeScenario:
    """Generate future sessions using the saved model's sim.synth process."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    start_hour = pd.Timestamp(adapter.current_hour).tz_convert("UTC")
    initial = dict(adapter.current_demand)
    sessions: list[dict[str, object]] = []
    entity_ids = set(adapter.station_to_entity.values())
    history = adapter.warmup_sessions
    if not history.empty:
        history = history[
            history["entity_id"].astype(str).isin(entity_ids)
            & (pd.to_datetime(history["connect_time_utc"], utc=True) <= start_hour)
            & (pd.to_datetime(history["charge_end_time_utc"], utc=True) > start_hour)
        ]
        sessions.extend(
            history.loc[:, SESSION_COLUMNS].to_dict(orient="records")
        )
    starts_by_hour: list[dict[str, tuple[pd.Timestamp, ...]]] = []
    for step_index in range(1, horizon + 1):
        hour = start_hour + pd.Timedelta(hours=step_index)
        local = hour.tz_convert("Europe/Paris")
        hour_starts: dict[str, tuple[pd.Timestamp, ...]] = {}
        for station_id, entity_id in adapter.station_to_entity.items():
            count = rng.poisson(
                synth.arrival_rate(
                    entity_id,
                    local.hour,
                    local.weekday() >= 5,
                    dayofyear=local.dayofyear,
                )
            )
            connection_times = tuple(
                hour + pd.Timedelta(minutes=float(rng.uniform(0, 60)))
                for _ in range(count)
            )
            hour_starts[station_id] = connection_times
            for connected_at in connection_times:
                duration_hours, energy_kwh = synth.sample_session(rng, entity_id)
                sessions.append({
                    "source_dataset": "paris",
                    "entity_id": entity_id,
                    "site": entity_id,
                    "region": "France",
                    "connect_time_utc": connected_at,
                    "charge_end_time_utc": connected_at
                    + pd.Timedelta(hours=duration_hours),
                    "energy_kwh": round(energy_kwh, 4),
                })
        starts_by_hour.append(hour_starts)

    session_frame = pd.DataFrame(sessions, columns=SESSION_COLUMNS)
    if session_frame.empty:
        session_frame = pd.DataFrame(columns=SESSION_COLUMNS)
    if len(session_frame):
        hourly = sessions_to_hourly(session_frame)
        demand_by_hour = {
            (str(row.entity_id), pd.Timestamp(row.hour).tz_convert("UTC")):
                float(row.kwh_allocated)
            for row in hourly.itertuples(index=False)
        }
    else:
        demand_by_hour = {}

    actual_steps = []
    for step_index in range(1, horizon + 1):
        target_hour = start_hour + pd.Timedelta(hours=step_index)
        actual_steps.append({
            station_id: demand_by_hour.get((entity_id, target_hour), 0.0)
            for station_id, entity_id in adapter.station_to_entity.items()
        })
    return HurdleEpisodeScenario(
        initial_demand=initial,
        start_hour=start_hour,
        actual_demand_steps=tuple(actual_steps),
        session_starts_steps=tuple(starts_by_hour),
        renewable_available=float(renewable_available),
        grid_capacity=float(grid_capacity),
        current_grid_load=float(current_grid_load),
    )


def generate_synthetic_scenarios(
    adapter: HurdleForecastAdapter,
    count: int,
    horizon: int,
    rng: np.random.Generator,
    *,
    renewable_available: float,
    grid_capacity: float,
) -> list[HurdleEpisodeScenario]:
    if count <= 0:
        raise ValueError("count must be positive.")
    scenarios = []
    for _ in range(count):
        adapter.reset()
        scenarios.append(
            generate_session_scenario(
                adapter,
                horizon,
                rng,
                renewable_available=renewable_available,
                grid_capacity=grid_capacity,
            )
        )
    return scenarios


def _advance_after_step(adapter, scenario, step_index, info):
    actuals = info["actual_demand_next"]
    expected_actuals = scenario.actual_demand_steps[step_index]
    if dict(actuals) != dict(expected_actuals):
        raise ValueError("Environment actuals differ from the hidden scenario step.")
    target_hour = scenario.start_hour + pd.Timedelta(hours=step_index + 1)
    adapter.advance(
        actuals,
        scenario.session_starts_steps[step_index],
        hour=target_hour,
    )


def _collect_episode(
    env: EnergyManagementEnv,
    agent: MAPPO,
    adapter: HurdleForecastAdapter,
    scenario: HurdleEpisodeScenario,
    buffer: RolloutBuffer,
) -> tuple[dict[str, float], tuple[Mapping[str, float], ...]]:
    adapter.reset()
    if dict(scenario.initial_demand) != dict(adapter.current_demand):
        raise ValueError("Scenario initial demand must match the saved warm-up endpoint.")
    forecast = dict(adapter.forecast(adapter.current_demand))
    observations, _ = env.reset(
        scenario.initial_demand,
        forecast,
        renewable_available=scenario.renewable_available,
        grid_capacity=scenario.grid_capacity,
        current_grid_load=scenario.current_grid_load,
        forecast_source=adapter.source_name,
    )
    forecasts = [forecast]
    reward_total = 0.0
    agent_rewards = []
    diagnostics = {
        "actual_unserved_demand": 0.0,
        "renewable_used": 0.0,
        "renewable_available": 0.0,
        "grid_used": 0.0,
        "grid_excess_requests": 0.0,
        "unused_energy": 0.0,
    }
    for step_index in range(scenario.horizon):
        local = observations_to_array(observations, env.station_ids, OBSERVATION_FIELDS)
        fractions, latent, log_probabilities, values = agent.act(local)
        action = fractions_to_environment_action(
            fractions, env.station_ids, env.station_capacities
        )
        rewards, _, info = env.step(
            action,
            forecast,
            lambda index=step_index: scenario.actual_demand_steps[index],
            forecast_source=adapter.source_name,
        )
        reward_array = np.asarray(
            [rewards[station_id] for station_id in env.station_ids],
            dtype=np.float32,
        )
        if agent.config.reward_mode == "team":
            reward_array[:] = reward_array.mean()
        buffer.add(
            local,
            latent,
            log_probabilities,
            reward_array,
            values,
            step_index + 1 == scenario.horizon,
        )
        reward_total += float(sum(rewards.values()))
        agent_rewards.extend(rewards.values())
        diagnostics["actual_unserved_demand"] += info["actual_unserved"]
        diagnostics["renewable_used"] += info["renewable_used"]
        diagnostics["renewable_available"] += info["renewable_available"]
        diagnostics["grid_used"] += info["grid_used"]
        diagnostics["grid_excess_requests"] += info["grid_excess_request"]
        diagnostics["unused_energy"] += info["unused_or_curtailed_energy"]

        _advance_after_step(adapter, scenario, step_index, info)
        if step_index + 1 < scenario.horizon:
            forecast = dict(adapter.forecast(adapter.current_demand))
            forecasts.append(forecast)
            observations = env.observe(forecast, forecast_source=adapter.source_name)

    diagnostics["actual_unserved_demand"] /= scenario.horizon
    diagnostics["unused_energy"] /= scenario.horizon
    diagnostics["renewable_utilization"] = (
        diagnostics["renewable_used"] / diagnostics["renewable_available"]
        if diagnostics["renewable_available"] > 0
        else 0.0
    )
    return {
        "episode_reward": reward_total,
        "mean_agent_reward": float(np.mean(agent_rewards)),
        **diagnostics,
    }, tuple(forecasts)


def evaluate_hurdle_policy(
    environment_config: MARLConfig,
    agent: MAPPO,
    scenarios: Sequence[HurdleEpisodeScenario],
    adapter_factory,
) -> tuple[dict[str, float], list[tuple[Mapping[str, float], ...]]]:
    episode_metrics = []
    forecast_paths = []
    for scenario in scenarios:
        env = _new_environment(environment_config)
        adapter = adapter_factory()
        adapter.reset()
        if dict(scenario.initial_demand) != dict(adapter.current_demand):
            raise ValueError("Evaluation scenario and model warm-up do not share initial demand.")
        forecast = dict(adapter.forecast(adapter.current_demand))
        forecasts = [forecast]
        observations, _ = env.reset(
            scenario.initial_demand,
            forecast,
            renewable_available=scenario.renewable_available,
            grid_capacity=scenario.grid_capacity,
            current_grid_load=scenario.current_grid_load,
            forecast_source=adapter.source_name,
        )
        rewards_total = 0.0
        actual_unserved = 0.0
        actual_total = 0.0
        served_total = 0.0
        renewable_used = 0.0
        renewable_available_total = 0.0
        grid_used = 0.0
        peak_realized_grid_usage = 0.0
        grid_excess = 0.0
        unused = 0.0
        for step_index in range(scenario.horizon):
            local = observations_to_array(
                observations, env.station_ids, OBSERVATION_FIELDS
            )
            fractions = np.stack([
                agent.act_local(index, local[index], deterministic=True)
                for index in range(agent.num_agents)
            ])
            action = fractions_to_environment_action(
                fractions, env.station_ids, env.station_capacities
            )
            rewards, _, info = env.step(
                action,
                forecast,
                lambda index=step_index: scenario.actual_demand_steps[index],
                forecast_source=adapter.source_name,
            )
            rewards_total += float(sum(rewards.values()))
            actual_unserved += info["actual_unserved"]
            actual_total += sum(info["actual_demand_next"].values())
            served_total += info["actual_energy_served"]
            renewable_used += info["renewable_used"]
            renewable_available_total += info["renewable_available"]
            grid_used += info["grid_used"]
            peak_realized_grid_usage = max(
                peak_realized_grid_usage, info["grid_used"]
            )
            grid_excess += info["grid_excess_request"]
            unused += info["unused_or_curtailed_energy"]
            _advance_after_step(adapter, scenario, step_index, info)
            if step_index + 1 < scenario.horizon:
                forecast = dict(adapter.forecast(adapter.current_demand))
                forecasts.append(forecast)
                observations = env.observe(
                    forecast, forecast_source=adapter.source_name
                )
        episode_metrics.append({
            "total_reward": rewards_total,
            "actual_unserved_demand": actual_unserved / scenario.horizon,
            "demand_served_percentage": (
                100.0 * served_total / actual_total if actual_total > 0 else 100.0
            ),
            "renewable_utilization": (
                renewable_used / renewable_available_total
                if renewable_available_total > 0
                else 0.0
            ),
            "grid_energy_used": grid_used,
            "peak_realized_grid_usage": peak_realized_grid_usage,
            "grid_excess_requests": grid_excess,
            "unused_energy": unused / scenario.horizon,
        })
        forecast_paths.append(tuple(forecasts))
    return _mean_metrics(episode_metrics), forecast_paths


def evaluate_hurdle_baseline(
    environment_config: MARLConfig,
    scenarios: Sequence[HurdleEpisodeScenario],
    forecast_paths: Sequence[Sequence[Mapping[str, float]]],
) -> dict[str, float]:
    if len(scenarios) != len(forecast_paths):
        raise ValueError("Each scenario must have exactly one replayed forecast path.")
    episode_metrics = []
    for scenario, forecasts in zip(scenarios, forecast_paths):
        if len(forecasts) != scenario.horizon:
            raise ValueError("Forecast path length must equal scenario horizon.")
        env = _new_environment(environment_config)
        env.reset(
            scenario.initial_demand,
            forecasts[0],
            renewable_available=scenario.renewable_available,
            grid_capacity=scenario.grid_capacity,
            current_grid_load=scenario.current_grid_load,
            forecast_source="model",
        )
        rewards_total = 0.0
        actual_unserved = 0.0
        actual_total = 0.0
        served_total = 0.0
        renewable_used = 0.0
        renewable_available_total = 0.0
        grid_used = 0.0
        peak_realized_grid_usage = 0.0
        grid_excess = 0.0
        unused = 0.0
        for step_index in range(scenario.horizon):
            forecast = forecasts[step_index]
            action = proportional_allocation(
                forecast,
                station_capacities=env.station_capacities,
                renewable_available=env.renewable_available,
                grid_capacity=env.grid_capacity,
                current_grid_load=env.current_grid_load,
            )
            rewards, _, info = env.step(
                action,
                forecast,
                lambda index=step_index: scenario.actual_demand_steps[index],
                forecast_source="model",
            )
            rewards_total += float(sum(rewards.values()))
            actual_unserved += info["actual_unserved"]
            actual_total += sum(info["actual_demand_next"].values())
            served_total += info["actual_energy_served"]
            renewable_used += info["renewable_used"]
            renewable_available_total += info["renewable_available"]
            grid_used += info["grid_used"]
            peak_realized_grid_usage = max(
                peak_realized_grid_usage, info["grid_used"]
            )
            grid_excess += info["grid_excess_request"]
            unused += info["unused_or_curtailed_energy"]
            if step_index + 1 < scenario.horizon:
                env.observe(forecasts[step_index + 1], forecast_source="model")
        episode_metrics.append({
            "total_reward": rewards_total,
            "actual_unserved_demand": actual_unserved / scenario.horizon,
            "demand_served_percentage": (
                100.0 * served_total / actual_total if actual_total > 0 else 100.0
            ),
            "renewable_utilization": (
                renewable_used / renewable_available_total
                if renewable_available_total > 0
                else 0.0
            ),
            "grid_energy_used": grid_used,
            "peak_realized_grid_usage": peak_realized_grid_usage,
            "grid_excess_requests": grid_excess,
            "unused_energy": unused / scenario.horizon,
        })
    return _mean_metrics(episode_metrics)


def train_hurdle_mappo(
    environment_config: MARLConfig,
    mappo_config: MAPPOConfig,
    run_config: TrainingRunConfig,
    *,
    artifact_directory: str | Path = MODEL_ARTIFACT_DIRECTORY,
) -> tuple[MAPPO, dict[str, object]]:
    destination = Path(run_config.output_directory)
    if (destination / "mappo_final.pt").exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Hurdle→MAPPO checkpoint in {destination}."
        )
    adapter_factory = lambda: HurdleForecastAdapter(
        environment_config.station_ids,
        artifact_directory=artifact_directory,
    )
    adapter = adapter_factory()
    seed_everything(run_config.seed)
    rng = np.random.default_rng(run_config.seed)
    eval_rng = np.random.default_rng(run_config.seed + 10_000_000)
    env = _new_environment(environment_config)
    agent = MAPPO(
        len(env.station_ids),
        len(OBSERVATION_FIELDS),
        action_size=2,
        config=mappo_config,
    )
    buffer = RolloutBuffer(
        mappo_config.rollout_length,
        len(env.station_ids),
        len(OBSERVATION_FIELDS),
        2,
    )
    destination.mkdir(parents=True, exist_ok=True)
    training_configuration = {
        **asdict(run_config),
        **asdict(mappo_config),
        "number_of_agents": len(env.station_ids),
        "station_ids": list(env.station_ids),
        "station_to_entity": adapter.station_to_entity,
        "observation_fields": list(OBSERVATION_FIELDS),
        "observation_dimension": len(OBSERVATION_FIELDS),
        "action_dimension": 2,
        "forecast_source": "existing_hurdle_xgboost",
        "model_artifact_directory": str(Path(artifact_directory)),
        "model_data_source": adapter.model.data_source,
        "model_target_mode": adapter.model.target_mode,
        "scenario_generator": "sim.synth.arrival_rate + sim.synth.sample_session + session_to_hourly",
        "normalization": {
            "enabled": environment_config.normalize_observations,
            "method": "environment-observation-scaling",
        },
    }
    (destination / "training_config.json").write_text(
        json.dumps(training_configuration, indent=2), encoding="utf-8"
    )
    for episode_index in range(1, run_config.total_episodes + 1):
        adapter.reset()
        scenario = generate_session_scenario(
            adapter,
            run_config.steps_per_episode,
            rng,
            renewable_available=environment_config.renewable_availability,
            grid_capacity=environment_config.grid_capacity,
        )
        diagnostics, _ = _collect_episode(
            env, agent, adapter, scenario, buffer
        )
        losses = agent.update(buffer)
        LOGGER.info(
            "episode=%d reward=%.3f mean_agent_reward=%.3f policy_loss=%.5f "
            "value_loss=%.5f entropy=%.5f actual_unserved_kwh=%.3f "
            "renewable_utilization=%.3f grid_used_kwh=%.3f "
            "grid_excess_requests_kwh=%.3f unused_kwh=%.3f",
            episode_index,
            diagnostics["episode_reward"],
            diagnostics["mean_agent_reward"],
            losses["policy_loss"],
            losses["value_loss"],
            losses["entropy"],
            diagnostics["actual_unserved_demand"],
            diagnostics["renewable_utilization"],
            diagnostics["grid_used"],
            diagnostics["grid_excess_requests"],
            diagnostics["unused_energy"],
        )
        if episode_index % run_config.evaluation_interval == 0:
            eval_adapter = adapter_factory()
            eval_scenarios = generate_synthetic_scenarios(
                eval_adapter,
                run_config.evaluation_episodes,
                run_config.steps_per_episode,
                eval_rng,
                renewable_available=environment_config.renewable_availability,
                grid_capacity=environment_config.grid_capacity,
            )
            policy_metrics, forecast_paths = evaluate_hurdle_policy(
                environment_config, agent, eval_scenarios, adapter_factory
            )
            baseline_metrics = evaluate_hurdle_baseline(
                environment_config, eval_scenarios, forecast_paths
            )
            LOGGER.info(
                "evaluation episode=%d mappo=%s proportional_baseline=%s",
                episode_index,
                json.dumps(policy_metrics, sort_keys=True),
                json.dumps(baseline_metrics, sort_keys=True),
            )
            agent.save_checkpoint(
                destination / f"mappo_episode_{episode_index:06d}.pt",
                training_config=training_configuration,
                normalization_config=training_configuration["normalization"],
            )
    agent.save_checkpoint(
        destination / "mappo_final.pt",
        training_config=training_configuration,
        normalization_config=training_configuration["normalization"],
    )
    import torch

    torch.save(
        [actor.state_dict() for actor in agent.policy.actors],
        destination / "actors_final.pt",
    )
    torch.save(agent.policy.critic.state_dict(), destination / "critic_final.pt")
    return agent, training_configuration


def _demo_environment_config() -> MARLConfig:
    return MARLConfig(
        station_ids=tuple(DEMO_STATIONS),
        station_capacities={
            station: entry["capacity"] for station, entry in DEMO_STATIONS.items()
        },
        renewable_availability=100.0,
        grid_capacity=350.0,
        normalize_observations=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps-per-episode", type=int, default=128)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--output-directory", default="artifacts/marl/hurdle_mappo")
    parser.add_argument("--model-artifact-directory", default=str(MODEL_ARTIFACT_DIRECTORY))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_config = TrainingRunConfig(
        seed=args.seed,
        total_episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        output_directory=args.output_directory,
    )
    mappo_config = MAPPOConfig(
        rollout_length=args.steps_per_episode,
        device=args.device,
    )
    train_hurdle_mappo(
        _demo_environment_config(),
        mappo_config,
        run_config,
        artifact_directory=args.model_artifact_directory,
    )


if __name__ == "__main__":
    main()
