"""Train and evaluate MAPPO against the station-level energy environment."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .baseline import proportional_allocation
from .config import DEMO_STATIONS, MARLConfig
from .environment import OBSERVATION_FIELDS, EnergyManagementEnv
from .forecasts import ForecastSource, PersistenceForecast
from .mappo import (
    MAPPO,
    MAPPOConfig,
    RolloutBuffer,
    fractions_to_environment_action,
    observations_to_array,
    seed_everything,
)

LOGGER = logging.getLogger("marl.train_mappo")


@dataclass(frozen=True)
class TrainingRunConfig:
    seed: int = 42
    total_episodes: int = 200
    steps_per_episode: int = 128
    evaluation_interval: int = 10
    evaluation_episodes: int = 5
    output_directory: str = "artifacts/marl"

    def __post_init__(self):
        if self.total_episodes <= 0 or self.steps_per_episode <= 0:
            raise ValueError("Training episodes and steps per episode must be positive.")
        if self.evaluation_interval <= 0 or self.evaluation_episodes <= 0:
            raise ValueError("Evaluation interval and episode count must be positive.")


@dataclass(frozen=True)
class EpisodeScenario:
    """A fixed evaluation/training path; future demand is only read by a callback."""

    initial_demand: Mapping[str, float]
    actual_demand_steps: tuple[Mapping[str, float], ...]
    renewable_available: float
    current_grid_load: float = 0.0

    @property
    def horizon(self) -> int:
        return len(self.actual_demand_steps)

    def __post_init__(self):
        if not self.actual_demand_steps:
            raise ValueError("An episode scenario must contain at least one step.")


def build_random_scenario(
    station_ids: Sequence[str],
    station_capacities: Mapping[str, float],
    horizon: int,
    renewable_available: float,
    rng: np.random.Generator,
    *,
    initial_demand: Mapping[str, float] | None = None,
) -> EpisodeScenario:
    """Create bounded numerical demand for a reproducible demonstration run."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    actuals = []
    previous = {
        station_id: float(
            initial_demand[station_id] if initial_demand is not None
            else rng.uniform(0.1, 0.65) * station_capacities[station_id]
        )
        for station_id in station_ids
    }
    initial = dict(previous)
    for _ in range(horizon):
        current_actual = {}
        for station_id in station_ids:
            capacity = station_capacities[station_id]
            mean = 0.58 * previous[station_id] + 0.42 * capacity * 0.45
            noise = rng.normal(0.0, max(capacity * 0.16, 0.05))
            current_actual[station_id] = float(np.clip(mean + noise, 0.0, capacity * 1.25))
        actuals.append(current_actual)
        previous = current_actual
    return EpisodeScenario(
        initial_demand=initial,
        actual_demand_steps=tuple(actuals),
        renewable_available=renewable_available,
    )


def _new_environment(config: MARLConfig) -> EnergyManagementEnv:
    return EnergyManagementEnv(config)


def _evaluate_action_runner(
    env: EnergyManagementEnv,
    scenario: EpisodeScenario,
    action_selector: Callable[
        [Mapping[str, Mapping[str, float]], Mapping[str, float]],
        Mapping[str, Mapping[str, float]],
    ],
    *,
    forecast_source: str,
    forecast_provider: ForecastSource,
    forecast_sequence: Sequence[Mapping[str, float]] | None = None,
) -> tuple[dict[str, float], tuple[Mapping[str, float], ...]]:
    recorded_forecasts = (
        [dict(forecast) for forecast in forecast_sequence]
        if forecast_sequence is not None
        else [dict(forecast_provider.forecast(scenario.initial_demand))]
    )
    if forecast_sequence is not None and len(recorded_forecasts) != scenario.horizon:
        raise ValueError("forecast_sequence length must match the scenario horizon.")
    forecast = dict(recorded_forecasts[0])
    observations, _ = env.reset(
        scenario.initial_demand,
        forecast,
        renewable_available=scenario.renewable_available,
        grid_capacity=env.config.grid_capacity,
        current_grid_load=scenario.current_grid_load,
        forecast_source=forecast_source,
    )
    rewards_total = 0.0
    agent_reward_sum = 0.0
    unserved = 0.0
    actual_demand = 0.0
    served = 0.0
    renewable_used = 0.0
    renewable_available_total = 0.0
    grid_used = 0.0
    peak_realized_grid_usage = 0.0
    grid_excess = 0.0
    unused = 0.0

    for step_index in range(scenario.horizon):
        action = action_selector(observations, forecast)
        rewards, _, info = env.step(
            action,
            forecast,
            lambda index=step_index: scenario.actual_demand_steps[index],
            forecast_source=forecast_source,
        )
        step_reward = float(sum(rewards.values()))
        rewards_total += step_reward
        agent_reward_sum += step_reward / len(env.station_ids)
        unserved += info["actual_unserved"]
        actual_demand += sum(info["actual_demand_next"].values())
        served += info["actual_energy_served"]
        renewable_used += info["renewable_used"]
        renewable_available_total += info["renewable_available"]
        grid_used += info["grid_used"]
        peak_realized_grid_usage = max(
            peak_realized_grid_usage, info["grid_used"]
        )
        grid_excess += info["grid_excess_request"]
        unused += info["unused_or_curtailed_energy"]
        if step_index + 1 < scenario.horizon:
            if forecast_sequence is None:
                forecast = dict(
                    forecast_provider.forecast(info["actual_demand_next"])
                )
                recorded_forecasts.append(forecast)
            else:
                forecast = dict(recorded_forecasts[step_index + 1])
            observations = env.observe(
                forecast, forecast_source=forecast_source
            )

    return {
        "total_reward": rewards_total,
        "mean_agent_reward": agent_reward_sum / scenario.horizon,
        "actual_unserved_demand": unserved / scenario.horizon,
        "renewable_utilization": (
            renewable_used / renewable_available_total
            if renewable_available_total > 0
            else 0.0
        ),
        "grid_energy_used": grid_used,
        "peak_realized_grid_usage": peak_realized_grid_usage,
        "grid_excess_requests": grid_excess,
        "unused_energy": unused / scenario.horizon,
        "demand_served_percentage": (
            100.0 * served / actual_demand if actual_demand > 0 else 100.0
        ),
    }, tuple(recorded_forecasts)


def evaluate_policy(
    env: EnergyManagementEnv,
    agent: MAPPO,
    scenarios: Sequence[EpisodeScenario],
    forecast_source: ForecastSource,
) -> dict[str, float]:
    """Evaluate without sampling; only each actor's local observation is passed in."""
    episode_metrics = []
    for scenario in scenarios:
        def select_action(observations, _):
            local = observations_to_array(
                observations, env.station_ids, OBSERVATION_FIELDS
            )
            fractions = np.stack([
                agent.act_local(index, local[index], deterministic=True)
                for index in range(agent.num_agents)
            ])
            return fractions_to_environment_action(
                fractions, env.station_ids, env.station_capacities
            )

        metrics, _ = _evaluate_action_runner(
            env,
            scenario,
            select_action,
            forecast_source=forecast_source.source_name,
            forecast_provider=forecast_source,
        )
        episode_metrics.append(metrics)
    return _mean_metrics(episode_metrics)


def evaluate_proportional_baseline(
    env: EnergyManagementEnv,
    scenarios: Sequence[EpisodeScenario],
    forecast_source: ForecastSource,
) -> dict[str, float]:
    episode_metrics = []
    for scenario in scenarios:
        def select_action(_, forecast):
            return proportional_allocation(
                forecast,
                station_capacities=env.station_capacities,
                renewable_available=env.renewable_available,
                grid_capacity=env.grid_capacity,
                current_grid_load=env.current_grid_load,
            )

        metrics, _ = _evaluate_action_runner(
            env,
            scenario,
            select_action,
            forecast_source=forecast_source.source_name,
            forecast_provider=forecast_source,
        )
        episode_metrics.append(metrics)
    return _mean_metrics(episode_metrics)


def compare_policy_to_baseline(
    env: EnergyManagementEnv,
    agent: MAPPO,
    scenarios: Sequence[EpisodeScenario],
    forecast_source: ForecastSource,
) -> dict[str, dict[str, float]]:
    """Run both controllers on the exact same forecast and demand scenarios."""
    policy_metrics = []
    baseline_metrics = []
    for scenario in scenarios:
        baseline_env = _new_environment(env.config)

        def select_policy_action(observations, _):
            local = observations_to_array(
                observations, env.station_ids, OBSERVATION_FIELDS
            )
            fractions = np.stack([
                agent.act_local(index, local[index], deterministic=True)
                for index in range(agent.num_agents)
            ])
            return fractions_to_environment_action(
                fractions, env.station_ids, env.station_capacities
            )

        def select_baseline_action(_, forecast):
            return proportional_allocation(
                forecast,
                station_capacities=baseline_env.station_capacities,
                renewable_available=baseline_env.renewable_available,
                grid_capacity=baseline_env.grid_capacity,
                current_grid_load=baseline_env.current_grid_load,
            )

        metrics, shared_forecasts = _evaluate_action_runner(
            env,
            scenario,
            select_policy_action,
            forecast_source=forecast_source.source_name,
            forecast_provider=forecast_source,
        )
        policy_metrics.append(metrics)
        metrics, _ = _evaluate_action_runner(
            baseline_env,
            scenario,
            select_baseline_action,
            forecast_source=forecast_source.source_name,
            forecast_provider=forecast_source,
            forecast_sequence=shared_forecasts,
        )
        baseline_metrics.append(metrics)
    return {
        "mappo": _mean_metrics(policy_metrics),
        "proportional_baseline": _mean_metrics(baseline_metrics),
    }


def _mean_metrics(metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not metrics:
        raise ValueError("At least one evaluation scenario is required.")
    return {
        key: float(np.mean([episode[key] for episode in metrics]))
        for key in metrics[0]
    }


def _collect_training_episode(
    env: EnergyManagementEnv,
    agent: MAPPO,
    scenario: EpisodeScenario,
    buffer: RolloutBuffer,
    forecast_source: ForecastSource,
) -> dict[str, float]:
    forecast = dict(forecast_source.forecast(scenario.initial_demand))
    observations, _ = env.reset(
        scenario.initial_demand,
        forecast,
        renewable_available=scenario.renewable_available,
        grid_capacity=env.config.grid_capacity,
        current_grid_load=scenario.current_grid_load,
        forecast_source=forecast_source.source_name,
    )
    episode_reward = 0.0
    rewards_per_agent = []
    diagnostic = {
        "actual_unserved_demand": 0.0,
        "renewable_utilization": 0.0,
        "renewable_available": 0.0,
        "grid_energy_used": 0.0,
        "grid_excess_requests": 0.0,
        "unused_energy": 0.0,
    }
    for step_index in range(scenario.horizon):
        local = observations_to_array(observations, env.station_ids, OBSERVATION_FIELDS)
        fractions, latent, log_probs, values = agent.act(local)
        action = fractions_to_environment_action(
            fractions, env.station_ids, env.station_capacities
        )
        rewards, _, info = env.step(
            action,
            forecast,
            lambda index=step_index: scenario.actual_demand_steps[index],
            forecast_source=forecast_source.source_name,
        )
        reward_array = np.asarray(
            [rewards[station_id] for station_id in env.station_ids],
            dtype=np.float32,
        )
        if agent.config.reward_mode == "team":
            reward_array[:] = reward_array.mean()
        done = step_index + 1 == scenario.horizon
        buffer.add(local, latent, log_probs, reward_array, values, done)

        episode_reward += float(sum(rewards.values()))
        rewards_per_agent.extend(rewards.values())
        diagnostic["actual_unserved_demand"] += info["actual_unserved"]
        diagnostic["renewable_available"] += info["renewable_available"]
        diagnostic["grid_energy_used"] += info["grid_used"]
        diagnostic["grid_excess_requests"] += info["grid_excess_request"]
        diagnostic["unused_energy"] += info["unused_or_curtailed_energy"]
        diagnostic["renewable_utilization"] += info["renewable_used"]
        if step_index + 1 < scenario.horizon:
            forecast = dict(forecast_source.forecast(info["actual_demand_next"]))
            observations = env.observe(
                forecast,
                forecast_source=forecast_source.source_name,
            )

    horizon = scenario.horizon
    diagnostic["actual_unserved_demand"] /= horizon
    diagnostic["unused_energy"] /= horizon
    diagnostic["renewable_utilization"] = (
        diagnostic["renewable_utilization"] / diagnostic["renewable_available"]
        if diagnostic["renewable_available"] > 0
        else 0.0
    )
    return {
        "episode_reward": episode_reward,
        "mean_agent_reward": float(np.mean(rewards_per_agent)),
        **diagnostic,
    }


def train(
    environment_config: MARLConfig,
    mappo_config: MAPPOConfig,
    run_config: TrainingRunConfig,
    forecast_source: ForecastSource,
    *,
    output_directory: str | Path | None = None,
) -> tuple[MAPPO, dict[str, object]]:
    if forecast_source.source_name == "oracle":
        raise ValueError("Oracle forecasts are prohibited during MAPPO training.")
    if run_config.steps_per_episode != mappo_config.rollout_length:
        raise ValueError(
            "steps_per_episode must match mappo_config.rollout_length; "
            "each episode is one on-policy rollout."
        )
    seed_everything(run_config.seed)
    rng = np.random.default_rng(run_config.seed)
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
    destination = Path(output_directory or run_config.output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    training_configuration: dict[str, object] = {
        **asdict(run_config),
        **asdict(mappo_config),
        "number_of_agents": len(env.station_ids),
        "station_ids": list(env.station_ids),
        "observation_fields": list(OBSERVATION_FIELDS),
        "observation_dimension": len(OBSERVATION_FIELDS),
        "action_dimension": 2,
        "forecast_source": forecast_source.source_name,
        "environment_normalization_enabled": environment_config.normalize_observations,
        "environment_normalization": {
            "method": "environment-observation-scaling",
            "configured": environment_config.normalize_observations,
        },
        "total_training_episodes": run_config.total_episodes,
        "total_training_steps": run_config.total_episodes * run_config.steps_per_episode,
    }
    (destination / "training_config.json").write_text(
        json.dumps(training_configuration, indent=2), encoding="utf-8"
    )

    for episode_index in range(1, run_config.total_episodes + 1):
        scenario = build_random_scenario(
            env.station_ids,
            env.station_capacities,
            run_config.steps_per_episode,
            environment_config.renewable_availability,
            rng,
        )
        diagnostics = _collect_training_episode(
            env, agent, scenario, buffer, forecast_source
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
            diagnostics["grid_energy_used"],
            diagnostics["grid_excess_requests"],
            diagnostics["unused_energy"],
        )
        if episode_index % run_config.evaluation_interval == 0:
            eval_rng = np.random.default_rng(run_config.seed + episode_index)
            evaluation_scenarios = [
                build_random_scenario(
                    env.station_ids,
                    env.station_capacities,
                    run_config.steps_per_episode,
                    environment_config.renewable_availability,
                    eval_rng,
                )
                for _ in range(run_config.evaluation_episodes)
            ]
            metrics = compare_policy_to_baseline(
                env, agent, evaluation_scenarios, forecast_source
            )
            LOGGER.info(
                "evaluation episode=%d mappo=%s proportional_baseline=%s",
                episode_index,
                json.dumps(metrics["mappo"], sort_keys=True),
                json.dumps(metrics["proportional_baseline"], sort_keys=True),
            )
            agent.save_checkpoint(
                destination / f"mappo_episode_{episode_index:06d}.pt",
                training_config=training_configuration,
                normalization_config=training_configuration["environment_normalization"],
            )

    agent.save_checkpoint(
        destination / "mappo_final.pt",
        training_config=training_configuration,
        normalization_config=training_configuration["environment_normalization"],
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
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument("--forecast-source", choices=("persistence",), default="persistence")
    parser.add_argument("--output-directory", default="artifacts/marl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    training_config = TrainingRunConfig(
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
    train(
        _demo_environment_config(),
        mappo_config,
        training_config,
        PersistenceForecast(),
    )


if __name__ == "__main__":
    main()
