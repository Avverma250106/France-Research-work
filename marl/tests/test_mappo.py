import numpy as np
import pytest
import torch

from marl.config import MARLConfig
from marl.environment import OBSERVATION_FIELDS, EnergyManagementEnv
from marl.forecasts import ModelForecast, OracleForecast, PersistenceForecast
from marl.mappo import (
    MAPPO,
    MAPPOConfig,
    MAPPOPolicy,
    RolloutBuffer,
    clipped_surrogate_loss,
    fractions_to_environment_action,
    generalized_advantage_estimate,
    observations_to_array,
)
from marl.train_mappo import (
    EpisodeScenario,
    TrainingRunConfig,
    _collect_training_episode,
    _evaluate_action_runner,
    compare_policy_to_baseline,
    evaluate_policy,
    train,
)


def tiny_config(**overrides):
    settings = {
        "hidden_size": 16,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "gamma": 0.9,
        "gae_lambda": 0.8,
        "clip_epsilon": 0.2,
        "entropy_coefficient": 0.01,
        "value_coefficient": 0.5,
        "rollout_length": 4,
        "minibatch_size": 2,
        "ppo_epochs": 2,
        "device": "cpu",
    }
    settings.update(overrides)
    return MAPPOConfig(**settings)


def make_environment():
    return EnergyManagementEnv(MARLConfig(
        station_ids=("A", "B", "C"),
        station_capacities={"A": 10.0, "B": 12.0, "C": 15.0},
        renewable_availability=8.0,
        grid_capacity=15.0,
        normalize_observations=True,
    ))


def scenario():
    return EpisodeScenario(
        initial_demand={"A": 3.0, "B": 4.0, "C": 5.0},
        actual_demand_steps=(
            {"A": 4.0, "B": 3.0, "C": 7.0},
            {"A": 5.0, "B": 5.0, "C": 8.0},
            {"A": 4.0, "B": 6.0, "C": 9.0},
            {"A": 3.0, "B": 7.0, "C": 6.0},
        ),
        renewable_available=8.0,
    )


def test_actor_and_centralized_critic_dimension_handling():
    policy = MAPPOPolicy(num_agents=3, observation_size=6, action_size=2, hidden_size=16)
    local = torch.zeros(5, 3, 6)
    global_state = local.reshape(5, -1)

    _, _, values = policy.evaluate_actions(
        local, global_state, torch.zeros(5, 3, 2)
    )

    assert values.shape == (5, 3)
    assert len(policy.actors) == 3
    assert policy.critic[0].in_features == 18
    assert policy.critic[-1].out_features == 3


def test_actor_action_distribution_is_bounded_and_correctly_shaped():
    agent = MAPPO(3, 6, config=tiny_config())
    fractions, latent, log_probability, values = agent.act(np.zeros((3, 6), np.float32))
    assert fractions.shape == (3, 2)
    assert latent.shape == (3, 2)
    assert log_probability.shape == (3,)
    assert values.shape == (3,)
    assert np.all((fractions >= 0) & (fractions <= 1))

    deterministic_a = agent.act(np.ones((3, 6), np.float32), deterministic=True)[0]
    deterministic_b = agent.act(np.ones((3, 6), np.float32), deterministic=True)[0]
    np.testing.assert_array_equal(deterministic_a, deterministic_b)


def test_decentralized_actor_inference_accepts_only_one_local_observation():
    agent = MAPPO(3, 6, config=tiny_config())
    local_observation = np.arange(6, dtype=np.float32)
    action = agent.act_local(1, local_observation, deterministic=True)

    assert action.shape == (2,)
    assert np.all((action >= 0) & (action <= 1))
    with pytest.raises(ValueError, match="local_observation"):
        agent.act_local(1, np.zeros((3, 6), np.float32))


def test_clipped_ppo_objective_and_policy_ratio():
    loss, ratios = clipped_surrogate_loss(
        torch.tensor([np.log(1.5)]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        clip_epsilon=0.2,
    )
    assert ratios.item() == pytest.approx(1.5)
    assert loss.item() == pytest.approx(-1.2)

    negative_loss, _ = clipped_surrogate_loss(
        torch.tensor([np.log(1.5)]),
        torch.tensor([0.0]),
        torch.tensor([-1.0]),
        clip_epsilon=0.2,
    )
    assert negative_loss.item() == pytest.approx(1.5)


def test_gae_matches_hand_calculated_terminal_returns():
    advantages, returns = generalized_advantage_estimate(
        rewards=np.array([[1.0], [2.0]], dtype=np.float32),
        values=np.array([[0.5], [0.25]], dtype=np.float32),
        dones=np.array([0.0, 1.0], dtype=np.float32),
        last_values=np.array([0.0], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.8,
    )
    assert advantages[:, 0].tolist() == pytest.approx([1.985, 1.75])
    assert returns[:, 0].tolist() == pytest.approx([2.485, 2.0])


def test_rollout_storage_preserves_time_agent_dimensions():
    buffer = RolloutBuffer(2, 3, 6, 2)
    for _ in range(2):
        buffer.add(
            np.zeros((3, 6)),
            np.zeros((3, 2)),
            np.zeros(3),
            np.ones(3),
            np.zeros(3),
            False,
        )
    batch = buffer.as_tensors(torch.device("cpu"))
    assert batch["local_observations"].shape == (2, 3, 6)
    assert batch["latent_actions"].shape == (2, 3, 2)
    assert batch["rewards"].shape == (2, 3)
    with pytest.raises(RuntimeError, match="full"):
        buffer.add(
            np.zeros((3, 6)), np.zeros((3, 2)), np.zeros(3),
            np.ones(3), np.zeros(3), False,
        )


def test_ppo_update_changes_actor_and_critic_weights():
    agent = MAPPO(3, 6, config=tiny_config())
    buffer = RolloutBuffer(4, 3, 6, 2)
    before_actor = [parameter.detach().clone() for parameter in agent.policy.actors[0].parameters()]
    before_critic = [parameter.detach().clone() for parameter in agent.policy.critic.parameters()]
    rng = np.random.default_rng(5)
    for step_index in range(4):
        local = rng.normal(size=(3, 6)).astype(np.float32)
        _, latent, log_probs, values = agent.act(local)
        buffer.add(
            local, latent, log_probs,
            rng.normal(size=3).astype(np.float32),
            values, step_index == 3,
        )

    losses = agent.update(buffer)

    assert set(losses) == {"policy_loss", "value_loss", "entropy"}
    assert not buffer.rewards
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_actor, agent.policy.actors[0].parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_critic, agent.policy.critic.parameters())
    )


def test_observation_conversion_and_environment_action_scaling():
    environment = make_environment()
    observations, _ = environment.reset(
        {"A": 2, "B": 3, "C": 4},
        {"A": 4, "B": 5, "C": 6},
        forecast_source="persistence",
    )
    array = observations_to_array(observations, environment.station_ids, OBSERVATION_FIELDS)
    action = fractions_to_environment_action(
        np.full((3, 2), 0.5, dtype=np.float32),
        environment.station_ids,
        environment.station_capacities,
    )
    assert array.shape == (3, len(OBSERVATION_FIELDS))
    assert action["B"] == {"renewable": pytest.approx(6.0), "grid": pytest.approx(6.0)}
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fractions_to_environment_action(
            np.full((3, 2), 1.1), environment.station_ids, environment.station_capacities
        )


def test_forecast_sources_feed_training_without_forecasting_internals():
    episode = scenario()
    persistence = PersistenceForecast()
    assert persistence.forecast(episode.initial_demand) == episode.initial_demand
    model_forecast = ModelForecast(
        lambda current: {station: value + 1.0 for station, value in current.items()}
    )
    forecast_t = model_forecast.forecast(episode.initial_demand)
    actual_revealed_after_t = episode.actual_demand_steps[0]
    forecast_t_plus_1 = model_forecast.forecast(actual_revealed_after_t)
    assert forecast_t["A"] == pytest.approx(4.0)
    assert forecast_t_plus_1["A"] == pytest.approx(5.0)


def test_training_reveals_actual_before_computing_the_following_forecast():
    events = []
    values = (
        {"A": 4.0, "B": 3.0, "C": 7.0},
        {"A": 5.0, "B": 5.0, "C": 8.0},
    )

    class TrackedActuals:
        def __len__(self):
            return len(values)

        def __getitem__(self, index):
            events.append("actual")
            return values[index]

    class TrackedModelForecast:
        source_name = "model"

        def forecast(self, current):
            events.append("forecast")
            return dict(current)

    environment = make_environment()
    agent = MAPPO(3, len(OBSERVATION_FIELDS), config=tiny_config(rollout_length=2))
    original_act = agent.act

    def tracked_act(*args, **kwargs):
        events.append("actor")
        return original_act(*args, **kwargs)

    agent.act = tracked_act
    test_scenario = EpisodeScenario(
        initial_demand={"A": 3.0, "B": 4.0, "C": 5.0},
        actual_demand_steps=TrackedActuals(),
        renewable_available=8.0,
    )
    buffer = RolloutBuffer(2, 3, len(OBSERVATION_FIELDS), 2)

    _collect_training_episode(
        environment, agent, test_scenario, buffer, TrackedModelForecast()
    )

    assert events == [
        "forecast", "actor", "actual",
        "forecast", "actor", "actual",
    ]


def test_deterministic_policy_evaluation_is_repeatable():
    environment = make_environment()
    agent = MAPPO(3, len(OBSERVATION_FIELDS), config=tiny_config())
    source = PersistenceForecast()
    first = evaluate_policy(environment, agent, [scenario()], source)
    second = evaluate_policy(environment, agent, [scenario()], source)
    assert first == second
    required = {
        "total_reward",
        "actual_unserved_demand",
        "renewable_utilization",
        "grid_energy_used",
        "peak_realized_grid_usage",
        "grid_excess_requests",
        "unused_energy",
        "demand_served_percentage",
    }
    assert required <= set(first)


def test_peak_realized_grid_usage_uses_realized_not_allocated_energy():
    environment = EnergyManagementEnv(MARLConfig(
        station_ids=("A", "B", "C"),
        station_capacities={"A": 10.0, "B": 10.0, "C": 10.0},
        renewable_availability=0.0,
        grid_capacity=30.0,
    ))
    test_scenario = EpisodeScenario(
        initial_demand={"A": 3.0, "B": 3.0, "C": 3.0},
        actual_demand_steps=(
            {"A": 1.0, "B": 1.0, "C": 1.0},
            {"A": 1.0, "B": 1.0, "C": 1.0},
        ),
        renewable_available=0.0,
    )

    def request_forecast(observations, _):
        return {
            station_id: {"renewable": 0.0, "grid": observation["predicted_demand"]}
            for station_id, observation in observations.items()
        }

    metrics, _ = _evaluate_action_runner(
        environment,
        test_scenario,
        request_forecast,
        forecast_source="persistence",
        forecast_provider=PersistenceForecast(),
    )

    assert metrics["peak_realized_grid_usage"] == pytest.approx(3.0)


def test_policy_and_baseline_evaluation_use_same_scenarios():
    environment = make_environment()
    agent = MAPPO(3, len(OBSERVATION_FIELDS), config=tiny_config())
    calls = []

    class CountingModelForecast:
        source_name = "model"

        def forecast(self, current):
            calls.append(dict(current))
            return dict(current)

    comparison = compare_policy_to_baseline(
        environment, agent, [scenario()], CountingModelForecast()
    )
    assert set(comparison) == {"mappo", "proportional_baseline"}
    assert set(comparison["mappo"]) == set(comparison["proportional_baseline"])
    assert len(calls) == scenario().horizon


def test_checkpoint_round_trip_preserves_deterministic_actions(tmp_path):
    agent = MAPPO(3, 6, config=tiny_config())
    observations = np.arange(18, dtype=np.float32).reshape(3, 6) / 10
    expected = agent.act(observations, deterministic=True)[0]
    checkpoint_path = tmp_path / "mappo.pt"
    agent.save_checkpoint(
        checkpoint_path,
        training_config={"seed": 7, "number_of_agents": 3},
        normalization_config={"enabled": True},
    )

    restored, checkpoint = MAPPO.load_checkpoint(checkpoint_path)
    actual = restored.act(observations, deterministic=True)[0]
    np.testing.assert_array_equal(expected, actual)
    assert checkpoint["training_config"]["seed"] == 7
    assert checkpoint["normalization_config"]["enabled"]


def test_end_to_end_training_smoke_updates_and_saves_checkpoint(tmp_path):
    environment_config = MARLConfig(
        station_ids=("A", "B", "C"),
        station_capacities={"A": 10.0, "B": 12.0, "C": 15.0},
        renewable_availability=8.0,
        grid_capacity=15.0,
        normalize_observations=True,
    )
    mappo_config = tiny_config(rollout_length=4)
    run_config = TrainingRunConfig(
        seed=17,
        total_episodes=1,
        steps_per_episode=4,
        evaluation_interval=1,
        evaluation_episodes=1,
        output_directory=str(tmp_path),
    )

    agent, recorded = train(
        environment_config,
        mappo_config,
        run_config,
        PersistenceForecast(),
    )

    assert recorded["seed"] == 17
    assert recorded["number_of_agents"] == 3
    assert recorded["observation_dimension"] == 6
    assert recorded["action_dimension"] == 2
    assert (tmp_path / "mappo_final.pt").exists()
    assert (tmp_path / "mappo_episode_000001.pt").exists()
    assert (tmp_path / "actors_final.pt").exists()
    assert (tmp_path / "critic_final.pt").exists()
    assert (tmp_path / "training_config.json").exists()
    restored, _ = MAPPO.load_checkpoint(tmp_path / "mappo_final.pt")
    assert restored.num_agents == agent.num_agents == 3


def test_oracle_forecasts_cannot_be_used_for_training(tmp_path):
    with pytest.raises(ValueError, match="prohibited"):
        train(
            make_environment().config,
            tiny_config(),
            TrainingRunConfig(total_episodes=1, steps_per_episode=2),
            OracleForecast(),
            output_directory=tmp_path,
        )
