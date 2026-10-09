import pytest

from marl.baseline import proportional_allocation
from marl.config import MARLConfig, RewardConfig
from marl.environment import EnergyManagementEnv
from marl.forecasts import ModelForecast, OracleForecast, PersistenceForecast


def make_env(
    station_ids=("A", "B"),
    *,
    capacities=None,
    renewable=10.0,
    grid=10.0,
    reward=None,
    normalize=False,
    allow_oracle=False,
):
    capacities = capacities or {station_id: 10.0 for station_id in station_ids}
    return EnergyManagementEnv(MARLConfig(
        station_ids=station_ids,
        station_capacities=capacities,
        renewable_availability=renewable,
        grid_capacity=grid,
        reward=reward or RewardConfig(),
        normalize_observations=normalize,
        allow_oracle_forecast=allow_oracle,
    ))


def started_env(
    *,
    station_ids=("A", "B"),
    capacities=None,
    renewable=10.0,
    grid=10.0,
    grid_load=0.0,
    reward=None,
    normalize=False,
    allow_oracle=False,
    forecast=None,
    forecast_source="model",
):
    env = make_env(
        station_ids,
        capacities=capacities,
        renewable=renewable,
        grid=grid,
        reward=reward,
        normalize=normalize,
        allow_oracle=allow_oracle,
    )
    current = {station_id: 2.0 for station_id in station_ids}
    forecast = forecast or {station_id: 4.0 for station_id in station_ids}
    observations, reset_info = env.reset(
        current,
        forecast,
        current_grid_load=grid_load,
        forecast_source=forecast_source,
    )
    return env, current, forecast, observations, reset_info


def step(env, forecast, action, actual, **kwargs):
    return env.step(action, forecast, lambda: actual, **kwargs)


def test_reset_provides_numerical_observation_per_station():
    env, current, forecast, observations, info = started_env()

    assert set(observations) == set(env.station_ids)
    assert observations["A"] == {
        "predicted_demand": forecast["A"],
        "current_demand": current["A"],
        "charging_capacity": 10.0,
        "renewable_available": 10.0,
        "remaining_grid_capacity": 10.0,
        "current_grid_load": 0.0,
    }
    assert info == {"time_step": 0, "forecast_source": "model"}


def test_step_allocates_and_advances_after_actual_revelation():
    env, _, forecast, _, _ = started_env()
    action = {
        "A": {"renewable": 2.0, "grid": 1.0},
        "B": {"renewable": 1.0, "grid": 1.0},
    }

    rewards, done, info = step(env, forecast, action, {"A": 5.0, "B": 6.0})

    assert not done
    assert set(rewards) == {"A", "B"}
    assert info["time_step"] == 1
    assert info["allocated_energy"] == pytest.approx(5.0)
    assert env.current_demand == {"A": 5.0, "B": 6.0}


def test_environment_supports_configurable_station_count():
    _, _, _, observations, _ = started_env(station_ids=("A", "B", "C"))
    assert set(observations) == {"A", "B", "C"}


def test_actions_reject_negative_and_clip_to_forecast_and_station_capacity():
    env, _, forecast, _, _ = started_env()
    with pytest.raises(ValueError, match="non-negative"):
        step(env, forecast, {
            "A": {"renewable": -1, "grid": 0},
            "B": {"renewable": 0, "grid": 0},
        }, {"A": 1, "B": 1})

    env, _, forecast, _, _ = started_env()
    _, _, info = step(env, forecast, {
        "A": {"renewable": 100, "grid": 100},
        "B": {"renewable": 0, "grid": 0},
    }, {"A": 1, "B": 1})
    assert info["allocations"]["A"]["allocated_energy"] == pytest.approx(4.0)
    assert info["station_capacity_violation"]["A"] == pytest.approx(190.0)
    assert info["forecast_demand_excess_request"]["A"] == pytest.approx(196.0)


def test_grid_constraint_projects_proportionally_and_reports_excess_request():
    env, _, forecast, _, _ = started_env(
        renewable=0, grid=6, forecast={"A": 10.0, "B": 10.0}
    )
    _, _, info = step(env, forecast, {
        "A": {"renewable": 0, "grid": 8},
        "B": {"renewable": 0, "grid": 4},
    }, {"A": 1, "B": 1})

    assert info["grid_allocated"] == pytest.approx(6.0)
    assert info["allocations"]["A"]["grid"] == pytest.approx(4.0)
    assert info["allocations"]["B"]["grid"] == pytest.approx(2.0)
    assert info["grid_excess_request"] == pytest.approx(6.0)
    assert info["grid_violation"] == info["grid_excess_request"]
    assert info["grid_load_after"] <= info["total_grid_capacity"]


def test_renewable_constraint_projects_proportionally_and_reports_excess_request():
    env, _, forecast, _, _ = started_env(
        renewable=6, grid=0, forecast={"A": 10.0, "B": 10.0}
    )
    _, _, info = step(env, forecast, {
        "A": {"renewable": 8, "grid": 0},
        "B": {"renewable": 4, "grid": 0},
    }, {"A": 1, "B": 1})

    assert info["renewable_allocated"] == pytest.approx(6.0)
    assert info["allocations"]["A"]["renewable"] == pytest.approx(4.0)
    assert info["allocations"]["B"]["renewable"] == pytest.approx(2.0)
    assert info["renewable_excess_request"] == pytest.approx(6.0)
    assert info["renewable_violation"] == info["renewable_excess_request"]


def test_remaining_grid_budget_includes_current_grid_load():
    env, _, forecast, observations, _ = started_env(
        renewable=0,
        grid=10,
        grid_load=3,
        forecast={"A": 10.0, "B": 10.0},
    )
    assert env.remaining_grid_capacity == pytest.approx(7.0)
    assert observations["A"]["remaining_grid_capacity"] == pytest.approx(7.0)
    _, _, info = step(env, forecast, {
        "A": {"renewable": 0, "grid": 10},
        "B": {"renewable": 0, "grid": 10},
    }, {"A": 10, "B": 10})
    assert info["grid_allocated"] == pytest.approx(7.0)
    assert info["grid_load_after"] == pytest.approx(10.0)


def test_station_capacity_constrains_combined_allocation():
    env, _, forecast, _, _ = started_env(
        capacities={"A": 3.0, "B": 10.0},
        renewable=20,
        grid=20,
        forecast={"A": 10.0, "B": 10.0},
    )
    _, _, info = step(env, forecast, {
        "A": {"renewable": 3.0, "grid": 3.0},
        "B": {"renewable": 0, "grid": 0},
    }, {"A": 1, "B": 1})
    assert info["allocations"]["A"]["allocated_energy"] == pytest.approx(3.0)
    assert info["station_capacity_violation"]["A"] == pytest.approx(3.0)


def test_forecast_underestimation_is_penalized_by_realized_unserved_demand():
    reward = RewardConfig(
        renewable_usage_reward=0,
        unserved_demand_penalty=2,
        grid_energy_cost=0,
        grid_excess_request_penalty=0,
        unused_energy_penalty=0,
    )
    env, _, forecast, _, _ = started_env(
        station_ids=("A",),
        capacities={"A": 200},
        renewable=100,
        grid=0,
        reward=reward,
        forecast={"A": 100},
    )
    rewards, _, info = step(env, forecast, {
        "A": {"renewable": 100, "grid": 0}
    }, {"A": 160})

    assert info["allocations"]["A"]["allocated_energy"] == 100
    assert info["forecast_unserved"] == 0
    assert info["actual_unserved"] == 60
    assert info["reward_components"]["A"]["actual_unserved_penalty"] == 120
    assert rewards["A"] == -120


def test_forecast_overestimation_reports_unused_energy_and_caps_served_at_actual():
    env, _, forecast, _, _ = started_env(
        station_ids=("A",),
        capacities={"A": 200},
        renewable=160,
        grid=0,
        forecast={"A": 160},
    )
    _, _, info = step(env, forecast, {
        "A": {"renewable": 160, "grid": 0}
    }, {"A": 100})

    assert info["actual_energy_served"] == 100
    assert info["actual_unserved"] == 0
    assert info["unused_or_curtailed_energy"] == 60
    assert info["renewable_used"] == 100
    assert info["reward_components"]["A"]["unused_energy_penalty"] == pytest.approx(3.0)


def test_exact_forecast_has_no_actual_shortage_or_unused_energy():
    env, _, forecast, _, _ = started_env(
        station_ids=("A",),
        capacities={"A": 200},
        renewable=100,
        grid=0,
        forecast={"A": 100},
    )
    _, _, info = step(env, forecast, {
        "A": {"renewable": 100, "grid": 0}
    }, {"A": 100})
    assert info["actual_unserved"] == 0
    assert info["unused_or_curtailed_energy"] == 0


def test_reward_uses_configured_realized_components():
    reward = RewardConfig(
        renewable_usage_reward=2.0,
        unserved_demand_penalty=3.0,
        grid_energy_cost=0.5,
        grid_excess_request_penalty=4.0,
        unused_energy_penalty=0.25,
    )
    env, _, forecast, _, _ = started_env(
        station_ids=("A",),
        capacities={"A": 20},
        renewable=4,
        grid=2,
        reward=reward,
        forecast={"A": 10},
    )
    rewards, _, info = step(env, forecast, {
        "A": {"renewable": 4.0, "grid": 4.0}
    }, {"A": 12.0})

    components = info["reward_components"]["A"]
    assert components == {
        "renewable_reward": 8.0,
        "actual_unserved_penalty": 18.0,
        "grid_cost": 1.0,
        "grid_excess_penalty": 8.0,
        "unused_energy_penalty": 0.0,
    }
    assert rewards["A"] == pytest.approx(-19.0)


def test_energy_accounting_is_conserved_and_nonnegative():
    env, _, forecast, _, _ = started_env()
    _, _, info = step(env, forecast, {
        "A": {"renewable": 2.0, "grid": 1.0},
        "B": {"renewable": 1.0, "grid": 1.0},
    }, {"A": 1.0, "B": 6.0})

    assert info["renewable_allocated"] <= info["renewable_available"]
    assert info["grid_allocated"] <= info["remaining_grid_capacity"]
    assert info["allocated_energy"] == pytest.approx(
        info["renewable_allocated"] + info["grid_allocated"]
    )
    assert info["allocated_energy"] == pytest.approx(
        info["actual_energy_served"] + info["unused_or_curtailed_energy"]
    )
    assert sum(info["actual_demand_next"].values()) == pytest.approx(
        info["actual_energy_served"] + info["actual_unserved"]
    )
    assert info["renewable_used"] + info["grid_used"] == pytest.approx(
        info["actual_energy_served"]
    )
    assert all(
        value >= 0
        for allocation in info["allocations"].values()
        for value in allocation.values()
    )


def test_actual_demand_is_not_in_observation_before_action():
    _, _, _, observations, _ = started_env()
    assert all("actual_demand_next" not in observation for observation in observations.values())
    assert all("actual_unserved" not in observation for observation in observations.values())


def test_actual_provider_is_called_only_after_action_projection():
    env, _, forecast, _, _ = started_env(station_ids=("A",), capacities={"A": 10.0})
    action = {"A": {"renewable": 2.0, "grid": 1.0}}
    calls = []

    def actual_provider():
        calls.append("actual")
        assert env._last_projected_action["A"]["allocated_energy"] == pytest.approx(3.0)
        return {"A": 5.0}

    env.step(action, forecast, actual_provider)
    assert calls == ["actual"]


def test_step_requires_callable_future_actual_provider():
    env, _, forecast, _, _ = started_env()
    with pytest.raises(TypeError, match="callable"):
        env.step({
            "A": {"renewable": 1, "grid": 1},
            "B": {"renewable": 1, "grid": 1},
        }, forecast, {"A": 3, "B": 3})


def test_persistence_and_model_forecast_sources_are_explicit():
    assert PersistenceForecast.source_name == "persistence"
    assert ModelForecast.source_name == "model"
    assert PersistenceForecast().forecast({"A": 10.0}) == {"A": 10.0}
    model = ModelForecast(lambda current: {key: value + 5 for key, value in current.items()})
    assert model.forecast({"A": 10.0}) == {"A": 15.0}
    assert OracleForecast().forecast({"A": 20.0}) == {"A": 20.0}
    env, _, forecast, _, _ = started_env(forecast_source="persistence")
    _, _, info = step(env, forecast, {
        "A": {"renewable": 1, "grid": 1},
        "B": {"renewable": 1, "grid": 1},
    }, {"A": 3, "B": 3}, forecast_source="persistence")
    assert info["forecast_source"] == "persistence"


def test_forecast_source_persists_when_observing_next_timestep():
    env, _, forecast, _, _ = started_env(forecast_source="model")
    action = {
        "A": {"renewable": 1, "grid": 1},
        "B": {"renewable": 1, "grid": 1},
    }
    step(env, forecast, action, {"A": 3, "B": 3})
    next_observations = env.observe({"A": 3, "B": 3})
    assert next_observations["A"]["predicted_demand"] == 3
    _, _, info = env.step(action, {"A": 3, "B": 3}, lambda: {"A": 4, "B": 4})
    assert info["forecast_source"] == "model"


def test_oracle_forecast_requires_explicit_upper_bound_opt_in():
    assert OracleForecast.source_name == "oracle"
    env = make_env(station_ids=("A",), capacities={"A": 10})
    with pytest.raises(ValueError, match="Oracle forecasts are disabled"):
        env.reset({"A": 2}, {"A": 3}, forecast_source="oracle")

    env = make_env(
        station_ids=("A",), capacities={"A": 10}, allow_oracle=True
    )
    env.reset({"A": 2}, {"A": 3}, forecast_source="oracle")
    _, _, info = step(
        env, {"A": 3}, {"A": {"renewable": 1, "grid": 1}}, {"A": 3},
        forecast_source="oracle",
    )
    assert info["forecast_source"] == "oracle"


def test_reset_override_values_are_used_for_observation_normalization():
    env = make_env(
        station_ids=("A",),
        capacities={"A": 10},
        renewable=100,
        grid=200,
        normalize=True,
    )
    observations, _ = env.reset(
        {"A": 2},
        {"A": 5},
        renewable_available=8,
        grid_capacity=20,
        current_grid_load=5,
    )
    assert observations["A"]["renewable_available"] == pytest.approx(1.0)
    assert observations["A"]["remaining_grid_capacity"] == pytest.approx(0.75)
    assert observations["A"]["current_grid_load"] == pytest.approx(0.25)


def test_baseline_proportion_is_correct_without_capacity_limits():
    allocation = proportional_allocation(
        {"A": 100.0, "B": 200.0, "C": 300.0},
        station_capacities={"A": 500.0, "B": 500.0, "C": 500.0},
        renewable_available=300.0,
        grid_capacity=0.0,
        current_grid_load=0.0,
    )
    assert {
        station: values["renewable"]
        for station, values in allocation.items()
    } == {
        "A": pytest.approx(50.0),
        "B": pytest.approx(100.0),
        "C": pytest.approx(150.0),
    }


def test_baseline_redistributes_after_one_capacity_binds():
    allocation = proportional_allocation(
        {"A": 100.0, "B": 200.0, "C": 300.0},
        station_capacities={"A": 20.0, "B": 500.0, "C": 500.0},
        renewable_available=300.0,
        grid_capacity=0.0,
        current_grid_load=0.0,
    )
    assert allocation["A"]["renewable"] == pytest.approx(20.0)
    assert allocation["B"]["renewable"] == pytest.approx(112.0)
    assert allocation["C"]["renewable"] == pytest.approx(168.0)


def test_baseline_redistributes_after_multiple_capacities_bind():
    allocation = proportional_allocation(
        {"A": 100.0, "B": 200.0, "C": 300.0},
        station_capacities={"A": 20.0, "B": 50.0, "C": 500.0},
        renewable_available=300.0,
        grid_capacity=0.0,
        current_grid_load=0.0,
    )
    assert allocation["A"]["renewable"] == pytest.approx(20.0)
    assert allocation["B"]["renewable"] == pytest.approx(50.0)
    assert allocation["C"]["renewable"] == pytest.approx(230.0)


def test_baseline_resource_limits_and_remaining_grid_budget():
    allocation = proportional_allocation(
        {"A": 100.0, "B": 200.0, "C": 300.0},
        station_capacities={"A": 500.0, "B": 500.0, "C": 500.0},
        renewable_available=200.0,
        grid_capacity=200.0,
        current_grid_load=100.0,
    )
    assert sum(v["renewable"] for v in allocation.values()) == pytest.approx(200.0)
    assert sum(v["grid"] for v in allocation.values()) == pytest.approx(100.0)


def test_baseline_and_environment_apply_identical_remaining_grid_budget():
    forecast = {"A": 10.0, "B": 20.0, "C": 30.0}
    capacities = {"A": 100.0, "B": 100.0, "C": 100.0}
    current_grid_load = 40.0
    renewable = 15.0
    grid_capacity = 100.0
    baseline = proportional_allocation(
        forecast,
        station_capacities=capacities,
        renewable_available=renewable,
        grid_capacity=grid_capacity,
        current_grid_load=current_grid_load,
    )
    env, _, _, _, _ = started_env(
        station_ids=tuple(forecast),
        capacities=capacities,
        renewable=renewable,
        grid=grid_capacity,
        grid_load=current_grid_load,
        forecast=forecast,
    )
    _, _, info = step(env, forecast, baseline, forecast)

    assert info["renewable_allocated"] == pytest.approx(
        sum(action["renewable"] for action in baseline.values())
    )
    assert info["grid_allocated"] == pytest.approx(
        sum(action["grid"] for action in baseline.values())
    )
    assert info["grid_allocated"] <= grid_capacity - current_grid_load


def test_reward_coefficients_are_nonnegative_and_configurable():
    with pytest.raises(ValueError, match="non-negative"):
        RewardConfig(unused_energy_penalty=-1)
