"""Run a deterministic proportional-allocation demo with saved T+1 forecasts."""

import numpy as np
import pandas as pd

from sim import synth
from sim.features_rt import FeatureBuffer
from sim.model_service import ModelService

from marl.baseline import proportional_allocation
from marl.config import DEMO_STATIONS, MARLConfig, RewardConfig
from marl.environment import EnergyManagementEnv


def _actual_next_hour(rng, entity_id, target_hour):
    actual = 0.0
    target_end = target_hour + pd.Timedelta(hours=1)
    first_start = target_hour - pd.Timedelta(hours=24)
    for start_hour in pd.date_range(first_start, target_hour, freq="h"):
        local = start_hour.tz_convert("Europe/Paris")
        count = rng.poisson(synth.arrival_rate(
            entity_id,
            local.hour,
            local.weekday() >= 5,
            dayofyear=local.dayofyear,
        ))
        for _ in range(count):
            connect = start_hour + pd.Timedelta(minutes=float(rng.uniform(0, 60)))
            duration, energy = synth.sample_session(rng, entity_id)
            end = connect + pd.Timedelta(hours=duration)
            total_minutes = (end - connect).total_seconds() / 60
            overlap_start = max(connect, target_hour)
            overlap_end = min(end, target_end)
            overlap_minutes = max(
                0.0, (overlap_end - overlap_start).total_seconds() / 60
            )
            if overlap_minutes:
                actual += energy * overlap_minutes / total_minutes
    return actual


def main():
    print("Forecast-driven MARL environment demonstration (no trained MARL policy)")
    model = ModelService()
    entity_ids = [station["entity_id"] for station in DEMO_STATIONS.values()]
    hourly = pd.read_parquet("artifacts/warmup_hourly.parquet")
    sessions = pd.read_parquet("artifacts/warmup_sessions.parquet")
    buffer = FeatureBuffer(entity_ids)
    buffer.warm(
        hourly,
        sessions,
        prior=model.meta.get("expanding_state"),
    )
    rows = buffer.feature_rows(target_mode="forecast").sort_values("entity_id")
    predictions, _ = model.predict(rows)
    by_entity = dict(zip(rows["entity_id"], map(float, predictions)))
    current_by_entity = {
        entity_id: float(buffer.hourly[entity_id][-1][1])
        for entity_id in entity_ids
    }
    target_hour = buffer.last_hour() + pd.Timedelta(hours=1)

    forecast = {
        station_id: by_entity[station["entity_id"]]
        for station_id, station in DEMO_STATIONS.items()
    }
    current = {
        station_id: current_by_entity[station["entity_id"]]
        for station_id, station in DEMO_STATIONS.items()
    }
    capacities = {
        station_id: station["capacity"]
        for station_id, station in DEMO_STATIONS.items()
    }
    config = MARLConfig(
        station_ids=tuple(DEMO_STATIONS),
        station_capacities=capacities,
        renewable_availability=100.0,
        grid_capacity=350.0,
        reward=RewardConfig(),
    )
    env = EnergyManagementEnv(config)
    current_grid_load = 25.0
    env.reset(
        current,
        forecast,
        current_grid_load=current_grid_load,
        forecast_source="model",
    )
    action = proportional_allocation(
        forecast,
        station_capacities=capacities,
        renewable_available=config.renewable_availability,
        grid_capacity=config.grid_capacity,
        current_grid_load=current_grid_load,
    )

    rng = np.random.default_rng(2026)
    def actual_next():
        actual_by_entity = {
            entity_id: _actual_next_hour(rng, entity_id, target_hour)
            for entity_id in entity_ids
        }
        return {
            station_id: actual_by_entity[station["entity_id"]]
            for station_id, station in DEMO_STATIONS.items()
        }

    rewards, _, info = env.step(
        action,
        forecast,
        actual_next,
        forecast_source="model",
    )

    print(f"Forecast target: {target_hour.isoformat()}")
    print("\nStation energy (kWh/timestep):")
    print(
        f"  {'Station':14s} {'Forecast':>10s} {'Actual':>10s} {'Allocated':>10s} "
        f"{'Served':>10s} {'Unserved':>10s} {'Unused':>10s} {'Renewable':>10s} {'Grid':>10s}"
    )
    for station_id in DEMO_STATIONS:
        allocation = info["allocations"][station_id]
        print(
            f"  {station_id:14s} {forecast[station_id]:10.2f} "
            f"{info['actual_demand_next'][station_id]:10.2f} "
            f"{allocation['allocated_energy']:10.2f} "
            f"{info['actual_energy_served_by_station'][station_id]:10.2f} "
            f"{info['actual_unserved_by_station'][station_id]:10.2f} "
            f"{info['unused_or_curtailed_by_station'][station_id]:10.2f} "
            f"{allocation['renewable']:10.2f} {allocation['grid']:10.2f}"
        )
    print(f"\nTotal forecast demand: {sum(forecast.values()):.2f} kWh/timestep")
    print(f"Total actual demand: {sum(info['actual_demand_next'].values()):.2f} kWh/timestep")
    print(f"Total allocated: {info['allocated_energy']:.2f} kWh/timestep")
    print(f"Total actual served: {info['actual_energy_served']:.2f} kWh/timestep")
    print(f"Total actual unserved: {info['actual_unserved']:.2f} kWh/timestep")
    print(f"Total unused energy: {info['unused_or_curtailed_energy']:.2f} kWh/timestep")
    print(f"Total renewable: {info['renewable_allocated']:.2f} kWh/timestep")
    print(f"Total grid: {info['grid_allocated']:.2f} kWh/timestep")
    print(f"Total reward: {info['global_reward']:.2f}")


if __name__ == "__main__":
    main()
