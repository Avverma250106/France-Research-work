"""Runtime bridge from simulation station frames to the saved six-agent MAPPO."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from marl.config import (
    GRID_CAPACITY_KWH,
    MAX_STATIONS,
    RENEWABLE_AVAILABILITY_KWH,
    MARLConfig,
)
from marl.environment import OBSERVATION_FIELDS, EnergyManagementEnv
from marl.mappo import MAPPO, fractions_to_environment_action, observations_to_array


class MAPPOController:
    """Apply the existing fixed-size checkpoint to the selected station order."""

    def __init__(
        self,
        stations: Sequence[Mapping[str, object]],
        checkpoint_path: str | Path = "artifacts/marl/mappo_final.pt",
    ):
        self.stations = tuple(stations)
        self.station_ids = tuple(str(station["id"]) for station in stations)
        if len(self.station_ids) != MAX_STATIONS or len(set(self.station_ids)) != MAX_STATIONS:
            raise ValueError(
                f"The existing MAPPO checkpoint requires exactly {MAX_STATIONS} "
                "unique simulation stations."
            )
        self.agent, checkpoint = MAPPO.load_checkpoint(
            checkpoint_path,
            device="cpu",
            num_agents=MAX_STATIONS,
        )
        if (
            self.agent.observation_size != len(OBSERVATION_FIELDS)
            or self.agent.action_size != 2
        ):
            raise ValueError(
                "The saved MAPPO checkpoint does not match the existing "
                "six-field observation and two-resource action."
            )
        self.capacities = {
            station_id: self._capacity_kwh(station)
            for station_id, station in zip(self.station_ids, self.stations)
        }
        self.capacity_sources = {
            station_id: (
                "osm_tagged"
                if station.get("rated_kw") is not None
                else "estimated_default_7.4_kw_per_point"
            )
            for station_id, station in zip(self.station_ids, self.stations)
        }
        training = checkpoint.get("training_config", {})
        normalized = bool(training.get("environment_normalization_enabled", True))
        self.environment = EnergyManagementEnv(
            MARLConfig(
                station_ids=self.station_ids,
                station_capacities=self.capacities,
                renewable_availability=RENEWABLE_AVAILABILITY_KWH,
                grid_capacity=GRID_CAPACITY_KWH,
                normalize_observations=normalized,
            )
        )
        self._initialized = False
        self._pending_forecast = None
        self._pending_action = None
        self.last_step = None

    @staticmethod
    def _capacity_kwh(station: Mapping[str, object]) -> float:
        tagged_kw = station.get("rated_kw")
        per_point_kw = float(tagged_kw) if tagged_kw is not None else 7.4
        point_count = max(1, int(station.get("n_points") or 1))
        return per_point_kw * point_count

    def advance(
        self,
        current_demand: Mapping[str, float],
        forecast: Mapping[str, float],
    ) -> dict:
        """Close the prior hourly allocation, then plan the next hour."""
        if set(current_demand) != set(self.station_ids):
            raise ValueError("Current demand IDs must match the selected station IDs.")
        if set(forecast) != set(self.station_ids):
            raise ValueError("Forecast IDs must match the selected station IDs.")

        if not self._initialized:
            observations, _ = self.environment.reset(
                current_demand,
                forecast,
                renewable_available=RENEWABLE_AVAILABILITY_KWH,
                grid_capacity=GRID_CAPACITY_KWH,
                forecast_source="model",
            )
            self._initialized = True
            completed = None
        else:
            _, _, completed = self.environment.step(
                self._pending_action,
                self._pending_forecast,
                lambda: dict(current_demand),
                forecast_source="model",
            )
            self.last_step = completed
            observations = self.environment.observe(
                forecast, forecast_source="model"
            )

        local_observations = observations_to_array(
            observations, self.station_ids, OBSERVATION_FIELDS
        )
        fractions = np.stack([
            self.agent.act_local(index, local_observations[index], deterministic=True)
            for index in range(MAX_STATIONS)
        ])
        action = fractions_to_environment_action(
            fractions, self.station_ids, self.capacities
        )
        self._pending_forecast = dict(forecast)
        self._pending_action = action
        completed_allocations = completed["allocations"] if completed else {}
        return {
            "station_count": len(self.station_ids),
            "station_ids": list(self.station_ids),
            "agent_ids": list(self.station_ids),
            "forecast_demand": dict(forecast),
            "current_demand": dict(current_demand),
            "capacity_kwh": dict(self.capacities),
            "capacity_source": dict(self.capacity_sources),
            "renewable_availability_kwh": RENEWABLE_AVAILABILITY_KWH,
            "grid_capacity_kwh": GRID_CAPACITY_KWH,
            "next_hour_plan": action,
            "planned_renewable_request": {
                station: allocation["renewable"]
                for station, allocation in action.items()
            },
            "planned_grid_request": {
                station: allocation["grid"] for station, allocation in action.items()
            },
            "renewable_allocation": {
                station: completed_allocations.get(station, {}).get("renewable", 0.0)
                for station in self.station_ids
            },
            "grid_allocation": {
                station: completed_allocations.get(station, {}).get("grid", 0.0)
                for station in self.station_ids
            },
            "renewable_utilization": (
                completed["renewable_used"] / completed["renewable_available"]
                if completed and completed["renewable_available"] > 0
                else None
            ),
            "completed_step": completed,
            "controller": "MAPPO",
        }
