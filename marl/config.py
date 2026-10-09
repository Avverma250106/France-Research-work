"""Configuration for the station-level energy-management environment."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping


MAX_STATIONS = 6
RENEWABLE_AVAILABILITY_KWH = 100.0
GRID_CAPACITY_KWH = 350.0


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights; energy terms are measured in kWh per one-hour step."""

    renewable_usage_reward: float = 0.1
    unserved_demand_penalty: float = 1.0
    grid_energy_cost: float = 0.05
    grid_excess_request_penalty: float = 2.0
    unused_energy_penalty: float = 0.05

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True)
class MARLConfig:
    """Environment limits in kWh per one-hour timestep.

    ``station_capacities`` and ``grid_capacity`` are interval-energy limits,
    not instantaneous kW ratings. ``current_grid_load`` passed to reset is
    energy already consumed within the same one-hour grid budget.
    """

    station_ids: tuple[str, ...]
    station_capacities: Mapping[str, float]
    renewable_availability: float
    grid_capacity: float
    reward: RewardConfig = field(default_factory=RewardConfig)
    normalize_observations: bool = False
    allow_oracle_forecast: bool = False

    def __post_init__(self):
        if not self.station_ids:
            raise ValueError("At least one station is required.")
        if len(set(self.station_ids)) != len(self.station_ids):
            raise ValueError("station_ids must be unique.")
        if set(self.station_capacities) != set(self.station_ids):
            raise ValueError("station_capacities must contain exactly one value per station.")
        for station_id, capacity in self.station_capacities.items():
            if not isfinite(capacity) or capacity < 0:
                raise ValueError(f"Capacity for {station_id} must be non-negative.")
        if not isfinite(self.renewable_availability) or self.renewable_availability < 0:
            raise ValueError("renewable_availability must be non-negative.")
        if not isfinite(self.grid_capacity) or self.grid_capacity < 0:
            raise ValueError("grid_capacity must be non-negative.")


DEMO_STATIONS = {
    "Bercy": {"entity_id": "paris_bercy", "capacity": 120.0},
    "Republique": {"entity_id": "paris_republique", "capacity": 100.0},
    "Montparnasse": {"entity_id": "paris_montparnasse", "capacity": 160.0},
    "La Defense": {"entity_id": "paris_la_defense", "capacity": 90.0},
    "Batignolles": {"entity_id": "paris_batignolles", "capacity": 75.0},
    "Ivry": {"entity_id": "paris_ivry", "capacity": 110.0},
}
