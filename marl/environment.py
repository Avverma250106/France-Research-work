"""Numerical multi-agent environment for station-level energy allocation.

One environment timestep is one hour. Every demand, capacity, allocation, and
resource quantity is energy in kWh per timestep.
"""

from collections.abc import Callable, Mapping
from math import isfinite

from .config import MARLConfig

OBSERVATION_FIELDS = (
    "predicted_demand",
    "current_demand",
    "charging_capacity",
    "renewable_available",
    "remaining_grid_capacity",
    "current_grid_load",
)
RESOURCE_FIELDS = ("renewable", "grid")
FORECAST_SOURCES = frozenset({"persistence", "model", "oracle"})


def _validate_station_values(values, station_ids, name):
    if set(values) != set(station_ids):
        missing = sorted(set(station_ids) - set(values))
        extra = sorted(set(values) - set(station_ids))
        raise ValueError(f"{name} station IDs mismatch; missing={missing}, extra={extra}.")
    result = {}
    for station_id in station_ids:
        value = float(values[station_id])
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name}[{station_id!r}] must be finite and non-negative.")
        result[station_id] = value
    return result


class EnergyManagementEnv:
    """One agent per station; actions allocate renewable and grid kWh.

    The agent observes a forecast and current state, chooses its action, then
    the environment projects that action before invoking the lazy next-demand
    provider. Oracle forecasts require explicit opt-in in ``MARLConfig``.
    """

    def __init__(self, config: MARLConfig):
        self.config = config
        self.station_ids = config.station_ids
        self.station_capacities = dict(config.station_capacities)
        self._reset = False

    def reset(
        self,
        current_demand: Mapping[str, float],
        forecast: Mapping[str, float],
        *,
        renewable_available: float | None = None,
        grid_capacity: float | None = None,
        current_grid_load: float = 0.0,
        forecast_source: str = "model",
    ):
        self.current_demand = _validate_station_values(
            current_demand, self.station_ids, "current_demand"
        )
        self.renewable_available = self._resource_value(
            self.config.renewable_availability if renewable_available is None
            else renewable_available,
            "renewable_available",
        )
        self.grid_capacity = self._resource_value(
            self.config.grid_capacity if grid_capacity is None else grid_capacity,
            "grid_capacity",
        )
        self.current_grid_load = self._resource_value(current_grid_load, "current_grid_load")
        if self.current_grid_load > self.grid_capacity:
            raise ValueError("current_grid_load cannot exceed grid_capacity.")
        self.time_step = 0
        self._reset = True
        observations = self.observe(forecast, forecast_source=forecast_source)
        return observations, {"time_step": 0, "forecast_source": forecast_source}

    @staticmethod
    def _resource_value(value, name):
        value = float(value)
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return value

    def _validate_forecast_source(self, forecast_source):
        if forecast_source not in FORECAST_SOURCES:
            raise ValueError(
                f"Unsupported forecast_source={forecast_source!r}; "
                f"expected one of {sorted(FORECAST_SOURCES)}."
            )
        if forecast_source == "oracle" and not self.config.allow_oracle_forecast:
            raise ValueError(
                "Oracle forecasts are disabled; enable allow_oracle_forecast only "
                "for an explicit evaluation upper-bound experiment."
            )
        return forecast_source

    @property
    def remaining_grid_capacity(self):
        return max(self.grid_capacity - self.current_grid_load, 0.0)

    def observe(self, forecast: Mapping[str, float], *, forecast_source: str | None = None):
        if not self._reset:
            raise RuntimeError("Call reset before requesting observations.")
        source = self._validate_forecast_source(
            forecast_source or self._current_forecast_source
        )
        predictions = _validate_station_values(forecast, self.station_ids, "forecast")
        self._current_forecast = predictions
        self._current_forecast_source = source
        observations = {}
        for station_id in self.station_ids:
            capacity = self.station_capacities[station_id]
            raw = {
                "predicted_demand": predictions[station_id],
                "current_demand": self.current_demand[station_id],
                "charging_capacity": capacity,
                "renewable_available": self.renewable_available,
                "remaining_grid_capacity": self.remaining_grid_capacity,
                "current_grid_load": self.current_grid_load,
            }
            if self.config.normalize_observations:
                renewable_scale = max(self.renewable_available, 1.0)
                grid_scale = max(self.grid_capacity, 1.0)
                raw["predicted_demand"] /= max(capacity, 1.0)
                raw["current_demand"] /= max(capacity, 1.0)
                raw["charging_capacity"] = 1.0 if capacity > 0 else 0.0
                raw["renewable_available"] /= renewable_scale
                raw["remaining_grid_capacity"] /= grid_scale
                raw["current_grid_load"] /= grid_scale
            observations[station_id] = raw
        return observations

    def step(
        self,
        action: Mapping[str, Mapping[str, float]],
        forecast: Mapping[str, float],
        actual_demand_next: Callable[[], Mapping[str, float]],
        *,
        forecast_source: str | None = None,
        renewable_available_next: float | None = None,
        current_grid_load_next: float = 0.0,
    ):
        if not self._reset:
            raise RuntimeError("Call reset before taking a step.")
        if self._current_forecast is None:
            raise RuntimeError("Call observe(forecast) before taking an action.")
        if not callable(actual_demand_next):
            raise TypeError("actual_demand_next must be a callable invoked after action projection.")
        source = self._validate_forecast_source(
            forecast_source or self._current_forecast_source
        )
        predictions = _validate_station_values(forecast, self.station_ids, "forecast")
        if predictions != self._current_forecast:
            raise ValueError("The step forecast must match the forecast used to build observations.")
        if source != self._current_forecast_source:
            raise ValueError("The step forecast_source must match the observed forecast source.")

        requested = self._validate_actions(action)
        (
            projected,
            station_violations,
            forecast_demand_violations,
        ) = self._project_station_limits(requested, predictions)

        requested_renewable_by_station = {
            station_id: allocation["renewable"]
            for station_id, allocation in projected.items()
        }
        requested_renewable = sum(requested_renewable_by_station.values())
        renewable_excess_request = max(
            requested_renewable - self.renewable_available, 0.0
        )
        if requested_renewable > self.renewable_available and requested_renewable > 0:
            factor = self.renewable_available / requested_renewable
            for allocation in projected.values():
                allocation["renewable"] *= factor

        requested_grid_by_station = {
            station_id: allocation["grid"]
            for station_id, allocation in projected.items()
        }
        requested_grid = sum(requested_grid_by_station.values())
        remaining_grid = self.remaining_grid_capacity
        grid_excess_request = max(requested_grid - remaining_grid, 0.0)
        if requested_grid > remaining_grid and requested_grid > 0:
            factor = remaining_grid / requested_grid
            for allocation in projected.values():
                allocation["grid"] *= factor

        allocations = {
            station_id: {
                "renewable": projected[station_id]["renewable"],
                "grid": projected[station_id]["grid"],
                "allocated_energy": (
                    projected[station_id]["renewable"] + projected[station_id]["grid"]
                ),
            }
            for station_id in self.station_ids
        }
        for allocation in allocations.values():
            allocation["delivered_energy"] = allocation["allocated_energy"]

        forecast_unserved_by_station = {
            station_id: max(
                predictions[station_id] - allocations[station_id]["allocated_energy"], 0.0
            )
            for station_id in self.station_ids
        }
        # This is the causal boundary: neither the action nor its projection can
        # depend on next-hour actual demand.
        self._last_projected_action = allocations
        actuals = _validate_station_values(
            actual_demand_next(), self.station_ids, "actual_demand_next"
        )
        actual_energy_served = {
            station_id: min(
                allocations[station_id]["allocated_energy"], actuals[station_id]
            )
            for station_id in self.station_ids
        }
        actual_unserved_by_station = {
            station_id: max(
                actuals[station_id] - allocations[station_id]["allocated_energy"], 0.0
            )
            for station_id in self.station_ids
        }
        unused_by_station = {
            station_id: max(
                allocations[station_id]["allocated_energy"] - actuals[station_id], 0.0
            )
            for station_id in self.station_ids
        }
        renewable_used_by_station = {
            station_id: min(allocations[station_id]["renewable"], actuals[station_id])
            for station_id in self.station_ids
        }
        grid_used_by_station = {
            station_id: min(
                allocations[station_id]["grid"],
                max(actuals[station_id] - allocations[station_id]["renewable"], 0.0),
            )
            for station_id in self.station_ids
        }

        next_renewable = self.renewable_available if renewable_available_next is None else (
            self._resource_value(renewable_available_next, "renewable_available_next")
        )
        next_grid_load = self._resource_value(current_grid_load_next, "current_grid_load_next")
        if next_grid_load > self.grid_capacity:
            raise ValueError("current_grid_load_next cannot exceed grid_capacity.")

        rewards, reward_components = self._rewards(
            allocations,
            actual_unserved_by_station,
            unused_by_station,
            renewable_used_by_station,
            grid_excess_request,
            requested_grid_by_station,
        )
        renewable_allocated_total = sum(a["renewable"] for a in allocations.values())
        grid_allocated_total = sum(a["grid"] for a in allocations.values())
        allocated_total = sum(a["allocated_energy"] for a in allocations.values())
        actual_served_total = sum(actual_energy_served.values())
        actual_unserved_total = sum(actual_unserved_by_station.values())
        unused_total = sum(unused_by_station.values())
        info = {
            "time_step": self.time_step + 1,
            "forecast_source": source,
            "allocations": allocations,
            "forecast_demand": dict(predictions),
            "actual_demand_next": dict(actuals),
            "forecast_unserved": sum(forecast_unserved_by_station.values()),
            "forecast_unserved_by_station": forecast_unserved_by_station,
            "actual_energy_served": actual_served_total,
            "actual_energy_served_by_station": actual_energy_served,
            "actual_unserved": actual_unserved_total,
            "actual_unserved_by_station": actual_unserved_by_station,
            "unused_or_curtailed_energy": unused_total,
            "unused_or_curtailed_by_station": unused_by_station,
            "allocated_energy": allocated_total,
            "allocated_energy_by_station": {
                station_id: allocation["allocated_energy"]
                for station_id, allocation in allocations.items()
            },
            "renewable_allocated": renewable_allocated_total,
            "grid_allocated": grid_allocated_total,
            "renewable_used": sum(renewable_used_by_station.values()),
            "grid_used": sum(grid_used_by_station.values()),
            "served_demand": actual_served_total,
            "unserved_demand": actual_unserved_total,
            "unused_energy": unused_total,
            "renewable_available": self.renewable_available,
            "total_grid_capacity": self.grid_capacity,
            "current_grid_load": self.current_grid_load,
            "remaining_grid_capacity": remaining_grid,
            "grid_load_after": self.current_grid_load + grid_allocated_total,
            "grid_excess_request": grid_excess_request,
            "renewable_excess_request": renewable_excess_request,
            "grid_violation": grid_excess_request,
            "renewable_violation": renewable_excess_request,
            "station_capacity_violation": station_violations,
            "forecast_demand_excess_request": forecast_demand_violations,
            "constraint_violations": {
                "grid_excess_request": grid_excess_request,
                "renewable_excess_request": renewable_excess_request,
                "station_capacity": station_violations,
                "forecast_demand_excess_request": forecast_demand_violations,
            },
            "requested_allocations": requested,
            "reward_components": reward_components,
            "global_reward": sum(rewards.values()),
        }

        self.current_demand = actuals
        self.renewable_available = next_renewable
        self.current_grid_load = next_grid_load
        self.time_step = info["time_step"]
        self._current_forecast = None
        return rewards, False, info

    def _validate_actions(self, action):
        if set(action) != set(self.station_ids):
            missing = sorted(set(self.station_ids) - set(action))
            extra = sorted(set(action) - set(self.station_ids))
            raise ValueError(f"Action station IDs mismatch; missing={missing}, extra={extra}.")
        requested = {}
        for station_id in self.station_ids:
            station_action = action[station_id]
            if set(station_action) != set(RESOURCE_FIELDS):
                raise ValueError(
                    f"Action for {station_id} must contain exactly {RESOURCE_FIELDS}."
                )
            requested[station_id] = {}
            for resource in RESOURCE_FIELDS:
                value = float(station_action[resource])
                if not isfinite(value) or value < 0:
                    raise ValueError(
                        f"Action {resource} for {station_id} must be finite and non-negative."
                    )
                requested[station_id][resource] = value
        return requested

    def _project_station_limits(self, requested, predictions):
        projected = {}
        capacity_violations = {}
        forecast_demand_violations = {}
        for station_id in self.station_ids:
            action = requested[station_id]
            local_limit = min(predictions[station_id], self.station_capacities[station_id])
            renewable = min(action["renewable"], local_limit)
            grid = min(action["grid"], local_limit)
            combined = renewable + grid
            station_violation = max(combined - local_limit, 0.0)
            if station_violation > 0:
                factor = local_limit / combined if combined else 0.0
                renewable *= factor
                grid *= factor
            projected[station_id] = {"renewable": renewable, "grid": grid}
            total_requested = action["renewable"] + action["grid"]
            capacity_violations[station_id] = max(
                total_requested - self.station_capacities[station_id], 0.0
            )
            forecast_demand_violations[station_id] = max(
                total_requested - predictions[station_id], 0.0
            )
        return projected, capacity_violations, forecast_demand_violations

    def _rewards(
        self,
        allocations,
        actual_unserved,
        unused,
        renewable_used,
        grid_excess_request,
        requested_grid,
    ):
        weights = self.config.reward
        total_requested_grid = sum(requested_grid.values())
        rewards = {}
        components = {}
        for station_id, allocation in allocations.items():
            excess_share = (
                grid_excess_request * requested_grid[station_id] / total_requested_grid
                if total_requested_grid > 0
                else 0.0
            )
            item = {
                "renewable_reward": weights.renewable_usage_reward * renewable_used[station_id],
                "actual_unserved_penalty": (
                    weights.unserved_demand_penalty * actual_unserved[station_id]
                ),
                "grid_cost": weights.grid_energy_cost * allocation["grid"],
                "grid_excess_penalty": weights.grid_excess_request_penalty * excess_share,
                "unused_energy_penalty": weights.unused_energy_penalty * unused[station_id],
            }
            components[station_id] = item
            rewards[station_id] = (
                item["renewable_reward"]
                - item["actual_unserved_penalty"]
                - item["grid_cost"]
                - item["grid_excess_penalty"]
                - item["unused_energy_penalty"]
            )
        return rewards, components
