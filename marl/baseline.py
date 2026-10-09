"""Deterministic proportional renewable-first energy allocation."""

from collections.abc import Mapping
from math import isfinite


def _proportional_allocate(demands, available, limits):
    allocation = {station_id: 0.0 for station_id in demands}
    remaining = min(float(available), sum(min(demands[s], limits[s]) for s in demands))
    open_stations = {
        station_id: {
            "weight": demands[station_id],
            "capacity": min(demands[station_id], limits[station_id]),
        }
        for station_id in demands
        if min(demands[station_id], limits[station_id]) > 0
    }
    while remaining > 1e-12 and open_stations:
        weight_total = sum(item["weight"] for item in open_stations.values())
        if weight_total <= 0:
            break
        proposed = {
            station_id: remaining * item["weight"] / weight_total
            for station_id, item in open_stations.items()
        }
        saturated = [
            station_id for station_id, proposal in proposed.items()
            if proposal >= open_stations[station_id]["capacity"] - 1e-12
        ]
        if not saturated:
            for station_id, proposal in proposed.items():
                allocation[station_id] += proposal
            remaining = 0.0
            break
        used = 0.0
        for station_id in saturated:
            amount = open_stations[station_id]["capacity"]
            allocation[station_id] += amount
            used += amount
        remaining -= used
        for station_id in saturated:
            open_stations.pop(station_id)
    return allocation


def proportional_allocation(
    demands: Mapping[str, float],
    *,
    station_capacities: Mapping[str, float],
    renewable_available: float,
    grid_capacity: float,
    current_grid_load: float,
):
    """Allocate forecast demand proportionally, renewables first, then grid.

    Returns the environment action shape: ``station_id -> {renewable, grid}``.
    All energy limits, including current_grid_load, are kWh per one-hour step.
    """
    if not demands:
        raise ValueError("demands must contain at least one station.")
    if set(demands) != set(station_capacities):
        raise ValueError("demands and station_capacities must contain the same station IDs.")

    clean_demands = {}
    clean_capacities = {}
    for station_id in demands:
        demand = float(demands[station_id])
        capacity = float(station_capacities[station_id])
        if not isfinite(demand) or demand < 0:
            raise ValueError(f"Demand for {station_id} must be finite and non-negative.")
        if not isfinite(capacity) or capacity < 0:
            raise ValueError(f"Capacity for {station_id} must be finite and non-negative.")
        clean_demands[station_id] = demand
        clean_capacities[station_id] = capacity
    renewable_available = float(renewable_available)
    grid_capacity = float(grid_capacity)
    current_grid_load = float(current_grid_load)
    if not isfinite(renewable_available) or renewable_available < 0:
        raise ValueError("renewable_available must be finite and non-negative.")
    if not isfinite(grid_capacity) or grid_capacity < 0:
        raise ValueError("grid_capacity must be finite and non-negative.")
    if not isfinite(current_grid_load) or current_grid_load < 0:
        raise ValueError("current_grid_load must be finite and non-negative.")
    if current_grid_load > grid_capacity:
        raise ValueError("current_grid_load cannot exceed grid_capacity.")

    renewable = _proportional_allocate(
        clean_demands, renewable_available, clean_capacities
    )
    remaining = {
        station_id: max(
            min(clean_demands[station_id], clean_capacities[station_id])
            - renewable[station_id],
            0.0,
        )
        for station_id in clean_demands
    }
    remaining_capacities = {
        station_id: max(clean_capacities[station_id] - renewable[station_id], 0.0)
        for station_id in clean_demands
    }
    remaining_grid = max(grid_capacity - current_grid_load, 0.0)
    grid = _proportional_allocate(remaining, remaining_grid, remaining_capacities)
    return {
        station_id: {
            "renewable": renewable[station_id],
            "grid": grid[station_id],
        }
        for station_id in clean_demands
    }
