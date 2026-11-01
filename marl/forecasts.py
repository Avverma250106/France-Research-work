"""Forecast-source adapters independent of the forecasting model implementation."""

from collections.abc import Callable, Mapping
from typing import Protocol


class ForecastSource(Protocol):
    def forecast(self, current_demand: Mapping[str, float]) -> Mapping[str, float]:
        """Return one next-hour demand forecast for each station."""


class PersistenceForecast:
    """Predict next-hour demand as the latest observed station demand."""

    source_name = "persistence"

    def forecast(self, current_demand: Mapping[str, float]) -> Mapping[str, float]:
        return dict(current_demand)


class ModelForecast:
    """Adapt any callable forecasting service to the station forecast protocol."""

    source_name = "model"

    def __init__(self, predictor: Callable[[Mapping[str, float]], Mapping[str, float]]):
        self.predictor = predictor

    def forecast(self, current_demand: Mapping[str, float]) -> Mapping[str, float]:
        return self.predictor(current_demand)


class OracleForecast:
    """Oracle upper-bound adapter for evaluation-only future actual demand."""

    source_name = "oracle"

    def forecast(self, actual_future_demand: Mapping[str, float]) -> Mapping[str, float]:
        return dict(actual_future_demand)
