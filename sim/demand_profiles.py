"""Historical dataset demand profiles for real-geography simulation runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from session_to_hourly import reindex_continuous, sessions_to_hourly

from .station_provider import DATASET_CONFIG

SESSION_COLUMNS = (
    "source_dataset",
    "entity_id",
    "site",
    "region",
    "connect_time_utc",
    "charge_end_time_utc",
    "energy_kwh",
)


class HistoricalDemandProfile:
    """Resample arrivals and session sizes from one dataset's observed history."""

    def __init__(self, dataset: str, sessions: pd.DataFrame):
        self.dataset = dataset
        config = DATASET_CONFIG[dataset]
        self.timezone = config["timezone"]
        self.entity_id = config["profile_entity_id"]
        self.region = config["region"]
        missing = set(SESSION_COLUMNS) - set(sessions.columns)
        if missing:
            raise ValueError(
                f"Historical profile for {dataset} is missing columns "
                f"{sorted(missing)}."
            )
        rows = sessions.loc[
            (sessions["source_dataset"].astype(str) == dataset)
            & (sessions["entity_id"].astype(str) == self.entity_id),
            SESSION_COLUMNS,
        ].copy()
        if rows.empty:
            raise ValueError(
                f"No historical session rows for {dataset}/{self.entity_id}."
            )
        rows["connect_time_utc"] = pd.to_datetime(
            rows["connect_time_utc"], utc=True, errors="coerce"
        )
        rows["charge_end_time_utc"] = pd.to_datetime(
            rows["charge_end_time_utc"], utc=True, errors="coerce"
        )
        rows["energy_kwh"] = pd.to_numeric(rows["energy_kwh"], errors="coerce")
        duration = (
            rows["charge_end_time_utc"] - rows["connect_time_utc"]
        ).dt.total_seconds() / 3600.0
        valid = (
            rows["connect_time_utc"].notna()
            & rows["charge_end_time_utc"].notna()
            & np.isfinite(rows["energy_kwh"])
            & rows["energy_kwh"].gt(0)
            & np.isfinite(duration)
            & duration.gt(0)
        )
        rows = rows.loc[valid].copy()
        if rows.empty:
            raise ValueError(
                f"Historical profile for {dataset}/{self.entity_id} has no "
                "valid positive-energy sessions."
            )
        rows["_duration_hours"] = duration.loc[valid].to_numpy()
        self.sessions = rows.reset_index(drop=True)
        self._rate_by_slot = self._build_arrival_rates()
        self._sessions_by_slot = self._build_session_slots()

    @classmethod
    def load(cls, dataset: str, session_directory: str | Path = "harmonized_sessions"):
        if dataset not in DATASET_CONFIG:
            raise ValueError(f"Unknown historical profile dataset {dataset!r}.")
        config = DATASET_CONFIG[dataset]
        path = Path(session_directory) / str(config.get("profile_file", ""))
        if not config.get("profile_file") or not path.is_file():
            raise FileNotFoundError(
                f"No historical session profile is configured for {dataset}: {path}."
            )
        sessions = pd.read_csv(path)
        return cls(dataset, sessions)

    def _slot_values(self, timestamps):
        local = timestamps.dt.tz_convert(self.timezone)
        return list(zip(local.dt.weekday >= 5, local.dt.hour))

    def _build_arrival_rates(self):
        starts = self.sessions["connect_time_utc"]
        first_hour = starts.min().floor("h")
        last_hour = starts.max().floor("h")
        hours = pd.date_range(first_hour, last_hour, freq="h", tz="UTC")
        counts = (
            self.sessions.groupby(self.sessions["connect_time_utc"].dt.floor("h"))
            .size()
            .reindex(hours, fill_value=0)
        )
        local = hours.tz_convert(self.timezone)
        table = pd.DataFrame({
            "weekend": local.weekday >= 5,
            "hour": local.hour,
            "count": counts.to_numpy(),
        })
        means = table.groupby(["weekend", "hour"])["count"].mean()
        return {
            (bool(weekend), int(hour)): float(means.get((weekend, hour), 0.0))
            for weekend in (False, True)
            for hour in range(24)
        }

    def _build_session_slots(self):
        slots = {}
        for index, slot in enumerate(
            self._slot_values(self.sessions["connect_time_utc"])
        ):
            slots.setdefault(slot, []).append(index)
        all_indices = np.arange(len(self.sessions), dtype=int)
        return slots, all_indices

    def arrival_rate(self, local_hour: int, is_weekend: bool) -> float:
        return self._rate_by_slot[(bool(is_weekend), int(local_hour))]

    def sample_session(self, rng: np.random.Generator, local_hour: int, is_weekend: bool):
        slots, all_indices = self._sessions_by_slot
        candidates = slots.get((bool(is_weekend), int(local_hour)), all_indices)
        row = self.sessions.iloc[int(rng.choice(candidates))]
        return float(row["_duration_hours"]), float(row["energy_kwh"])

    def warmup(self, hours: int = 336):
        """Return this dataset's recent hourly history, starts, and prior state."""
        sessions = self.sessions.loc[:, SESSION_COLUMNS].copy()
        hourly = reindex_continuous(sessions_to_hourly(sessions))
        if hourly.empty:
            raise ValueError(f"Historical profile {self.dataset} has no hourly demand.")
        end = pd.to_datetime(hourly["hour"], utc=True).max()
        start = end - pd.Timedelta(hours=hours - 1)
        warm_hourly = hourly.loc[
            pd.to_datetime(hourly["hour"], utc=True).between(start, end)
        ].copy()
        warm_sessions = sessions.loc[
            pd.to_datetime(sessions["connect_time_utc"], utc=True).between(
                start, end, inclusive="both"
            )
        ].copy()
        prior = {
            self.entity_id: [
                float(hourly.loc[hourly["hour"] <= end, "kwh_allocated"].sum()),
                int((hourly["hour"] <= end).sum()),
            ]
        }
        return warm_hourly, warm_sessions, prior
