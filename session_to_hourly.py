"""
Step 2: Convert unified sessions into a non-overlapping hourly demand series,
using time-weighted proportional energy splitting so multi-hour sessions don't
get dumped entirely into their start hour (the "overlap" problem).
"""

import pandas as pd
import numpy as np


def split_session_energy(row, freq="h"):
    """Allocate one session's energy_kwh across every hourly bin its active
    charging window overlaps, weighted by the number of minutes of overlap.

    Returns a list of dicts: {entity, hour, kwh_allocated}
    entity = (source_dataset, station_id) so stations are never mixed.
    """
    start = row["connect_time_utc"]
    end = row["charge_end_time_utc"]
    kwh = row["energy_kwh"]

    total_minutes = (end - start).total_seconds() / 60
    if total_minutes <= 0:
        return []

    bin_start_floor = start.floor(freq)
    bin_end_ceil = end.ceil(freq)
    bins = pd.date_range(bin_start_floor, bin_end_ceil, freq=freq)

    records = []
    bin_width = pd.tseries.frequencies.to_offset(freq)
    for bin_start in bins[:-1]:
        bin_end = bin_start + bin_width
        overlap_start = max(start, bin_start)
        overlap_end = min(end, bin_end)
        overlap_minutes = max(0.0, (overlap_end - overlap_start).total_seconds() / 60)
        if overlap_minutes > 0:
            records.append({
                "source_dataset": row["source_dataset"],
                "entity_id": row["entity_id"],
                "site": row["site"],
                "region": row["region"],
                "hour": bin_start,
                "kwh_allocated": kwh * (overlap_minutes / total_minutes),
            })
    return records


def sessions_to_hourly(sessions: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Apply the proportional split to every session and aggregate to
    (source_dataset, entity_id, hour) totals.

    entity_id is the modeling granularity chosen in harmonize.py: per-station
    for ACN, network-level for ElaadNL (see note there). All energy from
    overlapping sessions on the SAME entity in the SAME hour is summed here;
    since each session's own total is conserved exactly across its own bins
    (validated in validate_split), summing multiple sessions' allocations for
    one entity/hour is correct, not double-counting.
    """
    all_records = []
    for _, row in sessions.iterrows():
        all_records.extend(split_session_energy(row, freq=freq))

    exploded = pd.DataFrame(all_records)
    hourly = (
        exploded.groupby(["source_dataset", "entity_id", "site", "region", "hour"], as_index=False)
        ["kwh_allocated"].sum()
    )
    return hourly


def validate_split(sessions: pd.DataFrame, hourly: pd.DataFrame, tol=1e-6):
    """Sanity check: total energy per entity in the hourly table must equal
    total energy per entity in the original sessions table. Run this before
    trusting anything downstream."""
    original_totals = sessions.groupby("entity_id")["energy_kwh"].sum()
    split_totals = hourly.groupby("entity_id")["kwh_allocated"].sum()
    diff = (original_totals - split_totals).abs()
    bad = diff[diff > tol]
    if len(bad):
        print(f"  WARNING: {len(bad)} entities have energy mismatch after split (max diff {bad.max():.6f} kWh)")
    else:
        print(f"  OK: energy conserved across split for all {len(original_totals)} entities "
              f"(max diff {diff.max():.2e} kWh)")
    return bad


def reindex_continuous(hourly: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Reindex each (source_dataset, entity_id) series to a continuous hourly
    grid spanning its own observed date range, filling true zero-demand hours
    with 0 (not NaN, not dropped)."""
    filled_parts = []
    for (source, entity), group in hourly.groupby(["source_dataset", "entity_id"]):
        full_range = pd.date_range(group["hour"].min(), group["hour"].max(), freq=freq)
        g = group.set_index("hour").reindex(full_range)
        g["kwh_allocated"] = g["kwh_allocated"].fillna(0.0)
        g["source_dataset"] = source
        g["entity_id"] = entity
        g["site"] = group["site"].iloc[0]
        g["region"] = group["region"].iloc[0]
        g.index.name = "hour"
        filled_parts.append(g.reset_index())
    return pd.concat(filled_parts, ignore_index=True)