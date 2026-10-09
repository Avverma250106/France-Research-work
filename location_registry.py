from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


REGION_COUNTRY = {
    "California": "United States",
    "Colorado": "United States",
    "France": "France",
    "UK": "United Kingdom",
    "Australia": "Australia",
}


def build_location_registry(session_frames: Iterable[pd.DataFrame] | pd.DataFrame | None):
    """Build a dataset-derived location registry without any fabricated geography.

    The registry is generated from actual sessions in harmonized_sessions and is
    intentionally conservative: if there are no verified coordinates in the data,
    coordinate_resolution is set to 'none' and lat/lon remain null.
    """
    if session_frames is None:
        return []

    if isinstance(session_frames, pd.DataFrame):
        frames = [session_frames]
    else:
        frames = list(session_frames)

    rows = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        df = frame.copy()
        if "source_dataset" not in df.columns:
            continue
        for src, g in df.groupby("source_dataset", sort=True):
            entity_ids = sorted(map(str, g["entity_id"].dropna().astype(str).unique()))
            region = str(g["region"].dropna().iloc[0]) if not g["region"].dropna().empty else "unknown"
            if src == "acn":
                display_name = "ACN"
            else:
                display_name = str(src).replace("_", " ").title()
            start = pd.to_datetime(g["connect_time_utc"], utc=True).min()
            end = pd.to_datetime(g["connect_time_utc"], utc=True).max()
            rec = {
                "location_id": str(src),
                "display_name": display_name,
                "country": REGION_COUNTRY.get(region, "unknown"),
                "region": region,
                "source_dataset": str(src),
                "entity_ids": entity_ids,
                "latitude": None,
                "longitude": None,
                "coordinate_resolution": "none",
                "available_start": start.isoformat() if pd.notna(start) else None,
                "available_end": end.isoformat() if pd.notna(end) else None,
                "row_count": int(len(g)),
                "entity_count": int(len(entity_ids)),
            }
            rows.append(rec)
    return rows


def load_registry_from_directory(directory: str | Path | None = None):
    root = Path(directory) if directory is not None else Path("harmonized_sessions")
    if not root.exists():
        return []
    frames = []
    for path in sorted(root.glob("*.csv")) + sorted(root.glob("*.parquet")):
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            continue
        if {"source_dataset", "entity_id", "connect_time_utc"}.issubset(df.columns):
            df["connect_time_utc"] = pd.to_datetime(df["connect_time_utc"], utc=True)
            frames.append(df)
    return build_location_registry(frames)
