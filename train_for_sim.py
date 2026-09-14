"""Train the existing hurdle model and persist everything the simulation needs.

Reproduces hurdle_model_cv.ipynb's final protocol (cells 14-23) as a script:
global 85% development / 15% untouched test by hour, entity statistics from
development rows only, XGBClassifier x XGBRegressor(reg:tweedie, p=1.3),
n_estimators=500 with no early stopping.

The simulation runs against the model exactly as the notebook defines it.

Data source: harmonized_sessions/ if present, otherwise synthetic sessions.
Run:  python3 train_for_sim.py [--days 300]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

import pipeline as P
from session_to_hourly import sessions_to_hourly, reindex_continuous, validate_split
from sim import synth

ART = Path("artifacts")
WARMUP_HOURS = 336  # 2 weeks: enough for lag_168h and roll_mean_168h at t=0


def load_sessions(days):
    real = Path("harmonized_sessions")
    if real.is_dir():
        files = sorted(real.glob("*.csv"))
        if files:
            print(f"Real corpus: {len(files)} files in harmonized_sessions/")
            df = pd.concat(
                [pd.read_csv(f, parse_dates=["connect_time_utc", "charge_end_time_utc"])
                 for f in files],
                ignore_index=True,
            )
            for c in ("connect_time_utc", "charge_end_time_utc"):
                df[c] = pd.to_datetime(df[c], utc=True)
            return df, "real"
    print("No harmonized_sessions/ - generating synthetic corpus")
    print("  (metrics below demonstrate the pipeline, they are not results)")
    return synth.generate(days=days), "synthetic"


def wape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return np.abs(y - p).sum() / np.abs(y).sum() * 100


def main(days):
    ART.mkdir(exist_ok=True)
    sessions, source_kind = load_sessions(days)
    print(f"Sessions: {len(sessions):,} across {sessions['entity_id'].nunique()} entities")

    print("\nAllocating session energy to hours...")
    hourly = sessions_to_hourly(sessions)
    validate_split(sessions, hourly)
    hourly = reindex_continuous(hourly)
    print(f"Hourly rows: {len(hourly):,}")

    print("\nBuilding features...")
    data = P.build_feature_table(hourly, sessions)
    before = len(data)
    data = data.dropna().reset_index(drop=True)
    print(f"Feature rows: {before:,} -> {len(data):,} after dropna "
          f"({before - len(data):,} warm-up rows removed)")

    # Final protocol from cells 14-23: global 85/15 split by hour.
    data = data.sort_values("hour").reset_index(drop=True)
    cut = int(len(data) * 0.85)
    dev, test = data.iloc[:cut].copy(), data.iloc[cut:].copy()
    print(f"Development: {len(dev):,}   Untouched test: {len(test):,}")

    stats = P.entity_stats_from(dev)
    global_mean = dev[P.TARGET].mean()
    dev = P.attach_entity_stats(dev, stats, global_mean)
    test = P.attach_entity_stats(test, stats, global_mean)

    X_dev = P.design_matrix(dev)
    columns = list(X_dev.columns)
    X_test = P.design_matrix(test, columns=columns)
    y_dev, y_test = dev[P.TARGET], test[P.TARGET]
    print(f"Design matrix: {X_dev.shape[1]} features")

    common = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)

    print("\nTraining classifier...")
    clf = XGBClassifier(eval_metric="logloss", **common)
    clf.fit(X_dev, (y_dev > 0).astype(int), verbose=False)

    print("Training Tweedie regressor on positive rows...")
    pos = (y_dev > 0).values
    reg = XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.3, **common)
    reg.fit(X_dev[pos], y_dev[pos], verbose=False)

    pred, prob = P.hurdle_predict(clf, reg, X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "wape": float(wape(y_test, pred)),
        # Persistence baseline, for comparison against the model.
        "persistence_mae": float(mean_absolute_error(y_test, test["lag_1h"])),
        "persistence_rmse": float(np.sqrt(mean_squared_error(y_test, test["lag_1h"]))),
    }
    print("\nUntouched test:")
    print(f"  MAE  {metrics['mae']:.4f}   (persistence {metrics['persistence_mae']:.4f})")
    print(f"  RMSE {metrics['rmse']:.4f}   (persistence {metrics['persistence_rmse']:.4f})")
    print(f"  R2   {metrics['r2']:.4f}")
    print(f"  WAPE {metrics['wape']:.2f}%")
    if metrics["mae"] > metrics["persistence_mae"]:
        print("  NOTE: the model does NOT beat persistence on MAE here.")

    clf.save_model(ART / "clf.ubj")
    reg.save_model(ART / "reg.ubj")
    (ART / "columns.json").write_text(json.dumps(columns, indent=2))
    stats.to_csv(ART / "entity_stats.csv", index=False)

    # Warm-up state for the live ring buffer: the tail of each French site's
    # hourly series plus the sessions inside that window.
    paris = [e for e in data["entity_id"].unique() if str(e).startswith("paris_")] or \
            list(data["entity_id"].unique()[:1])
    # End the warm-up at 07:00 UTC so the simulation opens at 08:00 UTC
    # (09:00 Paris) - the morning peak. Starting at midnight is correct but
    # shows an empty map for the first minute of wall time.
    ph = hourly[hourly["entity_id"].isin(paris)]
    cut = ph.loc[ph["hour"].dt.hour == 7, "hour"].max()
    if pd.isna(cut):
        cut = ph["hour"].max()
    tail_start = cut - pd.Timedelta(hours=WARMUP_HOURS)
    wu = ph[(ph["hour"] > tail_start) & (ph["hour"] <= cut)].copy()
    ws = sessions[(sessions["entity_id"].isin(paris)) &
                  (sessions["connect_time_utc"] > tail_start) &
                  (sessions["connect_time_utc"] <= cut)].copy()
    wu.to_parquet(ART / "warmup_hourly.parquet", index=False)
    ws.to_parquet(ART / "warmup_sessions.parquet", index=False)

    # Running (sum, count) of every hourly observation per site. The live ring
    # buffer holds only a bounded window and cannot reconstruct expanding_mean
    # without this - see test_parity.py.
    exp_state = {
        str(e): [float(g["kwh_allocated"].sum()), int(len(g))]
        for e, g in ph[ph["hour"] <= cut].groupby("entity_id")
    }

    (ART / "meta.json").write_text(json.dumps({
        "data_source": source_kind,
        "expanding_state": exp_state,
        "n_sessions": int(len(sessions)),
        "n_feature_rows": int(len(data)),
        "n_features": len(columns),
        "entities": sorted(map(str, data["entity_id"].unique())),
        "paris_entities": sorted(map(str, paris)),
        "global_mean": float(global_mean),
        "hourly_end": str(hourly["hour"].max()),
        "sim_start": str(cut + pd.Timedelta(hours=1)),
        "metrics": metrics,
    }, indent=2))

    print(f"\nArtifacts written to {ART}/  (data_source={source_kind})")
    print(f"  warm-up: {len(wu):,} hourly rows, {len(ws):,} sessions across {len(paris)} sites")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=300)
    main(ap.parse_args().days)
