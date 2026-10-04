"""Train the hurdle model and persist the artifacts used by the live simulator.

The active evaluation protocol is causal, time-ordered, and removes leakage by
using an inner validation split for early stopping before scoring the untouched
outer validation fold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

import pipeline as P
from baselines import evaluate_baselines
from session_to_hourly import reindex_continuous, sessions_to_hourly, validate_split
from sim import synth

ART = Path("artifacts")
WARMUP_HOURS = 336
DEFAULT_TARGET_MODE = "nowcast"


def load_sessions(days):
    real = Path("harmonized_sessions")
    if real.is_dir():
        csv_files = sorted(real.glob("*.csv"))
        parquet_files = sorted(real.glob("*.parquet"))
        files = csv_files + parquet_files
        if files:
            print(f"Real corpus: {len(files)} files in harmonized_sessions/")
            frames = []
            for path in files:
                if path.suffix == ".csv":
                    frame = pd.read_csv(path, parse_dates=["connect_time_utc", "charge_end_time_utc"])
                else:
                    frame = pd.read_parquet(path)
                frames.append(frame)
            df = pd.concat(frames, ignore_index=True)
            for c in ("connect_time_utc", "charge_end_time_utc"):
                df[c] = pd.to_datetime(df[c], utc=True)
            return df, "real"
    print("No harmonized_sessions/ - generating synthetic corpus")
    print("  (metrics below demonstrate the pipeline, they are not results)")
    return synth.generate(days=days), "synthetic"


def _build_model_common():
    return {
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }


def _select_n_estimators(X, y, *, is_classifier, target_mode, n_splits=3, early_stopping_rounds=25):
    base = _build_model_common()
    target = (y > 0).astype(int) if is_classifier else y

    splitter = TimeSeriesSplit(n_splits=n_splits)
    best_iters = []
    for train_idx, val_idx in splitter.split(np.arange(len(X))):
        inner_cut = int(len(train_idx) * 0.8)
        inner_train_idx = train_idx[:inner_cut]
        inner_val_idx = train_idx[inner_cut:]
        if len(inner_val_idx) == 0:
            continue

        if is_classifier:
            params = {
                **base,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            }
            dtrain = xgb.DMatrix(X.iloc[inner_train_idx], label=target.iloc[inner_train_idx])
            dvalid = xgb.DMatrix(X.iloc[inner_val_idx], label=target.iloc[inner_val_idx])
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dvalid, "val")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
        else:
            params = {
                **base,
                "objective": "reg:tweedie",
                "eval_metric": "rmse",
                "tweedie_variance_power": 1.3,
            }
            dtrain = xgb.DMatrix(X.iloc[inner_train_idx], label=target.iloc[inner_train_idx])
            dvalid = xgb.DMatrix(X.iloc[inner_val_idx], label=target.iloc[inner_val_idx])
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dvalid, "val")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
        best_iters.append(int(booster.best_iteration))

    if not best_iters:
        return 64
    return int(round(float(np.mean(best_iters))))


def _build_model_metadata(source_kind, target_mode, metrics, clf_n_estimators, reg_n_estimators, columns,
                          space, global_mean, hourly, data, paris, cut):
    exp_state = {
        str(e): [float(g["kwh_allocated"].sum()), int(len(g))]
        for e, g in hourly[hourly["hour"] <= cut].groupby("entity_id")
    }
    return {
        "data_source": source_kind,
        "target_mode": target_mode,
        "target_type": "nowcast" if target_mode == "nowcast" else "forecast",
        "expanding_state": exp_state,
        "n_sessions": int(len(space)),
        "n_feature_rows": int(len(data)),
        "n_features": len(columns),
        "entities": sorted(map(str, data["entity_id"].unique())),
        "paris_entities": sorted(map(str, paris)),
        "global_mean": float(global_mean),
        "hourly_end": str(hourly["hour"].max()),
        "sim_start": str(cut + pd.Timedelta(hours=1)),
        "metrics": metrics,
        "clf_n_estimators": int(clf_n_estimators),
        "reg_n_estimators": int(reg_n_estimators),
        "cv_protocol": "causal_inner_early_stopping",
    }


def main(days, target_mode=DEFAULT_TARGET_MODE):
    ART.mkdir(exist_ok=True)
    sessions, source_kind = load_sessions(days)
    target_mode = str(target_mode).lower()
    if target_mode not in {"nowcast", "forecast"}:
        raise ValueError(f"Unsupported target_mode={target_mode!r}; expected 'nowcast' or 'forecast'.")
    print(f"Target mode: {target_mode} | data source: {source_kind}")
    print(f"Sessions: {len(sessions):,} across {sessions['entity_id'].nunique()} entities")

    print("\nAllocating session energy to hours...")
    hourly = sessions_to_hourly(sessions)
    validate_split(sessions, hourly)
    hourly = reindex_continuous(hourly)
    print(f"Hourly rows: {len(hourly):,}")

    print("\nBuilding features...")
    data = P.build_feature_table(hourly, sessions, target_mode=target_mode)
    before = len(data)
    data = data.dropna().reset_index(drop=True)
    print(f"Feature rows: {before:,} -> {len(data):,} after dropna "
          f"({before - len(data):,} warm-up rows removed)")

    data = data.sort_values("hour").reset_index(drop=True)
    cut = int(len(data) * 0.85)
    dev, test = data.iloc[:cut].copy(), data.iloc[cut:].copy()
    print(f"Development: {len(dev):,}   Untouched test: {len(test):,}")

    target_col = "target" if target_mode == "forecast" else P.TARGET
    stats = P.entity_stats_from(dev)
    global_mean = dev[target_col].mean()
    dev = P.attach_entity_stats(dev, stats, global_mean)
    test = P.attach_entity_stats(test, stats, global_mean)

    X_dev = P.design_matrix(dev)
    columns = list(X_dev.columns)
    X_test = P.design_matrix(test, columns=columns)
    y_dev, y_test = dev[target_col], test[target_col]
    print(f"Design matrix: {X_dev.shape[1]} features")

    clf_n_estimators = _select_n_estimators(X_dev, y_dev, is_classifier=True, target_mode=target_mode)
    reg_n_estimators = _select_n_estimators(X_dev, y_dev, is_classifier=False, target_mode=target_mode)
    print(f"CV-selected classifier n_estimators: {clf_n_estimators}")
    print(f"CV-selected regressor n_estimators: {reg_n_estimators}")

    common = _build_model_common()
    common.update({"n_estimators": clf_n_estimators})
    print("\nTraining classifier...")
    clf = XGBClassifier(eval_metric="logloss", **common)
    clf.fit(X_dev, (y_dev > 0).astype(int), verbose=False)

    reg_common = _build_model_common()
    reg_common.update({"n_estimators": reg_n_estimators})
    print("Training Tweedie regressor on positive rows...")
    pos = (y_dev > 0).values
    reg = XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.3, **reg_common)
    reg.fit(X_dev[pos], y_dev[pos], verbose=False)

    pred, prob = P.hurdle_predict(clf, reg, X_test)
    baseline_eval = evaluate_baselines(test, target_col=target_col, entity_col="entity_id", feature_columns=columns)

    metrics = {
        "mae": float(np.mean(np.abs(y_test - pred))),
        "rmse": float(np.sqrt(np.mean((y_test - pred) ** 2))),
        "wape": float(np.abs(y_test - pred).sum() / np.abs(y_test).sum() * 100.0 if np.abs(y_test).sum() > 0 else 0.0),
        "persistence_mae": float(np.mean(np.abs(y_test - test["lag_1h"]))),
        "persistence_rmse": float(np.sqrt(np.mean((y_test - test["lag_1h"]) ** 2))),
        "baseline_metrics": baseline_eval["summary"],
    }
    print("\nUntouched test:")
    print(f"  MAE  {metrics['mae']:.4f}   (persistence {metrics['persistence_mae']:.4f})")
    print(f"  RMSE {metrics['rmse']:.4f}   (persistence {metrics['persistence_rmse']:.4f})")
    print(f"  WAPE {metrics['wape']:.2f}%")
    if metrics["mae"] > metrics["persistence_mae"]:
        print("  NOTE: the model does NOT beat persistence on MAE here.")

    print("\nBaseline summary: ")
    for name, values in baseline_eval["summary"].items():
        print(f"  {name}: {values}")

    clf.save_model(ART / "clf.ubj")
    reg.save_model(ART / "reg.ubj")
    (ART / "columns.json").write_text(json.dumps(columns, indent=2))
    stats.to_csv(ART / "entity_stats.csv", index=False)

    paris = [e for e in data["entity_id"].unique() if str(e).startswith("paris_")] or list(data["entity_id"].unique()[:1])
    ph = hourly[hourly["entity_id"].isin(paris)]
    cut_hour = ph.loc[ph["hour"].dt.hour == 7, "hour"].max()
    if pd.isna(cut_hour):
        cut_hour = ph["hour"].max()
    tail_start = cut_hour - pd.Timedelta(hours=WARMUP_HOURS)
    wu = ph[(ph["hour"] > tail_start) & (ph["hour"] <= cut_hour)].copy()
    ws = sessions[(sessions["entity_id"].isin(paris)) &
                  (sessions["connect_time_utc"] > tail_start) &
                  (sessions["connect_time_utc"] <= cut_hour)].copy()
    wu.to_parquet(ART / "warmup_hourly.parquet", index=False)
    ws.to_parquet(ART / "warmup_sessions.parquet", index=False)

    meta = _build_model_metadata(
        source_kind,
        target_mode,
        metrics,
        clf_n_estimators,
        reg_n_estimators,
        columns,
        sessions,
        global_mean,
        hourly,
        data,
        paris,
        cut_hour,
    )
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nArtifacts written to {ART}/  (data_source={source_kind}, target_mode={target_mode})")
    print(f"  warm-up: {len(wu):,} hourly rows, {len(ws):,} sessions across {len(paris)} sites")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--target-mode", choices=["nowcast", "forecast"], default=DEFAULT_TARGET_MODE)
    args = ap.parse_args()
    main(args.days, args.target_mode)
