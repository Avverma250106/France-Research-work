"""Shared, leakage-checked utilities for isolated Hurdle-XGBoost experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

import pipeline as P
from session_to_hourly import reindex_continuous, sessions_to_hourly

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "harmonized_sessions"
HOURLY_CACHE_DIR = ROOT.parent / "hourly_output"
RESULTS_DIR = ROOT / "results"
SEED = 42
BASE_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": 4,
    "tree_method": "hist",
}
TWEEDIE_POWERS = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8)
SESSION_ROLLING_FEATURES = [
    f"sessions_started_roll_{stat}_{window}h"
    for window in (24, 168)
    for stat in ("mean", "std", "max")
]
SAME_HOUR_FEATURES = [
    "same_hour_previous_day",
    "same_hour_previous_week",
    "same_hour_mean_7d",
    "same_hour_median_7d",
    "same_hour_std_7d",
]


def load_experiment_data():
    """Build the same offline feature table as training, without touching artifacts."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Expected the real session corpus in {DATA_DIR}")
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No session CSVs found in {DATA_DIR}")
    sessions = pd.concat(
        [
            pd.read_csv(
                path,
                parse_dates=["connect_time_utc", "charge_end_time_utc"],
            )
            for path in files
        ],
        ignore_index=True,
    )
    for column in ("connect_time_utc", "charge_end_time_utc"):
        sessions[column] = pd.to_datetime(sessions[column], utc=True)

    cache_files = sorted(HOURLY_CACHE_DIR.glob("*_hourly.csv"))
    if cache_files:
        hourly_source = "verified hourly_output cache (valid-session energy conserved)"
        hourly = pd.concat(
            [pd.read_csv(path, parse_dates=["hour"]) for path in cache_files],
            ignore_index=True,
        )
        hourly["hour"] = pd.to_datetime(hourly["hour"], utc=True)
        duration = (
            pd.to_datetime(sessions["charge_end_time_utc"], utc=True)
            - pd.to_datetime(sessions["connect_time_utc"], utc=True)
        )
        valid_energy = (
            sessions.loc[duration > pd.Timedelta(0)]
            .groupby(["source_dataset", "entity_id"])["energy_kwh"]
            .sum()
        )
        cached_energy = hourly.groupby(
            ["source_dataset", "entity_id"]
        )["kwh_allocated"].sum()
        difference = valid_energy.subtract(cached_energy, fill_value=0).abs()
        if (difference > 1e-6).any():
            raise ValueError(
                "Precomputed hourly files do not conserve valid session energy: "
                f"{difference[difference > 1e-6].to_dict()}"
            )
        hourly = reindex_continuous(hourly)
    else:
        hourly_source = "recomputed from session records"
        hourly = reindex_continuous(sessions_to_hourly(sessions))
    frame = P.build_feature_table(hourly, sessions)
    frame = frame.dropna().sort_values(["hour", "entity_id"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Feature generation produced no complete training rows")

    frame = add_session_history_features(frame, sessions)
    frame = add_same_hour_demand_features(frame)
    frame.attrs["hourly_source"] = hourly_source
    return frame, sessions


def add_session_history_features(frame, sessions):
    """Add counts and rolling statistics calculated strictly before each row hour."""
    result = frame.copy().reset_index(drop=True)
    source = sessions[["source_dataset", "entity_id", "connect_time_utc"]].copy()
    source["hour"] = pd.to_datetime(source["connect_time_utc"], utc=True).dt.floor("h")
    counts = (
        source.groupby(["source_dataset", "entity_id", "hour"])
        .size()
        .rename("_session_count")
        .reset_index()
    )
    result = result.merge(
        counts,
        on=["source_dataset", "entity_id", "hour"],
        how="left",
        validate="one_to_one",
    )
    current = result["_session_count"].fillna(0).astype(float)
    keys = [result["source_dataset"], result["entity_id"]]
    grouped = current.groupby(keys)

    for lag in (24, 168):
        result[f"sessions_started_lag{lag}h"] = grouped.shift(lag)

    previous_hour = grouped.shift(1)
    for window in (24, 168):
        rolling = previous_hour.groupby(keys).rolling(
            window,
            min_periods=6 if window == 24 else 24,
        )
        result[f"sessions_started_roll_mean_{window}h"] = (
            rolling.mean().reset_index(level=[0, 1], drop=True)
        )
        result[f"sessions_started_roll_std_{window}h"] = (
            rolling.std().reset_index(level=[0, 1], drop=True)
        )
        result[f"sessions_started_roll_max_{window}h"] = (
            rolling.max().reset_index(level=[0, 1], drop=True)
        )

    result = result.drop(columns="_session_count")
    return result


def add_same_hour_demand_features(frame):
    """Add prior same-local-hour demand observations and their trailing stats."""
    result = frame.sort_values(
        ["source_dataset", "entity_id", "hour"]
    ).copy().reset_index(drop=True)
    keys = [
        result["source_dataset"],
        result["entity_id"],
        result["hour_of_day"],
    ]
    grouped = result[P.TARGET].groupby(keys)
    prior_same_hour = grouped.shift(1)
    result["same_hour_previous_day"] = prior_same_hour
    result["same_hour_previous_week"] = grouped.shift(7)

    prior_groups = prior_same_hour.groupby(keys).rolling(7, min_periods=3)
    result["same_hour_mean_7d"] = (
        prior_groups.mean().reset_index(level=[0, 1, 2], drop=True)
    )
    result["same_hour_median_7d"] = (
        prior_groups.median().reset_index(level=[0, 1, 2], drop=True)
    )
    result["same_hour_std_7d"] = (
        prior_groups.std().reset_index(level=[0, 1, 2], drop=True)
    )
    return result


def temporal_splits(frame):
    """Use row-quantile cutoffs aligned to complete timestamps, never splitting an hour."""
    ordered = frame.sort_values(["hour", "entity_id"]).reset_index(drop=True)
    if len(ordered) < 20:
        raise ValueError("At least 20 rows are required for temporal splits")
    test_hour = ordered.iloc[int(len(ordered) * 0.85)]["hour"]
    development = ordered[ordered["hour"] < test_hour].copy()
    validation_hour = development.iloc[int(len(development) * 0.80)]["hour"]
    train = development[development["hour"] < validation_hour].copy()
    validation = development[development["hour"] >= validation_hour].copy()
    test = ordered[ordered["hour"] >= test_hour].copy()
    if not (train["hour"].max() < validation["hour"].min()
            and validation["hour"].max() < test["hour"].min()):
        raise AssertionError("Temporal splits overlap or are not chronological")
    return train, validation, test


def attach_training_statistics(train, validation, test=None):
    stats = P.entity_stats_from(train)
    global_mean = float(train[P.TARGET].mean())
    train = P.attach_entity_stats(train, stats, global_mean)
    validation = P.attach_entity_stats(validation, stats, global_mean)
    if test is not None:
        test = P.attach_entity_stats(test, stats, global_mean)
    return train, validation, test


def base_feature_columns(train):
    matrix = P.design_matrix(train)
    extra_columns = {
        "sessions_started_lag168h",
        *SESSION_ROLLING_FEATURES,
        *SAME_HOUR_FEATURES,
    }
    return [column for column in matrix.columns if column not in extra_columns]


def matrix_for(frame, columns):
    return P.design_matrix(frame, columns=columns)


def fit_hurdle(
    train,
    validation,
    feature_columns,
    params=None,
    power=1.3,
    classifier_params=None,
    regressor_params=None,
):
    common = dict(BASE_PARAMS if params is None else params)
    clf_params = dict(common if classifier_params is None else classifier_params)
    reg_params = dict(common if regressor_params is None else regressor_params)
    clf_params.setdefault("random_state", SEED)
    reg_params.setdefault("random_state", SEED)
    X_train = matrix_for(train, feature_columns)
    X_validation = matrix_for(validation, feature_columns)
    y_train = train[P.TARGET].to_numpy(dtype=float)
    y_validation = validation[P.TARGET].to_numpy(dtype=float)

    classifier = XGBClassifier(eval_metric="logloss", **clf_params)
    classifier.fit(X_train, (y_train > 0).astype(int), verbose=False)
    positive = y_train > 0
    if not positive.any():
        raise ValueError("Training split has no positive-demand observations")
    regressor = XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=float(power),
        **reg_params,
    )
    regressor.fit(X_train.loc[positive], y_train[positive], verbose=False)

    probability = classifier.predict_proba(X_validation)[:, 1]
    magnitude = np.clip(regressor.predict(X_validation), 0, None)
    return {
        "classifier": classifier,
        "regressor": regressor,
        "probability": probability,
        "magnitude": magnitude,
        "y_validation": y_validation,
    }


def predictions_for(output, threshold=0.5):
    return np.where(output["probability"] >= threshold, output["magnitude"], 0.0)


def calculate_metrics(y_true, prediction, probability=None, threshold=0.5):
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    active = y_true > 0
    top_10 = y_true >= np.quantile(y_true, 0.90)
    top_1 = y_true >= np.quantile(y_true, 0.99)
    metrics = {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
        "wape": float(np.abs(y_true - prediction).sum() / np.abs(y_true).sum() * 100),
        "active_hour_mae": float(mean_absolute_error(y_true[active], prediction[active])),
        "top_10pct_demand_mae": float(mean_absolute_error(y_true[top_10], prediction[top_10])),
        "top_1pct_demand_mae": float(mean_absolute_error(y_true[top_1], prediction[top_1])),
        "n_observations": int(len(y_true)),
        "active_hours": int(active.sum()),
        "top_10pct_threshold_kwh": float(np.quantile(y_true, 0.90)),
        "top_1pct_threshold_kwh": float(np.quantile(y_true, 0.99)),
    }
    if probability is not None:
        predicted_active = np.asarray(probability) >= threshold
        truth_active = active.astype(int)
        metrics.update({
            "precision": float(precision_score(truth_active, predicted_active, zero_division=0)),
            "recall": float(np.sum(predicted_active & active) / active.sum()) if active.any() else 0.0,
            "f1": float(f1_score(truth_active, predicted_active, zero_division=0)),
            "threshold": float(threshold),
        })
    return metrics


def metrics_from_output(output, threshold=0.5):
    prediction = predictions_for(output, threshold)
    return calculate_metrics(
        output["y_validation"],
        prediction,
        probability=output["probability"],
        threshold=threshold,
    )


def date_range(frame):
    return {
        "start": str(frame["hour"].min()),
        "end": str(frame["hour"].max()),
    }


def save_experiment(name, configuration, feature_columns, train, validation, test, metrics,
                    model_parameters, random_seed=SEED):
    directory = RESULTS_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": name,
        "configuration": configuration,
        "training_date_range": date_range(train),
        "validation_date_range": date_range(validation),
        "test_date_range_reserved_not_evaluated": date_range(test),
        "n_training_rows": int(len(train)),
        "n_validation_rows": int(len(validation)),
        "n_test_rows_reserved": int(len(test)),
        "metrics": metrics,
        "model_parameters": model_parameters,
        "random_seed": int(random_seed),
    }
    (directory / "metrics.json").write_text(json.dumps(payload, indent=2))
    (directory / "feature_columns.json").write_text(
        json.dumps(list(feature_columns), indent=2)
    )
    return payload


def score_configuration(
    context,
    name,
    feature_columns,
    configuration,
    power=1.3,
    classifier_params=None,
    regressor_params=None,
    threshold=0.5,
):
    classifier_params = dict(BASE_PARAMS if classifier_params is None else classifier_params)
    regressor_params = dict(BASE_PARAMS if regressor_params is None else regressor_params)
    output = fit_hurdle(
        context["train"],
        context["validation"],
        feature_columns,
        power=power,
        classifier_params=classifier_params,
        regressor_params=regressor_params,
    )
    metrics = metrics_from_output(output, threshold)
    reference = context.get("baseline_metrics")
    if reference is not None:
        metrics["balanced_selection_score"] = balanced_selection_score(metrics, reference)
    parameters = {
        "stage_1_classifier": classifier_params,
        "stage_2_tweedie_regressor": {
            **regressor_params,
            "objective": "reg:tweedie",
            "tweedie_variance_power": float(power),
        },
        "threshold": float(threshold),
    }
    save_experiment(
        name,
        configuration,
        feature_columns,
        context["train"],
        context["validation"],
        context["test"],
        metrics,
        parameters,
    )
    return {
        "name": name,
        "configuration": configuration,
        "feature_columns": list(feature_columns),
        "power": float(power),
        "threshold": float(threshold),
        "classifier_params": classifier_params,
        "regressor_params": regressor_params,
        "metrics": metrics,
        "output": output,
    }


def balanced_selection_score(metrics, reference):
    """Give overall, active, top-decile, and top-percentile MAE equal relative weight."""
    metrics_to_balance = (
        "mae",
        "active_hour_mae",
        "top_10pct_demand_mae",
        "top_1pct_demand_mae",
    )
    ratios = []
    for name in metrics_to_balance:
        denominator = max(abs(float(reference[name])), 1e-12)
        ratios.append(float(metrics[name]) / denominator)
    return float(np.mean(ratios))


def gain_importance(classifier, regressor, feature_columns):
    def normalize(values):
        total = sum(values.values())
        return {
            name: (float(value / total * 100) if total else 0.0)
            for name, value in values.items()
        }

    def stage_values(model):
        scores = model.get_booster().get_score(importance_type="gain")
        result = {name: float(scores.get(name, scores.get(f"f{index}", 0.0)))
                  for index, name in enumerate(feature_columns)}
        return normalize(result)

    return {
        "stage_1_classifier_gain_pct": stage_values(classifier),
        "stage_2_tweedie_gain_pct": stage_values(regressor),
    }


def json_safe(value):
    return json.loads(json.dumps(value, default=str))
