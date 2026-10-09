"""Feature pipeline for the hourly EV load model.

Lifted verbatim in behaviour from data.ipynb (cells 5-10) and
hurdle_model_cv.ipynb, so that training and live serving share one code path.
The only deviation is that add_temporal_features groups by region instead of
calling df.iterrows() per row - same output, ~100x faster, which matters
because the simulation calls this once per simulated hour.

This module reproduces the notebook's behaviour exactly and deliberately
changes nothing about the model itself.
"""

import numpy as np
import pandas as pd
import holidays

TARGET = "kwh_allocated"

# From hurdle_model_cv.ipynb. entity_id is dropped here, so the three ACN sites
# share one representation.
DROP_COLUMNS = ["hour", "kwh_allocated", "target", "entity_id", "site"]

CATEGORICAL_COLS = ["source_dataset", "region"]

# Professor feedback: sessions_started must be strictly historical.
# 24 means the model sees the number of sessions that started at the same
# entity exactly 24 hours before the prediction hour. Set to 168 to use
# the same hour from the previous week for an alternative experiment.
SESSION_COUNT_LAG_HOURS = 24
SESSION_COUNT_FEATURE = f"sessions_started_lag{SESSION_COUNT_LAG_HOURS}h"

TZ_MAP = {
    "California": "America/Los_Angeles",
    "Colorado": "America/Denver",
    "France": "Europe/Paris",
    "UK": "Europe/London",
    "Australia": "Australia/Perth",
}

_HOLIDAY_FACTORY = {
    "California": holidays.US,
    "Colorado": holidays.US,
    "France": holidays.FR,
    "UK": holidays.UK,
    "Australia": holidays.AU,
}


def add_temporal_features(df):
    df = df.copy()
    df["hour"] = pd.to_datetime(df["hour"], utc=True)

    local_hour = np.zeros(len(df), dtype=int)
    local_day = np.zeros(len(df), dtype=int)
    local_month = np.zeros(len(df), dtype=int)
    is_holiday = np.zeros(len(df), dtype=int)

    for region, idx in df.groupby("region").groups.items():
        pos = df.index.get_indexer(idx)
        local = df.loc[idx, "hour"].dt.tz_convert(TZ_MAP[region])
        local_hour[pos] = local.dt.hour.to_numpy()
        local_day[pos] = local.dt.weekday.to_numpy()
        local_month[pos] = local.dt.month.to_numpy()

        dates = local.dt.date
        years = sorted({d.year for d in dates})
        cal = _HOLIDAY_FACTORY[region](years=years)
        is_holiday[pos] = np.fromiter((d in cal for d in dates), dtype=int, count=len(dates))

    df["hour_of_day"] = local_hour
    df["day_of_week"] = local_day
    df["month"] = local_month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = is_holiday

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_session_count_features(hourly_df, sessions):
    """Add leakage-safe historical session-arrival features.

    The raw session count for hour t is *not* exposed to the model because it
    is only known during/after hour t. Instead, ``SESSION_COUNT_FEATURE`` is the
    count from ``SESSION_COUNT_LAG_HOURS`` earlier. The default 24-hour lag
    follows the supervisor's suggested fix; 168 can be used for a previous-week
    experiment.

    ``sessions_started_roll24h`` remains a historical rolling mean and is built
    from counts shifted by one hour, so the current hour is never included.
    """
    s = sessions.copy()
    s["connect_time_utc"] = pd.to_datetime(s["connect_time_utc"], utc=True)
    s["hour"] = s["connect_time_utc"].dt.floor("h")

    counts = (
        s.groupby(["source_dataset", "entity_id", "hour"])
        .size()
        .reset_index(name="_sessions_started_current")
    )

    df = hourly_df.merge(
        counts,
        on=["source_dataset", "entity_id", "hour"],
        how="left",
    )
    df["_sessions_started_current"] = (
        df["_sessions_started_current"].fillna(0).astype(int)
    )
    df = df.sort_values(["source_dataset", "entity_id", "hour"]).copy()

    groups = [df["source_dataset"], df["entity_id"]]
    current = df["_sessions_started_current"]

    # This is the feature that enters XGBoost. It is strictly historical.
    df[SESSION_COUNT_FEATURE] = (
        current.groupby(groups).shift(SESSION_COUNT_LAG_HOURS)
    )

    # Historical 24-hour arrival intensity. shift(1) excludes the current
    # hour before the rolling window is calculated.
    previous_hour = current.groupby(groups).shift(1)
    df["sessions_started_roll24h"] = (
        previous_hour.groupby(groups)
        .rolling(24, min_periods=6)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Never let the contemporaneous count reach design_matrix().
    df = df.drop(columns=["_sessions_started_current"])
    return df


def add_lag_and_rolling_features(df):
    df = df.sort_values(["source_dataset", "entity_id", "hour"]).copy()
    g = df.groupby(["source_dataset", "entity_id"])[TARGET]
    shifted = g.shift(1)

    for lag in [1, 2, 3, 6, 12, 24, 168]:
        df[f"lag_{lag}h"] = g.shift(lag)

    keys = [df["source_dataset"], df["entity_id"]]

    df["ema_24h"] = (
        shifted.groupby(keys).transform(lambda x: x.ewm(span=24, min_periods=6).mean())
    )
    df["expanding_mean"] = (
        shifted.groupby(keys).transform(lambda x: x.expanding(min_periods=6).mean())
    )

    roll24 = shifted.groupby(keys).rolling(24, min_periods=6)
    df["roll_mean_24h"] = roll24.mean().reset_index(level=[0, 1], drop=True)
    df["roll_std_24h"] = roll24.std().reset_index(level=[0, 1], drop=True)
    df["roll_median_24h"] = roll24.median().reset_index(level=[0, 1], drop=True)
    df["roll_min_24h"] = roll24.min().reset_index(level=[0, 1], drop=True)
    df["roll_max_24h"] = roll24.max().reset_index(level=[0, 1], drop=True)
    df["roll_q25_24h"] = roll24.quantile(0.25).reset_index(level=[0, 1], drop=True)
    df["roll_q75_24h"] = roll24.quantile(0.75).reset_index(level=[0, 1], drop=True)

    roll168 = shifted.groupby(keys).rolling(168, min_periods=24)
    df["roll_mean_168h"] = roll168.mean().reset_index(level=[0, 1], drop=True)
    df["roll_std_168h"] = roll168.std().reset_index(level=[0, 1], drop=True)

    df["trend_1h"] = df["lag_1h"] - df["lag_2h"]
    df["accel_1h"] = df["trend_1h"] - (df["lag_2h"] - df["lag_3h"])
    df["trend_24h"] = df["roll_mean_24h"] - df["roll_mean_168h"]

    df["prev_day_peak"] = roll24.max().reset_index(level=[0, 1], drop=True)
    df["prev_week_peak"] = roll168.max().reset_index(level=[0, 1], drop=True)

    zero = shifted.eq(0)
    df["consec_zero_hours"] = zero.groupby(keys).transform(
        lambda x: x.groupby((~x).cumsum()).cumsum()
    )
    active = shifted.gt(0)
    df["consec_active_hours"] = active.groupby(keys).transform(
        lambda x: x.groupby((~x).cumsum()).cumsum()
    )

    df["volatility_24h"] = df["roll_std_24h"] / (df["roll_mean_24h"] + 1e-6)
    return df


def build_feature_table(hourly_reindexed, sessions):
    df = add_temporal_features(hourly_reindexed)
    df = add_session_count_features(df, sessions)
    df = add_lag_and_rolling_features(df)
    return df


def validate_forecast_feature_leakage(df):
    """Reject feature names that directly expose target-hour or future data."""
    forbidden = []
    for column in df.columns:
        name = str(column).lower()
        if (
            name == "sessions_started"
            or (
                "sessions_started" in name
                and not (
                    name.startswith("sessions_started_lag")
                    or name.startswith("sessions_started_roll")
                )
            )
            or "session_count_current" in name
            or name.startswith(
                ("current_", "actual_", "target_hour_", "future_", "lead_", "next_")
            )
            or "_current" in name
            or "target_hour" in name
            or "_future" in name
            or "future_" in name
        ):
            forbidden.append(str(column))
    if forbidden:
        raise ValueError(
            "Forecast features contain current-hour or future information: "
            + ", ".join(sorted(forbidden))
        )


def entity_stats_from(train_df):
    """Per-entity statistics over the training window.

    Persisted alongside the model so that serving reproduces training exactly.
    """
    stats = (
        train_df.groupby("entity_id")[TARGET]
        .agg(entity_mean_kwh="mean", entity_std_kwh="std", entity_max_kwh="max")
        .reset_index()
    )
    stats["entity_std_kwh"] = stats["entity_std_kwh"].fillna(0)
    return stats


def attach_entity_stats(df, stats, global_mean):
    df = df.merge(stats, on="entity_id", how="left")
    df["entity_mean_kwh"] = df["entity_mean_kwh"].fillna(global_mean)
    df["entity_std_kwh"] = df["entity_std_kwh"].fillna(0)
    df["entity_max_kwh"] = df["entity_max_kwh"].fillna(global_mean)
    return df


def design_matrix(df, columns=None):
    """Build X. When `columns` is given, reindex onto that exact order.

    Pinning the column order is mandatory for serving: re-deriving it with
    get_dummies at serve time can silently reorder or drop features, and the
    model then returns plausible nonsense.
    """
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    validate_forecast_feature_leakage(X)
    cats = [c for c in CATEGORICAL_COLS if c in X.columns]
    X = pd.get_dummies(X, columns=cats, drop_first=True)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X.astype(float)


def hurdle_predict(clf, reg, X, charge_threshold=0.5, rule="threshold"):
    """Combine the hurdle's two stages into one prediction.

    Two rules, because one point forecast cannot be optimal for both metrics:

      "threshold"  1{p >= charge_threshold} * magnitude
                   Minimises L1-style error. This is what the experiment
                   programme tuned (threshold selected on validation), and is
                   the default so saved artifacts keep their behaviour.

      "expected"   p * magnitude
                   The conditional mean, which is the minimum-MSE point
                   forecast. Report this when scoring RMSE or R2.

    Returns (prediction, charge probability).
    """
    if rule not in ("threshold", "expected"):
        raise ValueError(f"Unknown hurdle rule {rule!r}; use 'threshold' or 'expected'.")
    charge_prob = clf.predict_proba(X)[:, 1]
    magnitude = np.clip(reg.predict(X), 0, None)
    if rule == "expected":
        return charge_prob * magnitude, charge_prob
    will_charge = (charge_prob >= charge_threshold).astype(int)
    return will_charge * magnitude, charge_prob
