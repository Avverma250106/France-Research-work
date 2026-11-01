import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _as_numpy(values):
    return np.asarray(values, dtype=float)


def rmse(y_true, y_pred):
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true, y_pred):
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    denom = np.abs(y_true).sum()
    if denom == 0.0:
        return 0.0
    return float(np.abs(y_true - y_pred).sum() / denom * 100.0)


def nrmse(y_true, y_pred, scale=None):
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    if scale is None:
        scale = float(np.std(y_true)) if np.std(y_true) > 0 else float(np.abs(np.mean(y_true)) + 1e-9)
    return float(rmse(y_true, y_pred) / max(scale, 1e-9))


def persistence_forecast(df, target_col="kwh_allocated", entity_col="entity_id"):
    out = df.sort_values([entity_col, "hour"]).copy()
    if target_col == "target" and "kwh_allocated" in out.columns:
        # Forecast rows are indexed by feature time t and target y(t+1).
        # Persistence for that target is the observed y(t) in the same row.
        out["forecast"] = out["kwh_allocated"]
    else:
        out["forecast"] = out.groupby(entity_col, sort=False)[target_col].shift(1)
        out["forecast"] = out["forecast"].fillna(
            out.groupby(entity_col, sort=False)[target_col].transform("mean")
        )
    return out["forecast"].to_numpy()


def seasonal_naive_forecast(df, lag_hours, target_col="kwh_allocated", entity_col="entity_id"):
    out = df.sort_values([entity_col, "hour"]).copy()
    out["forecast"] = out.groupby(entity_col, sort=False)[target_col].shift(lag_hours)
    out["forecast"] = out["forecast"].fillna(out.groupby(entity_col, sort=False)[target_col].transform("mean"))
    return out["forecast"].to_numpy()


def hour_of_week_climatology(df, target_col="kwh_allocated", entity_col="entity_id", lookback_weeks=8):
    out = df.sort_values([entity_col, "hour"]).copy()
    if "hour_of_day" not in out.columns or "day_of_week" not in out.columns:
        raise ValueError("hour_of_week_climatology requires hour_of_day and day_of_week columns.")

    forecast = np.full(len(out), np.nan, dtype=float)
    entity_groups = out.groupby(entity_col, sort=False)
    for entity, g in entity_groups:
        g = g.sort_values("hour").copy()
        for idx, row in g.iterrows():
            key = (int(row["day_of_week"]), int(row["hour_of_day"]))
            mask = (
                (g["day_of_week"] == key[0])
                & (g["hour_of_day"] == key[1])
                & (g["hour"] < row["hour"])
            )
            prior = g.loc[mask]
            if prior.empty:
                forecast[out.index.get_loc(idx)] = float(g[target_col].mean())
                continue
            cutoff = row["hour"] - pd.Timedelta(weeks=lookback_weeks)
            prior = prior[prior["hour"] >= cutoff]
            forecast[out.index.get_loc(idx)] = float(prior[target_col].mean()) if not prior.empty else float(g[target_col].mean())
    forecast = np.asarray(forecast, dtype=float)
    return np.nan_to_num(forecast, nan=np.nanmean(out[target_col].to_numpy(dtype=float)))


def blend_forecast(base_forecast, climatology_forecast, weight=0.5):
    base = _as_numpy(base_forecast)
    climate = _as_numpy(climatology_forecast)
    return (weight * base) + ((1.0 - weight) * climate)


def _entity_metrics(y_true, y_pred, persistence):
    y = _as_numpy(y_true)
    p = _as_numpy(y_pred)
    q = _as_numpy(persistence)
    persist_rmse = rmse(y, q)
    pred_rmse = rmse(y, p)
    return {
        "rmse": pred_rmse,
        "nrmse": nrmse(y, p),
        "mae": mae(y, p),
        "wape": wape(y, p),
        "rmse_reduction_vs_persistence": float((persist_rmse - pred_rmse) / max(persist_rmse, 1e-9)) if persist_rmse > 0 else 0.0,
    }


def evaluate_baselines(df, target_col="kwh_allocated", entity_col="entity_id", feature_columns=None, ridge_alpha=1.0):
    """Evaluate causal baselines on the same entity-level test slice as the ML model.

    Returns a dictionary with per-entity metrics and a summary across entities.
    """
    work = df.sort_values([entity_col, "hour"]).reset_index(drop=True).copy()
    result = {"by_entity": {}, "summary": {}}

    if "lag_1h" not in work.columns:
        work["lag_1h"] = work.groupby(entity_col, sort=False)[target_col].shift(1)
    if "lag_24h" not in work.columns:
        work["lag_24h"] = work.groupby(entity_col, sort=False)[target_col].shift(24)
    if "lag_168h" not in work.columns:
        work["lag_168h"] = work.groupby(entity_col, sort=False)[target_col].shift(168)
    if "day_of_week" not in work.columns or "hour_of_day" not in work.columns:
        raise ValueError("Baseline evaluation requires day_of_week and hour_of_day columns.")

    pers = persistence_forecast(work, target_col=target_col, entity_col=entity_col)
    naive24 = seasonal_naive_forecast(work, 24, target_col=target_col, entity_col=entity_col)
    naive168 = seasonal_naive_forecast(work, 168, target_col=target_col, entity_col=entity_col)
    climatology = hour_of_week_climatology(work, target_col=target_col, entity_col=entity_col)
    blend = blend_forecast(pers, climatology, 0.5)

    ridge_X = None
    if feature_columns is not None:
        ridge_X = work.reindex(columns=feature_columns, fill_value=0.0)
        ridge_X = ridge_X.fillna(0.0)
    elif any(c.startswith("lag_") or c.startswith("roll_") or c.startswith("hour") or c.startswith("dow") or c.startswith("ema") for c in work.columns):
        ridge_X = work.drop(columns=[c for c in [target_col, entity_col, "hour", "source_dataset", "site", "region"] if c in work.columns])
        ridge_X = pd.get_dummies(ridge_X, columns=[c for c in ["source_dataset", "region"] if c in ridge_X.columns], drop_first=True)
        ridge_X = ridge_X.fillna(0.0)

    ridge_pred = np.zeros(len(work), dtype=float)
    if ridge_X is not None:
        model = Ridge(alpha=ridge_alpha)
        model.fit(ridge_X, work[target_col].to_numpy())
        ridge_pred = model.predict(ridge_X)

    predictors = {
        "persistence": pers,
        "seasonal_naive_24h": naive24,
        "seasonal_naive_168h": naive168,
        "hour_of_week_climatology": climatology,
        "blend": blend,
        "ridge": ridge_pred,
    }

    entity_rows = []
    for entity, g in work.groupby(entity_col, sort=False):
        indices = g.index.to_numpy()
        y = g[target_col].to_numpy(dtype=float)
        for name, pred in predictors.items():
            entity_pred = np.asarray(pred)[indices]
            valid = np.isfinite(y) & np.isfinite(entity_pred) & np.isfinite(pers[indices])
            if not np.any(valid):
                continue
            y_valid = y[valid]
            p_valid = entity_pred[valid]
            q_valid = pers[indices][valid]
            metrics = _entity_metrics(y_valid, p_valid, q_valid)
            entity_rows.append({"entity_id": entity, "method": name, **metrics})

    for name in predictors:
        values = [r for r in entity_rows if r["method"] == name]
        if not values:
            continue
        result["by_entity"][name] = values
        summary = {
            "rmse": float(np.mean([r["rmse"] for r in values])),
            "nrmse": float(np.mean([r["nrmse"] for r in values])),
            "mae": float(np.mean([r["mae"] for r in values])),
            "wape": float(np.mean([r["wape"] for r in values])),
            "rmse_reduction_vs_persistence": float(np.mean([r["rmse_reduction_vs_persistence"] for r in values])),
            "n_entities": len(values),
        }
        result["summary"][name] = summary

    return result
