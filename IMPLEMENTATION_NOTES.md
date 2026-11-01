# Implementation notes

This update preserves the live simulator while tightening the offline training protocol and adding the missing causal baseline layer.

## What the repo was doing before

- `pipeline.py` builds the shared hourly feature table and reuses the same feature logic in training and live serving.
- `train_for_sim.py` was training the hurdle model on a single 85/15 split by hour and hard-coding `n_estimators=500`.
- `sim/features_rt.py` rebuilds a bounded feature frame for the live ring buffer and carries `expanding_mean` as stored state so it stays aligned with training.
- `sim/model_service.py` reads the persisted metadata and serves the model without altering the feature pipeline.
- The live forecast path now scores each T+1 prediction against the actual for that exact target hour.

## Corrected protocol implemented

- Added a causal inner-validation loop so each outer fold validates only after the inner split chooses the number of boosting rounds.
- Kept the final training split and test window consistent between the model and the baseline suite.
- Added a reusable `baselines.py` module with causal persistence, seasonal naive, climatology, blend, and ridge baselines.
- Added explicit `target_mode` support (`nowcast` / `forecast`) and `data_source` metadata (`real` / `synthetic`) for metadata and UI clarity.
- Added a strict 336-hour per-entity warm-up requirement; mismatched artifacts fail clearly instead of starting with incomplete features.
- Corrected forecast persistence to `y_hat(t+1) = y(t)` in both live and offline evaluation.
- Added per-entity forecast-pair diagnostics and explicit valid-pair/target-hour counts to the rolling metrics.

## Files changed

- `pipeline.py`
- `train_for_sim.py`
- `sim/model_service.py`
- `sim/server.py`
- `sim/engine.py`
- `sim/features_rt.py`
- `sim/static/index.html`
- `baselines.py`
- `test_sim_forecast.py`
- `README.md`
- `DEMO.md`
- `IMPLEMENTATION_NOTES.md`

The old notebook-era R² table was not carried into the active training path; the live script now reports the corrected evaluation protocol only.
