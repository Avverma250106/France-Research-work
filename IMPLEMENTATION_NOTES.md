# Implementation notes

This update preserves the live simulator while tightening the offline training protocol and adding the missing causal baseline layer.

## What the repo was doing before

- `pipeline.py` builds the shared hourly feature table and reuses the same feature logic in training and live serving.
- `train_for_sim.py` was training the hurdle model on a single 85/15 split by hour and hard-coding `n_estimators=500`.
- `sim/features_rt.py` rebuilds a bounded feature frame for the live ring buffer and carries `expanding_mean` as stored state so it stays aligned with training.
- `sim/model_service.py` reads the persisted metadata and serves the model without altering the feature pipeline.
- The demo remains a nowcast because the model predicts current-hour load using session-start information available in the simulation.

## Corrected protocol implemented

- Added a causal inner-validation loop so each outer fold validates only after the inner split chooses the number of boosting rounds.
- Kept the final training split and test window consistent between the model and the baseline suite.
- Added a reusable `baselines.py` module with causal persistence, seasonal naive, climatology, blend, and ridge baselines.
- Added explicit `target_mode` support (`nowcast` / `forecast`) and `data_source` metadata (`real` / `synthetic`) for metadata and UI clarity.
- Left the live nowcast simulator unchanged while making the code ready for a proper forecast experiment.

## Files changed

- `pipeline.py`
- `train_for_sim.py`
- `sim/model_service.py`
- `sim/server.py`
- `baselines.py`
- `IMPLEMENTATION_NOTES.md`

The old notebook-era R² table was not carried into the active training path; the live script now reports the corrected evaluation protocol only.
