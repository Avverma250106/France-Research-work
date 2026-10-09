# Leakage-safe Hurdle-XGBoost experiments

Run the experiment suite with the repository environment:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_all
```

The scripts under this directory are separate from `pipeline.py`'s active
simulation model and never write to `artifacts/`. They use the real corpus in
`harmonized_sessions/`, generate historical features through the production
feature pipeline, and reserve the final 15% of distinct timestamps for a
single final evaluation. The validation window is the last 20% of development.
Hyperparameter selection uses two expanding windows inside the training
portion. Entity-level statistics are fitted on each corresponding training
portion only. Feature and power selection use an equal-weight relative-MAE
score over overall, active-hour, top-decile, and top-percentile demand MAE; the
decision threshold is selected by validation overall MAE as requested.

Rows are indexed by the target hour: features for target hour `t` are
computed from observations through `t-1`, so they represent a one-hour-ahead
forecast made at origin `t-1`. Session arrival lag features are verified
against exact 24-hour and 168-hour historical counts.

Results and model files are written below `experiments/results/`, including
configuration, features, date ranges, seed, model parameters, validation
metrics, the final test metrics, and separate Stage 1/Stage 2 gain importance.
The test window is only scored after the feature set, Tweedie power,
hyperparameters, and decision threshold have been selected using validation.
The existing `artifacts/` used by the simulation are not modified.

Each experimental change is in its own module:

- `experiment_01_session_arrivals.py`: 24h, 168h, and combined arrival lags.
- `experiment_02_historical_sessions.py`: lagged rolling arrival statistics.
- `experiment_03_same_hour_demand.py`: prior same-hour demand features.
- `experiment_04_threshold.py`: validation-only threshold selection.
- `experiment_05_tweedie_power.py`: Stage 2 variance power.
- `experiment_06_hyperparameters.py`: separate expanding-window searches.
- `experiment_07_error_analysis.py`: comparable metric table, including active
  and peak-demand errors.
- `experiment_08_feature_importance.py`: gain by Hurdle stage.

Run leakage-focused tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest experiments/test_temporal_leakage.py -q
```
