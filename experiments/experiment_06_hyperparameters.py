"""Small seeded, expanding-window searches for classifier and regressor separately."""

import json

import numpy as np
from sklearn.metrics import log_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from experiments.common import (
    BASE_PARAMS,
    RESULTS_DIR,
    SEED,
    attach_training_statistics,
    balanced_selection_score,
    matrix_for,
    score_configuration,
)


def _expanding_folds(train):
    hours = np.asarray(sorted(train["hour"].unique()))
    if len(hours) < 10:
        raise ValueError("Not enough training hours for expanding-window validation")
    for train_fraction, validation_start, validation_end in (
        (0.50, 0.50, 0.70),
        (0.70, 0.70, 0.90),
    ):
        train_end = int(len(hours) * train_fraction)
        val_start = int(len(hours) * validation_start)
        val_end = int(len(hours) * validation_end)
        train_end_time = hours[train_end]
        val_start_time = hours[val_start]
        val_end_time = hours[min(val_end, len(hours) - 1)]
        fold_train = train[train["hour"] < train_end_time]
        fold_validation = train[
            (train["hour"] >= val_start_time) & (train["hour"] < val_end_time)
        ]
        if len(fold_train) and len(fold_validation):
            yield fold_train, fold_validation


def _candidate_parameters(rng, count=2):
    candidates = [dict(BASE_PARAMS)]
    choices = {
        "n_estimators": [250, 400, 500],
        "learning_rate": [0.03, 0.05, 0.07],
        "max_depth": [3, 4, 6],
        "min_child_weight": [1, 5, 8],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.75, 0.85, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [1.0, 3.0, 5.0],
    }
    while len(candidates) < count:
        parameters = {
            name: choices[name][int(rng.integers(0, len(choices[name])))]
            for name in choices
        }
        parameters.update({"random_state": SEED, "n_jobs": 4, "tree_method": "hist"})
        if parameters not in candidates:
            candidates.append(parameters)
    return candidates


def _search_stage(train, columns, stage, rng):
    candidates = _candidate_parameters(rng)
    scored = []
    for parameters in candidates:
        fold_scores = []
        for fold_train, fold_validation in _expanding_folds(train):
            stat_columns = ["entity_mean_kwh", "entity_std_kwh", "entity_max_kwh"]
            fold_train = fold_train.drop(columns=stat_columns, errors="ignore")
            fold_validation = fold_validation.drop(columns=stat_columns, errors="ignore")
            fold_train, fold_validation, _ = attach_training_statistics(
                fold_train,
                fold_validation,
            )
            X_train = matrix_for(fold_train, columns)
            X_validation = matrix_for(fold_validation, columns)
            y_train = fold_train["kwh_allocated"].to_numpy(dtype=float)
            y_validation = fold_validation["kwh_allocated"].to_numpy(dtype=float)
            if stage == "classifier":
                model = XGBClassifier(eval_metric="logloss", **parameters)
                model.fit(X_train, (y_train > 0).astype(int), verbose=False)
                score = log_loss(
                    (y_validation > 0).astype(int),
                    model.predict_proba(X_validation)[:, 1],
                    labels=[0, 1],
                )
            else:
                positive_train = y_train > 0
                positive_validation = y_validation > 0
                if not positive_train.any() or not positive_validation.any():
                    continue
                model = XGBRegressor(
                    objective="reg:tweedie",
                    tweedie_variance_power=1.3,
                    **parameters,
                )
                model.fit(
                    X_train.loc[positive_train],
                    y_train[positive_train],
                    verbose=False,
                )
                positive_predictions = np.clip(
                    model.predict(X_validation.loc[positive_validation]), 0, None
                )
                score = mean_absolute_error(
                    y_validation[positive_validation], positive_predictions
                )
            fold_scores.append(float(score))
        scored.append({
            "parameters": parameters,
            "fold_scores": fold_scores,
            "mean_score": float(np.mean(fold_scores)),
        })
    return min(scored, key=lambda candidate: candidate["mean_score"]), scored


def run(context, selected):
    rng = np.random.default_rng(SEED)
    classifier_best, classifier_results = _search_stage(
        context["train"], selected["feature_columns"], "classifier", rng
    )
    regressor_best, regressor_results = _search_stage(
        context["train"], selected["feature_columns"], "regressor", rng
    )
    candidate = score_configuration(
        context,
        "hyperparameter_optimization",
        selected["feature_columns"],
        {
            "search": "2 candidates per stage; two expanding-window folds",
            "stage_1_selection_metric": "mean fold log loss",
            "stage_2_selection_metric": "mean positive-hour fold MAE",
        },
        power=selected["power"],
        classifier_params=classifier_best["parameters"],
        regressor_params=regressor_best["parameters"],
        threshold=selected["threshold"],
    )
    if balanced_selection_score(
        candidate["metrics"],
        context["baseline_metrics"],
    ) >= balanced_selection_score(
        selected["metrics"],
        context["baseline_metrics"],
    ):
        candidate = selected

    directory = RESULTS_DIR / "hyperparameter_optimization"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage_1_classifier_search": classifier_results,
        "stage_2_tweedie_regressor_search": regressor_results,
        "selected_validation_experiment": candidate["name"],
        "training_date_range": context["training_range"],
        "validation_date_range": context["validation_range"],
        "test_date_range_reserved_not_evaluated": context["test_range"],
        "random_seed": SEED,
    }
    (directory / "search.json").write_text(json.dumps(payload, indent=2))
    return candidate
