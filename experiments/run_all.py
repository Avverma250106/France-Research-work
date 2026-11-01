"""Run the validation-led experiments; evaluate the reserved test exactly once at the end."""

import argparse
import json

import pandas as pd

from experiments import (
    experiment_01_session_arrivals as exp01,
    experiment_02_historical_sessions as exp02,
    experiment_03_same_hour_demand as exp03,
    experiment_04_threshold as exp04,
    experiment_05_tweedie_power as exp05,
    experiment_06_hyperparameters as exp06,
    experiment_07_error_analysis as exp07,
    experiment_08_feature_importance as exp08,
)
from experiments.common import (
    RESULTS_DIR,
    SEED,
    attach_training_statistics,
    fit_hurdle,
    load_experiment_data,
    metrics_from_output,
    temporal_splits,
)


def run_all():
    print("Loading sessions and building past-only features...", flush=True)
    frame, sessions = load_experiment_data()
    print(
        f"Feature rows: {len(frame):,}; hourly source: "
        f"{frame.attrs.get('hourly_source', 'not recorded')}",
        flush=True,
    )
    train, validation, test = temporal_splits(frame)
    train, validation, test = attach_training_statistics(train, validation, test)
    context = {
        "train": train,
        "validation": validation,
        "test": test,
        "sessions": sessions,
        "training_range": {
            "start": str(train["hour"].min()),
            "end": str(train["hour"].max()),
        },
        "validation_range": {
            "start": str(validation["hour"].min()),
            "end": str(validation["hour"].max()),
        },
        "test_range": {
            "start": str(test["hour"].min()),
            "end": str(test["hour"].max()),
        },
    }

    print("Experiment 1: session-arrival lags...", flush=True)
    session_best = exp01.run(context)
    print("Experiment 2: historical session summaries...", flush=True)
    historical_best = exp02.run(context, session_best)
    print("Experiment 3: same-hour historical demand...", flush=True)
    demand_best = exp03.run(context, historical_best)
    print("Experiment 5: Tweedie power...", flush=True)
    power_best = exp05.run(context, demand_best)
    print("Experiment 6: stage-wise expanding-window tuning...", flush=True)
    tuned_best = exp06.run(context, power_best)
    print("Experiment 4: validation-only threshold selection...", flush=True)
    selected = exp04.run(context, tuned_best)

    print("Training selected model on development data; evaluating test once...", flush=True)
    development = pd.concat([train, validation], ignore_index=True)
    stat_columns = ["entity_mean_kwh", "entity_std_kwh", "entity_max_kwh"]
    development = development.drop(columns=stat_columns, errors="ignore")
    final_test = test.drop(columns=stat_columns, errors="ignore")
    development, final_test, _ = attach_training_statistics(
        development, final_test, None
    )

    final_output = fit_hurdle(
        development,
        final_test,
        selected["feature_columns"],
        power=selected["power"],
        classifier_params=selected["classifier_params"],
        regressor_params=selected["regressor_params"],
    )
    final_metrics = metrics_from_output(final_output, selected["threshold"])

    final_dir = RESULTS_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = final_dir / "hurdle_classifier.ubj"
    regressor_path = final_dir / "tweedie_regressor.ubj"
    final_output["classifier"].save_model(classifier_path)
    final_output["regressor"].save_model(regressor_path)
    final_payload = {
        "selected_from": selected["name"],
        "selection_metric": "chronological validation MAE",
        "selection_rule": (
            "Equal-weight average of relative overall, active-hour, top-10%, "
            "and top-1% validation MAE; threshold selected by overall validation MAE."
        ),
        "selected_threshold": selected["threshold"],
        "training_date_range": {
            "start": str(development["hour"].min()),
            "end": str(development["hour"].max()),
        },
        "validation_date_range": context["validation_range"],
        "test_date_range": context["test_range"],
        "test_evaluations": 1,
        "test_metrics": final_metrics,
        "configuration": {
            "features": selected["feature_columns"],
            "hourly_source": frame.attrs.get("hourly_source", "not recorded"),
            "stage_1_classifier": selected["classifier_params"],
            "stage_2_tweedie_regressor": {
                **selected["regressor_params"],
                "objective": "reg:tweedie",
                "tweedie_variance_power": selected["power"],
            },
            "random_seed": SEED,
        },
        "model_files": {
            "classifier": classifier_path.name,
            "regressor": regressor_path.name,
        },
        "baseline_test_metrics_previously_reported_not_recomputed": {
            "mae": 2.2144,
            "rmse": 6.3449,
            "r2": 0.8611,
            "wape": 30.32,
        },
    }
    (final_dir / "metrics.json").write_text(json.dumps(final_payload, indent=2))
    (final_dir / "feature_columns.json").write_text(
        json.dumps(selected["feature_columns"], indent=2)
    )

    experiments = []
    for metrics_path in sorted(RESULTS_DIR.glob("*/metrics.json")):
        if metrics_path.parent.name in {"threshold_tuned", "final"}:
            continue
        payload = json.loads(metrics_path.read_text())
        if "metrics" in payload:
            experiments.append({
                "name": payload["experiment"],
                "metrics": payload["metrics"],
            })
    experiments.append({
        "name": "threshold_tuned",
        "metrics": selected["metrics"],
    })
    table = exp07.write_table(experiments, final_metrics)
    importance = exp08.write_importance(
        final_output,
        selected["feature_columns"],
    )
    result = {
        "best_validation_experiment": selected["name"],
        "best_validation_metrics": selected["metrics"],
        "selected_threshold": selected["threshold"],
        "final_test_metrics": final_metrics,
        "final_test_evaluations": 1,
        "experiment_table": table,
        "top_classifier_importance": sorted(
            importance["stage_1_classifier_gain_pct"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:15],
        "top_regressor_importance": sorted(
            importance["stage_2_tweedie_gain_pct"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:15],
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_all()
