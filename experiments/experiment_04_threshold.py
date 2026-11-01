"""Choose the Hurdle classification threshold using validation data only."""

import json

import numpy as np

from experiments.common import RESULTS_DIR, calculate_metrics, predictions_for


def run(context, selected):
    output = selected["output"]
    rows = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.05), 2):
        prediction = predictions_for(output, float(threshold))
        rows.append({
            "threshold": float(threshold),
            **calculate_metrics(
                output["y_validation"],
                prediction,
                probability=output["probability"],
                threshold=float(threshold),
            ),
        })
    best = min(rows, key=lambda row: row["mae"])
    selected = dict(selected)
    selected["threshold"] = best["threshold"]
    selected["metrics"] = best

    directory = RESULTS_DIR / "threshold_tuned"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "threshold_tuned",
        "thresholds": rows,
        "selected_threshold": best["threshold"],
        "selection_metric": "validation MAE",
        "training_date_range": context["training_range"],
        "validation_date_range": context["validation_range"],
        "test_date_range_reserved_not_evaluated": context["test_range"],
        "model_parameters": {
            "stage_1_classifier": selected["classifier_params"],
            "stage_2_tweedie_regressor": {
                **selected["regressor_params"],
                "objective": "reg:tweedie",
                "tweedie_variance_power": selected["power"],
            },
        },
        "feature_columns": selected["feature_columns"],
        "random_seed": 42,
    }
    (directory / "metrics.json").write_text(json.dumps(payload, indent=2))
    (directory / "feature_columns.json").write_text(
        json.dumps(selected["feature_columns"], indent=2)
    )
    return selected
