"""Compare Tweedie variance powers on the fixed chronological validation window."""

import numpy as np
from xgboost import XGBRegressor

from experiments.common import (
    RESULTS_DIR,
    TWEEDIE_POWERS,
    balanced_selection_score,
    calculate_metrics,
    matrix_for,
    save_experiment,
)


def run(context, selected):
    results = [selected]
    X_train = matrix_for(context["train"], selected["feature_columns"])
    X_validation = matrix_for(context["validation"], selected["feature_columns"])
    y_train = context["train"]["kwh_allocated"].to_numpy(dtype=float)
    positive = y_train > 0
    for power in TWEEDIE_POWERS:
        if power == selected["power"]:
            output = dict(selected["output"])
        else:
            regressor = XGBRegressor(
                objective="reg:tweedie",
                tweedie_variance_power=float(power),
                **selected["regressor_params"],
            )
            regressor.fit(X_train.loc[positive], y_train[positive], verbose=False)
            output = dict(selected["output"])
            output["regressor"] = regressor
            output["magnitude"] = np.clip(regressor.predict(X_validation), 0, None)

        prediction = np.where(
            output["probability"] >= selected["threshold"],
            output["magnitude"],
            0.0,
        )
        metrics = calculate_metrics(
            output["y_validation"],
            prediction,
            probability=output["probability"],
            threshold=selected["threshold"],
        )
        metrics["balanced_selection_score"] = balanced_selection_score(
            metrics,
            context["baseline_metrics"],
        )
        name = f"tweedie_power_{power:.1f}"
        parameters = {
            "stage_1_classifier": selected["classifier_params"],
            "stage_2_tweedie_regressor": {
                **selected["regressor_params"],
                "objective": "reg:tweedie",
                "tweedie_variance_power": float(power),
            },
            "threshold": selected["threshold"],
        }
        save_experiment(
            name,
            {
                "tweedie_variance_power": power,
                "reason": "test the positive-demand dispersion assumption",
            },
            selected["feature_columns"],
            context["train"],
            context["validation"],
            context["test"],
            metrics,
            parameters,
        )
        candidate = {
            **selected,
            "name": name,
            "power": float(power),
            "metrics": metrics,
            "output": output,
        }
        results.append(candidate)
    return min(
        results,
        key=lambda result: balanced_selection_score(
            result["metrics"],
            context["baseline_metrics"],
        ),
    )
