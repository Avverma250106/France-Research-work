"""Test same-local-hour historical demand features."""

from experiments.common import (
    SAME_HOUR_FEATURES,
    balanced_selection_score,
    score_configuration,
)


def run(context, selected):
    columns = selected["feature_columns"] + SAME_HOUR_FEATURES
    candidate = score_configuration(
        context,
        "same_hour_demand",
        columns,
        {
            "added_demand_features": SAME_HOUR_FEATURES,
            "reason": "capture daily and weekly diurnal demand recurrence",
        },
        power=selected["power"],
        classifier_params=selected["classifier_params"],
        regressor_params=selected["regressor_params"],
        threshold=selected["threshold"],
    )
    if balanced_selection_score(
        candidate["metrics"],
        context["baseline_metrics"],
    ) < balanced_selection_score(
        selected["metrics"],
        context["baseline_metrics"],
    ):
        return candidate
    return selected
