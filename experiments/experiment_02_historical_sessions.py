"""Test whether historical session-arrival rolling summaries add value."""

from experiments.common import (
    SESSION_ROLLING_FEATURES,
    balanced_selection_score,
    score_configuration,
)


def run(context, selected):
    current = selected["feature_columns"]
    groups = (
        ("session_roll_means", [name for name in SESSION_ROLLING_FEATURES if "_mean_" in name]),
        ("session_roll_stds", [name for name in SESSION_ROLLING_FEATURES if "_std_" in name]),
        ("session_roll_maxima", [name for name in SESSION_ROLLING_FEATURES if "_max_" in name]),
        ("historical_features", list(SESSION_ROLLING_FEATURES)),
    )
    candidates = [selected]
    for name, features in groups:
        candidate = score_configuration(
            context,
            name,
            current + features,
            {
                "added_session_history_features": features,
                "reason": "summarize only session arrivals from completed hours",
            },
            power=selected["power"],
            classifier_params=selected["classifier_params"],
            regressor_params=selected["regressor_params"],
            threshold=selected["threshold"],
        )
        candidates.append(candidate)
    return min(
        candidates,
        key=lambda result: balanced_selection_score(
            result["metrics"],
            context["baseline_metrics"],
        ),
    )
