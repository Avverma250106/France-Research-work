"""Compare 24-hour, 168-hour, and combined session-arrival lags."""

import json

from experiments.common import (
    RESULTS_DIR,
    balanced_selection_score,
    base_feature_columns,
    score_configuration,
)


def run(context):
    base = base_feature_columns(context["train"])
    without_lag24 = [column for column in base if column != "sessions_started_lag24h"]
    variants = (
        ("baseline_24h", base, "sessions_started_lag24h only"),
        (
            "session_168h",
            without_lag24 + ["sessions_started_lag168h"],
            "sessions_started_lag168h only",
        ),
        (
            "session_24_168",
            base + ["sessions_started_lag168h"],
            "sessions_started_lag24h and sessions_started_lag168h",
        ),
    )
    results = []
    for name, columns, description in variants:
        result = score_configuration(
            context,
            name,
            columns,
            {"session_lags": description, "reason": "compare daily and weekly arrival cycles"},
        )
        results.append(result)
        if len(results) == 1:
            context["baseline_metrics"] = result["metrics"]
    baseline_metrics = results[0]["metrics"]
    for result in results:
        result["metrics"]["balanced_selection_score"] = balanced_selection_score(
            result["metrics"],
            baseline_metrics,
        )
        path = RESULTS_DIR / result["name"] / "metrics.json"
        payload = json.loads(path.read_text())
        payload["metrics"] = result["metrics"]
        payload["selection_policy"] = (
            "Equal-weight relative MAE across overall, active-hour, top-10%, and top-1%."
        )
        path.write_text(json.dumps(payload, indent=2))
    return min(
        results,
        key=lambda result: result["metrics"]["balanced_selection_score"],
    )
