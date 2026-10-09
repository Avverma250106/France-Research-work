"""Persist separate gain-based feature importance for both Hurdle stages."""

import json

from experiments.common import RESULTS_DIR, gain_importance


def write_importance(selected_final, feature_columns):
    importance = gain_importance(
        selected_final["classifier"],
        selected_final["regressor"],
        feature_columns,
    )
    path = RESULTS_DIR / "final" / "feature_importance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "method": "normalized XGBoost gain, reported separately by stage",
        "feature_importance": importance,
        "legacy_comparison": {
            "lag_1h_pct": 56.4,
            "lag_24h_pct": 7.8,
            "lag_168h_pct": 4.8,
            "prev_day_peak_pct": 3.7,
            "sessions_started_contemporaneous_pct": None,
            "note": "The legacy contemporaneous session importance is invalid and is not compared.",
        },
    }, indent=2))
    return importance
