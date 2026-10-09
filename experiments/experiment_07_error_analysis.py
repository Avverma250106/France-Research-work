"""Write a comparable validation/test metric table for all evaluated variants."""

import csv
import json

from experiments.common import RESULTS_DIR


TABLE_COLUMNS = [
    "Model/Experiment",
    "MAE",
    "RMSE",
    "R2",
    "WAPE",
    "Active MAE",
    "Top-10% MAE",
    "Top-1% MAE",
]


def write_table(experiments, final_test_metrics=None):
    rows = []
    for experiment in experiments:
        metrics = experiment["metrics"]
        rows.append({
            "Model/Experiment": experiment["name"],
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "R2": metrics["r2"],
            "WAPE": metrics["wape"],
            "Active MAE": metrics["active_hour_mae"],
            "Top-10% MAE": metrics["top_10pct_demand_mae"],
            "Top-1% MAE": metrics["top_1pct_demand_mae"],
        })
    if final_test_metrics is not None:
        rows.append({
            "Model/Experiment": "selected_final_untouched_test",
            "MAE": final_test_metrics["mae"],
            "RMSE": final_test_metrics["rmse"],
            "R2": final_test_metrics["r2"],
            "WAPE": final_test_metrics["wape"],
            "Active MAE": final_test_metrics["active_hour_mae"],
            "Top-10% MAE": final_test_metrics["top_10pct_demand_mae"],
            "Top-1% MAE": final_test_metrics["top_1pct_demand_mae"],
        })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "experiment_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / "experiment_table.json").write_text(json.dumps(rows, indent=2))
    return rows
