"""Create publication-ready figures from saved leakage-safe experiment results."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures_updated"

COLORS = {
    "blue": "#2878B5",
    "orange": "#E07A26",
    "green": "#3A923A",
    "red": "#C44536",
    "purple": "#7A5AA6",
    "gray": "#737373",
    "light_gray": "#B9C0C8",
}


def _load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_experiment_table():
    with (RESULTS / "experiment_table.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("experiment_table.csv has no saved metrics")
    return {row["Model/Experiment"]: row for row in rows}


def _set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D9DDE2",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
    })


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _metric_bars(rows, ylabel, title, filename, units):
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    y = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(8.0, max(3.3, 0.42 * len(values) + 1.2)))
    bars = ax.barh(y, values, color=COLORS["blue"], height=0.68)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title, pad=10, weight="bold")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.16)
    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f} {units}".rstrip(),
            va="center",
            ha="left",
            fontsize=8,
        )
    _save(fig, filename)


def figure_1_experiment_comparison(table):
    mapping = [
        ("baseline_24h", "Baseline: session lag 24h"),
        ("session_168h", "Session lag 168h"),
        ("session_24_168", "Session lags 24h + 168h"),
        ("session_roll_means", "Historical session rolling means"),
        ("same_hour_demand", "Same-hour historical demand"),
        ("tweedie_power_1.2", "Tweedie power p = 1.2"),
        ("threshold_tuned", "Selected threshold = 0.75"),
    ]
    display = {
        "baseline_24h": "Baseline: session lag 24h",
        "session_168h": "Session lag 168h",
        "session_24_168": "Session lags 24h + 168h",
        "session_roll_means": "Historical session rolling means",
        "same_hour_demand": "Same-hour historical demand",
        "tweedie_power_1.2": "Tweedie power p = 1.2",
        "threshold_tuned": "Selected threshold = 0.75",
    }
    missing = [name for name, _ in mapping if name not in table]
    if missing:
        raise ValueError(f"Missing required saved experiment rows: {missing}")
    mae_rows = [
        (display[name], float(table[name]["MAE"]))
        for name, _ in mapping
    ]
    rmse_rows = [
        (display[name], float(table[name]["RMSE"]))
        for name, _ in mapping
    ]
    _metric_bars(
        mae_rows,
        "Validation MAE (kWh)",
        "Validation Performance Across Leakage-Safe Experiments",
        "fig1_validation_experiment_mae.png",
        "kWh",
    )
    _metric_bars(
        rmse_rows,
        "Validation RMSE (kWh)",
        "Validation RMSE Across Leakage-Safe Experiments",
        "fig1b_validation_experiment_rmse.png",
        "kWh",
    )


def figure_2_final_vs_baseline():
    final = _load_json(RESULTS / "final" / "metrics.json")
    metrics = final["test_metrics"]
    baseline = final["baseline_test_metrics_previously_reported_not_recomputed"]
    panels = [
        ("MAE", "MAE (kWh)", "mae"),
        ("RMSE", "RMSE (kWh)", "rmse"),
        ("WAPE", "WAPE (%)", "wape"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.5))
    for ax, (title, label, key) in zip(axes, panels):
        values = [float(baseline[key]), float(metrics[key])]
        bars = ax.bar(
            [0, 1],
            values,
            color=[COLORS["gray"], COLORS["blue"]],
            width=0.62,
        )
        ax.set_xticks([0, 1], ["24h baseline", "Selected model"], rotation=12, ha="right")
        ax.set_ylabel(label)
        ax.set_title(title, weight="bold")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(values) * 1.22)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.4f}" if key != "wape" else f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle(
        "Final Leakage-Safe Model vs 24h Baseline (Untouched Test)",
        y=1.03,
        fontsize=11,
        weight="bold",
    )
    _save(fig, "fig2_final_model_vs_baseline_test.png")


def figure_3_feature_importance():
    data = _load_json(RESULTS / "final" / "feature_importance.json")
    stages = data["feature_importance"]
    stage1 = stages["stage_1_classifier_gain_pct"]
    stage2 = stages["stage_2_tweedie_gain_pct"]
    features = list(stage1)
    if set(features) != set(stage2):
        raise ValueError("Stage feature importance files do not contain matching features")
    if "sessions_started" in features:
        raise ValueError("Contemporaneous sessions_started found in final importance data")

    session_features = [name for name in features if name.startswith("sessions_started")]
    same_hour_features = [name for name in features if name.startswith("same_hour_")]
    core = sorted(
        (name for name in features if name not in session_features + same_hour_features),
        key=lambda name: max(stage1[name], stage2[name]),
        reverse=True,
    )[:9]
    selected = list(dict.fromkeys(core + session_features + same_hour_features))
    selected.sort(key=lambda name: max(stage1[name], stage2[name]))
    labels = [name.replace("_", " ") for name in selected]
    values1 = [float(stage1[name]) for name in selected]
    values2 = [float(stage2[name]) for name in selected]
    positions = np.arange(len(selected))
    height = 0.36

    fig, ax = plt.subplots(figsize=(9.2, max(5.0, 0.35 * len(selected) + 1.2)))
    ax.barh(
        positions - height / 2,
        values1,
        height,
        label="Stage 1: charge classifier",
        color=COLORS["blue"],
    )
    ax.barh(
        positions + height / 2,
        values2,
        height,
        label="Stage 2: Tweedie regressor",
        color=COLORS["orange"],
    )
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Normalized gain importance (%)")
    ax.set_title(
        "Final Leakage-Safe Hurdle-XGBoost Feature Importance",
        pad=10,
        weight="bold",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(max(values1), max(values2)) * 1.13)
    ax.text(
        0,
        -0.10,
        "Bars show selected high-importance features plus every final session-arrival "
        "and same-hour demand feature.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=COLORS["gray"],
    )
    _save(fig, "fig3_feature_importance_updated.png")


def figure_4_error_regimes():
    final = _load_json(RESULTS / "final" / "metrics.json")
    metrics = final["test_metrics"]
    rows = [
        ("All hours", float(metrics["mae"])),
        ("Active hours", float(metrics["active_hour_mae"])),
        ("Top 10% demand", float(metrics["top_10pct_demand_mae"])),
        ("Top 1% demand", float(metrics["top_1pct_demand_mae"])),
    ]
    _metric_bars(
        rows,
        "Test MAE (kWh)",
        "Final Hurdle-XGBoost Error Across Demand Regimes",
        "fig4_final_model_error_by_demand_regime.png",
        "kWh",
    )


def write_readme():
    text = """# Updated paper figures

Generated by `experiments/plot_updated_figures.py` using only already-saved
results. PNG images are rendered at 300 DPI. No model was trained and the
untouched test set was not rescored.

## Figure inventory

| Figure filename | What it shows | Exact source data | Split / model | Replaces old paper figure? |
|---|---|---|---|---|
| `fig1_validation_experiment_mae.png` | Validation MAE for the seven supplied leakage-safe experiment configurations. | `../experiment_table.csv` (`MAE` rows: `baseline_24h`, `session_168h`, `session_24_168`, `session_roll_means`, `same_hour_demand`, `tweedie_power_1.2`, `threshold_tuned`). | Chronological validation; contains candidate experiments, not solely the final model. | No; experiment comparison. |
| `fig1b_validation_experiment_rmse.png` | Validation RMSE for the same seven configurations. | `../experiment_table.csv` (`RMSE` for the same rows). | Chronological validation; candidate experiments. | No; supplementary experiment comparison. |
| `fig2_final_model_vs_baseline_test.png` | Separate MAE, RMSE and WAPE panels for the 24h baseline and selected model. | `../final/metrics.json`: saved baseline values and `test_metrics`. | Baseline values previously reported; final selected model, untouched test. | Yes; use for any old final-model-vs-baseline test-performance figure. |
| `fig3_feature_importance_updated.png` | Grouped Stage 1/Stage 2 normalized gain for important demand, session-arrival and same-hour features. Every final `sessions_started*` and `same_hour_*` feature is included. | `../final/feature_importance.json`. | Final selected Hurdle-XGBoost; importance from final trained models. | **Yes; replaces old Figure 3.** |
| `fig4_final_model_error_by_demand_regime.png` | Test MAE for all, active, top-decile, and top-percentile demand hours. | `../final/metrics.json` (`test_metrics`: `mae`, `active_hour_mae`, `top_10pct_demand_mae`, `top_1pct_demand_mae`). | Final selected Hurdle-XGBoost; untouched test. | Yes; replace any old final-model error-by-regime figure. |

## Figure not generated

**Figure 5 (new temporal-fold metrics): no new temporal-CV data was available
for Figure 5.** `hyperparameter_optimization/search.json` contains two
stage-specific fold scores (classifier log loss and positive-hour regressor
MAE), not fold-wise overall Hurdle predictions with MAE, RMSE, R² and WAPE.
It is not sufficient to create the requested four-metric selected-model
temporal validation plot. No values were fabricated.

No saved `y_true` / `y_pred` time series were found; these figures therefore
show saved aggregate metrics and importance only.

## Old paper figures

- **Replace Figure 3**: its cited contemporaneous `sessions_started`
  importance is invalid for the leakage-safe model.
- **Replace old final-model test/error plots** with Figures 2 and 4 when they
  describe the superseded model.
- **Old Figure 2 (temporal CV) must not be reused as new-model validation.**
  The saved `hurdle_model_cv.ipynb` reads `feature_output/*_features.csv`;
  every such CSV currently contains a `sessions_started` column, and the
  notebook's training drop-list does not exclude it. The notebook's CV uses
  Tweedie `p = 1.3` and threshold `0.50`, not the selected final configuration.
- The same precomputed feature table is the input to the notebook's pooled
  dataset evaluations. Treat LODO (old Figure 1), ACN (old Figure 4), and
  pooled multi-dataset results (old Figure 6) as potentially leakage-affected
  if they came from those feature files. Retain them only if their independent
  source code/data proves that current-hour sessions were excluded. Their
  manuscript-specific provenance is not in the repository, so this cannot
  establish which published panel used which run.
- The paper source was not found in this repository, so no manuscript figures
  were replaced in place.

## Warnings / provenance

- Final-model description: Tweedie `p = 1.2`, hurdle threshold `0.75`, 500
  trees, seed `42`.
- The updated feature-importance plot uses `sessions_started_lag168h` and
  historical rolling features. It excludes contemporaneous `sessions_started`.
- Figure 1 / 1b show validation scores; Figures 2 / 4 show test metrics. The
  split types are not combined on one graph.
- `data.ipynb` contains leakage-safe updated feature-building code, but its
  stored outputs still show a `sessions_started` column. Re-execute/clear those
  stale notebook outputs before using them as evidence.
- Do not cite the old `hurdle_model_cv.ipynb` output as the final selected
  model: it uses a feature table containing contemporaneous `sessions_started`,
  Tweedie `p = 1.3`, and threshold `0.50`; its saved test metrics are from that
  older run, not this selected model.
- The saved final artifact metadata under the separate simulation `artifacts/`
  directory describes a different model configuration. It was not used to
  generate these selected-model paper figures.
"""
    (FIGURES / "README.md").write_text(text, encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    _set_style()
    table = _load_experiment_table()
    figure_1_experiment_comparison(table)
    figure_2_final_vs_baseline()
    figure_3_feature_importance()
    figure_4_error_regimes()
    write_readme()
    for path in sorted(FIGURES.iterdir()):
        if path.is_file():
            print(f"Created: {path}")


if __name__ == "__main__":
    main()
