"""Loads the trained hurdle model and serves predictions.

The column order is read from artifacts/columns.json and applied with
design_matrix(..., columns=...). Re-deriving it at serve time with get_dummies
can silently reorder or drop features.
"""

import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

import pipeline as P

ART = Path("artifacts")


class ModelService:
    def __init__(self, art=ART):
        self.art = Path(art)
        missing = [f for f in ("clf.ubj", "reg.ubj", "columns.json", "meta.json")
                   if not (self.art / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifacts {missing} in {self.art}/. Run: python3 train_for_sim.py"
            )
        self.clf = XGBClassifier()
        self.clf.load_model(self.art / "clf.ubj")
        self.reg = XGBRegressor()
        self.reg.load_model(self.art / "reg.ubj")
        self.columns = json.loads((self.art / "columns.json").read_text())
        self.meta = json.loads((self.art / "meta.json").read_text())
        self.entity_stats = pd.read_csv(self.art / "entity_stats.csv")
        self.global_mean = float(self.meta["global_mean"])

    @property
    def data_source(self):
        return self.meta.get("data_source", "unknown")

    def predict(self, feature_rows: pd.DataFrame):
        """feature_rows: output of the shared pipeline, one row per entity."""
        df = P.attach_entity_stats(feature_rows, self.entity_stats, self.global_mean)
        X = P.design_matrix(df, columns=self.columns)
        pred, prob = P.hurdle_predict(self.clf, self.reg, X)
        return pred, prob
