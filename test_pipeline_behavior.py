import numpy as np
import pandas as pd

import pipeline as P


class DummyClassifier:
    def predict_proba(self, X):
        return np.array([
            [0.8, 0.2],
            [0.1, 0.9],
        ])


class DummyRegressor:
    def predict(self, X):
        return np.array([5.0, 7.0])


def test_hurdle_predict_uses_expected_load():
    pred, prob = P.hurdle_predict(DummyClassifier(), DummyRegressor(), np.zeros((2, 2)))
    np.testing.assert_allclose(prob, np.array([0.2, 0.9]))
    np.testing.assert_allclose(pred, np.array([1.0, 6.3]))


def test_design_matrix_keeps_entity_id_dummies():
    df = pd.DataFrame(
        {
            "hour": pd.to_datetime(["2020-01-01T00:00Z", "2020-01-01T01:00Z"]),
            "source_dataset": ["paris", "paris"],
            "region": ["France", "France"],
            "entity_id": ["paris_a", "paris_b"],
            "site": ["site_a", "site_b"],
            "kwh_allocated": [1.0, 2.0],
            "target": [1.0, 2.0],
        }
    )
    X = P.design_matrix(df)
    entity_cols = [c for c in X.columns if c.startswith("entity_id_")]
    assert entity_cols
