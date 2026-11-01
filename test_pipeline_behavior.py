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


def test_hurdle_predict_expected_rule_is_probability_weighted():
    """p * magnitude - the minimum-MSE point forecast, for RMSE/R2 reporting."""
    pred, prob = P.hurdle_predict(
        DummyClassifier(), DummyRegressor(), np.zeros((2, 2)), rule="expected"
    )
    np.testing.assert_allclose(prob, np.array([0.2, 0.9]))
    np.testing.assert_allclose(pred, np.array([1.0, 6.3]))


def test_hurdle_predict_threshold_rule_is_the_default():
    """The validation-tuned rule, and the one the saved artifacts were scored with."""
    pred, prob = P.hurdle_predict(
        DummyClassifier(), DummyRegressor(), np.zeros((2, 2)), charge_threshold=0.75
    )
    np.testing.assert_allclose(prob, np.array([0.2, 0.9]))
    np.testing.assert_allclose(pred, np.array([0.0, 7.0]))


def test_design_matrix_drops_entity_id():
    """entity_id is intentionally dropped (pipeline.DROP_COLUMNS).

    Keeping per-entity dummies would let the model separate the three ACN
    sites, but it also breaks at serve time the moment an unseen entity
    appears. The drop is the deliberate choice; this test pins it so the
    behaviour cannot change silently.
    """
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
    assert "entity_id" in P.DROP_COLUMNS
    assert [c for c in X.columns if c.startswith("entity_id")] == []


def test_validate_forecast_feature_leakage_fails_on_future_fields():
    df = pd.DataFrame({"future_total_energy": [1.0, 2.0], "target": [3.0, 4.0]})
    try:
        P.validate_forecast_feature_leakage(df)
        raise AssertionError("Expected ValueError for future leakage fields")
    except ValueError:
        pass
