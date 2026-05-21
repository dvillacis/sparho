"""Pickle + sklearn.clone round-trip tests for the wrapper estimators.

`check_estimator` covers this transitively, but we keep an explicit suite
because:
- Pickle support is what makes `Pipeline`, `GridSearchCV`, and MLflow
  autolog work — a regression here breaks every downstream integration.
- We exercise the fit-then-pickle path (covered by check_estimator) *and*
  the unfit-then-pickle path (not covered) to catch state introduced by
  `__init__`.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sparho import ElasticNetHO, LassoHO, LogisticRegressionHO

_RNG = np.random.default_rng(0)


def _regression_data(n_samples: int = 60, n_features: int = 8):
    X = _RNG.standard_normal((n_samples, n_features))
    beta_true = np.zeros(n_features)
    beta_true[:3] = [1.0, -0.5, 0.7]
    y = X @ beta_true + 0.1 * _RNG.standard_normal(n_samples)
    return X, y


def _classification_data(n_samples: int = 60, n_features: int = 8):
    X = _RNG.standard_normal((n_samples, n_features))
    logits = X[:, 0] - 0.5 * X[:, 1]
    y = (logits > 0).astype(int)
    return X, y


@pytest.mark.parametrize(
    "factory, data_fn",
    [
        (lambda: LassoHO(alpha_init=0.1, n_iter=5), _regression_data),
        (lambda: ElasticNetHO(alpha_init=0.1, rho=0.5, n_iter=5), _regression_data),
        (lambda: LogisticRegressionHO(alpha_init=0.1, n_iter=5), _classification_data),
    ],
    ids=["lasso", "elasticnet", "logistic"],
)
def test_unfit_pickle_round_trip(factory, data_fn):
    """An unfit estimator pickles and the loaded copy still fits."""
    est = factory()
    blob = pickle.dumps(est)
    est2 = pickle.loads(blob)
    assert est.get_params() == est2.get_params()
    X, y = data_fn()
    est2.fit(X, y)  # must not crash
    assert hasattr(est2, "coef_")


@pytest.mark.parametrize(
    "factory, data_fn",
    [
        (lambda: LassoHO(alpha_init=0.1, n_iter=5), _regression_data),
        (lambda: ElasticNetHO(alpha_init=0.1, rho=0.5, n_iter=5), _regression_data),
        (lambda: LogisticRegressionHO(alpha_init=0.1, n_iter=5), _classification_data),
    ],
    ids=["lasso", "elasticnet", "logistic"],
)
def test_fit_pickle_round_trip(factory, data_fn):
    """Fitted estimator survives pickle: coef_/alpha_/n_iter_ preserved; predict() unchanged."""
    est = factory()
    X, y = data_fn()
    est.fit(X, y)
    blob = pickle.dumps(est)
    est2 = pickle.loads(blob)
    np.testing.assert_array_equal(est.coef_, est2.coef_)
    assert est.alpha_ == est2.alpha_
    assert est.n_iter_ == est2.n_iter_
    np.testing.assert_array_equal(est.predict(X), est2.predict(X))


@pytest.mark.parametrize(
    "factory, data_fn",
    [
        (lambda: LassoHO(alpha_init=0.1, n_iter=5), _regression_data),
        (lambda: ElasticNetHO(alpha_init=0.1, rho=0.5, n_iter=5), _regression_data),
        (lambda: LogisticRegressionHO(alpha_init=0.1, n_iter=5), _classification_data),
    ],
    ids=["lasso", "elasticnet", "logistic"],
)
def test_clone_returns_unfit_copy_with_same_params(factory, data_fn):
    """`sklearn.clone` produces an unfit copy with the same constructor params."""
    est = factory()
    X, y = data_fn()
    est.fit(X, y)
    twin = clone(est)
    assert est.get_params() == twin.get_params()
    # `clone` strips fitted attrs.
    assert not hasattr(twin, "coef_")
    # Independently fittable.
    twin.fit(X, y)
    assert hasattr(twin, "coef_")
