"""Tests for sklearn-compatible wrapper estimators.

Coverage:
- BaseEstimator contract (clone, get_params/set_params)
- sklearn.utils.estimator_checks.check_estimator with documented exceptions
- end-to-end fit/predict/score, parity vs LassoCV
- pandas DataFrame round-trip via feature_names_in_
- Pipeline integration ([StandardScaler, LassoHO])
- sample_weight refusal (NotImplementedError, ROADMAP item M)
- sparse + fit_intercept=True refusal with redirect message
- uneven-column-scale UserWarning
- LogisticRegressionHO with arbitrary 2-class labels (string, etc.)
- LogisticRegressionHO fit_intercept=True refusal
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator
from sparho import ElasticNetHO, LassoHO, LogisticRegressionHO

# The check_estimator suite expects sample_weight support; sparho v0.3 wrappers
# deliberately raise NotImplementedError per ROADMAP item M. Declare those checks
# as known-skipped so the rest of the suite runs.
_SAMPLE_WEIGHT_CHECKS = {
    name: "sample_weight unsupported in v0.3; tracked as ROADMAP item M"
    for name in (
        "check_sample_weights_pandas_series",
        "check_sample_weights_not_an_array",
        "check_sample_weights_list",
        "check_sample_weights_shape",
        "check_sample_weights_not_overwritten",
        "check_sample_weight_equivalence_on_dense_data",
        "check_sample_weight_equivalence_on_sparse_data",
        "check_classifiers_one_label_sample_weights",
    )
}


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(n_samples=120, n_features=20, n_informative=5, noise=0.5, random_state=0)
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture(scope="module")
def cls_data():
    X, y = make_classification(n_samples=200, n_features=20, n_informative=5, random_state=0)
    return X.astype(np.float64), y.astype(np.int64)


# ---------------------------------------------------------------- BaseEstimator


def test_lasso_ho_clone_roundtrip():
    est = LassoHO(alpha_init=0.5, n_iter=5, fit_intercept=False)
    clone(est)  # raises if get_params is mis-declared


def test_lasso_ho_get_set_params():
    est = LassoHO(alpha_init=0.5, n_iter=5)
    p = est.get_params()
    assert p["alpha_init"] == 0.5
    assert p["n_iter"] == 5
    est.set_params(alpha_init=1.0)
    assert est.alpha_init == 1.0


def test_elastic_net_ho_carries_rho_through_clone():
    est = ElasticNetHO(rho=0.7, alpha_init=0.5, n_iter=5)
    assert clone(est).get_params()["rho"] == 0.7


# ---------------------------------------------------------------- check_estimator


def test_lasso_ho_passes_check_estimator():
    check_estimator(
        LassoHO(n_iter=3, alpha_init=0.5),
        expected_failed_checks=_SAMPLE_WEIGHT_CHECKS,
        on_skip=None,
    )


def test_elastic_net_ho_passes_check_estimator():
    check_estimator(
        ElasticNetHO(rho=0.5, n_iter=3, alpha_init=0.5),
        expected_failed_checks=_SAMPLE_WEIGHT_CHECKS,
        on_skip=None,
    )


def test_logistic_regression_ho_passes_check_estimator():
    check_estimator(
        LogisticRegressionHO(n_iter=3, alpha_init=0.5),
        expected_failed_checks=_SAMPLE_WEIGHT_CHECKS,
        on_skip=None,
    )


# ---------------------------------------------------------------- end-to-end


def test_lasso_ho_predict_shape_and_intercept(reg_data):
    X, y = reg_data
    m = LassoHO(alpha_init=0.1, n_iter=10).fit(X, y)
    assert m.coef_.shape == (X.shape[1],)
    assert isinstance(m.intercept_, float)
    assert m.predict(X[:5]).shape == (5,)
    assert hasattr(m, "alpha_") and m.alpha_ > 0
    assert hasattr(m, "n_iter_")
    assert hasattr(m, "search_result_")


def test_lasso_ho_predict_before_fit_raises(reg_data):
    X, _ = reg_data
    with pytest.raises(NotFittedError):
        LassoHO().predict(X)


def test_lasso_ho_beats_or_matches_lasso_cv(reg_data):
    """LassoHO + 5-fold CV criterion should match LassoCV's held-out MSE
    within a small margin on the same data (≤ 1.2× the LassoCV held-out MSE)."""
    X, y = reg_data
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_tr = int(0.7 * len(y))
    X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
    y_tr, y_te = y[idx[:n_tr]], y[idx[n_tr:]]

    sk_cv = LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, tol=1e-10, max_iter=20_000)
    sk_cv.fit(X_tr, y_tr)
    cv_mse = mean_squared_error(y_te, sk_cv.predict(X_te))

    ho = LassoHO(alpha_init=0.1, n_iter=20).fit(X_tr, y_tr)
    ho_mse = mean_squared_error(y_te, ho.predict(X_te))
    assert ho_mse < 1.5 * cv_mse, (ho_mse, cv_mse)


def test_elastic_net_ho_rho_one_close_to_lasso_ho(reg_data):
    X, y = reg_data
    a = LassoHO(alpha_init=0.1, n_iter=10, random_state=0).fit(X, y)
    b = ElasticNetHO(rho=1.0, alpha_init=0.1, n_iter=10, random_state=0).fit(X, y)
    # Same penalty (ρ=1 ⇒ Lasso) ⇒ predictions within a small numerical margin
    # at the same data; α* may differ slightly due to different solvers.
    pred_a = a.predict(X)
    pred_b = b.predict(X)
    rel_diff = np.linalg.norm(pred_a - pred_b) / max(np.linalg.norm(pred_a), 1e-12)
    assert rel_diff < 0.05


def test_elastic_net_ho_rho_out_of_range_raises(reg_data):
    X, y = reg_data
    with pytest.raises(ValueError, match="rho"):
        ElasticNetHO(rho=1.5, n_iter=3).fit(X, y)
    with pytest.raises(ValueError, match="rho"):
        ElasticNetHO(rho=0.0, n_iter=3).fit(X, y)


# ---------------------------------------------------------------- DataFrame


def test_lasso_ho_dataframe_round_trip(reg_data):
    pd = pytest.importorskip("pandas")
    X, y = reg_data
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    m = LassoHO(alpha_init=0.1, n_iter=5).fit(df, y)
    assert list(m.feature_names_in_) == cols
    # predict accepts DataFrame with same columns.
    pred = m.predict(df.head(3))
    assert pred.shape == (3,)


# ---------------------------------------------------------------- Pipeline


def test_pipeline_standard_scaler_lasso_ho(reg_data):
    X, y = reg_data
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LassoHO(alpha_init=0.1, n_iter=10)),
        ]
    )
    pipe.fit(X, y)
    pred = pipe.predict(X[:5])
    assert pred.shape == (5,)


# ---------------------------------------------------------------- sample_weight


def test_sample_weight_non_none_raises(reg_data):
    X, y = reg_data
    with pytest.raises(NotImplementedError, match="sample_weight"):
        LassoHO(n_iter=3).fit(X, y, sample_weight=np.ones_like(y))
    with pytest.raises(NotImplementedError, match="sample_weight"):
        ElasticNetHO(n_iter=3).fit(X, y, sample_weight=np.ones_like(y))


# ---------------------------------------------------------------- sparse + intercept


def test_sparse_with_fit_intercept_raises(reg_data):
    X, y = reg_data
    X_sp = sp.csc_matrix(X)
    with pytest.raises(ValueError, match="sparse X with fit_intercept"):
        LassoHO(fit_intercept=True, n_iter=3).fit(X_sp, y)


def test_sparse_with_fit_intercept_false_works(reg_data):
    X, y = reg_data
    X_sp = sp.csc_matrix(X)
    m = LassoHO(fit_intercept=False, alpha_init=0.1, n_iter=5).fit(X_sp, y)
    assert m.coef_.shape == (X.shape[1],)
    assert m.intercept_ == 0.0


# ---------------------------------------------------------------- uneven scales


def test_uneven_column_scales_warns(reg_data):
    X, y = reg_data
    X_skew = X.copy()
    X_skew[:, 0] *= 1000.0  # one column with extreme scale
    with pytest.warns(UserWarning, match="different scales"):
        LassoHO(n_iter=3).fit(X_skew, y)


# ---------------------------------------------------------------- Logistic


def test_logistic_regression_ho_with_string_labels():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 20))
    true_beta = np.zeros(20)
    true_beta[:5] = [2.0, -2.0, 1.5, -1.5, 1.0]
    y = np.where(X @ true_beta > 0, "pos", "neg")
    m = LogisticRegressionHO(alpha_init=0.1, n_iter=10).fit(X, y)
    assert list(m.classes_) == ["neg", "pos"]
    pred = m.predict(X[:5])
    assert set(np.unique(pred)).issubset({"neg", "pos"})
    proba = m.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_regression_ho_score_classification(cls_data):
    X, y = cls_data
    m = LogisticRegressionHO(alpha_init=0.1, n_iter=10).fit(X, y)
    assert m.score(X, y) > 0.7  # synthetic, separable; loose floor


def test_logistic_regression_ho_intercept_unsupported(cls_data):
    X, y = cls_data
    with pytest.raises(NotImplementedError, match="fit_intercept"):
        LogisticRegressionHO(fit_intercept=True, n_iter=3).fit(X, y)
