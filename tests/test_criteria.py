"""Criterion tests: sklearn parity for value, FD checks for hypergradient."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sparho import L1, LogisticLoss, Problem, SquaredLoss, WeightedL1
from sparho.adapters import (
    SklearnLasso,
    SklearnLogisticRegression,
    SklearnWeightedLasso,
)
from sparho.criteria import (
    Criterion,
    CrossVal,
    HeldOutLogistic,
    HeldOutMSE,
    Sure,
)
from sparho.hypergrad import implicit_forward

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def reg_problem_and_split():
    X, y = make_regression(n_samples=200, n_features=50, n_informative=10, random_state=0)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    rng = np.random.default_rng(0)
    idx = np.arange(200)
    rng.shuffle(idx)
    idx_train = idx[:140].astype(np.int32)
    idx_val = idx[140:].astype(np.int32)
    return Problem(SquaredLoss(), L1(), X, y), idx_train, idx_val


@pytest.fixture(scope="module")
def cls_problem_and_split():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 30))
    true_beta = np.zeros(30)
    true_beta[:5] = [2.0, -2.0, 1.5, -1.5, 1.0]
    logits = X @ true_beta
    y = np.where(logits > 0, 1.0, -1.0)
    idx = np.arange(300)
    rng.shuffle(idx)
    idx_train = idx[:200].astype(np.int32)
    idx_val = idx[200:].astype(np.int32)
    return Problem(LogisticLoss(), L1(), X, y), idx_train, idx_val


# ---------------------------------------------------------------- protocol


def test_held_out_mse_satisfies_criterion_protocol(reg_problem_and_split):
    _, idx_tr, idx_val = reg_problem_and_split
    assert isinstance(HeldOutMSE(idx_tr, idx_val), Criterion)


def test_held_out_logistic_satisfies_criterion_protocol(cls_problem_and_split):
    _, idx_tr, idx_val = cls_problem_and_split
    assert isinstance(HeldOutLogistic(idx_tr, idx_val), Criterion)


def test_cross_val_satisfies_criterion_protocol():
    cv = CrossVal.kfold(100, k=3, shuffle=False)
    assert isinstance(cv, Criterion)


# ---------------------------------------------------------------- HeldOutMSE


def test_held_out_mse_value_matches_sklearn(reg_problem_and_split):
    problem, idx_tr, idx_val = reg_problem_and_split
    alpha = 0.1
    crit = HeldOutMSE(idx_tr, idx_val)
    solver = SklearnLasso(tol=1e-12, max_iter=100_000)
    value = crit.value(problem, alpha, solver)
    # Direct sklearn computation.
    est = SkLasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=100_000)
    est.fit(problem.design[idx_tr], problem.target[idx_tr])
    expected = mean_squared_error(problem.target[idx_val], problem.design[idx_val] @ est.coef_)
    assert value == pytest.approx(expected, rel=1e-9)


def test_held_out_mse_hypergrad_finite_difference(reg_problem_and_split):
    problem, idx_tr, idx_val = reg_problem_and_split
    alpha = 0.1
    eps = 1e-5
    crit = HeldOutMSE(idx_tr, idx_val)
    solver = SklearnLasso(tol=1e-12, max_iter=100_000)
    result = crit.value_and_hypergrad(problem, alpha, solver, implicit_forward)
    v_plus = crit.value(problem, alpha + eps, solver)
    v_minus = crit.value(problem, alpha - eps, solver)
    fd = (v_plus - v_minus) / (2 * eps)
    assert result.hypergrad == pytest.approx(fd, rel=1e-3, abs=1e-4)
    # Coef and active set should mirror the converged inner solve.
    np.testing.assert_array_equal(result.active_set, np.flatnonzero(result.coef).astype(np.int32))


# ---------------------------------------------------------------- HeldOutLogistic


def test_held_out_logistic_value_matches_formula(cls_problem_and_split):
    problem, idx_tr, idx_val = cls_problem_and_split
    alpha = 0.1
    crit = HeldOutLogistic(idx_tr, idx_val)
    solver = SklearnLogisticRegression(tol=1e-10, max_iter=100_000)
    value = crit.value(problem, alpha, solver)
    # Recompute by hand using the adapter directly on the same subset.
    train_problem = Problem(
        problem.datafit, problem.penalty, problem.design[idx_tr], problem.target[idx_tr]
    )
    result_sk = solver(train_problem, alpha)
    y_val = problem.target[idx_val]
    Xb = problem.design[idx_val] @ result_sk.coef
    expected = float(np.mean(np.logaddexp(0.0, -y_val * Xb)))
    # liblinear runs aren't bit-reproducible across two fits at the same tol;
    # a 1e-6 relative match is the tightest we can rely on.
    assert value == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------- CrossVal


def test_cross_val_value_averages_over_folds(reg_problem_and_split):
    problem, _, _ = reg_problem_and_split
    cv = CrossVal.kfold(problem.n_samples, k=4, shuffle=False)
    solver = SklearnLasso(tol=1e-10, max_iter=100_000)
    value = cv.value(problem, 0.1, solver)
    # Compute the same K HeldOutMSE evaluations manually.
    total = 0.0
    for idx_tr, idx_val in cv.folds:
        total += HeldOutMSE(idx_tr, idx_val).value(problem, 0.1, solver)
    expected = total / len(cv.folds)
    assert value == pytest.approx(expected, rel=1e-12)


def test_cross_val_matches_lasso_cv(reg_problem_and_split):
    """At the α selected by sklearn ``LassoCV``, our CrossVal value matches its
    reported mean MSE within ``rtol=1e-4``."""
    problem, _, _ = reg_problem_and_split
    alphas = np.logspace(-2, 0, 5)
    sk_cv = LassoCV(
        alphas=alphas,
        cv=KFold(5, shuffle=False),
        fit_intercept=False,
        tol=1e-10,
        max_iter=100_000,
    )
    sk_cv.fit(problem.design, problem.target)
    alpha_best = float(sk_cv.alpha_)
    # sklearn's ``mse_path_`` is (n_alphas, n_folds); rows ordered by descending α.
    i_best = int(np.argmin(sk_cv.mse_path_.mean(axis=1)))
    mean_mse_at_best = float(sk_cv.mse_path_[i_best].mean())
    cv = CrossVal.kfold(problem.n_samples, k=5, shuffle=False)
    solver = SklearnLasso(tol=1e-10, max_iter=100_000)
    our_value = cv.value(problem, alpha_best, solver)
    assert our_value == pytest.approx(mean_mse_at_best, rel=1e-4)


def test_cross_val_hypergrad_finite_difference(reg_problem_and_split):
    problem, _, _ = reg_problem_and_split
    cv = CrossVal.kfold(problem.n_samples, k=4, shuffle=False)
    solver = SklearnLasso(tol=1e-12, max_iter=100_000)
    alpha = 0.1
    eps = 1e-5
    result = cv.value_and_hypergrad(problem, alpha, solver, implicit_forward)
    v_plus = cv.value(problem, alpha + eps, solver)
    v_minus = cv.value(problem, alpha - eps, solver)
    fd = (v_plus - v_minus) / (2 * eps)
    assert result.hypergrad == pytest.approx(fd, rel=1e-3, abs=1e-3)


# ---------------------------------------------------------------- end-to-end


def test_held_out_mse_classification_with_logistic_via_held_out_logistic(cls_problem_and_split):
    problem, idx_tr, idx_val = cls_problem_and_split
    crit = HeldOutLogistic(idx_tr, idx_val)
    solver = SklearnLogisticRegression(tol=1e-10, max_iter=100_000)
    result = crit.value_and_hypergrad(problem, 0.1, solver, implicit_forward)
    # Sanity bounds.
    assert 0.0 <= result.value <= np.log(2.0) * 1.1  # ≤ uniform-noise upper bound (loose)
    assert result.coef.shape == (problem.n_features,)
    assert result.active_set.dtype == np.int32


# ---------------------------------------------------------------- Sure


@pytest.fixture(scope="module")
def denoising_problem():
    """Sparse denoising: y = Xβ* + σ·noise, known σ, no train/val split."""
    rng = np.random.default_rng(2)
    n, p, k = 120, 80, 8
    X = rng.standard_normal((n, p)) / np.sqrt(n)
    beta_star = np.zeros(p)
    support = rng.choice(p, size=k, replace=False)
    beta_star[support] = rng.choice([-1.0, 1.0], size=k) * (1.0 + rng.random(k))
    sigma = 0.1
    y_clean = X @ beta_star
    y = y_clean + sigma * rng.standard_normal(n)
    return Problem(SquaredLoss(), L1(), X, y), beta_star, sigma


def test_sure_satisfies_criterion_protocol():
    assert isinstance(Sure(sigma=1.0), Criterion)


def test_sure_rejects_non_squared_loss(cls_problem_and_split):
    problem, _, _ = cls_problem_and_split
    solver = SklearnLogisticRegression(tol=1e-6, max_iter=10_000)
    crit = Sure(sigma=0.1)
    with pytest.raises(TypeError, match="SquaredLoss"):
        crit.value(problem, 0.1, solver)
    with pytest.raises(TypeError, match="SquaredLoss"):
        crit.value_and_hypergrad(problem, 0.1, solver, implicit_forward)


def test_sure_value_reproducible_under_same_seed(denoising_problem):
    problem, _, sigma = denoising_problem
    solver = SklearnLasso(tol=1e-10, max_iter=100_000)
    # α below α_max ≈ 0.023 on this fixture; matches the DOF-concentration
    # test below. At α=0.05 (above α_max) β̂ ≡ 0 and the FDMC DOF term
    # vanishes identically, making any seed-dependent SURE test trivially
    # pass on Linux and stochastically pass on macOS — see Land v0.9 fix.
    alpha = 5e-3
    v1 = Sure(sigma=sigma, random_state=7).value(problem, alpha, solver)
    v2 = Sure(sigma=sigma, random_state=7).value(problem, alpha, solver)
    assert v1 == pytest.approx(v2, rel=1e-12)


def test_sure_value_differs_across_seeds(denoising_problem):
    problem, _, sigma = denoising_problem
    solver = SklearnLasso(tol=1e-10, max_iter=100_000)
    # α below α_max so β̂ has a non-trivial active set; otherwise the FDMC
    # DOF term collapses to zero for every δ and std == 0 exactly.
    alpha = 5e-3
    values = [Sure(sigma=sigma, random_state=s).value(problem, alpha, solver) for s in range(5)]
    # Different δ probes ⇒ MC variance in the FDMC DOF estimate. Variance
    # should be small (sigma is small, ε is small) but strictly nonzero.
    assert float(np.std(values)) > 0.0


def test_sure_fdmc_dof_concentrates_at_lasso_support_size(denoising_problem):
    """For SquaredLoss + L1, closed-form DOF (Zou-Hui-Tibshirani 2007) equals
    |support(β̂)|. The FDMC estimator with a single δ has mean |A| and variance
    ~2|A| under Gaussian δ; averaging over many seeds should concentrate.
    """
    problem, _, sigma = denoising_problem
    solver = SklearnLasso(tol=1e-10, max_iter=100_000)
    # Pick an α small enough to give a non-trivial active set on this fixture.
    alpha = 5e-3
    # Solve once to get the closed-form DOF reference.
    r1 = solver(problem, alpha)
    dof_closed_form = int(np.flatnonzero(r1.coef).size)
    assert dof_closed_form > 0, "test fixture mis-tuned: empty active set at α"

    # Recover the FDMC DOF from the SURE value:
    #   SURE = (1/n) ‖resid‖² − σ² + (2σ²/n) · DOF_FDMC
    # ⇒ DOF_FDMC = (n / (2σ²)) · (SURE − (1/n)‖resid‖² + σ²)
    n = problem.n_samples
    resid = problem.design @ r1.coef - problem.target
    data_term = float(resid @ resid) / n

    # Use a small ε so the finite-difference approximation lives near the
    # true Jacobian limit (where DOF_FDMC = δᵀ P_A δ exactly).
    eps_fd = 1e-6
    dofs = []
    for s in range(40):
        v = Sure(sigma=sigma, epsilon=eps_fd, random_state=s).value(problem, alpha, solver)
        dofs.append((n / (2.0 * sigma**2)) * (v - data_term + sigma**2))
    mean_dof = float(np.mean(dofs))
    # Per-sample stdev ≈ √(2|A|); SE of the mean ≈ √(2|A|/M). With |A| small
    # and M=40 we should be well within 3 SE of the closed form.
    assert abs(mean_dof - dof_closed_form) < 3.0 * np.sqrt(2.0 * dof_closed_form / 40)


def test_sure_hypergrad_finite_difference(denoising_problem):
    problem, _, sigma = denoising_problem
    solver = SklearnLasso(tol=1e-12, max_iter=200_000)
    # α below α_max so β̂ has a non-trivial active set — otherwise both
    # hypergrad and FD are identically zero and the test passes vacuously.
    alpha = 5e-3
    eps = 1e-5
    crit = Sure(sigma=sigma, random_state=0)
    # Prime the probe so all three calls reuse the same δ → consistent FD.
    result = crit.value_and_hypergrad(problem, alpha, solver, implicit_forward)
    v_plus = crit.value(problem, alpha + eps, solver)
    v_minus = crit.value(problem, alpha - eps, solver)
    fd = (v_plus - v_minus) / (2 * eps)
    assert result.hypergrad == pytest.approx(fd, rel=5e-3, abs=5e-4)


def test_sure_warm_start_matches_cold_start(denoising_problem):
    """Lasso is convex; warm-started inner solves must converge to the same
    SURE value as cold-started ones at the same α."""
    problem, _, sigma = denoising_problem
    solver = SklearnLasso(tol=1e-10, max_iter=200_000)
    cold = Sure(sigma=sigma, random_state=3, warm_start=False)
    warm = Sure(sigma=sigma, random_state=3, warm_start=True)
    # Both αs below α_max ≈ 0.023 so β̂ is non-trivial; otherwise warm and
    # cold trivially agree at β̂ ≡ 0 and the test never exercises warm-start.
    _ = warm.value(problem, 1e-2, solver)
    v_cold = cold.value(problem, 5e-3, solver)
    v_warm = warm.value(problem, 5e-3, solver)
    assert v_warm == pytest.approx(v_cold, rel=1e-6)


def test_sure_weighted_l1_returns_vector_hypergrad():
    """Vector α (WeightedL1) ⇒ vector hypergradient with the right shape."""
    rng = np.random.default_rng(4)
    n, p = 60, 20
    X = rng.standard_normal((n, p)) / np.sqrt(n)
    beta_star = np.zeros(p)
    beta_star[:3] = [1.5, -1.5, 1.0]
    sigma = 0.1
    y = X @ beta_star + sigma * rng.standard_normal(n)
    problem = Problem(SquaredLoss(), WeightedL1(), X, y)
    alpha_vec = 0.05 * np.ones(p, dtype=np.float64)
    solver = SklearnWeightedLasso(tol=1e-10, max_iter=100_000)
    result = Sure(sigma=sigma, random_state=11).value_and_hypergrad(
        problem, alpha_vec, solver, implicit_forward
    )
    assert isinstance(result.hypergrad, np.ndarray)
    assert result.hypergrad.shape == (p,)


def test_sure_recovers_near_oracle_alpha_on_denoising(denoising_problem):
    """The α* selected by minimizing SURE should match the α* selected by
    minimizing the true (oracle) prediction MSE within an order of magnitude.
    This is the actual reason SURE exists; the test is intentionally loose
    to avoid being brittle on a single MC realization."""
    problem, beta_star, sigma = denoising_problem
    solver = SklearnLasso(tol=1e-10, max_iter=100_000)
    crit = Sure(sigma=sigma, random_state=0)
    alphas = np.logspace(-3, 0, 20)
    sure_values = np.array([crit.value(problem, float(a), solver) for a in alphas])
    # Oracle: true held-out MSE using the noise-free signal Xβ*.
    n = problem.n_samples
    y_clean = problem.design @ beta_star
    oracle_mse = np.empty(len(alphas))
    for i, a in enumerate(alphas):
        coef = solver(problem, float(a)).coef
        err = problem.design @ coef - y_clean
        oracle_mse[i] = float(err @ err) / n
    a_sure = float(alphas[int(np.argmin(sure_values))])
    a_oracle = float(alphas[int(np.argmin(oracle_mse))])
    assert abs(np.log10(a_sure) - np.log10(a_oracle)) < 1.0
