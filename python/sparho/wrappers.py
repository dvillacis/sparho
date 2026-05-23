"""sklearn-compatible wrapper estimators.

``LassoHO`` / ``ElasticNetHO`` / ``LogisticRegressionHO`` wrap the bilevel
search (``hoag_search`` by default) behind the ``BaseEstimator + Mixin`` API
the sklearn ecosystem expects: ``fit`` / ``predict`` / ``score`` /
``get_params`` / ``set_params``, plus the ``coef_`` / ``intercept_`` /
``alpha_`` / ``n_iter_`` / ``feature_names_in_`` fitted attributes. This is
the integration surface for ``Pipeline``, ``GridSearchCV``,
``cross_val_score``, ``clone``, ``permutation_importance``, MLflow autolog,
and EconML/DoubleML.

**Standardization.** ``fit_intercept=True`` is the default; no
``standardize=`` parameter (matches sklearn-modern after the
``normalize=`` deprecation). For feature scaling use ``Pipeline([
StandardScaler(), LassoHO()])``; see ``docs/how-to/
standardization-and-leakage.md`` for the recipe and the leakage trap that
follows from internal cross-validation seeing post-scaler folds.

**Sparse + fit_intercept.** v0.3 supports ``fit_intercept=True`` for dense
``X`` only (subtracts column means upfront). Sparse ``X`` with
``fit_intercept=True`` raises with a redirect to the
``StandardScaler(with_mean=False)`` pattern or to manual pre-centering;
the sparse-aware "offset-adjusted matvecs" path the ROADMAP commits to is
tracked as v0.4 polish — it requires plumbing ``X_mean`` through every
solver adapter and the hypergradient kernel.

**sample_weight.** Not supported in v0.3 (separate ROADMAP item M).
``fit(X, y, sample_weight=...)`` raises ``NotImplementedError`` when
``sample_weight`` is anything other than ``None``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from .adapters import (
    SklearnElasticNet,
    SklearnLasso,
    SklearnLogisticRegression,
)
from .core.types import Array, Hyperparam
from .criteria import Criterion, CrossVal, HeldOutLogistic, HeldOutMSE
from .problem import L1, LogisticLoss, Problem, SquaredLoss
from .problem import ElasticNet as ElasticNetPenalty
from .search import grad_search, hoag_search
from .solver import Solver
from .state import IterationRecord, SearchResult

OuterMethod = Literal["hoag", "grad"]

_UNEVEN_SCALE_RATIO = 10.0
_DEFAULT_CV_FOLDS = 5

_SPARSE_INTERCEPT_MSG = (
    "sparho v0.3 wrappers do not support sparse X with fit_intercept=True. "
    "Either set fit_intercept=False and pre-center your target, or wrap the "
    "estimator: Pipeline([('scaler', StandardScaler(with_mean=False)), "
    "('model', LassoHO(fit_intercept=False))]). Sparse-aware intercept "
    "centering (offset-adjusted matvecs) is tracked for v0.4."
)


# ---------------------------------------------------------------- helpers


def _center_dense(X: Array, y: Array, *, fit_intercept: bool) -> tuple[Array, Array, Array, float]:
    """Return ``(X_c, y_c, X_mean, y_mean)`` for the downstream solve.

    When ``fit_intercept`` is ``False``, returns the inputs unchanged with
    zero offsets — the inner solver sees the original problem.
    """
    if not fit_intercept:
        return X, y, np.zeros(X.shape[1], dtype=np.float64), 0.0
    X_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    return X - X_mean, y - y_mean, X_mean, y_mean


def _warn_if_uneven_scales(X: Array) -> None:
    if sp.issparse(X):
        return
    stds = np.asarray(X, dtype=np.float64).std(axis=0)
    mean_std = float(stds.mean())
    if mean_std == 0.0:
        return
    if float(np.ptp(stds)) > _UNEVEN_SCALE_RATIO * mean_std:
        warnings.warn(
            "Features have very different scales (range of column stds exceeds "
            f"{_UNEVEN_SCALE_RATIO:g}× the mean). α selection becomes scale-dependent — "
            "consider Pipeline([StandardScaler(), <estimator>]). See "
            "docs/how-to/standardization-and-leakage.md.",
            UserWarning,
            stacklevel=3,
        )


def _check_sample_weight(sample_weight: Array | None) -> None:
    if sample_weight is not None:
        raise NotImplementedError(
            "sample_weight is not supported in sparho v0.3 wrappers; it is tracked "
            "as ROADMAP item M for a future release. Pass sample_weight=None."
        )


class _VerbosePrinter:
    """Default callback wired by ``verbose > 0`` on the wrapper estimators.

    Output is plain text on ``stdout`` so it composes with notebooks and
    captured-stdout test fixtures. ``verbose=1`` prints one line per iter
    with α, criterion value, log-space gradient norm, and the
    ``cg_status`` from extras. ``verbose>=2`` also prints ``step_size`` and
    ``L_estimate`` (HOAG only — absent under ``grad_search``).
    """

    def __init__(self, level: int, *, name: str) -> None:
        self.level = int(level)
        self.name = name

    def __call__(self, record: IterationRecord) -> None:
        hp = record.hyperparam
        if isinstance(hp, np.ndarray):
            alpha_repr = f"|α|₂={float(np.linalg.norm(hp)):.4g}"
        else:
            alpha_repr = f"{float(hp):.4g}"
        cg_status = record.extras.get("cg_status", "ok")
        msg = (
            f"[{self.name}] iter {record.iteration:3d}: "
            f"α={alpha_repr}  value={record.value:.6g}  "
            f"|∇θ|={record.grad_norm:.3g}  cg={cg_status}"
        )
        if self.level >= 2:
            ss = record.extras.get("step_size")
            le = record.extras.get("L_estimate")
            if isinstance(ss, float) and isinstance(le, float):
                msg += f"  step={ss:.3g}  L={le:.3g}"
        print(msg)


def _run_search(
    problem: Problem,
    hp0: Hyperparam,
    *,
    solver: Solver,
    criterion: Criterion,
    outer: OuterMethod,
    n_iter: int,
    inner_tol: float,
    callback: Callable[[IterationRecord], None] | None = None,
) -> SearchResult:
    if outer == "hoag":
        return hoag_search(
            problem,
            hp0,
            solver=solver,
            criterion=criterion,
            n_iter=n_iter,
            inner_tol=inner_tol,
            callback=callback,
        )
    if outer == "grad":
        return grad_search(
            problem,
            hp0,
            solver=solver,
            criterion=criterion,
            n_iter=n_iter,
            tol=inner_tol,
            callback=callback,
        )
    raise ValueError(f"outer must be 'hoag' or 'grad', got {outer!r}")


def _resolve_cv(
    n_samples: int,
    *,
    cv_folds: int,
    base: Callable[..., Any],
    random_state: int | None,
) -> CrossVal:
    return CrossVal.kfold(
        n_samples,
        k=cv_folds,
        shuffle=True,
        random_state=random_state if random_state is not None else 0,
        base=base,
        warm_start=True,
    )


# ---------------------------------------------------------------- LassoHO


class LassoHO(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """Lasso with gradient-based α tuning, sklearn-compatible.

    The outer search is :func:`sparho.hoag_search` by default. The inner solver
    defaults to :class:`sparho.adapters.SklearnLasso`. The default validation
    criterion is 5-fold ``CrossVal`` over the training data.

    Parameters
    ----------
    alpha_init
        Starting α for the outer search (must be ``> 0``; the search optimizes
        in ``log α`` space).
    fit_intercept
        Center ``X`` and ``y`` before the inner solve, then reconstruct the
        intercept from the means and ``coef_``. **Dense X only** — sparse X
        with ``fit_intercept=True`` raises; use ``fit_intercept=False`` plus
        ``StandardScaler(with_mean=False)``.
    n_iter
        Maximum outer iterations.
    solver
        Inner solver; defaults to ``SklearnLasso(tol=inner_tol)``.
    criterion
        Validation oracle; defaults to a 5-fold ``CrossVal(HeldOutMSE,
        warm_start=True)``.
    outer
        ``"hoag"`` (default) or ``"grad"``.
    inner_tol
        Inner-solver tolerance. Threaded through to the default solver and
        the HOAG schedule.
    cv_folds
        Number of folds for the default ``CrossVal``. Ignored if ``criterion``
        is supplied.
    random_state
        Seed for the default ``CrossVal`` fold split. Ignored if ``criterion``
        is supplied.
    verbose
        ``0`` (default) is silent. ``1`` prints one line per outer iter via a
        default ``_VerbosePrinter`` callback wired into the search. ``2`` adds
        ``step_size`` / ``L_estimate`` to each line (HOAG only).

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    alpha_ : float
    n_iter_ : int
    feature_names_in_ : ndarray, present only when ``X`` is a DataFrame
    n_features_in_ : int
    search_result_ : :class:`sparho.SearchResult`
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        *,
        fit_intercept: bool = True,
        n_iter: int = 30,
        solver: Solver | None = None,
        criterion: Criterion | None = None,
        outer: OuterMethod = "hoag",
        inner_tol: float = 1e-6,
        cv_folds: int = _DEFAULT_CV_FOLDS,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.alpha_init = alpha_init
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
        self.solver = solver
        self.criterion = criterion
        self.outer = outer
        self.inner_tol = inner_tol
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: Array,
        y: Array,
        sample_weight: Array | None = None,
    ) -> LassoHO:
        _check_sample_weight(sample_weight)
        X_, y_ = validate_data(self, X, y, accept_sparse="csc", dtype=np.float64, y_numeric=True)
        if self.fit_intercept and sp.issparse(X_):
            raise ValueError(_SPARSE_INTERCEPT_MSG)
        _warn_if_uneven_scales(X_)
        X_c, y_c, X_mean, y_mean = _center_dense(X_, y_, fit_intercept=self.fit_intercept)
        problem = Problem(SquaredLoss(), L1(), X_c, y_c)
        solver = self.solver if self.solver is not None else SklearnLasso(tol=self.inner_tol)
        criterion = (
            self.criterion
            if self.criterion is not None
            else _resolve_cv(
                problem.n_samples,
                cv_folds=self.cv_folds,
                base=HeldOutMSE,
                random_state=self.random_state,
            )
        )
        callback = _VerbosePrinter(self.verbose, name="LassoHO") if self.verbose > 0 else None
        result = _run_search(
            problem,
            float(self.alpha_init),
            solver=solver,
            criterion=criterion,
            outer=self.outer,
            n_iter=self.n_iter,
            inner_tol=self.inner_tol,
            callback=callback,
        )
        self.coef_ = np.asarray(result.best_coef, dtype=np.float64)
        self.intercept_ = float(y_mean - X_mean @ self.coef_) if self.fit_intercept else 0.0
        self.alpha_ = float(result.best_hyperparam)
        self.n_iter_ = int(result.n_iter)
        self.search_result_ = result
        return self

    def predict(self, X: Array) -> Array:
        check_is_fitted(self, "coef_")
        X_ = validate_data(self, X, accept_sparse="csc", dtype=np.float64, reset=False)
        if sp.issparse(X_):
            yhat = np.asarray(X_ @ self.coef_).ravel()
        else:
            yhat = X_ @ self.coef_
        return np.asarray(yhat + self.intercept_, dtype=np.float64)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        # v0.3 supports dense X end-to-end; sparse needs fit_intercept=False (which
        # works but is not the default), so we declare sparse=False honestly.
        # Flipped to True in v0.4 when sparse-aware intercept centering lands.
        tags.input_tags.sparse = False
        tags.target_tags.required = True
        return tags


# ---------------------------------------------------------------- ElasticNetHO


class ElasticNetHO(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """ElasticNet with gradient-based α tuning, sklearn-compatible.

    Same shape as :class:`LassoHO` plus the structural mixing weight ``rho``
    (sklearn's ``l1_ratio``). The penalty is ``α · (ρ·‖β‖₁ + (1−ρ)/2·‖β‖²)``;
    only ``α`` is tuned by the outer search.

    Parameters
    ----------
    alpha_init, fit_intercept, n_iter, solver, criterion, outer, inner_tol, cv_folds, random_state
        See :class:`LassoHO`.
    rho
        Mixing weight ``ρ ∈ (0, 1]``. ``ρ = 1`` recovers Lasso. Structural —
        not tuned by the search.
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        *,
        rho: float = 0.5,
        fit_intercept: bool = True,
        n_iter: int = 30,
        solver: Solver | None = None,
        criterion: Criterion | None = None,
        outer: OuterMethod = "hoag",
        inner_tol: float = 1e-6,
        cv_folds: int = _DEFAULT_CV_FOLDS,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.alpha_init = alpha_init
        self.rho = rho
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
        self.solver = solver
        self.criterion = criterion
        self.outer = outer
        self.inner_tol = inner_tol
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: Array,
        y: Array,
        sample_weight: Array | None = None,
    ) -> ElasticNetHO:
        _check_sample_weight(sample_weight)
        X_, y_ = validate_data(self, X, y, accept_sparse="csc", dtype=np.float64, y_numeric=True)
        if self.fit_intercept and sp.issparse(X_):
            raise ValueError(_SPARSE_INTERCEPT_MSG)
        if not 0.0 < self.rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1]; got {self.rho}")
        _warn_if_uneven_scales(X_)
        X_c, y_c, X_mean, y_mean = _center_dense(X_, y_, fit_intercept=self.fit_intercept)
        problem = Problem(SquaredLoss(), ElasticNetPenalty(rho=float(self.rho)), X_c, y_c)
        solver = self.solver if self.solver is not None else SklearnElasticNet(tol=self.inner_tol)
        criterion = (
            self.criterion
            if self.criterion is not None
            else _resolve_cv(
                problem.n_samples,
                cv_folds=self.cv_folds,
                base=HeldOutMSE,
                random_state=self.random_state,
            )
        )
        callback = _VerbosePrinter(self.verbose, name="ElasticNetHO") if self.verbose > 0 else None
        result = _run_search(
            problem,
            float(self.alpha_init),
            solver=solver,
            criterion=criterion,
            outer=self.outer,
            n_iter=self.n_iter,
            inner_tol=self.inner_tol,
            callback=callback,
        )
        self.coef_ = np.asarray(result.best_coef, dtype=np.float64)
        self.intercept_ = float(y_mean - X_mean @ self.coef_) if self.fit_intercept else 0.0
        self.alpha_ = float(result.best_hyperparam)
        self.n_iter_ = int(result.n_iter)
        self.search_result_ = result
        return self

    def predict(self, X: Array) -> Array:
        check_is_fitted(self, "coef_")
        X_ = validate_data(self, X, accept_sparse="csc", dtype=np.float64, reset=False)
        if sp.issparse(X_):
            yhat = np.asarray(X_ @ self.coef_).ravel()
        else:
            yhat = X_ @ self.coef_
        return np.asarray(yhat + self.intercept_, dtype=np.float64)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        # v0.3 supports dense X end-to-end; sparse needs fit_intercept=False (which
        # works but is not the default), so we declare sparse=False honestly.
        # Flipped to True in v0.4 when sparse-aware intercept centering lands.
        tags.input_tags.sparse = False
        tags.target_tags.required = True
        return tags


# ---------------------------------------------------------------- LogisticRegressionHO


class LogisticRegressionHO(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Sparse-L1 logistic regression with gradient-based α tuning.

    Wraps :class:`sparho.adapters.SklearnLogisticRegression` (liblinear, L1).
    The fit accepts arbitrary 2-class labels and internally remaps to the
    ``{−1, +1}`` convention sparho's :class:`LogisticLoss` requires; predictions
    come back in the original label space via ``classes_``.

    ``fit_intercept=True`` is **not supported in v0.3** — the log-odds
    intercept is a separate degree of freedom, not recoverable from feature
    centering. Either append a constant column to ``X`` manually, or use
    ``fit_intercept=False`` (the default for this estimator).

    Parameters
    ----------
    alpha_init, n_iter, solver, criterion, outer, inner_tol, cv_folds, random_state
        See :class:`LassoHO`.
    fit_intercept
        Must be ``False`` in v0.3. Kept on the signature so future versions
        can add intercept support without breaking ``set_params`` callers.
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        *,
        fit_intercept: bool = False,
        n_iter: int = 30,
        solver: Solver | None = None,
        criterion: Criterion | None = None,
        outer: OuterMethod = "hoag",
        inner_tol: float = 1e-6,
        cv_folds: int = _DEFAULT_CV_FOLDS,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.alpha_init = alpha_init
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
        self.solver = solver
        self.criterion = criterion
        self.outer = outer
        self.inner_tol = inner_tol
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: Array,
        y: Array,
        sample_weight: Array | None = None,
    ) -> LogisticRegressionHO:
        _check_sample_weight(sample_weight)
        if self.fit_intercept:
            raise NotImplementedError(
                "LogisticRegressionHO does not support fit_intercept=True in v0.3 — "
                "the log-odds intercept is a separate degree of freedom from feature "
                "centering. Either pre-augment X with a constant column or pass "
                "fit_intercept=False."
            )
        X_, y_ = validate_data(self, X, y, accept_sparse="csc", dtype=np.float64)
        _warn_if_uneven_scales(X_)
        target_kind = type_of_target(y_, input_name="y", raise_unknown=True)
        if target_kind == "continuous":
            raise ValueError(
                "Unknown label type: continuous target passed to a classifier. "
                "LogisticRegressionHO is a classifier; use LassoHO / ElasticNetHO "
                "for regression."
            )
        if target_kind != "binary":
            raise ValueError(
                "Only binary classification is supported. "
                f"The type of the target is {target_kind}."
            )
        classes = np.unique(y_)
        if classes.size < 2:
            raise ValueError(
                "Classifier can't train when only one class is present. "
                f"Got y with a single unique value: {classes[0]!r}."
            )
        self.classes_ = classes
        # Map to {-1, +1} for sparho's LogisticLoss convention.
        y_signed = np.where(y_ == classes[1], 1.0, -1.0).astype(np.float64)
        problem = Problem(LogisticLoss(), L1(), X_, y_signed)
        solver = (
            self.solver
            if self.solver is not None
            else SklearnLogisticRegression(tol=self.inner_tol)
        )
        criterion = (
            self.criterion
            if self.criterion is not None
            else _resolve_cv(
                problem.n_samples,
                cv_folds=self.cv_folds,
                base=HeldOutLogistic,
                random_state=self.random_state,
            )
        )
        callback = (
            _VerbosePrinter(self.verbose, name="LogisticRegressionHO") if self.verbose > 0 else None
        )
        result = _run_search(
            problem,
            float(self.alpha_init),
            solver=solver,
            criterion=criterion,
            outer=self.outer,
            n_iter=self.n_iter,
            inner_tol=self.inner_tol,
            callback=callback,
        )
        self.coef_ = np.asarray(result.best_coef, dtype=np.float64).reshape(1, -1)
        self.intercept_ = np.zeros(1, dtype=np.float64)
        self.alpha_ = float(result.best_hyperparam)
        self.n_iter_ = int(result.n_iter)
        self.search_result_ = result
        return self

    def decision_function(self, X: Array) -> Array:
        check_is_fitted(self, "coef_")
        X_ = validate_data(self, X, accept_sparse="csc", dtype=np.float64, reset=False)
        coef = self.coef_.ravel()
        if sp.issparse(X_):
            scores = np.asarray(X_ @ coef).ravel()
        else:
            scores = X_ @ coef
        return np.asarray(scores, dtype=np.float64)

    def predict(self, X: Array) -> Array:
        scores = self.decision_function(X)
        return np.asarray(np.where(scores > 0, self.classes_[1], self.classes_[0]))

    def predict_proba(self, X: Array) -> Array:
        scores = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - p1, p1])

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        # Sparse path requires fit_intercept=False (the v0.3 default for the
        # classifier), so declaring sparse=True here is honest.
        tags.input_tags.sparse = True
        tags.target_tags.required = True
        tags.classifier_tags.multi_class = False
        return tags
