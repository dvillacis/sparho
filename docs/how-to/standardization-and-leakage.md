# Standardization and CV leakage

`sparho`'s sklearn-compatible wrappers (`LassoHO`, `ElasticNetHO`,
`LogisticRegressionHO`) deliberately ship without a `standardize=` /
`normalize=` parameter. Feature scaling is composed externally via
`sklearn.pipeline.Pipeline` and `sklearn.preprocessing.StandardScaler`.
This page covers the recommended recipe and one subtle leakage trap that
follows from putting cross-validation *inside* the wrapper while feature
scaling sits *outside*.

## Why no `standardize=` parameter

This decision (made 2026-05-20) follows sklearn's post-1.0 stance after
the `normalize=` deprecation
([sklearn#21238](https://github.com/scikit-learn/scikit-learn/issues/21238),
[sklearn#26359](https://github.com/scikit-learn/scikit-learn/issues/26359)).
Carrying a built-in `standardize=True` would:

- Replicate the historical `normalize=` ambiguity (centered vs scaled? per
  fold or per fit? before or after train/val split?).
- Make `α*` *not* directly comparable to sklearn `Lasso`'s `α*`.
- Encourage users to skip the explicit feature-engineering step that the
  sklearn ecosystem now expects as `Pipeline` composition.

The audience is sklearn-refugees, not glmnet-refugees. Users who want
glmnet-style on-by-default standardization should compose a Pipeline.

## Recipe: scale features upstream of the bilevel search

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sparho import LassoHO

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LassoHO(alpha_init=0.1, n_iter=30)),
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

When `LassoHO` warns about uneven column scales — *"Features have very
different scales..."* — this is the recommended response.

For sparse `X`, use `StandardScaler(with_mean=False)` to keep the sparse
representation, and pair with `LassoHO(fit_intercept=False)`:

```python
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("model", LassoHO(fit_intercept=False, alpha_init=0.1, n_iter=30)),
])
```

(Sparse `X` with `fit_intercept=True` is not supported in sparho v0.3 —
the wrapper raises with this exact redirect.)

## The leakage trap: nested CV inside an outer Pipeline

`LassoHO`'s default `criterion` is a 5-fold `CrossVal(HeldOutMSE)` over
the training data. When the wrapper sits *inside* a `Pipeline` that has
a `StandardScaler` upstream of it, every CV fold sees data that was
**scaled using the full training set's statistics** — not the
fold-train statistics. This is exactly the leakage `sklearn#26359`
describes for `LassoCV` inside a Pipeline.

For most use cases the leakage is small (StandardScaler is robust under
moderate fold-to-fold variation). When it matters — small `n`, heavy
tails, leakage-sensitive downstream evaluation — there are two safe
patterns:

### 1. Move scaling inside each fold via outer CV

If the goal is honest CV-based generalization estimation, wrap the
*whole pipeline* in `sklearn.model_selection.cross_validate` and let
`LassoHO` use a single held-out criterion for its α search:

```python
from sklearn.model_selection import cross_validate, KFold
from sparho import HeldOutMSE, LassoHO

# Use a fixed held-out split inside LassoHO; the outer cross_validate
# rotates the Pipeline (including the scaler) across folds.
rng = np.random.default_rng(0)
perm = rng.permutation(len(y))
n_inner = int(0.8 * len(y))
idx_train_inner, idx_val_inner = perm[:n_inner].astype(np.int32), perm[n_inner:].astype(np.int32)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LassoHO(
        criterion=HeldOutMSE(idx_train_inner, idx_val_inner),
        alpha_init=0.1,
        n_iter=30,
    )),
])
cv_scores = cross_validate(pipe, X, y, cv=KFold(5), scoring="r2")
```

### 2. Pre-scale once outside the bilevel search

If you trust your training data not to need fold-by-fold scaling
(common for genomics / EHR / finance with stable feature distributions),
scale once before the search and skip the Pipeline:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
m = LassoHO(alpha_init=0.1, n_iter=30).fit(X_scaled, y)
```

The internal `CrossVal` then sees fold-consistent scaled features and
no leakage exists.

## Recap

| Setup | α* comparable to sklearn `Lasso`? | Leakage-safe? |
|---|---|---|
| `LassoHO(fit_intercept=True)` on raw `X` | ✅ | ✅ (no scaler) |
| `Pipeline([StandardScaler, LassoHO])` | ⚠️ (α* now scaled-space) | ⚠️ (internal CV sees pre-scaled X) |
| Outer `cross_validate(Pipeline)` + internal `HeldOutMSE` | ⚠️ | ✅ |
| Manual one-time `StandardScaler.fit_transform` + `LassoHO` | ⚠️ | ✅ |

The wrapper's job is to make the bilevel search work; the Pipeline
boundary and the choice of criterion are the user's controls for
honesty.
