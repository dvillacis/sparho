sklearn-compatible wrappers
===========================

.. currentmodule:: sparho

The wrappers expose sparho's bilevel α-tuning behind the
``BaseEstimator + Mixin`` API the sklearn ecosystem expects. They unlock
``Pipeline`` composition, ``GridSearchCV`` over structural parameters (e.g.
ElasticNet's ``rho``), ``cross_val_score``, ``clone``,
``permutation_importance``, MLflow autolog, and EconML / DoubleML
integration.

.. autoclass:: LassoHO
   :members:

.. autoclass:: ElasticNetHO
   :members:

.. autoclass:: LogisticRegressionHO
   :members:
