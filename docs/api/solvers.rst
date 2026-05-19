Solvers
=======

.. currentmodule:: sparho

A ``Solver`` is anything callable with the signature
``(Problem, Hyperparam, *, x0=None, tol=None) -> SolverResult``. It is the
boundary between sparho's bilevel machinery and a concrete inner-problem
fitter (sklearn, celer, or a user callable).

.. autoclass:: Solver
   :members: __call__
   :special-members: __call__

sklearn adapters
----------------

.. currentmodule:: sparho.adapters

.. autoclass:: SklearnLasso
   :members: __call__
.. autoclass:: SklearnElasticNet
   :members: __call__
.. autoclass:: SklearnWeightedLasso
   :members: __call__
.. autoclass:: SklearnLogisticRegression
   :members: __call__

celer adapters
--------------

``celer`` adapters are available behind the ``[celer]`` extra. The
``celer`` package is imported lazily so the module loads even when the
extra is not installed.

.. currentmodule:: sparho.adapters.celer

.. autoclass:: CelerLasso
   :members: __call__
.. autoclass:: CelerElasticNet
   :members: __call__

Wrapping an arbitrary callable
------------------------------

.. currentmodule:: sparho.adapters

.. autofunction:: as_solver
