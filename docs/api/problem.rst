Problem definition
==================

.. currentmodule:: sparho

The bilevel problem is a single ``Problem`` dataclass that carries the
datafit + penalty + design + target. The hyperparameter ``α`` is **not**
stored — that's what the outer search tunes.

The ``Datafit`` and ``Penalty`` families are tagged unions of frozen
dataclasses. Algorithms dispatch on them via ``match`` statements with
``typing.assert_never`` exhaustiveness; mypy will flag any missing case.

.. autoclass:: Problem
   :members:

Datafits
--------

.. autoclass:: SquaredLoss
.. autoclass:: LogisticLoss

.. autodata:: Datafit
   :annotation: = SquaredLoss | LogisticLoss

Penalties
---------

.. autoclass:: L1
.. autoclass:: ElasticNet
.. autoclass:: WeightedL1

.. autodata:: Penalty
   :annotation: = L1 | ElasticNet | WeightedL1
