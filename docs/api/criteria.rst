Criteria
========

.. currentmodule:: sparho

A ``Criterion`` is the outer-loop validation oracle: it slices the full
``Problem`` into a training subproblem, drives the inner solver, and
returns the criterion value (and, when asked, the hypergradient chained
through ``implicit_forward``).

.. autoclass:: Criterion
   :members:

.. autoclass:: CriterionResult
   :members:

.. autoclass:: HeldOutMSE
   :members:

.. autoclass:: HeldOutLogistic
   :members:

.. autoclass:: CrossVal
   :members:

.. autoclass:: Sure
   :members:
