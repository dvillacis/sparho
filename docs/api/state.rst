Result and state dataclasses
============================

.. currentmodule:: sparho

All result types are ``frozen=True, slots=True`` dataclasses. The outer
search history is an immutable ``tuple[IterationRecord, ...]`` — there is
no mutable monitor.

.. autoclass:: SolverResult
   :members:

.. autoclass:: IterationRecord
   :members:

.. autoclass:: SearchState
   :members:

.. autoclass:: SearchResult
   :members:
