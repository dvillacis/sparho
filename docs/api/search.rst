Outer-search loops
==================

.. currentmodule:: sparho

Two outer loops ship at v0.1. Both step in ``θ = log α`` space (positive
``α`` without projection), both threadable with any
``Solver`` × ``Criterion`` × hypergradient triple, and both refit the
inner solver on the full problem at the best ``α`` seen before returning.

.. autofunction:: grad_search

.. autofunction:: hoag_search
