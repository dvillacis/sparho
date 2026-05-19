Hypergradient
=============

.. currentmodule:: sparho

At v0.1 we ship a single hypergradient mode — ``implicit_forward`` —
which restricts the KKT linear system to the inner-solver's active set
and solves it via matrix-free conjugate gradients with an auto-scaled
Tikhonov ridge.

.. autofunction:: implicit_forward
