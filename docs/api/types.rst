Type aliases
============

.. currentmodule:: sparho

The public type aliases used across the API. All are concrete
``typing.TypeAlias`` definitions over numpy / scipy.sparse types.

.. py:data:: Scalar
   :type: TypeAlias

   ``float | numpy.floating`` — a real scalar.

.. py:data:: Array
   :type: TypeAlias

   ``numpy.typing.NDArray[numpy.floating]`` — a real-valued numpy array of
   arbitrary shape.

.. py:data:: Hyperparam
   :type: TypeAlias

   ``float | Array``. The outer-loop hyperparameter — a scalar (e.g. the
   Lasso ``α``) or a per-feature vector (e.g. ``WeightedL1``'s ``α_j``).

.. py:data:: DesignMatrix
   :type: TypeAlias

   ``Array | scipy.sparse.csc_matrix | scipy.sparse.csc_array``. CSC is the
   v0.1 sparse format. CSR and COO inputs are expected to be converted by
   the caller via ``.tocsc()`` before reaching the inner solvers.

.. py:data:: IndexArray
   :type: TypeAlias

   ``numpy.typing.NDArray[numpy.int32]``. Index arrays (active set, fold
   indices). ``int32`` to match CSC indices.
