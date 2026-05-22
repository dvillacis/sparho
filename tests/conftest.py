"""Shared pytest configuration for the sparho test suite.

Currently provides one fixture: a session-scope autouse BLAS-thread pin.

Multi-threaded BLAS makes reduction order non-deterministic at the bit
level (see ``docs/reproducibility.md``), and several tests in this suite
— ``test_determinism.py``, ``test_determinism_matrix.py``, the golden
regression suite — rely on bit-identical floats. Pinning BLAS to a
single thread for the *entire* test process makes those guarantees
robust without each test having to opt in.

Users with deliberately multi-threaded inner solvers (e.g. running
sparho's tests embedded in a larger CI pipeline) can opt out by setting
``SPARHO_TEST_RESPECT_BLAS=1`` in the environment before invoking
pytest. The original BLAS-thread env state is then preserved.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _pin_blas_threads_to_one() -> Iterator[None]:
    """Session-scope autouse fixture pinning BLAS to 1 thread.

    Set ``SPARHO_TEST_RESPECT_BLAS=1`` to skip; useful when the surrounding
    CI orchestrator already pins threads or wants multi-thread coverage.
    """
    if os.environ.get("SPARHO_TEST_RESPECT_BLAS") == "1":
        yield
        return
    # Local import: the helper itself is in `sparho.testing`, which depends
    # on the compiled extension. Importing here (not at module top) keeps
    # collection-time errors localized to fixture setup rather than the
    # whole conftest.
    from sparho.testing import pin_blas_threads

    with pin_blas_threads(1):
        yield
