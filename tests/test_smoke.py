from __future__ import annotations

import sparho


def test_import_and_version():
    assert isinstance(sparho.__version__, str)
    assert len(sparho.__version__) > 0


def test_rust_core_round_trip():
    assert sparho._core.version() != ""
