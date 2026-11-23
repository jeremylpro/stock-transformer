from __future__ import annotations

from src.utils import library_versions


def test_library_versions_uses_distribution_metadata():
    versions = library_versions()
    # ``ta`` has no ``__version__`` attribute, so it must fall back to the
    # installed distribution metadata rather than reporting "not installed".
    assert versions["ta"] != "not installed"
    assert versions["pandas"] != "not installed"
