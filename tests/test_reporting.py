from __future__ import annotations

import math

from src.reporting import _interpretation


def test_interpretation_ranges_are_honest():
    assert "below 0.50" in _interpretation(0.4887)
    assert "[0.50, 0.53]" in _interpretation(0.50)
    assert "[0.50, 0.53]" in _interpretation(0.53)
    assert "(0.53, 0.58]" in _interpretation(0.55)
    assert "red flag" in _interpretation(0.61)
    assert "undefined" in _interpretation(math.nan)
    assert "undefined" in _interpretation(None)
