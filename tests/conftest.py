"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import EXAMPLES


@pytest.fixture
def example_gcd() -> Path:
    """Path to a real Shimadzu .gcd file; skip the test if examples are absent."""
    candidate = EXAMPLES / "2m2boh" / "2m2boh1.gcd"
    if not candidate.exists():
        pytest.skip(f"example file not found: {candidate}")
    return candidate
