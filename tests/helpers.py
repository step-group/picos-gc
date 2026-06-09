"""Synthetic-signal helpers shared across tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from picos_gc.reader import Chromatogram

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def gaussian(t: np.ndarray, center: float, height: float, width: float) -> np.ndarray:
    """A Gaussian peak. Analytic area above zero baseline = height * width * sqrt(2*pi)."""
    return height * np.exp(-0.5 * ((t - center) / width) ** 2)


def make_chrom(signal: np.ndarray, t_start: float = 0.0, t_end: float = 10.0) -> Chromatogram:
    """Wrap a signal array in a Chromatogram with uniform time sampling."""
    t = np.linspace(t_start, t_end, len(signal))
    return Chromatogram(filepath=Path("synthetic.gcd"), time_min=t, signal_mV=signal)
