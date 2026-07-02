"""Tests for EMG peak deconvolution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from picos_gc.deconvolution import emg

SQRT_2PI = np.sqrt(2 * np.pi)


@pytest.mark.parametrize("tau", [1e-4, 0.02, 0.05])
def test_emg_area_equals_area_param(tau):
    # The `area` parameter of an area-normalised EMG is its true integral,
    # for any tailing tau. Integrate over a window wide enough for the tail.
    area, mu, sigma = 3.0, 5.0, 0.03
    t = np.linspace(mu - 10 * sigma, mu + 10 * sigma + 12 * tau, 40000)
    y = emg(t, area, mu, sigma, tau)
    assert trapezoid(y, t) == pytest.approx(area, rel=0.01)
    assert np.all(np.isfinite(y))


def test_emg_small_tau_approaches_gaussian():
    # With negligible tau the EMG is a symmetric Gaussian of the same area,
    # so its apex sits at mu and area/height ratio matches sigma*sqrt(2*pi).
    area, mu, sigma = 2.0, 5.0, 0.04
    t = np.linspace(mu - 8 * sigma, mu + 8 * sigma, 20000)
    y = emg(t, area, mu, sigma, 1e-5)
    apex_t = t[int(np.argmax(y))]
    assert apex_t == pytest.approx(mu, abs=2 * sigma / 100)
    height = y.max()
    assert area / height == pytest.approx(sigma * SQRT_2PI, rel=0.02)
