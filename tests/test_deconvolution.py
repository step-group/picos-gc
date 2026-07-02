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


from picos_gc.deconvolution import deconvolve_group
from picos_gc.detector import DetectedPeak


def _two_emg_group(t, a1, mu1, a2, mu2, sigma, tau):
    """Build a signal of two EMGs and a hand-made 2-member fused group."""
    sig = emg(t, a1, mu1, sigma, tau) + emg(t, a2, mu2, sigma, tau)
    i1 = int(np.argmin(np.abs(t - mu1)))
    i2 = int(np.argmin(np.abs(t - mu2)))
    valley = i1 + int(np.argmin(sig[i1 : i2 + 1]))  # shared boundary
    left = DetectedPeak(index=i1, left_base=0, right_base=valley)
    right = DetectedPeak(index=i2, left_base=valley, right_base=len(t) - 1)
    return sig, [left, right]


def test_deconvolve_recovers_two_overlapping_areas():
    t = np.linspace(8.6, 9.6, 12000)
    a1, a2 = 5.0, 1.5
    sig, group = _two_emg_group(t, a1, 9.0, a2, 9.12, sigma=0.02, tau=0.03)
    areas, r2 = deconvolve_group(t, sig, group)
    assert areas is not None
    assert r2 > 0.99
    assert areas[0] == pytest.approx(a1, rel=0.05)
    assert areas[1] == pytest.approx(a2, rel=0.05)


def test_deconvolve_rejects_unfittable_group():
    # A flat/noise group has no peak structure: the fit is rejected (None) so the
    # caller falls back to drop rather than emitting a bogus area.
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 0.5, 400)
    sig = rng.normal(0, 1.0, t.shape)
    group = [
        DetectedPeak(index=100, left_base=0, right_base=200),
        DetectedPeak(index=300, left_base=200, right_base=399),
    ]
    areas, _r2 = deconvolve_group(t, sig, group)
    assert areas is None
