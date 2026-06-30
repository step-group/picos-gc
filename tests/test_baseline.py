"""Tests for optional global baseline correction."""

from __future__ import annotations

import numpy as np
import pytest

from picos_gc.baseline import ALLOWED_METHODS, DEFAULT_METHOD, correct_baseline
from picos_gc.detector import DetectionParams, detect_peaks
from picos_gc.integrator import integrate_all_peaks, integrate_peak
from tests.helpers import gaussian, make_chrom

SQRT_2PI = np.sqrt(2 * np.pi)


def test_default_method_is_arpls():
    assert DEFAULT_METHOD == "arpls"
    assert DEFAULT_METHOD in ALLOWED_METHODS


def test_unknown_method_raises():
    t = np.linspace(0, 10, 100)
    with pytest.raises(ValueError, match="unknown baseline method"):
        correct_baseline(np.zeros_like(t), t, method="bogus")


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="same shape"):
        correct_baseline(np.zeros(10), np.zeros(11), method="asls")


def test_short_signal_returned_unchanged():
    sig = np.array([3.0])
    out = correct_baseline(sig, np.array([0.0]), method="asls")
    assert np.array_equal(out, sig)


def test_correction_improves_area_on_curved_baseline():
    t = np.linspace(0, 10, 20000)
    height, width = 500.0, 0.1
    peak = gaussian(t, 5.0, height, width)
    drift = 100.0 + 5.0 * (t - 5.0) ** 2  # convex: a linear chord can't remove it
    raw = peak + drift
    expected = height * width * SQRT_2PI

    left = int(np.searchsorted(t, 4.0))
    right = int(np.searchsorted(t, 6.0))
    apex = int(np.searchsorted(t, 5.0))

    raw_area = integrate_peak(t, raw, apex, left, right)
    corrected = correct_baseline(raw, t, method="arpls", lam=1e5)
    corr_area = integrate_peak(t, corrected, apex, left, right)

    assert abs(corr_area - expected) < abs(raw_area - expected)
    assert corr_area == pytest.approx(expected, rel=0.05)


def test_correction_end_to_end_recovers_area():
    t = np.linspace(0, 10, 20000)
    height, width = 800.0, 0.08
    peak = gaussian(t, 4.0, height, width)
    drift = 50.0 + 12.0 * t + 1.5 * t**2  # monotone rising, curved
    raw = peak + drift
    expected = height * width * SQRT_2PI

    chrom = make_chrom(correct_baseline(raw, t, method="arpls", lam=1e6))
    results = integrate_all_peaks(chrom, detect_peaks(chrom, DetectionParams()))
    assert len(results) == 1
    assert results[0].area_mV_min == pytest.approx(expected, rel=0.05)
