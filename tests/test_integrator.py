"""Tests for peak integration with linear baseline subtraction."""

from __future__ import annotations

import numpy as np
import pytest

from picos_gc.detector import DetectedPeak
from picos_gc.integrator import integrate_all_peaks, integrate_peak
from tests.helpers import gaussian, make_chrom

SQRT_2PI = np.sqrt(2 * np.pi)


def test_gaussian_area_matches_analytic():
    # Analytic area of a Gaussian above zero baseline = height * width * sqrt(2*pi).
    t = np.linspace(0, 10, 20000)
    height, width = 1000.0, 0.1
    sig = gaussian(t, 5.0, height, width)
    # Integrate over a wide window around the peak.
    left = int(np.searchsorted(t, 5.0 - 1.0))
    right = int(np.searchsorted(t, 5.0 + 1.0))
    apex = int(np.searchsorted(t, 5.0))
    area = integrate_peak(t, sig, apex, left, right)
    expected = height * width * SQRT_2PI
    assert area == pytest.approx(expected, rel=0.01)


def test_linear_baseline_is_subtracted():
    # A Gaussian sitting on a sloped baseline should integrate to ~the same
    # area as the same Gaussian on a flat zero baseline.
    t = np.linspace(0, 10, 20000)
    height, width = 500.0, 0.1
    peak = gaussian(t, 5.0, height, width)
    sloped = peak + (50.0 + 30.0 * t)  # offset + slope
    left = int(np.searchsorted(t, 5.0 - 1.0))
    right = int(np.searchsorted(t, 5.0 + 1.0))
    apex = int(np.searchsorted(t, 5.0))
    area = integrate_peak(t, sloped, apex, left, right)
    expected = height * width * SQRT_2PI
    assert area == pytest.approx(expected, rel=0.02)


def test_degenerate_segment_returns_zero():
    t = np.linspace(0, 10, 100)
    sig = np.ones(100)
    assert integrate_peak(t, sig, 5, 5, 5) == 0.0  # single point
    assert integrate_peak(t, sig, 5, 5, 4) == 0.0  # empty/reversed


def test_integrate_all_peaks_numbers_and_fields():
    t = np.linspace(0, 10, 5000)
    sig = gaussian(t, 3.0, 1000, 0.05) + gaussian(t, 7.0, 800, 0.05)
    chrom = make_chrom(sig)
    detected = [
        DetectedPeak(index=1500, left_base=1400, right_base=1600),
        DetectedPeak(index=3500, left_base=3400, right_base=3600),
    ]
    results = integrate_all_peaks(chrom, detected)
    assert [r.peak_number for r in results] == [1, 2]
    assert results[0].time_min == pytest.approx(3.0, abs=0.01)
    assert results[1].time_min == pytest.approx(7.0, abs=0.01)
    assert all(r.area_mV_min > 0 for r in results)
    assert results[0].height_mV > results[1].height_mV
