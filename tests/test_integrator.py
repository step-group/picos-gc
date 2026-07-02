"""Tests for peak integration with linear baseline subtraction."""

from __future__ import annotations

import numpy as np
import pytest

from picos_gc.detector import DetectedPeak, DetectionParams, detect_peaks
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


def test_noisy_isolated_peak_area_accurate_with_clipping():
    # End-to-end: clipping keeps the area of a peak over a noisy baseline
    # close to analytic (was +60% with legacy wide windows).
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 60000)
    height, width = 500.0, 0.05
    sig = gaussian(t, 5.0, height, width) + rng.normal(0, 2.0, t.shape)
    chrom = make_chrom(sig)
    results = integrate_all_peaks(chrom, detect_peaks(chrom, DetectionParams()))
    assert len(results) == 1
    expected = height * width * SQRT_2PI
    assert results[0].area_mV_min == pytest.approx(expected, rel=0.03)


def test_drop_mode_conserves_fused_pair_area():
    # Valley mode integrates each fused peak above a chord rising to the
    # valley, cutting away the overlap; drop mode draws one baseline across
    # the group and conserves the total.
    t = np.linspace(0, 10, 60000)
    sig = gaussian(t, 5.0, 500, 0.08) + gaussian(t, 5.25, 300, 0.08)
    chrom = make_chrom(sig)
    detected = detect_peaks(chrom, DetectionParams(merge_shoulder_ratio=0.0))
    assert len(detected) == 2
    expected_total = (500 + 300) * 0.08 * SQRT_2PI

    valley = integrate_all_peaks(chrom, detected, split_mode="valley")
    drop = integrate_all_peaks(chrom, detected, split_mode="drop")

    total_valley = sum(r.area_mV_min for r in valley)
    total_drop = sum(r.area_mV_min for r in drop)
    assert total_drop == pytest.approx(expected_total, rel=0.02)
    assert total_valley < 0.6 * expected_total  # documents the valley-mode loss
    # the larger peak keeps the larger share
    assert drop[0].area_mV_min > drop[1].area_mV_min


def test_drop_mode_equals_valley_for_isolated_peaks():
    t = np.linspace(0, 10, 20000)
    sig = gaussian(t, 3.0, 1000, 0.05) + gaussian(t, 7.0, 800, 0.05)
    chrom = make_chrom(sig)
    detected = detect_peaks(chrom, DetectionParams(merge_shoulder_ratio=0.0))
    valley = integrate_all_peaks(chrom, detected, split_mode="valley")
    drop = integrate_all_peaks(chrom, detected, split_mode="drop")
    for v, d in zip(valley, drop, strict=True):
        assert d.area_mV_min == pytest.approx(v.area_mV_min, rel=1e-9)


def test_split_mode_validation():
    chrom = make_chrom(np.zeros(100))
    with pytest.raises(ValueError, match="split_mode"):
        integrate_all_peaks(chrom, [], split_mode="bogus")


def test_deconvolve_isolated_equals_drop():
    # Isolated peaks are integrated exactly as drop mode (no fitting).
    t = np.linspace(0, 10, 20000)
    sig = gaussian(t, 3.0, 1000, 0.05) + gaussian(t, 7.0, 800, 0.05)
    chrom = make_chrom(sig)
    detected = detect_peaks(chrom, DetectionParams(merge_shoulder_ratio=0.0))
    drop = integrate_all_peaks(chrom, detected, split_mode="drop")
    deconv = integrate_all_peaks(chrom, detected, split_mode="deconvolve")
    for d, x in zip(drop, deconv, strict=True):
        assert x.area_mV_min == pytest.approx(d.area_mV_min, rel=1e-9)


def test_deconvolve_beats_drop_on_tailing_overlap():
    # A small peak riding the TAIL of a large tailing peak: drop's vertical split
    # hands the big peak's tail to the small one (over-estimate); EMG recovers it.
    # The pair is a hand-built fused group so the test targets the integrator's
    # deconvolve branch, not detector tuning.
    from picos_gc.deconvolution import emg

    t = np.linspace(8.5, 9.8, 40000)
    a_big, a_small = 50.0, 2.0
    mu_big, mu_small = 9.0, 9.20
    sigma, tau = 0.02, 0.05  # pronounced tail on the big peak
    sig = emg(t, a_big, mu_big, sigma, tau) + emg(t, a_small, mu_small, sigma, tau)
    chrom = make_chrom(sig, t_start=8.5, t_end=9.8)

    i_big = int(np.argmin(np.abs(t - mu_big)))
    i_small = int(np.argmin(np.abs(t - mu_small)))
    valley = i_big + int(np.argmin(sig[i_big : i_small + 1]))  # shared boundary
    detected = [
        DetectedPeak(index=i_big, left_base=0, right_base=valley),
        DetectedPeak(index=i_small, left_base=valley, right_base=len(t) - 1),
    ]

    drop = integrate_all_peaks(chrom, detected, split_mode="drop")
    deconv = integrate_all_peaks(chrom, detected, split_mode="deconvolve")

    drop_small = drop[1].area_mV_min
    deconv_small = deconv[1].area_mV_min
    assert abs(deconv_small - a_small) < abs(drop_small - a_small)  # closer to truth
    assert deconv_small == pytest.approx(a_small, rel=0.15)


def test_deconvolve_mode_is_valid():
    chrom = make_chrom(np.zeros(100))
    # empty peak list is a no-op in any valid mode (must not raise)
    assert integrate_all_peaks(chrom, [], split_mode="deconvolve") == []
