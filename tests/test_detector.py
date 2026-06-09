"""Tests for peak detection, smoothing, and the merge passes."""

from __future__ import annotations

import numpy as np
import pytest

from picos_gc.detector import (
    DetectedPeak,
    DetectionParams,
    _smoothed,
    _valley_drop_ratio,
    detect_peaks,
)
from tests.helpers import gaussian, make_chrom

# Detection params with merging disabled — the baseline for most tests.
NO_MERGE = {"min_height": 50, "min_prominence": 20, "min_distance": 10, "merge_shoulder_ratio": 0.0}


def _tr(chrom, peaks):
    return [round(float(chrom.time_min[p.index]), 2) for p in peaks]


def test_detects_two_separated_peaks():
    t = np.linspace(0, 10, 5000)
    sig = gaussian(t, 3.0, 1000, 0.05) + gaussian(t, 7.0, 800, 0.05)
    chrom = make_chrom(sig)
    peaks = detect_peaks(chrom, DetectionParams(**NO_MERGE))
    assert len(peaks) == 2
    assert _tr(chrom, peaks) == [3.0, 7.0]
    # Peaks are returned sorted by retention time.
    assert peaks[0].index < peaks[1].index


def test_no_peaks_returns_empty():
    sig = np.zeros(2000)
    chrom = make_chrom(sig)
    assert detect_peaks(chrom, DetectionParams(**NO_MERGE)) == []


def test_fallback_thresholds_catch_weak_signal():
    # A small peak below the default min_height (50) but above the fallback (10).
    t = np.linspace(0, 10, 5000)
    sig = gaussian(t, 5.0, 25, 0.05)
    chrom = make_chrom(sig)
    peaks = detect_peaks(chrom, DetectionParams(min_height=50, min_prominence=20, min_distance=10))
    assert len(peaks) == 1


def test_merge_off_keeps_shoulder_pair_separate():
    t = np.linspace(0, 10, 5000)
    sig = gaussian(t, 3.0, 1000, 0.07) + gaussian(t, 3.2, 950, 0.07) + 300.0
    chrom = make_chrom(sig)
    peaks = detect_peaks(chrom, DetectionParams(**NO_MERGE))
    assert len(peaks) == 2


def test_baseline_aware_merge_combines_shoulder_on_elevated_baseline():
    # Shoulder pair sitting on a 300 mV baseline. A naive valley/apex ratio
    # would be fooled by the baseline; the baseline-aware metric should merge.
    t = np.linspace(0, 10, 5000)
    sig = gaussian(t, 3.0, 1000, 0.07) + gaussian(t, 3.2, 950, 0.07) + 300.0
    chrom = make_chrom(sig)
    merged = detect_peaks(
        chrom,
        DetectionParams(
            min_height=50, min_prominence=20, min_distance=10, merge_shoulder_ratio=0.5
        ),
    )
    assert len(merged) == 1
    # A high threshold only merges the very worst-resolved pairs — not this one.
    strict = detect_peaks(
        chrom,
        DetectionParams(
            min_height=50, min_prominence=20, min_distance=10, merge_shoulder_ratio=0.95
        ),
    )
    assert len(strict) == 2


def test_well_separated_peaks_never_merge():
    t = np.linspace(0, 10, 5000)
    sig = gaussian(t, 3.0, 1000, 0.05) + gaussian(t, 7.0, 800, 0.05) + 300.0
    chrom = make_chrom(sig)
    for ratio in (0.1, 0.5, 0.9):
        peaks = detect_peaks(
            chrom,
            DetectionParams(
                min_height=50, min_prominence=20, min_distance=10, merge_shoulder_ratio=ratio
            ),
        )
        assert len(peaks) == 2, f"ratio {ratio} wrongly merged separated peaks"


def test_proximity_merge():
    t = np.linspace(0, 10, 5000)
    sig = gaussian(t, 5.00, 1000, 0.03) + gaussian(t, 5.12, 900, 0.03)
    chrom = make_chrom(sig)
    # Resolved enough that the shoulder metric leaves them apart...
    separate = detect_peaks(chrom, DetectionParams(**NO_MERGE))
    assert len(separate) == 2
    # ...but a proximity gap of 0.2 min collapses them.
    merged = detect_peaks(
        chrom,
        DetectionParams(min_height=50, min_prominence=20, min_distance=10, merge_distance_min=0.2),
    )
    assert len(merged) == 1


def test_valley_drop_ratio_bounds():
    # Two peaks with a valley that drops fully to baseline -> ratio near 0.
    t = np.linspace(0, 10, 5000)
    sig = gaussian(t, 3.0, 1000, 0.04) + gaussian(t, 5.0, 1000, 0.04)
    a = DetectedPeak(index=1500, left_base=1400, right_base=1600)
    b = DetectedPeak(index=2500, left_base=2400, right_base=2600)
    assert _valley_drop_ratio(sig, a, b) < 0.05


def test_smoothed_validations():
    sig = np.linspace(0, 1, 100)
    # window 0 -> unchanged (same object).
    assert _smoothed(sig, 0, 3) is sig
    # even window -> error.
    with pytest.raises(ValueError):
        _smoothed(sig, 10, 3)
    # window <= polyorder -> error.
    with pytest.raises(ValueError):
        _smoothed(sig, 3, 3)
    # window longer than signal -> falls back to raw (no error).
    assert _smoothed(sig, 201, 3) is sig
    # valid smoothing returns a same-length, distinct array.
    out = _smoothed(sig, 11, 3)
    assert out.shape == sig.shape


def test_smoothing_reduces_noise_but_keeps_peak():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 5000)
    clean = gaussian(t, 5.0, 1000, 0.1)
    noisy = clean + rng.normal(0, 5, size=clean.shape)
    chrom = make_chrom(noisy)
    peaks = detect_peaks(
        chrom,
        DetectionParams(min_height=50, min_prominence=20, min_distance=10, smooth_window=21),
    )
    assert len(peaks) == 1
    assert float(chrom.time_min[peaks[0].index]) == pytest.approx(5.0, abs=0.05)
