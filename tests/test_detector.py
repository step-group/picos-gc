"""Tests for peak detection, smoothing, and the merge passes."""

from __future__ import annotations

import itertools

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


def test_clip_window_bounds_isolated_peak_in_noise():
    # Without clipping, an isolated peak's prominence bases land on the deepest
    # noise minima — minutes away — and the zero-clamped integral rectifies
    # baseline noise over the whole slack region (observed +60%).
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 60000)
    sig = gaussian(t, 5.0, 500, 0.05) + rng.normal(0, 2.0, t.shape)
    chrom = make_chrom(sig)
    peaks = detect_peaks(chrom, DetectionParams())
    assert len(peaks) == 1
    d = peaks[0]
    window = float(chrom.time_min[d.right_base] - chrom.time_min[d.left_base])
    assert window < 1.0  # ~4.3 min before clipping
    # apex stays inside the clipped window
    assert d.left_base <= d.index <= d.right_base


def test_clip_zero_restores_legacy_wide_windows():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 60000)
    sig = gaussian(t, 5.0, 500, 0.05) + rng.normal(0, 2.0, t.shape)
    chrom = make_chrom(sig)
    peaks = detect_peaks(chrom, DetectionParams(clip_window_frac=0.0))
    assert len(peaks) == 1
    d = peaks[0]
    window = float(chrom.time_min[d.right_base] - chrom.time_min[d.left_base])
    assert window > 1.0  # legacy prominence-base boundaries


def test_clip_keeps_fused_valley_boundaries_shared():
    # A fused pair splits at the valley; clipping must not move the shared
    # boundary (only the outer feet), or the valley split would leak.
    t = np.linspace(0, 10, 60000)
    sig = gaussian(t, 5.0, 500, 0.08) + gaussian(t, 5.25, 300, 0.08)
    chrom = make_chrom(sig)
    peaks = detect_peaks(chrom, DetectionParams(**NO_MERGE))
    assert len(peaks) == 2
    assert peaks[0].right_base == peaks[1].left_base


def test_clip_preserves_proximity_merged_envelope():
    # Two resolved peaks merged by proximity: the envelope must keep both
    # humps even though the valley between them touches baseline.
    t = np.linspace(0, 10, 60000)
    sig = gaussian(t, 5.0, 400, 0.05) + gaussian(t, 5.4, 350, 0.05)
    chrom = make_chrom(sig)
    merged = detect_peaks(
        chrom,
        DetectionParams(min_height=50, min_prominence=20, min_distance=10, merge_distance_min=0.5),
    )
    assert len(merged) == 1
    d = merged[0]
    # window spans both apexes (5.0 and 5.4) with feet outside them
    assert float(chrom.time_min[d.left_base]) < 4.9
    assert float(chrom.time_min[d.right_base]) > 5.5


def test_no_overlapping_windows_after_merges():
    # Merged envelopes take prominence-base boundaries, which can overlap the
    # next peak's window; the post-merge overlap pass must remove that.
    t = np.linspace(0, 10, 60000)
    sig = (
        gaussian(t, 4.0, 500, 0.08)
        + gaussian(t, 4.25, 300, 0.08)  # fuses/merges with the first
        + gaussian(t, 4.8, 400, 0.05)  # separate neighbour
    )
    chrom = make_chrom(sig)
    peaks = detect_peaks(
        chrom,
        DetectionParams(min_height=50, min_prominence=20, min_distance=10, merge_shoulder_ratio=0.5),
    )
    for a, b in itertools.pairwise(peaks):
        assert a.right_base <= b.left_base


def test_clip_separates_artifact_shared_windows():
    # Resolved peaks whose over-wide legacy windows merely overlapped (and were
    # split at an inter-peak noise minimum) must not be treated as fused: their
    # windows are clipped to the local feet, removing minutes of noisy baseline
    # that previously inflated zero-clamped areas by 6-9%.
    rng = np.random.default_rng(7)
    t = np.linspace(0, 10, 60000)
    sig = (
        gaussian(t, 3.0, 400, 0.05)
        + gaussian(t, 5.0, 500, 0.05)
        + gaussian(t, 7.0, 300, 0.06)
        + rng.normal(0, 1.0, t.shape)
    )
    chrom = make_chrom(sig)
    peaks = detect_peaks(chrom, DetectionParams())
    assert len(peaks) == 3
    for a, b in itertools.pairwise(peaks):
        assert a.right_base < b.left_base  # separated, not shared
    for d in peaks:
        window = float(chrom.time_min[d.right_base] - chrom.time_min[d.left_base])
        assert window < 1.0


def test_clip_does_not_collapse_tiny_peak_fused_to_giant_tail():
    # A small peak riding a huge neighbour's tail is judged fused (the valley
    # between them is elevated). Its outer-side clipping threshold must scale
    # with its own height, not the group's: a group-apex threshold once exceeded
    # the small peak's entire height and collapsed its window to the apex
    # (area ~0 for a real 58 mV peak).
    t = np.linspace(0, 10, 60000)
    sig = gaussian(t, 3.0, 70000, 0.15) + gaussian(t, 4.5, 58, 0.04)
    chrom = make_chrom(sig)
    peaks = detect_peaks(chrom, DetectionParams())
    small = [d for d in peaks if sig[d.index] < 1000]
    assert len(small) == 1
    d = small[0]
    window = float(chrom.time_min[d.right_base] - chrom.time_min[d.left_base])
    assert window > 0.08  # spans the peak, not collapsed to the apex
    assert d.left_base < d.index < d.right_base


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
