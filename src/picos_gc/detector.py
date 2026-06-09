"""Peak detection for GC chromatograms."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

from .reader import Chromatogram

# Looser thresholds used only if the supplied params find nothing at all.
_FALLBACK_HEIGHT = 10.0
_FALLBACK_PROMINENCE = 5.0


@dataclass
class DetectionParams:
    min_height: float = 50.0  # mV
    min_prominence: float = 20.0  # mV
    min_distance: int = 50  # data points
    min_width_min: float = 0.03  # minutes (~1.8 s) — filters sub-second artifacts
    smooth_window: int = 0  # Savitzky-Golay window for detection (odd, > polyorder); 0 = off
    smooth_polyorder: int = 3  # Savitzky-Golay polynomial order
    # Shoulder merge: combine two adjacent peaks when the baseline-corrected
    # valley between them rises above this fraction of the smaller apex.
    # 0 disables merging; values near 1 merge only the most poorly resolved pairs.
    merge_shoulder_ratio: float = 0.5
    # Proximity merge: combine adjacent peaks whose apex retention times are
    # within this many minutes of each other. 0 disables it.
    merge_distance_min: float = 0.0


@dataclass
class DetectedPeak:
    index: int  # index into the raw signal array (apex)
    left_base: int  # left integration boundary index
    right_base: int  # right integration boundary index
    # Prominence bases: where the signal descends to its local baseline on each
    # side (from find_peaks). Used to anchor a baseline for the shoulder-merge
    # metric. Default to the integration boundaries when not supplied.
    base_left: int = -1
    base_right: int = -1

    def __post_init__(self) -> None:
        if self.base_left < 0:
            self.base_left = self.left_base
        if self.base_right < 0:
            self.base_right = self.right_base


def _smoothed(signal: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Return a Savitzky-Golay-smoothed copy of *signal*, or *signal* unchanged.

    Smoothing is skipped (and the raw signal returned) when ``window <= 0`` or
    when the parameters are invalid for the data (window must be odd, greater
    than ``polyorder``, and no longer than the signal).
    """
    if window <= 0:
        return signal
    if window % 2 == 0:
        raise ValueError(f"smooth_window must be odd, got {window}")
    if window <= polyorder:
        raise ValueError(
            f"smooth_window ({window}) must be greater than smooth_polyorder ({polyorder})"
        )
    if window > len(signal):
        return signal
    return savgol_filter(signal, window, polyorder)


def _valley_drop_ratio(signal: np.ndarray, a: DetectedPeak, b: DetectedPeak) -> float:
    """Baseline-corrected valley-to-peak ratio between two adjacent peaks.

    Returns a value where ~1 means the valley between the peaks is nearly as
    tall as the smaller apex (poorly resolved → merge), and ~0 means the valley
    falls to the local baseline (well resolved → keep separate). The valley
    depth is measured above a straight baseline drawn between the pair's outer
    boundaries, so an elevated or sloping baseline does not trigger spurious
    merges. Returns 0.0 when the smaller apex sits at or below that baseline.
    """
    lo, hi = a.index, b.index
    if hi <= lo:
        return 0.0
    seg = signal[lo : hi + 1]
    valley_idx = lo + int(np.argmin(seg))
    valley_val = float(signal[valley_idx])

    # Baseline anchored on the pair's outer prominence bases — the points where
    # the signal descends to its local baseline — so a shoulder's high inner
    # boundary does not lift the baseline and mask the valley.
    left, right = a.base_left, b.base_right
    if right > left:
        frac = (valley_idx - left) / (right - left)
        base = float(signal[left]) + (float(signal[right]) - float(signal[left])) * frac
    else:
        base = min(float(signal[left]), float(signal[right]))

    smaller_apex = min(float(signal[a.index]), float(signal[b.index]))
    denom = smaller_apex - base
    if denom <= 0:
        return 0.0
    return (valley_val - base) / denom


def _merge_shoulders(
    detected: list[DetectedPeak], signal: np.ndarray, ratio: float
) -> list[DetectedPeak]:
    """Merge chains of adjacent peaks separated only by a shallow valley."""
    if ratio <= 0 or len(detected) < 2:
        return detected

    merged: list[DetectedPeak] = []
    i = 0
    while i < len(detected):
        current = detected[i]
        while (
            i + 1 < len(detected) and _valley_drop_ratio(signal, current, detected[i + 1]) > ratio
        ):
            nxt = detected[i + 1]
            rep = current.index if signal[current.index] >= signal[nxt.index] else nxt.index
            # A merged envelope is integrated down to the outer baseline, so its
            # integration boundaries become the outer prominence bases.
            current = DetectedPeak(
                index=rep,
                left_base=current.base_left,
                right_base=nxt.base_right,
                base_left=current.base_left,
                base_right=nxt.base_right,
            )
            i += 1
        merged.append(current)
        i += 1
    return merged


def _merge_proximity(
    detected: list[DetectedPeak],
    signal: np.ndarray,
    time_min: np.ndarray,
    max_gap: float,
) -> list[DetectedPeak]:
    """Merge chains of adjacent peaks whose apex retention times are close."""
    if max_gap <= 0 or len(detected) < 2:
        return detected

    out: list[DetectedPeak] = []
    i = 0
    while i < len(detected):
        current = detected[i]
        last_apex = current.index
        while (
            i + 1 < len(detected)
            and (time_min[detected[i + 1].index] - time_min[last_apex]) <= max_gap
        ):
            nxt = detected[i + 1]
            rep = current.index if signal[current.index] >= signal[nxt.index] else nxt.index
            current = DetectedPeak(
                index=rep,
                left_base=current.base_left,
                right_base=nxt.base_right,
                base_left=current.base_left,
                base_right=nxt.base_right,
            )
            last_apex = nxt.index
            i += 1
        out.append(current)
        i += 1
    return out


def detect_peaks(chrom: Chromatogram, params: DetectionParams) -> list[DetectedPeak]:
    """Detect all peaks and their integration boundaries in a chromatogram.

    Pipeline:
      1. Optionally Savitzky-Golay-smooth a *copy* of the signal for detection
         only (boundaries and apex positions are taken from this copy; the
         integrator always works on the raw signal).
      2. ``find_peaks`` with the supplied height / prominence / distance / width
         thresholds, falling back to looser thresholds if nothing is found.
      3. ``peak_widths`` at ``rel_height=1.0`` puts each boundary at the
         prominence reference level (the local valley floor) rather than out in
         flat baseline.
      4. Valley-clipping safety net for any boundaries that still overlap.
      5. Optional shoulder merge (baseline-aware) and proximity merge.

    Returns peaks sorted by retention time (left → right).
    """
    raw = chrom.signal_mV
    detect_signal = _smoothed(raw, params.smooth_window, params.smooth_polyorder)

    # Convert min_width from minutes to data points using the actual sampling rate.
    span = float(chrom.time_min[-1] - chrom.time_min[0])
    pts_per_min = len(chrom.time_min) / span if span > 0 else 0.0
    width_pts = params.min_width_min * pts_per_min if pts_per_min > 0 else 0.0
    width_arg = width_pts if width_pts > 0 else None

    peaks, props = find_peaks(
        detect_signal,
        height=params.min_height,
        prominence=params.min_prominence,
        distance=params.min_distance,
        width=width_arg,
    )

    if len(peaks) == 0:
        peaks, props = find_peaks(
            detect_signal,
            height=_FALLBACK_HEIGHT,
            prominence=_FALLBACK_PROMINENCE,
            distance=params.min_distance,
            width=width_arg,
        )

    if len(peaks) == 0:
        return []

    # peak_widths at rel_height=1.0: boundaries at the prominence reference
    # level. Reuses prominence data already computed by find_peaks.
    _, _, left_ips, right_ips = peak_widths(
        detect_signal,
        peaks,
        rel_height=1.0,
        prominence_data=(
            props["prominences"],
            props["left_bases"],
            props["right_bases"],
        ),
    )

    order = np.argsort(peaks)
    detected = [
        DetectedPeak(
            index=int(peaks[i]),
            left_base=int(np.round(left_ips[i])),
            right_base=int(np.round(right_ips[i])),
            base_left=int(props["left_bases"][i]),
            base_right=int(props["right_bases"][i]),
        )
        for i in order
    ]

    # Safety net: clip any remaining overlaps to the valley minimum.
    for a, b in itertools.pairwise(detected):
        if a.right_base > b.left_base:
            valley = int(np.argmin(detect_signal[a.index : b.index + 1])) + a.index
            a.right_base = valley
            b.left_base = valley

    detected = _merge_shoulders(detected, detect_signal, params.merge_shoulder_ratio)
    detected = _merge_proximity(detected, detect_signal, chrom.time_min, params.merge_distance_min)
    return detected
