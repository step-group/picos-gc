"""Peak integration with linear baseline subtraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from .deconvolution import deconvolve_group
from .detector import DetectedPeak
from .reader import Chromatogram


@dataclass
class PeakResult:
    peak_number: int  # 1-based, ordered by retention time
    time_min: float  # retention time (tR)
    height_mV: float
    area_mV_min: float
    left_min: float
    right_min: float


def integrate_peak(
    time: np.ndarray, signal: np.ndarray, peak_idx: int, left_idx: int, right_idx: int
) -> float:
    """Integrate a single peak with linear baseline subtraction.

    Uses scipy-supplied valley boundaries (left_idx, right_idx) directly —
    no custom boundary walking needed. Draws a straight baseline between the
    two valley points, subtracts it, clamps negatives to zero, and integrates
    with the trapezoidal rule.

    Returns:
        area in mV·min
    """
    t_seg = time[left_idx : right_idx + 1]
    s_seg = signal[left_idx : right_idx + 1]

    if len(t_seg) < 2 or t_seg[-1] == t_seg[0]:
        return 0.0

    baseline = s_seg[0] + (s_seg[-1] - s_seg[0]) * (t_seg - t_seg[0]) / (t_seg[-1] - t_seg[0])
    corrected = np.maximum(s_seg - baseline, 0)
    return float(trapezoid(corrected, t_seg))


def _fused_groups(detected: list[DetectedPeak]) -> list[list[DetectedPeak]]:
    """Split peaks into chains of fused neighbours (adjacent windows sharing a boundary)."""
    groups: list[list[DetectedPeak]] = []
    for dp in detected:
        if groups and groups[-1][-1].right_base == dp.left_base:
            groups[-1].append(dp)
        else:
            groups.append([dp])
    return groups


def _integrate_drop(time: np.ndarray, signal: np.ndarray, group: list[DetectedPeak]) -> list[float]:
    """Perpendicular-drop integration of one fused group.

    One straight baseline is drawn across the whole group (outer foot to outer
    foot) and each peak's area is the clamped integral of its vertical slice,
    split at the shared valley boundaries. This matches the Shimadzu-style
    vertical-division convention and conserves the group's total area, unlike
    valley-to-valley baselines which cut away the overlap region.
    """
    lo, hi = group[0].left_base, group[-1].right_base
    t_seg = time[lo : hi + 1]
    s_seg = signal[lo : hi + 1]
    if len(t_seg) < 2 or t_seg[-1] == t_seg[0]:
        return [0.0] * len(group)
    baseline = s_seg[0] + (s_seg[-1] - s_seg[0]) * (t_seg - t_seg[0]) / (t_seg[-1] - t_seg[0])
    corrected = np.maximum(s_seg - baseline, 0)
    areas: list[float] = []
    for dp in group:
        i0, i1 = dp.left_base - lo, dp.right_base - lo
        areas.append(float(trapezoid(corrected[i0 : i1 + 1], t_seg[i0 : i1 + 1])))
    return areas


def integrate_all_peaks(
    chrom: Chromatogram, detected: list[DetectedPeak], split_mode: str = "valley"
) -> list[PeakResult]:
    """Integrate every detected peak and return one PeakResult per peak.

    split_mode controls how fused peaks (adjacent windows sharing a valley
    boundary) are handled:
      - "valley" (default): each peak gets its own baseline between its two
        boundary points (valley-to-valley). Conservative; underestimates badly
        fused pairs because the baseline rises to the valley.
      - "drop": one baseline across each fused group, areas split vertically
        at the valleys (perpendicular drop). Conserves the group's total area.
      - "deconvolve": fit each fused group to a sum of exponentially-modified
        Gaussians and report each component's closed-form area; isolated peaks
        and any group whose fit fails fall back to "drop".
    Isolated peaks are identical in all modes.
    """
    if split_mode not in ("valley", "drop", "deconvolve"):
        raise ValueError(
            f"split_mode must be 'valley', 'drop' or 'deconvolve', got {split_mode!r}"
        )

    areas: list[float] = []
    if split_mode == "valley":
        areas = [
            integrate_peak(chrom.time_min, chrom.signal_mV, dp.index, dp.left_base, dp.right_base)
            for dp in detected
        ]
    elif split_mode == "drop":
        for group in _fused_groups(detected):
            areas.extend(_integrate_drop(chrom.time_min, chrom.signal_mV, group))
    else:  # deconvolve: fit fused groups to an EMG sum; isolated peaks use drop.
        for group in _fused_groups(detected):
            if len(group) < 2:
                areas.extend(_integrate_drop(chrom.time_min, chrom.signal_mV, group))
                continue
            fitted, _r2 = deconvolve_group(chrom.time_min, chrom.signal_mV, group)
            if fitted is None:  # non-convergence / poor / blown-up fit -> drop
                areas.extend(_integrate_drop(chrom.time_min, chrom.signal_mV, group))
            else:
                areas.extend(fitted)

    results: list[PeakResult] = []
    for number, (dp, area) in enumerate(zip(detected, areas, strict=True), start=1):
        results.append(
            PeakResult(
                peak_number=number,
                time_min=float(chrom.time_min[dp.index]),
                height_mV=float(chrom.signal_mV[dp.index]),
                area_mV_min=area,
                left_min=float(chrom.time_min[dp.left_base]),
                right_min=float(chrom.time_min[dp.right_base]),
            )
        )
    return results
