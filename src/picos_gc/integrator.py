"""Peak integration with linear baseline subtraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from .detector import DetectedPeak
from .reader import Chromatogram


@dataclass
class PeakResult:
    peak_number: int   # 1-based, ordered by retention time
    time_min: float    # retention time (tR)
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


def integrate_all_peaks(
    chrom: Chromatogram, detected: list[DetectedPeak]
) -> list[PeakResult]:
    """Integrate every detected peak and return one PeakResult per peak."""
    results: list[PeakResult] = []
    for number, dp in enumerate(detected, start=1):
        area = integrate_peak(
            chrom.time_min, chrom.signal_mV, dp.index, dp.left_base, dp.right_base
        )
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
