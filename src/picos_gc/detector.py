"""Peak detection for GC chromatograms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .reader import Chromatogram


@dataclass
class DetectionParams:
    min_height: float = 500.0  # mV
    min_prominence: float = 20.0  # mV
    min_distance: int = 50  # data points
    smooth_window: int = 11  # Savitzky-Golay window (odd); 0 = disabled
    smooth_polyorder: int = 3  # SG polynomial order


@dataclass
class DetectedPeak:
    index: int  # index into the raw signal array
    left_base: int  # left valley index (from scipy prominence)
    right_base: int  # right valley index (from scipy prominence)


def detect_peaks(chrom: Chromatogram, params: DetectionParams) -> list[DetectedPeak]:
    """Detect all peaks and their valley boundaries in a chromatogram.

    Applies optional Savitzky-Golay smoothing before calling find_peaks so
    that noise spikes are suppressed. The returned indices always refer to the
    *raw* (unsmoothed) signal.

    Valley boundaries (left_base / right_base) come directly from scipy's
    prominence calculation, giving correct valley-to-valley splits for
    overlapping peaks without any custom boundary-walking.

    Falls back to looser thresholds (height=10, prominence=5) if nothing is
    found with the supplied params.

    Returns peaks sorted by retention time (left → right).
    """
    signal = chrom.signal_mV

    if params.smooth_window > 0:
        window = params.smooth_window
        if window >= len(signal):
            window = len(signal) // 2 * 2 - 1
        signal = savgol_filter(signal, window, params.smooth_polyorder)

    peaks, props = find_peaks(
        signal,
        height=params.min_height,
        prominence=params.min_prominence,
        distance=params.min_distance,
    )

    if len(peaks) == 0:
        peaks, props = find_peaks(
            signal,
            height=10.0,
            prominence=5.0,
            distance=params.min_distance,
        )

    order = np.argsort(peaks)
    return [
        DetectedPeak(
            index=int(peaks[i]),
            left_base=int(props["left_bases"][i]),
            right_base=int(props["right_bases"][i]),
        )
        for i in order
    ]
