"""Peak detection for GC chromatograms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .reader import Chromatogram


@dataclass
class DetectionParams:
    min_height: float = 50.0  # mV
    min_prominence: float = 20.0  # mV
    min_distance: int = 50  # data points


@dataclass
class DetectedPeak:
    index: int  # index into the raw signal array
    left_base: int  # left valley index (from scipy prominence)
    right_base: int  # right valley index (from scipy prominence)


def detect_peaks(chrom: Chromatogram, params: DetectionParams) -> list[DetectedPeak]:
    """Detect all peaks and their valley boundaries in a chromatogram.

    Valley boundaries (left_base / right_base) come directly from scipy's
    prominence calculation, giving correct valley-to-valley splits for
    overlapping peaks. The prominence threshold already suppresses noise spikes
    without any additional smoothing.

    Falls back to looser thresholds (height=10, prominence=5) if nothing is
    found with the supplied params.

    Returns peaks sorted by retention time (left → right).
    """
    signal = chrom.signal_mV

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
    detected = [
        DetectedPeak(
            index=int(peaks[i]),
            left_base=int(props["left_bases"][i]),
            right_base=int(props["right_bases"][i]),
        )
        for i in order
    ]

    # Clip overlapping boundaries: scipy sets a tall peak's base far back past
    # shorter neighbours. For each consecutive pair whose bases overlap, use the
    # signal valley between the two peak maxima as the shared boundary.
    raw = chrom.signal_mV
    for a, b in zip(detected, detected[1:]):
        if a.right_base > b.left_base:
            valley = int(np.argmin(raw[a.index : b.index + 1])) + a.index
            a.right_base = valley
            b.left_base = valley

    return detected
