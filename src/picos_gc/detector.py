"""Peak detection for GC chromatograms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths

from .reader import Chromatogram


@dataclass
class DetectionParams:
    min_height: float = 50.0  # mV
    min_prominence: float = 20.0  # mV
    min_distance: int = 50  # data points


@dataclass
class DetectedPeak:
    index: int  # index into the raw signal array
    left_base: int  # left boundary index
    right_base: int  # right boundary index


def detect_peaks(chrom: Chromatogram, params: DetectionParams) -> list[DetectedPeak]:
    """Detect all peaks and their integration boundaries in a chromatogram.

    Boundary strategy:
      1. `peak_widths` at rel_height=1.0 finds where the signal crosses the
         prominence reference level (local valley floor) on each side — tight,
         physically meaningful boundaries that don't wander into flat baseline.
      2. Valley-clipping: if any adjacent pair still overlaps after step 1,
         the shared boundary is set to the signal minimum between the two peaks.

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

    # peak_widths at rel_height=1.0: boundaries at the prominence reference
    # level (local valley floor). Reuses prominence data already computed by
    # find_peaks to avoid a redundant pass over the signal.
    _, _, left_ips, right_ips = peak_widths(
        signal,
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
        )
        for i in order
    ]

    # Safety net: clip any remaining overlaps to the valley minimum
    for a, b in zip(detected, detected[1:]):
        if a.right_base > b.left_base:
            valley = int(np.argmin(signal[a.index : b.index + 1])) + a.index
            a.right_base = valley
            b.left_base = valley

    return detected
