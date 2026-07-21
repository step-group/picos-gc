"""Peak detection for GC chromatograms."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import median_abs_deviation

from .reader import Chromatogram

# Looser thresholds used only if the supplied params find nothing at all, or
# when auto-thresholding is requested but the signal is flat (noise sigma == 0).
_FALLBACK_HEIGHT = 10.0
_FALLBACK_PROMINENCE = 5.0


def estimate_noise(signal: np.ndarray) -> float:
    """Robust noise sigma via Gaussian-scaled MAD (1.4826·MAD).

    Sparse peaks don't move the median, so on a baseline-corrected chromatogram
    this reflects the flat-baseline noise rather than the peaks. Returns 0.0 for
    a degenerate/flat signal, letting the caller fall back to a fixed threshold.

    Not the derivative-MAD: GC data is heavily oversampled, so adjacent points
    barely differ and diff-based sigma underestimates the real noise floor ~10x.
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size < 2:
        return 0.0
    return float(median_abs_deviation(sig, scale="normal", nan_policy="omit"))


@dataclass
class DetectionParams:
    # None -> auto: threshold = noise_snr_* x robust noise sigma (see estimate_noise).
    # A number forces a fixed mV cut (the pre-auto behaviour; use min_height=50).
    min_height: float | None = None  # mV, or None for auto
    min_prominence: float | None = None  # mV, or None for auto
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
    # Window clipping: shrink each integration window to the contiguous region
    # around the apex where the chord-corrected signal stays above this
    # fraction of the apex height. Prominence-base boundaries can sit minutes
    # away from the peak in noisy baseline, inflating zero-clamped areas.
    # 0 disables clipping (legacy prominence-base boundaries).
    clip_window_frac: float = 0.001
    # Auto-threshold factors (used only when min_height/min_prominence is None):
    # threshold = factor x noise sigma. 10sigma ≈ limit of quantitation, 5sigma prominence.
    noise_snr_height: float = 10.0
    noise_snr_prominence: float = 5.0


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
    # Apex span: for a merged envelope, the leftmost and rightmost member apex.
    # Window clipping never shrinks inside this span, so a merged envelope
    # keeps all its humps even when the valley between them touches baseline.
    # Defaults to the apex itself for unmerged peaks.
    apex_left: int = -1
    apex_right: int = -1

    def __post_init__(self) -> None:
        if self.base_left < 0:
            self.base_left = self.left_base
        if self.base_right < 0:
            self.base_right = self.right_base
        if self.apex_left < 0:
            self.apex_left = self.index
        if self.apex_right < 0:
            self.apex_right = self.index


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
                apex_left=current.apex_left,
                apex_right=nxt.apex_right,
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
                apex_left=current.apex_left,
                apex_right=nxt.apex_right,
            )
            last_apex = nxt.index
            i += 1
        out.append(current)
        i += 1
    return out


def _clip_overlaps(detected: list[DetectedPeak], signal: np.ndarray) -> None:
    """Clip any overlapping or shared windows of adjacent peaks to the valley minimum."""
    for a, b in itertools.pairwise(detected):
        if a.right_base > b.left_base:
            valley = int(np.argmin(signal[a.index : b.index + 1])) + a.index
            a.right_base = valley
            b.left_base = valley


def _side_floor(signal: np.ndarray, seg_start: int, seg_end: int, at_start: bool) -> float | None:
    """Robust local-baseline estimate: median of the samples at one end of a range."""
    seg = signal[seg_start:seg_end]
    if len(seg) < 2:
        return None
    m = max(5, len(seg) // 50)
    return float(np.median(seg[:m] if at_start else seg[-m:]))


# A shared valley boundary marks a genuine fusion (kept exactly, so drop-mode
# grouping and valley splits survive) only when the valley rises above this
# fraction of the smaller corrected apex; otherwise the shared boundary is an
# artifact of over-wide prominence-base windows and both sides are clipped.
_FUSED_VALLEY_RATIO = 0.05

# A fused group's outer boundary is searched beyond the integration window
# (out to the prominence bases) only when it is truncated mid-flank — i.e. it
# sits above the local floor by at least this fraction of the member's own
# corrected apex height. Boundaries already at floor level never extend.
_TRUNCATED_BOUNDARY_RATIO = 0.25


def _is_fused(signal: np.ndarray, a: DetectedPeak, b: DetectedPeak) -> bool:
    """True when the shared boundary between *a* and *b* is a genuine fused valley.

    Resolved peaks can end up sharing a boundary merely because their legacy
    prominence-base windows overlapped and were split at the inter-peak noise
    minimum; the valley then sits at (or below) the local baseline. A genuine
    fusion has its valley well above the baseline. The baseline floor is taken
    as the lower of the two robust medians at the pair's outermost ends.

    Known limitation: two small peaks separated by a long *elevated plateau*
    (e.g. column bleed) read as fused, because the plateau keeps the valley
    above the global floor; their windows then stay valley-to-valley across
    the plateau, as in the legacy behaviour. A gap-width-aware criterion
    could refine this.
    """
    valley = float(signal[a.right_base])
    floors = [
        f
        for f in (
            _side_floor(signal, min(a.base_left, a.left_base), a.apex_left, at_start=True),
            _side_floor(signal, b.apex_right + 1, max(b.base_right, b.right_base) + 1, at_start=False),
        )
        if f is not None
    ]
    if not floors:
        return True  # nothing to estimate a baseline from — keep the valley split
    floor = min(floors)
    smaller_apex = min(float(signal[a.index]), float(signal[b.index]))
    if smaller_apex <= floor:
        return True
    return (valley - floor) > _FUSED_VALLEY_RATIO * (smaller_apex - floor)


def _clip_windows(detected: list[DetectedPeak], signal: np.ndarray, frac: float) -> None:
    """Shrink each integration window to the signal region around its apex.

    ``peak_widths(rel_height=1.0)`` anchors boundaries at the prominence
    reference level, which for dominant or isolated peaks can sit minutes away
    from the apex; the zero-clamped integral then accumulates baseline noise
    and drift over the whole slack region, inflating areas (worst observed on
    real data: +307% on a small peak). This pass walks outward from the apex
    span and stops where the chord-corrected signal falls below ``frac`` times
    the apex height, leaving boundaries at the local peak feet.

    Clipping is inward for all ordinary boundaries: each side is searched only
    between the existing boundary and the apex span, so a window cannot grow.
    (Extending every boundary over the prominence-base range is unsafe: for a
    small peak the threshold ``frac * height`` lies below the baseline ripple,
    and the extension swallowed minutes of baseline — observed +189% on a
    58 mV peak between two large neighbours.) The side's threshold is anchored
    on a robust floor estimate — the median of the samples adjacent to the far
    end of the search range — rather than a chord between the two window
    endpoints: a chord anchored on a deep noise minimum or a high fused valley
    tilts the whole correction and clips on the flank or not at all. The
    boundary moves to the below-threshold sample nearest the apex span.

    The one place a boundary may move *outward* is the outer side of a
    multi-peak fused group, and only when that boundary is demonstrably
    truncated mid-flank (it sits above the local floor by more than
    ``_TRUNCATED_BOUNDARY_RATIO`` of the member's own corrected apex — the
    smaller member's ``rel_height=1.0`` window stops at the valley contour
    level, halfway down its flank). The search range is then widened to the
    prominence bases so drop-mode integration can anchor the group baseline
    at the true feet and conserve the group's total area. The threshold always
    derives from the member's *own* apex: a group-wide apex once collapsed a
    58 mV member riding a 72 000 mV neighbour's tail to a zero-width window,
    because the group-scaled threshold exceeded the member's entire height.

    Boundaries shared with a *genuinely fused* neighbour (see ``_is_fused``)
    are left untouched so valley splits and drop-mode grouping survive; shared
    boundaries that merely stem from overlapping legacy windows between
    resolved peaks are clipped inward like any other boundary.
    ``frac <= 0`` disables clipping entirely.
    """
    if frac <= 0:
        return
    # Decide fusion per shared boundary before any boundary is mutated.
    fused_after = [
        i + 1 < len(detected)
        and detected[i + 1].left_base == d.right_base
        and _is_fused(signal, d, detected[i + 1])
        for i, d in enumerate(detected)
    ]
    # Fused groups as (first, last) index pairs; group apex height drives the
    # threshold for outer-side extension of multi-peak groups.
    groups: list[tuple[int, int]] = []
    g = 0
    for i in range(len(detected)):
        if i == len(detected) - 1 or not fused_after[i]:
            groups.append((g, i))
            g = i + 1
    group_of = {}
    for gs, ge in groups:
        for k in range(gs, ge + 1):
            group_of[k] = (gs, ge)

    for i, d in enumerate(detected):
        gs, ge = group_of[i]
        multi = ge > gs
        apex_val = float(signal[d.index])
        if i == gs:  # left side is an outer (or ordinary) boundary
            start = d.left_base
            if multi:
                # Widen the search to the prominence bases only when the
                # boundary is truncated mid-flank (well above the local floor);
                # a boundary already at floor level has nothing to extend to.
                ext = min(d.base_left, d.left_base)
                probe = signal[ext : d.apex_left]
                if len(probe) >= 2:
                    m = max(5, len(probe) // 50)
                    ext_floor = float(np.median(probe[:m]))
                    bound_val = float(signal[d.left_base])
                    if bound_val - ext_floor > _TRUNCATED_BOUNDARY_RATIO * (apex_val - ext_floor):
                        start = ext
            seg = signal[start : d.apex_left]
            if len(seg) >= 2:
                m = max(5, len(seg) // 50)
                floor = float(np.median(seg[:m]))
                if apex_val > floor:
                    thr = floor + frac * (apex_val - floor)
                    below = np.nonzero(seg <= thr)[0]
                    if len(below):
                        d.left_base = start + int(below[-1])
        if i == ge:  # right side is an outer (or ordinary) boundary
            end = d.right_base
            if multi:
                ext = max(d.base_right, d.right_base)
                probe = signal[d.apex_right + 1 : ext + 1]
                if len(probe) >= 2:
                    m = max(5, len(probe) // 50)
                    ext_floor = float(np.median(probe[-m:]))
                    bound_val = float(signal[d.right_base])
                    if bound_val - ext_floor > _TRUNCATED_BOUNDARY_RATIO * (apex_val - ext_floor):
                        end = ext
            seg = signal[d.apex_right + 1 : end + 1]
            if len(seg) >= 2:
                m = max(5, len(seg) // 50)
                floor = float(np.median(seg[-m:]))
                if apex_val > floor:
                    thr = floor + frac * (apex_val - floor)
                    below = np.nonzero(seg <= thr)[0]
                    if len(below):
                        d.right_base = d.apex_right + 1 + int(below[0])


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
      5. Optional shoulder merge (baseline-aware) and proximity merge, followed
         by a second overlap pass (merged envelopes take their boundaries from
         prominence bases, which can overlap a neighbour's window).
      6. Window clipping: boundaries pulled in to the local peak feet
         (see ``_clip_windows``); fused/shared boundaries stay at the valley.

    Returns peaks sorted by retention time (left → right).
    """
    raw = chrom.signal_mV
    detect_signal = _smoothed(raw, params.smooth_window, params.smooth_polyorder)

    # Convert min_width from minutes to data points using the actual sampling rate.
    span = float(chrom.time_min[-1] - chrom.time_min[0])
    pts_per_min = len(chrom.time_min) / span if span > 0 else 0.0
    width_pts = params.min_width_min * pts_per_min if pts_per_min > 0 else 0.0
    width_arg = width_pts if width_pts > 0 else None

    # Resolve auto thresholds (None) from the measured noise sigma; a flat signal
    # (sigma == 0) falls back to the fixed looser constants.
    sigma = estimate_noise(detect_signal)
    height = (
        params.min_height
        if params.min_height is not None
        else (params.noise_snr_height * sigma if sigma > 0 else _FALLBACK_HEIGHT)
    )
    prominence = (
        params.min_prominence
        if params.min_prominence is not None
        else (params.noise_snr_prominence * sigma if sigma > 0 else _FALLBACK_PROMINENCE)
    )

    peaks, props = find_peaks(
        detect_signal,
        height=height,
        prominence=prominence,
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
    _clip_overlaps(detected, detect_signal)

    detected = _merge_shoulders(detected, detect_signal, params.merge_shoulder_ratio)
    detected = _merge_proximity(detected, detect_signal, chrom.time_min, params.merge_distance_min)

    # Merged envelopes take prominence-base boundaries, which can overlap the
    # next peak's window — clip overlaps again before shrinking windows.
    _clip_overlaps(detected, detect_signal)
    _clip_windows(detected, detect_signal, params.clip_window_frac)
    # A fused group's outer boundary may have extended into a neighbour's
    # window; re-split any such overlap at the valley.
    _clip_overlaps(detected, detect_signal)
    return detected
