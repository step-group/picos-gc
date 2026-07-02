"""EMG peak deconvolution: fit fused groups as a sum of exponentially-modified
Gaussians and report each component's closed-form area.

The exponentially-modified Gaussian models chromatographic tailing (a Gaussian
convolved with a one-sided exponential). Parametrised by its area so the fitted
`area` is directly the peak integral (mV*min when fit against the minute axis).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.special import erfcx

from .detector import DetectedPeak

_R2_MIN = 0.90  # reject fits worse than this; caller falls back to drop
_AREA_BLOWUP = 1.5  # reject if total fitted area exceeds this x the raw integral


def emg(t, area, mu, sigma, tau):
    """Area-normalised exponentially-modified Gaussian; ``integral(emg) == area``.

    Numerically stable (Kalambet 2011) form using ``erfcx(z) = exp(z**2)*erfc(z)``:

        emg = area/(2*tau) * exp(-0.5*((t-mu)/sigma)**2) * erfcx(z),
        z = (sigma/tau - (t-mu)/sigma) / sqrt(2)

    ``tau -> 0`` recovers a Gaussian of the same area. Deep in the exponential
    tail the intermediate ``erfcx`` can overflow while the true value has already
    underflowed to ~0; those points are zeroed so the optimiser never sees NaN.
    """
    t = np.asarray(t, dtype=float)
    sigma = max(float(sigma), 1e-9)
    tau = max(float(tau), 1e-9)
    dt = t - mu
    z = (sigma / tau - dt / sigma) / np.sqrt(2.0)
    # Deep in the tail erfcx(z) overflows while exp(...) underflows: the 0*inf is
    # an expected, handled artifact (true value ~0), so silence its warning.
    with np.errstate(over="ignore", invalid="ignore"):
        val = (area / (2.0 * tau)) * np.exp(-0.5 * (dt / sigma) ** 2) * erfcx(z)
    return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)


def _emg_sum(t, *params):
    """Sum of N EMGs. ``params`` is ``[area, mu, sigma, tau]`` repeated N times."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for i in range(len(params) // 4):
        area, mu, sigma, tau = params[4 * i : 4 * i + 4]
        out = out + emg(t, area, mu, sigma, tau)
    return out


def deconvolve_group(
    time_min, signal_mV, group: list[DetectedPeak]
) -> tuple[list[float] | None, float]:
    """Fit a sum of EMGs to one fused group; return (areas_mV_min, r2).

    The fit runs on the group's linear-baseline-subtracted signal in a
    window-local time coordinate (t - t_lo) for conditioning; since the EMG is
    area-normalised, each component's fitted ``area`` is directly its integral in
    mV*min. Seeds come from the detected apexes; bounds keep areas non-negative,
    centres inside the window, and widths above the sampling interval. Returns
    ``(None, r2)`` when the fit fails, is poor (r2 < 0.90), or blows up (total
    area > 1.5x the raw group integral) so the caller can fall back to drop.
    """
    lo, hi = group[0].left_base, group[-1].right_base
    t = np.asarray(time_min[lo : hi + 1], dtype=float)
    y = np.asarray(signal_mV[lo : hi + 1], dtype=float)
    if len(t) < 4 or t[-1] == t[0]:
        return None, 0.0

    base = y[0] + (y[-1] - y[0]) * (t - t[0]) / (t[-1] - t[0])
    yc = y - base
    ts = t - t[0]  # local coordinate; centre seeds become small

    raw_area = float(trapezoid(np.maximum(yc, 0.0), t))
    span = float(ts[-1])
    dt_min = float(np.median(np.diff(ts)))
    sigma_floor = max(dt_min, 1e-6)

    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for dp in group:
        mu = float(time_min[dp.index] - t[0])
        h = max(float(signal_mV[dp.index] - base[dp.index - lo]), 1e-6)
        sig = max((time_min[dp.right_base] - time_min[dp.left_base]) / 4.0, 2 * sigma_floor)
        area0 = h * sig * np.sqrt(2 * np.pi)
        p0 += [area0, mu, sig, sig]
        lower += [0.0, max(mu - sig, 0.0), sigma_floor, sigma_floor]
        upper += [10.0 * (raw_area + 1.0), min(mu + sig, span), span, span]

    try:
        popt, _ = curve_fit(_emg_sum, ts, yc, p0=p0, bounds=(lower, upper), maxfev=10000)
    except (RuntimeError, ValueError):
        return None, 0.0

    fit = _emg_sum(ts, *popt)
    ss_res = float(np.sum((yc - fit) ** 2))
    ss_tot = float(np.sum((yc - yc.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot

    areas = [float(popt[4 * i]) for i in range(len(group))]
    if r2 < _R2_MIN or not all(np.isfinite(a) and a >= 0 for a in areas):
        return None, r2
    if sum(areas) > _AREA_BLOWUP * (raw_area + 1e-9):
        return None, r2
    return areas, r2
