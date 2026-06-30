"""Global baseline correction via the pybaselines library.

Estimates a slowly-varying baseline over the whole chromatogram and subtracts
it, so downstream peak detection and integration both operate on a flattened
signal. Runs before detection in the processing pipeline.
"""

from __future__ import annotations

import numpy as np

# Curated whole-signal correctors we expose. Each is a method on a
# pybaselines.Baseline instance taking (signal, ...) -> (baseline, params).
ALLOWED_METHODS = ("asls", "airpls", "arpls", "snip")

# arPLS: robust to noise and large peaks with little tuning — good GC default.
DEFAULT_METHOD = "arpls"


def correct_baseline(
    signal: np.ndarray,
    time: np.ndarray,
    method: str = DEFAULT_METHOD,
    lam: float | None = None,
) -> np.ndarray:
    """Return *signal* with a globally-estimated baseline subtracted.

    Negatives are deliberately NOT clamped here: the integrator's per-peak
    linear chord still needs the signal on both sides of zero to locate the
    local baseline, and rectifying noise to zero would bias areas upward.

    Args:
        signal: 1-D signal array (mV).
        time:   matching 1-D time array (min), used as the x-grid.
        method: one of ALLOWED_METHODS.
        lam:    penalty strength for asls/airpls/arpls (ignored by snip).
                None -> each method's own pybaselines default.

    Raises:
        ValueError: unknown method, mismatched/non-1-D arrays, or a
                    non-finite result.
    """
    if method not in ALLOWED_METHODS:
        raise ValueError(
            f"unknown baseline method {method!r}; choose from {', '.join(ALLOWED_METHODS)}"
        )
    if signal.shape != time.shape:
        raise ValueError(
            f"signal and time must have the same shape, got {signal.shape} and {time.shape}"
        )
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if len(signal) < 2:
        return np.asarray(signal, dtype=float)

    from pybaselines import Baseline

    fitter = Baseline(x_data=np.asarray(time, dtype=float))
    fit = getattr(fitter, method)  # method already validated against ALLOWED_METHODS
    kwargs = {} if (method == "snip" or lam is None) else {"lam": lam}
    baseline, _ = fit(np.asarray(signal, dtype=float), **kwargs)

    corrected = np.asarray(signal, dtype=float) - np.asarray(baseline, dtype=float)
    if not np.all(np.isfinite(corrected)):
        raise ValueError(
            f"baseline method {method!r} produced non-finite values; "
            "try a different method or --baseline-lam"
        )
    return corrected
