"""EMG peak deconvolution: fit fused groups as a sum of exponentially-modified
Gaussians and report each component's closed-form area.

The exponentially-modified Gaussian models chromatographic tailing (a Gaussian
convolved with a one-sided exponential). Parametrised by its area so the fitted
`area` is directly the peak integral (mV*min when fit against the minute axis).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfcx


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
