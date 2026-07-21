"""Tests for robust noise estimation and auto (noise-based) detection thresholds."""

from __future__ import annotations

import numpy as np
import pytest

from picos_gc.detector import DetectionParams, detect_peaks, estimate_noise
from tests.helpers import gaussian, make_chrom


def test_estimate_noise_recovers_sigma():
    noise = np.random.default_rng(0).normal(0, 0.05, 20000)
    assert estimate_noise(noise) == pytest.approx(0.05, rel=0.1)


def test_estimate_noise_robust_to_peaks():
    # A tall narrow peak (a sparse outlier) must not inflate the MAD estimate.
    noise = np.random.default_rng(1).normal(0, 0.05, 20000)
    noise[10000] += 500.0
    assert estimate_noise(noise) == pytest.approx(0.05, rel=0.1)


def test_estimate_noise_flat_signal_is_zero():
    assert estimate_noise(np.zeros(1000)) == 0.0
    assert estimate_noise(np.array([1.0])) == 0.0  # too short


def test_auto_threshold_detects_peak_above_noise_rejects_bump():
    # sigma≈0.05 → auto height = 10sigma ≈ 0.5 mV. A 1.0 mV peak is in, a 0.1 mV bump is out.
    t = np.linspace(0, 10, 5000)
    sig = (
        np.random.default_rng(2).normal(0, 0.05, t.size)
        + gaussian(t, 3.0, 1.0, 0.05)
        + gaussian(t, 6.0, 0.1, 0.05)
    )
    chrom = make_chrom(sig)
    peaks = detect_peaks(chrom, DetectionParams())  # all-auto: 10sigma height, 5sigma prominence
    trs = [round(float(chrom.time_min[p.index]), 1) for p in peaks]
    assert trs == [3.0]


def test_explicit_height_overrides_auto():
    # A fixed min_height still works (pre-auto behaviour) and ignores the noise.
    t = np.linspace(0, 10, 5000)
    sig = np.random.default_rng(3).normal(0, 0.05, t.size) + gaussian(t, 3.0, 1.0, 0.05)
    chrom = make_chrom(sig)
    # 5.0 mV cut rejects the 1.0 mV peak entirely.
    assert detect_peaks(chrom, DetectionParams(min_height=5.0, min_prominence=2.0)) == []
