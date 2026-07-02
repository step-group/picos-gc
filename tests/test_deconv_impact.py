"""Tests for the deconvolve-vs-drop impact report."""

from __future__ import annotations

import numpy as np

from deconv_impact import peak_area_deltas
from picos_gc.deconvolution import emg
from picos_gc.detector import DetectionParams
from tests.helpers import make_chrom


def test_peak_area_deltas_flags_fused_pair():
    # Two tailing peaks close enough to fuse (detected as a pair): drop's vertical
    # split and EMG deconvolution disagree, so the report shows a non-zero
    # pct_change for at least one peak.
    t = np.linspace(8.5, 9.8, 40000)
    sig = emg(t, 10.0, 9.0, 0.03, 0.04) + emg(t, 6.0, 9.15, 0.03, 0.04)
    chrom = make_chrom(sig, t_start=8.5, t_end=9.8)
    rows = peak_area_deltas(chrom, DetectionParams(merge_shoulder_ratio=0.0), baseline_method=None)
    assert len(rows) == 2
    assert max(abs(r["pct_change"]) for r in rows) > 5.0
