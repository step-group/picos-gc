"""Tests for sample-name labelling in label_terpenos.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from label_terpenos import sample_label
from picos_gc.reader import Chromatogram


def _chrom(name):
    return Chromatogram(
        filepath=Path("x.gcd"),
        time_min=np.linspace(0, 1, 3),
        signal_mV=np.zeros(3),
        sample_name=name,
    )


def test_sample_label_returns_real_name():
    assert sample_label(_chrom("I1_T1")) == "I1_T1"


def test_sample_label_blank_for_bnk():
    assert sample_label(_chrom("BNK001")) == "BLANK"


def test_sample_label_blank_for_blanco():
    # block A names its blank "Blanco" (Spanish), not BNK*
    assert sample_label(_chrom("Blanco")) == "BLANK"


def test_sample_label_blank_for_missing():
    assert sample_label(_chrom(None)) == "BLANK"
