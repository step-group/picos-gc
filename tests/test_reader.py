"""Tests for the .gcd reader and time/index helpers."""

from __future__ import annotations

import numpy as np
import pytest

from picos_gc.reader import read_gcd, time_to_index
from tests.helpers import make_chrom


def test_read_gcd_basic_shape(example_gcd):
    chrom = read_gcd(example_gcd)
    assert chrom.time_min.shape == chrom.signal_mV.shape
    assert chrom.time_min.ndim == 1
    assert len(chrom.time_min) > 100
    # Time axis is increasing and starts at zero.
    assert chrom.time_min[0] == pytest.approx(0.0)
    assert chrom.time_min[-1] > chrom.time_min[0]
    assert np.all(np.diff(chrom.time_min) > 0)
    # There is a real signal in there.
    assert float(chrom.signal_mV.max()) > 0


def test_read_gcd_str_path(example_gcd):
    # Accepts both str and Path.
    chrom = read_gcd(str(example_gcd))
    assert len(chrom.signal_mV) > 0


def test_read_gcd_rejects_non_ole(tmp_path):
    bad = tmp_path / "not_really.gcd"
    bad.write_bytes(b"this is not an OLE2 document")
    with pytest.raises(ValueError):
        read_gcd(bad)


def test_time_to_index_endpoints_and_clamp():
    chrom = make_chrom(np.zeros(101), t_start=0.0, t_end=10.0)
    assert time_to_index(chrom, 0.0) == 0
    assert time_to_index(chrom, 10.0) == 100
    assert time_to_index(chrom, 5.0) == 50
    # Out-of-range values clamp to the ends.
    assert time_to_index(chrom, -1.0) == 0
    assert time_to_index(chrom, 99.0) == 100


def test_time_to_index_zero_span():
    chrom = make_chrom(np.zeros(10), t_start=3.0, t_end=3.0)
    assert time_to_index(chrom, 3.0) == 0


def test_data_starts_at_offset_24_not_16(example_gcd):
    """Regression: the signal stream's data begins at byte 24.

    Reading from byte 16 (the old behavior) prepended the 8 zero-padding
    bytes as a spurious leading 0.0 sample and dropped the last real sample.
    """
    import struct

    import olefile

    ole = olefile.OleFileIO(example_gcd)
    raw = ole.openstream("LSS Raw Data/Chromatogram Ch1").read()
    ole.close()

    assert raw[:2] == b"RC"  # magic
    assert raw[16:24] == b"\x00" * 8  # zero padding before the data
    first_raw_sample_uV = struct.unpack("<d", raw[24:32])[0]

    chrom = read_gcd(example_gcd)
    # The reader must surface the real first sample, not the padding zero.
    assert chrom.signal_mV[0] != 0.0
    assert chrom.signal_mV[0] * 1000.0 == pytest.approx(first_raw_sample_uV)
