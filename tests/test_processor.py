"""Tests for the batch processor CSV output."""

from __future__ import annotations

import csv
from pathlib import Path

from picos_gc.integrator import PeakResult
from picos_gc.processor import FileResult, save_csv


def test_save_csv_has_sample_name_first_column(tmp_path):
    peak = PeakResult(
        peak_number=1,
        time_min=2.0,
        height_mV=10.0,
        area_mV_min=5.0,
        left_min=1.9,
        right_min=2.1,
    )
    results = [
        FileResult(
            filepath=Path("x.gcd"),
            filename="x.gcd",
            chromatogram=None,
            peaks=[peak],
            sample_name="I1_T1",
        )
    ]
    out = tmp_path / "tidy.csv"
    save_csv(results, out)
    rows = list(csv.reader(out.open()))
    assert rows[0][:2] == ["sample_name", "filename"]
    assert rows[1][0] == "I1_T1"
    assert rows[1][1] == "x.gcd"
