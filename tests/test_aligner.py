"""Tests for cross-file peak alignment and aligned-CSV output."""

from __future__ import annotations

import csv
from pathlib import Path

from picos_gc.aligner import align_peaks, save_aligned_csv
from picos_gc.integrator import PeakResult
from picos_gc.processor import FileResult


def _file_result(name: str, peaks: list[tuple[float, float]]) -> FileResult:
    """Build a FileResult from (tR, area) pairs."""
    prs = [
        PeakResult(
            peak_number=i + 1,
            time_min=tr,
            height_mV=area,  # height irrelevant to alignment
            area_mV_min=area,
            left_min=tr - 0.05,
            right_min=tr + 0.05,
        )
        for i, (tr, area) in enumerate(peaks)
    ]
    return FileResult(filepath=Path(name), filename=name, chromatogram=None, peaks=prs)


def test_two_compounds_clustered_by_tr():
    results = [
        _file_result("a.gcd", [(2.00, 100.0), (5.00, 200.0)]),
        _file_result("b.gcd", [(2.03, 110.0), (5.02, 190.0)]),
        _file_result("c.gcd", [(1.98, 90.0), (4.99, 210.0)]),
    ]
    al = align_peaks(results, tol_min=0.1)
    assert len(al.compounds) == 2
    c1, c2 = al.compounds
    assert c1.median_tR < c2.median_tR
    assert c1.n_detected == 3
    assert c2.n_detected == 3
    # Every file matched both compounds.
    for row in al.table.values():
        assert all(pair[0] is not None for pair in row)


def test_gap_larger_than_tol_splits_clusters():
    results = [_file_result("a.gcd", [(2.0, 100.0), (2.5, 100.0)])]
    one = align_peaks(results, tol_min=1.0)
    assert len(one.compounds) == 1  # 0.5 gap <= 1.0 tol -> single cluster
    two = align_peaks(results, tol_min=0.1)
    assert len(two.compounds) == 2  # 0.5 gap > 0.1 tol -> split


def test_missing_peak_is_none_in_table():
    results = [
        _file_result("a.gcd", [(2.0, 100.0), (5.0, 200.0)]),
        _file_result("b.gcd", [(2.0, 100.0)]),  # missing the 5.0 compound
    ]
    al = align_peaks(results, tol_min=0.1)
    assert len(al.compounds) == 2
    # File b has no peak for compound 2.
    assert al.table[1][1] == (None, None)
    assert al.compounds[1].n_detected == 1


def test_rsd_and_stats():
    results = [
        _file_result("a.gcd", [(3.0, 100.0)]),
        _file_result("b.gcd", [(3.0, 110.0)]),
        _file_result("c.gcd", [(3.0, 90.0)]),
    ]
    al = align_peaks(results, tol_min=0.1)
    (c,) = al.compounds
    assert c.mean_area == 100.0
    assert c.n_detected == 3
    assert c.rsd_pct > 0
    assert c.std_area > 0


def test_empty_results():
    al = align_peaks([], tol_min=0.1)
    assert al.compounds == []


def test_save_aligned_csv_roundtrip(tmp_path):
    results = [
        _file_result("a.gcd", [(2.0, 100.0), (5.0, 200.0)]),
        _file_result("b.gcd", [(2.0, 100.0)]),
    ]
    al = align_peaks(results, tol_min=0.1)
    out = tmp_path / "aligned.csv"
    save_aligned_csv(al, results, out)
    rows = list(csv.reader(out.open()))
    header = rows[0]
    assert header[0] == "sample_name"
    assert header[1] == "filename"
    assert "cmp1_tR_min" in header and "cmp1_area_mV_min" in header
    # Data rows for both files exist (filename now at index 1).
    data_filenames = {r[1] for r in rows[1:] if len(r) > 1 and r[1] in {"a.gcd", "b.gcd"}}
    assert data_filenames == {"a.gcd", "b.gcd"}
    # Footer labels present.
    labels = {r[0] for r in rows if r}
    for lbl in ("median_tR", "tR_std", "mean_area", "std_area", "rsd_pct", "n_detected"):
        assert lbl in labels
