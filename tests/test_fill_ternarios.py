"""Unit tests for the KF-anchored normalization + replicate-mismatch QC."""
from __future__ import annotations

import pytest

pytest.importorskip("openpyxl")  # fill_ternarios imports openpyxl at module load
from fill_ternarios import _replicate_mismatch, kf_anchor


def test_kf_anchor_identity_when_closed():
    norm, clo = kf_anchor(0.70, 0.10, 0.10, 0.10)
    assert clo == pytest.approx(1.0)
    assert norm == pytest.approx([0.70, 0.10, 0.10, 0.10])


def test_kf_anchor_fixes_water_and_keeps_ratio():
    # D1 organic: Σ≈0.34; water must stay at the KF value, 2PE:hba ratio preserved,
    # and 2PE lands near the binary (~0.886) instead of the inflated proportional value.
    norm, clo = kf_anchor(0.2515, 0.0048, 0.0052, 0.0788)
    assert clo == pytest.approx(0.3403, abs=1e-3)
    assert norm[3] == pytest.approx(0.0788, abs=1e-4)
    assert sum(norm) == pytest.approx(1.0)
    assert norm[0] / norm[1] == pytest.approx(0.2515 / 0.0048, rel=1e-6)
    assert norm[0] == pytest.approx(0.886, abs=1e-2)


def test_kf_anchor_gc_zero_is_pure_water():
    assert kf_anchor(0.0, 0.0, 0.0, 0.04) == ([0.0, 0.0, 0.0, 1.0], 0.04)


def test_replicate_mismatch_flags_uniform_dilution():
    # D1 organic injections: 356 vs 2194 (and terpenes scale the same ~6.2x).
    g = [{"L": 356.0, "M": 6.4, "N": 5.6}, {"L": 2193.9, "M": 39.0, "N": 34.8}]
    assert _replicate_mismatch(g, 0.25, 0.005, 0.005) == pytest.approx(6.16, abs=0.1)


def test_replicate_mismatch_ignores_trace_peak_wobble():
    # aqueous phase: dominant 2PE fraction is below the constituent gate -> not evaluated.
    g = [{"L": 44.3, "M": 0.0, "N": 0.0}, {"L": 11.1, "M": 0.0, "N": 0.0}]
    assert _replicate_mismatch(g, 0.006, 0.0, 0.0) == 0.0


def test_power_law_response_inverts_and_bends_below_linear():
    from fill_ternarios import CC_MF, PURITY, Response, cc_mf_models

    r = Response(150.0, 1.05)
    assert r.conc(150.0 * 10.0**1.05) == pytest.approx(10.0)
    assert r.conc(0.0) == 0.0  # a blank reads zero
    models = cc_mf_models(CC_MF)
    assert set(models) == set(PURITY)
    # beta > 1 in every compound: the curve that made through-origin linear read the
    # lowest standards 9-23 % low
    assert all(1.02 < m.beta < 1.07 for m in models.values())


def test_calibration_areas_come_from_the_chromatograms():
    # CC_MF col O was integrated by hand and missed its chromatograms by up to 7 %
    # (camphor's scatter 4.9 % in ln A); label_terpenos' areas hold every compound's
    # six standards within 2.5 %
    import math

    import numpy as np

    from fill_ternarios import CC_MF, PURITY, calibration_points, canon

    by: dict[str, list[tuple[float, float]]] = {}
    for sheet, _, w, a in calibration_points(CC_MF):
        k = canon(sheet)
        by.setdefault(k, []).append((math.log(w * PURITY[k]), math.log(a)))
    assert set(by) == set(PURITY) and all(len(p) == 6 for p in by.values())
    for k, pts in by.items():
        x, y = np.array(pts).T
        b, c = np.polyfit(x, y, 1)
        r = y - c - b * x
        assert math.sqrt(r @ r / 4) < 0.025, k
