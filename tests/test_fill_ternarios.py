"""Unit tests for the KF-anchored normalization, the replicate screen and the transfer."""
from __future__ import annotations

import pytest

pytest.importorskip("openpyxl")  # fill_ternarios imports openpyxl at module load
from types import SimpleNamespace

from fill_ternarios import kf_anchor, replicate_screen, results_rows, transfer_factors


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


def _v(s, a, b):
    return {"s": s, "a": a, "b": b}


def test_replicate_screen_names_the_failure():
    # F3 organic: vial 10 reads 1.64x vial 9 in every component -- a weighing slip
    assert replicate_screen([_v(0.30, 0.20, 0.20), _v(0.49, 0.33, 0.33)]) == "whole_vial"
    # B1 aqueous: droplets carry the terpenes 4x and leave 2PE at 1.16x
    assert replicate_screen([_v(0.02, 1e-4, 1e-4), _v(0.023, 4e-4, 4e-4)]) == (
        "component_spread"
    )
    # clean pairs agree to a few % per vial; one vial, or a binary's zero 2PE, is no pair
    assert replicate_screen([_v(0.02, 1e-4, 1e-4), _v(0.0205, 1.04e-4, 0.97e-4)]) == ""
    assert replicate_screen([_v(0.02, 1e-4, 1e-4)]) == ""
    assert replicate_screen([_v(0.0, 0.4, 0.4), _v(0.0, 0.41, 0.4)]) == ""


def test_aqueous_keep_drops_the_richer_vial_of_a_droplet_pair():
    from fill_ternarios import aqueous_keep

    lean = {"s": 0.020, "a": 1e-4, "b": 1e-4, "kf": [0.98]}
    rich = {"s": 0.023, "a": 4e-4, "b": 4e-4, "kf": [0.98]}  # B1: below the ceiling
    assert aqueous_keep([rich, lean], terp_max=0.003) == [lean]
    # a whole-vial slip is not droplets: which vial is right is not in the data
    slip = {"s": 0.030, "a": 1.5e-4, "b": 1.5e-4, "kf": [0.98]}
    assert aqueous_keep([lean, slip], terp_max=0.003) == [lean, slip]


class _Sheet(dict):
    title = "Bloque Z"

    def __getitem__(self, k):
        return SimpleNamespace(value=self.get(k))


def _rec(row, sysnum, ph, L, M, N, U=None):
    return {"row": row, "sysnum": sysnum, "ph": ph, "phase": ph, "L": L, "M": M, "N": N,
            "I": 0.2, "K": 1.0, "U": U, "V": None}  # fmt: skip


def test_transfer_factor_is_the_organic_closure_and_corrects_the_aqueous_only():
    ws = _Sheet(F2=100.0, G2=100.0, H2=100.0, F3=1.0, G3=1.0, H3=1.0, M3="Thymol", N3="Eugenol")
    # DF = K/I = 5 and C = A/100 %, so g = A x 5e-4. Organic: G = 0.5 + 0.2 + 0.155 =
    # 0.855 = 0.9 x (1 - 0.05 KF) -- a batch reading at phi = 0.9
    org = [_rec(r, 1, "T", 1000.0, 400.0, 310.0, U=5.0) for r in (5, 6)]
    aq = [_rec(r, 1, "B", 10.0, 0.5, 0.5) for r in (7, 8)]
    phi = transfer_factors([(ws, org + aq)])
    assert phi == {"Z1T1 AL Z5B2": pytest.approx(0.9)}
    rows = {r[2]: r for r in results_rows(ws, "Z", org + aq, phi=phi)}
    # aqueous 2PE: 10/100 % x DF 5 = 0.005, read 0.9 low -> 0.00556; water by difference
    assert float(rows["Inferior"][3]) == pytest.approx(0.005 / 0.9, abs=1e-5)
    assert float(rows["Inferior"][8]) == pytest.approx(1 - 0.0055 / 0.9, abs=1e-5)
    # organic: phi cancels in the KF anchor
    plain = {r[2]: r for r in results_rows(ws, "Z", org + aq)}
    assert rows["Superior"][3:9] == plain["Superior"][3:9]


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
