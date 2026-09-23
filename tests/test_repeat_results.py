from types import SimpleNamespace

import pytest

from fill_ternarios import results_rows
from repeat_results import (
    block_of,
    classify_repeat,
    compounds,
    parse_code,
    two_pe_endpoint,
)


def test_vial_codes_map_to_block_and_compounds():
    assert parse_code("C4_T1") == ("C4", "T", 1)
    assert parse_code("BIN3-B2") == ("BIN3", "B", 2)
    assert block_of("BIN1") == "A" and block_of("BIN3") == "C" and block_of("D2") == "D"
    assert compounds("D1") == ("2PE", "Thymol", "Eugenol")
    assert compounds("BIN1") == ("L-Carvone", "Thymol")  # the edge carries no 2PE
    assert compounds("2PE") == ("2PE",)


def _pk(t, area):
    return SimpleNamespace(time_min=t, area_mV_min=area)


def test_classify_keeps_largest_peak_per_window_and_ignores_other_blocks():
    peaks = [_pk(4.9, 9e4), _pk(10.7, 130.0), _pk(10.9, 2.0), _pk(13.39, 5.0),
             _pk(13.8, 7.0), _pk(14.03, 3.0), _pk(16.0, 1.0)]  # fmt: skip
    got = {(nm, p.time_min) for nm, p in classify_repeat(peaks, ("2PE", "Thymol", "Eugenol"))}
    assert ("2PE", 10.7) in got and ("other", 10.9) in got
    assert ("Thymol", 13.39) in got and ("Eugenol", 13.8) in got
    assert ("other", 14.03) in got  # carvacrol is not block D's: left unlabelled
    assert ("other", 16.0) in got


def test_two_pe_binary_endpoints():
    entry = {
        "2PE-T1": {"D": 2.0, "F": 2.3, "H": 3.0, "U": None, "V": None},
        "2PE-B1": {"D": None, "F": None, "H": None, "U": 9.0, "V": 9.2},
    }
    areas = {"2PE-T1": {"2pe": 100.0}}
    org, aq = two_pe_endpoint(areas, entry, f2=200.0)
    assert org["phase"] == "organic" and org["water"] == pytest.approx(0.091)
    assert org["w_2pe"] == pytest.approx(0.909)
    s = 100.0 / 200.0 * (1.0 / 0.3) / 100
    assert aq["w_2pe"] == pytest.approx(s) and aq["water"] == pytest.approx(1 - s)


class _Sheet(dict):
    def __getitem__(self, k):
        return SimpleNamespace(value=self.get(k))


def test_aqueous_phase_without_kf_takes_water_by_difference():
    ws = _Sheet(F2=200.0, G2=180.0, H2=170.0, M3="Thymol", N3="Eugenol")
    rec = {"row": 5, "sysnum": 1, "ph": "T", "phase": "Superior", "L": 100.0, "M": 0.1,
           "N": 0.1, "I": 0.3, "K": 1.0, "U": None, "V": None}  # fmt: skip
    (row,) = results_rows(ws, "Z", [rec])
    assert row[12] == "bydiff" and "incomplete" not in row[11]
    assert row[9] == ""  # closure is 1 by construction: not reported
    assert float(row[3]) + float(row[5]) + float(row[7]) + float(row[8]) == pytest.approx(
        1, abs=1e-4
    )
