import pytest

pytest.importorskip("openpyxl")
import openpyxl

from audit_repeats import audit


def _sheet():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Bloque Z"
    ws["F2"], ws["G2"], ws["H2"] = 100.0, 100.0, 100.0  # area per %m/m
    ws["M3"], ws["N3"] = "Thymol", "Carvone"
    # Z1: rows 5-6 Superior (both vials, closes), rows 7-8 Inferior (vial 8 never injected)
    # Z2: rows 9-12, no areas at all.
    # Z3: rows 13-16 aqueous (KF 98 %): Superior clean (0.05 % terpenes both vials),
    #     Inferior vial 15 clean, vial 16 carries 1 % terpenes (organic droplets).
    ws["A5"], ws["A9"], ws["A13"] = "Z1", "Z2", "Z3"
    phases = ["Superior", "Superior", "Inferior", "Inferior"] * 3
    for r, phase in zip(range(5, 17), phases, strict=True):
        ws[f"B{r}"] = phase
        ws[f"D{r}"], ws[f"F{r}"], ws[f"H{r}"] = 10.0, 11.0, 12.0  # dilution x2
        ws[f"U{r}"] = 98.0 if r >= 13 else 20.0  # KF % water
    for r in (5, 6, 7):  # 2PE 20 %, HBA 20 %, HBD 20 % (diluted 10 % each) + 20 % water = 0.8
        ws[f"L{r}"], ws[f"M{r}"], ws[f"N{r}"] = 1000.0, 1000.0, 1000.0
    for r in (13, 14, 15):  # 2PE 1 %, terpenes 0.025 % each
        ws[f"L{r}"], ws[f"M{r}"], ws[f"N{r}"] = 50.0, 1.25, 1.25
    ws["L16"], ws["M16"], ws["N16"] = 50.0, 25.0, 25.0  # 2PE 1 %, terpenes 0.5 % each
    return wb


def test_verdicts():
    by = {(r["system"], r["phase"]): r for r in audit(_sheet())}
    assert by[("Z1", "Superior")]["repeat"] == "no"
    assert by[("Z1", "Superior")]["closure"] == "0.80000"
    assert by[("Z1", "Superior")]["aq_terpene_max"] == ""  # organic phase: rule not applied
    assert by[("Z1", "Superior")]["codes"] == "Z1-T1, Z1-T2"  # complete point: both vials
    assert by[("Z1", "Inferior")]["reason"] == "single_vial"
    assert by[("Z1", "Inferior")]["codes"] == "Z1-B2"  # only the missing vial
    assert by[("Z2", "Superior")]["reason"] == "missing_vials"
    assert by[("Z2", "Superior")]["codes"] == "Z2-T1, Z2-T2"
    assert by[("Z2", "Superior")]["hbd"] == "Carvone"
    assert by[("Z2", "Inferior")]["n_kf"] == 2
    assert by[("Z3", "Superior")]["repeat"] == "no"
    assert by[("Z3", "Superior")]["aq_terpene_max"] == "0.00050"
    assert by[("Z3", "Inferior")]["reason"] == "aqueous_organics_suspect"
    assert by[("Z3", "Inferior")]["aq_terpene_max"] == "0.01000"
    assert len(by) == 6  # no binary block on a synthetic sheet
