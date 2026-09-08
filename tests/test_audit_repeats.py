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
    # Z4: rows 17-20 aqueous: Superior both vials 1 % terpenes; Inferior clean terpenes
    #     but 2PE 1 % vs 0.2 % (2PE-only disagreement).
    ws["A5"], ws["A9"], ws["A13"], ws["A17"] = "Z1", "Z2", "Z3", "Z4"
    phases = ["Superior", "Superior", "Inferior", "Inferior"] * 4
    for r, phase in zip(range(5, 21), phases, strict=True):
        ws[f"B{r}"] = phase
        ws[f"D{r}"], ws[f"F{r}"], ws[f"H{r}"] = 10.0, 11.0, 12.0  # dilution x2
        ws[f"U{r}"] = 98.0 if r >= 13 else 20.0  # KF % water
    for r in (5, 6, 7):  # 2PE 20 %, HBA 20 %, HBD 20 % (diluted 10 % each) + 20 % water = 0.8
        ws[f"L{r}"], ws[f"M{r}"], ws[f"N{r}"] = 1000.0, 1000.0, 1000.0
    for r in (13, 14, 15, 19):  # 2PE 1 %, terpenes 0.025 % each
        ws[f"L{r}"], ws[f"M{r}"], ws[f"N{r}"] = 50.0, 1.25, 1.25
    for r in (16, 17, 18):  # 2PE 1 %, terpenes 0.5 % each -> droplets
        ws[f"L{r}"], ws[f"M{r}"], ws[f"N{r}"] = 50.0, 25.0, 25.0
    ws["L20"], ws["M20"], ws["N20"] = 10.0, 1.25, 1.25  # 2PE 0.2 %, clean terpenes
    return wb


def test_verdicts():
    by = {(r["system"], r["phase"]): r for r in audit(_sheet())}
    assert by[("Z1", "Superior")]["repeat"] == "no"
    assert by[("Z1", "Superior")]["closure"] == "0.80000"
    assert by[("Z1", "Superior")]["aq_terpene_max"] == ""  # organic phase: rule not applied
    assert by[("Z1", "Superior")]["codes"] == "Z1-T1, Z1-T2"  # complete point: both vials
    # one clean vial is accepted: informational, not a repeat
    assert by[("Z1", "Inferior")]["repeat"] == "no"
    assert by[("Z1", "Inferior")]["flags"] == "single_vial"
    assert by[("Z1", "Inferior")]["codes"] == "Z1-B2"  # the missing vial
    assert by[("Z2", "Superior")]["reason"] == "missing_vials"
    assert by[("Z2", "Superior")]["codes"] == "Z2-T1, Z2-T2"
    assert by[("Z2", "Superior")]["hbd"] == "Carvone"
    assert by[("Z2", "Inferior")]["n_kf"] == 2
    assert by[("Z3", "Superior")]["repeat"] == "no"
    assert by[("Z3", "Superior")]["aq_terpene_max"] == "0.00050"
    # contaminated vial with a clean pair: cherry-picked by the pipeline, passes
    z3i = by[("Z3", "Inferior")]
    assert z3i["repeat"] == "no" and z3i["dropped"] == "Z3-B2" and z3i["n_vials_used"] == 1
    assert "dropped_replicate" in z3i["flags"]
    assert z3i["aq_terpene_max"] == "0.00050"  # only the kept vial is judged
    assert z3i["aq_ceiling"] == "0.00326"  # (thymol 1.11 + carvone 1.61 g/L at 30 °C) x 1.2
    # both vials contaminated: nothing to pick
    assert by[("Z4", "Superior")]["reason"] == "aqueous_organics_suspect"
    assert by[("Z4", "Superior")]["dropped"] == ""
    # 2PE-only disagreement (5x), no droplet signature
    assert by[("Z4", "Inferior")]["reason"] == "aqueous_replicate_mismatch"
    assert by[("Z4", "Inferior")]["aq_organics_ratio"] == "4.20"
    assert len(by) == 8  # no binary block on a synthetic sheet
