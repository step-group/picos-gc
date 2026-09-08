import pytest

pytest.importorskip("openpyxl")
import openpyxl

from audit_repeats import _2pe_outliers, audit


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
    # Binary edge rows 25-28 (add_binary_tielines layout): T1/T2 organic with KF 3 %,
    # B1 clean aqueous (0.1 g/L per terpene, no KF), B2 with 10 g/L per terpene (droplets).
    ws["A25"] = "Bin"
    for r in (25, 26, 27, 28):
        ws[f"D{r}"], ws[f"F{r}"], ws[f"H{r}"] = 10.0, 11.0, 12.0
    for r in (25, 26):
        ws[f"M{r}"], ws[f"N{r}"], ws[f"U{r}"] = 2350.0, 2350.0, 3.0  # 47 % + 47 % + 3 %
    ws["M27"], ws["N27"] = 0.5, 0.5
    ws["M28"], ws["N28"] = 50.0, 50.0
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
    # binary edge: organic endpoint closes (0.47+0.47+0.03), aqueous cherry-picks B1
    org, aq = by[("Z-bin", "organic")], by[("Z-bin", "aqueous")]
    assert org["repeat"] == "no" and org["closure"] == "0.97000" and org["water_src"] == "kf"
    assert org["codes"] == "Z-bin-T1, Z-bin-T2"  # block not in BIN_TO_BLOCK: no BIN number
    assert aq["repeat"] == "no" and aq["dropped"] == "Z-bin-B2" and aq["n_vials_used"] == 1
    assert aq["water_src"] == "bydiff" and aq["aq_terpene_max"] == "0.00020"
    assert len(by) == 10
    # 0.20 raw, renormalized by the KF anchor (0.20 water / 0.60 organics): 0.20/0.75
    assert by[("Z1", "Superior")]["w_2pe"] == "0.26667"


def _tieline(system: str, aq: str, org: str) -> list[dict]:
    base = {"system": system, "reason": "", "repeat": "no"}
    return [
        {**base, "water_src": "kf", "w_2pe": org},
        {**base, "water_src": "bydiff", "w_2pe": aq},
    ]


def test_the_odd_2pe_rich_endpoint_is_flagged():
    """Same organic phase (~0.88 2PE) in every block, so one aqueous number measured
    once per block; F2-like rows miss the median by more than 1.5x."""
    rows = []
    for system, aq in (
        ("A2", "0.01800"),
        ("B1", "0.01990"),
        ("C1", "0.01710"),
        ("E2", "0.01610"),
        ("F2", "0.02770"),
    ):
        rows += _tieline(system, aq, "0.88000")
    rows += _tieline("A1", "0.00090", "0.03600")  # solvent-rich end: not in the family
    _2pe_outliers(rows)
    by = {(r["system"], r["water_src"]): r for r in rows}
    assert by[("F2", "bydiff")]["reason"] == "aqueous_2pe_outlier"
    assert by[("F2", "bydiff")]["repeat"] == "yes"
    assert all(by[(s, "bydiff")]["repeat"] == "no" for s in ("A2", "B1", "C1", "E2", "A1"))
    assert by[("F2", "kf")]["repeat"] == "no"  # the organic phase is not the suspect
