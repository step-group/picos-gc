# /// script
# requires-python = ">=3.11"
# dependencies = ["openpyxl"]
# ///
"""Export the per-vial LLE inputs the uncertainty analysis needs.

fill_ternarios.py collapses each phase to a 2-vial average and emits only the
final composition. An uncertainty budget needs the level below that: the two
vials separately, their own weighings, and both Karl-Fischer readings. This
writes that level out verbatim -- no averaging, no unit changes, blanks stay
blank -- for data/raw/lle_ternary/ in the thesis repo.

Also exports the GC calibration standards from CC_MF.xlsx, with the certificate
purity fill_ternarios corrects them by. fill_ternarios reduces them to a power-law
response (alpha, beta); the uncertainty analysis needs the points themselves,
because the inverse-prediction interval cannot be recovered from two parameters.

Run: uv run export_vial_data.py       (campaign 1 only, from the filled workbook)
repeat_results.py writes the same table with every repeat round patched in,
out/vial_measurements_repeat.csv -- the one that matches the tie-lines TESIS uses.
"""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl

from fill_ternarios import PURITY, canon

_ROOT = Path(__file__).resolve().parent
WB = _ROOT / "Sistemas ternarios_MF_filled.xlsx"
CC = _ROOT / "CC_MF.xlsx"
MERGED = _ROOT / "out"
OUT_VIALS = _ROOT / "out" / "vial_measurements.csv"
OUT_CAL = _ROOT / "out" / "gc_calibration.csv"

# Rows 5-24 are the five ternary systems (4 rows each: 2 Superior, 2 Inferior);
# rows 25-28 are the water-solvent binary endpoint in the same pattern.
TERNARY_ROWS = range(5, 25)
BINARY_ROWS = range(25, 29)


def _num(ws, coord):
    v = ws[coord].value
    return v if isinstance(v, (int, float)) else None


def injections_per_vial() -> dict[str, int]:
    """{block: n_rep} -- injections per vial, from label_terpenos' merged CSVs.

    2 for blocks B, C, E, F and 1 for A, D, H, I. A single injection gets no
    1/M benefit in the inverse-prediction interval, so this difference between
    blocks is real and must not be averaged away downstream.
    """
    out = {}
    for path in sorted(MERGED.glob("*/merged_samples.csv")):
        block = path.parent.name[0]  # "A1T1 AL A5B2" -> "A"
        with path.open() as fh:
            reps = {int(r["n_rep"]) for r in csv.DictReader(fh) if r.get("n_rep")}
        if reps:
            out[block] = max(reps)
    return out


def _rows(ws, block, rows, kind, n_rep, single=frozenset()):
    out = []
    for r in rows:
        phase_position = ws[f"B{r}"].value
        if phase_position is None:
            continue
        # The system is the row's position, as fill_ternarios._vial_rows reads it; the
        # workbook's own column-A code is a template copy in F/H/I ("E1" on Bloque F).
        system = f"{block}{(r - 5) // 4 + 1}" if kind == "ternary" else f"{block}-bin"
        out.append(
            {
                "block": block,
                "kind": kind,
                "system": system,
                "system_label_raw": ws[f"A{r}"].value,
                "phase_position": str(phase_position).strip(),
                "vial": _num(ws, f"C{r}"),
                "n_injections": 1 if (ws.title, r) in single else n_rep,
                "m_vial_g": _num(ws, f"D{r}"),
                "m_vial_sample_g": _num(ws, f"F{r}"),
                "m_vial_sample_diluent_g": _num(ws, f"H{r}"),
                "area_2pe": _num(ws, f"L{r}"),
                "area_hba": _num(ws, f"M{r}"),
                "area_hbd": _num(ws, f"N{r}"),
                "kf_water_pct_1": _num(ws, f"U{r}"),
                "kf_water_pct_2": _num(ws, f"V{r}"),
                "hba": ws["M3"].value,
                "hbd": ws["N3"].value,
            }
        )
    return out


def vial_rows(wb=None, single=frozenset()):
    """Every vial row of *wb* (default: the filled workbook). *single* holds the
    (sheet, row) cells a repeat round overwrote: those were injected once."""
    wb = wb or openpyxl.load_workbook(WB, data_only=True)
    n_rep = injections_per_vial()
    rows = []
    for sheet in wb.sheetnames:
        block = sheet.replace("Bloque", "").strip()
        ws = wb[sheet]
        m = n_rep.get(block, 1)
        rows += _rows(ws, block, TERNARY_ROWS, "ternary", m, single)
        rows += _rows(ws, block, BINARY_ROWS, "binary", m, single)
    return rows


def calibration_rows():
    """Every (%m/m, area) standard in cols N/O -- the SM stock point plus E1..E5.

    Same selection fill_ternarios.cc_mf_models fits its power law over, so a fit of
    ln(area) on ln(w_pct_m_m * purity) from this CSV reproduces the workbook's alpha, beta.
    """
    wb = openpyxl.load_workbook(CC, data_only=True)
    rows = []
    for ws in wb.worksheets:
        n = 0
        for r in ws.iter_rows(values_only=True):
            if len(r) > 14 and isinstance(r[13], (int, float)) and isinstance(r[14], (int, float)):
                rows.append(
                    {"compound": ws.title, "standard": n, "w_pct_m_m": r[13], "area": r[14],
                     "purity": PURITY[canon(ws.title)]}  # fmt: skip
                )
                n += 1
    return rows


if __name__ == "__main__":
    OUT_VIALS.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((OUT_VIALS, vial_rows()), (OUT_CAL, calibration_rows())):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path} ({len(rows)} rows)")
