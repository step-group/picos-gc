# /// script
# requires-python = ">=3.11"
# dependencies = ["openpyxl"]
# ///
"""Fill `Sistemas ternarios_MF.xlsx` with picos-gc peak areas + emit a results CSV.

Each sheet `Bloque A..I` is a pseudoternary LLE workbook: Water + 2-phenylethanol
(Solute) + a two-terpene solvent (HBA + HBD). We write the GC areas into columns
L/M/N per vial; the sheet's own formulas turn area -> diluted %m/m -> real %m/m
(x dilution factor) -> 2-replicate average -> KF water -> normalized ternary point.

The GC response is a power law through the origin, A = α·C^β, fitted in log-log over
CC_MF.xlsx's SM stock point + E1..E5 with purity-corrected standards (see cc_mf_models),
their areas integrated by label_terpenos like every sample (see calibration_points).
α and β land in F2:H2 and F3:H3 of every block, and the sheet formulas invert it.

Prereq: `uv run python label_terpenos.py` has produced out/<batch>/merged_samples.csv.
Run:    uv run fill_ternarios.py   (PEP 723 header pulls openpyxl; paths are
        script-relative, so it works from any directory)
        (also shells out to pcsaft-quaternary/plot_experimental_tielines.py at the
        end, so the ternary diagrams under out/tielines/ are regenerated in one go)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import statistics
import subprocess
import tempfile
from glob import glob
from pathlib import Path
from typing import NamedTuple

import openpyxl

_ROOT = Path(__file__).resolve().parent  # script-relative so CWD doesn't matter
WB_IN = _ROOT / "Sistemas ternarios_MF.xlsx"
WB_OUT = _ROOT / "Sistemas ternarios_MF_filled.xlsx"
CC_MF = _ROOT / "CC_MF.xlsx"
CAL_AREAS = _ROOT / "out" / "AaCALIBRACION_TERPENOS" / "areas.csv"  # label_terpenos
DATA = _ROOT / "FLECK_TERPENOS2026"
OUT_CSV = _ROOT / "out" / "ternarios_resultados.csv"
BIN_OUT_CSV = _ROOT / "out" / "binarios_tielines.csv"  # computed Water-solvent binary endpoints
BINARIOS_CSV = _ROOT / "out" / "BINARIOS_TERPENOS" / "samples.csv"
# BIN system -> ternary block, SEQUENTIAL (BIN n = the nth block in order A,B,C,D,E,F,H,I):
# 1=ThyCarvone(A) 2=ThyGer(B) 3=ThyCarvacrol(C) 4=ThyEugenol(D) 5=ThyCamphor(E)
# 6=CamphorCarvone(F) 7=CamphorCarvacrol(H) 8=CamphorEugenol(I). Carvone reads as
# "Geraniol" in samples.csv (the binary-method RT is miscalibrated), so blocks A & F route
# their Carvone area through fill_binary's single-remaining-terpene fallback (right value,
# wrong label). All 8 blocks map; a block only resolves an endpoint once its masses + KF
# are hand-entered on rows 25-28 (all 8 blocks have masses + organic KF today).
BIN_TO_BLOCK = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "H", 8: "I"}

TERPS = ("camph", "carvone", "carvacrol", "geraniol", "thymol", "eugenol")
DISPLAY = {  # canon -> workbook-style name written into blank M3/N3 cells
    "camphor": "Camphor",
    "carvone": "Carvone",
    "carvacrol": "Carvacrol",
    "geraniol": "Geraniol",
    "thymol": "Thymol",
    "eugenol": "Eugenol",
    "2pe": "2PhEt",
}
# sample codes vary by batch: A1T1, B1-T1, E1_T1, even BM1-T1 (block F). Match the
# trailing (system number, phase, rep); prefix letters/separators are ignored.
CODE_RE = re.compile(r"(\d+)[-_ ]?([TB])([12])\s*$", re.I)

# Composition QC. KF water is an independent absolute assay, so anchor it and split
# the GC remainder by ratio rather than normalising all four by their (often <1) sum
# (that rescales the good water — the D1/I1 spurious-curve bug). Σ outside the band ⇒
# mass didn't close; two vials that disagree past the replicate screen ⇒ replicate_mismatch.
CLOSURE_BAND = (0.5, 1.5)
# Replicate screen, on each vial's dilution-corrected fractions. Clean pairs agree to
# 2.7 % (2PE) and ~4.5 % (terpenes) per vial (2026-09-28, 65 pairs). A whole-vial slip --
# a weighing or transcription error -- moves every component together (F3: 1.64x);
# organic droplets in an aqueous vial move the terpenes and leave 2PE (B1: 4x vs 1.16x).
# The raw-area gate this replaces (3x on the dominant peak) passed all 13 such pairs.
SCREEN_WHOLE_VIAL = 1.15  # geometric-mean vial ratio over the components, ~5 sigma
SCREEN_SPREAD = 1.3  # max/min of the per-component vial ratios
ORGANIC_WATER_MAX = 0.5  # KF anchor only the water-poor (organic) phase; see results_rows
# Pure-water solubility at 30 °C, mass fraction (g/L / 1000), from the thesis
# data/raw/solubility/water_terpene/measurements.csv: thymol, carvacrol, eugenol, geraniol
# = Martins et al. 2017 (303.15 K shake-flask); carvone = Smyrl & LeMaguer 1980 (303.15 K);
# camphor = Yalkowsky compilation, 1.0-2.1 g/L at 293-298 K and 2.5 at 310 K, so ~2.0.
# An aqueous vial whose HBA+HBD exceeds the PAIR's summed solubility took organic-phase
# droplets — even both terpenes at saturation cannot reach it (clean vials here sit at
# 0.1-0.6 g/L). ponytail: 30 °C only; geraniol climbs steeply above (5.5 g/L at 40 °C),
# so add a temperature argument if a block was equilibrated warmer.
AQ_SOLUBILITY_30C = {
    "thymol": 0.00111,
    "carvacrol": 0.00129,
    "eugenol": 0.00210,
    "geraniol": 0.00119,
    "carvone": 0.00161,
    "camphor": 0.00200,
}
AQ_SOLUBILITY_TOL = 1.2  # +20 %: shake-flask literature scatters by that much between labs
# Certificate purity, mass fraction: Sigma-Aldrich GC assay of the lot used for CC_MF, from
# aromas_equilibrios_vfinal.xlsx -> Componentes I/J (">99.9 %" taken as 0.999). A standard
# weighed as w %m/m of reagent holds w·P of the compound, and the inverse reading scales
# with P exactly, so camphor (96.9 %) was read 3 % high before this correction.
PURITY = {
    "2pe": 0.999,  # SDBB2371
    "thymol": 0.999,  # SHBQ5769
    "carvone": 0.992,  # SHBQ8402
    "camphor": 0.969,  # SDBB0718
    "geraniol": 0.989,  # SHBP7700
    "carvacrol": 0.997,  # SHBQ9085
    "eugenol": 0.992,  # SHBM4053
}


def aq_terpene_max(hba, hbd) -> float:
    """Droplet ceiling for a pair: sum of the two pure-water solubilities (mass fraction)
    × AQ_SOLUBILITY_TOL. An unknown name counts 0.0025, i.e. an unlabelled pair falls back
    to the old 0.5 % (× tolerance)."""
    return AQ_SOLUBILITY_TOL * sum(
        AQ_SOLUBILITY_30C.get(canon(n or ""), 0.0025) for n in (hba, hbd)
    )


def parse_key(code: str) -> tuple[int, str, int] | None:
    m = CODE_RE.search(str(code).strip())
    return (int(m.group(1)), m.group(2).upper(), int(m.group(3))) if m else None


def canon(s) -> str:
    """Canonical compound key shared across sheet names, CSV headers, CC_MF sheets."""
    k = re.sub(r"[^a-z0-9]", "", str(s).lower())
    for key in TERPS:
        if key in k:
            return "camphor" if key == "camph" else key
    if "phenyl" in k or k in ("2pe", "2phet", "pe"):
        return "2pe"
    return k


def hba_priority(wb) -> dict[str, float]:
    """Per-terpene tendency to be the HBA, learned from already-labelled sheets.

    HBA appearance scores 0, HBD scores 1, averaged; lower = more HBA-like. Used to
    order a block's two terpenes when its M3/N3 names are blank (e.g. Bloque F).
    """
    seen: dict[str, list[int]] = {}
    for ws in wb.worksheets:
        hba, hbd = ws["M3"].value, ws["N3"].value
        if hba and hbd:
            seen.setdefault(canon(hba), []).append(0)
            seen.setdefault(canon(hbd), []).append(1)
    return {k: sum(v) / len(v) for k, v in seen.items()}


class Response(NamedTuple):
    """GC response A = alpha·C^beta, area vs %m/m in the injected solution."""

    alpha: float
    beta: float

    def conc(self, area: float) -> float:
        """%m/m in the injected solution that gives *area*."""
        return (area / self.alpha) ** (1 / self.beta)


def calibration_points(xlsx: Path) -> list[tuple[str, int, float, float]]:
    """(CC_MF sheet, standard, %m/m, area) for every standard: the stock (0) plus E1..E5.

    %m/m is CC_MF col N, gravimetric. The area is NOT col O: those were integrated in
    LabSolutions and miss the chromatograms by up to 7 % (camphor E5, 2PE E1, thymol E1),
    while the samples are integrated by label_terpenos. CAL_AREAS is label_terpenos'
    integration of the standards themselves, so both sides of the response share one
    integrator.
    """
    with CAL_AREAS.open() as fh:
        areas = {
            (r["compound"], int(r["standard"])): float(r["area_mV_min"]) for r in csv.DictReader(fh)
        }
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    out = []
    for ws in wb.worksheets:
        ws_w = [
            float(r[13])
            for r in ws.iter_rows(values_only=True)
            if len(r) > 14 and isinstance(r[13], int | float) and isinstance(r[14], int | float)
        ]
        out += [(ws.title, n, w, areas[(ws.title, n)]) for n, w in enumerate(ws_w)]
    return out


def cc_mf_models(xlsx: Path) -> dict[str, Response]:
    """{canon: power-law Response} from CC_MF.xlsx standards.

    Fit over every calibration_points standard as ln A = ln α + β ln(w·P), ordinary least
    squares in log-log (constant relative error). Through-origin linear read every lowest
    standard 9-23 % low, a monotone residual in all seven compounds (adsorptive loss at
    low load); β = 1.03-1.07 takes it out, and a blank still reads zero.
    """
    by: dict[str, list[tuple[float, float]]] = {}
    for sheet, _, w, a in calibration_points(xlsx):
        by.setdefault(canon(sheet), []).append((w, a))
    out: dict[str, Response] = {}
    for key, pts in by.items():
        if len(pts) >= 2:
            x = [math.log(w * PURITY.get(key, 1.0)) for w, _ in pts]
            y = [math.log(a) for _, a in pts]
            xm, ym = sum(x) / len(x), sum(y) / len(y)
            beta = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y)) / sum(
                (xi - xm) ** 2 for xi in x
            )
            out[key] = Response(math.exp(ym - beta * xm), beta)
    return out


def response(ws, col: str) -> Response | None:
    """The block's Response for one compound column (F=2PE, G=HBA, H=HBD): α in row 2,
    β in row 3, as fill_block writes them. None when either is missing."""
    a, b = _num(ws, f"{col}2"), _num(ws, f"{col}3")
    return Response(a, b) if a and b else None


def load_vial_areas(merged_csv: Path) -> dict[tuple[int, str, int], dict[str, float]]:
    """{(sysnum, phase, rep): {canon_compound: area_mean}} from a merged CSV."""
    rows = list(csv.reader(merged_csv.open()))
    hdr = rows[0]
    cols = {i: canon(h[:-10]) for i, h in enumerate(hdr) if h.endswith("_area_mean")}
    out: dict[tuple[int, str, int], dict[str, float]] = {}
    for r in rows[1:]:
        if r and r[0] and (key := parse_key(r[0])) is not None:
            out[key] = {c: float(r[i] or 0) for i, c in cols.items()}
    return out


def load_binary_areas(
    csv_path: Path, warn: list[str]
) -> dict[tuple[int, str, int], dict[str, float]]:
    """{(binnum, phase, rep): {canon_compound: area}} from BINARIOS samples.csv.

    Same shape as load_vial_areas, but the columns end in `_area` (per-injection, no
    `_area_mean` averaging). Missing file -> {} + a warning (binary areas just stay blank).
    """
    if not csv_path.exists():
        warn.append(f"binary: {csv_path} missing - run label_terpenos.py (skipping binary fill)")
        return {}
    rows = list(csv.reader(csv_path.open()))
    hdr = rows[0]
    cols = {i: canon(h[:-5]) for i, h in enumerate(hdr) if h.endswith("_area")}
    out: dict[tuple[int, str, int], dict[str, float]] = {}
    for r in rows[1:]:
        if r and r[0] and (key := parse_key(r[0])) is not None:
            out[key] = {c: float(r[i] or 0) for i, c in cols.items()}
    return out


# BIN phase/rep -> tie-line row: Superior (organic) 25/26, Inferior (aqueous) 27/28.
_BIN_ROWS = {("T", 1): 25, ("T", 2): 26, ("B", 1): 27, ("B", 2): 28}


def fill_binary(ws, block: str, bin_areas: dict, warn: list[str]) -> None:
    """Write the Water-solvent tie-line HBA/HBD areas into M/N of rows 25-28.

    Areas come from BINARIOS samples.csv via BIN_TO_BLOCK; only M/N are written (masses
    D/F/H and KF U/V stay hand-entered). Every block maps to a BIN, so the guard below is
    just a safety net. Must run after fill_block so M3/N3 (HBA/HBD names) are set.
    """
    binnum = next((b for b, blk in BIN_TO_BLOCK.items() if blk == block), None)
    if binnum is None:
        return  # unmapped block (shouldn't happen — all 8 map)
    if ws["A25"].value != "Bin":
        warn.append(f"{block}: no binary tie-line block (run add_binary_tielines.py first)")
        return
    hba_key, hbd_key = canon(ws["M3"].value or ""), canon(ws["N3"].value or "")
    for (ph, rep), row in _BIN_ROWS.items():
        a = bin_areas.get((binnum, ph, rep))
        if a is None:
            warn.append(f"{block}: binary BIN{binnum} {ph}{rep} absent from samples.csv")
            continue
        terp = {c: v for c, v in a.items() if c != "ipa" and v}  # nonzero terpene areas
        if hba_key not in terp:
            warn.append(f"{block}: binary HBA {hba_key!r} not found in BIN{binnum} {ph}{rep}")
            continue
        ws[f"M{row}"] = terp[hba_key]
        if hbd_key in terp:
            ws[f"N{row}"] = terp[hbd_key]
        else:  # HBD label differs (classifier) -> use the single remaining terpene
            other = [v for c, v in terp.items() if c != hba_key]
            if len(other) == 1:
                ws[f"N{row}"] = other[0]
                warn.append(
                    f"{block}: binary HBD {hbd_key!r} not labelled in BIN{binnum} {ph}{rep}; "
                    "used the other terpene"
                )
            else:
                warn.append(
                    f"{block}: binary HBD ambiguous in BIN{binnum} {ph}{rep} ({list(terp)})"
                )


BIN_PAIRS = ((25, (25, 26)), (27, (27, 28)))  # binary endpoint anchor row -> its two vials


def binary_vials(ws, reps: tuple[int, int], g2, h2) -> list[dict]:
    """Per-vial fractions for the binary rows, same shape as vial_fractions (2PE = 0).
    Read from the RAW cells (masses D/F/H, areas M/N, KF U/V), never from the sheet's
    cached binary formulas: those go stale between recalcs (Bloque H's cached AA25 read
    0.35 while the raw cells close at 0.94). A vial with no KF is taken as aqueous with
    water by difference when its terpenes are < 0.5; an organic vial needs its KF."""
    out = []
    for r in reps:
        d, f, h = _num(ws, f"D{r}"), _num(ws, f"F{r}"), _num(ws, f"H{r}")
        m, n = _num(ws, f"M{r}"), _num(ws, f"N{r}")
        if None in (d, f, h, m, n) or f == d or not (g2 and h2):
            continue
        rec = {"row": r, "L": None, "M": m, "N": n, "I": f - d, "K": h - d,
               "U": _num(ws, f"U{r}"), "V": _num(ws, f"V{r}")}  # fmt: skip
        v = vial_fractions(rec, None, g2, h2)
        if v is None:
            continue
        v["s"] = 0.0
        if not v["kf"] and _terp(v) < ORGANIC_WATER_MAX:
            v["kf"] = [1.0 - _terp(v)]  # aqueous, water by difference
        if v["kf"]:
            out.append(v)
    return out


def binary_endpoint(
    vials: list[dict], terp_max: float, phi: float = 1.0
) -> tuple[list[float] | None, float | None, str, list[str], list[dict]]:
    """One binary endpoint from its vials: ([HBA, HBD, water], closure, water_src, flags,
    kept). Same chain as the ternaries: cherry-pick via aqueous_keep, organic (water <
    ORGANIC_WATER_MAX) KF-anchored, aqueous divided by its batch's *phi* (see
    transfer_factors) with water by difference. Closure only means something with a real
    KF, so it is None for by-difference aqueous endpoints."""
    kept = aqueous_keep(vials, terp_max)
    if not kept:
        return None, None, "", [], kept
    a = sum(_terp_a(v) for v in kept) / len(kept)
    b = sum(v["b"] or 0.0 for v in kept) / len(kept)
    kf = [c for v in kept for c in v["kf"]]
    w = sum(kf) / len(kf)
    flags = ["dropped_replicate"] if len(kept) < len(vials) else []
    if w < ORGANIC_WATER_MAX:
        (_, x, y, z), closure = kf_anchor(0.0, a, b, w)
        src = "kf"
        lo, hi = CLOSURE_BAND
        if not (lo <= closure <= hi):
            flags.append("low_closure")
    else:
        x, y = a / phi, b / phi
        z, closure, src = max(0.0, 1.0 - x - y), None, "bydiff"
    if mismatch_flag(replicate_screen(kept), organic=w < ORGANIC_WATER_MAX):
        flags.append("replicate_mismatch")
    return [x, y, z], closure, src, flags, kept


def _terp_a(v: dict) -> float:
    return v["a"] or 0.0


def binary_tieline_rows(wb_d, phi: dict[str, float] | None = None) -> list[list]:
    """Water-solvent binary endpoints, rows 25/26 and 27/28 of each sheet, computed from
    the raw cells via binary_endpoint (HBA, HBD, water mass fractions; 2PE = 0), the
    aqueous corrected by *phi* (transfer_factors) of its batch. A pair with no usable
    vial (no masses, no areas, or organic without KF) contributes nothing."""
    out = []
    for sheet in wb_d.sheetnames:
        ws = wb_d[sheet]
        if ws["A25"].value != "Bin":
            continue
        block = sheet.split()[-1]
        hba, hbd = ws["M3"].value, ws["N3"].value
        g2, h2 = response(ws, "G"), response(ws, "H")
        terp_max = aq_terpene_max(hba, hbd)
        for _row, reps in BIN_PAIRS:
            point, _closure, src, _flags, _kept = binary_endpoint(
                binary_vials(ws, reps, g2, h2),
                terp_max,
                (phi or {}).get(batch_of(ws, reps[0]), 1.0),
            )
            if point is None:
                continue
            phase = "aqueous" if point[2] > ORGANIC_WATER_MAX else "organic"
            out.append(
                [block, phase, hba, _fmt(point[0]), hbd, _fmt(point[1]), _fmt(point[2]), src]
            )
    return out


def _num(ws, coord: str) -> float | None:
    v = ws[coord].value
    return float(v) if isinstance(v, int | float) else None


def _vial_rows(ws):
    """Yield (row, sysnum, ph, rep, phase) for the 20 data rows (5..24).

    System number is the digit in `Cód. Sist` (its letter is unreliable: blocks
    F/H/I carry a stale `E1..E5`). Block identity comes from the sheet name.
    """
    sysnum, counts = None, {}
    for row in range(5, 25):
        a = ws[f"A{row}"].value
        if a:
            m = re.search(r"\d+", str(a))
            sysnum, counts = (int(m.group()) if m else sysnum), {}
        phase = str(ws[f"B{row}"].value or "").strip()
        ph = "T" if phase.lower().startswith("sup") else "B"
        counts[ph] = counts.get(ph, 0) + 1
        yield row, sysnum, ph, counts[ph], phase


def fill_block(
    ws, ws_d, block: str, areas: dict, cc: dict, priority: dict, warn: list[str]
) -> list[dict]:
    """Write missing names/slopes + L/M/N areas; return per-vial records for the CSV.

    `ws` is the formula-preserving sheet we write into; `ws_d` is the data_only
    twin, used to read the cached numeric masses/KF (those columns are formulas).
    """
    # Auto-label blank HBA/HBD names from this block's terpenes, ordered by the
    # convention learned from labelled sheets (Bloque F ships unlabelled).
    terps = sorted(
        {k for a in areas.values() for k in a if k != "2pe"},
        key=lambda c: priority.get(c, 1.0),
    )
    auto = []
    if not ws["M3"].value and terps:
        ws["M3"] = DISPLAY.get(terps[0], terps[0])
        auto.append(f"HBA={ws['M3'].value}")
    if not ws["N3"].value:
        hbd = next((t for t in terps if t != canon(ws["M3"].value or "")), None)
        if hbd:
            ws["N3"] = DISPLAY.get(hbd, hbd)
            auto.append(f"HBD={ws['N3'].value}")
    if auto:
        warn.append(
            f"{block}: auto-labelled {', '.join(auto)} from batch terpenes (confirm HBA/HBD roles)"
        )

    hba_key, hbd_key = canon(ws["M3"].value or ""), canon(ws["N3"].value or "")
    if not ws["M3"].value or not ws["N3"].value:
        warn.append(f"{block}: HBA/HBD names still blank; terpene areas not placed")

    # Every response comes from CC_MF (power law, SM-inclusive) — the single source of
    # truth: α over β, one column per compound. Row 2 held the old through-origin slopes.
    ws["E2"], ws["E3"] = "α  (A = α·C^β)", "β"
    for col, comp in (("F", "2pe"), ("G", hba_key), ("H", hbd_key)):
        if comp in cc:
            ws[f"{col}2"] = round(cc[comp].alpha, 6)
            ws[f"{col}3"] = round(cc[comp].beta, 8)
        elif comp:
            warn.append(f"{block}: no CC_MF response for {comp!r} ({col}2:{col}3)")

    recs = []
    for row, sysnum, ph, rep, phase in _vial_rows(ws):
        a = areas.get((sysnum, ph, rep))
        if a is None:
            warn.append(
                f"{block}: vial row {row} ({block}{sysnum}{ph}{rep}) absent from merged_samples"
            )
            continue
        if "2pe" in a:
            ws[f"L{row}"] = a["2pe"]
        if hba_key in a:
            ws[f"M{row}"] = a[hba_key]
        if hbd_key in a:
            ws[f"N{row}"] = a[hbd_key]
        # Dilution from the raw weighings — I ("Muestra, g") = F-D, K ("Mue+Met") = H-D —
        # rather than the sheet's cached I/K *formulas*. data_only reads a formula's last
        # cached value, which is stale if a mass is edited without an Excel recalc; the
        # raw D/F/H cells are always live. Verified F-D/H-D == the I/K formulas in every
        # block (160/160 rows). KF (U/V) and areas (L/M/N) are already raw/live.
        d, fg, hg = _num(ws_d, f"D{row}"), _num(ws_d, f"F{row}"), _num(ws_d, f"H{row}")
        recs.append(
            {
                "row": row,
                "sysnum": sysnum,
                "ph": ph,
                "phase": phase,
                "L": a.get("2pe"),
                "M": a.get(hba_key),
                "N": a.get(hbd_key),
                "I": (fg - d) if (fg is not None and d is not None) else None,
                "K": (hg - d) if (hg is not None and d is not None) else None,
                "U": _num(ws_d, f"U{row}"),
                "V": _num(ws_d, f"V{row}"),
            }
        )
    return recs


def _fmt(x: float | None) -> str:
    return f"{x:.5f}" if x is not None else ""


def kf_anchor(sol: float, hba: float, hbd: float, water: float) -> tuple[list[float], float]:
    """KF-anchored ternary point + closure Σ. Trust the KF water absolutely and split
    the remaining (1 - water) among the GC components by their measured ratio. Where
    mass already closes (Σ≈1) this equals proportional normalisation; it diverges only
    by the closure gap — exactly where the GC totals are untrustworthy."""
    closure = sol + hba + hbd + water
    gc = sol + hba + hbd
    w = min(max(water, 0.0), 1.0)
    if gc <= 0:
        return [0.0, 0.0, 0.0, 1.0], closure
    scale = (1.0 - w) / gc
    return [sol * scale, hba * scale, hbd * scale, w], closure


def replicate_screen(vials: list[dict]) -> str:
    """'' when a phase's two vials agree, else why not: 'whole_vial' (every component off
    together) or 'component_spread'. Judged on vial_fractions, i.e. after each vial's own
    dilution, so a vial that simply took more sample is not a mismatch."""
    if len(vials) != 2:
        return ""
    lr = [
        math.log(v2 / v1)
        for v1, v2 in ((vials[0][k], vials[1][k]) for k in "sab")
        if v1 and v2 and v1 > 0 and v2 > 0
    ]
    if not lr:
        return ""
    if max(lr) - min(lr) > math.log(SCREEN_SPREAD):  # first: it moves the mean too
        return "component_spread"
    if abs(sum(lr) / len(lr)) > math.log(SCREEN_WHOLE_VIAL):
        return "whole_vial"
    return ""


def mismatch_flag(screen: str, organic: bool) -> bool:
    """Whether a screen failure flags the POINT. An organic phase's whole-vial slip
    scales every component alike and cancels in kf_anchor -- its composition is sound
    (F1, F3, F4, H2, I2), so it stays unflagged; transfer_factors still skips it."""
    return bool(screen) and not (organic and screen == "whole_vial")


# GC batch of a vial row. A batch is one instrument sequence, and it is what the response
# transfer (transfer_factors) is a property of: the campaign-1 blocks ran Mar-Apr at
# 2PE t_R 9.75 min, the May calibration and the September repeats at 10.7-10.85 min.
# repeat_results.patch writes a patched row's batch here; an unwritten row is campaign 1.
BATCH_COL = "AZ"
BINARIES_BATCH = "BINARIOS_TERPENOS"


def batch_of(ws, row: int) -> str:
    if v := ws[f"{BATCH_COL}{row}"].value:
        return str(v)
    block = ws.title.split()[-1]
    return BINARIES_BATCH if row >= 25 else f"{block}1T1 AL {block}5B2"


def transfer_factors(sheets) -> dict[str, float]:
    """{GC batch: phi} -- the batch's response relative to the May calibration (CC_MF),
    the median over its clean organic endpoints of G/(1 - w_KF): an organic phase's true
    non-water mass is 1 - KF, so the GC total G reads phi times it. The organic phase
    needs no phi (it cancels in kf_anchor); the aqueous organics are divided by it
    (results_rows, binary_endpoint). The median keeps one endpoint-level failure (H4 at
    0.55) from moving its batch. *sheets*: (ws, recs) pairs, recs as fill_block returns.
    A batch with no clean organic endpoint has no entry: its aqueous stays uncorrected.
    """
    by: dict[str, list[float]] = {}

    def add(ws, row, use):
        kf = [c for v in use for c in v["kf"]]
        if not kf or sum(kf) / len(kf) >= ORGANIC_WATER_MAX or replicate_screen(use):
            return
        comps = [[v[k] for v in use] for k in "sab"]
        if any(None in c for c in comps):
            return
        G = sum(sum(c) / len(c) for c in comps)
        by.setdefault(batch_of(ws, row), []).append(G / (1 - sum(kf) / len(kf)))

    for ws, recs in sheets:
        f2, g2, h2 = response(ws, "F"), response(ws, "G"), response(ws, "H")
        terp_max = aq_terpene_max(ws["M3"].value, ws["N3"].value)
        groups: dict[tuple[int, str], list[dict]] = {}
        for r in recs:
            groups.setdefault((r["sysnum"], r["ph"]), []).append(r)
        for g in groups.values():
            vials = [v for r in g if (v := vial_fractions(r, f2, g2, h2)) is not None]
            add(ws, g[0]["row"], aqueous_keep(vials, terp_max))
        if ws["A25"].value == "Bin":
            for _, reps in BIN_PAIRS:
                add(ws, reps[0], aqueous_keep(binary_vials(ws, reps, g2, h2), terp_max))
    return {b: statistics.median(v) for b, v in by.items()}


def vial_fractions(r: dict, f2, g2, h2) -> dict | None:
    """One vial's raw mass fractions {s: 2PE, a: HBA, b: HBD, kf: [water...]} from its
    areas, Responses f2/g2/h2 and dilution K/I. None when the dilution is missing."""
    df = (r["K"] / r["I"]) if (r["K"] and r["I"]) else None
    if df is None:
        return None
    s = (f2.conc(r["L"]) * df / 100) if (f2 and r["L"] is not None) else None
    a = (g2.conc(r["M"]) * df / 100) if (g2 and r["M"] is not None) else None
    b = (h2.conc(r["N"]) * df / 100) if (h2 and r["N"] is not None) else None
    ks = [c / 100 for c in (r["U"], r["V"]) if c is not None]
    return {"s": s, "a": a, "b": b, "kf": ks, "raw": r}


def _terp(v: dict) -> float:
    return (v["a"] or 0.0) + (v["b"] or 0.0)


def aqueous_keep(vials: list[dict], terp_max: float) -> list[dict]:
    """The vials of one phase worth averaging. Drops (1) a vial that sampled the wrong
    phase — majority-water AND majority-organic at once (E2 Superior: ~96 % KF water with
    a full organic terpene load); (2) an aqueous vial that took organic-phase droplets —
    terpenes above `terp_max` (see aq_terpene_max) — when its pair is clean, since
    carryover only ever adds organics; (3) below that ceiling, the richer vial of an
    aqueous pair whose components spread apart. If every vial fails, keep them all (the
    point stays flagged downstream)."""

    def water(v):
        return sum(v["kf"]) / len(v["kf"]) if v["kf"] else 0.0

    def mixup(v):
        gc = sum(c for c in (v["s"], v["a"], v["b"]) if c is not None)
        return water(v) > ORGANIC_WATER_MAX and gc > 0.5

    keep = [v for v in vials if not mixup(v)] or vials
    if any(water(v) >= ORGANIC_WATER_MAX for v in keep):  # aqueous phase
        clean = [v for v in keep if _terp(v) <= terp_max]
        keep = clean or keep
        # (3) Droplets below saturation: a pair whose components spread apart
        # (replicate_screen) took organic phase into one vial -- the terpenes jump, 2PE
        # barely moves, B1's excess is the organic phase's own composition. Carryover
        # only adds, so the leaner vial is the aqueous phase (2026-09-28: A2, B1, B5, H2).
        # The terpenes must be what moved: droplets carry 2PE too (B1's is 1.16x) but
        # far less, relative to what is dissolved; a pair apart in 2PE alone is not
        # droplets and stays flagged.
        if replicate_screen(keep) == "component_spread" and all(
            v["s"] and _terp(v) for v in keep
        ):
            (s1, s2), (t1, t2) = ([v["s"] for v in keep], [_terp(v) for v in keep])
            if abs(math.log(t2 / t1)) > abs(math.log(s2 / s1)):
                keep = [min(keep, key=_terp)]
    return keep


def results_rows(
    ws,
    block: str,
    recs: list[dict],
    aqueous_bydiff: bool = True,
    phi: dict[str, float] | None = None,
) -> list[list]:
    """Replicate the sheet formula chain -> one normalized ternary point per (system, phase).

    Organic (water-poor) phases are always KF-anchored. Aqueous (water-rich) phases use
    water BY DIFFERENCE (`aqueous_bydiff=True`, default) or KF proportional normalisation
    (`False`, legacy); by difference, their organics are first divided by *phi* of their
    GC batch (transfer_factors; absent -> 1). The chosen source is emitted per row as
    `water_src` (kf|bydiff)."""
    f2, g2, h2 = response(ws, "F"), response(ws, "G"), response(ws, "H")
    hba_name, hbd_name = ws["M3"].value, ws["N3"].value
    groups: dict[tuple[int, str], list[dict]] = {}
    for r in recs:
        groups.setdefault((r["sysnum"], r["ph"]), []).append(r)  # one point per system+phase

    out = []
    for (sysnum, ph), g in sorted(groups.items()):
        system = f"{block}{sysnum}"
        phase = "Superior" if ph == "T" else "Inferior"
        # Per-vial composition (fractions) so a replicate that sampled a *different
        # phase* than its pair can be dropped — averaging a mixed pair otherwise makes a
        # mid-triangle phantom (E2 Superior = one genuine aqueous vial + one that read
        # ~96% KF water yet carried a full organic terpene load).
        vials = [v for r in g if (v := vial_fractions(r, f2, g2, h2)) is not None]
        use = aqueous_keep(vials, aq_terpene_max(hba_name, hbd_name))
        dropped = len(use) < len(vials)

        def avg(xs):
            return sum(xs) / len(xs) if xs else None

        sol = [v["s"] for v in use if v["s"] is not None]
        hba = [v["a"] for v in use if v["a"] is not None]
        hbd = [v["b"] for v in use if v["b"] is not None]
        # KF from the kept vials; if the kept vial was never titrated, fall back to the
        # pair's (I2 Superior: the clean vial has no KF). Aqueous water is by difference
        # anyway, so the KF only classifies the phase and feeds the closure diagnostic.
        kf = [c for v in use for c in v["kf"]] or [c for v in vials for c in v["kf"]]
        w, x, y, z = avg(sol), avg(hba), avg(hbd), avg(kf)
        # No KF at all (the repeat campaign titrated organic phases only): a phase the GC
        # finds mostly water is aqueous, water by difference — binary_vials' rule. Its
        # closure is 1 by construction, so it is left blank rather than reported.
        no_kf = not kf and None not in (w, x, y) and w + x + y < ORGANIC_WATER_MAX
        if no_kf:
            z = 1.0 - (w + x + y)
        flags = [] if (g2 and h2) else ["slopes_missing"]
        closure = None
        src = ""
        if None not in (w, x, y, z):
            closure = w + x + y + z
            if z < ORGANIC_WATER_MAX:
                # Water-poor (organic) phase: KF titrates its small water content
                # reliably, so anchor it and split the rest by GC ratio — this fixes the
                # closure-inflated D1/I1 curve. (By-difference is wrong here: water is the
                # minor component, so it would absorb all the GC closure error.)
                norm, _ = kf_anchor(w, x, y, z)
                src = "kf"
            elif aqueous_bydiff:
                # Water-rich (aqueous) phase, water BY DIFFERENCE: water = 1 - Σ(organics).
                # At ~0.95+ water the KF titration is unreliable, and water so dominates
                # that it's insensitive to the method; the dissolved GC species are
                # divided by their batch's response transfer and are otherwise as
                # measured. `closure` (below) keeps the raw KF sum as a drift diagnostic.
                p = phi.get(batch_of(ws, g[0]["row"]), 1.0) if phi else 1.0
                norm = [w / p, x / p, y / p, max(0.0, 1.0 - (w + x + y) / p)]
                src = "bydiff"
            else:
                # Water-rich phase, KF-normalised (legacy --aqueous-water kf).
                tot = closure if closure > 0 else 1.0
                norm = [w / tot, x / tot, y / tot, z / tot]
                src = "kf"
            lo, hi = CLOSURE_BAND
            if no_kf:
                closure = None
            elif not (lo <= closure <= hi):
                flags.append("low_closure")
            if mismatch_flag(replicate_screen(use), organic=z < ORGANIC_WATER_MAX):
                flags.append("replicate_mismatch")
            if dropped:
                flags.append("dropped_replicate")
        else:
            norm = [None, None, None, None]
            flags.append("incomplete")
        # Write the final point straight into AD..AG as static values so the workbook
        # matches this CSV. The in-sheet formulas average the phase's rows and can't
        # express the per-vial mixup drop (e.g. E2), so they'd otherwise show a phantom.
        if norm[0] is not None:
            anchor = min(r["row"] for r in g)
            for col, val in zip(("AD", "AE", "AF", "AG"), norm):
                ws[f"{col}{anchor}"] = val
        out.append(
            [
                block,
                system,
                phase,
                _fmt(norm[0]),
                hba_name,
                _fmt(norm[1]),
                hbd_name,
                _fmt(norm[2]),
                _fmt(norm[3]),
                _fmt(closure),
                len(use),
                ";".join(flags),
                src,
            ]
        )
    return out


# Result formulas, generated uniformly for every block. Only Bloque A shipped with
# these (and only partially — its AD..AG summary was Superior-only, rows 5-13), so we
# write the full set rather than copy A's gaps. Inputs (L/M/N areas, masses D-K, KF
# U/V, slopes F2-H2, names M3/N3) live in other cells and are never touched.
PER_VIAL = {  # one per data row 5..24
    "O": "=(L{r}/$F$2)^(1/$F$3)",
    "P": "=(M{r}/$G$2)^(1/$G$3)",
    "Q": "=(N{r}/$H$2)^(1/$H$3)",  # diluted %m/m = (area/α)^(1/β)
    "R": "=O{r}*$K{r}/$I{r}",
    "S": "=P{r}*$K{r}/$I{r}",
    "T": "=Q{r}*$K{r}/$I{r}",  # x dilution
}
PER_PHASE = {  # on the first row r of each (system, phase) pair; averages rows r, r+1
    "W": "=AVERAGE(R{r}:R{s})/100",
    "X": "=AVERAGE(S{r}:S{s})/100",  # mean fraction over reps
    "Y": "=AVERAGE(T{r}:T{s})/100",
    "Z": "=AVERAGE(U{r}:V{s})/100",  # Z = KF water
    "AA": "=SUM(W{r}:Z{r})",  # raw closure (≈1); flags a bad phase even after a drop
    # AD..AG (the normalized ternary point) are NOT formulas: results_rows writes them
    # as static values so the workbook matches the CSV. In-sheet formulas would average
    # both replicate rows and so can't express the per-vial mixup drop (e.g. E2).
}


def write_formulas(wb) -> None:
    """Write the full result-formula set into every block (per-vial + per-phase)."""
    for ws in wb.worksheets:
        if ws["A25"].value == "Bin":  # binary edge: add_binary_tielines wired M/$G$2 there
            for r in range(25, 29):
                ws[f"P{r}"], ws[f"Q{r}"] = (PER_VIAL[c].format(r=r) for c in ("P", "Q"))
        for r in range(5, 25):
            for col, f in PER_VIAL.items():
                ws[f"{col}{r}"] = f.format(r=r)
        for r in range(5, 25, 2):  # phase-pair anchor rows 5,7,...,23
            for col, f in PER_PHASE.items():
                ws[f"{col}{r}"] = f.format(r=r, s=r + 1)


# Forces LibreOffice to recompute formulas on load (OOXMLRecalcMode 0 = always).
RECALC_XCU = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="OOXMLRecalcMode" oor:op="fuse"><value>0</value></prop></item>
 <item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="ODFRecalcMode" oor:op="fuse"><value>0</value></prop></item>
</oor:items>"""


def recalc_workbook(path: Path, warn: list[str]) -> None:
    """Recompute all formulas in place via headless LibreOffice (openpyxl can't)."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        warn.append("LibreOffice not found: open the workbook and press Ctrl+Shift+F9 to recalc")
        return
    with tempfile.TemporaryDirectory() as tmp:
        prof = Path(tmp) / "profile"
        (prof / "user").mkdir(parents=True)
        (prof / "user" / "registrymodifications.xcu").write_text(RECALC_XCU)
        out = Path(tmp) / "out"
        out.mkdir()
        r = subprocess.run(
            [
                soffice,
                "--headless",
                f"-env:UserInstallation=file://{prof}",
                "--calc",
                "--convert-to",
                "xlsx",
                "--outdir",
                str(out),
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        produced = out / path.name
        if produced.exists():
            shutil.move(str(produced), str(path))
            print(f"Recalculated {path} via LibreOffice")
        else:
            warn.append(
                f"LibreOffice recalc failed ({r.stderr.strip()[:120]}); open + Ctrl+Shift+F9"
            )


def run_tieline_plots(warn: list[str]) -> None:
    """Run the pcsaft-quaternary tie-line plotter on the CSV we just wrote. It lives
    in a separate uv project (its venv has python-ternary/matplotlib, which this one
    does not), so shell out with `uv run` in that dir rather than importing it."""
    plot_dir = Path(__file__).parent / "pcsaft-quaternary"
    if not (plot_dir / "plot_experimental_tielines.py").exists():
        warn.append(f"tie-line plotter not found under {plot_dir}")
        return
    r = subprocess.run(
        ["uv", "run", "python", "plot_experimental_tielines.py"],
        cwd=plot_dir,
        capture_output=True,
        text=True,
    )
    print(r.stdout, end="")
    if r.returncode != 0:
        warn.append(f"tie-line plotting failed: {r.stderr.strip()[:200]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill the ternary workbook + emit results CSVs.")
    ap.add_argument(
        "--aqueous-water",
        choices=["difference", "kf"],
        default="difference",
        help="aqueous-phase water: 'difference' (1-Σorganics, default) or 'kf' (KF-normalised). "
        "Organic phases are always KF-anchored.",
    )
    args = ap.parse_args()
    aqueous_bydiff = args.aqueous_water == "difference"
    print(f"aqueous-phase water: {args.aqueous_water}")

    cc = cc_mf_models(CC_MF)
    wb = openpyxl.load_workbook(WB_IN, data_only=False)  # keep formulas (write target)
    wb_d = openpyxl.load_workbook(WB_IN, data_only=True)  # cached values (read masses/KF)
    priority = hba_priority(wb)  # HBA/HBD ordering learned from labelled sheets
    warn: list[str] = []
    all_rows: list[list] = []
    bin_areas = load_binary_areas(BINARIOS_CSV, warn)  # tie-line areas (rows 25-28)

    filled = []  # (ws, block, recs): results_rows needs every batch's phi first
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        block = sheet.split()[-1]  # "Bloque A" -> "A"
        folders = glob(str(DATA / f"{block}1T1 AL *"))
        if not folders:
            warn.append(f"{block}: no FLECK data folder")
            continue
        merged = _ROOT / "out" / Path(folders[0]).name / "merged_samples.csv"
        if not merged.exists():
            warn.append(f"{block}: {merged} missing - run label_terpenos.py")
            continue
        areas = load_vial_areas(merged)
        recs = fill_block(ws, wb_d[sheet], block, areas, cc, priority, warn)
        fill_binary(ws, block, bin_areas, warn)  # Water-solvent tie-line M/N (rows 25-28)
        filled.append((ws, block, recs))
        print(
            f"Bloque {block}: filled {len(recs)}/20 vials "
            f"(HBA={ws['M3'].value}, HBD={ws['N3'].value}, 2PhEt α={_num(ws, 'F2')} β={_num(ws, 'F3')})"
        )
    phi = transfer_factors([(ws, recs) for ws, _, recs in filled])
    print("response transfer phi per GC batch:", {b: round(p, 3) for b, p in sorted(phi.items())})
    for ws, block, recs in filled:
        all_rows += results_rows(ws, block, recs, aqueous_bydiff, phi)

    write_formulas(wb)
    wb.save(WB_OUT)
    recalc_workbook(WB_OUT, warn)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "block",
                "system",
                "phase",
                "solute_2phet",
                "HBA",
                "HBA_wt",
                "HBD",
                "HBD_wt",
                "water",
                "closure",
                "n_vials",
                "flags",
                "water_src",
            ]
        )
        w.writerows(all_rows)
    # Water-solvent binary tie-line endpoints (rows 25-28) for the plotter. Read from the
    # recalc'd file so the pre-wired AE/AF/AG formulas have cached values.
    wb_bin = openpyxl.load_workbook(WB_OUT, data_only=True)
    bin_rows = binary_tieline_rows(wb_bin, phi)
    with BIN_OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "phase", "HBA", "HBA_wt", "HBD", "HBD_wt", "water", "water_src"])
        w.writerows(bin_rows)
    print(
        f"\nWrote {WB_OUT}\nWrote {OUT_CSV}\nWrote {BIN_OUT_CSV} ({len(bin_rows)} binary endpoint(s))"
    )
    run_tieline_plots(warn)
    if warn:
        print("\nWARNINGS:")
        for m in warn:
            print("  -", m)


def _selfcheck() -> None:
    models = cc_mf_models(CC_MF)
    assert len(models) == 7 and all(1.0 < m.beta < 1.1 for m in models.values()), models
    # The power law reads every standard back within 6 % (through-origin linear: up to 23 %
    # low at the bottom); camphor, the noisiest curve, needs it: its lowest standard is +5.8 %.
    for sheet, _, w, a in calibration_points(CC_MF):
        m = models[canon(sheet)]
        w *= PURITY[canon(sheet)]
        assert abs(m.conc(a) / w - 1) < 0.06, (sheet, w, m.conc(a))
    assert abs(m.conc(m.alpha * 7.0**m.beta) - 7.0) < 1e-9, "conc does not invert alpha·C^beta"
    a = load_vial_areas(_ROOT / "out" / "A1T1 AL A5B2" / "merged_samples.csv")
    assert abs(a[(1, "T", 1)]["2pe"] - 143.81) < 1, a[(1, "T", 1)]
    assert canon("2PE") == "2pe" and canon("DL-Camphor") == "camphor", "canon broken"
    assert parse_key("BM1-T1") == (1, "T", 1) and parse_key("E3_B2") == (3, "B", 2), "parse_key"
    print("selfcheck OK")


if __name__ == "__main__":
    main()
