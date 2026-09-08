# /// script
# requires-python = ">=3.11"
# dependencies = ["openpyxl"]
# ///
"""Which LLE tie-line points must be repeated? One row per (system, phase) point.

Reads the *filled* workbook only (no out/ dependency): blank L/M/N areas mean no
chromatogram reached the sheet, i.e. the vial was never injected. Reuses
fill_ternarios' own composition chain (results_rows) so closure/flags here equal
the pipeline's. Binary edge rows 25-28 get the same treatment (fill_ternarios'
binary export carries no closure at all).

Repeat reasons: missing_vials, single_vial, low_closure, replicate_mismatch,
dropped_replicate, aqueous_organics_suspect. A single KF titration is reported
(n_kf) but is not a reason.

aqueous_organics_suspect = organic-phase droplets carried into an aqueous sample.
The pipeline's replicate_mismatch is gated on a component > 10 %, which an aqueous
phase never has, so it is blind there. Two signals, either one fires: a vial whose
terpenes (HBA+HBD) exceed AQ_TERPENE_MAX (water dissolves ~0.1-0.25 % of each of
these terpenes; typical aqueous vials here sit at 0.01-0.05 %), or the two vials'
total organics (2PE+HBA+HBD) differing by more than MISMATCH_MAX x.

Run: uv run audit_repeats.py   ->  out/repeat_list.csv + console table
"""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl

from fill_ternarios import (
    BIN_TO_BLOCK,
    CLOSURE_BAND,
    MISMATCH_MAX,
    ORGANIC_WATER_MAX,
    _binary_bydiff,
    _num,
    _vial_rows,
    results_rows,
    vial_fractions,
)

_ROOT = Path(__file__).resolve().parent
WB = _ROOT / "Sistemas ternarios_MF_filled.xlsx"
OUT = _ROOT / "out" / "repeat_list.csv"
REPEAT_FLAGS = ("low_closure", "replicate_mismatch", "dropped_replicate")
# ponytail: fixed ceiling above the summed water solubility of any two of these terpenes;
# make it per-pair (solubility table) if a legit 2PE-cosolvency case ever exceeds it.
AQ_TERPENE_MAX = 0.005
COLS = [
    "kind", "block", "system", "phase", "hba", "hbd", "codes", "n_expected", "n_with_areas",
    "n_vials_used", "n_kf", "closure", "water_src", "aq_terpene_max", "aq_organics_ratio",
    "flags", "repeat", "reason",
]  # fmt: skip
_BIN_CODE = {25: "T1", 26: "T2", 27: "B1", 28: "B2"}  # fill_ternarios._BIN_ROWS inverted


def _codes(prefix: str, reps: list[str], have: list[bool]) -> str:
    """Sample codes to prepare: only the missing ones when some are missing, else all."""
    missing = [c for c, ok in zip(reps, have, strict=True) if not ok]
    return ", ".join(f"{prefix}-{c}" for c in (missing or reps))


def _has_areas(ws, r: int) -> bool:
    return any(_num(ws, f"{c}{r}") is not None for c in "LMN")


def _n_kf(ws, rows) -> int:
    return sum(_num(ws, f"{c}{r}") is not None for r in rows for c in "UV")


def _aqueous_organics(vials: list[dict]) -> tuple[float | None, float | None, bool]:
    """(max terpene fraction, max/min total-organics ratio, suspect?) over an aqueous
    phase's vials. None/False when the phase is organic or has no usable vial."""
    kf = [c for v in vials for c in v["kf"]]
    if not kf or sum(kf) / len(kf) < ORGANIC_WATER_MAX:
        return None, None, False
    terp = [(v["a"] or 0) + (v["b"] or 0) for v in vials]
    tot = [(v["s"] or 0) + t for v, t in zip(vials, terp, strict=True)]
    ratio = (max(tot) / min(tot)) if len(tot) > 1 and min(tot) > 0 else None
    suspect = max(terp) > AQ_TERPENE_MAX or (ratio is not None and ratio > MISMATCH_MAX)
    return max(terp), ratio, suspect


def _verdict(row: dict, flags: str) -> dict:
    reason = []
    if row["n_with_areas"] == 0:
        reason.append("missing_vials")
    elif row["n_with_areas"] < row["n_expected"]:
        reason.append("single_vial")
    reason += [f for f in flags.split(";") if f in REPEAT_FLAGS]
    if row.pop("aq_suspect", False):
        reason.append("aqueous_organics_suspect")
    row["flags"] = flags
    row["repeat"] = "yes" if reason else "no"
    row["reason"] = ";".join(reason)
    return row


def _ternary(ws, block: str) -> list[dict]:
    groups: dict[tuple[int, str], list[int]] = {}
    for row, sysnum, ph, _rep, phase in _vial_rows(ws):
        if phase:
            groups.setdefault((sysnum, ph), []).append(row)
    recs = []
    for (sysnum, ph), rows in groups.items():
        for r in rows:
            if not _has_areas(ws, r):
                continue
            d, f, h = _num(ws, f"D{r}"), _num(ws, f"F{r}"), _num(ws, f"H{r}")
            rec = {
                "row": r, "sysnum": sysnum, "ph": ph, "phase": ws[f"B{r}"].value,
                "L": _num(ws, f"L{r}"), "M": _num(ws, f"M{r}"), "N": _num(ws, f"N{r}"),
                "I": (f - d) if None not in (f, d) else None,
                "K": (h - d) if None not in (h, d) else None,
                "U": _num(ws, f"U{r}"), "V": _num(ws, f"V{r}"),
            }  # fmt: skip
            recs.append(rec)
    # results_rows: [block, system, phase, ..., closure(9), n_vials(10), flags(11), src(12)]
    computed = {(r[1], r[2]): r for r in results_rows(ws, block, recs)}
    f2, g2, h2 = _num(ws, "F2"), _num(ws, "G2"), _num(ws, "H2")
    out = []
    for (sysnum, ph), rows in sorted(groups.items()):
        system, phase = f"{block}{sysnum}", "Superior" if ph == "T" else "Inferior"
        c = computed.get((system, phase))
        vials = [
            v
            for r in recs
            if (r["sysnum"], r["ph"]) == (sysnum, ph)
            and (v := vial_fractions(r, f2, g2, h2)) is not None
        ]
        terp, ratio, suspect = _aqueous_organics(vials)
        have = [_has_areas(ws, r) for r in rows]
        row = {
            "kind": "ternary", "block": block, "system": system, "phase": phase,
            "hba": ws["M3"].value, "hbd": ws["N3"].value,
            "codes": _codes(system, [f"{ph}{i}" for i in range(1, len(rows) + 1)], have),
            "n_expected": len(rows),
            "n_with_areas": sum(have),
            "n_vials_used": c[10] if c else 0,
            "n_kf": _n_kf(ws, rows),
            "closure": c[9] if c else "",
            "water_src": c[12] if c else "",
            "aq_terpene_max": f"{terp:.5f}" if terp is not None else "",
            "aq_organics_ratio": f"{ratio:.2f}" if ratio is not None else "",
            "aq_suspect": suspect,
        }  # fmt: skip
        out.append(_verdict(row, c[11] if c else ""))
    return out


def _binary(ws, block: str) -> list[dict]:
    if ws["A25"].value != "Bin":
        return []
    g2, h2 = _num(ws, "G2"), _num(ws, "H2")
    lo, hi = CLOSURE_BAND
    out = []
    for row, reps in ((25, (25, 26)), (27, (27, 28))):
        water, src = _num(ws, f"AG{row}"), "kf"
        if water is None:
            bd = _binary_bydiff(ws, reps, g2, h2)
            water, src = (bd[2], "bydiff") if bd else (None, "")
        closure = _num(ws, f"AA{row}")
        flags = "low_closure" if closure is not None and not (lo <= closure <= hi) else ""
        have = [_num(ws, f"M{x}") is not None or _num(ws, f"N{x}") is not None for x in reps]
        binnum = next(b for b, blk in BIN_TO_BLOCK.items() if blk == block)
        r = {
            "kind": "binary", "block": block, "system": f"{block}-bin",
            "phase": "" if water is None else ("aqueous" if water > 0.5 else "organic"),
            "hba": ws["M3"].value, "hbd": ws["N3"].value,
            "codes": _codes(f"BIN{binnum}", [_BIN_CODE[x] for x in reps], have),
            "n_expected": len(reps),
            "n_with_areas": sum(have),
            "n_vials_used": len(reps) if water is not None else 0,
            "n_kf": _n_kf(ws, reps),
            "closure": f"{closure:.5f}" if closure is not None else "",
            "water_src": src,
            "aq_terpene_max": "", "aq_organics_ratio": "",  # no 2PE, KF-less aqueous rows
        }  # fmt: skip
        out.append(_verdict(r, flags))
    return out


def audit(wb) -> list[dict]:
    """All (system, phase) points of a data_only workbook, with a repeat verdict each."""
    rows = []
    for sheet in wb.sheetnames:
        ws, block = wb[sheet], sheet.split()[-1]
        rows += _ternary(ws, block) + _binary(ws, block)
    return rows


def main() -> None:
    rows = audit(openpyxl.load_workbook(WB, data_only=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    rep = [r for r in rows if r["repeat"] == "yes"]
    print(
        f"{'point':<10}{'phase':<10}{'vials':<7}{'closure':<10}{'aq_terp':<9}{'aq_ratio':<10}reason"
    )
    for r in rep:
        vials = f"{r['n_with_areas']}/{r['n_expected']}"
        print(
            f"{r['system']:<10}{r['phase']:<10}{vials:<7}{r['closure']:<10}"
            f"{r['aq_terpene_max']:<9}{r['aq_organics_ratio']:<10}{r['reason']}"
        )
    print(f"\n{len(rep)} of {len(rows)} points need repeating. Wrote {OUT}")


if __name__ == "__main__":
    main()
