# /// script
# requires-python = ">=3.11"
# dependencies = ["openpyxl"]
# ///
"""Trim the lab prep workbook (aromas_equilibrios_vfinal.xlsx) to the repeat campaign.

Tubes = the tie-lines with a repeat=yes point in out/repeat_list.csv (ternary systems,
binaries as BIN<n> via fill_ternarios.BIN_TO_BLOCK) plus any extra codes on argv. Rows of
every other tube are HIDDEN, not deleted: Lab_DES' volume estimates are cross-sheet
formulas into '2-phenylethanol_DES' and openpyxl would not rewrite them. A re-prepared
tube is a new tie-line, so Sheet2 (GC vial weighing) gets all four vials (T1 T2 B1 B2)
of every kept ternary tube, coded like the binaries already are (C4-T1; fill_ternarios'
CODE_RE accepts the hyphen).

Run: uv run make_repeat_workbook.py [EXTRA_TUBE ...]   e.g.  uv run make_repeat_workbook.py D2
  -> out/aromas_equilibrios_repeat.xlsx
"""

from __future__ import annotations

import csv
import re
import sys
from copy import copy
from pathlib import Path

import openpyxl

from fill_ternarios import BIN_TO_BLOCK

_ROOT = Path(__file__).resolve().parent
SRC = _ROOT / "aromas_equilibrios_vfinal.xlsx"
LIST = _ROOT / "out" / "repeat_list.csv"
OUT = _ROOT / "out" / "aromas_equilibrios_repeat.xlsx"
_FEED_REF = re.compile(r"'2-phenylethanol_DES'!I(\d+)")
LAB_ROWS, LAB_BIN_HEADER = range(10, 67), 58
FEED_ROWS, GC_ROWS, GC_HEADER, GC_APPEND = range(6, 64), range(4, 43), 3, 44
# DES prep table (10 g, 1:1 molar): rows 17-25 of datos_des, DES name in B. Sheet1 is
# the 5 g variant: hidden in the output, every batch is made at 10 g to have spare.
DES_TABLES = (("datos_des", range(17, 26), "B"),)
HIDE_SHEETS = ("Sheet1",)
BIN_V_DES_ML = 4  # Lab_DES F59:F66 (binary rows carry V_DES in F, no density)


def tubes_from_list(path: Path) -> set[str]:
    bin_of = {blk: n for n, blk in BIN_TO_BLOCK.items()}
    tubes = set()
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["repeat"] == "yes":
                tubes.add(r["system"] if r["kind"] == "ternary" else f"BIN{bin_of[r['block']]}")
    return tubes


def _backfill(ws, row: int) -> None:
    """Block labels (DES/HBA/HBD in B-D) sit only on a block's first row: copy them down."""
    if ws[f"B{row}"].value is not None:
        return
    src = row - 1
    while ws[f"B{src}"].value is None:
        src -= 1
    for c in "BCD":
        ws[f"{c}{row}"].value = ws[f"{c}{src}"].value


def _hide(ws, rows, keep: set[int]) -> None:
    for r in rows:
        ws.row_dimensions[r].hidden = r not in keep


def _copy_row(ws, src: int, dst: int) -> None:
    for c in range(1, 10):
        ws.cell(dst, c).value = ws.cell(src, c).value
        ws.cell(dst, c)._style = copy(ws.cell(src, c)._style)
    ws.row_dimensions[dst].height = ws.row_dimensions[src].height


def trim(wb, tubes: set[str]) -> None:
    lab = wb["Lab_DES"]
    known = {lab[f"A{r}"].value for r in LAB_ROWS}
    if missing := tubes - known:
        raise ValueError(f"not in Lab_DES: {sorted(missing)}")
    keep_lab = {r for r in LAB_ROWS if lab[f"A{r}"].value in tubes}
    for r in keep_lab:
        _backfill(lab, r)
    has_bin = any(t.startswith("BIN") for t in tubes)
    _hide(lab, LAB_ROWS, keep_lab | ({LAB_BIN_HEADER} if has_bin else set()))

    des = {lab[f"B{r}"].value for r in keep_lab}  # after back-fill every kept row names it
    for sheet, rows, cols in DES_TABLES:
        ws = wb[sheet]
        if missing := des - {ws[f"{c}{r}"].value for r in rows for c in cols}:
            raise ValueError(f"{sheet}: no prep row for {sorted(missing)}")
        _hide(ws, rows, {r for r in rows if any(ws[f"{c}{r}"].value in des for c in cols)})
    for sheet in HIDE_SHEETS:
        wb[sheet].sheet_state = "hidden"

    feed = wb["2-phenylethanol_DES"]
    keep_feed = set()
    for r in keep_lab:
        if m := _FEED_REF.search(str(lab[f"G{r}"].value or "")):
            keep_feed.add(int(m.group(1)))
    for r in keep_feed:
        _backfill(feed, r)
    _hide(feed, FEED_ROWS, keep_feed)

    gc = wb["Sheet2"]
    keep_gc = set()
    for r in GC_ROWS:
        if gc[f"A{r}"].value in tubes:
            keep_gc |= set(range(r, r + 4))
    _hide(gc, GC_ROWS, keep_gc)
    row = GC_APPEND
    _copy_row(gc, GC_HEADER, row)
    for tube in sorted(t for t in tubes if not t.startswith("BIN")):
        for j, (phase, tag, rep) in enumerate(
            (("Top", "T", 1), ("Top", "T", 2), ("Bot", "B", 1), ("Bot", "B", 2))
        ):
            row += 1
            _copy_row(gc, GC_ROWS.start + j, row)  # borders/fills of the BIN1 block
            gc[f"A{row}"] = tube if j == 0 else None
            gc[f"B{row}"] = phase if rep == 1 else None
            gc[f"C{row}"], gc[f"D{row}"] = rep, f"{tube}-{tag}{rep}"


def des_need(tubes: set[str]) -> dict[str, float]:
    """Grams of each DES the kept tubes consume, from Lab_DES' cached estimates
    (ternary rows: V_DES est [I] x rho_DES est [F]; binary rows: 4 mL x that DES' rho)."""
    lab = openpyxl.load_workbook(SRC, data_only=True)["Lab_DES"]
    rows = {r: lab[f"A{r}"].value for r in LAB_ROWS}
    des_of, rho = {}, {}
    for r, code in rows.items():  # block labels only on first rows: carry the DES down
        if lab[f"B{r}"].value:
            block_des = lab[f"B{r}"].value
        if code:
            des_of[r] = block_des
            if r < LAB_BIN_HEADER:
                rho[block_des] = lab[f"F{r}"].value
    need: dict[str, float] = {}
    for r, code in rows.items():
        if code in tubes:
            ml = lab[f"I{r}"].value if r < LAB_BIN_HEADER else BIN_V_DES_ML
            need[des_of[r]] = need.get(des_of[r], 0) + ml * rho.get(des_of[r], 1.0)
    return need


def main() -> None:
    tubes = tubes_from_list(LIST) | set(sys.argv[1:])
    wb = openpyxl.load_workbook(SRC)
    trim(wb, tubes)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    order = sorted(tubes, key=lambda t: (t.startswith("BIN"), t))
    n_tern = sum(not t.startswith("BIN") for t in tubes)
    print(f"{len(tubes)} tubes: {' '.join(order)}; {4 * n_tern} ternary GC vials. Wrote {OUT}")
    print(
        "DES needed (est.): "
        + ", ".join(f"{d} {g:.1f} g" for d, g in sorted(des_need(tubes).items()))
    )


if __name__ == "__main__":
    main()
