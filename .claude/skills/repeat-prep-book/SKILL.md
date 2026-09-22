---
name: repeat-prep-book
description: How make_repeat_workbook.py builds the repeat campaign's lab prep book (out/aromas_equilibrios_repeat.xlsx) — hidden-not-deleted tubes, Sampling vial-sheet row heights and pagination, DES prep tables. Use before changing or debugging make_repeat_workbook.py or its test.
---

Lab prep book for the repeat campaign: `uv run make_repeat_workbook.py [EXTRA_TUBE ...]`
(e.g. `D2`) trims `aromas_equilibrios_vfinal.xlsx` (the campaign's prep workbook: feed
`z` per tie-line, `Lab_DES` tube-weighing sheet, `Sheet2` GC-vial sheet) to the tubes with
a `repeat=yes` point in `out/repeat_list.csv` (binaries as `BIN<n>` via `BIN_TO_BLOCK`)
and writes `out/aromas_equilibrios_repeat.xlsx`. `Lab_DES` keeps only the volume guide
(A–I): the weighing/record columns J–T are hidden, since the feed is just a way into the
two-phase region and the results come from the sampled phases. Other tubes are **hidden, not deleted**
(the volume estimates are cross-sheet formulas openpyxl would not rewrite; unhide to get
the full book back). The printable vial sheets (`PLANTILLA_IMPRESION.xlsx` → `Sampling`, copied in
cell by cell) keep the four vials of every kept tube plus their block's three header
rows; block I (CamEug) is not in that template and is cloned from the last ternary block
after the binaries, titled from `Lab_DES`. Vial codes stay `C4-T1` (hyphen; `CODE_RE`
accepts it). Every vial row is re-heighted to `SAMPLING_ROW_H` and every column header to
`SAMPLING_HEAD_H`: the template mixes 15 pt (ternary, too small to hand-write a mass into)
with 70.85 pt (binary, five rows to a page), and 23.85 pt clips `m_solvent (g)`. `_paginate`
then drops manual page breaks so a system's four vials, and a block header with the system
under it, never straddle a page; `SAMPLING_PAGE_H` is in row-height points, measured off a
render, because fit-to-width shrinks this sheet to ~78 % and a 451 pt page therefore holds
~580 pt of rows. `Sheet2` (the prep book's own binary vial sheet) is hidden as redundant. The 10 g
DES prep table (`datos_des`) is cropped to the DES the **ternary** tubes use, its
experimental masses (G/L) cleared for the new weighings, and every binary gets a
duplicated row below the legend (`_des_table`, labelled `<DES> (BINn)`, formulas
re-pointed by `_repoint`): a binary is its own Falcon of solvent, weighed apart even when
the pair repeats a ternary one. The console prints grams per Falcon, keyed the same way.
The 5 g variant
(`Sheet1`) is hidden — every batch is made at 10 g to have spare. The console still
prints the grams of each DES the tubes consume (cached `Lab_DES` estimates; binaries as
4 mL) as a check against that 10 g. Test:
`tests/test_make_repeat_workbook.py` (same no-project invocation).
