---
name: repeat-round
description: How to prepare and record a new repeat round of LLE vials — the `--round N TUBE ...` and `:aq` mechanics shared by make_repeat_workbook.py and make_entry_workbook.py, and how the typed entry sheet (data/entry/repeat<N>_entry.xlsx) is built, re-read and protected. Use when adding a round, re-running either script, or changing which cells the user can type into.
---

Later rounds: `--round N TUBE ...` on both `make_repeat_workbook.py` and
`make_entry_workbook.py` takes exactly the named tubes (no `repeat_list.csv`, no
`EXTRA_TUBES`), writes `out/aromas_equilibrios_repeat<N>.xlsx` / `data/entry/repeat<N>_entry.xlsx`
so round 1's typed record is never rebuilt over, and opens every organic KF cell (a new
tube is a new tie-line). A `:aq` suffix (`B2:aq`) makes a tube aqueous-only: the entry sheet
lists only its aqueous vials, so `repeat_results.patch` replaces only those and the organic
phase and its KF stay campaign 1's; the prep book's vial sheet hides the two organic rows
(which phase is organic comes from campaign-1 KF, `organic_phase`, now in
`make_repeat_workbook`) and notes it in Notes.

Data-entry sheet for the repeat vials: `uv run python lle/make_entry_workbook.py [EXTRA_TUBE ...]` →
`data/entry/repeat_entry.xlsx`, one row per vial of the same tubes plus `EXTRA_TUBES` (the `2PE`
water binary, `2PE-T1` … `2PE-B2`, and D2). **The user types into that file**: a re-run reads
every unlocked cell of the existing one (`typed`), writes it back by vial code (`carry`),
keeps the previous file as `.bak.xlsx`, and refuses to save if a typed value has no input
cell left. Masses on every vial; KF only on the **organic** phase (picked from
the campaign-1 KF < `ORGANIC_WATER_MAX`; `ORGANIC_OVERRIDE` for a tube with no history: `2PE`) and
only typeable for the tubes in `FRESH_KF` (organic KF re-measured) — the others show their
campaign-1 `U/V` read-only. Sheet protected without password; row 7 names the master
workbook column of each value. Test: `tests/test_make_entry_workbook.py`.

After the round is measured: add its GC folder under `data/gc/` and its entry sheet to
`ROUNDS` in `lle/repeat_results.py`, then re-run. One round = one GC batch + one entry
sheet: never add tubes to a round already run (the rule and the round history live in
`CLAUDE.md`).
