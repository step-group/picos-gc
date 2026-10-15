---
name: repeat-audit
description: The repeat-audit reason catalogue behind lle/audit_repeats.py and out/repeat_list.csv — what each reason (missing_vials, low_closure, replicate_mismatch, aqueous_organics_suspect, aqueous_2pe_outlier, edge_ternary_mismatch) tests, its threshold constant, why it exists and which points it caught. Use before changing audit_repeats.py or its thresholds, or when reading a repeat verdict.
---

`uv run python lle/audit_repeats.py` reads the filled workbook only (no other `out/` file)
and writes `out/repeat_list.csv`: one row per (system, phase) point, ternary **and** binary,
with a `repeat` verdict. Re-run `fill_ternarios.py` first if the master workbook changed.

Reasons: `missing_vials` (blank L/M/N = never injected),
`low_closure`, `replicate_mismatch` (both via `fill_ternarios.results_rows`, so they match
the pipeline; the latter is `replicate_screen` — the two vials' dilution-corrected fractions
off together by > `SCREEN_WHOLE_VIAL` (a weighing/transcription slip) or apart by >
`SCREEN_SPREAD` (droplets), either phase; it replaced a 3x raw-area gate that passed all 13
failed pairs, 2026-09-28), `aqueous_organics_suspect` (organic droplets in **every** kept aqueous vial: terpenes above
the pair's summed pure-water solubility at 30 °C +20 % method tolerance, `AQ_SOLUBILITY_30C`
× `AQ_SOLUBILITY_TOL` in `aq_terpene_max`, 2.8–4.9 g/L by pair) and
`aqueous_2pe_outlier` (the 2PE-rich endpoint of every block is the same system — organic
phase 87–89 % 2PE, ~8 % water in all eight — so their aqueous phases are one number
measured once per block, 21–27 g/L since the transfer correction; `_2pe_outliers` flags a miss
of the median by > `AQ_2PE_OUTLIER`. It is a cross-block pass, the only rule that needs
every sheet at once, and it cannot be an absolute literature bound: published 2PE
solubility at 30 °C spans 21–33 g/L, wider than the disagreement) and
`edge_ternary_mismatch` (a block's water–solvent edge and its solvent-richest tie-line —
the one fed ~2 % 2PE — sit against the same ~96 % solvent organic phase, so their aqueous
terpene must agree; `_edge_vs_ternary` flags a factor over `EDGE_TERNARY_MAX` and blames
the **richer** of the two, since droplets only add terpene. Catches A-bin, C-bin, H1).

Informational, never a repeat: `single_vial` (one clean vial is accepted) and
`dropped_replicate` — `fill_ternarios.aqueous_keep` already cherry-picks: it drops a vial
that sampled the wrong phase (E2) or an aqueous vial with droplets whose pair is clean (D2,
D5, F5, I2), and the `dropped` column names it. Points with no vials (C4, C5) appear here
but are **absent** from `ternarios_resultados.csv`. A single KF titration is reported as
`n_kf`, not a repeat reason.

Printable lab worksheet (checkboxes, sample codes, per-reason action):
`typst compile --root . lle/repeat_list.typ out/repeat_list.pdf` — it reads the CSV, no Python.
Test: `tests/test_audit_repeats.py`.
