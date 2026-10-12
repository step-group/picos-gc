# picos-gc repo layout — design (2026-10-12)

## Goal

Sort picos-gc's top level by role so it reads at a glance. The audience is the user and
the thesis: one repo, no split. Success = the export chain produces byte-identical outputs
before and after, the suite passes, no old path survives, and TESIS `bin/import-lle` keeps
working.

## Layout

```
picos-gc/
├── src/picos_gc/        GC integrator CLI — unchanged
├── tests/               unchanged location
├── lle/                 the terpene LLE chain (all scripts move together)
│   ├── label_terpenos.py  fill_ternarios.py  repeat_results.py  export_vial_data.py
│   ├── audit_repeats.py   make_repeat_workbook.py  make_entry_workbook.py
│   ├── add_binary_tielines.py  compare_aqueous_water.py  deconv_impact.py
│   └── repeat_list.typ
├── data/
│   ├── gc/              ← FLECK_TERPENOS2026/*  (raw .gcd, one folder per GC batch)
│   ├── workbooks/       ← Sistemas ternarios_MF.xlsx, CC_MF.xlsx, aromas_equilibrios_vfinal.xlsx
│   └── entry/           ← out/repeat_entry.xlsx, out/repeat{2,3,4}_entry.xlsx
├── templates/           ← PLANTILLA_IMPRESION.xlsx, PLANTILLA_IMPRESION_puros.xlsx
├── docs/  pcsaft-quaternary/  (unchanged)
└── out/                 regenerable only; nothing tracked
```

`Sistemas ternarios_MF_filled.xlsx` is derived (fill_ternarios writes it on every run), so
it moves to `out/` and is untracked. This ends the "every import leaves picos-gc dirty"
problem: `bin/import-lle` refuses a dirty tree and then rewrote a tracked file.

The entry sheets are hand-typed lab records; in `data/entry/` they are tracked normally
instead of force-added into an ignored directory. Their `.bak.xlsx` and
`repeat_entry.user-*.xlsx` copies stay in `out/` untouched (untracked, machine-local).

## Mechanics

- Every move is `git mv` (history follows). Scripts move as one set, so their mutual
  top-level imports keep resolving (Python puts the running script's directory on the path).
- Path constants to edit: each `_ROOT = Path(__file__).resolve().parent` → `.parent.parent`;
  `WB_IN`, `WB_OUT`, `CC_MF`, the `FLECK_TERPENOS2026` roots, the entry-sheet paths in
  `make_entry_workbook`/`repeat_results`, the `aromas_equilibrios_vfinal` and
  `PLANTILLA_IMPRESION` paths, and `fill_ternarios`' `pcsaft-quaternary` plot dir.
- `make_entry_workbook` (any `--round N`) writes the entry sheet to `data/entry/`, where the
  user types; its `.bak.xlsx` lands beside it and is gitignored (`*.bak.xlsx`).
  `make_repeat_workbook`'s prep books stay in `out/`.
- `pyproject.toml`: pytest `pythonpath = ["lle"]`.
- Docs and consumers: picos-gc `CLAUDE.md`, `README.md`, `.claude/skills/repeat-prep-book`,
  script docstrings/usage lines; TESIS `bin/import-lle` (`lle/<script>.py`, and it now
  needs no workbook commit after a run) and `data/README.md:763`.

## Out of scope

Console-script entry points (callers by name, not path): one consumer, YAGNI. Splitting the
integrator from the LLE chain into two repos. Deleting any file.

## Verification

1. Before moving: run the chain (`label_terpenos`, `fill_ternarios`, `export_vial_data`,
   `repeat_results`) on the current tree; snapshot `out/*.csv`.
2. After: same chain from `lle/`; `cmp` every snapshotted CSV — byte-identical required.
3. `pytest` green with the same skip set; `ruff check`.
4. `grep` for the old root paths (`FLECK_TERPENOS2026` at root, `Sistemas ternarios_MF`
   at root, `out/repeat*_entry`) finds only history/prose that is meant to.
5. Merge order: picos-gc branch first, then TESIS `bin/import-lle` + `data/README.md`
   (one TESIS commit); the next `bin/import-lle` run is the end-to-end check.
