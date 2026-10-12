# picos-gc Repo Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sort picos-gc's top level by role (`lle/`, `data/{gc,workbooks,entry}`, `templates/`, regenerable-only `out/`) without changing a byte of the export chain's output.

**Architecture:** Every move is a `git mv`. The ten LLE scripts move together into `lle/`, so their mutual top-level imports keep resolving (Python puts the running script's directory on `sys.path`). Every location lives in one new module, `lle/paths.py`; the scripts import it (`import paths`) instead of each computing `_ROOT`. Proof of no breakage = the chain run before and after produces byte-identical CSVs.

**Tech Stack:** Python 3.13, openpyxl, pytest, ruff, typst; uv (the main checkout's `.venv` — see Global Constraints).

**Spec:** `docs/superpowers/specs/2026-10-12-repo-layout-design.md`

## Global Constraints

- Work in the worktree `/Users/maximilianfleckenstein/projects/TESIS/picos-gc/.claude/worktrees/tidy-2026-10-12` (branch `worktree-tidy-2026-10-12`). Shell: absolute paths, `git -C <worktree>`, no `cd`, no shell loops/variables (the isolation guard refuses them).
- Python: `PY=/Users/maximilianfleckenstein/projects/TESIS/picos-gc/.venv/bin/python` (written out in full in each command), with `PYTHONPATH=src` (pre-move) or `PYTHONPATH=src:lle` (post-move) so the worktree's own code wins over the venv's editable install.
- Moves with `git mv` only; delete nothing; `out/` stays gitignored and holds nothing tracked.
- Output CSVs must be byte-identical before/after (`diff -rq -x '*.png' -x '*.xlsx' -x '*.pdf'`).
- Folder names in English, exactly: `lle/`, `data/gc/`, `data/workbooks/`, `data/entry/`, `templates/`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push the branch; never merge to master.

## Review Focus

1. Running a script from a CWD other than the repo root (e.g. `lle/`) — expected: same files written to the same `out/`. Pinned by `tests/test_paths.py` (all locations are absolute, rooted at the repo).
2. A re-run of `make_entry_workbook.py` writes its `.bak.xlsx` beside the entry sheet in `data/entry/` — expected: ignored by git. Pinned by a `git check-ignore` step in Task 2.
3. A fresh clone has no `out/`, so no filled workbook — expected: `export_vial_data`/`audit_repeats` fail with a FileNotFoundError naming `out/Sistemas ternarios_MF_filled.xlsx`; CLAUDE.md says to run `fill_ternarios.py` first (Task 3).
4. TESIS `bin/import-lle` run against a picos-gc master that has not merged this branch — expected: the TESIS change (Task 4) lands only after the picos-gc merge; the report states the order.
5. `repeat_list.typ` moved into `lle/` reads a CSV outside its own directory — expected: compiles with `typst compile --root . lle/repeat_list.typ out/repeat_list.pdf`. Checked in Task 3.

---

### Task 1: Baseline snapshot (no commit)

**Files:** none tracked. Writes the worktree's `out/` and `/Users/maximilianfleckenstein/.claude/jobs/c9efa1af/tmp/out_baseline/` (temporary, this session only).

- [ ] **Step 1: Run the chain on the current layout** (from the worktree root, the Bash cwd)

```
PYTHONPATH=src /Users/maximilianfleckenstein/projects/TESIS/picos-gc/.venv/bin/python label_terpenos.py
PYTHONPATH=src /Users/maximilianfleckenstein/projects/TESIS/picos-gc/.venv/bin/python fill_ternarios.py
PYTHONPATH=src /Users/maximilianfleckenstein/projects/TESIS/picos-gc/.venv/bin/python export_vial_data.py
PYTHONPATH=src /Users/maximilianfleckenstein/projects/TESIS/picos-gc/.venv/bin/python repeat_results.py
```
Expected: each exits 0 (fill_ternarios may warn "tie-line plotter not found" — the submodule is not initialised in the worktree; that is fine, it writes no CSV).

- [ ] **Step 2: Snapshot** `cp -R <worktree>/out /Users/maximilianfleckenstein/.claude/jobs/c9efa1af/tmp/out_baseline`

- [ ] **Step 3: Cross-check against the main checkout's `out/`** (also proves the lint commit changed no number)

```
diff -rq -x '*.png' -x '*.xlsx' -x '*.pdf' -x 'repeat_compare' /Users/maximilianfleckenstein/.claude/jobs/c9efa1af/tmp/out_baseline /Users/maximilianfleckenstein/projects/TESIS/picos-gc/out
```
Expected: only "Only in" lines for files the main checkout has and the worktree does not (e.g. `repeat_list.csv`); no "differ" line. A "differ" line stops the plan: investigate before moving anything.

- [ ] **Step 4: Restore the tracked filled workbook** the run rewrote: `git -C <worktree> checkout -- "Sistemas ternarios_MF_filled.xlsx"`.

### Task 2: Move by role, one paths module (one atomic commit)

**Files:**
- Create: `lle/paths.py`, `tests/test_paths.py`
- Move: the ten scripts + `repeat_list.typ` → `lle/`; `FLECK_TERPENOS2026/` → `data/gc/`; `Sistemas ternarios_MF.xlsx`, `CC_MF.xlsx`, `aromas_equilibrios_vfinal.xlsx` → `data/workbooks/`; `out/repeat_entry.xlsx`, `out/repeat{2,3,4}_entry.xlsx` → `data/entry/`; `PLANTILLA_IMPRESION.xlsx`, `PLANTILLA_IMPRESION_puros.xlsx` → `templates/`
- Untrack: `Sistemas ternarios_MF_filled.xlsx` (file moves to `out/`)
- Modify: `lle/fill_ternarios.py:40-48,853,897,966`, `lle/label_terpenos.py:38-39`, `lle/repeat_results.py:65-85`, `lle/export_vial_data.py:32-37`, `lle/audit_repeats.py:59-61`, `lle/make_entry_workbook.py:41`, `lle/make_repeat_workbook.py:36-39,51`, `lle/compare_aqueous_water.py:32-36`, `lle/deconv_impact.py:107`, `pyproject.toml`, `.gitignore`

**Interfaces:**
- Produces: `paths.ROOT, GC, WORKBOOKS, ENTRY, TEMPLATES, OUT, PCSAFT_QUATERNARY` (all absolute `Path`). Existing public constants keep their names and meaning (`fill_ternarios.WB_IN`, `WB_OUT`, `CC_MF`, `CAL_AREAS`, …); only their values move.

- [ ] **Step 1: Write the failing layout test** `tests/test_paths.py`

```python
from paths import ENTRY, GC, OUT, ROOT, TEMPLATES, WORKBOOKS


def test_locations_are_absolute_and_rooted_at_the_repo():
    assert ROOT.is_absolute() and (ROOT / "pyproject.toml").exists()
    assert all(p.is_relative_to(ROOT) for p in (GC, WORKBOOKS, ENTRY, TEMPLATES, OUT))


def test_tracked_inputs_are_where_paths_says():
    assert (GC / "BINARIOS_TERPENOS").is_dir()
    for name in ("Sistemas ternarios_MF.xlsx", "CC_MF.xlsx", "aromas_equilibrios_vfinal.xlsx"):
        assert (WORKBOOKS / name).exists(), name
    assert (TEMPLATES / "PLANTILLA_IMPRESION.xlsx").exists()
    assert sorted(p.name for p in ENTRY.glob("repeat*_entry.xlsx")) == [
        "repeat2_entry.xlsx", "repeat3_entry.xlsx", "repeat4_entry.xlsx", "repeat_entry.xlsx"
    ]
```

- [ ] **Step 2: Run it to see it fail:** `PYTHONPATH=src:lle <PY> -m pytest -q tests/test_paths.py` → `ModuleNotFoundError: No module named 'paths'`.

- [ ] **Step 3: Moves** (each one command, `git -C <worktree> mv …`; `mkdir -p` the targets `lle data/workbooks data/entry templates` first; `git mv FLECK_TERPENOS2026 data/gc` renames the directory)

```
git -C <wt> mv add_binary_tielines.py audit_repeats.py compare_aqueous_water.py deconv_impact.py export_vial_data.py fill_ternarios.py label_terpenos.py make_entry_workbook.py make_repeat_workbook.py repeat_results.py repeat_list.typ lle/
git -C <wt> mv FLECK_TERPENOS2026 data/gc
git -C <wt> mv "Sistemas ternarios_MF.xlsx" CC_MF.xlsx aromas_equilibrios_vfinal.xlsx data/workbooks/
git -C <wt> mv out/repeat_entry.xlsx out/repeat2_entry.xlsx out/repeat3_entry.xlsx out/repeat4_entry.xlsx data/entry/
git -C <wt> mv PLANTILLA_IMPRESION.xlsx PLANTILLA_IMPRESION_puros.xlsx templates/
git -C <wt> rm --cached "Sistemas ternarios_MF_filled.xlsx"
mv "<wt>/Sistemas ternarios_MF_filled.xlsx" "<wt>/out/"
```

- [ ] **Step 4: Create `lle/paths.py`**

```python
"""Where everything lives. A layout change is made here and nowhere else."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GC = ROOT / "data" / "gc"  # raw .gcd, one folder per GC batch
WORKBOOKS = ROOT / "data" / "workbooks"  # hand-made inputs
ENTRY = ROOT / "data" / "entry"  # typed repeat-round entry sheets (lab record)
TEMPLATES = ROOT / "templates"
OUT = ROOT / "out"  # regenerable only
PCSAFT_QUATERNARY = ROOT / "pcsaft-quaternary"
```

- [ ] **Step 5: Point the constants at it.** Add `import paths` to each script's first-party import block, delete its `_ROOT = …` line, then:

| File | Old | New |
|---|---|---|
| fill_ternarios.py | `WB_IN = _ROOT / "Sistemas ternarios_MF.xlsx"` | `WB_IN = paths.WORKBOOKS / "Sistemas ternarios_MF.xlsx"` |
| | `WB_OUT = _ROOT / "Sistemas ternarios_MF_filled.xlsx"` | `WB_OUT = paths.OUT / "Sistemas ternarios_MF_filled.xlsx"` |
| | `CC_MF = _ROOT / "CC_MF.xlsx"` | `CC_MF = paths.WORKBOOKS / "CC_MF.xlsx"` |
| | `DATA = _ROOT / "FLECK_TERPENOS2026"` | `DATA = paths.GC` |
| | every `_ROOT / "out"` (lines 44, 46-48, 897, 966) | `paths.OUT` |
| | `plot_dir = Path(__file__).parent / "pcsaft-quaternary"` | `plot_dir = paths.PCSAFT_QUATERNARY` |
| label_terpenos.py | `DATA_DIR = Path("FLECK_TERPENOS2026")` / `OUT_DIR = Path("out")` | `DATA_DIR = paths.GC` / `OUT_DIR = paths.OUT` |
| repeat_results.py | `_ROOT / "FLECK_TERPENOS2026" / X` (3×) | `paths.GC / X` |
| | `_ROOT / "out" / "repeat*_entry.xlsx"` (4×) | `paths.ENTRY / "repeat*_entry.xlsx"` |
| | `CC_MF = _ROOT / "CC_MF.xlsx"` / `OUT = _ROOT / "out"` | `CC_MF = paths.WORKBOOKS / "CC_MF.xlsx"` / `OUT = paths.OUT` |
| export_vial_data.py | `WB = _ROOT / "Sistemas ternarios_MF_filled.xlsx"` | `WB = paths.OUT / "Sistemas ternarios_MF_filled.xlsx"` |
| | `CC = _ROOT / "CC_MF.xlsx"`; `_ROOT / "out"` (3×) | `CC = paths.WORKBOOKS / "CC_MF.xlsx"`; `paths.OUT` |
| audit_repeats.py | `WB = _ROOT / "Sistemas …_filled.xlsx"`; `OUT = _ROOT / "out" / …` | `WB = paths.OUT / "Sistemas ternarios_MF_filled.xlsx"`; `OUT = paths.OUT / "repeat_list.csv"` |
| make_entry_workbook.py | `OUT = Path(__file__).resolve().parent / "out" / "repeat_entry.xlsx"` | `OUT = paths.ENTRY / "repeat_entry.xlsx"` |
| make_repeat_workbook.py | `SRC = _ROOT / "aromas…"`; `LIST`/`OUT = _ROOT / "out" / …`; `SAMPLING_SRC = _ROOT / "PLANTILLA_IMPRESION.xlsx"` | `paths.WORKBOOKS / …`; `paths.OUT / …`; `paths.TEMPLATES / "PLANTILLA_IMPRESION.xlsx"` |
| compare_aqueous_water.py | `_ROOT / "out" / …` (3×); `PLOT_DIR = _ROOT / "pcsaft-quaternary"` | `paths.OUT / …`; `paths.PCSAFT_QUATERNARY` |
| deconv_impact.py | `out_dir = Path("out")` | `out_dir = paths.OUT` |

Then `grep -n "_ROOT\|Path(\"out\")\|Path(\"FLECK" <wt>/lle/*.py` → no hits. Drop any `from pathlib import Path` ruff now flags unused.

- [ ] **Step 6: Config.** `pyproject.toml`: `[tool.pytest.ini_options] pythonpath = [".", "lle"]`; `[tool.ruff]` add `src = ["src", "lle"]`. `.gitignore` under "Generated outputs": `*.bak.xlsx`.

- [ ] **Step 7: Tests + lint:** `PYTHONPATH=src:lle <PY> -m pytest -q` → green, same skip count as before (plus the new file passing); `<ruff> check <wt>` and `<ruff> format --check <wt>` clean.

- [ ] **Step 8: Review Focus 2:** `git -C <wt> check-ignore -v data/entry/repeat_entry.bak.xlsx` → prints the `*.bak.xlsx` rule.

- [ ] **Step 9: Chain on the new layout, byte-compared.** First delete every CSV the baseline run left in the worktree's `out/`, so a file the new layout fails to write shows up as "Only in …out_baseline" instead of passing stale:

```
find <wt>/out -name '*.csv' -delete
PYTHONPATH=src /Users/maximilianfleckenstein/projects/TESIS/picos-gc/.venv/bin/python lle/label_terpenos.py
PYTHONPATH=src /Users/maximilianfleckenstein/projects/TESIS/picos-gc/.venv/bin/python lle/fill_ternarios.py
PYTHONPATH=src /Users/maximilianfleckenstein/projects/TESIS/picos-gc/.venv/bin/python lle/export_vial_data.py
PYTHONPATH=src /Users/maximilianfleckenstein/projects/TESIS/picos-gc/.venv/bin/python lle/repeat_results.py
diff -rq -x '*.png' -x '*.xlsx' -x '*.pdf' /Users/maximilianfleckenstein/.claude/jobs/c9efa1af/tmp/out_baseline <wt>/out
```
Expected: no output from `diff`. (Deleting is safe: no tracked file lives in `out/` any more, and every CSV there is regenerated by the chain.)

- [ ] **Step 10: Commit and push**

```
git -C <wt> add -A
git -C <wt> commit -m "refactor(layout): sort the repo by role -- lle/, data/{gc,workbooks,entry}, templates/

Every location lives in lle/paths.py. The filled workbook is derived, so it is
untracked and written to out/. The export chain's CSVs are byte-identical
before and after.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git -C <wt> push
```

### Task 3: Docs inside picos-gc

**Files:** `CLAUDE.md`, `README.md:121`, `.claude/skills/repeat-prep-book/SKILL.md:3-14,34`, `lle/repeat_list.typ:2,7`, script docstrings' `Run:` lines (`audit_repeats.py:30`, `make_entry_workbook.py:14-15`, `make_repeat_workbook.py:17`, `add_binary_tielines.py:18`, `deconv_impact.py:7`, `fill_ternarios.py:17`).

- [ ] **Step 1:** `repeat_list.typ` line 7 → `csv("/out/repeat_list.csv", row-type: dictionary)`; line 2 → `// Regenerate:  uv run python lle/audit_repeats.py && typst compile --root . lle/repeat_list.typ out/repeat_list.pdf`.
- [ ] **Step 2: Review Focus 5:** `PYTHONPATH=src <PY> lle/audit_repeats.py` then `typst compile --root <wt> <wt>/lle/repeat_list.typ <wt>/out/repeat_list.pdf` → exit 0.
- [ ] **Step 3:** CLAUDE.md: replace the "Which file holds what" inputs bullet with the new locations (`data/workbooks/`, `data/entry/` tracked normally, `data/gc/`), the derived bullet with "the filled workbook in `out/` (untracked; run `lle/fill_ternarios.py` first on a fresh clone)", add one line "Layout: `lle/paths.py` is the only place a location is defined"; rewrite every `uv run X.py` → `uv run python lle/X.py`, `FLECK_TERPENOS2026/` → `data/gc/`, `out/repeat*_entry.xlsx` → `data/entry/repeat*_entry.xlsx`, and drop "force-tracked … `git add -f`". Same substitutions in the skill and README line 121 and the docstring `Run:` lines.
- [ ] **Step 4:** `grep -rn "FLECK_TERPENOS2026/\|out/repeat.*_entry\|git add -f\|uv run [a-z_]*\.py" <wt>/CLAUDE.md <wt>/README.md <wt>/.claude/skills <wt>/lle` → only hits that are history prose (e.g. the folder name of a batch); fix the rest.
- [ ] **Step 5:** Commit `docs: point CLAUDE.md, README, skill and usage lines at the new layout` and push.

### Task 4: TESIS consumers (separate repo, after the picos-gc merge)

**Files:** TESIS `bin/import-lle:36-39`, `data/README.md:763`.

- [ ] **Step 1:** From a TESIS worktree (EnterWorktree is per-session; use `git -C /Users/maximilianfleckenstein/projects/TESIS worktree add .claude/worktrees/picos-layout -b picos-layout` and absolute paths), edit `bin/import-lle`: `uv run python lle/label_terpenos.py`, `uv run python lle/fill_ternarios.py`, `uv run python lle/export_vial_data.py`, `uv run python lle/repeat_results.py`. The `cp "$PICOS/out/…"` lines are unchanged.
- [ ] **Step 2:** `data/README.md:763`: `picos-gc/aromas_equilibrios_vfinal.xlsx` → `picos-gc/data/workbooks/aromas_equilibrios_vfinal.xlsx`.
- [ ] **Step 3:** `bash -n bin/import-lle` → exit 0. Commit `chore(import-lle): follow picos-gc's lle/ layout` (the TESIS commit gate runs the TESIS suite; it must be green), push the branch, do not merge.
- [ ] **Step 4: Review Focus 4 — order:** report that picos-gc's branch merges first, TESIS's second; the next real `bin/import-lle` run is the end-to-end check (it should leave picos-gc clean now).
