# picos-gc

Run: `uv run picos-gc *.gcd` (or a directory). Test: `uv run pytest`. Uses `uv`, not pip/python directly.
`pcsaft-quaternary` is a submodule: a fresh clone needs `git submodule update --init` before uv can sync.

## Which file holds what (LLE measurements)
- **Inputs, hand-made, never written by a script:** `data/workbooks/Sistemas ternarios_MF.xlsx`
  (campaign 1: masses, KF), one typed entry sheet per repeat round
  (`data/entry/repeat_entry.xlsx`, `repeat2_entry.xlsx` … `repeat4_entry.xlsx`), one GC
  folder per batch under `data/gc/`, `data/workbooks/CC_MF.xlsx`.
- **The complete record:** `out/vial_ledger.csv` (`repeat_results.py`) — every vial ever
  injected, campaign 1 (`round` 0) and each round in `ROUNDS`, with `status`
  current/superseded and `superseded_by`. "Did I re-do X?" is answered here.
- **Derived views:** `out/vial_measurements_repeat.csv` (the ledger's current rows; main
  checks they agree), the tie-line exports `out/*_repeat.csv`, the filled workbook
  (`out/Sistemas ternarios_MF_filled.xlsx`, untracked: on a fresh clone run
  `lle/label_terpenos.py` and `lle/fill_ternarios.py` first). TESIS
  copies the ledger, the current vials, the tie-lines and `gc_calibration.csv`.
- A new round: add its GC folder and entry sheet, append it to `ROUNDS`, re-run.

## Architecture (`src/picos_gc/`)
Deconvolution impact on real data (`lle/deconv_impact.py`) is small: quantified analytes are well-resolved, so sizable peaks (area ≥ 1) move <~23% only for a handful of fused shoulders and the giant solvent peak — quantified terpene/2PE areas barely change. Run: `uv run python lle/deconv_impact.py data/gc/<batch>`.

## Excel mapping (downstream quantification)
`label_terpenos.py` labels each injection by its embedded `sample_name` (blanks →
"BLANK", dropped) and writes `out/<batch>/merged_samples.csv`. `fill_ternarios.py`
reads that CSV and joins to `Sistemas ternarios_MF.xlsx` rows via
`parse_key(name) -> (system_number, phase T/B, replicate 1/2)` — e.g. `I1_T1 → (1,"T",1)`.
The sample name is the join key, so it must be the real embedded name, not the
opaque `BATCH..._NNN.gcd` filename.

The CC_MF standards are integrated the same way (`process_calibration` →
`out/AaCALIBRACION_TERPENOS/areas.csv`; `CAM3` = camphor E3, `CAM-SM` = stock), and
`fill_ternarios.calibration_points` joins those areas to CC_MF's %m/m. CC_MF's own col O
was integrated by hand and misses its chromatograms by up to 7 %, so it is not read. The
standards skip the global arPLS baseline, which clips their tails (up to 9 %) while
leaving every sample peak above 30 mV·min within 0.4 %.

Per-phase water source (`--aqueous-water`, default `difference`): **organic** (water-poor)
phases are KF-anchored; **aqueous** (water-rich) phases take water **by difference**
(`1−Σ organics`, KF used only to classify + as the `closure` QC diagnostic) — KF is
unreliable at ~0.95+ water. `--aqueous-water kf` restores the legacy proportional
normalisation exactly. Each ternary CSV row carries a `water_src` (kf|bydiff) column.
`compare_aqueous_water.py` scores KF vs by-difference (closure drift); it needs a KF
baseline (`--aqueous-water kf`) and refuses a by-difference CSV.

Each ternary sheet also carries its Water–solvent **binary edge tie-line** (HBA+HBD+Water,
no 2PE) at rows 25–28, added by `add_binary_tielines.py` (pre-wired formulas reusing the
sheet's own `$G$2`/`$H$2` cells; `write_formulas` rewrites their P/Q to the power law, α in
row 2 and β in row 3). `fill_ternarios.py` auto-fills the HBA/HBD **areas**
(`M/N`) there from `out/BINARIOS_TERPENOS/samples.csv` via `BIN_TO_BLOCK` (**sequential**:
BIN _n_ → the _n_-th block in order A,B,C,D,E,F,H,I — 1=ThyCarvone(A) … 8=CamphorEugenol(I);
all 8 map); masses `D/F/H` and KF water `U/V` stay hand-entered. Carvone reads as "Geraniol"
in `samples.csv` (binary-method RT miscalibrated), so blocks A & F route their Carvone area
through `fill_binary`'s single-remaining-terpene fallback (right value, wrong label). The
block sits below the pipeline's hardcoded rows 5–24, so the ternary CSV/`results_rows` never
sees it. `fill_ternarios.binary_tieline_rows` emits the endpoints to `out/binarios_tielines.csv`
computed **from the raw cells** (masses D/F/H, areas M/N, KF U/V) through the same chain as
the ternaries (`binary_vials` → `aqueous_keep` cherry-pick → `binary_endpoint`): the
**organic** endpoint (water < 0.5) is KF-anchored (`water_src=kf`), the **aqueous** one takes
water **by difference** (`bydiff`), so aqueous KF is optional. Never read the sheet's cached
binary formula cells (AA/AE–AG rows 25–28): they go stale between recalcs — Bloque H's cached
AA25 read 0.35 while its raw cells close at 0.94. A block needs masses + the organic KF. All 8 blocks resolve both endpoints (every aqueous one `bydiff`: no
aqueous KF was measured).

## Response transfer (per GC batch)
The CC_MF calibration (13–14 May) is applied to sequences run in another column state: the
campaign-1 blocks (Mar–Apr, 2PE t_R 9.75 min) read G/(1 − w_KF) = 0.81–1.08 on their organic
endpoints, the May binaries 0.97, the September repeats (t_R 10.7, the calibration's state)
0.996. `transfer_factors` takes each batch's median over its clean organic endpoints;
`results_rows`, `binary_endpoint` and `two_pe_endpoint` divide the **aqueous** organics by it
(it cancels in the organic KF anchor). The batch is the block's own sequence for campaign 1,
`BINARIOS_TERPENOS` for the edges, and the round folder for a row `repeat_results.patch`
overwrote (`BATCH_COL`, in-memory only). A batch with no clean organic endpoint stays
at 1 unless `TRANSFER_PROXY` lends it another batch's (round 2 borrows round 1's). Evidence it is the instrument, not the sample: the 2PE-rich aqueous family read 7.5 %
low in campaign-1 blocks against round-1 ones before the correction, +0.5 % after.

## Repeat audit
`uv run python lle/audit_repeats.py` reads the filled workbook only (no other `out/` file)
and writes `out/repeat_list.csv`: one row per (system, phase) point, ternary **and** binary,
with a `repeat` verdict. The reason catalogue, each threshold constant and what it caught
are in the `repeat-audit` skill — read it before touching the script or a threshold.
Re-run `fill_ternarios.py` first if the master workbook changed. Printable lab worksheet
(checkboxes, sample codes, per-reason action):
`typst compile --root . lle/repeat_list.typ out/repeat_list.pdf` — it reads the CSV, no Python.

Lab prep book for the repeat campaign: `uv run python lle/make_repeat_workbook.py [EXTRA_TUBE ...]` →
`out/aromas_equilibrios_repeat.xlsx` plus the **bench printout** `*_print.xlsx` + `.pdf`
(print that one). How it trims, hides and paginates, and what the printout hides, is in the
`repeat-prep-book` skill — read it before touching that script.

Later rounds: `--round N TUBE ...` on both `make_repeat_workbook.py` and
`make_entry_workbook.py`; the mechanics (exact tubes only, per-round files, the `:aq`
aqueous-only suffix) and how the entry sheet is built and re-read are in the `repeat-round`
skill. Round 2 (2026-09-23), every tube aqueous-only: `--round 2 A1:aq A4:aq A5:aq F5:aq I2:aq B2:aq B3:aq B4:aq
BIN6:aq`. Block A's organic phase
floats when thymol–carvone-rich (A1 A3 A4) and sinks when 2PE-rich (A2 A5); the entry
sheet reads that off campaign-1 KF, so it is right, not a mix-up.

Data-entry sheet for the repeat vials: `uv run python lle/make_entry_workbook.py [EXTRA_TUBE ...]` →
`data/entry/repeat_entry.xlsx`. **The user types into that file**; a re-run carries the typed
cells over and keeps the previous file as `.bak.xlsx` (details in `repeat-round`). The filled
`data/entry/repeat_entry.xlsx` is the lab record of the repeat campaign's weighings and KF, so
it is **tracked** (later edits show as modified; its `.bak.xlsx` is ignored).

## Repeat campaign results
`uv run python lle/repeat_results.py` integrates `data/gc/REPETICIONES_22SEP2026/` with its **own** retention-time map
(`REPEAT_TR`: a new 17-min method, 2PE at 10.72, carvone at 9.67 — `label_terpenos`' maps
would read carvone as 2PE) and the block taken from the sample-name prefix (the folder mixes
blocks; `BIN<n>` → `BIN_TO_BLOCK`). It patches each weighed repeat vial (masses, KF from
`data/entry/repeat_entry.xlsx`, areas) over its campaign-1 row in an in-memory copy of
`out/Sistemas ternarios_MF_filled.xlsx`, whole tubes at a time, and runs the unchanged chain
(`results_rows`, `binary_tieline_rows`, `audit`). Response: CC_MF, the May calibration, as a
power law (`cc_mf_models`; the batch has no standards). The water–2PE binary has no sheet: organic 2PE = 1 − KF, aqueous from the
GC. Writes `out/repeat_tielines.csv` (old vs new per repeated tube), `out/repeat_list_after.csv`,
`out/repeat_compare/<block>.png`. Repeat aqueous vials carry no KF, so `results_rows` takes a
KF-less mostly-water phase as by-difference with a blank closure (`binary_vials`' rule).
Rounds stack in `ROUNDS` order (round 2 = `REPETICIONES_26SEP2026` + `data/entry/repeat2_entry.xlsx`,
aqueous vials only): "old" is every round but the last, the export carries all of them;
`ALIAS` maps round 2's phase-less `BIN6_1/2` to `BIN6_B1/B2`.
Round 3 (prepared 2026-09-28, both phases, fresh organic KF): `--round 3 B5 BIN2`, 8 vials (the water-2PE binary was on the entry sheet too, but the user did not run it -- dropped 2026-09-29; round 1's stays) -- the
two campaign-1 points round 2 left above block B's trend. Measured by the user on
2026-09-28 (the thymol-geraniol points); landed with round 4, below. **One round = one GC batch + one entry sheet**: never add tubes to a round already run.

Round 4 (prepared 2026-09-28, aqueous only -- the user's call: the organic phases are sound,
a whole-vial slip cancels in the anchor): `--round 4 H2:aq H5:aq I4:aq BIN5:aq`
(8 vials) -- two of the replicate screen's undecidable aqueous pairs (H5, I4), the E edge,
and H2 (its one surviving vial sits off both its block's terpene trend and K). Not
repeated, the user's calls: F3 and F4 also fail the screen, but their means sit on block
F's trend, so they keep the two-point interval. C2 reads above its own water-solvent edge
(1218 ± 97 vs 1018 ± 75 ppm, 1.63x their combined U since 2026-09-29); it was put back
into round 4 on paper (branch `round3-plan`, 56f3ac7) but the user did not run it
(2026-10-12), so it stays campaign 1's. The droplet pairs
(A2, B1, B5, H2) otherwise need no repeat: aqueous_keep takes their leaner vial. An
aqueous-only batch has no organic endpoint to estimate phi from: run a CC_MF standard at
the start, middle and end of the sequence (or add it to TRANSFER_PROXY, as round 2 is).

Rounds 3 and 4 landed together (2026-10-12): the user injected both in one sequence,
`REPETICIONES_07OCT2026` (8-9 Oct, 16 vials, no standard), so `ROUNDS` holds them as one
entry with both entry sheets (`ROUNDS`' second element is a tuple of sheets) and the
ledger calls all 16 vials round 3. Its phi (0.937) comes from round 3's organic phases.
Round 3's organic KF was not re-measured: the user chose to reuse campaign 1's (written
into the sheet as `reused: campaign 1`). B5 and the B edge cleared, so `TENTATIVE` is
empty; H5, I4 and the E edge cleared; H2's aqueous now fails the screen on carvacrol at
the floor (0.015 vs 0.037 mV·min, its 2PE agrees to 10 %). Both entry sheets hide every
column but the three masses (user's request); a rebuild unhides them.
