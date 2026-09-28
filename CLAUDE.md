# picos-gc

CLI tool for automatic multi-peak integration of Shimadzu `.gcd` GC files.

Run: `uv run picos-gc *.gcd` (or a directory). Test: `uv run pytest`. Uses `uv`, not pip/python directly.

## Which file holds what (LLE measurements)
- **Inputs, hand-made, never written by a script:** `Sistemas ternarios_MF.xlsx` (campaign 1:
  masses, KF), one typed entry sheet per repeat round (`out/repeat_entry.xlsx`,
  `out/repeat2_entry.xlsx`, `out/repeat3_entry.xlsx` — force-tracked although `out/` is
  ignored; `git add -f`), one GC folder per batch under `FLECK_TERPENOS2026/`, `CC_MF.xlsx`.
- **The complete record:** `out/vial_ledger.csv` (`repeat_results.py`) — every vial ever
  injected, campaign 1 (`round` 0) and each round in `ROUNDS`, with `status`
  current/superseded and `superseded_by`. "Did I re-do X?" is answered here.
- **Derived views:** `out/vial_measurements_repeat.csv` (the ledger's current rows; main
  checks they agree), the tie-line exports `out/*_repeat.csv`, the filled workbook. TESIS
  copies the ledger, the current vials, the tie-lines and `gc_calibration.csv`.
- A new round: add its GC folder and entry sheet, append it to `ROUNDS`, re-run.

## Architecture (`src/picos_gc/`)
- `deconvolution.py` — EMG (exponentially-modified Gaussian) curve-fit for fused peak groups; `integrator` uses it for `--split-mode deconvolve` (fits N≥2 groups, falls back to `drop` on any fit failure; isolated peaks unchanged, so only fused-group `area_mV_min` differs)

Deconvolution impact on real data (`deconv_impact.py`, repo root) is small: quantified analytes are well-resolved, so sizable peaks (area ≥ 1) move <~23% only for a handful of fused shoulders and the giant solvent peak — quantified terpene/2PE areas barely change. Run: `uv run python deconv_impact.py <dir>`.

## `.gcd` metadata: sample name
The operator-entered sample name lives in the OLE2 `File Property` stream as
`<smpl_name>@StoX@<hex></smpl_name>` (`@StoX@` = Shimadzu "store as hex"; hex-decode
to text). `read_gcd` surfaces it as `Chromatogram.sample_name` (`None` if absent).
Verified present in all project `.gcd` files; `smpl_name == smpl_id`; blanks are
named `BNK*`. It appears as the leading `sample_name` column in both output CSVs.

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
AA25 read 0.35 while its raw cells close at 0.94. A block needs masses + the organic KF. `plot_experimental_tielines.py` draws the endpoints as
a green tie-line on the 2PE-free edge in all three views (`_experimental`, `_experimental_zoom`,
`_aqueous_log`). All 8 blocks resolve both endpoints (every aqueous one `bydiff`: no
aqueous KF was measured).

## Response transfer (per GC batch)
The CC_MF calibration (13–14 May) is applied to sequences run in another column state: the
campaign-1 blocks (Mar–Apr, 2PE t_R 9.75 min) read G/(1 − w_KF) = 0.81–1.08 on their organic
endpoints, the May binaries 0.97, the September repeats (t_R 10.7, the calibration's state)
0.996. `transfer_factors` takes each batch's median over its clean organic endpoints;
`results_rows`, `binary_endpoint` and `two_pe_endpoint` divide the **aqueous** organics by it
(it cancels in the organic KF anchor). The batch is the block's own sequence for campaign 1,
`BINARIOS_TERPENOS` for the edges, and the round folder for a row `repeat_results.patch`
overwrote (`BATCH_COL`, in-memory only). A batch with no clean organic endpoint (round 2) stays
at 1. Evidence it is the instrument, not the sample: the 2PE-rich aqueous family read 7.5 %
low in campaign-1 blocks against round-1 ones before the correction, +0.5 % after.

## Repeat audit
`uv run audit_repeats.py` reads `Sistemas ternarios_MF_filled.xlsx` only (no `out/` needed)
and writes `out/repeat_list.csv`: one row per (system, phase) point, ternary **and** binary,
with a `repeat` verdict. Reasons: `missing_vials` (blank L/M/N = never injected),
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
`n_kf`, not a repeat reason. Re-run `fill_ternarios.py` first if the master
workbook changed. Printable lab worksheet (checkboxes, sample codes, per-reason action):
`typst compile repeat_list.typ out/repeat_list.pdf` — it reads the CSV, no Python. Test without the project venv (it cannot sync while `pcsaft-quaternary`
is an empty gitlink): `uv run --no-project --with openpyxl --with pytest pytest --noconftest
-o pythonpath=. tests/test_audit_repeats.py tests/test_fill_ternarios.py`.

Lab prep book for the repeat campaign: `uv run make_repeat_workbook.py [EXTRA_TUBE ...]` →
`out/aromas_equilibrios_repeat.xlsx`; how it trims, hides and paginates is in the
`repeat-prep-book` skill — read it before touching that script. It also writes the **bench
printout**, `*_print.xlsx` + `.pdf` (headless LibreOffice): only DES prep, tube prep and the
GC vial sheets, the columns computed from what gets written in hidden (`PRINT_HIDE_COLS`),
their width given to Notes so the vial-sheet page breaks still hold. Print that one.

Later rounds: `--round N TUBE ...` on both `make_repeat_workbook.py` and
`make_entry_workbook.py` takes exactly the named tubes (no `repeat_list.csv`, no
`EXTRA_TUBES`), writes `out/aromas_equilibrios_repeat<N>.xlsx` / `out/repeat<N>_entry.xlsx`
so round 1's typed record is never rebuilt over, and opens every organic KF cell (a new
tube is a new tie-line). A `:aq` suffix (`B2:aq`) makes a tube aqueous-only: the entry sheet
lists only its aqueous vials, so `repeat_results.patch` replaces only those and the organic
phase and its KF stay campaign 1's; the prep book's vial sheet hides the two organic rows
(which phase is organic comes from campaign-1 KF, `organic_phase`, now in
`make_repeat_workbook`) and notes it in Notes. Round 2 (2026-09-23), every tube aqueous-only: `--round 2 A1:aq A4:aq A5:aq F5:aq I2:aq B2:aq B3:aq B4:aq
BIN6:aq`; `repeat_results.TENTATIVE` lists the same points. Block A's organic phase
floats when thymol–carvone-rich (A1 A3 A4) and sinks when 2PE-rich (A2 A5); the entry
sheet reads that off campaign-1 KF, so it is right, not a mix-up.

Data-entry sheet for the repeat vials: `uv run make_entry_workbook.py [EXTRA_TUBE ...]` →
`out/repeat_entry.xlsx`, one row per vial of the same tubes plus `EXTRA_TUBES` (the `2PE`
water binary, `2PE-T1` … `2PE-B2`, and D2). **The user types into that file**: a re-run reads
every unlocked cell of the existing one (`typed`), writes it back by vial code (`carry`),
keeps the previous file as `.bak.xlsx`, and refuses to save if a typed value has no input
cell left. Masses on every vial; KF only on the **organic** phase (picked from
the campaign-1 KF < `ORGANIC_WATER_MAX`; `ORGANIC_OVERRIDE` for a tube with no history: `2PE`) and
only typeable for the tubes in `FRESH_KF` (organic KF re-measured) — the others show their
campaign-1 `U/V` read-only. Sheet protected without password; row 7 names the master
workbook column of each value. Test: `tests/test_make_entry_workbook.py`. The filled
`out/repeat_entry.xlsx` is the lab record of the repeat campaign's weighings and KF, so it is
**tracked** (force-added despite `out/` being ignored; later edits show as modified).

## Repeat campaign results
`PYTHONPATH=src:. .venv/bin/python repeat_results.py` (the project venv cannot sync, see above)
integrates `FLECK_TERPENOS2026/REPETICIONES_22SEP2026/` with its **own** retention-time map
(`REPEAT_TR`: a new 17-min method, 2PE at 10.72, carvone at 9.67 — `label_terpenos`' maps
would read carvone as 2PE) and the block taken from the sample-name prefix (the folder mixes
blocks; `BIN<n>` → `BIN_TO_BLOCK`). It patches each weighed repeat vial (masses, KF from
`out/repeat_entry.xlsx`, areas) over its campaign-1 row in an in-memory copy of
`Sistemas ternarios_MF_filled.xlsx`, whole tubes at a time, and runs the unchanged chain
(`results_rows`, `binary_tieline_rows`, `audit`). Response: CC_MF, the May calibration, as a
power law (`cc_mf_models`; the batch has no standards). The water–2PE binary has no sheet: organic 2PE = 1 − KF, aqueous from the
GC. Writes `out/repeat_tielines.csv` (old vs new per repeated tube), `out/repeat_list_after.csv`,
`out/repeat_compare/<block>.png`. Repeat aqueous vials carry no KF, so `results_rows` takes a
KF-less mostly-water phase as by-difference with a blank closure (`binary_vials`' rule).
Rounds stack in `ROUNDS` order (round 2 = `REPETICIONES_26SEP2026` + `out/repeat2_entry.xlsx`,
aqueous vials only): "old" is every round but the last, the export carries all of them;
`ALIAS` maps round 2's phase-less `BIN6_1/2` to `BIN6_B1/B2`.
Round 3 (prepared 2026-09-28, both phases, fresh organic KF): `--round 3 B5 BIN2` (entry sheet: `B5 BIN2 2PE`, the water-2PE binary re-measured too, hand-prepared, not in the prep book) -- the
two campaign-1 points round 2 left above block B's trend; `TENTATIVE` holds them until it
lands, then append its `(batch, out/repeat3_entry.xlsx)` to `ROUNDS`. Extended the same
day with the replicate screen's four undecidable aqueous pairs and the E edge, aqueous only
(the user's call: the organic phases are sound, a whole-vial slip cancels in the anchor):
prep book `--round 3 B5 BIN2 F3:aq F4:aq H5:aq I4:aq BIN5:aq`, entry sheet the same plus
`2PE` -- 22 vials. The droplet pairs (A2, B1, B5, H2 aqueous) need no repeat: aqueous_keep
takes their leaner vial.
