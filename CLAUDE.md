# picos-gc

CLI tool for automatic multi-peak integration of Shimadzu `.gcd` GC files.

Run: `uv run picos-gc *.gcd` (or a directory). Test: `uv run pytest`. Uses `uv`, not pip/python directly.

## Architecture (`src/picos_gc/`)
- `reader.py` — OLE2 binary parsing → `Chromatogram` (also extracts `sample_name`, see below)
- `baseline.py` — optional global baseline correction (`pybaselines`, arPLS default, on by default)
- `detector.py` — `find_peaks` + valley-clipping → `list[DetectedPeak]`
- `integrator.py` — linear baseline + trapezoid → `list[PeakResult]`; `--split-mode` picks how fused peaks are split: `valley` (default), `drop` (perpendicular drop), `deconvolve` (see below)
- `deconvolution.py` — EMG (exponentially-modified Gaussian) curve-fit for fused peak groups; `integrator` uses it for `--split-mode deconvolve` (fits N≥2 groups, falls back to `drop` on any fit failure; isolated peaks unchanged, so only fused-group `area_mV_min` differs)
- `processor.py` — batch pipeline → `list[FileResult]` + tidy `save_csv`
- `aligner.py` — cross-file tR clustering → wide `save_aligned_csv`
- `cli.py` — argparse entry; defaults from `DetectionParams()`

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

Per-phase water source (`--aqueous-water`, default `difference`): **organic** (water-poor)
phases are KF-anchored; **aqueous** (water-rich) phases take water **by difference**
(`1−Σ organics`, KF used only to classify + as the `closure` QC diagnostic) — KF is
unreliable at ~0.95+ water. `--aqueous-water kf` restores the legacy proportional
normalisation exactly. Each ternary CSV row carries a `water_src` (kf|bydiff) column.
`compare_aqueous_water.py` scores KF vs by-difference (closure drift); it needs a KF
baseline (`--aqueous-water kf`) and refuses a by-difference CSV.

Each ternary sheet also carries its Water–solvent **binary edge tie-line** (HBA+HBD+Water,
no 2PE) at rows 25–28, added by `add_binary_tielines.py` (pre-wired formulas reusing the
sheet's own `$G$2`/`$H$2` slopes). `fill_ternarios.py` auto-fills the HBA/HBD **areas**
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

## Repeat audit
`uv run audit_repeats.py` reads `Sistemas ternarios_MF_filled.xlsx` only (no `out/` needed)
and writes `out/repeat_list.csv`: one row per (system, phase) point, ternary **and** binary,
with a `repeat` verdict. Reasons: `missing_vials` (blank L/M/N = never injected),
`low_closure`, `replicate_mismatch` (both via `fill_ternarios.results_rows`, so they match
the pipeline), `aqueous_organics_suspect` (organic droplets in **every** kept aqueous vial: terpenes above
the pair's summed pure-water solubility at 30 °C +20 % method tolerance, `AQ_SOLUBILITY_30C`
× `AQ_SOLUBILITY_TOL` in `aq_terpene_max`, 2.8–4.9 g/L by pair) and `aqueous_replicate_mismatch` (kept aqueous
vials' total organics differ > 3x with no droplet signature; the pipeline's
`replicate_mismatch` is gated on a component > 10 % and never fires on aqueous phases).
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
accepts it). `Sheet2` (the prep book's own binary vial sheet) is hidden as redundant. The 10 g
DES prep table (`datos_des`) is cropped to the DES those tubes use; the 5 g variant
(`Sheet1`) is hidden — every batch is made at 10 g to have spare. The console still
prints the grams of each DES the tubes consume (cached `Lab_DES` estimates; binaries as
4 mL) as a check against that 10 g. Test:
`tests/test_make_repeat_workbook.py` (same no-project invocation).
