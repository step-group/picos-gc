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
