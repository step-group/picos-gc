# picos-gc

Automatic multi-peak integration for Shimadzu GC `.gcd` files.

Detects all peaks in a chromatogram, integrates each one with linear baseline
subtraction, and — when processing a batch — aligns peaks across files by
retention time so you get a ready-to-use compound table with areas, mean, std,
and RSD.

## Installation

```bash
git clone <repo>
cd picos-gc
uv sync
```

For plotting support (optional — `matplotlib` is not installed by default):

```bash
uv sync --extra plot
```

## Quick start

```bash
# single file
uv run picos-gc sample.gcd

# a whole folder (one batch per folder; subfolders become their own batches)
uv run picos-gc data/

# several files (produces an aligned summary automatically)
uv run picos-gc *.gcd

# with plots (requires the 'plot' extra)
uv run picos-gc data/ --plot

# relax detection for weak signals
uv run picos-gc data/ --height 20 --prominence 5
```

### Inputs and batches

Each positional argument can be a `.gcd` file or a directory:

- Loose `.gcd` files are grouped into one batch labelled `batch`.
- A directory containing `.gcd` files directly becomes one batch named after the
  directory.
- A directory with no direct `.gcd` files but with subdirectories that contain
  them becomes one batch per subdirectory (one level deep).

## Outputs

Results are written under `<outdir>/<batch-label>/` (default `out/`):

| File | Description |
|---|---|
| `resultados_integracion.csv` | Tidy/long format — one row per peak per file |
| `resultados_integracion_aligned.csv` | Wide format — one row per file, one column-pair per compound + stats footer |
| `<name>_peaks.png` | Chromatogram with shaded peak areas (requires `--plot`) |

### Tidy CSV columns
`sample_name, filename, peak_n, tR_min, height_mV, area_mV_min, left_min, right_min`

### Aligned CSV columns
`sample_name, filename, cmp1_tR_min, cmp1_area_mV_min, cmp2_tR_min, cmp2_area_mV_min, ...`

Footer rows (one value per compound, in the area cell): `median_tR`, `tR_std`,
`mean_area`, `std_area`, `rsd_pct`, `n_detected`.

## CLI reference

```
usage: picos-gc [-h] [--height FLOAT] [--prominence FLOAT]
                [--height-snr FLOAT] [--prominence-snr FLOAT] [--distance INT]
                [--outdir DIR] [--min-width FLOAT] [--smooth-window INT]
                [--smooth-polyorder INT] [--merge-ratio FLOAT]
                [--merge-distance FLOAT] [--clip-frac FLOAT]
                [--split-mode {valley,drop}]
                [--baseline {none,asls,airpls,arpls,snip}] [--baseline-lam FLOAT]
                [--align-tol FLOAT] [--plot]
                FILES_OR_DIRS ...
```

| Option | Default | Description |
|---|---|---|
| `--height` | auto | Minimum peak height. Default: auto = `--height-snr` × the measured noise σ (robust MAD on the baseline-corrected signal). Pass a number to force a fixed mV cut (e.g. `--height 50` for the pre-auto behaviour) |
| `--prominence` | auto | Minimum peak prominence. Default: auto = `--prominence-snr` × noise σ |
| `--height-snr` | 10 | Auto height in noise-σ units (limit-of-quantitation ≈ 10σ). Used only when `--height` is unset |
| `--prominence-snr` | 5 | Auto prominence in noise-σ units. Used only when `--prominence` is unset |
| `--distance` | 50 pts | Minimum separation between peaks |
| `--outdir` | `out` | Base output directory |
| `--min-width` | 0.03 min | Minimum peak width (`0` = off) |
| `--smooth-window` | 0 | Savitzky-Golay window for **detection only** (odd integer; `0` = off) |
| `--smooth-polyorder` | 3 | Savitzky-Golay polynomial order |
| `--merge-ratio` | 0.5 | Merge two adjacent peaks when the baseline-corrected valley between them rises above this fraction of the smaller peak (`0` = off; values toward `1` merge only the worst-resolved pairs) |
| `--merge-distance` | 0 min | Merge adjacent peaks whose retention times are within this distance (`0` = off) |
| `--clip-frac` | 0.001 | Clip each integration window to where the baseline-corrected signal stays above this fraction of the peak height (`0` = off, legacy wide windows) |
| `--split-mode` | `valley` | How to integrate fused peaks: `valley` = per-peak valley-to-valley baseline; `drop` = one baseline per fused group with vertical splits at the valleys (Shimadzu-style, conserves the group total) |
| `--baseline` | `arpls` | Global baseline correction applied before detection (`none` = off; other choices: `asls`, `airpls`, `snip`) |
| `--baseline-lam` | method's own | Penalty strength (`lam`) for the `asls`/`airpls`/`arpls` smoothers; larger = smoother baseline. Ignored by `snip` |
| `--align-tol` | 0.1 min | Retention time tolerance for cross-file alignment (`0` = skip) |
| `--plot` | off | Save a `<name>_peaks.png` per file (requires `matplotlib`) |

## Baseline correction

By default, picos-gc subtracts a global baseline from the whole chromatogram
**before** peak detection, using the [pybaselines](https://github.com/derb12/pybaselines)
arPLS algorithm. This removes column bleed and slow drift that a per-peak linear
baseline cannot follow, so both detection and integration see a flat signal.

- Pass `--baseline none` to disable it, or pick another method (`asls`, `airpls`,
  `snip`). arPLS is robust to noise and large peaks with little tuning.
- `--baseline-lam` tunes the Whittaker smoothing for `asls`/`airpls`/`arpls`
  (larger = smoother). The optimal value scales with sampling density, so adjust
  it if the baseline looks over- or under-corrected. `snip` ignores it.

**Reproducibility:** because correction is on by default, reported areas differ
from runs made before this feature existed. Use `--baseline none` to reproduce
pre-baseline results. The standalone `label_terpenos.py` script calls the
detection/integration functions directly and does **not** apply correction
unless it calls `correct_baseline` itself.

## How it works

1. **Read** — parses the OLE2 binary stream in the `.gcd` file to extract time
   (min), signal (mV), and the operator-entered sample name (`File Property` →
   `smpl_name`).
2. **Smooth (optional)** — when `--smooth-window` > 0, a Savitzky-Golay filter is
   applied to a *copy* of the signal for detection only; integration always uses
   the raw signal, so smoothing never changes reported areas. Off by default.
3. **Detect** — `scipy.signal.find_peaks` with height, prominence, distance, and
   width thresholds. By default height/prominence are **auto**: the robust noise σ
   (Gaussian-scaled MAD of the baseline-corrected signal, `estimate_noise`) is
   measured per file and the thresholds are set to `10σ` / `5σ`, so detection adapts
   to each run's noise instead of a fixed mV constant. Falls back to looser fixed
   thresholds if nothing is found.
4. **Bound** — `peak_widths` at `rel_height=1.0` places each integration boundary
   at the prominence reference level (the local valley floor), with a
   valley-clipping safety net for any boundaries that still overlap.
5. **Merge (optional)** — a *baseline-aware* shoulder merge combines two adjacent
   peaks when the valley between them, measured above a baseline drawn across the
   pair's outer bases, stays above `--merge-ratio` of the smaller peak. A separate
   proximity merge (`--merge-distance`) combines peaks that are simply close in
   retention time. Both are configurable; the shoulder merge is on by default.
6. **Clip** — each integration window is shrunk to the contiguous region around
   the apex where the baseline-corrected signal stays above `--clip-frac` of the
   peak height. Prominence-base boundaries can otherwise sit minutes away from
   the peak, and the zero-clamped integral accumulates baseline noise and drift
   over the slack (observed up to +307% on small peaks in real data). Boundaries
   shared by genuinely fused peaks stay at their valley.
7. **Integrate** — for each peak, draws a linear baseline between its boundaries,
   subtracts it, clamps negatives to zero, and integrates with the trapezoidal
   rule (on the global-baseline-corrected signal unless `--baseline none`; never
   on the smoothed copy). With `--split-mode drop`, fused groups instead get
   one baseline across the whole group and are split vertically at the valleys,
   which conserves the group's total area.
8. **Align** — clusters all detected peaks across files by retention time
   proximity, assigns a compound ID to each cluster, and matches each file's peaks
   back to it.

## Development

```bash
uv sync                 # install runtime + dev dependencies
uv run pytest           # run the test suite
uv run ruff check .     # lint
uv run mypy             # type-check
```
