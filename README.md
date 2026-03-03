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

For plotting support:

```bash
uv sync --extra plot
```

## Quick start

```bash
# single file
uv run picos-gc sample.gcd

# batch (produces aligned summary automatically)
uv run picos-gc *.gcd

# batch with plots
uv run picos-gc *.gcd --plot

# relax detection for weak signals
uv run picos-gc *.gcd --height 100 --prominence 10
```

## Outputs

| File | Description |
|---|---|
| `resultados_integracion.csv` | Tidy/long format — one row per peak per file |
| `resultados_integracion_aligned.csv` | Wide format — one row per file, one column-pair per compound + stats footer |
| `<name>_peaks.png` | Chromatogram with shaded peak areas (requires `--plot`) |

### Tidy CSV columns
`filename, peak_n, tR_min, height_mV, area_mV_min, left_min, right_min`

### Aligned CSV columns
`filename, cmp1_tR_min, cmp1_area_mV_min, cmp2_tR_min, cmp2_area_mV_min, ...`

Footer rows: `median_tR`, `tR_std`, `mean_area`, `std_area`, `rsd_pct`, `n_detected` per compound.

## CLI reference

```
usage: picos-gc [-h] [--height FLOAT] [--prominence FLOAT] [--distance INT]
                [--smooth-window INT] [--smooth-polyorder INT]
                [--align-tol FLOAT] [--output PATH] [--plot]
                FILES ...
```

| Option | Default | Description |
|---|---|---|
| `--height` | 500 mV | Minimum peak height for detection |
| `--prominence` | 20 mV | Minimum peak prominence |
| `--distance` | 50 pts | Minimum separation between peaks |
| `--smooth-window` | 11 | Savitzky-Golay window size (odd integer; `0` = off) |
| `--smooth-polyorder` | 3 | Savitzky-Golay polynomial order |
| `--align-tol` | 0.1 min | Retention time tolerance for cross-file alignment (`0` = skip) |
| `--output` | `resultados_integracion.csv` | Output CSV path |
| `--plot` | off | Save a `<name>_peaks.png` per file |

## How it works

1. **Read** — parses the OLE2 binary stream in the `.gcd` file to extract time (min) and signal (mV)
2. **Smooth** — applies a Savitzky-Golay filter to a copy of the signal (detection only; integration always uses the raw signal)
3. **Detect** — `scipy.signal.find_peaks` with height, prominence, and distance thresholds; falls back to looser thresholds if nothing is found
4. **Integrate** — for each peak, draws a linear baseline between the valley boundaries returned by scipy's prominence algorithm and integrates the corrected area with the trapezoidal rule
5. **Align** — clusters all detected peaks across files by retention time proximity; assigns a compound ID to each cluster and matches each file's peaks back to it
