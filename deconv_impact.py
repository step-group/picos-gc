"""Quantify how much EMG deconvolution changes peak areas vs perpendicular-drop.

Detection runs once per file; only integration is repeated per split mode, so
peaks line up 1:1 and only their area changes. Writes out/deconv_impact.csv
and prints a summary.

Run: uv run python deconv_impact.py FLECK_TERPENOS2026/BINARIOS_TERPENOS
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from picos_gc.baseline import correct_baseline
from picos_gc.detector import DetectionParams, detect_peaks
from picos_gc.integrator import integrate_all_peaks
from picos_gc.reader import Chromatogram, read_gcd


def peak_area_deltas(
    chrom: Chromatogram, params: DetectionParams, baseline_method: str | None = "arpls"
) -> list[dict]:
    """One dict per peak comparing drop vs deconvolve areas for a chromatogram."""
    if baseline_method is not None:
        from dataclasses import replace

        corrected = correct_baseline(chrom.signal_mV, chrom.time_min, method=baseline_method)
        chrom = replace(chrom, signal_mV=corrected)

    detected = detect_peaks(chrom, params)
    drop = integrate_all_peaks(chrom, detected, split_mode="drop")
    deconv = integrate_all_peaks(chrom, detected, split_mode="deconvolve")

    rows: list[dict] = []
    for d, x in zip(drop, deconv, strict=True):
        denom = d.area_mV_min if abs(d.area_mV_min) > 1e-12 else 1e-12
        rows.append(
            {
                "peak_n": d.peak_number,
                "tR_min": round(d.time_min, 4),
                "drop_area": round(d.area_mV_min, 4),
                "deconv_area": round(x.area_mV_min, 4),
                "pct_change": round(100.0 * (x.area_mV_min - d.area_mV_min) / denom, 2),
            }
        )
    return rows


def _iter_gcd(inputs: list[Path]):
    for p in inputs:
        if p.is_file() and p.suffix == ".gcd":
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.gcd"))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: uv run python deconv_impact.py FILES_OR_DIRS", file=sys.stderr)
        sys.exit(1)

    files = list(_iter_gcd([Path(a) for a in argv]))
    if not files:
        print("no .gcd files found", file=sys.stderr)
        sys.exit(1)

    params = DetectionParams()
    out_rows: list[list] = []
    n_changed = 0
    max_abs = 0.0
    # Above-noise view: relative changes on near-zero peaks blow up (denominator
    # ~0), so also track only peaks whose drop area clears a noise floor — that's
    # the change that matters for quantification.
    noise_floor = 1.0  # mV*min
    n_big = 0
    n_big_changed = 0
    max_big = 0.0
    for fp in files:
        try:
            chrom = read_gcd(fp)
        except ValueError as exc:
            print(f"  SKIP {fp.name}: {exc}")
            continue
        for r in peak_area_deltas(chrom, params):
            out_rows.append(
                [fp.name, r["peak_n"], r["tR_min"], r["drop_area"], r["deconv_area"], r["pct_change"]]
            )
            if abs(r["pct_change"]) > 0.1:
                n_changed += 1
            max_abs = max(max_abs, abs(r["pct_change"]))
            if r["drop_area"] >= noise_floor:
                n_big += 1
                if abs(r["pct_change"]) > 1.0:
                    n_big_changed += 1
                max_big = max(max_big, abs(r["pct_change"]))

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "deconv_impact.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "peak_n", "tR_min", "drop_area", "deconv_area", "pct_change"])
        w.writerows(out_rows)

    print(f"\n{len(files)} files, {len(out_rows)} peaks")
    print(f"peaks changed >0.1%: {n_changed}  (raw largest |change|: {max_abs:.1f}% — near-zero peaks)")
    print(
        f"peaks with area >= {noise_floor:g}: {n_big}  |  of those changed >1%: "
        f"{n_big_changed}  |  largest |change|: {max_big:.2f}%"
    )
    print(f"detail -> {out_path}")


if __name__ == "__main__":
    main()
