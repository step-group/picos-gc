"""Command-line interface for picos-gc."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aligner import align_peaks, save_aligned_csv
from .detector import DetectionParams
from .processor import FileResult, process_batch, save_csv


def _print_summary(results: list[FileResult]) -> None:
    print("\n" + "=" * 75)
    print("RESUMEN DE PEAKS")
    print("=" * 75)
    header = f"{'Archivo':<22} {'Peak':>4}  {'tR(min)':>8}  {'Altura(mV)':>10}  {'Area(mV*min)':>13}"
    print(header)
    print("-" * 75)

    for result in results:
        if result.error:
            print(f"{result.filename:<22}  ERROR: {result.error}")
        elif not result.peaks:
            print(f"{result.filename:<22}  (no peaks found)")
        else:
            for peak in result.peaks:
                print(
                    f"{result.filename:<22} {peak.peak_number:>4}  "
                    f"{peak.time_min:>8.3f}  {peak.height_mV:>10.2f}  {peak.area_mV_min:>13.4f}"
                )

    print("=" * 75)


def _print_alignment_summary(alignment) -> None:
    from .aligner import AlignmentResult

    if not alignment.compounds:
        print("\n(no compounds detected for alignment)")
        return

    print("\n" + "=" * 75)
    print("COMPOUND ALIGNMENT SUMMARY")
    print("=" * 75)
    header = f"{'Compound':>8}  {'median tR':>10}  {'tR std':>8}  {'mean area':>12}  {'std area':>12}  {'RSD%':>6}  {'n':>4}"
    print(header)
    print("-" * 75)
    for c in alignment.compounds:
        print(
            f"{c.compound_id:>8}  {c.median_tR:>10.3f}  {c.tR_std:>8.4f}  "
            f"{c.mean_area:>12.2f}  {c.std_area:>12.2f}  {c.rsd_pct:>6.1f}  {c.n_detected:>4}"
        )
    print("=" * 75)


def _plot_file(result: FileResult, output_dir: Path) -> None:
    """Save a PNG with shaded peak areas for a single file result."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  matplotlib is not installed. Install it with: uv add matplotlib  "
            "(or pip install matplotlib)"
        )
        return

    if result.chromatogram is None or result.error:
        return

    chrom = result.chromatogram
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(chrom.time_min, chrom.signal_mV, color="steelblue", linewidth=0.8, label="signal")

    for peak in result.peaks:
        # Find index bounds
        left_idx = int((peak.left_min - chrom.time_min[0]) / (chrom.time_min[-1] - chrom.time_min[0]) * (len(chrom.time_min) - 1))
        right_idx = int((peak.right_min - chrom.time_min[0]) / (chrom.time_min[-1] - chrom.time_min[0]) * (len(chrom.time_min) - 1))
        left_idx = max(0, min(left_idx, len(chrom.time_min) - 1))
        right_idx = max(0, min(right_idx, len(chrom.time_min) - 1))

        t_seg = chrom.time_min[left_idx : right_idx + 1]
        s_seg = chrom.signal_mV[left_idx : right_idx + 1]
        ax.fill_between(t_seg, s_seg, alpha=0.35, label=f"Peak {peak.peak_number}")
        ax.annotate(
            f"P{peak.peak_number}\n{peak.time_min:.2f} min\n{peak.area_mV_min:.3f} mV·min",
            xy=(peak.time_min, peak.height_mV),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Signal (mV)")
    ax.set_title(result.filename)
    ax.legend(fontsize=7)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
    fig.tight_layout()

    stem = Path(result.filename).stem
    out_path = output_dir / f"{stem}_peaks.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main(argv: list[str] | None = None) -> None:
    _d = DetectionParams()  # single source of truth for defaults

    parser = argparse.ArgumentParser(
        prog="picos-gc",
        description="Integrate all peaks in Shimadzu .gcd chromatogram files.",
    )
    parser.add_argument("files", metavar="FILES", nargs="+", help="one or more .gcd files")
    parser.add_argument(
        "--height",
        metavar="FLOAT",
        type=float,
        default=_d.min_height,
        help=f"min peak height in mV (default: {_d.min_height})",
    )
    parser.add_argument(
        "--prominence",
        metavar="FLOAT",
        type=float,
        default=_d.min_prominence,
        help=f"min peak prominence in mV (default: {_d.min_prominence})",
    )
    parser.add_argument(
        "--distance",
        metavar="INT",
        type=int,
        default=_d.min_distance,
        help=f"min distance between peaks in data points (default: {_d.min_distance})",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        default=Path("out/resultados_integracion.csv"),
        help="CSV output path (default: out/resultados_integracion.csv)",
    )
    parser.add_argument(
        "--smooth-window",
        metavar="INT",
        type=int,
        default=_d.smooth_window,
        help=f"Savitzky-Golay window for detection smoothing, odd integer (default: {_d.smooth_window}; 0 = off)",
    )
    parser.add_argument(
        "--smooth-polyorder",
        metavar="INT",
        type=int,
        default=_d.smooth_polyorder,
        help=f"Savitzky-Golay polynomial order (default: {_d.smooth_polyorder})",
    )
    parser.add_argument(
        "--align-tol",
        metavar="FLOAT",
        type=float,
        default=0.1,
        help="retention time tolerance (min) for cross-file peak alignment (default: 0.1; 0 = skip)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="save a <name>_peaks.png per file (requires matplotlib)",
    )

    args = parser.parse_args(argv)

    filepaths = [Path(f) for f in args.files]
    missing = [f for f in filepaths if not f.exists()]
    if missing:
        for m in missing:
            print(f"ERROR: file not found: {m}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    params = DetectionParams(
        min_height=args.height,
        min_prominence=args.prominence,
        min_distance=args.distance,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
    )

    smooth_label = f"SG(window={params.smooth_window}, poly={params.smooth_polyorder})" if params.smooth_window > 0 else "off"

    print("=" * 75)
    print("INTEGRACION DE PEAKS - GC Shimadzu (.gcd)")
    print("=" * 75)
    print(f"Files     : {len(filepaths)}")
    print(f"Height    : {params.min_height} mV")
    print(f"Prominence: {params.min_prominence} mV")
    print(f"Distance  : {params.min_distance} pts")
    print(f"Smoothing : {smooth_label}")
    print()

    results = process_batch(filepaths, params)

    _print_summary(results)

    save_csv(results, args.output)
    print(f"\nResults saved to: {args.output}")

    if args.align_tol > 0 and len(results) > 1:
        alignment = align_peaks(results, tol_min=args.align_tol)
        aligned_path = args.output.with_stem(args.output.stem + "_aligned")
        save_aligned_csv(alignment, results, aligned_path)
        print(f"Alignment saved to: {aligned_path}")
        _print_alignment_summary(alignment)

    if args.plot:
        plot_dir = args.output.parent
        print("\nGenerating plots...")
        for result in results:
            _plot_file(result, plot_dir)

    print("Done.")


if __name__ == "__main__":
    main()
