"""Batch processing pipeline: files → FileResult list + CSV output."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .detector import DetectionParams, detect_peaks
from .integrator import PeakResult, integrate_all_peaks
from .reader import Chromatogram, read_gcd


@dataclass
class FileResult:
    filepath: Path
    filename: str
    chromatogram: Chromatogram | None
    peaks: list[PeakResult] = field(default_factory=list)
    error: str | None = None


def process_file(filepath: Path | str, params: DetectionParams) -> FileResult:
    """Read, detect peaks, and integrate a single .gcd file."""
    filepath = Path(filepath)
    filename = filepath.name

    try:
        chrom = read_gcd(filepath)
    except ValueError as exc:
        return FileResult(
            filepath=filepath, filename=filename, chromatogram=None, error=str(exc)
        )

    try:
        peak_indices = detect_peaks(chrom, params)
        peaks = integrate_all_peaks(chrom, peak_indices)
    except Exception as exc:
        return FileResult(
            filepath=filepath, filename=filename, chromatogram=chrom, error=str(exc)
        )

    return FileResult(
        filepath=filepath, filename=filename, chromatogram=chrom, peaks=peaks
    )


def process_batch(
    filepaths: list[Path | str], params: DetectionParams
) -> list[FileResult]:
    """Process a list of .gcd files and return one FileResult each.

    Prints a one-line progress message per file to stdout.
    """
    results: list[FileResult] = []
    for fp in filepaths:
        fp = Path(fp)
        result = process_file(fp, params)
        if result.error:
            print(f"  ERROR  {fp.name}: {result.error}")
        else:
            n = len(result.peaks)
            print(f"  OK     {fp.name}: {n} peak{'s' if n != 1 else ''} found")
        results.append(result)
    return results


def save_csv(results: list[FileResult], output: Path) -> None:
    """Write results to a tidy CSV (one row per peak per file).

    Columns: filename, peak_n, tR_min, height_mV, area_mV_min, left_min, right_min

    Files with errors get a single row with empty peak columns.
    """
    output = Path(output)
    with output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "filename",
                "peak_n",
                "tR_min",
                "height_mV",
                "area_mV_min",
                "left_min",
                "right_min",
            ]
        )
        for result in results:
            if result.error or not result.peaks:
                writer.writerow([result.filename, "", "", "", "", "", ""])
            else:
                for peak in result.peaks:
                    writer.writerow(
                        [
                            result.filename,
                            peak.peak_number,
                            f"{peak.time_min:.4f}",
                            f"{peak.height_mV:.2f}",
                            f"{peak.area_mV_min:.4f}",
                            f"{peak.left_min:.4f}",
                            f"{peak.right_min:.4f}",
                        ]
                    )
