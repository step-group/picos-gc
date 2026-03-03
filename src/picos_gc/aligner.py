"""Cross-file peak alignment by retention time clustering."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .processor import FileResult


@dataclass
class Compound:
    compound_id: int           # 1-based, sorted by median tR
    median_tR: float           # representative retention time
    tR_std: float              # std of tR across files (drift indicator)
    mean_area: float
    std_area: float
    rsd_pct: float             # relative std dev of area (%)
    n_detected: int            # how many files detected this compound


@dataclass
class AlignmentResult:
    compounds: list[Compound]
    # per-file data: index in results -> list of (tR | None, area | None) per compound
    table: dict[int, list[tuple[float | None, float | None]]]


def align_peaks(results: list[FileResult], tol_min: float = 0.1) -> AlignmentResult:
    """Cluster peaks across files by retention time proximity.

    Algorithm:
      1. Collect every (tR, area, filename) triple from all successful results.
      2. Sort by tR and split into clusters wherever the gap between consecutive
         tR values exceeds `tol_min`. This is single-linkage clustering — simple,
         fast, and explainable.
      3. Assign a compound ID (1-based, left to right in time) to each cluster.
      4. For each file × compound, pick the peak whose tR is closest to the
         cluster median (there should normally be at most one per file).

    Args:
        results:  output of process_batch
        tol_min:  max gap in minutes between two tR values to be considered the
                  same compound (default 0.1 min = 6 s)

    Returns:
        AlignmentResult with per-compound stats and a per-file lookup table.
    """
    # Collect all peaks with file attribution
    all_peaks: list[tuple[float, float, str]] = []  # (tR, area, filename)
    for r in results:
        if not r.error:
            for p in r.peaks:
                all_peaks.append((p.time_min, p.area_mV_min, r.filename))

    if not all_peaks:
        return AlignmentResult(compounds=[], table={i: [] for i in range(len(results))})

    all_peaks.sort(key=lambda x: x[0])

    # Split into clusters on gaps > tol_min
    clusters: list[list[tuple[float, float, str]]] = [[all_peaks[0]]]
    for entry in all_peaks[1:]:
        if entry[0] - clusters[-1][-1][0] <= tol_min:
            clusters[-1].append(entry)
        else:
            clusters.append([entry])

    # Build Compound summaries
    compounds: list[Compound] = []
    for cid, cluster in enumerate(clusters, start=1):
        trs = [e[0] for e in cluster]
        areas = [e[1] for e in cluster]
        median_tR = statistics.median(trs)
        tR_std = statistics.stdev(trs) if len(trs) > 1 else 0.0
        mean_area = statistics.mean(areas)
        std_area = statistics.stdev(areas) if len(areas) > 1 else 0.0
        rsd = (std_area / mean_area * 100.0) if mean_area > 0 else 0.0
        compounds.append(
            Compound(
                compound_id=cid,
                median_tR=median_tR,
                tR_std=tR_std,
                mean_area=mean_area,
                std_area=std_area,
                rsd_pct=rsd,
                n_detected=len(cluster),
            )
        )

    # Build per-file lookup keyed by result index (not filename, which may collide).
    # Match by tR range of each cluster [min_tR - tol, max_tR + tol] so that a
    # peak included in a cluster is always matched back to it regardless of drift.
    cluster_ranges = [
        (min(e[0] for e in cl) - tol_min, max(e[0] for e in cl) + tol_min)
        for cl in clusters
    ]
    table: dict[int, list[tuple[float | None, float | None]]] = {}
    for i, r in enumerate(results):
        row: list[tuple[float | None, float | None]] = []
        for (lo, hi), compound in zip(cluster_ranges, compounds):
            best = None
            best_dist = float("inf")
            for p in r.peaks:
                if lo <= p.time_min <= hi:
                    dist = abs(p.time_min - compound.median_tR)
                    if dist < best_dist:
                        best_dist = dist
                        best = (p.time_min, p.area_mV_min)
            row.append(best if best is not None else (None, None))
        table[i] = row

    return AlignmentResult(compounds=compounds, table=table)


def save_aligned_csv(alignment: AlignmentResult, results: list[FileResult], output: Path) -> None:
    """Write a wide-format CSV: one row per file, one column-pair per compound.

    Header:
        filename, cmp1_tR_min, cmp1_area_mV_min, cmp2_tR_min, cmp2_area_mV_min, ...

    Footer rows (after a blank line):
        median_tR, mean_area, std_area, rsd_pct per compound
    """
    output = Path(output)
    n = len(alignment.compounds)

    header = ["filename"]
    for c in alignment.compounds:
        header += [f"cmp{c.compound_id}_tR_min", f"cmp{c.compound_id}_area_mV_min"]

    with output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for i, r in enumerate(results):
            row_vals = alignment.table.get(i, [(None, None)] * n)
            row = [r.filename]
            for tR, area in row_vals:
                row.append(f"{tR:.4f}" if tR is not None else "")
                row.append(f"{area:.4f}" if area is not None else "")
            writer.writerow(row)

        # Summary footer
        writer.writerow([])
        for label, getter in [
            ("median_tR",  lambda c: f"{c.median_tR:.4f}"),
            ("tR_std",     lambda c: f"{c.tR_std:.4f}"),
            ("mean_area",  lambda c: f"{c.mean_area:.4f}"),
            ("std_area",   lambda c: f"{c.std_area:.4f}"),
            ("rsd_pct",    lambda c: f"{c.rsd_pct:.2f}"),
            ("n_detected", lambda c: str(c.n_detected)),
        ]:
            row = [label]
            for c in alignment.compounds:
                row += ["", getter(c)]  # blank tR cell, value in area cell
            writer.writerow(row)
