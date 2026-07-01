"""One-off: integrate, name, and MERGE-duplicates for FLECK_TERPENOS2026 (batches A-I).

Each sample (DB-WAX UI / polar PEG column) contains IPA (solvent, ignored), 2PE
(2-phenylethanol, ~9.75 min) and two named terpenes (different per batch, see BATCH_TERPENES).

Two labelling dimensions:
  * COMPOUND of each peak  -> IPA / terpene1 / terpene2 / 2PE, named per batch.
  * SAMPLE identity of each injection (e.g. B1-T1, E3_B2) -> read from the Shimadzu .gcb
    batch table; batches B, C, E, F injected every sample in DUPLICATE (rep _1/_2 or -1/-2),
    which are averaged here (mean + %RSD). Batches A, D, H, I are single injection.

BINARIOS_TERPENOS is handled separately (different method / retention times).

Run from the repo root:  uv run python label_terpenos.py
"""

from __future__ import annotations

import csv
import dataclasses
import re
import shutil
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from picos_gc.baseline import ALLOWED_METHODS, correct_baseline
from picos_gc.detector import DetectionParams, detect_peaks, estimate_noise
from picos_gc.integrator import PeakResult, integrate_all_peaks
from picos_gc.reader import read_gcd, time_to_index

# --- configuration ----------------------------------------------------------
DATA_DIR = Path("FLECK_TERPENOS2026")
OUT_DIR = Path("out")
PARAMS = DetectionParams()
BASELINE_METHOD = "arpls"  # global drift removal (library default); None disables


def load_chrom(fp: Path):
    """Read a .gcd and apply global baseline correction (matches the picos-gc CLI).

    The integrator's per-peak chord only sees each peak's own span; this removes
    the slow whole-run drift so the plotted trace is flat and sub-mV drift
    artifacts don't register as peaks. Analyte areas move <0.2% (chord already
    handles local drift), so it does not affect the mass-fraction results.
    """
    ch = read_gcd(fp)
    if BASELINE_METHOD:
        corr = correct_baseline(ch.signal_mV, ch.time_min, method=BASELINE_METHOD)
        ch = dataclasses.replace(ch, signal_mV=corr)
    return ch


def auto_height(chrom):
    """The height threshold detect_peaks resolves for this chrom (for plots/report)."""
    if PARAMS.min_height is not None:
        return PARAMS.min_height
    sigma = estimate_noise(chrom.signal_mV)
    return PARAMS.noise_snr_height * sigma if sigma > 0 else None


IPA_WINDOW = (4.4, 5.3)
TPE_WINDOW = (9.55, 10.05)
DETECT_HEIGHT_MIN = 300.0
CLUSTER_GAP = 0.12
TERP_TOL = 0.15

# two terpenes per batch, ordered (earlier-eluting, later-eluting)
BATCH_TERPENES: dict[str, tuple[str, str]] = {
    "A1T1 AL A5B2": ("L-Carvone", "Thymol"),
    "B1T1 AL B5B2": ("Geraniol", "Thymol"),
    "C1T1 AL C5B2": ("Thymol", "Carvacrol"),
    "D1T1 AL D5B2": ("Thymol", "Eugenol"),
    "E1T1 AL E5B2": ("DL-Camphor", "Thymol"),
    "F1T1 AL F5B2": ("DL-Camphor", "L-Carvone"),
    "H1T1 AL H5B2": ("DL-Camphor", "Carvacrol"),
    "I1T1 AL I5B2": ("DL-Camphor", "Eugenol"),
}
CANONICAL_TR = {
    "IPA": 4.85,
    "DL-Camphor": 7.96,
    "L-Carvone": 8.94,
    "Geraniol": 9.17,
    "2PE": 9.75,
    "Thymol": 11.73,
    "Eugenol": 11.93,
    "Carvacrol": 12.15,
}
COMPOUND_COLORS = {
    "IPA": "0.6",
    "DL-Camphor": "tab:green",
    "L-Carvone": "tab:olive",
    "Geraniol": "tab:cyan",
    "2PE": "tab:red",
    "Thymol": "tab:purple",
    "Eugenol": "tab:orange",
    "Carvacrol": "tab:brown",
    "other": "tab:blue",
}

# BINARIOS_TERPENOS: same elution order as A-I but a different (stretched) method,
# so its own retention-time map (confirmed with the user). Binary mixtures, NO 2PE.
BIN_IPA_WINDOW = (4.7, 5.3)
BIN_TERP_CENTERS = {
    "DL-Camphor": 8.55,
    "L-Carvone": 9.15,
    "Geraniol": 9.90,
    "Thymol": 13.65,
    "Eugenol": 14.15,
    "Carvacrol": 14.32,
}
BIN_TOL = 0.25  # absorbs the 9.8/10.0 Geraniol drift; Eugenol/Carvacrol split by nearest-center
BIN_TERPENES = list(BIN_TERP_CENTERS)

# a sample name carrying a replicate suffix -> capture the sample key without the rep
REP = re.compile(r"(.+[TB][12])[-_][12]$")


def sample_key(name: str) -> str:
    m = REP.match(name)
    return m.group(1) if m else name


def sample_label(chrom) -> str:
    """Embedded Shimadzu sample name, or 'BLANK' for blank/wash injections.

    Blanks are named 'BNK...'; downstream code drops 'BLANK' from the merged
    and Excel outputs. ponytail: broaden the blank test if other blank prefixes
    appear.
    """
    nm = chrom.sample_name
    return "BLANK" if not nm or nm.upper().startswith("BNK") else nm


def in_window(tr: float, w: tuple[float, float]) -> bool:
    return w[0] <= tr <= w[1]


def force_window_area(chrom, win: tuple[float, float]) -> float:
    """Integrate the signal inside *win* with a chord baseline (no peak required).

    Used by --trace to put a measured value on a compound that fell below the
    detection threshold: for a genuinely empty window this returns the rectified
    noise (~0.0X, an upper bound / LOD-level estimate), not a real peak.
    """
    m = (chrom.time_min >= win[0]) & (chrom.time_min <= win[1])
    t, s = chrom.time_min[m], chrom.signal_mV[m]
    if len(t) < 3:
        return 0.0
    base = np.linspace(s[0], s[-1], len(s))  # chord between the window edges
    return float(np.trapezoid(np.clip(s - base, 0.0, None), t))


# --- peak / compound handling (unchanged model) -----------------------------
def cluster_tr(points):
    pts = sorted(points, key=lambda p: p[0])
    clusters, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[-1][0] <= CLUSTER_GAP:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    return clusters


def derive_terpene_windows(all_peaks, names):
    pts = [
        (p.time_min, p.area_mV_min)
        for p in all_peaks
        if p.height_mV >= DETECT_HEIGHT_MIN
        and not in_window(p.time_min, IPA_WINDOW)
        and not in_window(p.time_min, TPE_WINDOW)
    ]
    if not pts:
        return {}
    report = [
        {"median_tR": statistics.median([t for t, _ in c]), "total_area": sum(a for _, a in c)}
        for c in cluster_tr(pts)
    ]
    report.sort(key=lambda r: r["total_area"], reverse=True)
    top2 = sorted(report[:2], key=lambda r: r["median_tR"])
    windows = {}
    for cluster, nm in zip(top2, names):
        c = cluster["median_tR"]
        windows[nm] = (c - TERP_TOL, c + TERP_TOL)
    return windows


def classify(peaks, terp_windows):
    centers = {nm: 0.5 * (w[0] + w[1]) for nm, w in terp_windows.items()}
    claims, leftovers = {}, []
    for p in peaks:
        if in_window(p.time_min, IPA_WINDOW):
            claims.setdefault("IPA", []).append(p)
        elif in_window(p.time_min, TPE_WINDOW):
            claims.setdefault("2PE", []).append(p)
        else:
            best, bd = None, TERP_TOL + 1.0
            for nm, c in centers.items():
                d = abs(p.time_min - c)
                if d <= TERP_TOL and d < bd:
                    best, bd = nm, d
            if best:
                claims.setdefault(best, []).append(p)
            else:
                leftovers.append(p)
    out = []
    for nm, plist in claims.items():
        plist.sort(key=lambda p: p.area_mV_min, reverse=True)
        out.append((nm, plist[0]))
        out.extend(("other", p) for p in plist[1:])
    out.extend(("other", p) for p in leftovers)
    out.sort(key=lambda ap: ap[1].time_min)
    return out


def plot_labeled(chrom, assignments, title, out_path, threshold=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(chrom.time_min, chrom.signal_mV, color="black", linewidth=0.7)
    if threshold and threshold > 0:  # detection floor: peaks below this are noise
        ax.axhline(
            threshold,
            color="tab:red",
            linewidth=0.6,
            linestyle="--",
            alpha=0.6,
            label=f"detect floor {threshold:.3f} mV",
        )
    seen = set()
    for i, (name, pk) in enumerate(assignments):
        li, ri = time_to_index(chrom, pk.left_min), time_to_index(chrom, pk.right_min)
        color = COMPOUND_COLORS.get(name, "tab:blue")
        ax.fill_between(
            chrom.time_min[li : ri + 1],
            chrom.signal_mV[li : ri + 1],
            alpha=0.4,
            color=color,
            label=name if name not in seen else None,
        )
        seen.add(name)
        tag = "IPA\n(solvent)" if name == "IPA" else name
        ax.annotate(
            f"{tag}\n{pk.time_min:.2f}\n{pk.area_mV_min:.1f}",
            xy=(pk.time_min, pk.height_mV),
            xytext=(0, 6 + 14 * (i % 2)),  # stagger to avoid label collisions on close peaks
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="0.4" if name == "IPA" else color,
        )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Signal (mV)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    # symlog so tiny terpene analytes (~5 mV) stay visible next to IPA/2PE (~1e4 mV).
    # ponytail: linthresh=1 keeps the ~0.2 mV baseline noise in the flat linear zone.
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(bottom=-1.0, top=ax.get_ylim()[1] * 3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- BINARIOS (binary terpene mixtures: same elution order, no 2PE) ---------
def classify_bin(peaks):
    """Label BINARIOS peaks; keep only the 2 largest terpenes (binary), rest -> other."""
    claims, leftovers = {}, []
    for p in peaks:
        if in_window(p.time_min, BIN_IPA_WINDOW):
            claims.setdefault("IPA", []).append(p)
            continue
        best, bd = None, BIN_TOL + 1.0
        for nm, c in BIN_TERP_CENTERS.items():
            d = abs(p.time_min - c)
            if d <= BIN_TOL and d < bd:
                best, bd = nm, d
        if best:
            claims.setdefault(best, []).append(p)
        else:
            leftovers.append(p)
    chosen = {}
    for nm, plist in claims.items():
        plist.sort(key=lambda p: p.area_mV_min, reverse=True)
        chosen[nm] = plist[0]
        leftovers.extend(plist[1:])
    terps = sorted(
        ((nm, pk) for nm, pk in chosen.items() if nm != "IPA"),
        key=lambda x: x[1].area_mV_min,
        reverse=True,
    )
    assignments = []
    if "IPA" in chosen:
        assignments.append(("IPA", chosen["IPA"]))
    assignments += terps[:2]  # the two real terpenes
    leftovers += [pk for _, pk in terps[2:]]  # demote the 3rd (impurity) -> other
    assignments += [("other", pk) for pk in leftovers]
    assignments.sort(key=lambda ap: ap[1].time_min)
    return assignments


def process_binarios(bd: Path):
    out_dir = OUT_DIR / bd.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_file = []
    for fp in sorted(bd.glob("*.gcd")):
        try:
            ch = load_chrom(fp)
            pk = integrate_all_peaks(ch, detect_peaks(ch, PARAMS))
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {fp.name}: {exc}")
            continue
        per_file.append((fp, ch, pk))
    names = {fp: sample_label(ch) for fp, ch, _ in per_file}

    tidy, sample_rows = [], []
    for fp, ch, pk in per_file:
        sname = names.get(fp, "BLANK")
        assigns = classify_bin(pk)
        for nm, p in assigns:
            tidy.append(
                [
                    sname,
                    fp.name,
                    nm,
                    f"{p.time_min:.4f}",
                    f"{p.height_mV:.2f}",
                    f"{p.area_mV_min:.4f}",
                ]
            )
        plot_labeled(
            ch,
            assigns,
            f"{sname}  [{fp.name}]",
            out_dir / f"{safe(sname) if sname != 'BLANK' else 'BLANK_' + fp.stem}_labeled.png",
            threshold=auto_height(ch),
        )
        if sname == "BLANK":
            continue
        by = {nm: p for nm, p in assigns if nm != "other"}
        row = [sname, f"{by['IPA'].area_mV_min:.2f}" if "IPA" in by else ""]
        for t in BIN_TERPENES:
            row.append(f"{by[t].area_mV_min:.2f}" if t in by else "")
        sample_rows.append(row)

    with (out_dir / "peaks_by_injection.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "file", "compound", "tR_min", "height_mV", "area_mV_min"])
        w.writerows(tidy)
    with (out_dir / "samples.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "IPA_area"] + [f"{t}_area" for t in BIN_TERPENES])
        w.writerows(sample_rows)
    print(
        f"{bd.name}: binary mixtures, single injection | {len(per_file)} files "
        f"-> {len(sample_rows)} samples (no 2PE)"
    )


# --- main -------------------------------------------------------------------
def safe(s: str) -> str:
    return re.sub(r"[^\w.-]", "_", s)


def main(argv=None) -> None:
    global PARAMS, BASELINE_METHOD
    import argparse

    ap = argparse.ArgumentParser(description="Integrate, name, and merge FLECK_TERPENOS2026.")
    ap.add_argument(
        "--height",
        type=float,
        default=PARAMS.min_height,
        help=f"min peak height mV (default: auto = {PARAMS.noise_snr_height:g}sigma over noise; "
        "pass a number to force a fixed cut)",
    )
    ap.add_argument(
        "--prominence",
        type=float,
        default=PARAMS.min_prominence,
        help=f"min peak prominence mV (default: auto = {PARAMS.noise_snr_prominence:g}sigma over noise)",
    )
    ap.add_argument(
        "--height-snr",
        type=float,
        default=PARAMS.noise_snr_height,
        help=f"auto height in noise-sigma units when --height is unset (default {PARAMS.noise_snr_height:g})",
    )
    ap.add_argument(
        "--prominence-snr",
        type=float,
        default=PARAMS.noise_snr_prominence,
        help=f"auto prominence in noise-sigma units when --prominence is unset "
        f"(default {PARAMS.noise_snr_prominence:g})",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="comma-separated batch letters to (re)process, e.g. B,D (default: all)",
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        help="for an undetected compound, integrate its retention window anyway "
        "(below-threshold trace value, ~noise floor) instead of reporting 0",
    )
    ap.add_argument(
        "--baseline",
        default=BASELINE_METHOD,
        choices=[*ALLOWED_METHODS, "none"],
        help=f"global baseline correction before integration (default {BASELINE_METHOD}; "
        "'none' disables, using the per-peak chord only)",
    )
    args = ap.parse_args(argv)
    BASELINE_METHOD = None if args.baseline == "none" else args.baseline

    PARAMS = DetectionParams(
        min_height=args.height,
        min_prominence=args.prominence,
        noise_snr_height=args.height_snr,
        noise_snr_prominence=args.prominence_snr,
    )
    only = {s.strip().upper() for s in args.only.split(",")} if args.only else None

    batches = [
        sub
        for sub in sorted(DATA_DIR.iterdir())
        if sub.is_dir()
        and re.match(r"[A-I]1T1 AL ", sub.name)
        and (only is None or sub.name[0] in only)
    ]
    print(
        f"Processing {len(batches)} batches {[b.name[0] for b in batches]} "
        f"(height={args.height}, prominence={args.prominence})\n"
    )
    combined: list[list] = []

    for bd in batches:
        label = bd.name
        out_dir = OUT_DIR / label
        if out_dir.exists():
            shutil.rmtree(out_dir)  # clean re-run
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(bd.glob("*.gcd"))

        # pass 1: integrate every file
        per_file = []
        for fp in files:
            try:
                ch = load_chrom(fp)
                pk = integrate_all_peaks(ch, detect_peaks(ch, PARAMS))
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {fp.name}: {exc}")
                continue
            per_file.append((fp, ch, pk))
        if PARAMS.min_height is None and per_file:  # report the data-driven floor
            med = statistics.median(estimate_noise(ch.signal_mV) for _, ch, _ in per_file)
            print(
                f"  {label[0]}: noise sigma (median) = {med:.4f} mV  ->  auto height "
                f"≈ {PARAMS.noise_snr_height * med:.3f}, prom ≈ {PARAMS.noise_snr_prominence * med:.3f} mV"
            )

        names = {fp: sample_label(ch) for fp, ch, _ in per_file}
        terp_windows = derive_terpene_windows(
            [p for _, _, pk in per_file for p in pk], BATCH_TERPENES[label]
        )
        terp_names = list(BATCH_TERPENES[label])
        compounds = [*terp_names, "2PE"]
        is_dup = any(REP.match(n) for n in names.values())

        # pass 2: classify + plot, collect per-injection compound areas
        per_sample: dict[str, list[dict]] = {}
        tidy = []
        n_trace = 0
        for fp, ch, pk in per_file:
            sname = names.get(fp, "BLANK")
            assigns = classify(pk, terp_windows)
            by = {nm: p for nm, p in assigns if nm != "other"}
            for nm, p in assigns:
                tidy.append(
                    [
                        sname,
                        fp.name,
                        nm,
                        f"{p.time_min:.4f}",
                        f"{p.height_mV:.2f}",
                        f"{p.area_mV_min:.4f}",
                    ]
                )
            plot_labeled(
                ch,
                assigns,
                f"{sname}  [{fp.name}]",
                out_dir / f"{safe(sname) if sname != 'BLANK' else 'BLANK_' + fp.stem}_labeled.png",
                threshold=auto_height(ch),
            )
            if sname == "BLANK":
                continue
            rec = {"IPA": by["IPA"].area_mV_min if "IPA" in by else 0.0}
            for cmp in compounds:
                if cmp in by:
                    rec[cmp] = by[cmp].area_mV_min
                    rec[cmp + "_tR"] = by[cmp].time_min
                else:
                    win = terp_windows.get(cmp) or (TPE_WINDOW if cmp == "2PE" else None)
                    if args.trace and win is not None:
                        rec[cmp] = force_window_area(ch, win)  # below-threshold trace value
                        rec[cmp + "_tR"] = 0.5 * (win[0] + win[1])
                        n_trace += 1
                    else:
                        rec[cmp] = 0.0
                        rec[cmp + "_tR"] = None
            per_sample.setdefault(sample_key(sname), []).append(rec)

        # merge replicates -> mean + RSD
        merged_rows = []
        flags = []
        for key in sorted(per_sample):
            recs = per_sample[key]
            row = [key, len(recs)]
            means = {}
            for cmp in compounds:
                areas = [r[cmp] for r in recs]
                trs = [r[cmp + "_tR"] for r in recs if r[cmp + "_tR"] is not None]
                mean = statistics.mean(areas)
                rsd = (
                    (statistics.pstdev(areas) / mean * 100) if mean > 0 and len(areas) > 1 else 0.0
                )
                tr = statistics.mean(trs) if trs else None
                means[cmp] = mean
                row += [f"{tr:.3f}" if tr is not None else "", f"{mean:.4f}", f"{rsd:.1f}"]
                if len(recs) > 1 and rsd > 25 and mean > 50:
                    flags.append(f"{key}/{cmp} RSD={rsd:.0f}%")
            merged_rows.append(row)
            combined.append(
                [
                    label,
                    key,
                    len(recs),
                    f"{means[terp_names[0]]:.2f}",
                    f"{means[terp_names[1]]:.2f}",
                    f"{means['2PE']:.2f}",
                ]
            )

        # write merged CSV
        hdr = ["sample", "n_rep"]
        for cmp in compounds:
            hdr += [f"{cmp}_tR", f"{cmp}_area_mean", f"{cmp}_area_RSD%"]
        with (out_dir / "merged_samples.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            w.writerows(merged_rows)
        with (out_dir / "peaks_by_injection.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sample", "file", "compound", "tR_min", "height_mV", "area_mV_min"])
            w.writerows(tidy)

        kind = "DUPLICATE (merged reps)" if is_dup else "single injection"
        trace_note = f" | {n_trace} below-threshold trace fills" if n_trace else ""
        print(
            f"{label}: {kind} | {len(per_file)} files -> {len(merged_rows)} samples "
            f"({terp_names[0]} + {terp_names[1]} + 2PE){trace_note}"
        )
        if flags:
            print("   high-RSD merges to check: " + ", ".join(flags))

    if only is None:  # don't clobber the combined table / binarios on a partial re-run
        with (OUT_DIR / "FLECK_TERPENOS2026_merged.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["batch", "sample", "n_rep", "terp1_area", "terp2_area", "2PE_area"])
            w.writerows(combined)
        print(f"\nCombined merged table: {OUT_DIR / 'FLECK_TERPENOS2026_merged.csv'}")

        bin_dir = DATA_DIR / "BINARIOS_TERPENOS"
        if bin_dir.is_dir():
            print()
            process_binarios(bin_dir)


if __name__ == "__main__":
    main()
