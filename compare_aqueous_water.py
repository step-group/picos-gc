# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compare aqueous-phase water: Karl Fischer (KF) titration vs water-by-difference.

KF titrates the organic phase's small water content reliably, but on the water-rich
aqueous phase (~95-99% water) it is awkward (huge titrant volumes, tiny samples). This
script checks whether taking aqueous water *by difference* (water = 1 - Σ organic mass
fractions) is cleaner, judged by KF mass-balance closure drift + regenerated plots.

No pipeline re-run needed: `out/ternarios_resultados.csv` already carries, per aqueous
row, the normalized `water` and the raw `closure` (= Σ of all four raw fractions incl.
KF). The aqueous branch of fill_ternarios normalized by `closure`, so the by-difference
endpoint is recoverable exactly (see `water_bydiff`). A row is aqueous iff its reported
water > 0.5 (aqueous 0.97-0.999 vs organic 0.01-0.08 — no overlap), matching the
pipeline's ORGANIC_WATER_MAX=0.5 discriminator.

Run: uv run compare_aqueous_water.py        (prints the table, writes the bydiff CSV,
                                             regenerates out/tielines_bydiff/)
     uv run compare_aqueous_water.py --check (self-check only)
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
SRC_CSV = _ROOT / "out" / "ternarios_resultados.csv"
BYDIFF_CSV = _ROOT / "out" / "ternarios_resultados_bydiff.csv"
PLOT_DIR = _ROOT / "pcsaft-quaternary"
BYDIFF_TIELINES = _ROOT / "out" / "tielines_bydiff"

AQUEOUS_WATER_MIN = 0.5  # reported water above this ⇒ aqueous phase
# The four composition value columns (names HBA/HBD are copied verbatim).
COMP_COLS = ("solute_2phet", "HBA_wt", "HBD_wt", "water")


def water_bydiff(water: float, closure: float) -> float:
    """Aqueous water with KF dropped: water = 1 - Σ(organic mass fractions).

    For an aqueous row the pipeline wrote water_norm = z/closure and each organic
    comp_norm = raw/closure, so Σ raw organics = (1 - water_norm)·closure and
    water_bydiff = 1 - (1 - water_norm)·closure. Reduces to water when closure == 1."""
    return 1.0 - (1.0 - water) * closure


def bydiff_row(row: dict) -> dict:
    """Return a copy of an aqueous CSV row with its four composition columns replaced
    by the by-difference endpoint (organics·closure, water = 1 - their sum). The
    endpoint sums to 1 exactly; all other columns (names, closure, flags…) are kept."""
    water = float(row["water"])
    closure = float(row["closure"])
    out = dict(row)
    out["solute_2phet"] = f"{float(row['solute_2phet']) * closure:.5f}"
    out["HBA_wt"] = f"{float(row['HBA_wt']) * closure:.5f}"
    out["HBD_wt"] = f"{float(row['HBD_wt']) * closure:.5f}"
    out["water"] = f"{water_bydiff(water, closure):.5f}"
    return out


def _is_aqueous(row: dict) -> bool:
    try:
        return float(row["water"]) > AQUEOUS_WATER_MIN
    except (TypeError, ValueError):
        return False  # incomplete row (empty composition)


def analyze(rows: list[dict]) -> None:
    """Print the aqueous KF closure-drift table + per-block and overall summaries."""
    aq = [r for r in rows if _is_aqueous(r)]
    print(f"\nAqueous-phase KF closure drift  ({len(aq)} rows, water > {AQUEOUS_WATER_MIN})")
    print(f"{'system':<8}{'water_KF':>10}{'closure':>10}{'drift':>9}"
          f"{'Δwater':>11}{'organics':>10}")
    print("-" * 58)
    per_block: dict[str, list[float]] = {}
    worst = (0.0, "")
    for r in sorted(aq, key=lambda r: (r["block"], r["system"])):
        water = float(r["water"])
        closure = float(r["closure"])
        drift = closure - 1.0
        dwater = water_bydiff(water, closure) - water
        per_block.setdefault(r["block"], []).append(abs(drift))
        if abs(drift) > worst[0]:
            worst = (abs(drift), r["system"])
        # organics scale by exactly `closure`, i.e. (closure-1) relative change
        print(f"{r['system']:<8}{water:>10.5f}{closure:>10.5f}{drift:>+9.4f}"
              f"{dwater:>+11.6f}{drift * 100:>+9.2f}%")

    print("\nPer-block mean / max |drift|:")
    all_ad: list[float] = []
    for block in sorted(per_block):
        ad = per_block[block]
        all_ad += ad
        print(f"  {block}: mean={sum(ad) / len(ad):.4f}  max={max(ad):.4f}  (n={len(ad)})")

    mean_ad = sum(all_ad) / len(all_ad)
    rms = (sum(d * d for d in all_ad) / len(all_ad)) ** 0.5
    print(f"\nOverall: mean|drift|={mean_ad:.4f}  rms={rms:.4f}  "
          f"worst={worst[1]} ({worst[0]:.4f})")
    print("Reading: |drift| is how far KF misses aqueous mass balance (=1). By-difference "
          "forces it to 0;\n         reported water barely moves (Δwater), but the dissolved "
          "2PE/terpene solubilities\n         scale by `closure` (the `organics` column).")


def write_bydiff_csv(rows: list[dict], fieldnames: list[str]) -> None:
    BYDIFF_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BYDIFF_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(bydiff_row(r) if _is_aqueous(r) else r)
    print(f"\nWrote {BYDIFF_CSV}  (aqueous rows recomputed by-difference; organic rows verbatim)")


def run_bydiff_plots() -> None:
    """Regenerate tie-line plots from the by-difference CSV into out/tielines_bydiff/,
    reusing the pcsaft-quaternary plotter via its TIELINE_CSV/TIELINE_OUT env overrides."""
    if not (PLOT_DIR / "plot_experimental_tielines.py").exists():
        print(f"  (plotter not found under {PLOT_DIR}; skipping plots)")
        return
    env = {**os.environ, "TIELINE_CSV": str(BYDIFF_CSV), "TIELINE_OUT": str(BYDIFF_TIELINES)}
    r = subprocess.run(
        ["uv", "run", "python", "plot_experimental_tielines.py"],
        cwd=PLOT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    print(r.stdout, end="")
    if r.returncode != 0:
        print(f"  plotting failed: {r.stderr.strip()[:200]}")


def _check() -> None:
    # closure == 1 (mass closes) ⇒ by-difference agrees with KF, organics unscaled.
    assert abs(water_bydiff(0.97, 1.0) - 0.97) < 1e-12
    # A synthetic normalized aqueous row round-trips to a known endpoint summing to 1.
    row = {"solute_2phet": "0.02000", "HBA_wt": "0.01000", "HBD_wt": "0.01000",
           "water": "0.96000", "closure": "0.90000", "HBA": "X", "HBD": "Y"}
    out = bydiff_row(row)
    vals = [float(out[c]) for c in COMP_COLS]
    assert abs(sum(vals) - 1.0) < 1e-6, vals  # endpoint closes to 1
    # organics scaled by closure; water = 1 - their sum
    assert abs(float(out["solute_2phet"]) - 0.02 * 0.9) < 1e-6
    assert abs(float(out["water"]) - (1 - (0.02 + 0.01 + 0.01) * 0.9)) < 1e-6
    assert _is_aqueous(row) and not _is_aqueous({"water": "0.05"})
    print("self-check OK")


def main() -> None:
    if not SRC_CSV.exists():
        sys.exit(f"missing {SRC_CSV} — run `uv run fill_ternarios.py` first")
    with SRC_CSV.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    # This tool recomputes by-difference from a KF-normalised baseline via the closure
    # identity — feeding it an already-by-difference CSV would double-apply. Since
    # fill_ternarios now defaults to --aqueous-water difference, require a KF baseline.
    if any(r.get("water_src") == "bydiff" for r in rows):
        sys.exit(
            f"{SRC_CSV.name} is already by-difference; this tool needs a KF baseline. "
            "Regenerate it with:\n  uv run fill_ternarios.py --aqueous-water kf"
        )
    analyze(rows)
    write_bydiff_csv(rows, list(fieldnames))
    run_bydiff_plots()


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        _check()  # cheap guardrail; keeps the identity honest on every run
        main()
