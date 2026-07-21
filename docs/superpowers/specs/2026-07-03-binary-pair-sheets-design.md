# Binary-pair sheets (Water–solvent edge) — Design

**Date:** 2026-07-03

**Goal:** Give every ternary block in `Sistemas ternarios_MF.xlsx` a sibling
`Bloque X bin` sheet for the **Water–solvent binary edge** (HBA + HBD + Water, no
2PE), at the same temperature, with the raw-measurement cells left blank as
placeholders the user fills in by hand.

## Context

Each `Bloque A..I` sheet is one pseudo-ternary LLE system: Water + 2-phenylethanol
(Solute) + a fixed-ratio two-terpene solvent (HBA + HBD). The measurement scaffold
per vial is: vial masses (D/F/H) → 2-propanol dilution (I/J/K) → GC areas (L/M/N) →
diluido %m/m (O/P/Q = area/slope) → real %m/m (R/S/T = ×K/I) → KF water (U/V) →
per-phase averages (W/X/Y/Z) → closure `AA=SUM(W:Z)` → normalized point
(AD/AE/AF/AG = Solute/HBA/HBD/Water). Slopes (F2/G2/H2) come from `CC_MF.xlsx`.

The binary edge drops the Solute (2PE) component: it is the same experiment run on
the terpene solvent + water alone (the HBA:HBD ratio is fixed per block, so at one
temperature this is a single mutual-solubility tie-line — but the sheet mirrors the
full ternary 20-vial / 5-system grid per the user's choice, to be filled as needed).

Two workbooks matter: `Sistemas ternarios_MF.xlsx` (master, hand-edited, the write
target here) and `CC_MF.xlsx` (calibration slopes). `fill_ternarios.py` reads the
master and writes `Sistemas ternarios_MF_filled.xlsx`.

## Sheet layout — mirror of the ternary, Solute column dropped

The generator **clones** the ternary sheet (`wb.copy_worksheet`) to inherit column
widths, borders and number formats, then rewrites the cells below. Column addresses
stay identical to the ternary (HBA in M, HBD in N, Water in Z, normalized point in
AE/AF/AG); the Solute column (L/O/R/W/AD) and its slope (F1/F2) are cleared.

| Cell(s) | Value |
|---|---|
| `G1` | pair label (e.g. `Thymol:Carvone`), copied from the ternary sheet |
| `F1`, `F2` | **cleared** (no solute label, no 2PhEt slope) |
| `E2` | `Pend. CC` (kept) |
| `G2`, `H2` | HBA / HBD through-origin slopes from `CC_MF.xlsx` (`cc_mf_slopes`) |
| `M3`, `N3` | HBA / HBD display names. Block F ships blank → derive from `G1` (`Camph:Carvone` → Camphor, Carvone) |
| row 4 headers | copied; Solute-role labels `L4/O4/R4/W4/AD4` **cleared**; `AE4/AF4` set to HBA/HBD names, `AG4`=`Water` |
| `A` on rows 5,9,13,17,21 | clean system codes `X1`…`X5` (block letter + 1..5) — avoids the stale `E1..E5` that F/H/I carry |
| `B`, `C` (all data rows) | Fase `Superior/Superior/Inferior/Inferior` per system; vial code 1..20 |
| `AC` on rows 5,9,13,17,21 | 1..5 (system index, cosmetic summary helper) |
| **placeholders — left blank** | `D` Vial g, `E` Muestra µL, `F` V+muestra g, `H` V+M+Met g, `M` Area HBA, `N` Area HBD, `U`/`V` KF water rep1/rep2 |
| `G` (rows 5–24) | `=1000-E{r}` |
| `I`,`J`,`K` (5–24) | `=F{r}-D{r}`, `=H{r}-F{r}`, `=J{r}+I{r}` |
| `L`,`O`,`R` (5–24) | **cleared** (no solute) |
| `P`,`Q` (5–24) | `=M{r}/$G$2`, `=N{r}/$H$2` (diluido %m/m) |
| `S`,`T` (5–24) | `=P{r}*$K{r}/$I{r}`, `=Q{r}*$K{r}/$I{r}` (real %m/m) |
| `W` on phase rows 5,7,…,23 | **cleared** (no solute; so `AA=SUM(W:Z)=X+Y+Z`) |
| `X`,`Y`,`Z` on phase rows | `=AVERAGE(S{r}:S{r+1})/100`, `…T…`, `=AVERAGE(U{r}:V{r+1})/100` |
| `AA` on phase rows | `=SUM(W{r}:Z{r})` (closure ≈ 1) |
| `AD` on phase rows | **cleared** (Solute = 0) |
| `AE`,`AF`,`AG` on phase rows | `=IFERROR(X{r}/$AA{r},"")`, `…Y…`, `…Z…` → HBA/HBD/Water, sum to 1; blank (not `#DIV/0`) until data is entered |

Phase-anchor rows are 5,7,9,…,23 (each phase = 2 replicate rows); system-anchor rows
are 5,9,13,17,21 (each system = Superior + Inferior = 4 rows).

## Components

### New: `add_binary_sheets.py` (repo root)

PEP 723 header pulling `openpyxl`; `_ROOT = Path(__file__).resolve().parent` for
CWD-independence (matches `fill_ternarios.py`). Reuses `cc_mf_slopes`, `canon`,
`DISPLAY` from `fill_ternarios` (importing it is side-effect-free — it is guarded by
`if __name__ == "__main__"`).

Flow:
1. Load `CC_MF.xlsx` slopes and the master workbook (`data_only=False`, keep formulas).
2. For each ternary sheet (name **not** ending in ` bin`), if `f"{name} bin"` does
   **not** already exist, `copy_worksheet` it and rename. The clone carries the
   ternary's vial data and its (incomplete/offset) result formulas, so first
   **clear the whole data region** — rows 5–24, columns D–AG — then write the binary
   placeholders/formulas from the layout table (this wipes copied areas, KF, solute
   data, and the stray `AD..AG` formulas on non-anchor rows in one pass). Columns
   A/B/C and the row-1..4 header cells are set explicitly per the table.
   If the bin sheet already exists, **skip** (print a note) — idempotent, never
   clobbers entered data.
3. Insert each new bin sheet immediately after its ternary sheet.
4. Save the master in place. Print which sheets were created / skipped.

HBA/HBD names: read `M3`/`N3`; if blank (F), split `G1` on `:` and `canon` each side
to look up the CC_MF slope and `DISPLAY` name.

### Modify: `fill_ternarios.py` — skip bin sheets

`write_formulas()` and `main()` both iterate every worksheet and would stamp the
ternary (solute-based) formulas over the bin sheets. Add a `" bin"`-suffix skip in
both loops (a small `_is_bin(name)` helper or inline `name.endswith(" bin")`). With
the guard the bin sheets pass through the master→`_filled` copy untouched.

## Idempotency & safety

- `add_binary_sheets.py` skips existing bin sheets → safe to re-run; it only adds
  sheets for blocks that lack one.
- It writes into the master in place but touches **only** newly created sheets; the
  ternary sheets are never modified.
- No LibreOffice recalc needed — the template has no data; Excel/LibreOffice compute
  the formulas when the user opens and fills the sheet.

## Verification

- `uv run add_binary_sheets.py` prints 8 `created Bloque X bin` lines on first run;
  a second run prints 8 `skip` lines and changes nothing.
- Open `Sistemas ternarios_MF.xlsx`: 16 sheets, interleaved ternary/bin. Each bin
  sheet has G2/H2 slopes, HBA/HBD names, blank D/E/F/H/M/N/U/V, and AE/AF/AG blank.
- Self-check in the generator (`__main__`-guarded `demo()`/assert): fill one bin
  sheet's row with synthetic masses/areas/KF in-memory, recalc the P/Q/S/T/X/Y/Z/AA
  chain in Python from the same formulas, and assert AE+AF+AG == 1.
- `fill_ternarios._selfcheck()` still prints `selfcheck OK`; a full
  `uv run fill_ternarios.py` still fills the 8 ternary blocks and does **not** warn
  about or write into the bin sheets.

## Out of scope (add when asked)

- Wiring the binary edge into `plot_experimental_tielines.py` or the results CSV.
- Auto-filling binary GC areas from a merged CSV (these are hand-entered for now).
- Any change to `CC_MF.xlsx` or the ternary sheets' contents.
