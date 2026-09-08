// Printable lab worksheet of the LLE points to repeat, built from out/repeat_list.csv.
// Regenerate:  uv run audit_repeats.py && typst compile repeat_list.typ out/repeat_list.pdf
#set page(paper: "a4", flipped: true, margin: (x: 1.4cm, y: 1.4cm), numbering: "1 / 1")
#set text(font: "Helvetica", size: 9.5pt)
#set table(stroke: 0.4pt + luma(160), inset: 5pt)

#let rows = csv("out/repeat_list.csv", row-type: dictionary)
#let todo = rows.filter(r => r.repeat == "yes")
#let tern = todo.filter(r => r.kind == "ternary")
#let bin = todo.filter(r => r.kind == "binary")

#let action = (
  missing_vials: "Never injected: prepare both vials, KF + GC.",
  low_closure: "Mass does not close: re-sample the phase, re-weigh, KF, GC.",
  replicate_mismatch: "Replicate injections differ > 3x: re-dilute and re-inject both vials.",
  aqueous_organics_suspect: "Organic droplets in both aqueous vials: re-sample after longer settling, from mid-layer, avoiding the interface.",
  aqueous_replicate_mismatch: "Aqueous vials disagree on 2PE > 3x (no droplet signature) and the organic phase gives no reference: re-sample both.",
  aqueous_2pe_outlier: "Aqueous 2PE misses the other 2PE-rich tie-lines by > 1.5x, though they share the same organic phase (≈88 % 2PE): re-sample the aqueous phase, re-weigh, re-inject.",
)
#let why(r) = r.reason.split(";").map(k => action.at(k)).join(linebreak())
#let gl(s) = calc.round(float(s) * 1000, digits: 2)
#let box = $square$

#let sheet(rows, title) = {
  heading(level: 2, [#title (#rows.len())])
  table(
    columns: (1.4cm, 1.7cm, 3.2cm, 3.4cm, 3.2cm, 1fr, 1cm, 1cm, 1cm, 2.6cm),
    align: (x, y) => if x >= 6 { center + horizon } else { left + horizon },
    fill: (x, y) => if y == 0 { luma(235) },
    table.header(
      [*Point*], [*Phase*], [*Solvent pair*], [*Sample codes*], [*Current data*],
      [*Why · what to do*], [*Prep*], [*KF*], [*GC*], [*Date / init.*],
    ),
    ..rows
      .map(r => (
        strong(r.system),
        r.phase,
        [#r.hba + #r.hbd],
        raw(r.codes),
        {
          let d = ()
          if r.closure != "" { d.push([closure Σ #r.closure]) }
          if r.aq_terpene_max != "" { d.push([terpenes #gl(r.aq_terpene_max) g/L (max #gl(r.aq_ceiling))]) }
          if r.aq_organics_ratio != "" { d.push([vial ratio #r.aq_organics_ratio]) }
          if r.reason.contains("2pe_outlier") { d.push([2PE #gl(r.w_2pe) g/L (others 16–20)]) }
          d.join(linebreak())
        },
        why(r),
        box, box, box, [],
      ))
      .flatten(),
  )
}

#align(center)[
  #text(15pt, weight: "bold")[LLE points to repeat] \
  #text(luma(90))[#todo.len() of #rows.len() points · water + 2‑phenylethanol + terpene pairs · generated #datetime.today().display()]
]

#sheet(tern, "Ternary tie-lines")
#sheet(bin, "Binary water–solvent edge (no 2‑phenylethanol)")

#v(6pt)
#block(inset: 8pt, stroke: 0.4pt + luma(160), radius: 3pt, width: 100%)[
  *Reference values for judging the new vials (mass basis, 30 °C).*
  Clean aqueous phases here carry 0.1–0.6 g/L of each terpene (up to 1.3 g/L for camphor + carvacrol).
  Pure-water solubility at 30 °C (Martins 2017; Smyrl 1980 for carvone; Yalkowsky for camphor): thymol 1.1, carvacrol 1.3, carvone 1.6, camphor ≈2.0, eugenol 2.1, geraniol 1.2 g/L.
  The audit ceiling per point ("max" above) is the pair's summed solubility +20 % for method scatter, 2.8–4.9 g/L: even both terpenes at saturation cannot exceed it.
  2‑phenylethanol in water: 23–30 g/L across the literature, too wide a spread to judge a
  single point; but the eight 2PE‑rich tie‑lines share one organic phase (≈88 % 2PE), so
  their aqueous phases must agree, and the five sound ones measured 16–20 g/L. Carried-over organic droplets show as both terpenes rising together at the organic phase's ratio.
  Organic phases: closure Σ must fall in 0.5–1.5; replicate injections must agree within 3×.
]
