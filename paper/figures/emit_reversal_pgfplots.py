#!/usr/bin/env python3
"""Emit the pgfplots figure for Figure B (load-reversal mechanism) of
paper.tex from the released derived CSV. No re-derivation; Table A is
batch_size==1 only.

The figure shows one clean reversal group (AGX Orin / mobilenet-v2-100) on an
absolute external-demand axis. Each power mode is plotted over the demand range
where it is feasible (up to its measured capacity); past the energy-optimal
mode's capacity the region is shaded, marking where that mode leaves the
feasible set. In-plot text is kept minimal (deadline line and the capacity
line); mode names are carried by the legend row.

Usage:
  python emit_reversal_pgfplots.py            # 2-panel -> figures/reversal_plot.tex
  python emit_reversal_pgfplots.py --compact  # single-panel fallback
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "data", "derived", "full_dvfs_lambda_cell_summary.csv")
OUT = os.path.join(HERE, "reversal_plot.tex")
DEVICE, MODEL = "orin", "mobilenet-v2-100"
ORACLE, MAXPERF, DEADLINE = "MODE_30W", "MAXN", 100.0
# Display names for the mode labels.
DISP = {"MODE_30W": "30\\,W (energy-optimal)", "MAXN": "MAXN (max-perf.)"}


def load_group_cells(csv_path=CSV, device=DEVICE, model=MODEL):
    """Return {mode_label: [(demand_rps, energy, p95), ...]} sorted by demand,
    plus {mode_label: capacity_rps}. Demand is the measured achieved rate at
    each load fraction, so each mode spans up to its own capacity."""
    cells = {}
    caps = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["device"] != device or r["model"] != model:
                continue
            if int(r["batch_size"]) != 1:
                continue
            m = r["dvfs_mode_label"]
            cells.setdefault(m, []).append((
                float(r["achieved_rps_median"]),
                float(r["marginal_wall_energy_j_per_inf_median"]),
                float(r["p95_latency_ms_median"]),
            ))
            caps[m] = float(r["capacity_ips"])
    for m in cells:
        cells[m].sort(key=lambda t: t[0])
    return cells, caps


def _ecoords(pts):
    return " ".join(f"({d:.1f},{e:.4f})" for (d, e, p) in pts)


def _pcoords(pts):
    return " ".join(f"({d:.1f},{p:.1f})" for (d, e, p) in pts)


def _ranges(cells):
    es = [e for pts in cells.values() for (d, e, p) in pts]
    ps = [p for pts in cells.values() for (d, e, p) in pts]
    ds = [d for pts in cells.values() for (d, e, p) in pts]
    return min(es), max(es), max(ps), max(ds)


def _interp_energy(pts, x):
    """Linear interpolation of energy at demand x from sorted (demand, e, p)."""
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (da, ea, _), (db, eb, _) in zip(pts, pts[1:]):
        if da <= x <= db:
            w = (x - da) / (db - da)
            return ea * (1.0 - w) + eb * w
    return pts[-1][1]


def render_tikz(cells, caps, oracle=ORACLE, maxperf=MAXPERF, deadline=DEADLINE):
    others = [m for m in cells if m not in (oracle, maxperf)]
    emin, emax, pmax, dmax = _ranges(cells)
    cap_o = caps[oracle]
    xmax = dmax * 1.05
    e_lo, e_hi = emin * 0.93, emax * 1.07
    p_hi = max(pmax * 1.10, deadline * 1.12)
    L = []
    a = L.append
    # legend row (mode names live here, not inside the plot)
    a(r"{\scriptsize "
      r"\tikz[baseline=-0.5ex]{\draw[blue!55!black,line width=1.1pt](0,0)--(0.36,0);"
      r"\fill[blue!55!black](0.135,-0.045) rectangle (0.225,0.045);}\,MAXN (max-perf.)\enspace "
      r"\tikz[baseline=-0.5ex]{\draw[red!75!black,line width=1.1pt](0,0)--(0.36,0);"
      r"\fill[red!75!black](0.18,0) circle (0.055);}\,30\,W (energy-optimal)\enspace "
      r"\tikz[baseline=-0.5ex]{\draw[gray!60,line width=0.8pt](0,0)--(0.36,0);}\,other power modes}")
    a(r"\par\vspace{2pt}")
    a(r"\begin{tikzpicture}")
    a(r"\begin{groupplot}[")
    a(r"  group style={group size=1 by 2, vertical sep=0.85cm,")
    a(r"    x descriptions at=edge bottom},")
    a(r"  width=\columnwidth, height=0.40\columnwidth,")
    a(rf"  xmin=0, xmax={xmax:.0f}, xtick={{0,200,400,600}},")
    a(r"  ymajorgrids=true, grid style={black!12},")
    a(r"  label style={font=\footnotesize}, tick label style={font=\footnotesize},")
    a(r"]")

    def shade_and_capline(ymin, ymax):
        a(rf"\addplot[forget plot, draw=none, fill=red!7] coordinates "
          rf"{{({cap_o:.1f},{ymin}) ({xmax:.0f},{ymin}) ({xmax:.0f},{ymax}) "
          rf"({cap_o:.1f},{ymax})}} \closedcycle;")
        a(rf"\draw[red!70!black, dotted, line width=0.9pt] "
          rf"(axis cs:{cap_o:.1f},{ymin}) -- (axis cs:{cap_o:.1f},{ymax});")

    # --- panel 1: energy ---
    a(rf"\nextgroupplot[ylabel={{AC input energy (J/inf.)}}, "
      rf"ymin={e_lo:.4f}, ymax={e_hi:.4f}, ytick={{0.015,0.020}}, "
      rf"yticklabels={{0.015,0.020}}, scaled y ticks=false]")
    shade_and_capline(f"{e_lo:.4f}", f"{e_hi:.4f}")
    for m in others:
        a(rf"\addplot[forget plot, gray!55, line width=0.4pt, mark=*, mark size=0.8pt] "
          rf"coordinates {{{_ecoords(cells[m])}}};")
    a(rf"\addplot[forget plot, blue!55!black, line width=1.3pt, mark=square*, mark size=1.5pt] "
      rf"coordinates {{{_ecoords(cells[maxperf])}}};")
    a(rf"\addplot[forget plot, red!75!black, line width=1.3pt, mark=*, mark size=1.5pt] "
      rf"coordinates {{{_ecoords(cells[oracle])}}};")
    # energy-gap marker between the cheap mode and max-perf at a moderate demand
    dx, e_c = cells[oracle][1][0], cells[oracle][1][1]
    e_m = _interp_energy(cells[maxperf], dx)
    gap = round(100 * (e_m - e_c) / e_m)
    arrow_x = dx + 18.0
    e_c_arrow = _interp_energy(cells[oracle], arrow_x)
    e_m_arrow = _interp_energy(cells[maxperf], arrow_x)
    arrow_pad = (e_m_arrow - e_c_arrow) * 0.08
    a(rf"\draw[{{Stealth[length=3.0pt,width=3.2pt]}}-{{Stealth[length=3.0pt,width=3.2pt]}}, "
      rf"red!75!black, line width=0.85pt] "
      rf"(axis cs:{arrow_x:.0f},{e_c_arrow + arrow_pad:.4f}) -- "
      rf"(axis cs:{arrow_x:.0f},{e_m_arrow - arrow_pad:.4f});")
    a(rf"\node[font=\scriptsize, red!75!black, anchor=west, fill=white, "
      rf"fill opacity=0.85, text opacity=1, inner sep=0.6pt] "
      rf"at (axis cs:{arrow_x + 7:.0f},{(e_c_arrow + e_m_arrow) / 2:.4f}) "
      rf"{{$\approx${gap}\% cheaper}};")

    # --- panel 2: latency ---
    a(r"\nextgroupplot[ylabel={p95 latency (ms)}, "
      r"xlabel={External demand (req/s)},")
    a(rf"  ymin=0, ymax={p_hi:.0f}, ytick={{0,50,100}}]")
    shade_and_capline("0", f"{p_hi:.0f}")
    for m in others:
        a(rf"\addplot[forget plot, gray!55, line width=0.4pt, mark=*, mark size=0.8pt] "
          rf"coordinates {{{_pcoords(cells[m])}}};")
    a(rf"\addplot[forget plot, blue!55!black, line width=1.3pt, mark=square*, "
      rf"mark size=1.5pt] coordinates {{{_pcoords(cells[maxperf])}}};")
    a(rf"\addplot[forget plot, red!75!black, line width=1.3pt, mark=*, "
      rf"mark size=1.5pt] coordinates {{{_pcoords(cells[oracle])}}};")
    a(rf"\addplot[forget plot, dashed, black!55, line width=0.8pt, samples=2, "
      rf"domain=0:{xmax:.0f}] {{{deadline:g}}};")
    a(rf"\node[font=\scriptsize, black!55, anchor=north west] "
      rf"at (axis cs:8,{deadline - 3:g}) {{100\,ms latency SLO}};")
    a(r"\end{groupplot}")
    a(r"\end{tikzpicture}")
    return "\n".join(L) + "\n"


def render_compact(cells, caps, oracle=ORACLE, maxperf=MAXPERF, deadline=DEADLINE):
    others = [m for m in cells if m not in (oracle, maxperf)]
    emin, emax, _, dmax = _ranges(cells)
    cap_o = caps[oracle]
    xmax = dmax * 1.05
    e_lo, e_hi = emin * 0.93, emax * 1.10
    L = []
    a = L.append
    a(r"\begin{tikzpicture}")
    a(r"\begin{axis}[")
    a(r"  width=\columnwidth, height=0.62\columnwidth,")
    a(r"  xlabel={External demand (req/s)},")
    a(r"  ylabel={Marginal wall energy (J/inf.)},")
    a(rf"  xmin=0, xmax={xmax:.0f}, ymin={e_lo:.4f}, ymax={e_hi:.4f},")
    a(r"  xtick={0,200,400,600}, ymajorgrids=true, grid style={black!12},")
    a(r"  label style={font=\footnotesize}, tick label style={font=\footnotesize},")
    a(r"  legend style={draw=none, fill=none, font=\scriptsize, at={(0.02,0.98)},")
    a(r"    anchor=north west}, legend cell align=left]")
    a(rf"\addplot[forget plot, draw=none, fill=red!7] coordinates "
      rf"{{({cap_o:.1f},{e_lo:.4f}) ({xmax:.0f},{e_lo:.4f}) ({xmax:.0f},{e_hi:.4f}) "
      rf"({cap_o:.1f},{e_hi:.4f})}} \closedcycle;")
    for m in others:
        a(rf"\addplot[forget plot, gray!55, line width=0.4pt] "
          rf"coordinates {{{_ecoords(cells[m])}}};")
    a(rf"\addplot[blue!55!black, line width=1.3pt, mark=square*, mark size=1.5pt] "
      rf"coordinates {{{_ecoords(cells[maxperf])}}};")
    a(r"\addlegendentry{MAXN (max-perf.)}")
    a(rf"\addplot[red!75!black, line width=1.3pt, mark=*, mark size=1.5pt] "
      rf"coordinates {{{_ecoords(cells[oracle])}}};")
    a(r"\addlegendentry{30\,W (energy-optimal)}")
    a(r"\end{axis}")
    a(r"\end{tikzpicture}")
    return "\n".join(L) + "\n"


def main():
    cells, caps = load_group_cells()
    compact = "--compact" in sys.argv
    tex = render_compact(cells, caps) if compact else render_tikz(cells, caps)
    with open(OUT, "w") as fh:
        fh.write(tex)
    print(f"wrote {OUT} ({'compact' if compact else '2-panel'}), {len(cells)} modes")


if __name__ == "__main__":
    main()
