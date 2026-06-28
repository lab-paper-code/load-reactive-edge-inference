#!/usr/bin/env python3
"""Emit two native-pgfplots figures for paper.tex from the released artifacts.
No re-derivation; Table A is batch_size==1 only.

  fleet_cloud.tex    Figure A: all 396 measured cells in the (p95, energy) plane,
                     log-log, colored by segment, sized by load, 100 ms deadline.
                     Pure raw measurement.
  ranking_flip.tex   Device energy-rank slopegraph, isolated profiling vs serving,
                     one panel per model (decision_flip.csv). Shows that the
                     profile's best device is never the serving best, and that the
                     reorder is non-uniform, so no single scale recovers it.
  savings_dist.tex   Savings distribution over the 37 decisions that admit a choice
                     (dynamic vs max-perf, scheduler_policy_eval.csv), sorted,
                     colored by segment. Interpolated policy layer.
"""
import csv, os
from collections import defaultdict

DER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "derived")
CSV_A = os.path.join(DER, "full_dvfs_lambda_cell_summary.csv")
CSV_B = os.path.join(DER, "scheduler_policy_eval.csv")
CSV_C = os.path.join(DER, "decision_flip.csv")
HERE = os.path.dirname(os.path.abspath(__file__))

SEG_OF = {"gpu-server": "Server", "orin": "Jetson", "orin-nano": "Jetson",
          "xavier": "Jetson", "jetson": "Jetson", "orangepi-npu": "SB-NPU",
          "orangepi": "SB-CPU", "rasp5": "SB-CPU", "lattepanda": "SB-CPU"}
SEG_HEX = {"Server": "D55E00", "Jetson": "0072B2", "SB-NPU": "009E73", "SB-CPU": "CC79A7"}
SEG_KEY = {"Server": "segserver", "Jetson": "segjetson", "SB-NPU": "segnpu", "SB-CPU": "segcpu"}
SEG_LABEL = {"Server": "Server", "Jetson": "Jetson",
             "SB-NPU": "Small-board NPU", "SB-CPU": "Small-board CPU"}
# fleet scatter draw order: small-board NPU drawn last so its sparse points stay on top
FLEET_SEGS = ["Server", "Jetson", "SB-CPU", "SB-NPU"]
LEGEND_SEGS = ["Server", "Jetson", "SB-NPU", "SB-CPU"]
SIZE_OF = {0.25: 0.5, 0.5: 1.0, 0.75: 1.7, 1.0: 2.5}
DEADLINE = 100.0


def load_A():
    rows = []
    with open(CSV_A) as fh:
        for r in csv.DictReader(fh):
            if int(r["batch_size"]) != 1:
                continue
            rows.append((SEG_OF[r["device"]],
                         r["model"],
                         round(float(r["lambda_frac"]), 2),
                         float(r["p95_latency_ms_median"]),
                         float(r["marginal_wall_energy_j_per_inf_median"])))
    return rows


def colordefs(segs):
    return "\n".join(rf"\definecolor{{{SEG_KEY[s]}}}{{HTML}}{{{SEG_HEX[s]}}}"
                     for s in segs)


MODELS = [("mobilenet-v2-050", "MobileNet-V2 0.5"),
          ("mobilenet-v2-100", "MobileNet-V2 1.0"),
          ("efficientnet-b4", "EfficientNet-B4")]


def fig_fleet_cloud(rows):
    """Two-column figure: one (p95, energy) panel per model, shared log-log axes."""
    ps = [p for (_, _, _, p, _) in rows]
    es = [e for (_, _, _, _, e) in rows]
    pmin, pmax = min(ps), max(ps)
    emin, emax = min(es), max(es)
    xmin, xmax = pmin * 0.85, pmax * 1.08
    ymin, ymax = emin * 0.85, emax * 1.1
    L = [colordefs(FLEET_SEGS), r"\begin{tikzpicture}", r"\begin{groupplot}["]
    a = L.append
    a(r"  group style={group size=3 by 1, horizontal sep=0.35cm,")
    a(r"    ylabels at=edge left, yticklabels at=edge left},")
    a(r"  width=0.345\textwidth, height=0.18\textwidth,")
    a(r"  xmode=log, ymode=log,")
    a(r"  xlabel={p95 latency (ms)}, ylabel={Marginal wall energy (J/inf.)},")
    a(rf"  xmin={xmin:.1f}, xmax={xmax:.0f}, ymin={ymin:.4f}, ymax={ymax:.3f},")
    a(r"  ymajorgrids=true, grid style={black!10},")
    a(r"  title style={font=\footnotesize, yshift=-2pt},")
    a(r"  label style={font=\footnotesize}, tick label style={font=\scriptsize},")
    a(r"]")
    for mkey, mlabel in MODELS:
        a(rf"\nextgroupplot[title={{{mlabel}}}]")
        # shade infeasible region p95 > deadline
        a(rf"\addplot[draw=none, fill=black!6, forget plot] coordinates "
          rf"{{({DEADLINE:g},{ymin:.4f}) ({xmax:.0f},{ymin:.4f}) "
          rf"({xmax:.0f},{ymax:.3f}) ({DEADLINE:g},{ymax:.3f})}} \closedcycle;")
        a(rf"\draw[dashed, black!55, line width=0.7pt] (axis cs:{DEADLINE:g},{ymin:.4f}) -- "
          rf"(axis cs:{DEADLINE:g},{ymax:.3f});")
        # one scatter per (segment, load) within this model
        for seg in FLEET_SEGS:
            for ld in [0.25, 0.5, 0.75, 1.0]:
                pts = [(p, e) for (s, m, l, p, e) in rows
                       if m == mkey and s == seg and l == ld]
                if not pts:
                    continue
                coords = " ".join(f"({p:.1f},{e:.4f})" for (p, e) in pts)
                a(rf"\addplot[only marks, mark=*, mark size={SIZE_OF[ld]}pt, "
                  rf"color={SEG_KEY[seg]}, fill={SEG_KEY[seg]}, fill opacity=0.55, "
                  rf"draw opacity=0.55, forget plot] coordinates {{{coords}}};")
    a(r"\end{groupplot}")
    a(r"\end{tikzpicture}")
    # legend as plain LaTeX colored bullets, centered below (colors defined at the
    # top of this file so they are in scope outside the tikzpicture)
    a(r"\par\vspace{1pt}")
    leg = r" \quad ".join(
        rf"\textcolor{{{SEG_KEY[s]}}}{{$\bullet$}}\,{SEG_LABEL[s]}"
        for s in LEGEND_SEGS)
    a(rf"{{\footnotesize {leg}}}")
    # size key: marker area encodes load fraction of capacity
    sizes = r" \quad ".join(
        rf"\tikz[baseline=-0.5ex]\draw[fill=black] (0,0) circle ({SIZE_OF[ld]}pt);\,{ld:g}"
        for ld in [0.25, 0.5, 0.75, 1.0])
    a(r"\par\vspace{1pt}")
    a(rf"{{\footnotesize marker size $=$ load (fraction of capacity): {sizes}}}")
    return "\n".join(L) + "\n", (pmin, pmax, emin, emax)


# --- ranking-flip slopegraph: isolated profile vs serving, per model ---------
RF_POLICY = "e_inc"          # ranks are identical under both serving policies
RF_HEX = {"isobest": "D55E00", "srvbest": "009E73", "server": "0072B2"}
RF_SHORT = {"gpu-server": "Server", "orin": "Orin", "xavier": "Xavier",
            "jetson": "Nano", "orangepi-npu": "NPU", "orangepi": "OPi",
            "rasp5": "RPi5", "lattepanda": "Latte"}


def load_rankflip():
    g = defaultdict(list)
    with open(CSV_C) as fh:
        for r in csv.DictReader(fh):
            if r["policy"] != RF_POLICY:
                continue
            g[r["model"]].append((r["device"], int(r["iso_rank"]),
                                  int(r["srv_rank"])))
    return g


def _rf_panel(label, rows):
    rows = sorted(rows, key=lambda t: t[1])
    iso_best = min(rows, key=lambda t: t[1])[0]
    srv_best = min(rows, key=lambda t: t[2])[0]
    L = []
    a = L.append
    a(rf"\nextgroupplot[title={{{label}}}]")
    for (dev, ir, sr) in rows:
        if dev == iso_best:
            col, lw = "isobest", "1.3pt"
        elif dev == srv_best:
            col, lw = "srvbest", "1.3pt"
        elif dev == "gpu-server":
            col, lw = "server", "1.0pt"
        else:
            col, lw = "gray!60", "0.5pt"
        a(rf"\addplot[forget plot, {col}, line width={lw}, mark=*, "
          rf"mark size=1.3pt] coordinates {{(0,{ir}) (1,{sr})}};")
    # label only the three role devices, to keep the panel readable
    ib_ir = [ir for (d, ir, sr) in rows if d == iso_best][0]
    a(rf"\node[font=\tiny, color=isobest, anchor=east, xshift=-2pt] "
      rf"at (axis cs:0,{ib_ir}) {{Orin}};")
    sb_sr = [sr for (d, ir, sr) in rows if d == srv_best][0]
    a(rf"\node[font=\tiny, color=srvbest, anchor=west, xshift=2pt] "
      rf"at (axis cs:1,{sb_sr}) {{{RF_SHORT[srv_best]}}};")
    sv_ir = [ir for (d, ir, sr) in rows if d == "gpu-server"]
    if sv_ir:
        a(rf"\node[font=\tiny, color=server, anchor=east, xshift=-2pt] "
          rf"at (axis cs:0,{sv_ir[0]}) {{Server}};")
    return "\n".join(L)


def fig_ranking_flip(g):
    maxn = max(len(g[m]) for m, _ in MODELS)
    L = []
    a = L.append
    for c, h in RF_HEX.items():
        a(rf"\definecolor{{{c}}}{{HTML}}{{{h}}}")
    a(r"\begin{tikzpicture}")
    a(r"\begin{groupplot}[")
    a(r"  group style={group size=3 by 1, horizontal sep=0.95cm},")
    a(r"  width=0.18\textwidth, height=0.285\textwidth,")
    a(r"  clip=false,")
    a(r"  y dir=reverse,")
    a(r"  xmin=-0.1, xmax=1.1,")
    a(r"  xtick={0,1}, xticklabels={Profile, Serving},")
    a(rf"  ymin=-0.7, ymax={maxn - 1 + 0.7:.1f},")
    a(r"  ytick=\empty,")
    a(r"  axis y line=none, axis x line=bottom,")
    a(r"  x axis line style={black!55}, xtick style={black!55},")
    a(r"  tick label style={font=\scriptsize}, "
      r"title style={font=\footnotesize, yshift=-3pt},")
    a(r"]")
    for idx, (mk, ml) in enumerate(MODELS):
        panel = _rf_panel(ml, g[mk])
        if idx == 0:
            # y-direction cue in the left margin of the first panel: top is best
            panel += (
                "\n\\draw[-{Stealth[length=3.5pt,width=3.5pt]}, black!45, "
                "line width=0.7pt] (rel axis cs:-0.30,0.06) -- (rel axis cs:-0.30,0.94);"
                "\n\\node[font=\\scriptsize, black!45, rotate=90, anchor=south] "
                "at (rel axis cs:-0.42,0.5) {less energy};")
        a(panel)
    a(r"\end{groupplot}")
    a(r"\end{tikzpicture}")
    # color key (line colors are defined before the tikzpicture, so in scope here)
    a(r"\par\vspace{1pt}")
    a(r"{\scriptsize \textcolor{isobest}{$\bullet$}\,best when profiled\enspace "
      r"\textcolor{srvbest}{$\bullet$}\,best when serving\enspace "
      r"\textcolor{server}{$\bullet$}\,GPU server\enspace "
      r"\textcolor{black!45}{$\bullet$}\,other devices}")
    info = {ml: (min(g[mk], key=lambda t: t[1])[0],
                 min(g[mk], key=lambda t: t[2])[0], len(g[mk]))
            for mk, ml in MODELS}
    return "\n".join(L) + "\n", info


# --- savings distribution: dynamic vs max-perf default, per decision ----------
# per-bar identity: each decision is one (device, model, load) point
SAV_SHORT = {"gpu-server": "Server", "orin": "Orin", "xavier": "Xavier",
             "jetson": "Nano", "orin-nano": "OrinN"}
SAV_MOD = {"mobilenet-v2-050": "M0.5", "mobilenet-v2-100": "M1.0",
           "efficientnet-b4": "E-B4"}


def load_savings():
    rows = []
    with open(CSV_B) as fh:
        for r in csv.DictReader(fh):
            if float(r["n_sla_safe_modes"]) <= 1:
                continue
            label = (f"{SAV_SHORT[r['device']]} {SAV_MOD[r['model']]} "
                     f"{float(r['demand_frac_of_group_max']):.2f}")
            rows.append((SEG_OF[r["device"]],
                         float(r["dynamic_energy_saving_vs_max_perf_pct"]),
                         label))
    rows.sort(key=lambda t: -t[1])
    return rows


def fig_savings(rows):
    import statistics
    vals = [v for (_, v, _) in rows]
    med = statistics.median(vals)
    vmax = max(vals)
    n = len(rows)
    L = [r"\begin{tikzpicture}", colordefs(["Server", "Jetson"]), r"\begin{axis}["]
    a = L.append
    a(r"  width=\columnwidth, height=0.50\columnwidth,")
    a(r"  ylabel={Energy saved vs max-perf (\%)},")
    a(rf"  xmin=0, xmax={n+1}, ymin=0, ymax={vmax*1.12:.0f},")
    ticks = ",".join(str(i + 1) for i in range(n))
    labels = ",".join("{" + lab + "}" for (_, _, lab) in rows)
    a(rf"  xtick={{{ticks}}}, xticklabels={{{labels}}},")
    a(r"  x tick label style={rotate=60, anchor=east, font=\tiny, yshift=0.4pt},")
    a(r"  ymajorgrids=true, grid style={black!10},")
    a(r"  label style={font=\footnotesize}, tick label style={font=\footnotesize},")
    a(r"  legend style={draw=none, fill=none, font=\scriptsize, at={(0.98,0.98)}, anchor=north east},")
    a(r"  legend cell align=left,")
    a(r"  area legend,")
    a(r"]")
    for seg in ["Server", "Jetson"]:
        coords = " ".join(f"({i+1},{v:.2f})" for i, (s, v, _) in enumerate(rows) if s == seg)
        if not coords.strip():
            continue
        a(rf"\addplot[ybar, bar width=3.0pt, bar shift=0pt, draw={SEG_KEY[seg]}, "
          rf"fill={SEG_KEY[seg]}] coordinates {{{coords}}};")
        a(rf"\addlegendentry{{{seg}}}")
    a(rf"\draw[densely dotted, black, line width=0.9pt] (axis cs:0,{med:.2f}) -- (axis cs:{n+1},{med:.2f});")
    a(rf"\node[font=\scriptsize, anchor=south east] at (axis cs:{n},{med:.2f}) {{median {med:.1f}\%}};")
    a(rf"\node[font=\scriptsize, color=segserver, anchor=south west] at (axis cs:1,{vmax:.2f}) {{max {vmax:.1f}\%}};")
    a(r"\end{axis}")
    a(r"\end{tikzpicture}")
    return "\n".join(L) + "\n", (n, med, vmax)


if __name__ == "__main__":
    A = load_A()
    rf = load_rankflip()
    sv = load_savings()
    print(f"A cells: {len(A)} (expect 396)")
    print(f"savings rows: {len(sv)} (expect 37)")
    texA, infoA = fig_fleet_cloud(A)
    texR, infoR = fig_ranking_flip(rf)
    texS, infoS = fig_savings(sv)
    print(f"A ranges p95 {infoA[0]:.1f}-{infoA[1]:.1f}, energy {infoA[2]:.4f}-{infoA[3]:.3f} "
          f"(span {infoA[3]/infoA[2]:.0f}x)")
    for ml, (ib, sb, n) in infoR.items():
        print(f"  {ml}: iso-best={ib}  srv-best={sb}  n={n}")
    print(f"savings n={infoS[0]} median {infoS[1]:.1f}% max {infoS[2]:.1f}%")
    with open(os.path.join(HERE, "fleet_cloud.tex"), "w") as fh:
        fh.write(texA)
    with open(os.path.join(HERE, "ranking_flip.tex"), "w") as fh:
        fh.write(texR)
    with open(os.path.join(HERE, "savings_dist.tex"), "w") as fh:
        fh.write(texS)
    print("wrote fleet_cloud.tex, ranking_flip.tex, savings_dist.tex")
