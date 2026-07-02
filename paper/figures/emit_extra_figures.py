#!/usr/bin/env python3
"""Emit two native-pgfplots figures for paper.tex from the released artifacts.
No re-derivation; Table A is batch_size==1 only.

  fleet_cloud.tex    Figure 1: EDP efficiency (inf/J) vs load group (% of MST),
                     one panel per model, colored by device type; filled marker =
                     SLO-feasible, open = infeasible. From the 396 measured cells.
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


LOAD_RANK = {0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}
# Per-device-type horizontal offset within a load group, and half-width of the
# deterministic within-group spread (points ordered by y). Cosmetic jitter only;
# the y value, load group, and feasibility are the measured data.
SEG_XOFF = {"Server": -0.24, "Jetson": -0.08, "SB-NPU": 0.08, "SB-CPU": 0.24}
FLEET_HW = 0.037
FLEET_TITLES = {"mobilenet-v2-050": "MobileNet-V2-050",
                "mobilenet-v2-100": "MobileNet-V2-100",
                "efficientnet-b4": "EfficientNet-B4"}
FLEET_LEGEND = ["SB-CPU", "SB-NPU", "Jetson", "Server"]


def fig_fleet_cloud(rows):
    """Figure 1: EDP efficiency (inf/J) vs load group (% of MST), one panel per
    model, colored by device type; filled marker = SLO-feasible, open = infeasible.
    EDP efficiency = 100 / (AC input energy per inf * p95 latency). Points within
    each (model, load group, device type) are spread horizontally, ordered by y."""
    per_model = defaultdict(lambda: defaultdict(list))
    n_pts = n_infeas = 0
    for seg, model, ld, p95, energy in rows:
        rank = LOAD_RANK[round(ld, 2)]
        y = DEADLINE / (energy * p95)
        feasible = p95 <= DEADLINE
        per_model[model][(rank, seg)].append((y, feasible))
        n_pts += 1
        n_infeas += 0 if feasible else 1
    L = [colordefs(FLEET_SEGS)]
    a = L.append
    leg = r" \qquad ".join(
        rf"\tikz[baseline=-0.55ex]\draw[draw=white, line width=0.12pt, "
        rf"fill={SEG_KEY[s]}] (0,0) circle (2.25pt);\,{SEG_LABEL[s]}"
        for s in FLEET_LEGEND)
    a(rf"{{\footnotesize\centering \textbf{{Device type:}}\quad {leg}\par}}")
    a(r"\vspace{2pt}")
    a(r"\begin{tikzpicture}")
    a(r"\begin{groupplot}[")
    a(r"  group style={group size=3 by 1, horizontal sep=0.20cm,")
    a(r"    ylabels at=edge left, yticklabels at=edge left},")
    a(r"  width=0.334\textwidth, height=0.285\textwidth, clip=true,")
    a(r"  ymode=log,")
    a(r"  ylabel={EDP efficiency (inf/J)},")
    a(r"  xmin=0.55, xmax=4.45, ymin=0.45, ymax=3000.0,")
    a(r"  xtick={1,2,3,4}, xticklabels={25,50,75,100},")
    a(r"  ytick={1,10,100,1000}, yticklabels={1,10,100,1000},")
    a(r"  minor tick num=1,")
    a(r"  ymajorgrids=true, grid style={black!12, densely dotted},")
    a(r"  axis line style={black!60}, tick style={black!55},")
    a(r"  title style={font=\small, yshift=-1pt},")
    a(r"  label style={font=\small}, tick label style={font=\footnotesize},")
    a(r"  ylabel style={yshift=-2pt}, xlabel style={yshift=2pt},")
    a(r"]")
    for mkey, _ in MODELS:
        head = rf"\nextgroupplot[title={{{FLEET_TITLES[mkey]}}}"
        if mkey == "mobilenet-v2-100":
            head += r", xlabel={Load group (\% of MST)}"
        a(head + "]")
        placed = defaultdict(lambda: {"filled": [], "open": []})
        for (rank, seg), pts in per_model[mkey].items():
            pts.sort(key=lambda t: t[0])
            n = len(pts)
            center = rank + SEG_XOFF[seg]
            for i, (y, feasible) in enumerate(pts):
                x = center if n == 1 else center - FLEET_HW + 2 * FLEET_HW * i / (n - 1)
                placed[seg]["filled" if feasible else "open"].append((x, y))
        for seg in FLEET_SEGS:  # filled markers first (the cloud)
            fld = sorted(placed[seg]["filled"])
            if fld:
                coords = " ".join(f"({x:.3f},{y:.6f})" for x, y in fld)
                a(rf"\addplot[only marks, mark=*, mark size=2.0pt, mark options="
                  rf"{{fill={SEG_KEY[seg]}, fill opacity=0.82, draw=white, "
                  rf"line width=0.16pt}}, forget plot] coordinates {{{coords}}};")
        for seg in FLEET_SEGS:  # infeasible (open) markers on top
            opn = sorted(placed[seg]["open"])
            if opn:
                coords = " ".join(f"({x:.3f},{y:.6f})" for x, y in opn)
                a(rf"\addplot[only marks, mark=*, mark size=2.0pt, mark options="
                  rf"{{fill=white, draw={SEG_KEY[seg]}, line width=0.55pt}}, "
                  rf"forget plot] coordinates {{{coords}}};")
    a(r"\end{groupplot}")
    a(r"\end{tikzpicture}\par")
    return "\n".join(L) + "\n", (n_pts, n_infeas)


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
    a(r"  ylabel={AC input energy savings (\%)},")
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
    print(f"fleet_cloud: {infoA[0]} points, {infoA[1]} infeasible (open markers)")
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
