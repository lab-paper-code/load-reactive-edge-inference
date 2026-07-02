#!/usr/bin/env python3
"""Generate the full result tables (LaTeX) from the official derived artifacts.

Single source of truth, no re-derivation:
  Table A  raw measurement grid   <- full_dvfs_lambda_cell_summary.csv (batch_size==1)
  Table B  oracle decision grid   <- scheduler_policy_eval.csv

Output: oracle_full_tables.tex (standalone, landscape longtable), pdflatex-able.
"""
import csv, os

DERIVED = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
OUT = os.path.join(os.path.dirname(__file__), "oracle_full_tables.tex")

SEG_OF = {
    "gpu-server": "Server",
    "orin": "Jetson", "orin-nano": "Jetson", "xavier": "Jetson", "jetson": "Jetson",
    "orangepi-npu": "SBC", "orangepi": "SBC", "rasp5": "SBC", "lattepanda": "SBC",
}
SEG_ORDER = {"Server": 0, "Jetson": 1, "SBC": 2}
DEV_NAME = {
    "gpu-server": "GPU Server", "orin": "AGX Orin", "orin-nano": "Orin Nano",
    "xavier": "Xavier NX", "jetson": "Jetson Nano", "orangepi-npu": "OrangePi NPU",
    "orangepi": "OrangePi CPU", "rasp5": "Raspberry Pi 5", "lattepanda": "LattePanda",
}
MODEL_NAME = {
    "mobilenet-v2-050": "mobilenet-v2-050",
    "mobilenet-v2-100": "mobilenet-v2-100",
    "efficientnet-b4": "efficientnet-b4",
}
LOADS = ["0.25", "0.5", "0.75", "1.0"]
SLA = 100.0


def esc(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fnum(x, nd=3, dash="--"):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return dash


def inum(x, dash="--"):
    try:
        return f"{round(float(x)):d}"
    except (TypeError, ValueError):
        return dash


def load_csv(name):
    with open(os.path.join(DERIVED, name)) as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------------
# Table A: raw measurement grid (per device/model/mode x 4 own-capacity loads)
# ----------------------------------------------------------------------------
def table_a():
    rows = [r for r in load_csv("full_dvfs_lambda_cell_summary.csv")
            if r.get("batch_size") in ("1", "1.0")]
    # index by (device, model, mode) -> {lambda_frac: row}
    cells = {}
    cap = {}
    for r in rows:
        key = (r["device"], r["model"], r["dvfs_mode_label"])
        cells.setdefault(key, {})[r["lambda_frac"]] = r
        cap[key] = r["capacity_ips"]

    def sortkey(k):
        dev, model, mode = k
        seg = SEG_OF[dev]
        try:
            c = float(cap[k])
        except ValueError:
            c = 0.0
        return (SEG_ORDER[seg], dev, model, c)

    out = []
    out.append(r"\begin{center}\footnotesize")
    out.append(r"\begin{longtable}{@{}lll r *{4}{r r}@{}}")
    out.append(r"\caption{Table A. Full raw measurement grid: every measured "
               r"device--model--mode at four request loads expressed as a fraction "
               r"of that mode's own confirmed capacity. Cells report "
               r"AC input energy (J/inf, median) and p95 latency (ms, median). "
               r"Gray p95 marks a cell over the 100\,ms deadline. "
               r"Source: \texttt{full\_dvfs\_lambda\_cell\_summary.csv} (batch\_size=1), "
               r"99 modes $\times$ 4 loads = 396 measured load-cells.}\\")
    out.append(r"\toprule")
    out.append(r"\multicolumn{4}{@{}l}{} & \multicolumn{2}{c}{load 0.25} & "
               r"\multicolumn{2}{c}{load 0.50} & \multicolumn{2}{c}{load 0.75} & "
               r"\multicolumn{2}{c}{load 1.00} \\")
    out.append(r"Segment & Device / Model & Mode & Capacity (ips) & "
               + " & ".join([r"J/inf & p95 (ms)"] * 4) + r" \\")
    out.append(r"\midrule\endfirsthead")
    out.append(r"\toprule Segment & Device / Model & Mode & Capacity (ips) & "
               + " & ".join([r"J/inf & p95 (ms)"] * 4) + r" \\\midrule\endhead")

    prev_seg = prev_dm = None
    for k in sorted(cells, key=sortkey):
        dev, model, mode = k
        seg = SEG_OF[dev]
        dm = f"{DEV_NAME[dev]} / {MODEL_NAME.get(model, model)}"
        seg_cell = esc(seg) if seg != prev_seg else ""
        dm_cell = esc(dm) if dm != prev_dm else ""
        if dm != prev_dm and prev_dm is not None:
            out.append(r"\addlinespace[2pt]")
        prev_seg, prev_dm = seg, dm
        cellvals = []
        for lf in LOADS:
            r = cells[k].get(lf)
            if r is None:
                cellvals += ["--", "--"]
                continue
            jinf = fnum(r["marginal_wall_energy_j_per_inf_median"], 3)
            p95v = r["p95_latency_ms_median"]
            p95s = inum(p95v)
            try:
                if float(p95v) > SLA:
                    p95s = r"\textcolor{gray}{" + p95s + "}"
            except (TypeError, ValueError):
                pass
            cellvals += [jinf, p95s]
        out.append(f"{seg_cell} & {dm_cell} & {esc(mode)} & {inum(cap[k],dash='--')} & "
                   + " & ".join(cellvals) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{longtable}\end{center}")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Table B: oracle decision grid (per device/model x 4 external-demand loads)
# ----------------------------------------------------------------------------
def table_b():
    rows = load_csv("scheduler_policy_eval.csv")

    def sortkey(r):
        dev = r["device"]
        return (SEG_ORDER[SEG_OF[dev]], dev, r["model"], float(r["demand_frac_of_group_max"]))

    # detect reversal per (device, model): oracle label changes across demand
    by_group = {}
    for r in rows:
        by_group.setdefault((r["device"], r["model"]), []).append(r)
    reversal = {}
    for g, rs in by_group.items():
        labs = [x["dynamic_dvfs_mode_label"] for x in
                sorted(rs, key=lambda x: float(x["demand_frac_of_group_max"]))
                if x.get("dynamic_dvfs_mode_label")]
        reversal[g] = len(set(labs)) > 1

    hdr_group = (r"& & \multicolumn{3}{c}{modes} "
                 r"& \multicolumn{4}{c}{Oracle (feasible min-energy)} "
                 r"& \multicolumn{3}{c}{Max-performance proxy} "
                 r"& \multicolumn{3}{c}{Capacity-only proxy} & & \\")
    hdr_rules = r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}\cmidrule(lr){10-12}\cmidrule(lr){13-15}"
    hdr_cols = (r"Device / Model & Load & modes & feasible & SLA-safe "
                r"& mode & category & J/inf & p95 (ms) "
                r"& mode & J/inf & p95 (ms) "
                r"& mode & J/inf & p95 (ms) "
                r"& regret vs max-perf & robustness \\")

    out = []
    out.append(r"\begin{center}\tiny")
    out.append(r"\setlength{\tabcolsep}{2pt}")
    out.append(r"\begin{longtable}{@{}l l r r r l l r r l r r l r r r l@{}}")
    out.append(r"\caption{Table B. Full oracle decision grid: every device--model "
               r"group at four external request loads (fraction of the group's max "
               r"capacity, the figure's load axis). \emph{feasible} counts modes whose "
               r"capacity sustains the demand; \emph{SLA-safe} counts those that also "
               r"keep p95 $\le 100$\,ms (the oracle chooses among SLA-safe modes, so a "
               r"genuine choice exists where SLA-safe $> 1$). For the oracle, the "
               r"maximum-performance proxy, and the capacity-only proxy the table lists "
               r"mode, estimated J/inf and p95 (ms); regret is the oracle's energy "
               r"saving over the max-performance proxy. $\dagger$ marks an SLA-unsafe "
               r"selection (p95 over 100\,ms); $\star$ marks a group whose oracle mode "
               r"changes with load. Source: \texttt{scheduler\_policy\_eval.csv}, "
               r"22 groups $\times$ 4 loads = 88 rows.}\\")
    out.append(r"\toprule")
    out.append(hdr_group)
    out.append(hdr_rules)
    out.append(hdr_cols)
    out.append(r"\midrule\endfirsthead")
    out.append(r"\toprule")
    out.append(hdr_group)
    out.append(hdr_rules)
    out.append(hdr_cols)
    out.append(r"\midrule\endhead")

    prev_dm = None
    for r in sorted(rows, key=sortkey):
        dev, model = r["device"], r["model"]
        g = (dev, model)
        dm = f"{DEV_NAME[dev]} / {MODEL_NAME.get(model, model)}"
        star = r"$\star$" if reversal.get(g) else ""
        dm_cell = (esc(dm) + star) if dm != prev_dm else ""
        if dm != prev_dm and prev_dm is not None:
            out.append(r"\addlinespace[2pt]")
        prev_dm = dm
        load = f'{float(r["demand_frac_of_group_max"]):.2f}'

        def safe_mark(flag):
            return "" if flag in ("True", "1", "true") else r"\,$\dagger$"

        ora_mode = esc(r["dynamic_dvfs_mode_label"]) if r["dynamic_dvfs_mode_label"] else r"\textcolor{gray}{none}"
        ora_cat = esc(r.get("dynamic_mode_category", ""))
        ora_j = fnum(r["dynamic_energy_j_per_inf_est"], 3)
        ora_p = inum(r["dynamic_p95_latency_ms_est"]) + safe_mark(r.get("dynamic_sla_safe", ""))
        mp_mode = esc(r["max_perf_dvfs_mode_label"]) if r["max_perf_dvfs_mode_label"] else "--"
        mp_j = fnum(r["max_perf_energy_j_per_inf_est"], 3)
        mp_p = inum(r["max_perf_p95_latency_ms_est"]) + safe_mark(r.get("max_perf_sla_safe", ""))
        cap_mode = esc(r["capacity_only_dvfs_mode_label"]) if r["capacity_only_dvfs_mode_label"] else "--"
        cap_j = fnum(r["capacity_only_energy_j_per_inf_est"], 3)
        cap_p = inum(r["capacity_only_p95_latency_ms_est"]) + safe_mark(r.get("capacity_only_sla_safe", ""))
        regret = r.get("dynamic_energy_saving_vs_max_perf_pct", "")
        regret_s = fnum(regret, 1, dash="--")
        if regret_s not in ("--",):
            regret_s = regret_s + r"\%"
        robust = esc(r.get("decision_robustness", ""))

        out.append(f"{dm_cell} & {load} & {esc(r['n_modes'])} & {esc(r['n_feasible_modes'])} "
                   f"& {esc(r['n_sla_safe_modes'])} & "
                   f"{ora_mode} & {ora_cat} & {ora_j} & {ora_p} & "
                   f"{mp_mode} & {mp_j} & {mp_p} & "
                   f"{cap_mode} & {cap_j} & {cap_p} & {regret_s} & {robust} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{longtable}\end{center}")
    return "\n".join(out)


def main():
    doc = []
    doc.append(r"\documentclass[10pt]{article}")
    doc.append(r"\usepackage[landscape,margin=1cm]{geometry}")
    doc.append(r"\usepackage{longtable,booktabs,array,xcolor}")
    doc.append(r"\setlength{\tabcolsep}{3pt}")
    doc.append(r"\renewcommand{\arraystretch}{1.05}")
    doc.append(r"\begin{document}")
    doc.append(r"\section*{Full result tables (working artifact)}")
    doc.append(r"\noindent Generated from official derived artifacts with no re-derivation. "
               r"Deadline p95 $\le 100$\,ms; loads $\{0.25,0.5,0.75,1.0\}$; AC input energy. "
               r"$\dagger$ = SLA-unsafe (p95 over deadline). "
               r"Table~A load = fraction of each mode's own capacity (raw measurement); "
               r"Table~B load = fraction of the group's max capacity (the oracle-sheet axis).\par\medskip")
    doc.append(table_b())
    doc.append(r"\clearpage")
    doc.append(table_a())
    doc.append(r"\end{document}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(doc) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
