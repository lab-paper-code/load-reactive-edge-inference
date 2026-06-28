"""
Build T2 central artifact: capacity + serving-load power + CI per pair.

# T2: internal artifact stage label for the frozen capacity-power table.
Central empirical artifact: capacity and power per device/model. It joins:
  - capacity_measured.csv / capacity_dvfs_overrides.csv (capacity point estimate)
  - serving_power_measured.csv (serving-load power)
  - capacity_power_ci.csv (CI bounds from confirmation runs)
  - constants.py ACCURACY, SLO_INFEASIBLE (metadata)

Output:
  - results/derived/t2_capacity_power.csv, one row per (device, model, policy)
  - results/derived/t2_summary.json, fleet-level statistics

Freeze rule: once produced and validated, no more re-measurement. Subsequent
paper analysis operates on the frozen T2 artifact only.

Usage:
    python3 build_t2_artifact.py           # dry-run
    python3 build_t2_artifact.py --apply   # write CSVs + summary
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from _util import POWER_EXCLUDED_DEVICES, atomic_csv_write, atomic_json_write
from constants import (ACCURACY, MODELS, SLO_INFEASIBLE, MEASURED_CAPACITY_EINC,
                       MEASURED_CAPACITY_CAP, DEVICES)

RESULTS_DIR = Path(__file__).parent / "results"
CAPACITY_CSV = RESULTS_DIR / "derived" / "capacity_measured.csv"
CAPACITY_OVERRIDES_CSV = RESULTS_DIR / "derived" / "capacity_dvfs_overrides.csv"
SERVING_POWER_CSV = RESULTS_DIR / "derived" / "serving_power_measured.csv"
CI_CSV = RESULTS_DIR / "derived" / "capacity_power_ci.csv"
T2_CSV = RESULTS_DIR / "derived" / "t2_capacity_power.csv"
T2_SUMMARY_JSON = RESULTS_DIR / "derived" / "t2_summary.json"


def load_csv_by_key(path: Path, key_cols: list[str]) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = tuple(row[c] for c in key_cols)
            out[key] = row
    return out


def _f(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _load_bs1_power(path: Path) -> dict[tuple, dict]:
    """serving_power rows restricted to batch_size==1, keyed by
    (device, model, policy). T2 is a bs=1 capacity-power anchor: bs>1 rows
    must never widen this join (a plain (d,m,p) key would last-write-win onto
    a bs>1 row). Legacy rows with no batch_size column are treated as bs=1."""
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if str(row.get("batch_size") or "1") != "1":
                continue
            out[(row["device"], row["model"], row["policy"])] = row
    return out


def build_t2(policy: str) -> list[dict]:
    """Build one T2 row per (device, model) for the given policy.

    capacity_ips / capacity_ci_min / capacity_ci_max are all derived from the
    same statistic: the achieved-RPS distribution across the n=3 confirmation
    rounds (median / min / max). This keeps the point estimate inside its own
    range by construction. For SLO-infeasible pairs or pairs missing CI data,
    we fall back to the binary-search-confirmed capacity in MEASURED_CAPACITY_*.
    """
    capacity_cap = MEASURED_CAPACITY_CAP if policy == "capacity" else MEASURED_CAPACITY_EINC
    # bs=1-only join: T2 is the frozen bs=1 anchor; bs>1 serving-power rows
    # (Phase-1 batch extension) must not widen or shadow this lookup.
    power_rows = _load_bs1_power(SERVING_POWER_CSV)
    ci_rows = load_csv_by_key(CI_CSV, ["device", "model", "policy"])

    # T2 must present a complete (device, model) grid. Iterating only
    # MEASURED_CAPACITY_* keys silently drops SLO-infeasible pairs that aren't
    # in those dicts, leaving undocumented gaps in the evidence surface.
    # Emit an explicit row for every (device, model) the harness knows about.
    grid_pairs = sorted(
        {(d, m) for d in DEVICES for m in MODELS if d not in POWER_EXCLUDED_DEVICES}
    )

    out = []
    for (device, model) in grid_pairs:
        binary_search_cap = capacity_cap.get((device, model))
        slo_infeasible = (device, model) in SLO_INFEASIBLE
        p = power_rows.get((device, model, policy)) or {}
        c = ci_rows.get((device, model, policy)) or {}

        # warm_idle: server-loaded idle baseline power (watts), measured before
        # inference traffic starts.  delta_w = avg_serving_power - warm_idle.
        warm_idle = _f(p.get("warm_idle_power_w"))
        serving = _f(p.get("avg_serving_power_w"))
        delta = _f(p.get("delta_power_w"))

        capacity_median = _f(c.get("capacity_median_ips"))
        capacity_min = _f(c.get("capacity_min_ips"))
        capacity_max = _f(c.get("capacity_max_ips"))
        # Prefer empirical median (same source as the range); fall back to the
        # binary-search-confirmed number only when CI data is missing.
        capacity_point = capacity_median if capacity_median is not None else binary_search_cap

        # Recompute marginal wall energy against the T2 point estimate so the
        # row is algebraically coherent: E = delta_w / capacity_point. The
        # pre-computed value in serving_power_measured.csv is anchored to the
        # binary-search mu, which differs from the CI median for some rows.
        if delta is not None and capacity_point is not None and capacity_point > 0:
            energy = round(delta / capacity_point, 4)
        else:
            energy = None

        out.append({
            "device": device,
            "model": model,
            "policy": policy,
            "accuracy": ACCURACY.get(model, ""),
            "capacity_ips": capacity_point,
            "capacity_ci_min": capacity_min,
            "capacity_ci_max": capacity_max,
            "capacity_binary_search_ips": binary_search_cap,
            "p95_median_ms": _f(c.get("p95_median_ms")),
            "p95_ci_max": _f(c.get("p95_max_ms")),
            "warm_idle_w": warm_idle,
            "serving_w": serving,
            "delta_w": delta,
            "marginal_wall_energy_j_per_inf": energy,
            "n_conf_runs": _f(c.get("n_runs")),
            "slo_infeasible": slo_infeasible,
            "power_source": p.get("source_json", ""),
        })
    return out


def compute_summary(rows_einc: list[dict], rows_cap: list[dict]) -> dict:
    def agg(rows, field, where=None):
        vals = []
        for r in rows:
            if where and not where(r):
                continue
            v = r.get(field)
            if v is not None:
                vals.append(v)
        if not vals:
            return {}
        return {
            "n": len(vals),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "median": round(statistics.median(vals), 3),
        }

    feasible = lambda r: not r["slo_infeasible"]
    has_power = lambda r: r.get("warm_idle_w") is not None

    return {
        "policy_e_inc": {
            "n_pairs": len(rows_einc),
            "n_feasible": sum(1 for r in rows_einc if feasible(r)),
            "n_with_power": sum(1 for r in rows_einc if has_power(r)),
            "capacity_ips": agg(rows_einc, "capacity_ips", feasible),
            "warm_idle_w": agg(rows_einc, "warm_idle_w", lambda r: feasible(r) and has_power(r)),
            "serving_w": agg(rows_einc, "serving_w", lambda r: feasible(r) and has_power(r)),
            "delta_w": agg(rows_einc, "delta_w", lambda r: feasible(r) and has_power(r)),
            "marginal_wall_energy_j_per_inf": agg(rows_einc, "marginal_wall_energy_j_per_inf", lambda r: feasible(r) and has_power(r)),
        },
        "policy_capacity": {
            "n_pairs": len(rows_cap),
            "n_feasible": sum(1 for r in rows_cap if feasible(r)),
            "n_with_power": sum(1 for r in rows_cap if has_power(r)),
            "capacity_ips": agg(rows_cap, "capacity_ips", feasible),
            "warm_idle_w": agg(rows_cap, "warm_idle_w", lambda r: feasible(r) and has_power(r)),
            "serving_w": agg(rows_cap, "serving_w", lambda r: feasible(r) and has_power(r)),
            "delta_w": agg(rows_cap, "delta_w", lambda r: feasible(r) and has_power(r)),
            "marginal_wall_energy_j_per_inf": agg(rows_cap, "marginal_wall_energy_j_per_inf", lambda r: feasible(r) and has_power(r)),
        },
        "excluded_devices": sorted(POWER_EXCLUDED_DEVICES),
        "excluded_reasons": {d: "no Shelly plug" for d in sorted(POWER_EXCLUDED_DEVICES)},
        "slo_threshold_ms": 100.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    atomic_csv_write(path, list(rows[0].keys()), rows)


def write_json(path: Path, payload) -> None:
    atomic_json_write(path, payload)


def main():
    parser = argparse.ArgumentParser(description="Build T2 central artifact")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    print("=" * 100)
    print("T2 Policy A (e_inc)")
    print("=" * 100)
    rows_einc = build_t2("e_inc")
    for r in rows_einc:
        tag = " [SLO-INFEASIBLE]" if r["slo_infeasible"] else ""
        has_pw = "PW" if r["warm_idle_w"] is not None else "--"
        has_ci = "CI" if r["capacity_ci_min"] is not None else "--"
        cap = f"{r['capacity_ips']:>7.1f}" if r["capacity_ips"] is not None else "     --"
        print(f"  {r['device']:<13} {r['model']:<20} cap={cap} "
              f"idle={r['warm_idle_w']} serving={r['serving_w']} "
              f"delta={r['delta_w']} E={r['marginal_wall_energy_j_per_inf']} "
              f"[{has_pw}/{has_ci}]{tag}")

    print()
    print("=" * 100)
    print("T2 Policy B (capacity)")
    print("=" * 100)
    rows_cap = build_t2("capacity")

    summary = compute_summary(rows_einc, rows_cap)
    print()
    print("=" * 60)
    print("Fleet Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    if args.apply:
        write_csv(T2_CSV, rows_einc + rows_cap)
        print(f"\nWritten: {T2_CSV} ({len(rows_einc) + len(rows_cap)} rows)")
        write_json(T2_SUMMARY_JSON, summary)
        print(f"Written: {T2_SUMMARY_JSON}")
    else:
        print(f"\nDry run. Use --apply to write.")


if __name__ == "__main__":
    main()
