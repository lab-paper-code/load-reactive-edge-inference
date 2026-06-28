"""Build first-pass full-DVFS analysis tables.

This script consumes the mode-keyed full-DVFS artifacts only. It does not read
or mutate the policy-keyed T2/T5 artifacts.

Inputs:
  results/derived/full_dvfs_capacity.csv
  results/derived/full_dvfs_capacity_summary.json
  results/derived/full_dvfs_lambda_sweep.csv
  results/derived/full_dvfs_lambda_cell_summary.csv
  results/derived/full_dvfs_lambda_summary.json

Outputs:
  results/derived/full_dvfs_analysis_cells.csv
  results/derived/full_dvfs_pareto.csv
  results/derived/full_dvfs_analysis_summary.json

Usage:
  python3 analyze_full_dvfs.py
  python3 analyze_full_dvfs.py --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from _util import atomic_csv_write, atomic_json_write
from constants import L_MAX_MS

RESULTS_DIR = Path(__file__).parent / "results" / "derived"

CAPACITY_CSV = RESULTS_DIR / "full_dvfs_capacity.csv"
CAPACITY_SUMMARY_JSON = RESULTS_DIR / "full_dvfs_capacity_summary.json"
LAMBDA_SWEEP_CSV = RESULTS_DIR / "full_dvfs_lambda_sweep.csv"
LAMBDA_CELL_CSV = RESULTS_DIR / "full_dvfs_lambda_cell_summary.csv"
LAMBDA_SUMMARY_JSON = RESULTS_DIR / "full_dvfs_lambda_summary.json"

ANALYSIS_CSV = RESULTS_DIR / "full_dvfs_analysis_cells.csv"
PARETO_CSV = RESULTS_DIR / "full_dvfs_pareto.csv"
SUMMARY_JSON = RESULTS_DIR / "full_dvfs_analysis_summary.json"

LOW_DELTA_W = 0.5


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return [dict(r) for r in csv.DictReader(f)]


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _float(row: dict, key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return val


def _int(row: dict, key: str) -> int | None:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _key(row: dict) -> tuple[str, str, int]:
    return (row["device"], row["model"], int(row["dvfs_mode"]))


def _cell_key(row: dict) -> tuple[str, str, int, float]:
    return (row["device"], row["model"], int(row["dvfs_mode"]),
            float(row["lambda_frac"]))


def _round(val: float | None, digits: int = 6) -> float | None:
    return round(val, digits) if val is not None else None


def _median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return num / den


def _run_quality_by_cell(run_rows: list[dict]) -> dict[tuple, dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[_cell_key(row)].append(row)

    out = {}
    for key, rows in grouped.items():
        p95s = [_float(r, "p95_latency_ms") for r in rows]
        deltas = [_float(r, "delta_w") for r in rows]
        energies = [_float(r, "marginal_wall_energy_j_per_inf") for r in rows]
        p95s_clean = [v for v in p95s if v is not None]
        deltas_clean = [v for v in deltas if v is not None]
        energies_clean = [v for v in energies if v is not None]
        out[key] = {
            "n_run_rows": len(rows),
            "n_runs_p95_gt_lmax": sum(v > L_MAX_MS for v in p95s_clean),
            "n_runs_delta_le_zero": sum(v <= 0 for v in deltas_clean),
            "n_runs_delta_lt_low": sum(v < LOW_DELTA_W for v in deltas_clean),
            "n_runs_energy_le_zero": sum(v <= 0 for v in energies_clean),
            "run_p95_median": _round(_median(p95s_clean), 6),
            "run_delta_median": _round(_median(deltas_clean), 6),
            "run_energy_median": _round(_median(energies_clean), 6),
        }
    return out


def build_analysis_rows() -> tuple[list[dict], list[dict], dict]:
    capacity_rows = _read_csv(CAPACITY_CSV)
    lambda_cell_rows = _read_csv(LAMBDA_CELL_CSV)
    lambda_run_rows = _read_csv(LAMBDA_SWEEP_CSV)
    capacity_summary = _read_json(CAPACITY_SUMMARY_JSON)
    lambda_summary = _read_json(LAMBDA_SUMMARY_JSON)

    capacity_by_key = {_key(r): r for r in capacity_rows}
    run_quality = _run_quality_by_cell(lambda_run_rows)

    rows: list[dict] = []
    for cell in sorted(lambda_cell_rows,
                       key=lambda r: (r["device"], r["model"],
                                      int(r["dvfs_mode"]),
                                      float(r["lambda_frac"]))):
        key = _key(cell)
        cap = capacity_by_key.get(key)
        if cap is None:
            raise RuntimeError(f"lambda cell without capacity anchor: {key}")

        lambda_frac = _float(cell, "lambda_frac")
        target_rps = _float(cell, "target_rps")
        achieved_rps = _float(cell, "achieved_rps_median")
        p95_median = _float(cell, "p95_latency_ms_median")
        p95_max = _float(cell, "p95_latency_ms_max")
        delta_median = _float(cell, "delta_w_median")
        energy_median = _float(cell, "marginal_wall_energy_j_per_inf_median")
        quality = run_quality.get(_cell_key(cell), {})

        row = {
            "device": cell["device"],
            "model": cell["model"],
            "dvfs_mode": int(cell["dvfs_mode"]),
            "dvfs_mode_label": cell.get("dvfs_mode_label", ""),
            "condition_tag": cell.get("condition_tag", ""),
            "segment": cell.get("segment", ""),
            "lambda_frac": lambda_frac,
            "capacity_ips": _float(cap, "capacity_ips"),
            "capacity_selection": cap.get("capacity_selection", ""),
            "capacity_power_status": cap.get("power_status", ""),
            "capacity_delta_w": _float(cap, "delta_w"),
            "target_rps": target_rps,
            "achieved_rps_median": achieved_rps,
            "achieved_target_ratio": _round(
                _safe_ratio(achieved_rps, target_rps), 6),
            "n_runs": _int(cell, "n_runs"),
            "p95_latency_ms_median": p95_median,
            "p95_latency_ms_max": p95_max,
            "warm_idle_w_median": _float(cell, "warm_idle_w_median"),
            "serving_w_median": _float(cell, "serving_w_median"),
            "delta_w_median": delta_median,
            "marginal_wall_energy_j_per_inf_median": energy_median,
            "marginal_wall_energy_j_per_inf_min": _float(
                cell, "marginal_wall_energy_j_per_inf_min"),
            "marginal_wall_energy_j_per_inf_max": _float(
                cell, "marginal_wall_energy_j_per_inf_max"),
            "flag_slo_median_violation": int(
                p95_median is not None and p95_median > L_MAX_MS),
            "flag_slo_any_run_violation": int(
                quality.get("n_runs_p95_gt_lmax", 0) > 0),
            "flag_low_delta_median": int(
                delta_median is not None and delta_median < LOW_DELTA_W),
            "flag_nonpositive_delta_run": int(
                quality.get("n_runs_delta_le_zero", 0) > 0),
            "flag_nonpositive_energy_run": int(
                quality.get("n_runs_energy_le_zero", 0) > 0),
            "n_runs_p95_gt_lmax": quality.get("n_runs_p95_gt_lmax", 0),
            "n_runs_delta_le_zero": quality.get("n_runs_delta_le_zero", 0),
            "n_runs_delta_lt_low": quality.get("n_runs_delta_lt_low", 0),
            "n_runs_energy_le_zero": quality.get("n_runs_energy_le_zero", 0),
            "source_capacity_json": cap.get("source_json", ""),
        }
        rows.append(row)

    pareto_rows = _pareto_rows(rows)
    summary = _summary(rows, pareto_rows, capacity_summary, lambda_summary)
    return rows, pareto_rows, summary


def _pareto_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["device"], row["model"], row["lambda_frac"])].append(row)

    out: list[dict] = []
    for group_key, candidates in sorted(grouped.items()):
        valid = [
            r for r in candidates
            if r["marginal_wall_energy_j_per_inf_median"] is not None
            and r["p95_latency_ms_median"] is not None
        ]
        energy_sorted = sorted(
            valid,
            key=lambda r: (r["marginal_wall_energy_j_per_inf_median"],
                           r["p95_latency_ms_median"], r["dvfs_mode"]),
        )
        energy_rank = {
            id(r): idx + 1
            for idx, r in enumerate(energy_sorted)
        }
        for row in candidates:
            dominated_by: list[str] = []
            e = row["marginal_wall_energy_j_per_inf_median"]
            p95 = row["p95_latency_ms_median"]
            if e is not None and p95 is not None:
                for other in valid:
                    if other is row:
                        continue
                    oe = other["marginal_wall_energy_j_per_inf_median"]
                    op95 = other["p95_latency_ms_median"]
                    if oe <= e and op95 <= p95 and (oe < e or op95 < p95):
                        dominated_by.append(
                            f"dvfs{other['dvfs_mode']}:{oe:.6f}J,{op95:.3f}ms")
            out.append({
                "device": row["device"],
                "model": row["model"],
                "lambda_frac": row["lambda_frac"],
                "dvfs_mode": row["dvfs_mode"],
                "dvfs_mode_label": row["dvfs_mode_label"],
                "segment": row["segment"],
                "capacity_ips": row["capacity_ips"],
                "p95_latency_ms_median": row["p95_latency_ms_median"],
                "marginal_wall_energy_j_per_inf_median": (
                    row["marginal_wall_energy_j_per_inf_median"]),
                "delta_w_median": row["delta_w_median"],
                "flag_slo_median_violation": row["flag_slo_median_violation"],
                "flag_low_delta_median": row["flag_low_delta_median"],
                "energy_rank_within_device_model_lambda": energy_rank.get(id(row)),
                "pareto_e_latency": int(not dominated_by),
                "dominated_by": ";".join(dominated_by),
                "n_modes_in_group": len(valid),
            })
    return sorted(out, key=lambda r: (r["device"], r["model"],
                                      r["lambda_frac"], r["dvfs_mode"]))


def _counter(rows: list[dict], field: str) -> dict:
    return dict(sorted(Counter(str(r.get(field, "")) for r in rows).items()))


def _flag_count(rows: list[dict], field: str) -> int:
    return sum(int(r.get(field) or 0) for r in rows)


def _summary(rows: list[dict], pareto_rows: list[dict],
             capacity_summary: dict, lambda_summary: dict) -> dict:
    invalid_rows = capacity_summary.get("invalid_rows", [])
    coverage = lambda_summary.get("coverage", {})
    capacity_cells = {
        (r["device"], r["model"], r["dvfs_mode"])
        for r in rows
    }

    by_model = defaultdict(list)
    by_segment = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
        by_segment[row["segment"]].append(row)

    def _group_summary(grouped: dict[str, list[dict]]) -> dict:
        return {
            key: {
                "lambda_cells": len(vals),
                "capacity_cells": len({
                    (r["device"], r["model"], r["dvfs_mode"])
                    for r in vals
                }),
                "slo_median_violation_cells": _flag_count(
                    vals, "flag_slo_median_violation"),
                "slo_any_run_violation_cells": _flag_count(
                    vals, "flag_slo_any_run_violation"),
                "low_delta_median_cells": _flag_count(
                    vals, "flag_low_delta_median"),
                "nonpositive_delta_cells": _flag_count(
                    vals, "flag_nonpositive_delta_run"),
            }
            for key, vals in sorted(grouped.items())
        }

    pareto_frontier_count = sum(r["pareto_e_latency"] for r in pareto_rows)
    dominated_count = len(pareto_rows) - pareto_frontier_count

    return {
        "artifact": "full_dvfs_analysis",
        "inputs": {
            "capacity_csv": str(CAPACITY_CSV),
            "capacity_summary_json": str(CAPACITY_SUMMARY_JSON),
            "lambda_sweep_csv": str(LAMBDA_SWEEP_CSV),
            "lambda_cell_csv": str(LAMBDA_CELL_CSV),
            "lambda_summary_json": str(LAMBDA_SUMMARY_JSON),
        },
        "outputs": {
            "analysis_cells_csv": str(ANALYSIS_CSV),
            "pareto_csv": str(PARETO_CSV),
            "summary_json": str(SUMMARY_JSON),
        },
        "evidence_boundary": {
            "planned_capacity_cells": capacity_summary.get("planned_cells"),
            "valid_capacity_cells": len(capacity_cells),
            "missing_capacity_cells": len(capacity_summary.get("missing_cells", [])),
            "invalid_capacity_cells": len(invalid_rows),
            "invalid_cells": invalid_rows,
            "lambda_planned_cells": coverage.get("planned_cells"),
            "lambda_complete_cells": coverage.get("complete_cells"),
            "lambda_planned_raw_slots": coverage.get("planned_raw_slots"),
            "lambda_missing_raw_slots": coverage.get("missing_raw_slots"),
            "lambda_rows": len(rows),
        },
        "counts": {
            "analysis_lambda_cells": len(rows),
            "analysis_capacity_cells": len(capacity_cells),
            "by_model": _counter(rows, "model"),
            "by_segment": _counter(rows, "segment"),
            "by_device": _counter(rows, "device"),
        },
        "qa_flags": {
            "l_max_ms": L_MAX_MS,
            "low_delta_w": LOW_DELTA_W,
            "slo_median_violation_cells": _flag_count(
                rows, "flag_slo_median_violation"),
            "slo_any_run_violation_cells": _flag_count(
                rows, "flag_slo_any_run_violation"),
            "run_rows_p95_gt_lmax": sum(r["n_runs_p95_gt_lmax"] for r in rows),
            "low_delta_median_cells": _flag_count(
                rows, "flag_low_delta_median"),
            "run_rows_delta_lt_low": sum(r["n_runs_delta_lt_low"] for r in rows),
            "nonpositive_delta_cells": _flag_count(
                rows, "flag_nonpositive_delta_run"),
            "run_rows_delta_le_zero": sum(r["n_runs_delta_le_zero"] for r in rows),
            "nonpositive_energy_cells": _flag_count(
                rows, "flag_nonpositive_energy_run"),
            "run_rows_energy_le_zero": sum(r["n_runs_energy_le_zero"] for r in rows),
            "by_model": _group_summary(by_model),
            "by_segment": _group_summary(by_segment),
        },
        "pareto": {
            "definition": (
                "Within each (device, model, lambda_frac), a mode is dominated "
                "when another mode has <= median marginal wall energy and <= "
                "median p95 latency, with one strict improvement."),
            "rows": len(pareto_rows),
            "pareto_frontier_rows": pareto_frontier_count,
            "dominated_rows": dominated_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write analysis CSV + summary JSON")
    args = parser.parse_args()

    rows, pareto_rows, summary = build_analysis_rows()

    print("=" * 80)
    print("Full-DVFS analysis build")
    print("=" * 80)
    boundary = summary["evidence_boundary"]
    print(f"  capacity planned:      {boundary['planned_capacity_cells']}")
    print(f"  capacity valid:        {boundary['valid_capacity_cells']}")
    print(f"  capacity missing:      {boundary['missing_capacity_cells']}")
    print(f"  capacity invalid:      {boundary['invalid_capacity_cells']}")
    print(f"  lambda complete cells: {boundary['lambda_complete_cells']}/"
          f"{boundary['lambda_planned_cells']}")
    print(f"  lambda missing slots:  {boundary['lambda_missing_raw_slots']}/"
          f"{boundary['lambda_planned_raw_slots']}")
    qa = summary["qa_flags"]
    print(f"  SLO median flags:      {qa['slo_median_violation_cells']}")
    print(f"  SLO any-run flags:     {qa['slo_any_run_violation_cells']}")
    print(f"  low-delta flags:       {qa['low_delta_median_cells']}")
    print(f"  nonpositive delta:     {qa['nonpositive_delta_cells']}")
    print(f"  pareto rows:           {summary['pareto']['pareto_frontier_rows']}/"
          f"{summary['pareto']['rows']}")

    if args.apply:
        atomic_csv_write(ANALYSIS_CSV, list(rows[0].keys()), rows)
        atomic_csv_write(PARETO_CSV, list(pareto_rows[0].keys()), pareto_rows)
        atomic_json_write(SUMMARY_JSON, summary)
        print(f"Written: {ANALYSIS_CSV} ({len(rows)} rows)")
        print(f"Written: {PARETO_CSV} ({len(pareto_rows)} rows)")
        print(f"Written: {SUMMARY_JSON}")
    else:
        print("Dry run. Use --apply to write.")


if __name__ == "__main__":
    main()
