"""Build full-DVFS lambda-sweep artifacts from raw *_lsweep.json files.

This is intentionally separate from the removed policy-keyed λ builders. The T5
artifact is keyed by (device, model, policy, lambda_frac), while the full-DVFS
campaign must preserve dvfs_mode as part of the measurement cell.

Inputs:
  results/raw/*_lsweep.json
  results/derived/full_dvfs_capacity.csv

Outputs:
  results/derived/full_dvfs_lambda_sweep.csv
  results/derived/full_dvfs_lambda_cell_summary.csv
  results/derived/full_dvfs_lambda_summary.json

Usage:
  python3 build_full_dvfs_lambda_artifact.py
  python3 build_full_dvfs_lambda_artifact.py --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from _util import (atomic_csv_write, atomic_json_write, POWER_EXCLUDED_DEVICES,
                   marginal_wall_energy_per_image)
from constants import L_MAX_MS
from measurement_jobs import LAMBDA_FRACS_DEFAULT

FULL_DVFS_EXPECTED_RUNS_DEFAULT = 10

RESULTS_DIR = Path(__file__).parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
CAPACITY_CSV = RESULTS_DIR / "derived" / "full_dvfs_capacity.csv"
RUN_CSV = RESULTS_DIR / "derived" / "full_dvfs_lambda_sweep.csv"
CELL_CSV = RESULTS_DIR / "derived" / "full_dvfs_lambda_cell_summary.csv"
SUMMARY_JSON = RESULTS_DIR / "derived" / "full_dvfs_lambda_summary.json"

SEGMENTS: dict[str, set[str]] = {
    "server": {"gpu-server"},
    "jetson": {"orin", "orin-nano", "xavier", "jetson"},
    "sbc_npu": {"orangepi-npu", "rasp5", "orangepi", "lattepanda"},
}


@dataclass
class LambdaRun:
    device: str
    model: str
    dvfs_mode: int
    dvfs_mode_label: str
    condition_tag: str
    segment: str
    capacity_ips: float
    lambda_frac: float
    target_rps: float
    run_idx: int
    achieved_rps: float
    p95_latency_ms: float
    warm_idle_w: float | None
    serving_w: float | None
    delta_w: float | None
    marginal_wall_energy_j_per_inf: float | None
    n_valid_power_runs: int | None
    power_status: str
    source_json: str
    source_mtime_ns: int
    batch_size: int = 1


def _segment_of(device: str) -> str:
    for seg, members in SEGMENTS.items():
        if device in members:
            return seg
    return "unassigned"


def _load_capacity_rows() -> dict[tuple[str, str, int, int], dict]:
    if not CAPACITY_CSV.exists():
        raise FileNotFoundError(
            f"{CAPACITY_CSV} does not exist; run "
            "build_full_dvfs_capacity_artifact.py --apply first")
    out: dict[tuple[str, str, int, int], dict] = {}
    with open(CAPACITY_CSV) as f:
        for row in csv.DictReader(f):
            try:
                mode = int(row["dvfs_mode"])
                capacity = float(row["capacity_ips"])
                bs = int(row.get("batch_size") or 1)
            except (KeyError, TypeError, ValueError):
                continue
            if capacity <= 0:
                continue
            row = dict(row)
            row["dvfs_mode"] = mode
            row["capacity_ips"] = capacity
            row["batch_size"] = bs
            out[(row["device"], row["model"], mode, bs)] = row
    return out


def _iter_lsweep_jsons() -> list[Path]:
    return sorted(RAW_DIR.glob("*_lsweep.json"))


def _compute_marginal_energy(delta_w, achieved_rps, batch_size: int = 1) -> float | None:
    # Coercion guard stays here (legacy behaviour: non-numeric inputs → None).
    # The pure-quotient helper subsumes the achieved<=0 → None case; the round
    # to 6 dp and the None-guard stay at THIS call site so bs=1 stays
    # byte-identical to the frozen artifact (== round(delta/achieved, 6)).
    try:
        delta = float(delta_w)
        achieved = float(achieved_rps)
    except (TypeError, ValueError):
        return None
    e = marginal_wall_energy_per_image(delta, achieved, batch_size)
    return round(e, 6) if e is not None else None


def _parse_run(path: Path, capacity_map: dict[tuple[str, str, int, int], dict]) -> tuple[LambdaRun | None, str | None]:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{exc}"

    if data.get("run_type") != "lambda_sweep":
        return None, "not_lambda_sweep"
    if data.get("run_family") != "full_dvfs":
        return None, "not_full_dvfs"
    if data.get("policy") != "full_dvfs":
        return None, "not_full_dvfs_policy"

    device = data.get("device")
    model = data.get("model")
    if device in POWER_EXCLUDED_DEVICES:
        return None, "power_excluded_device"
    try:
        dvfs_mode = int(data.get("dvfs_mode"))
        lambda_frac = float(data.get("lambda_frac"))
        run_idx = int(data.get("run_idx"))
        achieved_rps = float(data.get("achieved_rps"))
        p95_latency_ms = float(data.get("p95_ms"))
        target_rps = float(data.get("target_rps"))
    except (TypeError, ValueError):
        return None, "invalid_numeric_fields"

    # v5 (batch-aware) raws MUST carry batch_size; never silently default to 1
    # for a v5 raw (that would divide by the wrong N). v4 raws are implied bs=1.
    if int(data.get("schema_version") or 0) >= 5 and "batch_size" not in data:
        return None, "v5_missing_batch_size"
    batch_size = int(data.get("batch_size") or 1)
    cap_row = capacity_map.get((device, model, dvfs_mode, batch_size))
    if cap_row is None:
        return None, "missing_capacity_anchor"
    if achieved_rps <= 0:
        return None, "nonpositive_achieved_rps"

    power = data.get("power_trace") or {}
    if power.get("status") != "ok":
        return None, f"power_status:{power.get('status')!r}"
    summary = power.get("summary") or {}
    warm_idle_w = summary.get("warm_idle_median_watts")
    serving_w = summary.get("serving_median_watts")
    delta_w = summary.get("delta_median_watts")
    if warm_idle_w is None or serving_w is None or delta_w is None:
        return None, "missing_power_summary"

    return LambdaRun(
        device=str(device),
        model=str(model),
        dvfs_mode=dvfs_mode,
        dvfs_mode_label=str(data.get("dvfs_mode_label") or ""),
        condition_tag=str(data.get("condition_tag") or f"dvfs{dvfs_mode}"),
        segment=_segment_of(str(device)),
        capacity_ips=float(cap_row["capacity_ips"]),
        lambda_frac=lambda_frac,
        target_rps=target_rps,
        run_idx=run_idx,
        achieved_rps=achieved_rps,
        p95_latency_ms=p95_latency_ms,
        warm_idle_w=float(warm_idle_w),
        serving_w=float(serving_w),
        delta_w=float(delta_w),
        marginal_wall_energy_j_per_inf=_compute_marginal_energy(
            delta_w, achieved_rps, batch_size),
        n_valid_power_runs=summary.get("n_valid_runs"),
        power_status=str(power.get("status", "missing")),
        source_json=path.name,
        source_mtime_ns=path.stat().st_mtime_ns,
        batch_size=batch_size,
    ), None


def _run_key(r: LambdaRun) -> tuple:
    return (r.device, r.model, r.dvfs_mode, r.lambda_frac, r.run_idx, r.batch_size)


def _cell_key(r: LambdaRun) -> tuple:
    return (r.device, r.model, r.dvfs_mode, r.lambda_frac, r.batch_size)


def _run_to_dict(r: LambdaRun) -> dict:
    return {
        "device": r.device,
        "model": r.model,
        "dvfs_mode": r.dvfs_mode,
        "dvfs_mode_label": r.dvfs_mode_label,
        "condition_tag": r.condition_tag,
        "segment": r.segment,
        "capacity_ips": round(r.capacity_ips, 4),
        "lambda_frac": r.lambda_frac,
        "target_rps": round(r.target_rps, 2),
        "run_idx": r.run_idx,
        "achieved_rps": round(r.achieved_rps, 2),
        "p95_latency_ms": round(r.p95_latency_ms, 2),
        "warm_idle_w": round(r.warm_idle_w, 4) if r.warm_idle_w is not None else None,
        "serving_w": round(r.serving_w, 4) if r.serving_w is not None else None,
        "delta_w": round(r.delta_w, 4) if r.delta_w is not None else None,
        "marginal_wall_energy_j_per_inf": r.marginal_wall_energy_j_per_inf,
        "n_valid_power_runs": r.n_valid_power_runs,
        "power_status": r.power_status,
        "source_json": r.source_json,
        "batch_size": r.batch_size,
    }


def _agg(vals: list[float | None]) -> dict:
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {
        "median": round(statistics.median(clean), 6),
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
        "n": len(clean),
    }


def _cell_summary(runs: list[LambdaRun]) -> dict:
    first = runs[0]
    achieved = _agg([r.achieved_rps for r in runs])
    p95 = _agg([r.p95_latency_ms for r in runs])
    idle = _agg([r.warm_idle_w for r in runs])
    serving = _agg([r.serving_w for r in runs])
    delta = _agg([r.delta_w for r in runs])
    energy = _agg([r.marginal_wall_energy_j_per_inf for r in runs])
    return {
        "device": first.device,
        "model": first.model,
        "dvfs_mode": first.dvfs_mode,
        "dvfs_mode_label": first.dvfs_mode_label,
        "condition_tag": first.condition_tag,
        "segment": first.segment,
        "capacity_ips": round(first.capacity_ips, 4),
        "lambda_frac": first.lambda_frac,
        "target_rps": round(first.target_rps, 2),
        "n_runs": len(runs),
        "achieved_rps_median": achieved["median"],
        "achieved_rps_min": achieved["min"],
        "achieved_rps_max": achieved["max"],
        "p95_latency_ms_median": p95["median"],
        "p95_latency_ms_max": p95["max"],
        "warm_idle_w_median": idle["median"],
        "serving_w_median": serving["median"],
        "delta_w_median": delta["median"],
        "marginal_wall_energy_j_per_inf_median": energy["median"],
        "marginal_wall_energy_j_per_inf_min": energy["min"],
        "marginal_wall_energy_j_per_inf_max": energy["max"],
        "batch_size": first.batch_size,
    }


def _coverage_summary(capacity_map: dict[tuple[str, str, int, int], dict],
                      run_rows: list[dict],
                      expected_fracs: tuple[float, ...],
                      expected_runs: int) -> tuple[dict, list[str]]:
    observed: dict[tuple[str, str, int, float, int], set[int]] = {}
    for row in run_rows:
        key = (row["device"], row["model"], int(row["dvfs_mode"]),
               float(row["lambda_frac"]), int(row.get("batch_size") or 1))
        observed.setdefault(key, set()).add(int(row["run_idx"]))

    per_cell = {}
    missing_slots: list[str] = []
    complete_cells = 0
    complete_lambda_cells = 0
    for device, model, mode, bs in sorted(capacity_map):
        tag = f"dvfs{mode}" if bs == 1 else f"dvfs{mode}_bs{bs}"
        cell_complete = True
        frac_counts = {}
        for frac in expected_fracs:
            got = observed.get((device, model, mode, frac, bs), set())
            frac_counts[str(frac)] = len(got)
            if len(got) >= expected_runs:
                complete_lambda_cells += 1
            else:
                cell_complete = False
            for run_idx in range(expected_runs):
                if run_idx not in got:
                    missing_slots.append(
                        f"{device}/{model}/{tag}/l{frac:g}/r{run_idx}")
        if cell_complete:
            complete_cells += 1
        per_cell[f"{device}/{model}/{tag}"] = {
            "complete": cell_complete,
            "lambda_counts": frac_counts,
        }

    return {
        "planned_cells": len(capacity_map),
        "planned_lambda_cells": len(capacity_map) * len(expected_fracs),
        "planned_raw_slots": len(capacity_map) * len(expected_fracs) * expected_runs,
        "complete_cells": complete_cells,
        "complete_lambda_cells": complete_lambda_cells,
        "missing_raw_slots": len(missing_slots),
        "per_cell": per_cell,
    }, missing_slots


def build_artifact(expected_fracs: tuple[float, ...],
                   expected_runs: int) -> tuple[list[dict], list[dict], dict]:
    capacity_map = _load_capacity_rows()
    jsons = _iter_lsweep_jsons()

    latest_by_slot: dict[tuple, LambdaRun] = {}
    rejected: dict[str, int] = {}
    duplicate_slots = 0
    for path in jsons:
        run, reason = _parse_run(path, capacity_map)
        if run is None:
            rejected[reason or "unknown"] = rejected.get(reason or "unknown", 0) + 1
            continue
        key = _run_key(run)
        prior = latest_by_slot.get(key)
        if prior is None or run.source_mtime_ns > prior.source_mtime_ns:
            if prior is not None:
                duplicate_slots += 1
            latest_by_slot[key] = run
        else:
            duplicate_slots += 1

    runs = sorted(latest_by_slot.values(), key=_run_key)
    cells: dict[tuple, list[LambdaRun]] = {}
    for run in runs:
        cells.setdefault(_cell_key(run), []).append(run)

    run_rows = [_run_to_dict(r) for r in runs]
    cell_rows = sorted((_cell_summary(rs) for rs in cells.values()),
                       key=lambda r: (r["device"], r["model"], r["dvfs_mode"],
                                      r["lambda_frac"], r["batch_size"]))
    coverage, missing_slots = _coverage_summary(
        capacity_map, run_rows, expected_fracs, expected_runs)
    summary = {
        "artifact": "full_dvfs_lambda",
        "capacity_source": str(CAPACITY_CSV),
        "n_raw_lsweep_files": len(jsons),
        "n_runs_admitted": len(run_rows),
        "n_lambda_cells_observed": len(cell_rows),
        "n_duplicate_run_slots_shadowed": duplicate_slots,
        "rejected_raw_counts": dict(sorted(rejected.items())),
        "expected_lambda_fracs": list(expected_fracs),
        "expected_runs_per_lambda": expected_runs,
        "l_max_ms": L_MAX_MS,
        "segments": {name: sorted(members) for name, members in SEGMENTS.items()},
        "coverage": coverage,
        "missing_slots": missing_slots,
    }
    return run_rows, cell_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write CSV + JSON outputs")
    parser.add_argument("--expected-fracs", default=",".join(str(f) for f in LAMBDA_FRACS_DEFAULT),
                        help="comma-separated planned lambda fractions")
    parser.add_argument("--expected-runs", type=int,
                        default=FULL_DVFS_EXPECTED_RUNS_DEFAULT,
                        help="planned repeats per lambda fraction")
    args = parser.parse_args()

    expected_fracs = tuple(float(x) for x in args.expected_fracs.split(",") if x)
    run_rows, cell_rows, summary = build_artifact(
        expected_fracs=expected_fracs,
        expected_runs=args.expected_runs,
    )

    print("=" * 80)
    print("Full-DVFS lambda artifact build")
    print("=" * 80)
    print(f"  raw *_lsweep files scanned: {summary['n_raw_lsweep_files']}")
    print(f"  admitted run slots:         {summary['n_runs_admitted']}")
    print(f"  lambda cells observed:      {summary['n_lambda_cells_observed']}")
    print(f"  duplicate slots shadowed:   {summary['n_duplicate_run_slots_shadowed']}")
    cov = summary["coverage"]
    print(f"  complete cells:             {cov['complete_cells']}/{cov['planned_cells']}")
    print(f"  complete lambda cells:      {cov['complete_lambda_cells']}/"
          f"{cov['planned_lambda_cells']}")
    print(f"  missing raw slots:          {cov['missing_raw_slots']}/"
          f"{cov['planned_raw_slots']}")
    if summary["rejected_raw_counts"]:
        print("  rejected raw counts:")
        for reason, count in summary["rejected_raw_counts"].items():
            print(f"    {reason}: {count}")

    if args.apply:
        if run_rows:
            atomic_csv_write(RUN_CSV, list(run_rows[0].keys()), run_rows)
            print(f"Written: {RUN_CSV} ({len(run_rows)} rows)")
        if cell_rows:
            atomic_csv_write(CELL_CSV, list(cell_rows[0].keys()), cell_rows)
            print(f"Written: {CELL_CSV} ({len(cell_rows)} rows)")
        atomic_json_write(SUMMARY_JSON, summary)
        print(f"Written: {SUMMARY_JSON}")
    else:
        print("Dry run. Use --apply to write.")


if __name__ == "__main__":
    main()
