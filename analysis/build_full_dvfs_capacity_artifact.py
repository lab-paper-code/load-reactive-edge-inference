"""Build the full-DVFS capacity artifact from validated capacity raws.

This builder is intentionally separate from the policy-keyed authoritative manifest:
policy-keyed rows are keyed by (device, model, policy), while full-DVFS evidence is
keyed by (device, model, dvfs_mode). Mixing the two would collapse distinct
operating points into one `policy=full_dvfs` bucket.

Outputs:
  results/derived/full_dvfs_capacity.csv
  results/derived/full_dvfs_capacity_summary.json

Usage:
  python3 build_full_dvfs_capacity_artifact.py
  python3 build_full_dvfs_capacity_artifact.py --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _util import atomic_csv_write, atomic_json_write
from full_dvfs_smoke import validate_full_dvfs_raw
from measurement_jobs import iter_full_dvfs_jobs, job_tag, server_batch_map

RESULTS_DIR = Path(__file__).parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
CAPACITY_CSV = RESULTS_DIR / "derived" / "full_dvfs_capacity.csv"
SUMMARY_JSON = RESULTS_DIR / "derived" / "full_dvfs_capacity_summary.json"

FIELDNAMES = [
    "device",
    "model",
    "dvfs_mode",
    "dvfs_mode_label",
    "condition_tag",
    "expected_ep",
    "capacity_ips",
    "capacity_ci_low",
    "capacity_ci_high",
    "capacity_selection",
    "capacity_confirmed",
    "power_status",
    "power_samples",
    "warm_idle_w",
    "serving_w",
    "delta_w",
    "source_json",
    "batch_size",
]


def _parse_name_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def _capacity_key(data: dict) -> tuple[str, str, int | None, int]:
    # batch_size defaults to 1 for legacy (v4, bs=1) raws that carry no field,
    # so a v4 raw keys to (d, m, mode, 1), matching a bs=1 planned job.
    return (data.get("device"), data.get("model"), data.get("dvfs_mode"),
            int(data.get("batch_size") or 1))


def _is_full_dvfs_capacity(data: dict) -> bool:
    if data.get("run_family") != "full_dvfs":
        return False
    if data.get("policy") != "full_dvfs":
        return False
    if data.get("run_type") == "lambda_sweep":
        return False
    return isinstance(data.get("dvfs_mode"), int)


def _latest_raws_for_plan(jobs: list) -> tuple[dict, dict]:
    planned = {(j.device, j.model, j.dvfs_mode, j.batch_size): j for j in jobs}
    latest: dict[tuple, tuple[Path, dict]] = {}
    duplicates: dict[str, list[str]] = {}

    for path in RAW_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not _is_full_dvfs_capacity(data):
            continue
        key = _capacity_key(data)
        if key not in planned:
            continue
        prev = latest.get(key)
        if prev is None or path.stat().st_mtime > prev[0].stat().st_mtime:
            if prev is not None:
                duplicates.setdefault(job_tag(planned[key]), []).append(prev[0].name)
            latest[key] = (path, data)
        else:
            duplicates.setdefault(job_tag(planned[key]), []).append(path.name)

    return latest, duplicates


def _row_for(job, path: Path, data: dict) -> dict:
    power = data.get("power_trace") or {}
    summary = power.get("summary") or {}
    rng = data.get("capacity_range") or [None, None]
    return {
        "device": job.device,
        "model": job.model,
        "dvfs_mode": job.dvfs_mode,
        "dvfs_mode_label": data.get("dvfs_mode_label") or job.dvfs_label,
        "condition_tag": data.get("condition_tag") or f"dvfs{job.dvfs_mode}",
        "expected_ep": job.expected_ep,
        "capacity_ips": data.get("capacity_ips"),
        "capacity_ci_low": rng[0] if len(rng) > 0 else None,
        "capacity_ci_high": rng[1] if len(rng) > 1 else None,
        "capacity_selection": data.get("capacity_selection"),
        "capacity_confirmed": str(data.get("capacity_confirmed")).lower(),
        "power_status": power.get("status", "missing"),
        "power_samples": power.get("total_sample_count"),
        "warm_idle_w": summary.get("warm_idle_median_watts"),
        "serving_w": summary.get("serving_median_watts"),
        "delta_w": summary.get("delta_median_watts"),
        "source_json": path.name,
        "batch_size": int(data.get("batch_size") or 1),
    }


def build_capacity_artifact(models: set[str] | None = None,
                            batch_sizes: dict[str, tuple[int, ...]] | None = None
                            ) -> tuple[list[dict], dict]:
    jobs = iter_full_dvfs_jobs(models=models, batch_sizes=batch_sizes)
    planned = {(j.device, j.model, j.dvfs_mode, j.batch_size): j for j in jobs}
    latest, duplicates = _latest_raws_for_plan(jobs)

    rows: list[dict] = []
    invalid: list[dict] = []
    missing = sorted(set(planned) - set(latest))

    for key, job in sorted(planned.items()):
        if key not in latest:
            continue
        path, data = latest[key]
        report = validate_full_dvfs_raw(path, expected_job=job, require_power=True)
        if not report.ok:
            invalid.append({
                "cell": job_tag(job),
                "source_json": path.name,
                "errors": report.errors,
            })
            continue
        rows.append(_row_for(job, path, data))

    summary = {
        "artifact": "full_dvfs_capacity",
        "planned_cells": len(planned),
        "rows": len(rows),
        "missing_cells": [
            f"{d}/{m}/dvfs{mode}" if bs == 1 else f"{d}/{m}/dvfs{mode}_bs{bs}"
            for d, m, mode, bs in missing
        ],
        "invalid_rows": invalid,
        "duplicate_cells": duplicates,
        "output_csv": str(CAPACITY_CSV),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="mobilenet-v2-050",
                        help="comma-separated model filter "
                             "(default: mobilenet-v2-050)")
    parser.add_argument("--batch-sizes", default=None,
                        help="comma-separated server batch widths (e.g. 1,2,4,8); "
                             "default bs=1 only. Only gpu-server carries bs>1.")
    parser.add_argument("--apply", action="store_true",
                        help="write CSV + JSON")
    args = parser.parse_args()

    batch_req = ({int(x) for x in args.batch_sizes.split(",") if x.strip()}
                 if args.batch_sizes else None)
    rows, summary = build_capacity_artifact(
        models=_parse_name_set(args.models),
        batch_sizes=server_batch_map(batch_req))

    print("=" * 80)
    print("Full-DVFS capacity artifact")
    print("=" * 80)
    print(f"  planned: {summary['planned_cells']}")
    print(f"  rows:    {summary['rows']}")
    print(f"  missing: {len(summary['missing_cells'])}")
    print(f"  invalid: {len(summary['invalid_rows'])}")
    print(f"  duplicate cells with older raws: {len(summary['duplicate_cells'])}")

    if summary["missing_cells"]:
        print("Missing:")
        for cell in summary["missing_cells"]:
            print(f"  {cell}")
    if summary["invalid_rows"]:
        print("Invalid:")
        for item in summary["invalid_rows"]:
            print(f"  {item['cell']}: {item['errors']}")

    if args.apply:
        atomic_csv_write(CAPACITY_CSV, FIELDNAMES, rows)
        atomic_json_write(SUMMARY_JSON, summary)
        print(f"Written: {CAPACITY_CSV} ({len(rows)} rows)")
        print(f"Written: {SUMMARY_JSON}")
    else:
        print("Dry run. Use --apply to write.")


if __name__ == "__main__":
    main()
