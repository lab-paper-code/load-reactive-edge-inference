"""
Surrogate vs measured energy contrast.

Joins:
  - eep-profiler DB (isolated profiling, bs=1): E_inc, idle_W, active_W
  - serving_power_measured.csv (serving-load): warm_idle_W, serving_W, delta_W
    + marginal_wall_energy_j_per_inf

Produces:
  - results/derived/surrogate_contrast.csv, per (device, model, policy) row
  # surrogate: isolated profile used as a pre-serving estimate of serving energy.
  # iso/srv: "iso" = isolated-profile side; "srv" = serving-load measured side.
  - results/derived/surrogate_contrast_summary.json, aggregate stats

Research question:
  "When does isolated profiling predict serving energy well, and when does it fail?"

Comparisons (per pair):
  - isolated E_inc (J/inf) vs marginal_wall_energy_j_per_inf (J/inf)
  - isolated idle_W vs warm_idle_W (serving-load)
  - isolated active_W vs serving_W (serving-load at confirmed MST)

Usage:
    python3 surrogate_contrast.py           # dry-run
    python3 surrogate_contrast.py --apply   # write CSV + summary JSON
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from constants import DB_PATH, ACCURACY
from profiler_bridge import load_best_batch1_profiles

RESULTS_DIR = Path(__file__).parent / "results"
SERVING_POWER_CSV = RESULTS_DIR / "derived" / "serving_power_measured.csv"
CONTRAST_CSV = RESULTS_DIR / "derived" / "surrogate_contrast.csv"
CONTRAST_JSON = RESULTS_DIR / "derived" / "surrogate_contrast_summary.json"


def load_serving_power() -> dict[tuple[str, str, str], dict]:
    """Load serving_power_measured.csv, keyed by (device, model, policy)."""
    out = {}
    if not SERVING_POWER_CSV.exists():
        return out
    with open(SERVING_POWER_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["device"], row["model"], row["policy"])
            out[key] = {
                "capacity_ips": float(row["capacity_ips"]) if row["capacity_ips"] else None,
                "warm_idle_power_w": float(row["warm_idle_power_w"]) if row["warm_idle_power_w"] else None,
                "avg_serving_power_w": float(row["avg_serving_power_w"]) if row["avg_serving_power_w"] else None,
                "delta_power_w": float(row["delta_power_w"]) if row["delta_power_w"] else None,
                "marginal_wall_energy_j_per_inf": float(row["marginal_wall_energy_j_per_inf"]) if row["marginal_wall_energy_j_per_inf"] else None,
            }
    return out


def load_isolated_profiles() -> dict[tuple[str, str], dict]:
    """Load isolated bs=1 profiles from eep-profiler DB."""
    out = {}
    for p in load_best_batch1_profiles(str(DB_PATH)):
        out[(p["device"], p["model"])] = p
    return out


def _pct_diff(measured: float, surrogate: float) -> float | None:
    if measured is None or surrogate is None or measured == 0:
        return None
    return round((surrogate - measured) / measured * 100, 1)


def build_contrast_rows(serving: dict, isolated: dict) -> list[dict]:
    rows = []
    for key, sv in sorted(serving.items()):
        device, model, policy = key
        iso = isolated.get((device, model))
        if iso is None:
            print(f"  SKIP {device}/{model}/{policy}: no isolated profile")
            continue

        iso_e_inc = iso.get("energy_inc_per_inf_j")
        iso_idle = iso.get("idle_power_w")
        iso_active = iso.get("p_wall_avg_w")
        flag_unreliable = iso.get("flag_incremental_unreliable", 0)

        srv_energy = sv.get("marginal_wall_energy_j_per_inf")
        srv_idle = sv.get("warm_idle_power_w")
        srv_serving = sv.get("avg_serving_power_w")

        energy_pct_diff = _pct_diff(srv_energy, iso_e_inc)
        idle_pct_diff = _pct_diff(srv_idle, iso_idle)
        active_pct_diff = _pct_diff(srv_serving, iso_active)

        print(f"  {device:<13} {model:<20} {policy:<9}"
              f" | E: iso={iso_e_inc} vs srv={srv_energy}J (Δ%={energy_pct_diff})"
              f" | idle: {iso_idle}W vs {srv_idle}W (Δ%={idle_pct_diff})"
              f" | active: {iso_active}W vs {srv_serving}W (Δ%={active_pct_diff})")

        rows.append({
            "device": device,
            "model": model,
            "policy": policy,
            "accuracy": ACCURACY.get(model, ""),
            "iso_e_inc_j_per_inf": iso_e_inc,
            "srv_marginal_wall_energy_j_per_inf": srv_energy,
            "energy_surrogate_pct_diff": energy_pct_diff,
            "iso_idle_w": iso_idle,
            "srv_warm_idle_w": srv_idle,
            "idle_surrogate_pct_diff": idle_pct_diff,
            "iso_active_w": iso_active,
            "srv_avg_serving_w": srv_serving,
            "active_surrogate_pct_diff": active_pct_diff,
            "iso_dvfs_mode": iso.get("dvfs_mode"),
            "iso_e_inc_reliable": not bool(flag_unreliable),
            "iso_incremental_source": iso.get("incremental_source", ""),
        })
    return rows


def compute_summary(rows: list[dict]) -> dict:
    def _collect(field):
        return [r[field] for r in rows if r.get(field) is not None]

    def _stats(vals):
        if not vals:
            return {}
        return {
            "n": len(vals),
            "median": round(statistics.median(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "mean": round(statistics.mean(vals), 2),
        }

    reliable_rows = [r for r in rows if r.get("iso_e_inc_reliable")]

    return {
        "n_pairs_total": len(rows),
        "n_pairs_e_inc_reliable": len(reliable_rows),
        "energy_pct_diff": _stats(_collect("energy_surrogate_pct_diff")),
        "energy_pct_diff_reliable_only": _stats(
            [r["energy_surrogate_pct_diff"] for r in reliable_rows
             if r.get("energy_surrogate_pct_diff") is not None]),
        "idle_pct_diff": _stats(_collect("idle_surrogate_pct_diff")),
        "active_pct_diff": _stats(_collect("active_surrogate_pct_diff")),
        "interpretation": (
            "Positive pct_diff = isolated profiling OVERESTIMATES the "
            "serving-load value. Negative = underestimates. "
            "iso_e_inc_reliable=False rows are eep-profiler calibration "
            "artifacts (e.g. orangepi-npu); treat as exploratory."
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Surrogate vs measured energy contrast")
    parser.add_argument("--apply", action="store_true",
                        help="Write derived CSVs and summary JSON")
    args = parser.parse_args()

    serving = load_serving_power()
    isolated = load_isolated_profiles()

    print(f"Serving power rows: {len(serving)}")
    print(f"Isolated profiles (bs=1): {len(isolated)}")
    print()
    print("=" * 120)
    print("Surrogate contrast (isolated bs=1 vs serving-load)")
    print("=" * 120)

    rows = build_contrast_rows(serving, isolated)
    summary = compute_summary(rows)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    if args.apply:
        write_csv(CONTRAST_CSV, rows)
        print(f"\nWritten: {CONTRAST_CSV} ({len(rows)} rows)")
        with open(CONTRAST_JSON, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Written: {CONTRAST_JSON}")
    else:
        print(f"\nDry run. Use --apply to write.")


if __name__ == "__main__":
    main()
