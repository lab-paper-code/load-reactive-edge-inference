"""
Decision-flip analysis: does isolated-profile ranking disagree with
measured-serving ranking, and if so, by how much per (model, policy)?

Inputs:
  - results/derived/surrogate_contrast.csv
      per (device, model, policy) row with both sides in J/inf:
        iso_e_inc_j_per_inf            (profile-side E/inf)
        srv_marginal_wall_energy_j_per_inf (measured serving-side E/inf)
        iso_e_inc_reliable             (True/False; profile-side calibration bit)
  - results/derived/t2_capacity_power.csv (frozen)
      per (device, model, policy) row; we link only for slo_infeasible gating.

Per (model, policy):
  1. gather devices with both iso and srv energy values and iso_e_inc_reliable=True
  2. rank devices by iso_e (ascending = lower-energy better)
  3. rank devices by srv_e (ascending)
  4. compute:
     - top1_different: is the profile-best device the same as the measured-best?
     - pairwise_disagreement: fraction of ordered device pairs whose order flips
       (Kendall-tau distance / n_pairs)
     - kendall_tau: 1 - 2 * pairwise_disagreement
     - rank_delta_max: max |iso_rank - srv_rank| across devices

Per-pair row output (decision_flip.csv):
  device, model, policy, iso_e_j_per_inf, srv_e_j_per_inf,
  iso_rank, srv_rank, rank_delta, top1_different, profile_reliable

Summary output (decision_flip_summary.json):
  per (model, policy) aggregate + fleet totals.

Freeze rule: decision_flip.csv is a derived artifact; it becomes admitted
into the evidence map only when the freshness gate validates it against
T2 and surrogate_contrast upstream mtimes.

Usage:
    python3 decision_flip.py           # dry run
    python3 decision_flip.py --apply   # write CSV + JSON
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _util import atomic_csv_write, atomic_json_write

RESULTS_DIR = Path(__file__).parent / "results" / "derived"
SURROGATE_CSV = RESULTS_DIR / "surrogate_contrast.csv"
T2_CSV = RESULTS_DIR / "t2_capacity_power.csv"
DECISION_FLIP_CSV = RESULTS_DIR / "decision_flip.csv"
DECISION_FLIP_JSON = RESULTS_DIR / "decision_flip_summary.json"


def _load_surrogate_rows() -> list[dict]:
    with open(SURROGATE_CSV) as f:
        return list(csv.DictReader(f))


def _load_t2_infeasible_keys() -> set[tuple[str, str, str]]:
    """Collect (device, model, policy) keys marked SLO-infeasible in frozen T2."""
    out = set()
    if not T2_CSV.exists():
        return out
    with open(T2_CSV) as f:
        for r in csv.DictReader(f):
            if r.get("slo_infeasible", "").lower() == "true":
                out.add((r["device"], r["model"], r["policy"]))
    return out


def _to_float(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _rank(values: list[float]) -> list[int]:
    """Ascending rank (0-based) with stable ordering on ties."""
    idx = sorted(range(len(values)), key=lambda i: (values[i], i))
    out = [0] * len(values)
    for r, i in enumerate(idx):
        out[i] = r
    return out


def _pairwise_disagreement(xs: list[float], ys: list[float]) -> tuple[int, int]:
    """Count (disagreeing pairs, total ordered pairs) for two ranking vectors."""
    n = len(xs)
    disagree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue  # tie; not counted against either side
            total += 1
            if (dx > 0) != (dy > 0):
                disagree += 1
    return disagree, total


def analyze() -> tuple[list[dict], dict]:
    rows = _load_surrogate_rows()
    infeasible = _load_t2_infeasible_keys()

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["model"], r["policy"])
        groups.setdefault(key, []).append(r)

    out_rows: list[dict] = []
    summary_per_group: dict[str, dict] = {}
    total_flips_top1 = 0
    total_groups = 0
    total_pair_disagreements = 0
    total_pair_comparisons = 0

    for (model, policy), grp in sorted(groups.items()):
        keep = []
        for r in grp:
            iso_e = _to_float(r.get("iso_e_inc_j_per_inf"))
            srv_e = _to_float(r.get("srv_marginal_wall_energy_j_per_inf"))
            reliable = r.get("iso_e_inc_reliable", "").strip().lower() == "true"
            key = (r["device"], model, policy)
            if key in infeasible:
                continue
            if iso_e is None or srv_e is None:
                continue
            if not reliable:
                # Keep but mark; iso_e_inc_reliable=False signals a calibration artifact.
                pass
            keep.append({"row": r, "iso_e": iso_e, "srv_e": srv_e, "reliable": reliable})

        # Require ≥ 2 devices to define any ranking
        if len(keep) < 2:
            continue

        iso_vals = [k["iso_e"] for k in keep]
        srv_vals = [k["srv_e"] for k in keep]
        iso_ranks = _rank(iso_vals)
        srv_ranks = _rank(srv_vals)

        # Top-1 comparison: best device by each side
        iso_best_idx = iso_ranks.index(0)
        srv_best_idx = srv_ranks.index(0)
        top1_diff = iso_best_idx != srv_best_idx

        disagree_pairs, total_pairs = _pairwise_disagreement(iso_vals, srv_vals)
        kendall_tau = (
            1.0 - 2.0 * disagree_pairs / total_pairs if total_pairs > 0 else float("nan")
        )
        rank_deltas = [abs(iso_ranks[i] - srv_ranks[i]) for i in range(len(keep))]
        max_rank_delta = max(rank_deltas) if rank_deltas else 0

        for i, k in enumerate(keep):
            out_rows.append({
                "device": k["row"]["device"],
                "model": model,
                "policy": policy,
                "iso_e_j_per_inf": round(k["iso_e"], 6),
                "srv_e_j_per_inf": round(k["srv_e"], 6),
                "iso_rank": iso_ranks[i],
                "srv_rank": srv_ranks[i],
                "rank_delta": srv_ranks[i] - iso_ranks[i],
                "top1_different": top1_diff,
                "profile_reliable": k["reliable"],
            })

        summary_per_group[f"{model}/{policy}"] = {
            "n_devices": len(keep),
            "top1_different": top1_diff,
            "iso_best_device": keep[iso_best_idx]["row"]["device"],
            "srv_best_device": keep[srv_best_idx]["row"]["device"],
            "kendall_tau": round(kendall_tau, 4) if total_pairs > 0 else None,
            "pairwise_disagreement_ratio": (
                round(disagree_pairs / total_pairs, 4) if total_pairs > 0 else None
            ),
            "n_pairwise": total_pairs,
            "n_disagreeing_pairs": disagree_pairs,
            "max_rank_delta": max_rank_delta,
        }

        total_groups += 1
        if top1_diff:
            total_flips_top1 += 1
        total_pair_disagreements += disagree_pairs
        total_pair_comparisons += total_pairs

    summary = {
        "per_model_policy": summary_per_group,
        "fleet": {
            "n_groups": total_groups,
            "n_groups_with_top1_flip": total_flips_top1,
            "top1_flip_rate": (
                round(total_flips_top1 / total_groups, 4) if total_groups > 0 else None
            ),
            "fleet_kendall_tau": (
                round(1.0 - 2.0 * total_pair_disagreements / total_pair_comparisons, 4)
                if total_pair_comparisons > 0
                else None
            ),
            "fleet_pairwise_disagreement_ratio": (
                round(total_pair_disagreements / total_pair_comparisons, 4)
                if total_pair_comparisons > 0
                else None
            ),
            "n_pairwise_total": total_pair_comparisons,
            "n_disagreeing_pairs_total": total_pair_disagreements,
        },
        "methodology": {
            "ranking_metric": "energy-per-inference (ascending = better)",
            "profile_side_column": "iso_e_inc_j_per_inf",
            "measured_side_column": "srv_marginal_wall_energy_j_per_inf",
            "grouping": "(model, policy)",
            "inclusion_rule": (
                "device rows with both iso and srv values, not SLO-infeasible in T2; "
                "iso_e_inc_reliable=False rows are retained but flagged"
            ),
            "tie_handling": "ties are excluded from pairwise disagreement count",
        },
    }
    return out_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    rows, summary = analyze()

    print("=" * 80)
    print("Per (model, policy) summary")
    print("=" * 80)
    for key, s in summary["per_model_policy"].items():
        flip = "FLIP" if s["top1_different"] else "ok  "
        print(
            f"  {key:<30} n={s['n_devices']} {flip}  "
            f"iso_best={s['iso_best_device']:<13} srv_best={s['srv_best_device']:<13}  "
            f"kendall_tau={s['kendall_tau']}  pairwise_disagree={s['pairwise_disagreement_ratio']}  "
            f"max_rank_delta={s['max_rank_delta']}"
        )

    print()
    print("=" * 80)
    print("Fleet totals")
    print("=" * 80)
    print(json.dumps(summary["fleet"], indent=2))

    if args.apply:
        atomic_csv_write(DECISION_FLIP_CSV, list(rows[0].keys()), rows)
        print(f"\nWritten: {DECISION_FLIP_CSV} ({len(rows)} rows)")
        atomic_json_write(DECISION_FLIP_JSON, summary)
        print(f"Written: {DECISION_FLIP_JSON}")
    else:
        print(f"\nDry run. Use --apply to write.")


if __name__ == "__main__":
    main()
