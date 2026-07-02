"""Build research-focused validity analysis for power-aware scheduling.

This script turns the full-DVFS sweep into scheduler-facing evidence.  It does
not compare raw lambda fractions across modes directly.  Instead, it fixes an
external demand level for each `(device, model)` group and evaluates each power
mode at the corresponding load fraction on that mode:

    load_fraction_on_mode = external_demand_rps / mode_capacity_ips

The measured lambda profiles are then linearly interpolated.  This keeps the
policy comparison tied to the actual scheduling question: for the same offered
load, which mode satisfies latency while minimizing power/energy?

Inputs:
  results/derived/full_dvfs_analysis_cells.csv
  results/derived/full_dvfs_lambda_sweep.csv

Outputs:
  results/derived/scheduler_action_space.csv
  results/derived/scheduler_policy_eval.csv
  results/derived/scheduler_validity_summary.json
  results/figures/scheduling_validity/*.png

Usage:
  python3 analyze_scheduling_validity.py
  python3 analyze_scheduling_validity.py --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle

from _util import atomic_csv_write, atomic_json_write
from constants import DEVICES, MODELS, L_MAX_MS

RESULTS_DIR = Path(__file__).parent / "results"
DERIVED_DIR = RESULTS_DIR / "derived"
FIG_DIR = RESULTS_DIR / "figures" / "scheduling_validity"

ANALYSIS_CSV = DERIVED_DIR / "full_dvfs_analysis_cells.csv"
LAMBDA_SWEEP_CSV = DERIVED_DIR / "full_dvfs_lambda_sweep.csv"

ACTION_SPACE_CSV = DERIVED_DIR / "scheduler_action_space.csv"
POLICY_EVAL_CSV = DERIVED_DIR / "scheduler_policy_eval.csv"
SUMMARY_JSON = DERIVED_DIR / "scheduler_validity_summary.json"

DEMAND_FRACS = (0.25, 0.5, 0.75, 1.0)
LAMBDA_FRACS = (0.25, 0.5, 0.75, 1.0)

DEVICE_ORDER = {name: i for i, name in enumerate(DEVICES)}
MODEL_ORDER = {name: i for i, name in enumerate(MODELS)}
MAXN_ORDER_VALUE = 9999.0

MODEL_COLORS = {
    "mobilenet-v2-050": "tab:blue",
    "mobilenet-v2-100": "tab:orange",
    "efficientnet-b4": "tab:green",
}

DECISION_CATEGORY_ORDER = (
    "fixed",
    "lowest",
    "intermediate",
    "max",
    "no_sla_safe_mode",
    "no_feasible_mode",
)
DECISION_CATEGORY_COLORS = {
    "fixed": "#d0d0d0",
    "lowest": "#9ecae1",
    "intermediate": "#3182bd",
    "max": "#756bb1",
    "no_sla_safe_mode": "#f2f2f2",
    "no_feasible_mode": "#fff2cc",
}
DECISION_CATEGORY_LABELS = {
    "fixed": "fixed device",
    "lowest": "lowest mode",
    "intermediate": "intermediate mode",
    "max": "max-MST mode",
    "no_sla_safe_mode": "no SLA-safe mode",
    "no_feasible_mode": "infeasible demand",
}
DECISION_CATEGORY_HATCHES = {
    "no_sla_safe_mode": "///",
    "no_feasible_mode": "xx",
}
DECISION_CATEGORY_EDGES = {
    "no_sla_safe_mode": "#d62728",
    "no_feasible_mode": "#ff7f0e",
}


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return [dict(r) for r in csv.DictReader(f)]


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


def _round(val: float | None, digits: int = 6) -> float | None:
    return round(val, digits) if val is not None else None


def _pct(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return 100.0 * num / den


def _pct_reduction(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None or baseline == 0:
        return None
    return 100.0 * (baseline - value) / baseline


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    weight = pos - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def _mode_sort_key(row: dict) -> tuple:
    label = str(row.get("dvfs_mode_label", "")).upper()
    mode = int(row["dvfs_mode"])
    if label == "FIXED":
        return (0, 0.0, 0, mode)

    mhz = re.search(r"(\d+(?:\.\d+)?)\s*MHZ", label)
    if mhz:
        return (1, float(mhz.group(1)), 0, mode)

    watts = re.search(r"(\d+(?:\.\d+)?)\s*W", label)
    if watts:
        core = re.search(r"(\d+)\s*CORE", label)
        core_order = int(core.group(1)) if core else 0
        if "DESKTOP" in label:
            core_order = 99
        return (2, float(watts.group(1)), core_order, mode)

    if "MAXN" in label:
        return (2, MAXN_ORDER_VALUE, 0, mode)

    return (9, float(mode), 0, mode)


def _group_sort_key(key: tuple[str, str]) -> tuple:
    device, model = key
    return (MODEL_ORDER.get(model, 999), DEVICE_ORDER.get(device, 999))


def _short_label(label: str) -> str:
    out = label.replace("MODE_", "")
    out = out.replace("_", "-").replace("CORE", "C")
    out = out.replace("MAXN-SUPER", "MAXN+")
    out = out.replace("DESKTOP", "DESK")
    return out if len(out) <= 12 else out[:11] + "."


def _model_tag(model: str) -> str:
    return model.replace("-", "_")


def _build_profiles() -> tuple[dict[tuple[str, str], list[dict]], dict]:
    cell_rows = _read_csv(ANALYSIS_CSV)
    run_rows = _read_csv(LAMBDA_SWEEP_CSV)

    profiles_by_key: dict[tuple[str, str, int], dict] = {}
    for row in cell_rows:
        key = (row["device"], row["model"], int(row["dvfs_mode"]))
        profile = profiles_by_key.setdefault(
            key,
            {
                "device": row["device"],
                "model": row["model"],
                "dvfs_mode": int(row["dvfs_mode"]),
                "dvfs_mode_label": row.get("dvfs_mode_label", ""),
                "capacity_ips": _float(row, "capacity_ips"),
                "capacity_delta_w": _float(row, "capacity_delta_w"),
                "lambdas": {},
            },
        )
        frac = _float(row, "lambda_frac")
        if frac is None:
            continue
        profile["lambdas"][frac] = {
            "target_rps": _float(row, "target_rps"),
            "achieved_rps_median": _float(row, "achieved_rps_median"),
            "p95_latency_ms_median": _float(row, "p95_latency_ms_median"),
            "serving_w_median": _float(row, "serving_w_median"),
            "delta_w_median": _float(row, "delta_w_median"),
        }

    profiles_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for profile in profiles_by_key.values():
        profiles_by_group[(profile["device"], profile["model"])].append(profile)

    for profiles in profiles_by_group.values():
        profiles.sort(key=_mode_sort_key)
        for rank, profile in enumerate(profiles):
            profile["semantic_rank"] = rank
            profile["n_modes"] = len(profiles)

    run_stats: dict[tuple[str, str, int, float], dict] = {}
    grouped_runs: dict[tuple[str, str, int, float], list[dict]] = defaultdict(list)
    for row in run_rows:
        mode = _int(row, "dvfs_mode")
        frac = _float(row, "lambda_frac")
        if mode is None or frac is None:
            continue
        grouped_runs[(row["device"], row["model"], mode, frac)].append(row)

    for key, rows in grouped_runs.items():
        deltas = [
            v for v in (_float(row, "delta_w") for row in rows)
            if v is not None
        ]
        p95s = [
            v for v in (_float(row, "p95_latency_ms") for row in rows)
            if v is not None
        ]
        delta_q1 = _percentile(deltas, 0.25)
        delta_q3 = _percentile(deltas, 0.75)
        p95_q1 = _percentile(p95s, 0.25)
        p95_q3 = _percentile(p95s, 0.75)
        run_stats[key] = {
            "n_runs": len(rows),
            "delta_w_iqr": (
                delta_q3 - delta_q1
                if delta_q1 is not None and delta_q3 is not None else None
            ),
            "p95_latency_ms_iqr": (
                p95_q3 - p95_q1
                if p95_q1 is not None and p95_q3 is not None else None
            ),
        }

    return profiles_by_group, run_stats


def _interpolate(profile: dict, load_frac: float, metric: str) -> tuple[float | None, str]:
    points = [
        (frac, vals[metric])
        for frac, vals in profile["lambdas"].items()
        if vals.get(metric) is not None
    ]
    points.sort()
    if not points:
        return None, "missing"

    min_frac, min_val = points[0]
    max_frac, max_val = points[-1]
    if load_frac <= min_frac + 1e-9:
        status = "exact" if abs(load_frac - min_frac) < 1e-9 else "clamped_low"
        return min_val, status
    if load_frac >= max_frac - 1e-9:
        status = "exact" if abs(load_frac - max_frac) < 1e-9 else "clamped_high"
        return max_val, status

    for (lo_frac, lo_val), (hi_frac, hi_val) in zip(points, points[1:]):
        if lo_frac <= load_frac <= hi_frac:
            if abs(load_frac - lo_frac) < 1e-9:
                return lo_val, "exact"
            if abs(load_frac - hi_frac) < 1e-9:
                return hi_val, "exact"
            weight = (load_frac - lo_frac) / (hi_frac - lo_frac)
            return lo_val * (1.0 - weight) + hi_val * weight, "interpolated"

    return None, "missing"


def _nearest_lambda_frac(load_frac: float) -> float:
    return min(LAMBDA_FRACS, key=lambda frac: abs(frac - load_frac))


def _evaluate_mode(profile: dict, demand_rps: float, run_stats: dict) -> dict | None:
    capacity = profile.get("capacity_ips")
    if capacity in (None, 0) or demand_rps <= 0:
        return None
    load_frac = demand_rps / capacity
    if load_frac > 1.0 + 1e-9:
        return None
    load_frac = min(1.0, max(0.0, load_frac))

    delta_w, delta_status = _interpolate(profile, load_frac, "delta_w_median")
    serving_w, serving_status = _interpolate(
        profile, load_frac, "serving_w_median")
    p95_ms, p95_status = _interpolate(
        profile, load_frac, "p95_latency_ms_median")
    achieved_rps, achieved_status = _interpolate(
        profile, load_frac, "achieved_rps_median")
    if delta_w is None or serving_w is None or p95_ms is None:
        return None

    nearest_frac = _nearest_lambda_frac(load_frac)
    stats = run_stats.get(
        (profile["device"], profile["model"], profile["dvfs_mode"], nearest_frac),
        {},
    )
    energy = delta_w / demand_rps
    delta_iqr = stats.get("delta_w_iqr")
    p95_iqr = stats.get("p95_latency_ms_iqr")
    return {
        "device": profile["device"],
        "model": profile["model"],
        "dvfs_mode": profile["dvfs_mode"],
        "dvfs_mode_label": profile.get("dvfs_mode_label", ""),
        "semantic_rank": profile["semantic_rank"],
        "n_modes": profile["n_modes"],
        "capacity_ips": capacity,
        "load_frac_on_mode": load_frac,
        "nearest_measured_lambda_frac": nearest_frac,
        "delta_w_est": delta_w,
        "serving_w_est": serving_w,
        "p95_latency_ms_est": p95_ms,
        "achieved_rps_est": achieved_rps,
        "energy_j_per_inf_est": energy,
        "sla_safe": p95_ms <= L_MAX_MS,
        "interp_status": ",".join(sorted({delta_status, serving_status, p95_status, achieved_status})),
        "delta_w_iqr_nearest": delta_iqr,
        "p95_latency_ms_iqr_nearest": p95_iqr,
        "delta_w_iqr_pct_nearest": _pct(delta_iqr, delta_w),
    }


def _mode_category(eval_row: dict | None) -> str:
    if eval_row is None:
        return ""
    if eval_row["n_modes"] == 1:
        return "fixed"
    if eval_row["semantic_rank"] == 0:
        return "lowest"
    if eval_row["semantic_rank"] == eval_row["n_modes"] - 1:
        return "max"
    return "intermediate"


def _decision_category(row: dict) -> str:
    if row["policy_status"] != "ok":
        return row["policy_status"]
    return row["dynamic_mode_category"]


def _category_facecolor(category: str) -> str:
    return DECISION_CATEGORY_COLORS.get(category, "#ffffff")


def _category_edgecolor(category: str) -> str:
    return DECISION_CATEGORY_EDGES.get(category, "white")


def _category_hatch(category: str) -> str | None:
    return DECISION_CATEGORY_HATCHES.get(category)


def _category_text_color(category: str) -> str:
    return "white" if category in {"intermediate", "max"} else "black"


def _decision_cell_label(row: dict) -> str:
    category = _decision_category(row)
    if category == "no_sla_safe_mode":
        return "no\nSLA"
    if category == "no_feasible_mode":
        return "infeasible"
    if category == "fixed":
        return "fixed"
    return _short_label(row["dynamic_dvfs_mode_label"])


def _decision_legend_handles(include_infeasible: bool = True) -> list[Patch]:
    handles = []
    for category in DECISION_CATEGORY_ORDER:
        if category == "no_feasible_mode" and not include_infeasible:
            continue
        handles.append(Patch(
            facecolor=_category_facecolor(category),
            edgecolor=_category_edgecolor(category),
            hatch=_category_hatch(category),
            label=DECISION_CATEGORY_LABELS[category],
        ))
    return handles


def _choice_fields(prefix: str, choice: dict | None) -> dict:
    fields = {
        f"{prefix}_dvfs_mode": None,
        f"{prefix}_dvfs_mode_label": None,
        f"{prefix}_semantic_rank": None,
        f"{prefix}_mode_category": "",
        f"{prefix}_load_frac_on_mode": None,
        f"{prefix}_nearest_measured_lambda_frac": None,
        f"{prefix}_delta_w_est": None,
        f"{prefix}_serving_w_est": None,
        f"{prefix}_p95_latency_ms_est": None,
        f"{prefix}_energy_j_per_inf_est": None,
        f"{prefix}_sla_safe": None,
        f"{prefix}_interp_status": "",
    }
    if choice is None:
        return fields
    fields.update({
        f"{prefix}_dvfs_mode": choice["dvfs_mode"],
        f"{prefix}_dvfs_mode_label": choice["dvfs_mode_label"],
        f"{prefix}_semantic_rank": choice["semantic_rank"],
        f"{prefix}_mode_category": _mode_category(choice),
        f"{prefix}_load_frac_on_mode": _round(choice["load_frac_on_mode"]),
        f"{prefix}_nearest_measured_lambda_frac": choice["nearest_measured_lambda_frac"],
        f"{prefix}_delta_w_est": _round(choice["delta_w_est"]),
        f"{prefix}_serving_w_est": _round(choice["serving_w_est"]),
        f"{prefix}_p95_latency_ms_est": _round(choice["p95_latency_ms_est"]),
        f"{prefix}_energy_j_per_inf_est": _round(choice["energy_j_per_inf_est"]),
        f"{prefix}_sla_safe": int(choice["sla_safe"]),
        f"{prefix}_interp_status": choice["interp_status"],
    })
    return fields


def build_action_space_rows(profiles_by_group: dict[tuple[str, str], list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for (device, model), profiles in sorted(profiles_by_group.items(), key=lambda kv: _group_sort_key(kv[0])):
        capacities = [p["capacity_ips"] for p in profiles if p.get("capacity_ips") is not None]
        delta_ws = [p["capacity_delta_w"] for p in profiles if p.get("capacity_delta_w") is not None]
        cap_min = min(capacities) if capacities else None
        cap_max = max(capacities) if capacities else None
        delta_min = min(delta_ws) if delta_ws else None
        delta_max = max(delta_ws) if delta_ws else None
        rows.append({
            "device": device,
            "model": model,
            "n_modes": len(profiles),
            "has_dvfs_choice": int(len(profiles) > 1),
            "capacity_min_ips": _round(cap_min),
            "capacity_max_ips": _round(cap_max),
            "capacity_span_ips": _round(cap_max - cap_min if cap_min is not None and cap_max is not None else None),
            "capacity_ratio_max_over_min": _round(cap_max / cap_min if cap_min not in (None, 0) and cap_max is not None else None),
            "capacity_delta_w_min": _round(delta_min),
            "capacity_delta_w_max": _round(delta_max),
            "capacity_delta_w_span": _round(delta_max - delta_min if delta_min is not None and delta_max is not None else None),
            "semantic_mode_order": " -> ".join(
                f"m{p['dvfs_mode']}:{p.get('dvfs_mode_label', '')}"
                for p in profiles
            ),
        })
    return rows


def build_policy_rows(profiles_by_group: dict[tuple[str, str], list[dict]],
                      run_stats: dict) -> list[dict]:
    rows: list[dict] = []
    for (device, model), profiles in sorted(profiles_by_group.items(), key=lambda kv: _group_sort_key(kv[0])):
        max_capacity = max(p["capacity_ips"] for p in profiles if p.get("capacity_ips") is not None)
        max_capacity_profile = max(
            profiles,
            key=lambda p: (
                p["capacity_ips"] if p.get("capacity_ips") is not None else -1,
                p["semantic_rank"],
            ),
        )

        for demand_frac in DEMAND_FRACS:
            demand_rps = demand_frac * max_capacity
            evals = [
                ev for ev in (
                    _evaluate_mode(profile, demand_rps, run_stats)
                    for profile in profiles
                )
                if ev is not None
            ]
            safe_evals = [ev for ev in evals if ev["sla_safe"]]
            # dynamic: oracle policy, feasible min-energy selection.
            # max_perf: max-performance proxy (highest MST mode).
            # capacity_only: capacity-only proxy (mode with highest MST, ignoring energy).
            dynamic = min(
                safe_evals,
                key=lambda ev: (
                    ev["energy_j_per_inf_est"],
                    ev["p95_latency_ms_est"],
                    ev["semantic_rank"],
                ),
                default=None,
            )
            max_perf = _evaluate_mode(max_capacity_profile, demand_rps, run_stats)
            capacity_only = min(
                evals,
                key=lambda ev: (ev["capacity_ips"], ev["semantic_rank"]),
                default=None,
            )

            second_best = None
            if dynamic is not None:
                ordered = sorted(
                    safe_evals,
                    key=lambda ev: (
                        ev["energy_j_per_inf_est"],
                        ev["p95_latency_ms_est"],
                        ev["semantic_rank"],
                    ),
                )
                for ev in ordered:
                    if ev["dvfs_mode"] != dynamic["dvfs_mode"]:
                        second_best = ev
                        break

            status = "ok"
            if not evals:
                status = "no_feasible_mode"
            elif not safe_evals:
                status = "no_sla_safe_mode"

            margin_w = (
                second_best["delta_w_est"] - dynamic["delta_w_est"]
                if dynamic is not None and second_best is not None else None
            )
            margin_pct = _pct(margin_w, dynamic["delta_w_est"] if dynamic else None)
            selected_iqr_pct = dynamic.get("delta_w_iqr_pct_nearest") if dynamic else None
            second_iqr_pct = second_best.get("delta_w_iqr_pct_nearest") if second_best else None
            max_iqr_pct = max(
                [v for v in (selected_iqr_pct, second_iqr_pct) if v is not None],
                default=None,
            )
            if dynamic is None:
                robustness = "no_decision"
            elif second_best is None:
                robustness = "only_one_safe_mode"
            elif max_iqr_pct is not None and margin_pct is not None and margin_pct > max_iqr_pct:
                robustness = "margin_gt_iqr"
            else:
                robustness = "margin_within_iqr"

            row = {
                "device": device,
                "model": model,
                "demand_frac_of_group_max": demand_frac,
                "demand_rps": _round(demand_rps),
                "group_max_capacity_ips": _round(max_capacity),
                "n_modes": len(profiles),
                "n_feasible_modes": len(evals),
                "n_sla_safe_modes": len(safe_evals),
                "policy_status": status,
            }
            row.update(_choice_fields("dynamic", dynamic))
            row.update(_choice_fields("max_perf", max_perf))
            row.update(_choice_fields("capacity_only", capacity_only))
            row.update({
                "dynamic_delta_w_saving_vs_max_perf_pct": _round(
                    _pct_reduction(
                        max_perf["delta_w_est"] if max_perf else None,
                        dynamic["delta_w_est"] if dynamic else None,
                    )
                ),
                "dynamic_energy_saving_vs_max_perf_pct": _round(
                    _pct_reduction(
                        max_perf["energy_j_per_inf_est"] if max_perf else None,
                        dynamic["energy_j_per_inf_est"] if dynamic else None,
                    )
                ),
                "dynamic_delta_w_saving_vs_capacity_only_pct": _round(
                    _pct_reduction(
                        capacity_only["delta_w_est"] if capacity_only else None,
                        dynamic["delta_w_est"] if dynamic else None,
                    )
                ),
                "capacity_only_sla_safe": (
                    int(capacity_only["sla_safe"]) if capacity_only else None
                ),
                "decision_second_best_dvfs_mode": (
                    second_best["dvfs_mode"] if second_best else None
                ),
                "decision_second_best_dvfs_mode_label": (
                    second_best["dvfs_mode_label"] if second_best else None
                ),
                "decision_delta_w_margin": _round(margin_w),
                "decision_delta_w_margin_pct": _round(margin_pct),
                "decision_selected_delta_w_iqr_pct": _round(selected_iqr_pct),
                "decision_second_best_delta_w_iqr_pct": _round(second_iqr_pct),
                "decision_robustness": robustness,
                "full_sweep_policy_relevant": int(
                    dynamic is not None and _mode_category(dynamic) == "intermediate"
                ),
            })
            rows.append(row)

    return rows


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def build_summary(action_rows: list[dict], policy_rows: list[dict]) -> dict:
    ok_rows = [r for r in policy_rows if r["policy_status"] == "ok"]
    savings = [
        r["dynamic_delta_w_saving_vs_max_perf_pct"]
        for r in ok_rows
        if r["dynamic_delta_w_saving_vs_max_perf_pct"] is not None
    ]
    multi_mode_ok_rows = [r for r in ok_rows if r["n_modes"] > 1]
    multi_mode_savings = [
        r["dynamic_delta_w_saving_vs_max_perf_pct"]
        for r in multi_mode_ok_rows
        if r["dynamic_delta_w_saving_vs_max_perf_pct"] is not None
    ]
    multi_mode_nonzero_savings = [
        v for v in multi_mode_savings
        if abs(v) > 1e-9
    ]
    grouped_selected: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in ok_rows:
        grouped_selected[(row["device"], row["model"])].add(row["dynamic_dvfs_mode"])
    category_counts = Counter(
        row["dynamic_mode_category"] if row["policy_status"] == "ok" else row["policy_status"]
        for row in policy_rows
    )
    robustness_counts = Counter(row["decision_robustness"] for row in policy_rows)
    return {
        "artifact": "scheduler_validity",
        "research_question": (
            "Does the current edge testbed expose useful measured actions for "
            "power-aware dynamic scheduling under latency constraints?"
        ),
        "methodology": {
            "demand_model": (
                "For each device/model, external demand is evaluated at "
                "25/50/75/100% of that group's maximum measured MST. "
                "Each mode is evaluated at demand_rps / mode_capacity_ips."
            ),
            "profile_estimation": (
                "Measured lambda profiles at 0.25/0.5/0.75/1.0 are linearly "
                "interpolated for the resulting per-mode load fraction."
            ),
            "dynamic_policy": (
                "Choose the SLA-safe mode with the lowest estimated marginal "
                "AC input energy, equivalent to the lowest delta power at fixed demand."
            ),
            "baseline_max_perf": (
                "Evaluate the maximum-MST mode at the same external demand."
            ),
            "baseline_capacity_only": (
                "Choose the lowest-MST feasible mode using MST only, "
                "without latency or energy awareness."
            ),
            "sla_threshold_ms": L_MAX_MS,
        },
        "counts": {
            "device_model_groups": len(action_rows),
            "multi_mode_groups": sum(1 for r in action_rows if r["n_modes"] > 1),
            "fixed_mode_groups": sum(1 for r in action_rows if r["n_modes"] == 1),
            "policy_eval_rows": len(policy_rows),
            "policy_ok_rows": len(ok_rows),
            "multi_mode_policy_ok_rows": len(multi_mode_ok_rows),
            "multi_mode_nonzero_saving_rows": len(multi_mode_nonzero_savings),
            "policy_status_counts": dict(Counter(r["policy_status"] for r in policy_rows)),
            "dynamic_mode_category_counts": dict(category_counts),
            "load_dependent_selection_groups": sum(
                1 for modes in grouped_selected.values() if len(modes) > 1
            ),
            "full_sweep_policy_relevant_rows": sum(
                r["full_sweep_policy_relevant"] for r in policy_rows
            ),
            "decision_robustness_counts": dict(robustness_counts),
        },
        "benefit_vs_max_perf": {
            "median_delta_w_saving_pct": _round(_median(savings)),
            "min_delta_w_saving_pct": _round(min(savings) if savings else None),
            "max_delta_w_saving_pct": _round(max(savings) if savings else None),
            "multi_mode_median_delta_w_saving_pct": _round(
                _median(multi_mode_savings)),
            "multi_mode_nonzero_median_delta_w_saving_pct": _round(
                _median(multi_mode_nonzero_savings)),
            "multi_mode_nonzero_min_delta_w_saving_pct": _round(
                min(multi_mode_nonzero_savings)
                if multi_mode_nonzero_savings else None),
            "multi_mode_nonzero_max_delta_w_saving_pct": _round(
                max(multi_mode_nonzero_savings)
                if multi_mode_nonzero_savings else None),
        },
        "inputs": {
            "analysis_cells_csv": str(ANALYSIS_CSV),
            "lambda_sweep_csv": str(LAMBDA_SWEEP_CSV),
        },
        "outputs": {
            "action_space_csv": str(ACTION_SPACE_CSV),
            "policy_eval_csv": str(POLICY_EVAL_CSV),
            "summary_json": str(SUMMARY_JSON),
            "figure_dir": str(FIG_DIR),
        },
        "scope_limitations": [
            "This validates measured profiling and policy decision space, not online switching overhead.",
            "Demand-level evaluation uses interpolation between measured lambda profiles.",
            "Invalid MST cells from the full-DVFS build are outside this policy table.",
        ],
    }


def _plot_action_space(action_rows: list[dict]) -> str:
    out = FIG_DIR / "action_space_validity.png"
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for model in MODELS:
        rows = [r for r in action_rows if r["model"] == model]
        if not rows:
            continue
        xs = [r["capacity_ratio_max_over_min"] or 1.0 for r in rows]
        ys = [r["capacity_delta_w_span"] or 0.0 for r in rows]
        sizes = [45 + r["n_modes"] * 10 for r in rows]
        ax.scatter(xs, ys, s=sizes, color=MODEL_COLORS.get(model),
                   alpha=0.78, label=model, edgecolors="white", linewidth=0.7)
        for x, y, r in zip(xs, ys, rows):
            if r["n_modes"] > 1 and (x >= 2.2 or y >= 5.0):
                ax.text(x * 1.03, y + 0.03, r["device"], fontsize=7)
    ax.axvline(1.05, color="0.5", linestyle="--", linewidth=1.0)
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("MST control span (max/min, log scale)")
    ax.set_ylabel("MST-load delta-power span (W)")
    ax.set_title("Action-space validity: power-mode control span per device/model")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(1.07, ylim[1] * 0.92, "no MST\ncontrol",
            fontsize=8, color="0.35", ha="left", va="top")
    ax.text(xlim[0] * 1.08, 0.9, "no delta-power control",
            fontsize=8, color="0.35", ha="left", va="bottom")
    ax.text(xlim[0] * 1.1, ylim[1] * 0.07, "weak-control cluster",
            fontsize=8, color="0.35", ha="left", va="bottom")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _policy_rows_for_model(policy_rows: list[dict], model: str) -> list[dict]:
    return [r for r in policy_rows if r["model"] == model]


def _plot_selection_maps(policy_rows: list[dict]) -> list[str]:
    outputs = []
    for model in MODELS:
        rows = _policy_rows_for_model(policy_rows, model)
        if not rows:
            continue
        devices = sorted({r["device"] for r in rows},
                         key=lambda d: DEVICE_ORDER.get(d, 999))
        by_key = {
            (r["device"], r["demand_frac_of_group_max"]): r
            for r in rows
        }

        fig, ax = plt.subplots(figsize=(7.8, max(3.0, 0.55 * len(devices) + 1.4)))
        for i, device in enumerate(devices):
            for j, demand_frac in enumerate(DEMAND_FRACS):
                row = by_key.get((device, demand_frac))
                if row is None:
                    continue
                category = _decision_category(row)
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=_category_facecolor(category),
                    edgecolor=_category_edgecolor(category),
                    linewidth=1.4 if category in {"intermediate", "no_sla_safe_mode"} else 0.8,
                    hatch=_category_hatch(category),
                ))
                ax.text(j, i, _decision_cell_label(row), ha="center",
                        va="center", fontsize=7.5,
                        color=_category_text_color(category))

        ax.set_xlim(-0.5, len(DEMAND_FRACS) - 0.5)
        ax.set_ylim(len(devices) - 0.5, -0.5)
        ax.set_xticks(range(len(DEMAND_FRACS)))
        ax.set_xticklabels([f"{int(frac * 100)}%" for frac in DEMAND_FRACS])
        ax.set_yticks(range(len(devices)))
        ax.set_yticklabels(devices)
        ax.set_xlabel("external demand (% of group max MST) -> increasing")
        ax.set_title(f"Selected SLA-safe energy-aware mode - {model}")
        ax.set_facecolor("#ffffff")
        ax.grid(False)
        ax.legend(handles=_decision_legend_handles(), fontsize=7,
                  ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=True)
        fig.tight_layout(rect=[0, 0, 0.78, 1])
        out = FIG_DIR / f"selection_map_{_model_tag(model)}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        outputs.append(str(out))
    return outputs


def _plot_savings_vs_max(policy_rows: list[dict]) -> str:
    out = FIG_DIR / "saving_vs_max_performance.png"
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.4 * len(MODELS), 6.0),
                             squeeze=False)
    cmap = plt.get_cmap("YlGn")
    norm = Normalize(vmin=0, vmax=50)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    for ax, model in zip(axes[0], MODELS):
        rows = _policy_rows_for_model(policy_rows, model)
        devices = sorted({r["device"] for r in rows},
                         key=lambda d: DEVICE_ORDER.get(d, 999))
        by_key = {
            (r["device"], r["demand_frac_of_group_max"]): r
            for r in rows
        }
        for i, device in enumerate(devices):
            for j, demand_frac in enumerate(DEMAND_FRACS):
                row = by_key.get((device, demand_frac))
                if row is None:
                    continue
                if row["policy_status"] != "ok":
                    facecolor = _category_facecolor("no_sla_safe_mode")
                    edgecolor = _category_edgecolor("no_sla_safe_mode")
                    hatch = _category_hatch("no_sla_safe_mode")
                    label = "no\nSLA"
                elif row["dynamic_mode_category"] == "fixed":
                    facecolor = _category_facecolor("fixed")
                    edgecolor = "white"
                    hatch = ".."
                    label = "fixed"
                else:
                    value = row["dynamic_delta_w_saving_vs_max_perf_pct"]
                    if value is None or value <= 1e-9:
                        facecolor = "#ffffff"
                        edgecolor = "#cfcfcf"
                        hatch = None
                        label = "0%"
                    else:
                        facecolor = cmap(norm(min(value, 50.0)))
                        edgecolor = "white"
                        hatch = None
                        label = f"{value:.0f}%"
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=0.9,
                    hatch=hatch,
                ))
                text_color = (
                    "white"
                    if row["policy_status"] == "ok"
                    and row["dynamic_mode_category"] != "fixed"
                    and row["dynamic_delta_w_saving_vs_max_perf_pct"] is not None
                    and row["dynamic_delta_w_saving_vs_max_perf_pct"] >= 35
                    else "black"
                )
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=7, color=text_color)

        ax.set_xlim(-0.5, len(DEMAND_FRACS) - 0.5)
        ax.set_ylim(len(devices) - 0.5, -0.5)
        ax.set_title(model)
        ax.set_xticks(range(len(DEMAND_FRACS)))
        ax.set_xticklabels([f"{int(frac * 100)}%" for frac in DEMAND_FRACS])
        ax.set_yticks(range(len(devices)))
        ax.set_yticklabels(devices)
        ax.set_xlabel("external demand (% of group max MST)")
    fig.suptitle("Estimated delta-power saving vs max measured-MST baseline", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.91, top=0.86, bottom=0.12, wspace=0.28)
    cax = fig.add_axes([0.93, 0.22, 0.015, 0.56])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("positive saving (%)")
    fig.legend(handles=[
        Patch(facecolor="#ffffff", edgecolor="#cfcfcf", label="valid 0% saving"),
        Patch(facecolor=_category_facecolor("fixed"), edgecolor="white",
              hatch="..", label="fixed device"),
        Patch(facecolor=_category_facecolor("no_sla_safe_mode"),
              edgecolor=_category_edgecolor("no_sla_safe_mode"),
              hatch=_category_hatch("no_sla_safe_mode"),
              label="no SLA-safe mode"),
    ], fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _plot_full_sweep_necessity(policy_rows: list[dict]) -> str:
    out = FIG_DIR / "full_sweep_necessity.png"
    groups = sorted(
        {(r["device"], r["model"]) for r in policy_rows},
        key=_group_sort_key,
    )
    by_key = {
        (r["device"], r["model"], r["demand_frac_of_group_max"]): r
        for r in policy_rows
    }

    fig, ax = plt.subplots(figsize=(10.5, max(5.4, 0.36 * len(groups) + 1.8)))
    for i, (device, model) in enumerate(groups):
        intermediate_count = 0
        for j, demand_frac in enumerate(DEMAND_FRACS):
            row = by_key.get((device, model, demand_frac))
            if row is None:
                continue
            category = _decision_category(row)
            if category == "intermediate":
                intermediate_count += 1
            ax.add_patch(Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=_category_facecolor(category),
                edgecolor=_category_edgecolor(category),
                linewidth=1.4 if category == "intermediate" else 0.8,
                hatch=_category_hatch(category),
            ))
            short = {
                "fixed": "fixed",
                "lowest": "lowest",
                "intermediate": "interm.",
                "max": "max",
                "no_sla_safe_mode": "no\nSLA",
                "no_feasible_mode": "infeasible",
            }[category]
            ax.text(j, i, short, ha="center", va="center", fontsize=7,
                    color=_category_text_color(category))
        ax.text(len(DEMAND_FRACS) + 0.05, i, f"{intermediate_count}/4",
                ha="left", va="center", fontsize=8,
                fontweight="bold" if intermediate_count else "normal")

    ax.set_xlim(-0.5, len(DEMAND_FRACS) + 0.85)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xticks(list(range(len(DEMAND_FRACS))) + [len(DEMAND_FRACS) + 0.25])
    ax.set_xticklabels(
        [f"{int(frac * 100)}%" for frac in DEMAND_FRACS] + ["intermediate\nused"]
    )
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{device}/{model}" for device, model in groups], fontsize=8)
    ax.set_xlabel("external demand (% of group max MST) -> increasing")
    ax.set_title("Full-sweep necessity across four demand levels")
    ax.legend(handles=_decision_legend_handles(), fontsize=7, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=True)
    ax.grid(False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _plot_decision_margin(policy_rows: list[dict]) -> str:
    out = FIG_DIR / "decision_margin_robustness.png"
    rows = [
        r for r in policy_rows
        if r["policy_status"] == "ok"
        and r["decision_delta_w_margin_pct"] is not None
        and r["decision_selected_delta_w_iqr_pct"] is not None
        and r["decision_second_best_delta_w_iqr_pct"] is not None
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    lim = max(
        [20.0]
        + [r["decision_delta_w_margin_pct"] for r in rows]
        + [
            max(
                r["decision_selected_delta_w_iqr_pct"],
                r["decision_second_best_delta_w_iqr_pct"],
            )
            for r in rows
        ]
    )
    lim = min(max(lim, 20.0), 160.0)
    xs = np.linspace(0, lim, 100)
    ax.fill_between(xs, 0, xs, color="tab:green", alpha=0.08, zorder=0)
    ax.fill_between(xs, xs, lim, color="tab:orange", alpha=0.08, zorder=0)
    for status, color in (
        ("margin_gt_iqr", "tab:green"),
        ("margin_within_iqr", "tab:orange"),
        ("only_one_safe_mode", "tab:blue"),
    ):
        sub = [r for r in rows if r["decision_robustness"] == status]
        if not sub:
            continue
        label = {
            "margin_gt_iqr": "robust: margin > repeat IQR",
            "margin_within_iqr": "uncertain: margin <= repeat IQR",
            "only_one_safe_mode": "only one SLA-safe mode",
        }[status]
        ax.scatter(
            [r["decision_delta_w_margin_pct"] for r in sub],
            [
                max(
                    r["decision_selected_delta_w_iqr_pct"],
                    r["decision_second_best_delta_w_iqr_pct"],
                )
                for r in sub
            ],
            s=38,
            alpha=0.75,
            color=color,
            label=label,
            edgecolors="white",
            linewidth=0.5,
        )
    ax.plot([0, lim], [0, lim], color="0.4", linestyle="--", linewidth=1.0)
    ax.text(lim * 0.62, lim * 0.56, "margin = repeat IQR",
            rotation=37, fontsize=8, color="0.35")
    ax.text(lim * 0.7, lim * 0.18, "robust choice",
            fontsize=9, color="tab:green")
    ax.text(lim * 0.08, lim * 0.86, "tie-like / noise-sensitive",
            fontsize=9, color="tab:orange")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("decision margin: best vs second-best delta power (%)")
    ax.set_ylabel("measurement uncertainty: max repeat IQR (%)")
    ax.set_title("Decision robustness: margin vs measurement uncertainty")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def build_figures(action_rows: list[dict], policy_rows: list[dict]) -> list[str]:
    outputs = [
        _plot_action_space(action_rows),
        _plot_savings_vs_max(policy_rows),
        _plot_full_sweep_necessity(policy_rows),
        _plot_decision_margin(policy_rows),
    ]
    outputs.extend(_plot_selection_maps(policy_rows))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write CSV/JSON and PNG figures")
    args = parser.parse_args()

    profiles_by_group, run_stats = _build_profiles()
    action_rows = build_action_space_rows(profiles_by_group)
    policy_rows = build_policy_rows(profiles_by_group, run_stats)
    summary = build_summary(action_rows, policy_rows)

    print("=" * 80)
    print("Power-aware scheduling validity analysis")
    print("=" * 80)
    print(f"  device/model groups:      {summary['counts']['device_model_groups']}")
    print(f"  multi-mode groups:        {summary['counts']['multi_mode_groups']}")
    print(f"  policy eval rows:         {summary['counts']['policy_eval_rows']}")
    print(f"  policy ok rows:           {summary['counts']['policy_ok_rows']}")
    print(f"  load-dependent groups:    {summary['counts']['load_dependent_selection_groups']}")
    print(f"  intermediate selections:  {summary['counts']['full_sweep_policy_relevant_rows']}")
    print(f"  median saving vs max:     {summary['benefit_vs_max_perf']['median_delta_w_saving_pct']}%")

    if args.apply:
        atomic_csv_write(ACTION_SPACE_CSV, list(action_rows[0].keys()), action_rows)
        atomic_csv_write(POLICY_EVAL_CSV, list(policy_rows[0].keys()), policy_rows)
        figures = build_figures(action_rows, policy_rows)
        summary = dict(summary)
        summary["outputs"] = dict(summary["outputs"])
        summary["outputs"]["figures"] = figures
        atomic_json_write(SUMMARY_JSON, summary)
        print(f"Written: {ACTION_SPACE_CSV} ({len(action_rows)} rows)")
        print(f"Written: {POLICY_EVAL_CSV} ({len(policy_rows)} rows)")
        print(f"Written: {SUMMARY_JSON}")
        print(f"Written figures: {len(figures)} under {FIG_DIR}")
    else:
        print("Dry run. Use --apply to write.")


if __name__ == "__main__":
    main()
