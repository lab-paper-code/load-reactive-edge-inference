"""
Bridge S1 experiment code to the current eep-profiler DB semantics.

Selection policy for batch=1 profile rows:
- prefer reliable incremental-energy rows when available
- prefer canonical incremental energy when available
- fall back to the older incremental-energy field only when canonical is absent
- expose true idle power as the human-facing idle reference
"""

from __future__ import annotations

import sys

from constants import DEVICES, EEP_ROOT

_eep_root = str(EEP_ROOT)
sys.path.insert(0, _eep_root)
from src.db import get_connection


def load_best_batch1_profiles(db_path: str) -> list[dict]:
    """Load one best-power-mode profile row per (device, model) for S1 consumers."""
    conn = get_connection(db_path)
    device_list = ",".join(f"'{d}'" for d in DEVICES)
    rows = conn.execute(f"""
        SELECT device, model, dvfs_mode,
               latency_ms_p50, latency_ms_p95, latency_ms_std,
               throughput_ips, p_wall_avg_w,
               watts_idle, watts_idle_true, p_plateau_w,
               energy_inc_per_inf_j, energy_inc_canonical_per_inf_j,
               incremental_source, flag_incremental_unreliable
        FROM profiles
        WHERE batch_size = 1
          AND COALESCE(energy_inc_canonical_per_inf_j, energy_inc_per_inf_j) IS NOT NULL
          AND device IN ({device_list})
        ORDER BY device, model,
                 flag_incremental_unreliable ASC,
                 COALESCE(energy_inc_canonical_per_inf_j, energy_inc_per_inf_j) ASC
    """).fetchall()
    conn.close()

    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["device"], row["model"])
        if key in best:
            continue

        selected_e_inc = row["energy_inc_canonical_per_inf_j"]
        selection = "canonical"
        if selected_e_inc is None:
            selected_e_inc = row["energy_inc_per_inf_j"]
            selection = "fallback"

        best[key] = {
            "device": row["device"],
            "model": row["model"],
            "dvfs_mode": row["dvfs_mode"],
            "latency_ms_p50": row["latency_ms_p50"],
            "latency_ms_p95": row["latency_ms_p95"],
            "latency_ms_std": row["latency_ms_std"],
            "throughput_ips": row["throughput_ips"],
            "p_wall_avg_w": row["p_wall_avg_w"],
            "watts_idle": row["watts_idle"],
            "watts_idle_true": row["watts_idle_true"],
            "idle_power_w": (
                row["watts_idle_true"]
                if row["watts_idle_true"] is not None
                else row["watts_idle"]
            ),
            "p_plateau_w": row["p_plateau_w"],
            "energy_inc_per_inf_j": selected_e_inc,
            "energy_inc_raw_per_inf_j": row["energy_inc_per_inf_j"],
            "energy_inc_canonical_per_inf_j": row["energy_inc_canonical_per_inf_j"],
            "e_inc_selection": selection,
            "incremental_source": row["incremental_source"] or "none",
            "flag_incremental_unreliable": int(row["flag_incremental_unreliable"]),
            "extra_energy_reliable": not bool(row["flag_incremental_unreliable"]),
        }

    return list(best.values())
