"""
S1-Capacity shared constants.

Single source of truth for measured maximum sustainable throughput (MST), model accuracy,
SLO-infeasible pairs, and analytical correction factors.
All values sourced from results/derived/capacity_measured.csv unless noted.
"""
from __future__ import annotations

from pathlib import Path

# --- Shared paths ---
S1_DIR = Path(__file__).resolve().parent
REPO_ROOT = S1_DIR.parent                      # analysis/ -> repo root
EEP_ROOT = REPO_ROOT / "measurement" / "stage1-isolated"
DB_PATH = REPO_ROOT / "data" / "stage1" / "eep_profiler.db"
MODELS_DIR = REPO_ROOT / "models"
RKNN_MODELS_DIR = MODELS_DIR / "rknn"
RESULTS_DIR = S1_DIR / "results"
CSV_DIR = RESULTS_DIR / "derived"
CSV_CAPACITY_ANALYTICAL = CSV_DIR / "capacity_analytical.csv"
CSV_CAPACITY_MEASURED = CSV_DIR / "capacity_measured.csv"
CSV_CAPACITY_DVFS = CSV_DIR / "capacity_dvfs_overrides.csv"

# --- Power-mode policy names ---
# Policy A (e_inc): power mode that minimizes incremental energy per inference above idle.
# Policy B (capacity): power mode that maximizes sustained throughput (MST).
# e_inc: incremental energy per inference above idle (J/inf above idle baseline).
POLICY_EINC = "e_inc"       # Policy A: E_inc-minimizing power mode
POLICY_CAPACITY = "capacity" # Policy B: MST-maximizing power mode
DVFS_POLICIES = [POLICY_EINC, POLICY_CAPACITY]

# --- Device/model lists ---
DEVICES = ["orin", "orin-nano", "xavier", "jetson", "rasp5", "orangepi", "orangepi-npu", "lattepanda", "gpu-server"]
MODELS = ["mobilenet-v2-050", "mobilenet-v2-100", "efficientnet-b4"]

# --- Server-side static batch widths (Phase-1 batch extension, server-only) ---
# Per-call N-wide forward on the GPU server. Maps device → batch grid; a device
# absent from this map is measured at bs=1 only. Only the GPU server (gpu-server)
# carries bs>1 in Phase 1; the rest of the fleet stays bs=1 until Phase 2.
SERVER_BATCH_SIZES: dict[str, tuple[int, ...]] = {
    "gpu-server": (1, 2, 4, 8),
}

# --- Model accuracy (ImageNet top-1) ---
ACCURACY: dict[str, float] = {
    "mobilenet-v2-050": 0.517,
    "mobilenet-v2-100": 0.591,
    "efficientnet-b4": 0.732,
}

# --- SLO-infeasible pairs (service time > L_max at best power mode) ---
SLO_INFEASIBLE: set[tuple[str, str]] = {
    ("jetson", "efficientnet-b4"),       # S1 load-test: p95 278ms
    ("lattepanda", "efficientnet-b4"),   # S1 load-test: p95 309ms
    ("orangepi", "efficientnet-b4"),     # S1 load-test: p95 462ms
    # ("orangepi", "mobilenet-v2-100") evening clean rerun confirmed feasible at 10.6 ips
    ("rasp5", "efficientnet-b4"),        # S1 load-test: p95 593ms
    ("orangepi-npu", "efficientnet-b4"),   # S1 load-test: p95=111ms at 0.5rps (single-inference >> 100ms)
}

# --- Measured maximum sustainable throughput (MST) (from S1-3 binary search, CSV authoritative) ---
# ips: inferences per second.
# Policy A: E_inc-minimizing power mode, 2026-04-18 (full-fleet power-annotated rerun).
# All devices re-measured in 20260418T Phase 2 campaign; values sourced from
# capacity_measured.csv rebuilt from power-annotated authoritative runs.
MEASURED_CAPACITY_EINC: dict[tuple[str, str], float] = {
    ("gpu-server", "efficientnet-b4"): 90.8,
    ("gpu-server", "mobilenet-v2-050"): 585.0,
    ("gpu-server", "mobilenet-v2-100"): 507.7,
    ("jetson", "mobilenet-v2-050"): 98.9,
    ("jetson", "mobilenet-v2-100"): 50.0,
    ("lattepanda", "mobilenet-v2-050"): 132.3,
    ("lattepanda", "mobilenet-v2-100"): 48.1,
    ("orangepi", "mobilenet-v2-050"): 102.4,
    ("orangepi", "mobilenet-v2-100"): 43.8,
    ("orangepi-npu", "efficientnet-b4"): 0.0,    # SLO-infeasible (p95=111ms)
    ("orangepi-npu", "mobilenet-v2-050"): 222.7,
    ("orangepi-npu", "mobilenet-v2-100"): 154.1,
    ("orin", "efficientnet-b4"): 21.9,
    ("orin", "mobilenet-v2-050"): 390.6,
    ("orin", "mobilenet-v2-100"): 318.8,
    ("rasp5", "mobilenet-v2-050"): 43.0,
    ("rasp5", "mobilenet-v2-100"): 13.0,
    ("xavier", "efficientnet-b4"): 1.4,
    ("xavier", "mobilenet-v2-050"): 212.5,
    ("xavier", "mobilenet-v2-100"): 142.2,
}

# Policy B: MST-maximizing power mode, 2026-04-18.
# Only orin/effb4 (MAXN) and gpu-server/effb4 (1545MHz) have distinct MST-max
# measurements; every other pair's hardware collapses to the same power mode, so
# Policy B MST equals Policy A MST.
MEASURED_CAPACITY_CAP: dict[tuple[str, str], float] = {
    ("gpu-server", "efficientnet-b4"): 135.6,  # cap-max power mode (1545MHz)
    ("gpu-server", "mobilenet-v2-050"): 585.0,  # same as e_inc
    ("gpu-server", "mobilenet-v2-100"): 507.7,  # same as e_inc
    ("jetson", "mobilenet-v2-050"): 98.9,  # same as e_inc
    ("jetson", "mobilenet-v2-100"): 50.0,  # same as e_inc
    ("lattepanda", "mobilenet-v2-050"): 132.3,  # same as e_inc
    ("lattepanda", "mobilenet-v2-100"): 48.1,  # same as e_inc
    ("orangepi", "mobilenet-v2-050"): 102.4,  # same as e_inc
    ("orangepi", "mobilenet-v2-100"): 43.8,  # same as e_inc
    ("orangepi-npu", "efficientnet-b4"): 0.0,  # same as e_inc (SLO-infeasible)
    ("orangepi-npu", "mobilenet-v2-050"): 222.7,  # same as e_inc (NPU, no DVFS)
    ("orangepi-npu", "mobilenet-v2-100"): 154.1,  # same as e_inc (NPU, no DVFS)
    ("orin", "efficientnet-b4"): 81.6,  # cap-max power mode (MAXN)
    ("orin", "mobilenet-v2-050"): 390.6,  # same as e_inc
    ("orin", "mobilenet-v2-100"): 318.8,  # same as e_inc
    ("rasp5", "mobilenet-v2-050"): 43.0,  # same as e_inc
    ("rasp5", "mobilenet-v2-100"): 13.0,  # same as e_inc
    ("xavier", "efficientnet-b4"): 1.4,  # same as e_inc
    ("xavier", "mobilenet-v2-050"): 212.5,  # same as e_inc
    ("xavier", "mobilenet-v2-100"): 142.2,  # same as e_inc
}

# E_inc values for MST-max power mode (different from best-E_inc power mode)
E_INC_CAP_OVERRIDE: dict[tuple[str, str], float] = {
    ("orin", "efficientnet-b4"): 0.197,     # MAXN: positive, not artifact
    ("gpu-server", "efficientnet-b4"): 1.623, # 1545MHz: +18% vs mode 4
}

# Default alias (backward compat)
MEASURED_CAPACITY = MEASURED_CAPACITY_EINC

# --- Analytical model correction ---
# Median of measured/analytical ratios (excluding xavier eff-b4 outlier 0.12):
# sorted(0.68, 0.74, 0.74, 0.79, 1.07, 1.61) → median = (0.74 + 0.79) / 2 = 0.765
ANALYTICAL_CORRECTION: float = 0.765

# --- SLO threshold ---
L_MAX_MS: float = 100.0

# --- E_inc clamping floor ---
# Negative E_inc = measurement artifact (idle power > active power in
# lightweight-model-on-GPU profiles where inference < 2ms).
# Clamp to small positive value to preserve LP ordering without
# creating 0-cost ties that cause objective degeneracy.
E_INC_FLOOR: float = 1e-4
