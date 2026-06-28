"""
S1-1: Analytical Capacity Estimate (M/G/1 based)

Uses existing EEP tensor (batch=1, best-DVFS) data to estimate SLO-feasible sustained
throughput C(d,m; L_max) for each (device, model) pair.

Three models for range bracketing:
  - C_md1:  M/D/1 (deterministic service), optimistic upper bound
  - C_mg1:  M/G/1 (measured variance), center estimate
  - C_mm1:  M/M/1 (exponential service), conservative lower bound

Usage:
    python3 measurement/stage2-serving/capacity_estimate.py [--l-max 100] [--db data/stage1/eep_profiler.db]
"""

from __future__ import annotations

import argparse
import csv
from math import log

from constants import DB_PATH, CSV_CAPACITY_ANALYTICAL, L_MAX_MS as DEFAULT_L_MAX
from profiler_bridge import load_best_batch1_profiles

# --- Analytical model constants ---
RHO_UPPER_BOUND = 0.999        # Binary search upper limit for utilization
BISECT_ITERATIONS = 100        # Iterations for rho binary search
MM1_P95_FACTOR = log(20)       # ln(1/0.05) = ln(20) ≈ 2.996 for M/M/1 p95

def best_dvfs_profiles(db_path: str) -> list[dict]:
    """Load batch=1 best-DVFS rows using current profiler DB semantics."""
    return load_best_batch1_profiles(db_path)


def solve_rho_max(mu_s: float, cs2: float, l_max_s: float,
                  l_service_s: float) -> float:
    """Binary search: max ρ where approximate p95 response time ≤ l_max_s.

    M/G/1 mean response time:
        E[T] = E[S] + ρ·E[S]·(1 + C_s²) / (2·(1 - ρ))

    p95 approximation (interpolation between service p95 and queuing tail):
        p95(ρ) ≈ E[T](ρ) × (1 + (ratio - 1) × ρ)
    where ratio = service_p95 / service_mean captures inherent variability.
    At ρ→0, p95 → E[S] × ratio ≈ service_p95.
    At ρ→1, p95 → E[T] × ratio (heavy-traffic exponential tail scaling).

    Empirical validation (S1-3):
        measured/analytical ratios = 0.17~0.99, median 0.67.
        xavier/eff-b4 is an extreme outlier at 0.17; most pairs are 1.2x~1.8x
        overestimated. Downstream consumers apply ANALYTICAL_CORRECTION=0.67
        (see constants.py) to calibrate these estimates.
    """
    if l_service_s >= l_max_s:
        return 0.0  # SLO violation: service time alone exceeds L_max

    lo, hi = 0.0, RHO_UPPER_BOUND
    for _ in range(BISECT_ITERATIONS):
        rho = (lo + hi) / 2
        # M/G/1 mean response time
        e_s = 1.0 / mu_s
        e_t = e_s + rho * e_s * (1 + cs2) / (2 * (1 - rho))
        # p95 scaling: at low ρ → service p95, at high ρ → heavy-traffic tail
        ratio = max(l_service_s / e_s, 1.0)
        p95_approx = e_t * (1 + (ratio - 1) * rho)
        if p95_approx <= l_max_s:
            lo = rho
        else:
            hi = rho
    return lo


def estimate_capacity(profiles: list[dict], l_max_ms: float) -> list[dict]:
    """Estimate analytical capacity for each (device, model) pair."""
    l_max_s = l_max_ms / 1000.0
    results = []

    for p in profiles:
        l_p50_ms = p["latency_ms_p50"]
        l_p95_ms = p["latency_ms_p95"]
        l_std_ms = p["latency_ms_std"]

        # Service parameters (seconds)
        # NOTE: L_p50 is used as a proxy for mean service time E[S].
        # Under isolated sequential profiling, p50 ≈ mean for symmetric
        # service time distributions. This is a heuristic, not exact.
        e_s = l_p50_ms / 1000.0            # proxy service time from p50
        mu_s = 1.0 / e_s                    # service rate (req/s)
        l_p95_s = l_p95_ms / 1000.0
        var_s = (l_std_ms / 1000.0) ** 2
        cs2 = var_s / (e_s ** 2)            # coefficient of variation squared

        # ── M/D/1 (C_s² = 0): optimistic upper bound ──
        rho_md1 = solve_rho_max(mu_s, 0.0, l_max_s, e_s)
        c_md1 = mu_s * rho_md1

        # ── M/G/1 (measured C_s²): center estimate ──
        rho_mg1 = solve_rho_max(mu_s, cs2, l_max_s, l_p95_s)
        c_mg1 = mu_s * rho_mg1

        # ── M/M/1 (C_s² = 1): conservative lower bound ──
        # Exact: p95 = -ln(0.05) / (μ - λ), solve for λ
        c_mm1_exact = max(0.0, mu_s - MM1_P95_FACTOR / l_max_s)

        results.append({
            "device": p["device"],
            "model": p["model"],
            "dvfs_mode": p["dvfs_mode"],
            "L_p50_ms": round(l_p50_ms, 2),
            "L_p95_ms": round(l_p95_ms, 2),
            "L_std_ms": round(l_std_ms, 3),
            "Cs2": round(cs2, 4),
            "mu_ips": round(mu_s, 1),
            "C_md1": round(c_md1, 1),
            "C_mg1": round(c_mg1, 1),
            "C_mm1": round(c_mm1_exact, 1),
            "C_naive": round(mu_s, 1),  # = 1000/L_p50, no queuing
            "E_inc_J": round(p["energy_inc_per_inf_j"], 4),
        })

    return results


def print_table(results: list[dict]):
    hdr = (f"{'Device':<12} {'Model':<20} {'DVFS':>4} {'L_p50':>7} {'L_p95':>7} "
           f"{'Cs2':>6} {'mu':>7} {'C_md1':>7} {'C_mg1':>7} {'C_mm1':>7} {'E_inc':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: (x["device"], x["model"])):
        print(f"{r['device']:<12} {r['model']:<20} {r['dvfs_mode']:>4} "
              f"{r['L_p50_ms']:>7.2f} {r['L_p95_ms']:>7.2f} {r['Cs2']:>6.3f} "
              f"{r['mu_ips']:>7.1f} {r['C_md1']:>7.1f} {r['C_mg1']:>7.1f} "
              f"{r['C_mm1']:>7.1f} {r['E_inc_J']:>8.4f}")


def save_csv(results: list[dict], path: str):
    if not results:
        return
    keys = results[0].keys()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {path} ({len(results)} rows)")


def main():
    parser = argparse.ArgumentParser(description="S1-1: Analytical Capacity Estimate")
    parser.add_argument("--l-max", type=float, default=DEFAULT_L_MAX, help="SLO p95 latency (ms)")
    parser.add_argument("--db", type=str,
                        default=str(DB_PATH),
                        help="Path to eep_profiler.db")
    parser.add_argument("--out", type=str,
                        default=str(CSV_CAPACITY_ANALYTICAL))
    args = parser.parse_args()

    print(f"DB: {args.db}")
    print(f"L_max: {args.l_max} ms (p95 SLO)\n")

    profiles = best_dvfs_profiles(args.db)
    print(f"Loaded {len(profiles)} (device, model) pairs at batch=1, best DVFS\n")

    results = estimate_capacity(profiles, args.l_max)
    print_table(results)
    save_csv(results, args.out)


if __name__ == "__main__":
    main()
