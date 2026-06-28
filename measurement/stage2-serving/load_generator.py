"""
S1-2: Poisson Load Generator with Binary Search Capacity Finder

Runs on meterless-a. Sends open-loop Poisson arrivals to a remote infer_server.py
and records per-request latency.

Modes:
  1) Fixed rate: load test at a specified lambda for duration_s seconds
  2) Binary search: automatically find the maximum lambda where p95 = L_max

Usage:
    # Fixed rate test
    python3 load_generator.py --target http://orin:8090 --rate 50 --duration 60

    # Binary search for capacity
    python3 load_generator.py --target http://orin:8090 --find-capacity \
        --l-max 100 --mu-hint 38.0 --duration 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from math import log
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np

# --- Constants ---
INF_LATENCY_MS = 1e6           # Sentinel for failed/timed-out requests (SLO violation)
POISSON_FLOOR = 1e-12          # Numerical floor for Poisson inter-arrival
HTTP_CLIENT_TIMEOUT = 30       # aiohttp per-request timeout (seconds)
SEARCH_PRECISION = 0.05        # Binary search convergence ratio (high-low < precision*low)
DEFAULT_N_CONFIRM = 3          # Boundary confirmation repeat count
DEFAULT_COOLDOWN_S = 15.0      # Cool-down after overload probe (seconds)
CONFIRM_BACKOFF = 0.95         # Capacity reduction factor on failed confirmation
MAX_BACKOFF_ROUNDS = 5         # Max confirmation-backoff iterations before giving up


async def poisson_load_test(
    target_url: str,
    rate_rps: float,
    duration_s: float,
    warmup_s: float = 5.0,
    batch_size: int = 1,
) -> dict:
    """Send Poisson-distributed requests at given rate, collect latencies."""
    url = target_url.rstrip("/") + "/infer"
    latencies = []
    errors = 0
    total_sent = 0
    t_global_start = time.monotonic()
    t_warmup_end = t_global_start + warmup_s
    t_deadline = t_global_start + warmup_s + duration_s
    # Wall-clock anchors for power correlation (monotonic → wall-clock offset)
    _wall_offset = time.time() - time.monotonic()
    active_start_ts_unix = t_warmup_end + _wall_offset
    active_end_ts_unix = t_deadline + _wall_offset

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=HTTP_CLIENT_TIMEOUT)
    ) as session:

        async def send_one(send_time: float):
            nonlocal errors, total_sent
            total_sent += 1
            try:
                t0 = time.monotonic()
                async with session.post(url) as resp:
                    t1 = time.monotonic()
                    lat_ms = (t1 - t0) * 1000
                    if send_time >= t_warmup_end:
                        if resp.status == 200:
                            latencies.append(lat_ms)
                        else:
                            errors += 1
                            latencies.append(INF_LATENCY_MS)
            except Exception:
                if send_time >= t_warmup_end:
                    errors += 1
                    latencies.append(INF_LATENCY_MS)

        pending = set()
        t_next = t_global_start
        while t_next < t_deadline:
            now = time.monotonic()
            if t_next <= now:
                task = asyncio.create_task(send_one(t_next))
                pending.add(task)
                task.add_done_callback(pending.discard)
                # Poisson inter-arrival
                interval = -log(max(random.random(), POISSON_FLOOR)) / rate_rps
                t_next += interval
            else:
                await asyncio.sleep(t_next - now)

        # Wait for remaining in-flight requests
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    if not latencies:
        return {
            "rate_rps": rate_rps,
            "duration_s": duration_s,
            "n_requests": 0,
            "errors": errors,
            "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "mean_ms": 0,
            "achieved_rps": 0,
            "batch_size": batch_size,
            "active_start_ts": round(active_start_ts_unix, 3),
            "active_end_ts": round(time.time(), 3),
        }

    # Finalize wall-clock end (actual, not planned)
    active_end_ts_unix = time.time()

    lat = np.array(latencies)
    return {
        "rate_rps": round(rate_rps, 2),
        "duration_s": duration_s,
        "n_requests": len(latencies),
        "errors": errors,
        "mean_ms": round(float(lat.mean()), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "std_ms": round(float(lat.std()), 2),
        "achieved_rps": round(len(latencies) / duration_s, 2),
        "batch_size": batch_size,
        "active_start_ts": round(active_start_ts_unix, 3),
        "active_end_ts": round(active_end_ts_unix, 3),
    }


async def collect_device_status(target_url: str) -> dict:
    """Fetch /status from infer_server for measurement provenance."""
    url = target_url.rstrip("/") + "/status"
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return {}


async def find_capacity(
    target_url: str,
    l_max_ms: float,
    mu_hint: float,
    duration_s: float = 60.0,
    warmup_s: float = 5.0,
    precision: float = SEARCH_PRECISION,
    max_iter: int = 10,
    n_confirm: int = DEFAULT_N_CONFIRM,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
    batch_size: int = 1,
) -> dict:
    """Binary search for max lambda where p95 <= l_max_ms.

    Search features:
      - Auto-expands upper bound if mu_hint is too low
      - Cool-down after overload probes to avoid thermal carry-over
      - Confirms boundary with n_confirm repeat runs, reports median
      - Records device hardware status for provenance
    """
    low = 0.0
    high = mu_hint
    history = []

    # Collect device status before measurement
    device_status = await collect_device_status(target_url)
    if device_status:
        print(f"  Device: {device_status.get('hostname', '?')}, "
              f"EP: {device_status.get('ep', '?')}")
        gpu_info = []
        if "nvpmodel" in device_status:
            gpu_info.append(device_status["nvpmodel"])
        if "gpu_max_freq_hz" in device_status:
            gpu_info.append(f"GPU max {device_status['gpu_max_freq_hz']//1_000_000}MHz")
        if "gpu_clock_mhz" in device_status:
            gpu_info.append(f"GPU {device_status['gpu_clock_mhz']}MHz")
        if "gpu_temp_c" in device_status:
            gpu_info.append(f"{device_status['gpu_temp_c']}°C")
        if gpu_info:
            print(f"  HW state: {', '.join(gpu_info)}")
    print()

    # Quick sanity: test at low rate first
    result = await poisson_load_test(target_url, max(1.0, mu_hint * 0.1),
                                     duration_s, warmup_s, batch_size=batch_size)
    if result["p95_ms"] > l_max_ms:
        print(f"  WARNING: p95={result['p95_ms']:.1f}ms > L_max={l_max_ms}ms "
              f"even at λ={mu_hint*0.1:.1f}. Server may be unhealthy.")

    # Expand upper bound until a fail point is found
    probe = await poisson_load_test(target_url, high, duration_s, warmup_s,
                                    batch_size=batch_size)
    while probe["p95_ms"] <= l_max_ms:
        low = high
        high *= 2
        print(f"  Expanding upper bound: λ={high:.1f} (previous {low:.1f} still feasible)")
        probe = await poisson_load_test(target_url, high, duration_s, warmup_s,
                                        batch_size=batch_size)
        probe["iteration"] = -1
        history.append(probe)

    print(f"  Upper bound set: fail at λ={high:.1f} (p95={probe['p95_ms']:.1f}ms)")
    # Cool-down after overload to avoid thermal carry-over
    print(f"  Cool-down {cooldown_s:.0f}s after overload probe...")
    await asyncio.sleep(cooldown_s)

    for i in range(max_iter):
        mid = (low + high) / 2
        if mid < 0.5:
            break

        result = await poisson_load_test(target_url, mid, duration_s, warmup_s,
                                         batch_size=batch_size)
        result["iteration"] = i
        history.append(result)

        p95 = result["p95_ms"]
        achieved = result["achieved_rps"]
        status = "OK" if p95 <= l_max_ms else "OVER"
        print(f"  [{i+1}/{max_iter}] λ={mid:>7.1f} rps → p95={p95:>7.1f}ms "
              f"(mean={result['mean_ms']:>6.1f}, achieved={achieved:>.1f}, "
              f"n={result['n_requests']}) [{status}]")

        if p95 <= l_max_ms:
            low = mid
        else:
            high = mid
            # Brief cool-down after overload iteration
            await asyncio.sleep(cooldown_s / 3)

        if (high - low) < precision * low and low > 0:
            print(f"  Converged: C ∈ [{low:.1f}, {high:.1f}]")
            break

    # Confirm boundary with repeat measurements.
    # If confirmation fails at the binary-search boundary, back off
    # multiplicatively and re-confirm until a confirmed rate is found.
    # Report the highest tested arrival rate that passes repeated SLO confirmation.
    confirmation_rounds = []
    capacity_achieved_rps = None
    capacity_confirmed = False
    capacity_selection = "unconfirmed"

    if n_confirm > 1 and low > 0.5:
        candidate = low
        boundary_lambda = low  # original binary-search boundary

        for backoff_round in range(MAX_BACKOFF_ROUNDS):
            label = (f"boundary λ={candidate:.1f}"
                     if backoff_round == 0
                     else f"backoff #{backoff_round} λ={candidate:.1f}")
            print(f"\n  Confirming at {label} ({n_confirm} runs)...")

            confirm_runs = []
            for r in range(n_confirm):
                cr = await poisson_load_test(target_url, candidate,
                                             duration_s, warmup_s,
                                             batch_size=batch_size)
                confirm_runs.append(cr)
                tag = "OK" if cr["p95_ms"] <= l_max_ms else "OVER"
                print(f"    [{r+1}/{n_confirm}] p95={cr['p95_ms']:>7.1f}ms "
                      f"achieved={cr['achieved_rps']:.1f} [{tag}]")

            p95_vals = [c["p95_ms"] for c in confirm_runs]
            achieved_vals = [c["achieved_rps"] for c in confirm_runs]
            median_p95 = sorted(p95_vals)[len(p95_vals) // 2]
            median_achieved = sorted(achieved_vals)[len(achieved_vals) // 2]
            passed = median_p95 <= l_max_ms

            confirmation_rounds.append({
                "round": backoff_round,
                "candidate_lambda": round(candidate, 1),
                "p95_values": [round(v, 1) for v in p95_vals],
                "achieved_rps_values": [round(v, 1) for v in achieved_vals],
                "median_p95": round(median_p95, 1),
                "pass": passed,
                "runs": [{
                    "active_start_ts": c.get("active_start_ts"),
                    "active_end_ts": c.get("active_end_ts"),
                } for c in confirm_runs],
            })

            if passed:
                low = candidate
                capacity_achieved_rps = median_achieved
                capacity_confirmed = True
                capacity_selection = ("confirmed" if backoff_round == 0
                                      else "confirmed_after_backoff")
                print(f"  Confirmed: C={candidate:.1f} ips"
                      + (f" (backoff #{backoff_round} from {boundary_lambda:.1f})"
                         if backoff_round > 0 else ""))
                break

            print(f"  Confirmation FAILED: median p95={median_p95:.1f}ms "
                  f"> {l_max_ms}ms")
            candidate *= CONFIRM_BACKOFF

        else:
            # Exhausted all backoff rounds without confirmation.
            # candidate has already been multiplied by CONFIRM_BACKOFF after
            # the last failure, so it sits one step below the last tested rate.
            low = candidate
            capacity_selection = "backoff_unconfirmed"
            print(f"  Exhausted {MAX_BACKOFF_ROUNDS} backoff rounds; "
                  f"unconfirmed, fallback lambda={low:.1f}")

    confirmation = {
        "boundary_lambda": round(boundary_lambda if confirmation_rounds
                                 else low, 1),
        "rounds": confirmation_rounds,
        "rule": "median_p95 <= l_max",
    } if confirmation_rounds else None

    # Collect device status after measurement (for thermal comparison)
    device_status_after = await collect_device_status(target_url)

    return {
        "schema_version": 4,
        "batch_size": batch_size,
        "capacity_ips": round(low, 1),
        "capacity_confirmed": capacity_confirmed,
        "capacity_selection": capacity_selection,
        "capacity_achieved_rps": round(capacity_achieved_rps, 1) if capacity_achieved_rps else None,
        "capacity_range": [round(low, 1), round(high, 1)],
        "l_max_ms": l_max_ms,
        "mu_hint": mu_hint,
        "duration_s": duration_s,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device_status_before": device_status,
        "device_status_after": device_status_after,
        "history": history,
        "confirmation": confirmation,
    }


async def amain(args):
    if args.find_capacity:
        print(f"Binary search: target={args.target}, L_max={args.l_max}ms, "
              f"mu_hint={args.mu_hint} ips, duration={args.duration}s\n")
        result = await find_capacity(
            args.target, args.l_max, args.mu_hint,
            duration_s=args.duration, warmup_s=args.warmup,
            batch_size=args.batch_size,
        )
        print(f"\n{'='*60}")
        print(f"RESULT: C(d,m; {args.l_max}ms) = {result['capacity_ips']} ips")
        print(f"  Range: [{result['capacity_range'][0]}, {result['capacity_range'][1]}]")

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {args.out}")
    else:
        print(f"Fixed rate: target={args.target}, λ={args.rate} rps, "
              f"duration={args.duration}s\n")
        result = await poisson_load_test(
            args.target, args.rate, args.duration, args.warmup,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, indent=2))

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved: {args.out}")


def main():
    parser = argparse.ArgumentParser(description="S1-2: Poisson Load Generator")
    parser.add_argument("--target", required=True, help="http://host:port")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")

    # Fixed rate mode
    parser.add_argument("--rate", type=float, default=10.0, help="Arrival rate (rps)")

    # Binary search mode
    parser.add_argument("--find-capacity", action="store_true")
    parser.add_argument("--l-max", type=float, default=100.0, help="SLO p95 (ms)")
    parser.add_argument("--mu-hint", type=float, default=None,
                        help="Service rate hint (ips), upper bound for search")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Server-side batch width N (recorded for provenance; "
                             "load is body-less, the server builds the N-wide tensor)")

    args = parser.parse_args()

    if args.find_capacity and (args.mu_hint is None or args.mu_hint <= 0):
        parser.error("--find-capacity requires --mu-hint with a positive value")
    if not args.find_capacity and args.rate <= 0:
        parser.error("--rate must be positive")

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
