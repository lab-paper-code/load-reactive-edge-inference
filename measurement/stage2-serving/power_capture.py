"""Shelly AC input power trace orchestration for measurement runs.

Encapsulates the lifecycle of a DirectShellyTraceCollector: start (0.5s polling),
stop, slice per-run windows, compute warm-idle + per-confirmation-run summaries,
and persist into the raw measurement JSON under a canonical `power_trace` block.

Consumers: measurement_runner.run_single_job, lambda_sweep.run_lambda_sweep_job.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from constants import EEP_ROOT
from device_topology import SHELLY_HOSTNAMES

_eep_root = str(EEP_ROOT)
_eep_src = str(EEP_ROOT / "src")
sys.path.insert(0, _eep_root)
sys.path.insert(0, _eep_src)
from power_reader import PrometheusReader
from power_trace import DirectShellyTraceCollector, _time_weighted_average
from config import profiling_defaults

# Lazy-init Prometheus reader, created at first use, not at import time.
_prom_reader: PrometheusReader | None = None

# Power-trace analysis config loaded from
# measurement/stage1-isolated/configs/profiling.yaml to avoid drift from
# hardcoded duplicates.
_POWER_TRACE_CFG = profiling_defaults()


def get_prom_reader() -> PrometheusReader:
    global _prom_reader
    if _prom_reader is None:
        _prom_reader = PrometheusReader()
        print(f"  Prometheus: {_prom_reader.url}")
    return _prom_reader


def start_power_trace(device: str) -> DirectShellyTraceCollector | None:
    """Start direct Shelly trace collection (0.5s polling) for a device."""
    shelly_host = SHELLY_HOSTNAMES.get(device)
    if not shelly_host:
        return None
    try:
        prom = get_prom_reader()
        poll_s = _POWER_TRACE_CFG.get("direct_poll_interval_s", 0.5)
        collector = DirectShellyTraceCollector(prom, shelly_host, poll_interval_s=poll_s)
        collector.start()
        return collector
    except Exception as e:
        print(f"  POWER: trace start failed for {device}: {e}")
        return None


def slice_trace(trace: list[dict], t0: float, t1: float) -> list[dict]:
    """Extract samples within [t0, t1] from a power trace."""
    return [s for s in trace if t0 <= float(s["ts"]) <= t1]


def save_power_trace(trace: list[dict], warm_idle_end_ts: float | None,
                     collector, out_path: Path,
                     raw: dict) -> str:
    """Analyze and save power trace to raw JSON. Returns power status string.

    Structure:
      power_trace.raw_samples: full sample sequence (for re-analysis)
      power_trace.warm_idle: explicit idle baseline (server ready, no requests)
      power_trace.confirmation_runs: per-run power observations

    `raw` is the already-parsed measurement JSON dict (avoids double read).
    """
    if not trace:
        print(f"  POWER: no samples collected")
        return "no_samples"

    if warm_idle_end_ts is None:
        return "no_idle_baseline"

    # 1. Warm idle window: [first_sample_ts, warm_idle_end_ts]
    trace_start_ts = float(trace[0]["ts"])
    idle_samples = slice_trace(trace, trace_start_ts, warm_idle_end_ts)
    idle_avg, idle_count = _time_weighted_average(idle_samples)
    idle_watts = [float(s["watts"]) for s in idle_samples]

    warm_idle = {
        "start_ts": trace_start_ts,
        "end_ts": warm_idle_end_ts,
        "duration_s": round(warm_idle_end_ts - trace_start_ts, 1),
        "avg_watts": round(idle_avg, 2) if idle_avg else None,
        "median_watts": round(statistics.median(idle_watts), 2) if idle_watts else None,
        "std_watts": round(statistics.pstdev(idle_watts), 3) if len(idle_watts) >= 2 else None,
        "sample_count": idle_count,
    }

    # 2. Per-confirmation-run power
    confirmation_runs = []
    conf = raw.get("confirmation") or {}
    passing_round = None
    for rnd in conf.get("rounds", []):
        if rnd.get("pass"):
            passing_round = rnd

    if passing_round and passing_round.get("runs"):
        for i, run_info in enumerate(passing_round["runs"]):
            t0 = run_info.get("active_start_ts")
            t1 = run_info.get("active_end_ts")
            if not t0 or not t1:
                confirmation_runs.append({
                    "run_index": i,
                    "avg_watts": None,
                    "sample_count": 0,
                    "error": "missing timestamps",
                })
                continue
            run_samples = slice_trace(trace, t0, t1)
            run_avg, run_count = _time_weighted_average(run_samples)
            confirmation_runs.append({
                "run_index": i,
                "active_start_ts": t0,
                "active_end_ts": t1,
                "duration_s": round(t1 - t0, 1),
                "avg_watts": round(run_avg, 2) if run_avg else None,
                "sample_count": run_count,
            })

    # 3. Compute delta per run (warm idle median as baseline)
    idle_baseline = warm_idle.get("median_watts")
    for run in confirmation_runs:
        if run.get("avg_watts") is not None and idle_baseline is not None:
            run["delta_watts"] = round(run["avg_watts"] - idle_baseline, 2)
        else:
            run["delta_watts"] = None

    # 4. Summary
    valid_runs = [r for r in confirmation_runs if r.get("avg_watts") is not None]
    if valid_runs:
        run_watts = sorted([r["avg_watts"] for r in valid_runs])
        median_serving = run_watts[len(run_watts) // 2]
        delta_watts = sorted([r["delta_watts"] for r in valid_runs
                              if r.get("delta_watts") is not None])
        median_delta = delta_watts[len(delta_watts) // 2] if delta_watts else None
    else:
        median_serving = None
        median_delta = None

    # 5. Persist
    raw["power_trace"] = {
        "schema_version": 1,
        "source": "direct_shelly",
        "shelly_hostname": collector.hostname,
        "poll_interval_s": collector.poll_interval_s,
        "trace_start_ts": trace_start_ts,
        "total_sample_count": len(trace),
        "collector_errors": collector.errors,
        "warm_idle": warm_idle,
        "passing_round": passing_round["round"] if passing_round else None,
        "confirmation_runs": confirmation_runs,
        "summary": {
            "warm_idle_median_watts": warm_idle.get("median_watts"),
            "serving_median_watts": round(median_serving, 2) if median_serving else None,
            "delta_median_watts": round(median_delta, 2) if median_delta else None,
            "n_valid_runs": len(valid_runs),
        },
        "raw_samples": [{"ts": round(s["ts"], 3), "watts": round(float(s["watts"]), 2)}
                        for s in trace],
    }

    status = "ok" if valid_runs else "no_valid_runs"
    raw["power_trace"]["status"] = status

    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)

    if valid_runs:
        idle_text = (
            f"{warm_idle.get('median_watts'):.2f}W"
            if warm_idle.get("median_watts") is not None else "n/a")
        delta_text = (
            f"{median_delta:.2f}W"
            if median_delta is not None else "n/a")
        print(f"  POWER: {len(trace)} samples, "
              f"idle={idle_text}, "
              f"serving={median_serving:.1f}W, "
              f"delta={delta_text} "
              f"({len(valid_runs)} runs)")
    else:
        print(f"  POWER: {len(trace)} samples but no valid confirmation runs")

    return status
