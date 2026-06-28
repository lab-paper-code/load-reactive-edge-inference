"""Per-job measurement runners.

Orchestrates one MeasurementJob through the shared
DVFS → deploy → validate → sanity → power-trace → load → teardown pipeline,
with two load variants at Phase 4:

  - capacity-finding (`run_capacity_single_job`): binary search via
    load_generator --find-capacity
  - lambda-sweep (`run_lambda_sweep_single_job`): fixed-rate Poisson load at
    λ_frac × C for (fracs × runs) cells in one server session

Phases 0-3 and 5 are identical between the two and delegate to
server_lifecycle + power_capture.

Consumers: remeasure.py CLI dispatcher.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from remote import RemoteError, ssh, file_sha256
from constants import L_MAX_MS
from device_topology import DEVICE_HOSTS, SHELLY_HOSTNAMES, device_port
from measurement_jobs import (MeasurementJob, LOAD_GENERATOR, INFER_SERVER,
                              condition_label, condition_tag, job_err, job_tag,
                              make_output_path,
                              make_lambda_output_path, model_src)
from power_capture import start_power_trace, save_power_trace
from server_lifecycle import (
    SSH_CMD_TIMEOUT,
    apply_dvfs_setup, deploy_server, wait_and_validate,
    sanity_check_latency, kill_all_infer_servers,
)

# --- Measurement defaults ---
MEASURE_DURATION_S = 60        # load_generator --duration
MEASURE_WARMUP_S = 5           # load_generator --warmup
WARM_IDLE_DURATION_S = 8.0
INTER_RUN_COOLDOWN_S = 6.0

# Cached at import time; same for all jobs in a run.
_INFER_SERVER_SHA256: str = file_sha256(str(INFER_SERVER))
# v4 = legacy bs=1 (batch implied). v5 = batch-aware: a `batch_size` field is
# required so the per-image energy divisor (ΔW/(rate×N)) is never silently
# defaulted to N=1. Existing v4 raws stay accepted as bs=1 (FULL_DVFS_RAW_SCHEMA_MIN);
# only freshly-measured full_dvfs raws are stamped at the current version.
FULL_DVFS_RAW_SCHEMA_MIN = 4
FULL_DVFS_RAW_SCHEMA_VERSION = 5


def inject_job_metadata(job: MeasurementJob, result: dict) -> dict:
    """Attach the requested experiment condition to a raw result dict."""
    label = condition_label(job)
    result.update({
        "device": job.device,
        "model": job.model,
        "policy": job.policy,
        "dvfs_mode": job.dvfs_mode,
        "dvfs_mode_label": label,
        "condition_tag": condition_tag(job),
        "run_family": "full_dvfs" if job.dvfs_mode is not None else "policy_keyed",
        "batch_size": job.batch_size,
    })
    result["operating_condition"] = {
        "kind": "dvfs_mode" if job.dvfs_mode is not None else "policy_key",
        "condition_tag": condition_tag(job),
        "policy_label": job.policy,
        "dvfs_mode": job.dvfs_mode,
        "dvfs_mode_label": label,
        "batch_size": job.batch_size,
        "expected_ep": job.expected_ep,
        "expected_nvpmodel": job.expected_nvpmodel or None,
        "expected_gpu_freq_mhz": job.expected_gpu_freq_mhz or None,
    }
    if job.dvfs_mode is not None:
        try:
            schema = int(result.get("schema_version", 0))
        except (TypeError, ValueError):
            schema = 0
        result["schema_version"] = max(schema, FULL_DVFS_RAW_SCHEMA_VERSION)
    return result


def inject_deploy_provenance(job: MeasurementJob, result: dict) -> dict:
    """Attach infer_server + model hash + runtime version to a result dict."""
    result["deploy_provenance"] = {
        "infer_server_sha256": _INFER_SERVER_SHA256,
        "model_filename": job.model_file,
        "model_sha256": file_sha256(str(model_src(job))),
    }
    try:
        if job.model_file.endswith(".rknn"):
            rt_ver = ssh(job.device,
                         "python3 -c 'import rknnlite; print(rknnlite.__version__)'",
                         timeout=SSH_CMD_TIMEOUT, check=False)
            if rt_ver:
                result["deploy_provenance"]["rknnlite_version"] = rt_ver
        else:
            ort_ver = ssh(job.device,
                          "python3 -c 'import onnxruntime; print(onnxruntime.__version__)'",
                          timeout=SSH_CMD_TIMEOUT, check=False)
            if ort_ver:
                result["deploy_provenance"]["ort_version"] = ort_ver
    except Exception:
        pass
    return result


# --- Phase 4 variants: load drivers ---

async def run_find_capacity_measurement(job: MeasurementJob, target_url: str,
                                        out_path: Path) -> dict:
    """Run load_generator in binary-search mode. Returns result dict."""
    print(f"  Measuring (mu_hint={job.mu_hint})...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(LOAD_GENERATOR),
        "--target", target_url,
        "--find-capacity", "--l-max", str(L_MAX_MS),
        "--mu-hint", str(job.mu_hint),
        "--duration", str(MEASURE_DURATION_S), "--warmup", str(MEASURE_WARMUP_S),
        "--out", str(out_path),
    ]
    if job.batch_size != 1:                       # bs=1 argv byte-identical to before
        cmd += ["--batch-size", str(job.batch_size)]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode()

    if proc.returncode != 0:
        print(f"  load_generator exited with rc={proc.returncode}")
        print(output[-500:] if output else "  (no output)")
        return job_err(job, "load_generator_failed", returncode=proc.returncode)

    if not out_path.exists():
        print(f"  FAILED: no output file despite rc=0")
        print(output[-500:] if output else "  (no output)")
        return job_err(job, "no_output_file")

    try:
        with open(out_path) as f:
            result = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  FAILED: invalid JSON in {out_path.name}: {e}")
        return job_err(job, "invalid_json", detail=str(e), output=str(out_path))

    cap = result.get("capacity_ips")
    rng = result.get("capacity_range")
    if (not isinstance(cap, (int, float))
            or not isinstance(rng, list) or len(rng) != 2
            or not all(isinstance(v, (int, float)) for v in rng)
            or rng[0] > rng[1]):
        print(f"  FAILED: invalid result schema (capacity_ips={cap}, "
              f"capacity_range={rng})")
        return job_err(job, "invalid_result_schema", output=str(out_path))

    inject_job_metadata(job, result)
    inject_deploy_provenance(job, result)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  RESULT: {cap} ips [{rng[0]}-{rng[1]}]")
    print(f"  Saved: {out_path.name}")

    return {"status": "ok", "capacity_ips": cap,
            "job": job_tag(job), "output": str(out_path)}


async def run_fixed_rate_load(job: MeasurementJob, target_url: str,
                              target_rps: float, duration_s: int,
                              warmup_s: int, out_path: Path) -> dict:
    """Drive load_generator at a fixed Poisson rate for one sustained run.

    Returns {"status": "ok", "raw": dict, "output": str} on success; caller
    is responsible for enrichment (lambda_frac, run_idx) and persistence.
    """
    print(f"  Measuring fixed rate ({target_rps:.1f} rps, {duration_s}s)...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(LOAD_GENERATOR),
        "--target", target_url,
        "--rate", f"{target_rps:.2f}",
        "--duration", str(duration_s),
        "--warmup", str(warmup_s),
        "--out", str(out_path),
    ]
    if job.batch_size != 1:                       # bs=1 argv byte-identical to before
        cmd += ["--batch-size", str(job.batch_size)]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode()

    if proc.returncode != 0:
        print(f"  load_generator exited with rc={proc.returncode}")
        print(output[-500:] if output else "  (no output)")
        return job_err(job, "load_generator_failed", returncode=proc.returncode)

    if not out_path.exists():
        print(f"  FAILED: no output file despite rc=0")
        return job_err(job, "no_output_file")

    try:
        with open(out_path) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        return job_err(job, "invalid_json", detail=str(e), output=str(out_path))

    return {"status": "ok", "raw": raw, "output": str(out_path)}


# --- Shared Phase 3.5 helper: open a power-trace window with explicit warm idle ---

async def _open_power_trace_window(device: str) -> tuple:
    """Start a power trace and observe the warm-idle window.

    Returns (collector, warm_idle_end_ts). Either can be None if the device
    has no Shelly plug.
    """
    collector = start_power_trace(device)
    warm_idle_end_ts = None
    if collector:
        sample_wait_deadline = time.time() + 10.0
        while not collector.samples and time.time() < sample_wait_deadline:
            await asyncio.sleep(0.25)
        await asyncio.sleep(WARM_IDLE_DURATION_S)
        warm_idle_end_ts = time.time()
        print(f"  POWER: trace started ({collector.hostname}), "
              f"warm idle {WARM_IDLE_DURATION_S}s collected "
              f"({len(collector.samples)} samples so far)")
    return collector, warm_idle_end_ts


def _power_trace_guard(job: MeasurementJob, collector) -> dict | None:
    """Fail fast when a Shelly-instrumented device has no power samples."""
    if job.device not in SHELLY_HOSTNAMES:
        return None
    if collector is None:
        return job_err(job, "power_trace_unavailable",
                       error_type="instrumentation",
                       detail="Shelly is configured for this device, but the trace collector did not start")
    if collector.samples:
        return None
    errors = getattr(collector, "errors", None)
    collector.stop()
    return job_err(job, "power_trace_unavailable",
                   error_type="instrumentation",
                   detail="Shelly trace collector started but produced no warm-idle samples",
                   collector_errors=errors)


# --- Capacity-finding per-job ---

async def run_capacity_single_job(job: MeasurementJob,
                                  dry_run: bool = False) -> dict:
    """DVFS → deploy → wait → sanity → power trace → find-capacity → cleanup."""
    if job.device not in DEVICE_HOSTS:
        return job_err(job, f"unknown device '{job.device}'; not in DEVICE_HOSTS",
                       error_type="config")

    ip = DEVICE_HOSTS[job.device]
    target_url = f"http://{ip}:{device_port(job.device)}"
    out_path = make_output_path(job)

    print(f"\n{'─'*60}")
    print(f"  {job_tag(job)} → {out_path.name}")

    if dry_run:
        print(f"  [DRY RUN] would measure at mu_hint={job.mu_hint}")
        return {"status": "dry_run", "job": job_tag(job)}

    await apply_dvfs_setup(job)
    server_launch_ts = await deploy_server(job)

    try:
        err = await wait_and_validate(job, target_url, server_launch_ts)
        if err:
            return err
        err = await sanity_check_latency(job, target_url)
        if err:
            return err

        power_collector, warm_idle_end_ts = await _open_power_trace_window(job.device)
        power_err = _power_trace_guard(job, power_collector)
        if power_err:
            return power_err

        result = await run_find_capacity_measurement(job, target_url, out_path)

        power_status = "skipped"
        if power_collector and result.get("status") == "ok" and out_path.exists():
            trace = power_collector.stop()
            with open(out_path) as f:
                raw = json.load(f)
            power_status = save_power_trace(
                trace, warm_idle_end_ts, power_collector, out_path, raw)
        elif power_collector:
            power_collector.stop()

        if result.get("status") == "ok":
            result["power_status"] = power_status
        return result

    finally:
        try:
            kill_all_infer_servers(job.device)
        except RemoteError:
            print(f"  WARNING: post-measurement cleanup failed on {job.device}")


# --- lambda-sweep per-job (multi-cell in one server session) ---

async def _run_one_lambda_cell(job: MeasurementJob, target_url: str,
                               lambda_frac: float, target_rps: float,
                               run_idx: int) -> dict:
    """One (λ_frac, run_idx) measurement: power trace → fixed-rate → stop trace."""
    out_path = make_lambda_output_path(job, lambda_frac, run_idx)

    power_collector, warm_idle_end_ts = await _open_power_trace_window(job.device)
    power_err = _power_trace_guard(job, power_collector)
    if power_err:
        return power_err

    load_result = await run_fixed_rate_load(
        job, target_url, target_rps,
        duration_s=MEASURE_DURATION_S, warmup_s=MEASURE_WARMUP_S,
        out_path=out_path,
    )

    if load_result.get("status") != "ok":
        if power_collector:
            power_collector.stop()
        return load_result

    # Enrich raw JSON with run metadata for T5 builder join.
    raw = load_result["raw"]
    raw.update({
        "run_type": "lambda_sweep",
        "lambda_frac": lambda_frac,
        "target_rps": round(target_rps, 2),
        "run_idx": run_idx,
        "l_max_ms": L_MAX_MS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    inject_job_metadata(job, raw)
    inject_deploy_provenance(job, raw)

    # Synthesize a `confirmation` block for power_capture.save_power_trace.
    # Single-cell λ-sweep → one run in the passing round.
    raw["confirmation"] = {
        "rounds": [{
            "round": 1, "pass": True,
            "runs": [{
                "active_start_ts": raw.get("active_start_ts"),
                "active_end_ts": raw.get("active_end_ts"),
            }],
        }],
    }

    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)

    power_status = "skipped"
    if power_collector and out_path.exists():
        trace = power_collector.stop()
        with open(out_path) as f:
            raw = json.load(f)
        power_status = save_power_trace(
            trace, warm_idle_end_ts, power_collector, out_path, raw)

    achieved = raw.get("achieved_rps", 0)
    p95 = raw.get("p95_ms", 0)
    print(f"  achieved={achieved:.1f} ips  p95={p95:.1f}ms  "
          f"power={power_status}  saved={out_path.name}")

    return {
        "status": "ok", "job": job_tag(job),
        "lambda_frac": lambda_frac, "run_idx": run_idx,
        "achieved_rps": achieved, "p95_ms": p95,
        "power_status": power_status, "output": str(out_path),
    }


async def run_lambda_sweep_single_job(job: MeasurementJob,
                                      capacity_ips: float,
                                      lambda_fracs: tuple[float, ...],
                                      n_runs: int,
                                      dry_run: bool = False) -> list[dict]:
    """All (λ_frac × run_idx) cells for one pair in one server session.

    Phase 0-3 and 5 mirror `run_capacity_single_job`; Phase 4 is replaced by
    a (fracs x runs) loop; each cell owns its own Shelly trace window.
    """
    if job.device not in DEVICE_HOSTS:
        return [job_err(job, f"unknown device '{job.device}'")]

    ip = DEVICE_HOSTS[job.device]
    target_url = f"http://{ip}:{device_port(job.device)}"

    print(f"\n{'═'*70}")
    print(f"  PAIR: {job_tag(job)}  C_measured={capacity_ips:.1f} ips")
    print(f"  λ-sweep plan: fracs={list(lambda_fracs)}  runs={n_runs}  "
          f"→ {len(lambda_fracs) * n_runs} cell runs")

    if dry_run:
        for lf in lambda_fracs:
            for r in range(n_runs):
                print(f"    [DRY] λ={lf:.2f}  target={lf*capacity_ips:.1f} ips  run={r}")
        return [{"status": "dry_run", "job": job_tag(job)}]

    await apply_dvfs_setup(job)
    server_launch_ts = await deploy_server(job)

    results: list[dict] = []
    try:
        err = await wait_and_validate(job, target_url, server_launch_ts)
        if err:
            return [err]
        err = await sanity_check_latency(job, target_url)
        if err:
            return [err]

        for lambda_frac in lambda_fracs:
            target_rps = lambda_frac * capacity_ips
            for run_idx in range(n_runs):
                r = await _run_one_lambda_cell(job, target_url,
                                               lambda_frac, target_rps, run_idx)
                results.append(r)

                if r.get("status") != "ok":
                    print(f"  FAIL at λ={lambda_frac:.2f} run={run_idx}; "
                          f"skipping rest of this pair")
                    return results

                await asyncio.sleep(INTER_RUN_COOLDOWN_S)

    finally:
        try:
            kill_all_infer_servers(job.device)
        except RemoteError:
            print(f"  WARNING: post-sweep cleanup failed on {job.device}")

    return results
