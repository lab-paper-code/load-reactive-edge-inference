"""Remote infer_server lifecycle: deploy, wait-and-validate, sanity, teardown.

Every measurement runner (capacity find, lambda sweep) shares the same server
lifecycle on each device. This module owns that full lifecycle so both
runners can reuse the exact same discipline (liveness probe with probe/dead
disambiguation, 30s dead-confirm window, DVFS validation via /status, etc.).

Consumers: measurement_runner.run_single_job, lambda_sweep.run_lambda_sweep_job.
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

from remote import RemoteError, ssh, scp_to
from device_topology import (DEVICE_HOSTS, PHYSICAL_HOST, REMOTE_WORK_DIR,
                             device_port)
from measurement_jobs import (MeasurementJob, INFER_SERVER, job_err,
                              model_src, expected_model_name)

# --- Timing (seconds) ---
REBOOT_MAX_WAIT = 150          # max wait for device to become operational
REBOOT_STABILIZE_DELAY = 10    # post-reboot stabilization before jetson_clocks
POLL_INTERVAL = 5              # /status polling interval
SERVER_SOFT_CAP = 900          # absolute max wait for TRT engine build (15min)
SERVER_STALE_THRESHOLD = 120   # max stale rounds × poll_interval before hung
SERVER_LAUNCH_TIMEOUT = 5      # SSH timeout for background server start
STALE_SERVER_CLOCK_DRIFT = 60  # max acceptable clock drift for stale detection
PROGRESS_LOG_INTERVAL = 30     # seconds between "still building..." messages
GPU_CLOCK_LOCK_RATIO = 0.9     # cur_freq / max_freq threshold for clock-lock warning
GPU_FREQ_TOLERANCE_MHZ = 50    # allowed deviation from expected GPU freq
SSH_CMD_TIMEOUT = 10           # default timeout for non-critical SSH commands
NVPMODEL_SET_TIMEOUT = 15      # timeout for nvpmodel -m (may trigger reboot)
REBOOT_WAIT_GENERIC = 60       # seconds to wait after non-Jetson device reboot

# --- Sanity probe ---
SANITY_LATENCY_RATIO = 10      # single-request latency > ratio × expected → degraded
SANITY_REQUESTS = 3            # number of probe requests for sanity check


def server_proc_pattern(device: str) -> str:
    # [i] breaks self-match: the pkill/pgrep cmdline contains the literal
    # string '[i]nfer_server', which does NOT match the regex [i]nfer_server
    # (character class expects 'i', not '[i]').
    return f"python3 [i]nfer_server.py.*--port {device_port(device)}"


async def wait_for_device_operational(device: str, is_jetson: bool = True,
                                      max_wait: int = REBOOT_MAX_WAIT) -> None:
    """Wait until device is fully operational after reboot.

    Checks command executability only. Nvpmodel mode validation is done later
    via /status (single authority for DVFS state).
    """
    stages: list[tuple[str, ...]] = [
        ("ssh",        lambda: ssh(device, "echo up", timeout=POLL_INTERVAL)),
        ("filesystem", lambda: ssh(device, f"mkdir -p {REMOTE_WORK_DIR}",
                                   timeout=SSH_CMD_TIMEOUT)),
        ("python",     lambda: ssh(device, "python3 --version",
                                   timeout=SSH_CMD_TIMEOUT)),
    ]
    if is_jetson:
        stages.append(
            ("nvpmodel_cmd", lambda: ssh(device, "nvpmodel -q",
                                         timeout=SSH_CMD_TIMEOUT)))

    last_error = ""
    for stage_name, check_fn in stages:
        for attempt in range(max_wait // POLL_INTERVAL):
            await asyncio.sleep(POLL_INTERVAL)
            try:
                check_fn()
                print(f"    [{stage_name}] ready (attempt {attempt+1})")
                break
            except Exception as e:
                last_error = str(e)
                if (attempt + 1) * POLL_INTERVAL % PROGRESS_LOG_INTERVAL == 0:
                    print(f"    [{stage_name}] still waiting... "
                          f"({(attempt+1)*POLL_INTERVAL}s)")
        else:
            raise RemoteError("reboot_timeout", device,
                              f"stage '{stage_name}' not ready after {max_wait}s",
                              -1, last_error[:200])

    print(f"    Stabilization wait {REBOOT_STABILIZE_DELAY}s...")
    await asyncio.sleep(REBOOT_STABILIZE_DELAY)
    for attempt in range(6):
        try:
            ssh(device, "sudo jetson_clocks", timeout=SSH_CMD_TIMEOUT)
            print(f"    jetson_clocks applied")
            break
        except Exception as e:
            if attempt == 5:
                raise
            print(f"    jetson_clocks retry {attempt + 1}/5 after SSH instability...")
            await asyncio.sleep(POLL_INTERVAL)


async def check_server_liveness(device: str,
                                remote_log: str) -> tuple[bool, bool, int, str]:
    """One SSH round-trip: (probe_ok, alive, log_size, tail). Non-blocking.

    probe_ok=False means the SSH probe itself failed (timeout, transient network
    error, etc.). The server state is *unknown* in that case; the caller must
    not conflate an unknown probe with confirmed death.
    """
    pattern = server_proc_pattern(device)
    cmd = (f"pgrep -f '{pattern}' >/dev/null 2>&1 && echo alive || echo dead; "
           f"stat -c%s {remote_log} 2>/dev/null || echo 0; "
           f"tail -5 {remote_log} 2>/dev/null")
    try:
        out = await asyncio.to_thread(ssh, device, cmd, POLL_INTERVAL, False)
        lines = out.strip().splitlines()
        alive = lines[0].strip() == "alive" if lines else False
        log_size = (int(lines[1].strip())
                    if len(lines) > 1 and lines[1].strip().isdigit() else 0)
        tail = "\n".join(lines[2:]) if len(lines) > 2 else ""
        return True, alive, log_size, tail
    except Exception:
        return False, False, 0, ""


def kill_and_verify_port(device: str, port: int) -> None:
    """Kill existing server processes and verify port is free.

    Escalation: SIGTERM → wait → port check → SIGKILL → port check → abort.
    Combined into 2 SSH calls (down from 5) to reduce round-trip overhead.
    """
    pattern = server_proc_pattern(device)
    out = ssh(device,
              f"pkill -f '{pattern}' 2>/dev/null; sleep 3; "
              f"fuser {port}/tcp 2>/dev/null || echo __FREE__",
              check=False)
    if "__FREE__" in out:
        return
    print(f"  Port {port} still occupied after SIGTERM; escalating to SIGKILL")
    out = ssh(device,
              f"pkill -9 -f '{pattern}' 2>/dev/null; "
              f"fuser -k {port}/tcp 2>/dev/null; sleep 2; "
              f"fuser {port}/tcp 2>/dev/null || echo __FREE__",
              check=False)
    if "__FREE__" in out:
        return
    raise RemoteError("port_cleanup", device,
                      f"port {port} still occupied after SIGKILL",
                      -1, f"fuser: {out.strip()}")


def kill_all_infer_servers(device: str) -> None:
    """Kill ALL infer_server.py processes on device, regardless of port.

    Prevents resource contention when multiple servers ran on different ports
    (e.g. orangepi CPU on 8090 + NPU on 8091 sharing the same board).
    """
    ssh(device,
        "pkill -f 'python3 [i]nfer_server.py' 2>/dev/null; "
        "sleep 1; "
        "pkill -9 -f 'python3 [i]nfer_server.py' 2>/dev/null",
        check=False)


async def deploy_server(job: MeasurementJob) -> float:
    """Deploy infer_server + model to device, launch server, return launch ts."""
    port = device_port(job.device)
    kill_all_infer_servers(job.device)
    kill_and_verify_port(job.device, port)
    ssh(job.device, f"mkdir -p {REMOTE_WORK_DIR}")
    scp_to(str(INFER_SERVER), job.device, f"{REMOTE_WORK_DIR}/infer_server.py")
    scp_to(str(model_src(job)), job.device, f"{REMOTE_WORK_DIR}/{job.model_file}")

    server_launch_ts = time.time()
    try:
        bs_flag = f" --batch-size {job.batch_size}" if job.batch_size != 1 else ""
        r = subprocess.run(
            ["ssh", job.device,
             f"cd {REMOTE_WORK_DIR} && nohup python3 infer_server.py {job.model_file} "
             f"--port {port}{bs_flag} > server.log 2>&1 < /dev/null &"],
            capture_output=True, text=True, timeout=SERVER_LAUNCH_TIMEOUT,
        )
        if r.returncode != 0:
            raise RemoteError("remote_cmd", job.device,
                              "server launch", r.returncode, r.stderr.strip())
    except subprocess.TimeoutExpired:
        # SSH can hang even with nohup ... & due to FD inheritance.
        # Readiness is verified via /status polling below; this timeout is expected.
        print(f"  SSH launch timed out ({SERVER_LAUNCH_TIMEOUT}s); "
              f"expected FD hang, continuing to readiness check")

    return server_launch_ts


async def wait_and_validate(job: MeasurementJob, target_url: str,
                            server_launch_ts: float) -> dict | None:
    """Adaptive wait for server ready + validate /status. Returns error dict or None."""
    import aiohttp
    print(f"  Waiting for server (adaptive)...")
    prev_log_size = 0
    stale_rounds = 0
    elapsed = 0
    status = None
    stale_max_rounds = SERVER_STALE_THRESHOLD // POLL_INTERVAL
    remote_log = f"{REMOTE_WORK_DIR}/server.log"

    http_client = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=POLL_INTERVAL))
    try:
        consec_dead = 0
        # 6 consecutive confirmed-dead probes = 30s at POLL_INTERVAL=5s.
        # Cold TRT builds can produce no log output for 15-25s; tighter
        # thresholds misfire on slow cold starts.
        DEAD_CONFIRM_COUNT = 6
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

            try:
                async with http_client.get(f"{target_url}/status") as resp:
                    if resp.status == 200:
                        status = await resp.json()
                        break
            except Exception:
                pass

            probe_ok, proc_alive, log_size, log_tail = await check_server_liveness(
                job.device, remote_log)

            # Death accounting rules:
            # - probe_ok=False  → SSH hiccup. State unknown; don't confirm death.
            # - proc_alive or log_growing → clearly alive; reset counter.
            # - else            → confirmed dead; increment counter.
            log_growing = log_size > prev_log_size
            if not probe_ok:
                pass
            elif proc_alive or log_growing:
                consec_dead = 0
            else:
                consec_dead += 1
                if consec_dead >= DEAD_CONFIRM_COUNT:
                    print(f"  Server process died (after {elapsed}s, "
                          f"{consec_dead} consecutive dead signals)")
                    if log_tail:
                        print(f"  Last log: {log_tail[:200]}")
                    return job_err(job, "server_crashed", elapsed_s=elapsed)

            if not probe_ok:
                # Transient SSH failure; skip stale_rounds / prev_log_size update.
                pass
            elif log_size > prev_log_size:
                stale_rounds = 0
                if elapsed % PROGRESS_LOG_INTERVAL < POLL_INTERVAL:
                    print(f"    [{elapsed}s] server building... "
                          f"(log {log_size//1024}KB, growing)")
                prev_log_size = log_size
            else:
                # Log not growing but process may be alive. TRT build may
                # produce no log output. Only count stale if process is alive.
                stale_rounds += 1
                if elapsed % PROGRESS_LOG_INTERVAL < POLL_INTERVAL:
                    print(f"    [{elapsed}s] process alive, log stable "
                          f"({log_size//1024}KB, stale {stale_rounds * POLL_INTERVAL}s)")
                prev_log_size = log_size

            if elapsed >= SERVER_SOFT_CAP:
                print(f"  Server build exceeded soft cap ({elapsed}s)")
                return job_err(job, "server_build_timeout", elapsed_s=elapsed)
            if stale_rounds >= stale_max_rounds and not proc_alive:
                print(f"  Server log stale for "
                      f"{stale_rounds * POLL_INTERVAL}s; "
                      f"assuming hung after {elapsed}s")
                return job_err(job, "server_hung", elapsed_s=elapsed)
    finally:
        await http_client.close()

    # --- Validate /status fields ---
    actual_ep = status.get("ep", "")
    actual_model = status.get("model", "")
    expected_model = expected_model_name(job)
    print(f"  Server ready after {elapsed}s: model={actual_model}, EP={actual_ep}")

    if actual_ep != job.expected_ep:
        print(f"  ABORT: expected EP {job.expected_ep}, got {actual_ep}")
        return job_err(job, "ep_mismatch",
                       expected_ep=job.expected_ep, actual_ep=actual_ep)
    if actual_model != expected_model:
        print(f"  ABORT: expected model {expected_model}, got {actual_model}")
        return job_err(job, "model_mismatch",
                       expected_model=expected_model, actual_model=actual_model)

    start_ts = status.get("server_start_ts", 0)
    if start_ts < server_launch_ts - STALE_SERVER_CLOCK_DRIFT:
        print(f"  ABORT: stale server (start_ts={start_ts:.0f}, "
              f"expected after {server_launch_ts:.0f})")
        return job_err(job, "stale_server")

    if job.expected_nvpmodel:
        actual_nv = status.get("nvpmodel", "")
        if job.expected_nvpmodel not in actual_nv:
            print(f"  ABORT: expected nvpmodel '{job.expected_nvpmodel}', "
                  f"got '{actual_nv}'")
            return job_err(job, "dvfs_mismatch",
                           detail=f"nvpmodel: expected {job.expected_nvpmodel}, got {actual_nv}")

    if job.expected_gpu_freq_mhz:
        actual_mhz = status.get("gpu_clock_mhz", 0)
        if actual_mhz and abs(actual_mhz - job.expected_gpu_freq_mhz) > GPU_FREQ_TOLERANCE_MHZ:
            print(f"  ABORT: expected GPU ~{job.expected_gpu_freq_mhz}MHz, "
                  f"got {actual_mhz}MHz")
            return job_err(job, "dvfs_mismatch",
                           detail=f"gpu_freq: expected {job.expected_gpu_freq_mhz}, got {actual_mhz}")

    # Server batch width must equal the requested N: the --batch-size launch flag
    # must have taken effect, or every J/image for this cell would be wrong by ×N.
    actual_bs = status.get("batch_size")
    if actual_bs is not None and int(actual_bs) != job.batch_size:
        print(f"  ABORT: expected batch_size {job.batch_size}, got {actual_bs}")
        return job_err(job, "batch_size_mismatch",
                       expected_batch_size=job.batch_size, actual_batch_size=actual_bs)

    # Jetson clock-lock: warning only (sampling timing can cause false reads)
    gpu_cur = status.get("gpu_cur_freq_hz")
    gpu_max = status.get("gpu_max_freq_hz")
    if gpu_cur and gpu_max:
        if gpu_cur < gpu_max * GPU_CLOCK_LOCK_RATIO:
            print(f"  WARNING: GPU cur_freq ({gpu_cur/1e6:.0f}MHz) < "
                  f"max_freq ({gpu_max/1e6:.0f}MHz). "
                  f"jetson_clocks may not have taken effect yet.")
        else:
            print(f"  GPU clock-locked: {gpu_cur/1e6:.0f}MHz")

    return None  # All checks passed


async def sanity_check_latency(job: MeasurementJob,
                               target_url: str) -> dict | None:
    """Send probe requests to detect NPU/accelerator degradation.

    Expected single-request latency ≈ 1000/mu_hint ms.
    If measured latency exceeds SANITY_LATENCY_RATIO × expected, the device
    is likely degraded (e.g. RKNN NPU core locked).

    Returns error dict if degraded, None if healthy.
    """
    import aiohttp
    # A bs=N request runs an N-wide forward, so per-request latency scales with
    # the batch width while mu_hint is an images/s rate; scale the expected
    # single-request latency by N. Without this, slow (heavy-model × low-clock)
    # bs>1 cells trip the degradation threshold spuriously. bs=1 ⇒ ×1 (unchanged).
    expected_ms = 1000.0 / max(job.mu_hint, 1.0) * job.batch_size
    threshold_ms = expected_ms * SANITY_LATENCY_RATIO
    url = target_url.rstrip("/") + "/infer"

    latencies = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        for _ in range(SANITY_REQUESTS):
            try:
                t0 = time.monotonic()
                async with session.post(url) as resp:
                    lat = (time.monotonic() - t0) * 1000
                    if resp.status == 200:
                        latencies.append(lat)
            except Exception:
                pass

    if not latencies:
        print(f"  SANITY: all {SANITY_REQUESTS} probe requests failed")
        return job_err(job, "sanity_probe_failed")

    median_lat = sorted(latencies)[len(latencies) // 2]
    print(f"  SANITY: probe latency {median_lat:.1f}ms "
          f"(expected ~{expected_ms:.1f}ms, threshold {threshold_ms:.0f}ms)")

    if median_lat > threshold_ms:
        print(f"  ABORT: accelerator degraded. "
              f"{median_lat:.1f}ms >> {expected_ms:.1f}ms expected. "
              f"Device reboot required.")
        return job_err(job, "accelerator_degraded",
                       error_type="device_health",
                       median_latency_ms=round(median_lat, 1),
                       expected_ms=round(expected_ms, 1),
                       threshold_ms=round(threshold_ms, 0))

    return None


def _dvfs_establishing_job(device: str, policy: str):
    """Find the (device, policy) job in JOBS_ALL that establishes DVFS state.

    In JOBS_ALL, jobs are ordered so that one "establishing" job per
    (device, policy) carries needs_reboot / nvpmodel_mode / dvfs_setup
    commands; the remaining jobs for the same (device, policy) inherit
    that state. When λ-sweep targets an inheriting job directly we must
    replay the establishing job's DVFS setup so the device reaches the
    canonical state for that policy.
    """
    from measurement_jobs import JOBS_ALL
    for j in JOBS_ALL:
        if j.device != device or j.policy != policy:
            continue
        if j.needs_reboot or j.dvfs_setup:
            return j
    return None


async def apply_dvfs_setup(job: MeasurementJob) -> None:
    """Phase 0 of any measurement job: optional reboot-for-nvpmodel + DVFS cmds.

    If `job` itself does not carry DVFS commands (i.e. it inherits DVFS from
    an earlier job in JOBS_ALL), fall back to the (device, policy)
    establishing job so λ-sweep on an inheriting pair still reaches the
    canonical state.
    """
    target = job
    if job.dvfs_mode is None and not (job.needs_reboot or job.dvfs_setup):
        est = _dvfs_establishing_job(job.device, job.policy)
        if est is not None and est is not job:
            print(f"  DVFS inherit: replaying establishing-job setup "
                  f"from {est.device}/{est.model}/{est.policy}")
            target = est

    if target.nvpmodel_mode >= 0:
        if target.needs_reboot:
            print(f"  Reboot required for nvpmodel mode {target.nvpmodel_mode}...")
            try:
                ssh(job.device,
                    f"printf 'YES\\n' | sudo nvpmodel -m {target.nvpmodel_mode}",
                    timeout=NVPMODEL_SET_TIMEOUT)
            except RemoteError as e:
                # nvpmodel mode change triggers reboot on Jetson Orin.
                # Expected: stderr contains "reboot", or SSH closed (rc=255) as
                # the device reboots mid-command.
                if e.error_type not in ("ssh_connect", "ssh_timeout") \
                   and "reboot" not in e.stderr.lower():
                    raise RemoteError("nvpmodel_set_failed", job.device,
                                      f"nvpmodel -m {target.nvpmodel_mode}",
                                      e.returncode, e.stderr)
            await wait_for_device_operational(job.device)
        else:
            print(f"  DVFS: sudo nvpmodel -m {target.nvpmodel_mode}")
            ssh(job.device,
                f"sudo nvpmodel -m {target.nvpmodel_mode}",
                timeout=NVPMODEL_SET_TIMEOUT)
            print(f"  DVFS: sudo jetson_clocks")
            ssh(job.device, "sudo jetson_clocks", timeout=SSH_CMD_TIMEOUT)

    for cmd in target.dvfs_setup:
        print(f"  DVFS: {cmd}")
        ssh(job.device, cmd)


async def reboot_and_wait_generic(device: str) -> None:
    """Reboot a non-Jetson device and wait for SSH to come back."""
    print(f"  Rebooting {device}...")
    ssh(device, "sudo reboot", check=False)
    await asyncio.sleep(10)
    for attempt in range(REBOOT_WAIT_GENERIC // POLL_INTERVAL):
        await asyncio.sleep(POLL_INTERVAL)
        try:
            ssh(device, "echo up", timeout=POLL_INTERVAL)
            print(f"  {device} back up after {(attempt+1)*POLL_INTERVAL + 10}s")
            await asyncio.sleep(5)
            return
        except Exception:
            pass
    raise RemoteError("reboot_timeout", device,
                      "generic reboot", -1,
                      f"not reachable after {REBOOT_WAIT_GENERIC}s")
