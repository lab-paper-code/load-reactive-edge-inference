"""
EEP tensor profiler.
# EEP = edge energy profiler.

Measures a single (device, model, config, batch) combination and records (E, Q, L, T).
Bench script: scripts/bench.py. Configuration loaded from configs/.
"""

import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Optional

from config import (
    MODEL_DIR,
    remote_dir,
    profiling_defaults,
)
from rknn_artifacts import sha256_file
from rknn_runtime import (
    bench_command as rknn_bench_command,
    ensure_local_artifact as ensure_local_rknn_artifact,
    remote_model_artifact as remote_rknn_artifact,
)
from ssh import run, deploy_script, scp
from power_reader import PrometheusReader
from power_trace import DirectShellyTraceCollector, analyze_power_trace
from runtime_budget import compute_runtime_budget, classify_bench_failure

_BENCH_SCRIPT = {}


@dataclass
class BenchRunResult:
    trials: list
    events: list
    budget: dict


class BenchExecutionError(RuntimeError):
    def __init__(self, ssh_alias: str, returncode: int, stderr: str,
                 failure_class: str, last_event: Optional[dict],
                 events: list, budget: dict):
        self.ssh_alias = ssh_alias
        self.returncode = returncode
        self.stderr = stderr
        self.failure_class = failure_class
        self.last_event = last_event
        self.events = events
        self.budget = budget
        last_event_name = last_event.get('event') if last_event else None
        msg = (f'Bench failed on {ssh_alias}: rc={returncode} class={failure_class} '
               f'last_event={last_event_name} err={stderr[:300]}')
        super().__init__(msg)


def _load_bench_script() -> str:
    return _load_named_script('bench.py')


def _load_named_script(filename: str) -> str:
    global _BENCH_SCRIPT
    if filename not in _BENCH_SCRIPT:
        script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', filename)
        with open(script_path) as f:
            _BENCH_SCRIPT[filename] = f.read()
    return _BENCH_SCRIPT[filename]


def _runtime_class(device_cfg: Optional[dict]) -> str:
    if not device_cfg:
        return 'ort'
    return device_cfg.get('runtime_class', 'ort')


def _local_model_artifact(model_name: str, batch_size: int,
                          device_cfg: Optional[dict]) -> str:
    runtime_class = _runtime_class(device_cfg)
    if runtime_class == 'rknn':
        return ensure_local_rknn_artifact(model_name, batch_size, device_cfg)
    return os.path.join(MODEL_DIR, f'{model_name}.onnx')


def _remote_model_artifact(model_name: str, batch_size: int,
                           device_cfg: Optional[dict]) -> str:
    if _runtime_class(device_cfg) == 'rknn':
        return remote_rknn_artifact(model_name, batch_size, device_cfg)
    local = _local_model_artifact(model_name, batch_size, device_cfg)
    return f'{remote_dir()}/{os.path.basename(local)}'


def _remote_sha256(ssh_alias: str, path: str) -> Optional[str]:
    rc, out, _ = run(ssh_alias, f"sha256sum {path} 2>/dev/null | awk '{{print $1}}'")
    if rc != 0:
        return None
    value = out.strip()
    return value or None


def ensure_bench_script(ssh_alias: str, device_cfg: Optional[dict] = None) -> str:
    """Deploy bench script to the remote device. Returns the remote path."""
    runtime_class = _runtime_class(device_cfg)
    if runtime_class == 'rknn':
        return deploy_script(ssh_alias, _load_named_script('bench_rknn.py'), '_bench_rknn.py')
    return deploy_script(ssh_alias, _load_bench_script(), '_bench.py')


def ensure_model(ssh_alias: str, model_name: str,
                 batch_size: int = 1,
                 device_cfg: Optional[dict] = None):
    """Deploy runtime-specific model artifact to the remote device (skip if already present)."""
    local = _local_model_artifact(model_name, batch_size, device_cfg)
    if not os.path.exists(local):
        raise FileNotFoundError(f'{local} not found')
    remote = _remote_model_artifact(model_name, batch_size, device_cfg)
    rc, out, _ = run(ssh_alias, f'test -f {remote} && stat -c %s {remote}')
    if rc == 0:
        remote_size = int(out.strip())
        local_size = os.path.getsize(local)
        if remote_size == local_size:
            local_sha = sha256_file(local)
            remote_sha = _remote_sha256(ssh_alias, remote)
            if remote_sha and remote_sha == local_sha:
                return
    scp(ssh_alias, local)


def run_inference_bench(ssh_alias: str, model_name: str,
                        batch_size: int, duration_s: float,
                        warmup: int, ep=None,
                        trials: int = 1, cooldown_s: float = 0,
                        latency_hint_ms: float = None,
                        cfg: Optional[dict] = None,
                        device_cfg: Optional[dict] = None) -> BenchRunResult:
    """Run runtime-specific bench on the remote device; return trial/event results and budget."""
    runtime_class = _runtime_class(device_cfg)
    model_path = _remote_model_artifact(model_name, batch_size, device_cfg)
    idle_window_s = float((cfg or {}).get('power_idle_window_s', 10.0))
    if runtime_class == 'rknn':
        cmd = rknn_bench_command(
            model_name=model_name,
            batch_size=batch_size,
            duration_s=duration_s,
            warmup=warmup,
            trials=trials,
            cooldown_s=cooldown_s,
            idle_window_s=idle_window_s,
            device_cfg=device_cfg,
        )
    else:
        script_path = f'{remote_dir()}/_bench.py'
        ep_str = ','.join(ep) if isinstance(ep, list) else (ep or 'auto')
        cmd = (f'python3 {script_path} {model_path} {batch_size} '
               f'{duration_s} {warmup} {ep_str} {trials} {cooldown_s} {idle_window_s}')

    budget = compute_runtime_budget(
        batch_size=batch_size,
        duration_s=duration_s,
        warmup=warmup,
        trials=trials,
        cooldown_s=cooldown_s,
        latency_hint_ms=latency_hint_ms,
        cfg=cfg or {},
    )
    timeout = budget.total_s
    rc, out, err = run(ssh_alias, cmd, timeout=timeout)
    events = []
    results = []
    for line in out.strip().split('\n'):
        line = line.strip()
        if line:
            if not (line.startswith('{') and line.endswith('}')):
                continue
            parsed = json.loads(line)
            if 'event' in parsed:
                parsed['_raw_ndjson'] = line
                events.append(parsed)
            else:
                parsed['_raw_ndjson'] = line  # preserve original stdout line from bench.py
                results.append(parsed)
    if rc != 0:
        last_event = events[-1] if events else None
        failure_class = classify_bench_failure(
            err, last_event.get('event') if last_event else None, returncode=rc
        )
        raise BenchExecutionError(
            ssh_alias=ssh_alias,
            returncode=rc,
            stderr=err,
            failure_class=failure_class,
            last_event=last_event,
            events=events,
            budget=budget.as_dict(),
        )
    return BenchRunResult(trials=results, events=events, budget=budget.as_dict())


def _get_gpu_temp_ssh(ssh_alias: str) -> float:
    """Query GPU temperature directly via SSH. Tries nvidia-smi first, then Jetson sysfs."""
    rc, out, _ = run(ssh_alias,
                     'nvidia-smi --query-gpu=temperature.gpu '
                     '--format=csv,noheader 2>/dev/null | head -1',
                     timeout=3)
    if rc == 0 and out.strip():
        try:
            return float(out.strip())
        except ValueError:
            pass
    rc, out, _ = run(ssh_alias,
                     'for z in /sys/devices/virtual/thermal/thermal_zone*; do '
                     'if [ "$(cat $z/type)" = "GPU-therm" ]; then '
                     'v=$(cat $z/temp); echo $v; fi; done',
                     timeout=3)
    if rc == 0 and out.strip():
        try:
            val = float(out.strip()) / 1000.0
            return val if val > 0 else -1.0
        except ValueError:
            pass
    return -1.0


def _read_temps(prom: PrometheusReader, hostname: str,
                ssh_alias: str, has_gpu: bool) -> tuple:
    """Return (cpu_temp, gpu_temp)."""
    if not hostname:
        return -1.0, -1.0
    cpu = prom.get_cpu_temperature(hostname)
    gpu = prom.get_gpu_temperature(hostname)
    if gpu <= 0 and has_gpu:
        gpu = _get_gpu_temp_ssh(ssh_alias)
    return cpu, gpu


# Keys aggregated across multiple trials
_AGG_KEYS = [
    'p_wall_avg_w', 'energy_total_j', 'energy_per_inf_j', 'energy_inc_j',
    'energy_inc_per_inf_j', 'watts_idle', 'watts_idle_true', 'watts_active',
    'watts_idle_median', 'watts_idle_std', 'idle_baseline_margin_w',
    'watts_idle_true_median', 'watts_idle_true_std',
    'p_plateau_w', 'plateau_duration_s', 'plateau_sample_count',
    'energy_total_plateau_j', 'energy_inc_plateau_j',
    'energy_inc_canonical_j', 'energy_inc_canonical_per_inf_j',
    'latency_ms_mean', 'latency_ms_p50', 'latency_ms_p95',
    'latency_ms_p99', 'latency_ms_std', 'throughput_ips',
    'bench_elapsed_s', 'wall_elapsed_s', 'n_power_samples',
    'temp_cpu_before_c', 'temp_cpu_after_c',
    'temp_gpu_before_c', 'temp_gpu_after_c',
]

# Keys for which standard deviation is recorded
_STD_KEYS = {
    'energy_per_inf_j': 'energy_per_inf_j_std',
    'latency_ms_p95': 'latency_ms_p95_std',
    'throughput_ips': 'throughput_ips_std',
    'p_wall_avg_w': 'p_wall_avg_w_std',
}

_AGG_DROP_KEYS = {
    '_power_samples_npz',
    'power_trace_json',
    'power_windowing_json',
    'bench_events_json',
    'runtime_budget_json',
    'idle_start_ts',
    'idle_end_ts',
    'active_start_ts',
    'active_end_ts',
}


def profile(ssh_alias: str, shelly_host,
            model_name: str, batch_size: int,
            prom: PrometheusReader, cfg: dict = None,
            ep=None, hostname: str = None,
            has_gpu: bool = False,
            latency_hint_ms: float = None,
            device_cfg: Optional[dict] = None) -> tuple:
    """
    Profile a (device, model, batch) combination. Runs multi-trial in a single SSH session.

    bench.py builds the ORT session once, repeats N times, and outputs NDJSON.
    When trials > 1, returns the median; standard deviation is recorded with _std suffix.

    Returns:
        (aggregate_dict, raw_trials). aggregate has the same dict shape as before;
        raw_trials is list[dict] of per-trial measurements, each containing bench_json.
    """
    if cfg is None:
        cfg = profiling_defaults()
    trials = cfg.get('trials', 1)
    cooldown_s = cfg.get('cooldown_s', 10)
    duration_s = cfg.get('duration_s', 30.0)
    warmup_n = cfg.get('warmup', 50)
    post_roll_s = float(cfg.get('power_postroll_s', 3.0))
    direct_poll_interval_s = float(cfg.get('direct_poll_interval_s', 0.5))

    has_shelly = shelly_host is not None

    trace_collector = None
    power_trace = []

    # 1. Start direct trace + record initial temperature
    if has_shelly:
        trace_collector = DirectShellyTraceCollector(
            prom, shelly_host, poll_interval_s=direct_poll_interval_s)
        trace_collector.start()
    cpu_before, gpu_before = _read_temps(prom, hostname, ssh_alias, has_gpu)

    # 2. Run all trials in a single SSH session
    bench_exec = run_inference_bench(
        ssh_alias, model_name, batch_size, duration_s, warmup_n,
        ep=ep, trials=trials, cooldown_s=cooldown_s,
        latency_hint_ms=latency_hint_ms, cfg=cfg, device_cfg=device_cfg)
    bench_results = bench_exec.trials
    bench_events = bench_exec.events
    runtime_budget = bench_exec.budget

    # 3. Record final temperature (immediately after inference, before GPU power gating)
    cpu_after, gpu_after = _read_temps(prom, hostname, ssh_alias, has_gpu)
    watts_active = prom.get_shelly_watts(shelly_host) if has_shelly else None

    # 4. Short post-roll to flush last samples, then stop trace
    if has_shelly:
        time.sleep(post_roll_s)
        power_trace = trace_collector.stop()

    # 5. Compute energy per trial.
    # Internal semantics use idle / pre_active / avg / stable / extra windows;
    # mapping to legacy column names happens only at DB write time.
    results = []
    for bench in bench_results:
        n_infer = bench['n_infer']
        total_inferences = n_infer * batch_size

        if has_shelly:
            trace_analysis = analyze_power_trace(
                power_trace, bench, cfg,
            )
            trace_analysis['trace_summary']['collector_errors'] = trace_collector.errors if trace_collector else None

            avg_window = trace_analysis['avg_window']
            stable_window = trace_analysis['stable_window']
            pre_active_stats = trace_analysis['pre_active_window']
            idle_stats = trace_analysis['idle_window']

            avg_power = avg_window['avg_watts']
            n_power_samples = avg_window['sample_count']
            if avg_power is None:
                avg_power, n_power_samples = prom.get_avg_watts_range(
                    shelly_host, bench['active_start_ts'], bench['active_end_ts'])
                avg_window['avg_watts'] = avg_power
                avg_window['sample_count'] = n_power_samples
                avg_window['duration_s'] = bench['active_end_ts'] - bench['active_start_ts']
                avg_window['complete'] = avg_power is not None
            avg_energy_j = (
                avg_power * bench['elapsed_s']
                if avg_power is not None else None
            )
            energy_per_inf_j = (
                avg_energy_j / total_inferences
                if avg_energy_j is not None and total_inferences > 0 else None
            )
            idle_power = idle_stats['median_watts']
            avg_extra_energy_j = (
                (avg_power - idle_power) * bench['elapsed_s']
                if avg_power is not None and idle_power is not None else None
            )
            avg_extra_energy_per_inf_j = (
                avg_extra_energy_j / total_inferences
                if avg_extra_energy_j is not None and total_inferences > 0 else None
            )
            extra_energy_choice = trace_analysis['extra_energy_choice']
            energy_inc_canonical_j = extra_energy_choice.get('energy_inc_j')
            energy_inc_canonical_per_inf_j = (
                energy_inc_canonical_j / total_inferences
                if energy_inc_canonical_j is not None and total_inferences > 0
                else None
            )
        else:
            avg_power = avg_energy_j = energy_per_inf_j = None
            avg_extra_energy_j = avg_extra_energy_per_inf_j = None
            energy_inc_canonical_j = energy_inc_canonical_per_inf_j = None
            n_power_samples = 0
            idle_power = None
            trace_analysis = {
                'idle_window': {'complete': False},
                'pre_active_window': {'complete': False},
                'avg_window': {'complete': False},
                'stable_window': {'complete': False},
                'flags': {
                    'no_stable_window': True,
                    'trace_jitter_high': False,
                    'incremental_unreliable': True,
                    'plateau_incremental_unreliable': False,
                },
                'trace_summary': {'sample_count': 0, 'poll_interval_median_s': None, 'collector_errors': None},
                'extra_energy_choice': {'source': 'none', 'energy_inc_j': None, 'complete': False},
            }
            avg_window = trace_analysis['avg_window']
            stable_window = trace_analysis['stable_window']
            pre_active_stats = trace_analysis['pre_active_window']
            idle_stats = trace_analysis['idle_window']
            extra_energy_choice = trace_analysis['extra_energy_choice']

        power_manifest = {
            'storage': 'npz_pending' if has_shelly else 'none',
            'idle_start_ts': bench.get('idle_start_ts'),
            'idle_end_ts': bench.get('idle_end_ts'),
            'active_start_ts': bench.get('active_start_ts'),
            'active_end_ts': bench.get('active_end_ts'),
            'idle_sample_count': idle_stats.get('sample_count', 0),
            'active_sample_count': avg_window.get('sample_count', 0),
            'trace_sample_count': trace_analysis['trace_summary'].get('sample_count', 0),
            'stable_detection_attempted': bool(avg_window.get('complete')),
        }
        power_windowing = {
            k: v for k, v in trace_analysis.items()
            if k not in ('idle_window', 'avg_window', 'stable_window', 'pre_active_window')
        }
        power_windowing.update({
            'idle_window': {
                key: value for key, value in idle_stats.items() if key != 'samples'
            },
            'pre_active_window': {
                key: value for key, value in pre_active_stats.items() if key != 'samples'
            },
            'avg_window': {
                key: value for key, value in avg_window.items() if key != 'samples'
            },
            'stable_window': {
                key: value for key, value in stable_window.items() if key != 'samples'
            },
        })

        results.append({
            'model': model_name,
            'batch_size': batch_size,
            'trial_index': bench.get('trial', len(results)),
            'n_infer': n_infer,
            'duration_s': duration_s,
            'ep': bench['ep'],
            'idle_start_ts': bench.get('idle_start_ts', 0),
            'idle_end_ts': bench.get('idle_end_ts', 0),
            'elapsed_s': bench['elapsed_s'],
            't_start': bench.get('t_start', 0),
            't_end': bench.get('t_end', 0),
            'active_start_ts': bench.get('active_start_ts', 0),
            'active_end_ts': bench.get('active_end_ts', 0),
            'p_wall_avg_w': round(avg_power, 3) if avg_power is not None else None,
            'energy_total_j': round(avg_energy_j, 2) if avg_energy_j is not None else None,
            'energy_per_inf_j': round(energy_per_inf_j, 6) if energy_per_inf_j is not None else None,
            'energy_inc_j': round(avg_extra_energy_j, 2) if avg_extra_energy_j is not None else None,
            'energy_inc_per_inf_j': (
                round(avg_extra_energy_per_inf_j, 6)
                if avg_extra_energy_per_inf_j is not None else None
            ),
            'watts_idle': idle_power,
            'watts_idle_true': idle_stats.get('median_watts'),
            'watts_active': avg_power if avg_power is not None else watts_active,
            'latency_ms_mean': bench['latency_ms_mean'],
            'latency_ms_p50': bench['latency_ms_p50'],
            'latency_ms_p95': bench['latency_ms_p95'],
            'latency_ms_p99': bench['latency_ms_p99'],
            'latency_ms_std': bench['latency_ms_std'],
            'throughput_ips': bench['throughput_ips'],
            'bench_elapsed_s': bench['elapsed_s'],
            'wall_elapsed_s': bench['elapsed_s'],
            'n_power_samples': n_power_samples,
            'temp_cpu_before_c': cpu_before,
            'temp_cpu_after_c': cpu_after,
            'temp_gpu_before_c': gpu_before,
            'temp_gpu_after_c': gpu_after,
            'bench_json': bench.get('_raw_ndjson', json.dumps(bench)),
            'bench_events_json': json.dumps(bench_events),
            'runtime_budget_json': json.dumps(runtime_budget),
            'watts_idle_mean': idle_stats.get('mean_watts'),
            'watts_idle_median': idle_stats.get('median_watts'),
            'watts_idle_std': idle_stats.get('std_watts'),
            'watts_idle_true_mean': idle_stats.get('mean_watts'),
            'watts_idle_true_median': idle_stats.get('median_watts'),
            'watts_idle_true_std': idle_stats.get('std_watts'),
            'idle_baseline_margin_w': pre_active_stats.get('baseline_margin_w'),
            'p_plateau_w': stable_window.get('avg_watts'),
            'plateau_start_ts': stable_window.get('start_ts'),
            'plateau_end_ts': stable_window.get('end_ts'),
            'plateau_duration_s': stable_window.get('duration_s'),
            'plateau_sample_count': stable_window.get('sample_count'),
            'energy_total_plateau_j': stable_window.get('energy_total_j'),
            'energy_inc_plateau_j': stable_window.get('energy_inc_j'),
            'flag_no_plateau_detected': int(trace_analysis['flags'].get('no_stable_window', False)),
            'flag_trace_jitter_high': int(trace_analysis['flags'].get('trace_jitter_high', False)),
            'flag_incremental_unreliable': int(trace_analysis['flags'].get('incremental_unreliable', False)),
            'flag_plateau_incremental_unreliable': int(
                trace_analysis['flags'].get('plateau_incremental_unreliable', False)),
            'energy_inc_canonical_j': (
                round(energy_inc_canonical_j, 2) if energy_inc_canonical_j is not None else None),
            'energy_inc_canonical_per_inf_j': (
                round(energy_inc_canonical_per_inf_j, 6)
                if energy_inc_canonical_per_inf_j is not None else None),
            'incremental_source': extra_energy_choice.get('source', 'none'),
            'power_trace_json': json.dumps(power_manifest),
            'power_windowing_json': json.dumps(power_windowing),
            '_power_samples_npz': {
                'idle_ts': [float(s['ts']) for s in idle_stats.get('samples', [])],
                'idle_watts': [float(s['watts']) for s in idle_stats.get('samples', [])],
                'active_ts': [float(s['ts']) for s in avg_window.get('samples', [])],
                'active_watts': [float(s['watts']) for s in avg_window.get('samples', [])],
            },
        })

    # 6. Aggregate
    raw_trials = list(results)  # preserve raw trials

    if len(results) == 1:
        out = dict(results[0])
        out['n_trials'] = 1
        for dst in _STD_KEYS.values():
            out[dst] = None
        for key in _AGG_DROP_KEYS:
            out.pop(key, None)
        return out, raw_trials

    out = dict(results[0])
    out['n_trials'] = len(results)

    for key in _AGG_KEYS:
        vals = [r[key] for r in results if r[key] is not None]
        if not vals:
            out[key] = None
            continue
        out[key] = round(statistics.median(vals), 6)

    out['flag_no_plateau_detected'] = max(
        int(r.get('flag_no_plateau_detected', 0)) for r in results)
    out['flag_trace_jitter_high'] = max(
        int(r.get('flag_trace_jitter_high', 0)) for r in results)
    out['flag_incremental_unreliable'] = max(
        int(r.get('flag_incremental_unreliable', 0)) for r in results)
    out['flag_plateau_incremental_unreliable'] = max(
        int(r.get('flag_plateau_incremental_unreliable', 0)) for r in results)

    sources = {r.get('incremental_source', 'none') for r in results}
    out['incremental_source'] = sources.pop() if len(sources) == 1 else 'mixed'

    for src, dst in _STD_KEYS.items():
        vals = [r[src] for r in results if r[src] is not None]
        if len(vals) < 2:
            out[dst] = None
        else:
            out[dst] = round(statistics.stdev(vals), 6)

    # Bimodal latency detection
    lat_vals = [r['latency_ms_p95'] for r in results if r['latency_ms_p95'] is not None]
    if len(lat_vals) >= 2:
        med = statistics.median(lat_vals)
        std = statistics.stdev(lat_vals)
        if med > 0 and std > med * 0.5:
            print(f'    WARNING: bimodal latency, '
                  f'median={med:.2f}ms std={std:.2f}ms')

    for key in _AGG_DROP_KEYS:
        out.pop(key, None)
    return out, raw_trials


if __name__ == '__main__':
    import sys
    from config import get_device

    device = sys.argv[1] if len(sys.argv) > 1 else 'orin'
    model = sys.argv[2] if len(sys.argv) > 2 else 'mobilenet-v2-050'
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    dev = get_device(device)
    prom = PrometheusReader()
    shelly = dev.get('shelly_hostname')

    print(f'Profiling: {device} / {model} / batch={bs}')
    ensure_bench_script(dev['ssh'], dev)
    ensure_model(dev['ssh'], model, bs, dev)
    result, raw_trials = profile(dev['ssh'], shelly, model, bs, prom,
                                  ep=dev.get('ep'), hostname=dev.get('hostname'),
                                  has_gpu=dev.get('gpu', False),
                                  device_cfg=dev)

    for k, v in result.items():
        print(f'  {k:>25s}: {v}')
    print(f'\n  Raw trials: {len(raw_trials)}')
