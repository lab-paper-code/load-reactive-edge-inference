"""
EEP tensor full-sweep profiler.

Data management: upsert mode. (device, model, batch_size, dvfs_mode) is the primary key.
- Existing key: replace; new key: insert.
- A specific device can be re-measured while preserving all other data.
- --fresh: delete existing data for the target device before re-measuring.

Usage:
    python3 src/run_sweep.py --devices orin           # re-measure orin only (others preserved)
    python3 src/run_sweep.py --all                    # all Shelly-connected devices
    python3 src/run_sweep.py --devices orin --dvfs 0  # orin DVFS mode 0 only
    python3 src/run_sweep.py --devices orin --all-dvfs  # iterate all configured DVFS modes for orin
    python3 src/run_sweep.py --plan configs/sweeps/canonical_parallel.yaml --dry-run
    python3 src/run_sweep.py --all --fresh            # ignore existing data, start fresh
"""

from __future__ import annotations

import atexit
import argparse
from datetime import datetime
import json
import os
import random
import signal
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import yaml

from device_lock import acquire_device_lock, release_device_lock, DeviceBusy
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (get_all_devices, shelly_devices, profiling_defaults,
                    batch_sizes as cfg_batch_sizes, DB_PATH,
                    MEASURED_RESULTS_DIR, RAW_RESULTS_DIR, ensure_results_layout)
from config import physical_host_id_from_cfg
from db import (get_connection, init_db, upsert_profile, export_tensor_json,
                create_run, finish_run, insert_trial,
                reconcile_stale_running_runs, abort_run,
                DEFAULT_ORPHAN_THRESHOLD_HOURS)
from dvfs import apply_dvfs_mode, selected_dvfs_modes
from measurement_signals import (refresh_profile_annotations,
                                 write_coverage_report,
                                 write_export_integrity_report)
from profiler import (profile, ensure_bench_script, ensure_model,
                      BenchExecutionError)
from power_reader import PrometheusReader
from runtime_budget import compute_runtime_budget, format_runtime_budget


def _load_plan(path: str) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Invalid sweep plan: {path}')
    return data


def _device_entry(base: dict, override: Optional[dict] = None) -> dict:
    dev = dict(base)
    if override:
        for key in ('compatible_models', 'batch_sizes'):
            if key in override:
                dev[key] = list(override[key])
    return dev


def _resolve_plan(plan_path: Optional[str],
                  devices_arg: Optional[list[str]],
                  all_devices_flag: bool,
                  dvfs_override: Optional[int],
                  all_dvfs: bool,
                  duration: Optional[float],
                  trials: Optional[int],
                  fresh: bool,
                  output: str) -> tuple[dict, dict, str, bool]:
    all_devs = get_all_devices()
    base_cfg = profiling_defaults()
    batch_default = cfg_batch_sizes()
    resolved_output = output
    resolved_fresh = fresh
    device_specs = {}

    if plan_path:
        plan = _load_plan(plan_path)
        plan_cfg = dict(plan.get('profiling', {}))
        pcfg = {**base_cfg, **plan_cfg}
        if duration is not None:
            pcfg['duration_s'] = duration
        if trials is not None:
            pcfg['trials'] = trials
        if 'output' in plan and output == DB_PATH:
            resolved_output = plan['output']
        if 'fresh' in plan and not fresh:
            resolved_fresh = bool(plan['fresh'])

        if 'devices' not in plan or not isinstance(plan['devices'], dict):
            raise ValueError('Plan must define devices: {device_name: {...}}')

        for dev_name, override in plan['devices'].items():
            if dev_name not in all_devs:
                raise ValueError(f'Unknown device in plan: {dev_name}')
            override = override or {}
            dev = _device_entry(all_devs[dev_name], override)
            dev['batch_sizes'] = list(override.get('batch_sizes', dev.get('batch_sizes', batch_default)))
            if dvfs_override is not None:
                dev['selected_dvfs_modes'] = [int(dvfs_override)]
            elif 'dvfs_modes' in override:
                dev['selected_dvfs_modes'] = [int(x) for x in override['dvfs_modes']]
            else:
                dev['selected_dvfs_modes'] = selected_dvfs_modes(
                    dev, dvfs_override=None, all_dvfs=all_dvfs
                )
            device_specs[dev_name] = dev
        return device_specs, pcfg, resolved_output, resolved_fresh

    pcfg = dict(base_cfg)
    if duration is not None:
        pcfg['duration_s'] = duration
    if trials is not None:
        pcfg['trials'] = trials

    if all_devices_flag:
        chosen = shelly_devices()
    elif devices_arg:
        chosen = {d: all_devs[d] for d in devices_arg if d in all_devs}
    else:
        chosen = shelly_devices()

    for dev_name, dev in chosen.items():
        dev = dict(dev)
        dev['batch_sizes'] = list(dev.get('batch_sizes', batch_default))
        dev['selected_dvfs_modes'] = selected_dvfs_modes(
            dev, dvfs_override=dvfs_override, all_dvfs=all_dvfs
        )
        device_specs[dev_name] = dev

    return device_specs, pcfg, resolved_output, resolved_fresh


def _print_plan(device_specs: dict, pcfg: dict, db_path: str, fresh: bool):
    total = 0
    print('Sweep config:')
    print(f'  Devices: {list(device_specs.keys())} (parallel)')
    print(f'  Duration: {pcfg["duration_s"]}s per trial')
    print(f'  Warmup: {pcfg.get("warmup", 0)}')
    print(f'  Trials: {pcfg.get("trials", 1)} (single SSH session)')
    print(f'  Cooldown: {pcfg.get("cooldown_s", 10)}s (inter-trial)')
    print(f'  Mode: {"fresh" if fresh else "upsert"}')
    print(f'  Output: {db_path}')
    print('  Device matrix:')
    for dev_name, dev in device_specs.items():
        models = dev['compatible_models']
        batches = dev['batch_sizes']
        modes = dev['selected_dvfs_modes']
        n_profiles = len(models) * len(batches) * len(modes)
        total += n_profiles
        print(f'    - {dev_name}: models={models} batches={batches} dvfs={modes} -> {n_profiles}')
    print(f'  Total profiles: {total}')


def _assert_no_physical_device_overlap(devices: dict):
    grouped = {}
    for dev_name, dev in devices.items():
        key = physical_host_id_from_cfg(dev)
        grouped.setdefault(key, []).append(dev_name)
    overlaps = [names for names in grouped.values() if len(names) > 1]
    if overlaps:
        joined = ', '.join('/'.join(names) for names in overlaps)
        raise ValueError(
            f'Parallel sweep mixes logical devices on the same physical board: {joined}. '
            'Split them into separate runs to avoid latency/power contamination.'
        )


def _generate_run_id() -> str:
    """timestamp + pid + random 4-char suffix to prevent sub-second collisions."""
    ts = time.strftime('%Y%m%d_%H%M%S')
    pid = os.getpid()
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f'run_{ts}_{pid}_{rand}'


def _get_git_commit() -> str:
    """Return current git HEAD commit hash."""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else ''
    except Exception:
        return ''


def _make_atexit_run_finalizer(db_path: str, run_id: str):
    def _finalize():
        conn = None
        try:
            conn = get_connection(db_path)
            row = conn.execute(
                'SELECT status FROM runs WHERE run_id=?',
                (run_id,),
            ).fetchone()
            if not row or row['status'] != 'running':
                return
            abort_run(conn, run_id, 'atexit auto-finalize')
        except Exception:
            pass
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    return _finalize


def _raise_system_exit_130(signum, frame):
    raise SystemExit(130)


def _orphan_threshold_hours() -> float:
    raw = os.environ.get('ORPHAN_THRESHOLD_HOURS')
    if raw is None:
        return DEFAULT_ORPHAN_THRESHOLD_HOURS
    try:
        return float(raw)
    except ValueError:
        print(f'Invalid ORPHAN_THRESHOLD_HOURS={raw!r}; '
              f'using {DEFAULT_ORPHAN_THRESHOLD_HOURS}')
        return DEFAULT_ORPHAN_THRESHOLD_HOURS


def _reconcile_startup_runs(db_path: str):
    conn_main = get_connection(db_path)
    try:
        init_db(conn_main)
        reconciled = reconcile_stale_running_runs(
            conn_main,
            threshold_hours=_orphan_threshold_hours(),
        )
    finally:
        conn_main.close()
    if reconciled:
        print(f'Startup reconciliation: aborted stale runs {reconciled}')


def _persist_power_samples(raw_dir: str, device: str, model: str,
                           batch_size: int, dvfs_mode: int,
                           trial: dict) -> dict:
    payload = trial.pop('_power_samples_npz', None)
    manifest = json.loads(trial.get('power_trace_json', '{}') or '{}')
    if not payload:
        trial['power_trace_json'] = json.dumps(manifest)
        return trial

    power_dir = os.path.join(raw_dir, 'power')
    os.makedirs(power_dir, exist_ok=True)
    trial_idx = int(trial.get('trial_index', 0))
    base = f'{device}_{model}_bs{batch_size}_dvfs{dvfs_mode}_trial{trial_idx}_power.npz'
    npz_path = os.path.join(power_dir, base)
    np.savez_compressed(
        npz_path,
        idle_ts=np.asarray(payload.get('idle_ts', []), dtype=float),
        idle_watts=np.asarray(payload.get('idle_watts', []), dtype=float),
        active_ts=np.asarray(payload.get('active_ts', []), dtype=float),
        active_watts=np.asarray(payload.get('active_watts', []), dtype=float),
    )

    manifest.update({
        'storage': 'npz',
        'npz_path': npz_path,
    })
    trial['power_trace_json'] = json.dumps(manifest)
    return trial


def run_sweep(devices: dict, db_path: str, pcfg: dict, fresh: bool = False):
    prom = PrometheusReader()
    _assert_no_physical_device_overlap(devices)

    ensure_results_layout()
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn_main = get_connection(db_path)
    init_db(conn_main)

    # Create run
    import json as _json
    run_id = _generate_run_id()
    create_run(
        conn_main, run_id,
        operator="",
        git_commit=_get_git_commit(),
        config_snapshot=_json.dumps(pcfg),
        devices=_json.dumps(list(devices.keys())),
    )
    atexit.register(_make_atexit_run_finalizer(db_path, run_id))
    signal.signal(signal.SIGTERM, _raise_system_exit_130)
    signal.signal(signal.SIGINT, _raise_system_exit_130)
    print(f'Run: {run_id}')

    # Output directory for raw trial JSON
    raw_dir = os.path.join(RAW_RESULTS_DIR, run_id)
    os.makedirs(raw_dir, exist_ok=True)

    if fresh:
        device_list = list(devices.keys())
        placeholders = ', '.join('?' * len(device_list))
        fresh_ts = time.strftime('%Y-%m-%dT%H:%M:%S')

        # Query affected run_ids before deleting trials and record them in runs notes.
        # Preserve the runs rows themselves (audit log) but mark that their data was cleared.
        affected = conn_main.execute(
            f'SELECT DISTINCT run_id FROM trials WHERE device IN ({placeholders})',
            device_list
        ).fetchall()
        if affected:
            note = f'[fresh-cleared {fresh_ts} devices={device_list}]'
            run_phs = ', '.join('?' * len(affected))
            run_ids = [r[0] for r in affected]
            conn_main.execute(
                f"UPDATE runs SET notes = CASE WHEN notes = '' THEN ? "
                f"ELSE notes || ' ' || ? END "
                f"WHERE run_id IN ({run_phs})",
                [note, note] + run_ids
            )

        conn_main.execute(
            f'DELETE FROM profiles WHERE device IN ({placeholders})', device_list
        )
        conn_main.execute(
            f'DELETE FROM trials WHERE device IN ({placeholders})', device_list
        )
        conn_main.commit()

        # raw JSON files: results/raw/{old_run_id}/{device}_*.json
        raw_base = RAW_RESULTS_DIR
        n_json_deleted = 0
        if os.path.isdir(raw_base):
            for old_run in os.listdir(raw_base):
                run_dir = os.path.join(raw_base, old_run)
                if not os.path.isdir(run_dir):
                    continue
                for fname in os.listdir(run_dir):
                    for dev in device_list:
                        if fname.startswith(f'{dev}_') and fname.endswith('.json'):
                            os.remove(os.path.join(run_dir, fname))
                            n_json_deleted += 1
        print(f'Fresh mode: cleared profiles/trials for {device_list}, '
              f'{n_json_deleted} raw JSON(s) removed. '
              f'Affected runs annotated in notes (rows retained).')

    n_before = conn_main.execute('SELECT COUNT(*) FROM profiles').fetchone()[0]

    if not fresh and n_before > 0:
        print(f'Loaded {n_before} existing profiles from {db_path}')

    new_count = [0]
    trial_count = [0]
    error_count = [0]
    count_lock = threading.Lock()

    total = sum(
        len(d['compatible_models']) * len(d['batch_sizes'])
        * len(d['selected_dvfs_modes'])
        for d in devices.values()
    )
    expected_trials = total * pcfg.get('trials', 1)
    done = [0]
    done_lock = threading.Lock()

    def sweep_device(dev_name, dev):
        lock_fh = acquire_device_lock(dev_name)
        try:
            _sweep_device_locked(dev_name, dev)
        finally:
            release_device_lock(lock_fh)

    def _sweep_device_locked(dev_name, dev):
        ssh = dev['ssh']
        shelly = dev.get('shelly_hostname')
        models = dev['compatible_models']
        batch_sizes = dev['batch_sizes']
        dvfs_modes = dev['selected_dvfs_modes']
        n_profiles = len(models) * len(batch_sizes) * len(dvfs_modes)

        print(f'[{dev_name}] Start: {len(models)} models × {len(batch_sizes)} batches '
              f'× {len(dvfs_modes)} dvfs = {n_profiles} profiles (modes={dvfs_modes})')

        conn = get_connection(db_path)

        ensure_bench_script(ssh, dev)

        for dvfs_mode in dvfs_modes:
            print(f'  [{dev_name}] Apply DVFS mode {dvfs_mode}')
            apply_dvfs_mode(ssh, dev_name, dev, dvfs_mode)
            # Some Jetson nvpmodel transitions reboot the board and clear /tmp.
            # Re-deploy the benchmark script after each DVFS transition so
            # rebooting and non-rebooting devices follow the same path.
            ensure_bench_script(ssh, dev)

            for model_name in models:
                for bs in batch_sizes:
                    with done_lock:
                        done[0] += 1
                        tag = f'[{done[0]}/{total}]'

                    try:
                        ensure_model(ssh, model_name, bs, dev)
                        # Latency hint for dynamic timeout: linear scale from the smallest-bs result of the same device+model.
                        row = conn.execute(
                            'SELECT latency_ms_p95, batch_size FROM profiles '
                            'WHERE device=? AND model=? AND latency_ms_p95 IS NOT NULL '
                            'ORDER BY batch_size ASC LIMIT 1',
                            (dev_name, model_name)
                        ).fetchone()
                        latency_hint_ms = None
                        if row:
                            ref_lat, ref_bs = row
                            latency_hint_ms = ref_lat * bs / ref_bs
                        budget = compute_runtime_budget(
                            batch_size=bs,
                            duration_s=pcfg.get('duration_s', 30.0),
                            warmup=pcfg.get('warmup', 50),
                            trials=pcfg.get('trials', 1),
                            cooldown_s=pcfg.get('cooldown_s', 10),
                            latency_hint_ms=latency_hint_ms,
                            cfg=pcfg,
                        )
                        print(f'    [{dev_name}] {model_name}/bs={bs} budget: '
                              f'{format_runtime_budget(budget)}')

                        result, raw_trials = profile(
                            ssh, shelly, model_name, bs, prom, pcfg,
                            ep=dev.get('ep'),
                            hostname=dev.get('hostname'),
                            has_gpu=dev.get('gpu', False),
                            latency_hint_ms=latency_hint_ms,
                            device_cfg=dev)
                        result['device'] = dev_name
                        result['dvfs_mode'] = dvfs_mode
                        result['run_id'] = run_id
                        result['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

                        # TODO(atomic-write): upsert_profile + insert_trial(s) + JSON write
                        # should be one transaction. Currently each commits individually;
                        # a crash between writes leaves partial state.
                        upsert_profile(conn, result)

                        # Save raw trial (DB + JSON)
                        ts = result['timestamp']
                        persisted_trials = []
                        for trial in raw_trials:
                            trial['run_id'] = run_id
                            trial['device'] = dev_name
                            trial['dvfs_mode'] = dvfs_mode
                            trial['timestamp'] = ts
                            trial = _persist_power_samples(
                                raw_dir, dev_name, model_name, bs, dvfs_mode, trial)
                            persisted_trials.append(trial)
                            insert_trial(conn, trial)

                        # Raw trial JSON file
                        import json as _json
                        trial_file = os.path.join(
                            raw_dir,
                            f'{dev_name}_{model_name}_bs{bs}_dvfs{dvfs_mode}.json')
                        with open(trial_file, 'w') as tf:
                            _json.dump({
                                'run_id': run_id,
                                'device': dev_name,
                                'model': model_name,
                                'batch_size': bs,
                                'dvfs_mode': dvfs_mode,
                                'n_trials': len(persisted_trials),
                                'trials': persisted_trials,
                                'aggregate': result,
                            }, tf, indent=2, default=str)

                        refresh_profile_annotations(conn)

                        with count_lock:
                            new_count[0] += 1
                            trial_count[0] += len(persisted_trials)

                        e = result.get('energy_per_inf_j')
                        l = result.get('latency_ms_p95')
                        t = result.get('throughput_ips')
                        pw = result.get('p_wall_avg_w')
                        nt = result.get('n_trials', 1)
                        print(f'  {tag} [{dev_name}] dvfs={dvfs_mode} {model_name}/bs={bs} '
                              f'OK  E={e}J L_p95={l}ms T={t}ips P={pw}W trials={nt}')

                    except Exception as ex:
                        with count_lock:
                            error_count[0] += 1
                        failure_class = 'unknown'
                        last_event = None
                        budget_dict = None
                        stderr = None
                        if isinstance(ex, BenchExecutionError):
                            failure_class = ex.failure_class
                            last_event = ex.last_event
                            budget_dict = ex.budget
                            stderr = ex.stderr
                        import json as _json
                        fail_dir = os.path.join(raw_dir, 'failures')
                        os.makedirs(fail_dir, exist_ok=True)
                        fail_file = os.path.join(
                            fail_dir,
                            f'{dev_name}_{model_name}_bs{bs}_dvfs{dvfs_mode}.json'
                        )
                        with open(fail_file, 'w') as ff:
                            _json.dump({
                                'run_id': run_id,
                                'device': dev_name,
                                'model': model_name,
                                'batch_size': bs,
                                'dvfs_mode': dvfs_mode,
                                'failure_class': failure_class,
                                'error': str(ex),
                                'stderr': stderr,
                                'last_event': last_event,
                                'runtime_budget': budget_dict,
                                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                            }, ff, indent=2, default=str)
                        print(f'  {tag} [{dev_name}] dvfs={dvfs_mode} {model_name}/bs={bs} '
                              f'FAIL[{failure_class}]: {ex}')

        conn.close()

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {executor.submit(sweep_device, name, dev): name
                   for name, dev in devices.items()}
        for future in as_completed(futures):
            dev_name = futures[future]
            try:
                future.result()
                print(f'[{dev_name}] Complete')
            except Exception as ex:
                # Device-level failure (ensure_bench_script, ensure_model, etc.)
                dev_cfg = devices[dev_name]
                n_dev_profiles = (
                    len(dev_cfg['compatible_models']) * len(dev_cfg['batch_sizes'])
                    * len(dev_cfg['selected_dvfs_modes'])
                )
                with count_lock:
                    error_count[0] += n_dev_profiles
                print(f'[{dev_name}] FAILED: {ex}')

    # Run complete; determine status.
    if error_count[0] == 0:
        run_status = 'completed'
    elif new_count[0] == 0:
        run_status = 'failed'
    else:
        run_status = 'partial'
    finish_run(conn_main, run_id, status=run_status,
               n_trials_total=expected_trials, n_trials_ok=trial_count[0])

    # Final statistics
    n_after = conn_main.execute('SELECT COUNT(*) FROM profiles').fetchone()[0]
    n_trials_db = conn_main.execute(
        'SELECT COUNT(*) FROM trials WHERE run_id=?', (run_id,)
    ).fetchone()[0]
    conn_main.close()
    n_added = n_after - n_before
    print(f'\nSweep complete: {new_count[0]} measured '
          f'(+{n_added} new rows) -> {n_after} total in {db_path}')
    print(f'Run {run_id}: {n_trials_db} trials saved, '
          f'raw JSON in {raw_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', help='YAML sweep plan path')
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--all', action='store_true', help='Shelly-connected devices only')
    parser.add_argument('--dvfs', type=int, default=None, help='DVFS mode override')
    parser.add_argument('--all-dvfs', action='store_true',
                        help='iterate all DVFS modes defined in device config')
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--trials', type=int, default=None)
    parser.add_argument('--fresh', action='store_true', help='delete existing data for the device then re-measure')
    parser.add_argument('--output', default=DB_PATH)
    parser.add_argument('--dry-run', action='store_true', help='print sweep space without running')
    args = parser.parse_args()

    devices, pcfg, output, fresh = _resolve_plan(
        plan_path=args.plan,
        devices_arg=args.devices,
        all_devices_flag=args.all,
        dvfs_override=args.dvfs,
        all_dvfs=args.all_dvfs,
        duration=args.duration,
        trials=args.trials,
        fresh=args.fresh,
        output=args.output,
    )

    _print_plan(devices, pcfg, output, fresh)
    if args.dry_run:
        return

    _reconcile_startup_runs(output)
    run_sweep(devices, output, pcfg, fresh=fresh)

    # Only canonical DB sweeps update the canonical measured/manifests artifacts.
    # Alternate DBs remain their own truth set; canonical artifacts are not modified.
    if os.path.realpath(output) == os.path.realpath(DB_PATH):
        tensor_path = os.path.join(MEASURED_RESULTS_DIR, 'eep_tensor.json')
        n = export_tensor_json(output, tensor_path)
        write_coverage_report(output)
        write_export_integrity_report(output, export_path=tensor_path)
        print(f'Updated canonical measured artifacts ({n} entries)')
    else:
        print('Skipped canonical artifact refresh: non-canonical DB output')


if __name__ == '__main__':
    main()
