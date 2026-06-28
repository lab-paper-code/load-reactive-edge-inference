"""
Phase 3 helpers: measurement-quality signals, coverage disclosure, export integrity.

Boundary:
- expose warning signals and disclosure
- do not decide exclusion, admissibility, or fair-comparison policy
"""

from __future__ import annotations

import json
import math
import os
import time

from config import (DB_PATH, MANIFESTS_RESULTS_DIR, MEASURED_RESULTS_DIR,
                    RAW_RESULTS_DIR, batch_sizes, ensure_results_layout,
                    get_all_devices, profiling_defaults)

THERMAL_RISE_THRESHOLD_C = 15.0
BIMODAL_LATENCY_STD_RATIO = 0.5
LOW_POWER_SAMPLE_SLACK = 1
LOW_POWER_SAMPLE_FLOOR = 3


def _bool(val) -> bool:
    return bool(val)


def _rank_quality_record(rec: dict) -> tuple[int, int]:
    task_rank = 0 if rec.get('task_type') == 'classification' else 1
    source_rank = 0 if rec.get('source') == 'measured' else 1
    return (task_rank, source_rank)


def build_quality_index(records: list[dict]) -> tuple[dict[str, dict], set[str]]:
    """Return best-per-model quality record and full model set."""
    best = {}
    model_set = set()
    for rec in records:
        model = rec['model']
        model_set.add(model)
        cur = best.get(model)
        if cur is None or _rank_quality_record(rec) < _rank_quality_record(cur):
            best[model] = rec
    return best, model_set


def expected_power_samples(duration_s: float | None = None) -> int:
    pcfg = profiling_defaults()
    duration = float(duration_s or pcfg.get('duration_s', 30.0))
    scrape_interval = float(pcfg.get('scrape_interval_s', 5.0))
    return max(1, math.floor(duration / scrape_interval) + 1)


def low_power_samples_threshold(duration_s: float | None = None) -> int:
    return max(
        LOW_POWER_SAMPLE_FLOOR,
        expected_power_samples(duration_s) - LOW_POWER_SAMPLE_SLACK,
    )


def _temperature_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if before < 0 or after < 0:
        return None
    return float(after) - float(before)


def coverage_for_profile(row: dict) -> dict:
    """Measurement coverage only: what telemetry is present on this row."""
    cpu_rise = _temperature_delta(row.get('temp_cpu_before_c'),
                                  row.get('temp_cpu_after_c'))
    gpu_rise = _temperature_delta(row.get('temp_gpu_before_c'),
                                  row.get('temp_gpu_after_c'))
    power_measurement = (
        row.get('energy_per_inf_j') is not None
        and row.get('p_wall_avg_w') is not None
        and int(row.get('n_power_samples') or 0) > 0
    )
    temperature_observation = any(v is not None for v in (cpu_rise, gpu_rise))

    missing = []
    if not power_measurement:
        missing.append('power_measurement')
    if not temperature_observation:
        missing.append('temperature_observation')

    return {
        'power_measurement': power_measurement,
        'temperature_observation': temperature_observation,
        'missing': missing,
        'complete': not missing,
    }


def provenance_for_profile(row: dict, trial_count: int | None = None) -> dict:
    """Run/trial provenance disclosure. Not a measurement-coverage flag."""
    n_trials = int(row.get('n_trials') or 0)
    run_id = row.get('run_id') or ''

    run_provenance = bool(run_id)
    trial_truth = (trial_count or 0) > 0
    trial_count_consistent = (
        True if not run_provenance
        else (trial_count == n_trials) if n_trials > 0 else True
    )

    missing = []
    if not run_provenance:
        missing.append('run_provenance')
    if not trial_truth:
        missing.append('trial_truth')
    if not trial_count_consistent:
        missing.append('trial_count_consistency')

    return {
        'run_provenance': run_provenance,
        'trial_truth': trial_truth,
        'trial_count_consistent': trial_count_consistent,
        'missing': missing,
        'complete': not missing,
    }


def integrity_for_profile(row: dict, quality_models: set[str]) -> dict:
    """Metadata integrity disclosure local to profiler exports."""
    quality_metadata = row.get('model') in quality_models

    missing = []
    if not quality_metadata:
        missing.append('quality_metadata')

    return {
        'quality_metadata': quality_metadata,
        'missing': missing,
        'complete': not missing,
    }


def measurement_flags_for_profile(row: dict, coverage: dict) -> dict:
    cpu_rise = _temperature_delta(row.get('temp_cpu_before_c'),
                                  row.get('temp_cpu_after_c'))
    gpu_rise = _temperature_delta(row.get('temp_gpu_before_c'),
                                  row.get('temp_gpu_after_c'))
    temp_rises = [v for v in (cpu_rise, gpu_rise) if v is not None]
    max_temp_rise = max(temp_rises) if temp_rises else None

    latency_p95 = row.get('latency_ms_p95')
    latency_p95_std = row.get('latency_ms_p95_std')
    bimodal_latency = (
        int(row.get('n_trials') or 0) >= 2
        and latency_p95 not in (None, 0)
        and latency_p95_std not in (None, 0)
        and float(latency_p95_std) / float(latency_p95) >= BIMODAL_LATENCY_STD_RATIO
    )

    low_power_samples = (
        coverage['power_measurement']
        and int(row.get('n_power_samples') or 0)
        < low_power_samples_threshold(row.get('duration_s'))
    )

    return {
        'flag_negative_incremental_energy': int(
            row.get('energy_inc_per_inf_j') is not None
            and float(row.get('energy_inc_per_inf_j')) < 0
        ),
        'flag_low_power_samples': int(low_power_samples),
        'flag_thermal_rise': int(
            max_temp_rise is not None and max_temp_rise >= THERMAL_RISE_THRESHOLD_C
        ),
        'flag_bimodal_latency': int(bimodal_latency),
        'flag_partial_coverage': int(not coverage['complete']),
        'coverage_gaps_json': json.dumps(coverage['missing'], sort_keys=True),
    }


def annotate_profile(row: dict, quality_models: set[str],
                     trial_count: int | None = None) -> tuple[dict, dict]:
    coverage = coverage_for_profile(row)
    flags = measurement_flags_for_profile(row, coverage)
    return flags, coverage


def trial_count_index(conn) -> dict[tuple[str, str, str, int, int], int]:
    rows = conn.execute(
        "SELECT run_id, device, model, batch_size, dvfs_mode, COUNT(*) AS n "
        "FROM trials GROUP BY run_id, device, model, batch_size, dvfs_mode"
    ).fetchall()
    return {
        (r['run_id'], r['device'], r['model'], int(r['batch_size']), int(r['dvfs_mode'])): int(r['n'])
        for r in rows
    }


def refresh_profile_annotations(conn) -> int:
    """Backfill / refresh Phase 3 annotations for all profiles."""
    q_rows = conn.execute("SELECT model, task_type, metric_name, source FROM quality").fetchall()
    quality_models = {r['model'] for r in q_rows}
    trial_counts = trial_count_index(conn)
    profiles = conn.execute(
        "SELECT device, model, batch_size, dvfs_mode, run_id, "
        "energy_per_inf_j, p_wall_avg_w, n_power_samples, energy_inc_per_inf_j, "
        "latency_ms_p95, latency_ms_p95_std, n_trials, duration_s, "
        "temp_cpu_before_c, temp_cpu_after_c, temp_gpu_before_c, temp_gpu_after_c "
        "FROM profiles"
    ).fetchall()

    for row in profiles:
        rec = dict(row)
        key = (
            rec.get('run_id') or '',
            rec['device'],
            rec['model'],
            int(rec['batch_size']),
            int(rec['dvfs_mode']),
        )
        flags, _ = annotate_profile(rec, quality_models, trial_count=trial_counts.get(key, 0))
        conn.execute(
            "UPDATE profiles SET "
            "flag_negative_incremental_energy=?, "
            "flag_low_power_samples=?, "
            "flag_thermal_rise=?, "
            "flag_bimodal_latency=?, "
            "flag_partial_coverage=?, "
            "coverage_gaps_json=?, "
            "updated_at=strftime('%Y-%m-%dT%H:%M:%S', 'now') "
            "WHERE device=? AND model=? AND batch_size=? AND dvfs_mode=?",
            (
                flags['flag_negative_incremental_energy'],
                flags['flag_low_power_samples'],
                flags['flag_thermal_rise'],
                flags['flag_bimodal_latency'],
                flags['flag_partial_coverage'],
                flags['coverage_gaps_json'],
                rec['device'],
                rec['model'],
                rec['batch_size'],
                rec['dvfs_mode'],
            ),
        )
    conn.commit()
    return len(profiles)


def measurement_flags_payload(row: dict) -> dict:
    return {
        'negative_incremental_energy': _bool(row.get('flag_negative_incremental_energy')),
        'incremental_unreliable': _bool(row.get('flag_incremental_unreliable')),
        'low_power_samples': _bool(row.get('flag_low_power_samples')),
        'thermal_rise': _bool(row.get('flag_thermal_rise')),
        'bimodal_latency': _bool(row.get('flag_bimodal_latency')),
        'partial_coverage': _bool(row.get('flag_partial_coverage')),
    }


def coverage_payload(row: dict, quality_models: set[str] | None = None,
                     trial_count: int | None = None) -> dict:
    coverage = coverage_for_profile(row)
    return {
        'power_measurement': coverage['power_measurement'],
        'temperature_observation': coverage['temperature_observation'],
        'missing': coverage['missing'],
        'complete': coverage['complete'],
    }


def provenance_payload(row: dict, trial_count: int | None = None) -> dict:
    provenance = provenance_for_profile(row, trial_count=trial_count)
    return {
        'run_provenance': provenance['run_provenance'],
        'trial_truth': provenance['trial_truth'],
        'trial_count_consistent': provenance['trial_count_consistent'],
        'missing': provenance['missing'],
        'complete': provenance['complete'],
    }


def integrity_payload(row: dict, quality_models: set[str]) -> dict:
    integrity = integrity_for_profile(row, quality_models)
    return {
        'quality_metadata': integrity['quality_metadata'],
        'missing': integrity['missing'],
        'complete': integrity['complete'],
    }


def configured_profile_space() -> dict:
    """Configured measurement universe from devices.yaml + profiling.yaml."""
    space = {}
    for device, cfg in get_all_devices().items():
        if 'dvfs_modes' in cfg:
            dvfs_modes = [int(k) for k in cfg['dvfs_modes'].keys()]
        elif 'dvfs_clocks' in cfg:
            dvfs_modes = list(range(len(cfg['dvfs_clocks'])))
        else:
            dvfs_modes = [int(cfg.get('current_mode', 0))]
        combos = []
        for model in cfg.get('compatible_models', []):
            for bs in batch_sizes():
                for dvfs_mode in dvfs_modes:
                    combos.append((device, model, int(bs), int(dvfs_mode)))
        space[device] = {
            'dvfs_modes': dvfs_modes,
            'compatible_models': list(cfg.get('compatible_models', [])),
            'combos': combos,
        }
    return space


def build_coverage_report(db_path: str = DB_PATH) -> dict:
    from db import get_connection, init_db, load_all_profiles, load_quality_records

    ensure_results_layout()
    conn = get_connection(db_path)
    init_db(conn)
    refresh_profile_annotations(conn)
    trial_counts = trial_count_index(conn)
    profiles = load_all_profiles(db_path)
    quality_records = load_quality_records(db_path)
    conn.close()

    _, quality_models = build_quality_index(quality_records)
    configured = configured_profile_space()
    measured_set = {
        (row['device'], row['model'], int(row['batch_size']), int(row['dvfs_mode']))
        for row in profiles
    }
    configured_set = {
        combo for cfg in configured.values() for combo in cfg['combos']
    }
    out_of_registry = sorted(measured_set - configured_set)

    summary = {
        'configured_profiles': len(configured_set),
        'measured_profiles': len(measured_set),
        'missing_profiles': len(configured_set - measured_set),
        'out_of_registry_profiles': len(out_of_registry),
        'quality_models': len(quality_models),
        'profiles_with_partial_coverage': sum(
            1 for row in profiles if row.get('flag_partial_coverage')
        ),
        'profiles_with_provenance_gaps': sum(
            1 for row in profiles
            if not provenance_for_profile(
                row,
                trial_count=trial_counts.get((
                    row.get('run_id') or '',
                    row['device'],
                    row['model'],
                    int(row['batch_size']),
                    int(row['dvfs_mode']),
                ), 0),
            )['complete']
        ),
        'profiles_with_integrity_gaps': sum(
            1 for row in profiles if not integrity_for_profile(row, quality_models)['complete']
        ),
    }

    by_device = []
    for device, cfg in sorted(configured.items()):
        configured_combos = set(cfg['combos'])
        device_rows = [row for row in profiles if row['device'] == device]
        measured_combos = {
            (row['device'], row['model'], int(row['batch_size']), int(row['dvfs_mode']))
            for row in device_rows
        }
        missing = sorted(configured_combos - measured_combos)
        flag_counts = {
            'negative_incremental_energy': sum(
                1 for row in device_rows if row.get('flag_negative_incremental_energy')
            ),
            'low_power_samples': sum(
                1 for row in device_rows if row.get('flag_low_power_samples')
            ),
            'thermal_rise': sum(
                1 for row in device_rows if row.get('flag_thermal_rise')
            ),
            'bimodal_latency': sum(
                1 for row in device_rows if row.get('flag_bimodal_latency')
            ),
            'partial_coverage': sum(
                1 for row in device_rows if row.get('flag_partial_coverage')
            ),
        }
        measured_facets = {
            'power_measurement_rows': sum(
                1 for row in device_rows
                if row.get('energy_per_inf_j') is not None and row.get('p_wall_avg_w') is not None
            ),
            'temperature_observation_rows': sum(
                1 for row in device_rows
                if coverage_for_profile(row)['temperature_observation']
            ),
        }
        provenance_facets = {
            'run_provenance_rows': sum(1 for row in device_rows if row.get('run_id')),
            'trial_backed_rows': sum(
                1 for row in device_rows
                if provenance_for_profile(
                    row,
                    trial_count=trial_counts.get((
                        row.get('run_id') or '',
                        row['device'],
                        row['model'],
                        int(row['batch_size']),
                        int(row['dvfs_mode']),
                    ), 0),
                )['trial_truth']
            ),
            'trial_count_consistent_rows': sum(
                1 for row in device_rows
                if provenance_for_profile(
                    row,
                    trial_count=trial_counts.get((
                        row.get('run_id') or '',
                        row['device'],
                        row['model'],
                        int(row['batch_size']),
                        int(row['dvfs_mode']),
                    ), 0),
                )['trial_count_consistent']
            ),
        }
        integrity_facets = {
            'quality_metadata_rows': sum(
                1 for row in device_rows if row.get('model') in quality_models
            ),
        }
        by_device.append({
            'device': device,
            'configured_profiles': len(configured_combos),
            'measured_profiles': len(measured_combos),
            'missing_profiles': len(missing),
            'missing_examples': [
                {
                    'model': model,
                    'batch_size': bs,
                    'dvfs_mode': dvfs_mode,
                }
                for _, model, bs, dvfs_mode in missing[:10]
            ],
            'measured_facets': measured_facets,
            'provenance_facets': provenance_facets,
            'integrity_facets': integrity_facets,
            'flag_counts': flag_counts,
        })

    report = {
        'version': '1.0',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'scope': 'coverage disclosure only; no authoritative subset selection',
        'summary': summary,
        'quality_coverage': {
            'models_with_quality_records': sorted(quality_models),
            'models_missing_quality_records': sorted(
                {
                    model
                    for cfg in configured.values()
                    for model in cfg['compatible_models']
                } - quality_models
            ),
        },
        'configured_space': {
            device: {
                'dvfs_modes': cfg['dvfs_modes'],
                'compatible_models': cfg['compatible_models'],
            }
            for device, cfg in sorted(configured.items())
        },
        'by_device': by_device,
        'out_of_registry_examples': [
            {
                'device': device,
                'model': model,
                'batch_size': bs,
                'dvfs_mode': dvfs_mode,
            }
            for device, model, bs, dvfs_mode in out_of_registry[:10]
        ],
    }
    return report


def write_coverage_report(db_path: str = DB_PATH,
                          output_path: str | None = None) -> dict:
    ensure_results_layout()
    report = build_coverage_report(db_path)
    out_path = output_path or os.path.join(MEASURED_RESULTS_DIR, 'coverage_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    return report


def _count_tensor_entries(tensor: dict) -> int:
    return sum(
        len(bs_map)
        for dev_map in tensor.values()
        for model_map in dev_map.values()
        for bs_map in model_map.values()
    )


def _raw_trial_path(run_id: str, device: str, model: str,
                    batch_size: int, dvfs_mode: int) -> str:
    return os.path.join(
        RAW_RESULTS_DIR,
        run_id,
        f'{device}_{model}_bs{batch_size}_dvfs{dvfs_mode}.json',
    )


def build_export_integrity_report(db_path: str = DB_PATH,
                                  export_path: str | None = None,
                                  coverage_path: str | None = None) -> dict:
    from db import get_connection, init_db, load_all_profiles, load_quality_records

    ensure_results_layout()
    tensor_path = export_path or os.path.join(MEASURED_RESULTS_DIR, 'eep_tensor.json')
    cov_path = coverage_path or os.path.join(MEASURED_RESULTS_DIR, 'coverage_report.json')

    conn = get_connection(db_path)
    init_db(conn)
    refresh_profile_annotations(conn)
    profiles = load_all_profiles(db_path)
    quality_records = load_quality_records(db_path)
    trial_counts = trial_count_index(conn)
    conn.close()

    _, quality_models = build_quality_index(quality_records)

    if os.path.exists(tensor_path):
        with open(tensor_path) as f:
            tensor_export = json.load(f)
    else:
        tensor_export = {'n_entries': 0, 'tensor': {}, 'quality_meta': {}}

    coverage_report = None
    if os.path.exists(cov_path):
        with open(cov_path) as f:
            coverage_report = json.load(f)

    export_entry_count = _count_tensor_entries(tensor_export.get('tensor', {}))
    db_entry_count = sum(1 for row in profiles if row.get('energy_per_inf_j') is not None)

    missing_entry_metadata = 0
    missing_quality_metadata = 0
    for dev_map in tensor_export.get('tensor', {}).values():
        for model, model_map in dev_map.items():
            for dvfs_map in model_map.values():
                for entry in dvfs_map.values():
                    required_entry_fields = {
                        'measurement_flags', 'coverage', 'provenance', 'integrity'
                    }
                    if not required_entry_fields.issubset(entry.keys()):
                        missing_entry_metadata += 1
                    if entry.get('Q') is not None:
                        required = {'Q_task_type', 'Q_metric', 'Q_direction', 'Q_source'}
                        if not required.issubset(entry.keys()):
                            missing_quality_metadata += 1
                        if model not in tensor_export.get('quality_meta', {}):
                            missing_quality_metadata += 1

    provenance_gaps = 0
    trial_mismatches = 0
    missing_raw_json = 0
    for row in profiles:
        key = (
            row.get('run_id') or '',
            row['device'],
            row['model'],
            int(row['batch_size']),
            int(row['dvfs_mode']),
        )
        trial_count = trial_counts.get(key, 0)
        provenance = provenance_for_profile(row, trial_count=trial_count)
        if not provenance['run_provenance'] or not provenance['trial_truth']:
            provenance_gaps += 1
        if not provenance['trial_count_consistent']:
            trial_mismatches += 1
        if row.get('run_id'):
            if not os.path.exists(_raw_trial_path(
                row['run_id'], row['device'], row['model'],
                int(row['batch_size']), int(row['dvfs_mode'])
            )):
                missing_raw_json += 1

    checks = {
        'db_vs_export_entry_count': db_entry_count == export_entry_count,
        'header_vs_nested_entry_count': export_entry_count == int(tensor_export.get('n_entries', 0)),
        'entry_quality_metadata_present': missing_quality_metadata == 0,
        'entry_measurement_metadata_present': missing_entry_metadata == 0,
        'coverage_report_present': coverage_report is not None,
        'coverage_summary_matches_db': (
            coverage_report is not None
            and coverage_report.get('summary', {}).get('measured_profiles') == len(profiles)
        ),
    }

    hard_fail = [
        name for name in (
            'db_vs_export_entry_count',
            'header_vs_nested_entry_count',
            'entry_quality_metadata_present',
            'entry_measurement_metadata_present',
        )
        if not checks[name]
    ]
    warn = provenance_gaps > 0 or trial_mismatches > 0 or missing_raw_json > 0

    return {
        'version': '1.0',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'scope': 'export integrity only; not a downstream fairness or admissibility checker',
        'status': 'fail' if hard_fail else 'warn' if warn else 'pass',
        'checks': checks,
        'counts': {
            'db_profiles': len(profiles),
            'db_exportable_profiles': db_entry_count,
            'export_header_n_entries': int(tensor_export.get('n_entries', 0)),
            'export_nested_entries': export_entry_count,
            'missing_entry_metadata': missing_entry_metadata,
            'missing_quality_metadata': missing_quality_metadata,
            'provenance_gaps': provenance_gaps,
            'trial_count_mismatches': trial_mismatches,
            'missing_raw_json_artifacts': missing_raw_json,
        },
        'paths': {
            'db': db_path,
            'export': tensor_path,
            'coverage_report': cov_path,
            'manifests_dir': MANIFESTS_RESULTS_DIR,
        },
    }


def write_export_integrity_report(db_path: str = DB_PATH,
                                  export_path: str | None = None,
                                  coverage_path: str | None = None,
                                  output_path: str | None = None) -> dict:
    ensure_results_layout()
    report = build_export_integrity_report(
        db_path=db_path,
        export_path=export_path,
        coverage_path=coverage_path,
    )
    out_path = output_path or os.path.join(MANIFESTS_RESULTS_DIR, 'export_integrity.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    return report
