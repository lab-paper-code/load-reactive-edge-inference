"""
EEP Profiler, SQLite access facade.

Schema, DDL, and migrations live in `db_schema.py`.
This module keeps the runtime API smaller:
- profile/trial upserts
- quality CRUD helpers
- export helpers
- run/trial loaders
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import os
import sqlite3

try:
    from .db_schema import (
        FLOAT_COLS as _FLOAT_COLS,
        INT_COLS as _INT_COLS,
        NULLABLE_COLS as _NULLABLE,
        PROFILE_COLS as _PROFILE_COLS,
        PROFILE_INSERT_SQL as _INSERT_SQL,
        TEXT_NULLABLE_COLS as _TEXT_NULLABLE,
        TRIAL_COLS as _TRIAL_COLS,
        TRIAL_INSERT_SQL as _TRIAL_INSERT_SQL,
        get_connection,
        init_db,
    )
except ImportError:
    from db_schema import (
        FLOAT_COLS as _FLOAT_COLS,
        INT_COLS as _INT_COLS,
        NULLABLE_COLS as _NULLABLE,
        PROFILE_COLS as _PROFILE_COLS,
        PROFILE_INSERT_SQL as _INSERT_SQL,
        TEXT_NULLABLE_COLS as _TEXT_NULLABLE,
        TRIAL_COLS as _TRIAL_COLS,
        TRIAL_INSERT_SQL as _TRIAL_INSERT_SQL,
        get_connection,
        init_db,
    )


def _coerce(result: dict) -> tuple:
    """Convert dict to profiles INSERT parameter tuple."""

    def _v(col):
        val = result.get(col)
        if col == 'coverage_gaps_json':
            return val if val not in (None, '') else '[]'
        if col in _NULLABLE:
            return float(val) if val not in (None, '') else None
        if col in _TEXT_NULLABLE:
            return val
        if col in _INT_COLS:
            return int(val) if val not in (None, '') else 0
        if col in _FLOAT_COLS:
            return float(val) if val not in (None, '') else 0.0
        return val if val is not None else ''

    return tuple(_v(col) for col in _PROFILE_COLS)


def _trial_tuple(trial: dict) -> tuple:
    """Convert dict to trials INSERT parameter tuple."""

    def _v(col):
        val = trial.get(col)
        if col in _NULLABLE:
            return float(val) if val not in (None, '') else None
        if col in _INT_COLS:
            return int(val) if val not in (None, '') else 0
        return val if val is not None else ''

    return tuple(_v(col) for col in _TRIAL_COLS)


def upsert_profile(conn: sqlite3.Connection, result: dict):
    """Upsert a single measurement result."""
    conn.execute(_INSERT_SQL, _coerce(result))
    conn.commit()


def load_all_profiles(path: str) -> list:
    conn = get_connection(path)
    init_db(conn)
    rows = conn.execute(
        "SELECT * FROM profiles ORDER BY device, dvfs_mode, model, batch_size"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def load_profiles_df(path: str):
    import pandas as pd

    conn = get_connection(path)
    init_db(conn)
    df = pd.read_sql_query(
        "SELECT * FROM profiles ORDER BY device, dvfs_mode, model, batch_size",
        conn
    )
    conn.close()
    df["batch_size"] = df["batch_size"].astype(int)
    df["dvfs_mode"] = df["dvfs_mode"].astype(str)
    return df


def upsert_quality(conn: sqlite3.Connection, quality: dict | list):
    if isinstance(quality, dict):
        records = [
            {
                'model': model,
                'q_value': value,
                'task_type': 'classification',
                'metric_name': 'top1_accuracy',
                'metric_direction': 'higher_is_better',
                'dataset': 'imagenet-1k-val',
                'source': 'measured',
            }
            for model, value in quality.items()
        ]
    else:
        records = quality

    for rec in records:
        conn.execute(
            "INSERT OR REPLACE INTO quality "
            "(model, q_value, task_type, metric_name, metric_direction, "
            " dataset, source, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%S', 'now'))",
            (
                rec['model'],
                float(rec['q_value']),
                rec.get('task_type', 'classification'),
                rec.get('metric_name', 'top1_accuracy'),
                rec.get('metric_direction', 'higher_is_better'),
                rec.get('dataset', 'imagenet-1k-val'),
                rec.get('source', 'measured'),
            ),
        )
    conn.commit()


def load_quality(path: str, task_type: str = 'classification',
                 source: str = 'measured') -> dict:
    conn = get_connection(path)
    rows = conn.execute(
        "SELECT model, q_value FROM quality WHERE task_type = ? AND source = ?",
        (task_type, source),
    ).fetchall()
    conn.close()
    return {row['model']: row['q_value'] for row in rows}


def load_quality_records(path: str) -> list:
    conn = get_connection(path)
    rows = conn.execute(
        "SELECT model, q_value, task_type, metric_name, metric_direction, "
        "dataset, source, updated_at FROM quality"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def export_tensor_json(path: str, output_path: str):
    try:
        from .config import ensure_results_layout
        from .measurement_signals import (
            build_quality_index,
            coverage_payload,
            integrity_payload,
            measurement_flags_payload,
            provenance_payload,
            refresh_profile_annotations,
            trial_count_index,
        )
    except ImportError:
        from config import ensure_results_layout
        from measurement_signals import (
            build_quality_index,
            coverage_payload,
            integrity_payload,
            measurement_flags_payload,
            provenance_payload,
            refresh_profile_annotations,
            trial_count_index,
        )

    ensure_results_layout()
    conn = get_connection(path)
    init_db(conn)
    refresh_profile_annotations(conn)
    rows = conn.execute(
        "SELECT * FROM profiles WHERE energy_per_inf_j IS NOT NULL "
        "ORDER BY device, dvfs_mode, model, batch_size"
    ).fetchall()
    q_rows = conn.execute(
        "SELECT model, q_value, task_type, metric_name, metric_direction, "
        "dataset, source, updated_at FROM quality"
    ).fetchall()
    trial_counts = trial_count_index(conn)
    conn.close()

    quality_records = [dict(row) for row in q_rows]
    best_quality, quality_models = build_quality_index(quality_records)

    tensor = {}
    for row in rows:
        r = dict(row)
        q_rec = best_quality.get(r['model'])
        trial_count = trial_counts.get((
            r.get('run_id') or '',
            r['device'],
            r['model'],
            int(r['batch_size']),
            int(r['dvfs_mode']),
        ), 0)
        idle_power = (
            r.get('watts_idle_true')
            if r.get('watts_idle_true') is not None
            else r.get('watts_idle')
        )
        stable_detected = (
            not bool(r.get('flag_no_plateau_detected'))
            and r.get('p_plateau_w') is not None
        )
        extra_energy_reliable = not bool(r.get('flag_incremental_unreliable'))
        extra_energy_basis = r.get('incremental_source') or 'none'
        entry = {
            'E_mean': r['energy_per_inf_j'],
            'E_inc': (
                r.get('energy_inc_canonical_per_inf_j')
                if r.get('energy_inc_canonical_per_inf_j') is not None
                else r.get('energy_inc_per_inf_j')
            ),
            'idle_power_w': idle_power or 0.0,
            'avg_power_w': r['p_wall_avg_w'] or 0.0,
            'stable_power_w': r.get('p_plateau_w'),
            'L_mean': r['latency_ms_mean'],
            'L_p50': r['latency_ms_p50'],
            'L_p95': r['latency_ms_p95'],
            'L_p99': r['latency_ms_p99'],
            'L_std': r['latency_ms_std'],
            'T_mean': r['throughput_ips'],
            'avg_energy_j': r.get('energy_total_j'),
            'stable_energy_j': r.get('energy_total_plateau_j'),
            'avg_extra_energy_j': r.get('energy_inc_j'),
            'stable_extra_energy_j': r.get('energy_inc_plateau_j'),
            'stable_detected': stable_detected,
            'extra_energy_reliable': extra_energy_reliable,
            'extra_energy_basis': extra_energy_basis,
            'Q': q_rec['q_value'] if q_rec else None,
            'n_infer': r['n_infer'],
            'ep': r['ep'],
            'timestamp': r['timestamp'],
            'measurement_flags': measurement_flags_payload(r),
            'coverage': coverage_payload(r, quality_models, trial_count=trial_count),
            'provenance': provenance_payload(r, trial_count=trial_count),
            'integrity': integrity_payload(r, quality_models),
            'power_windowing': {
                'idle_power_w': idle_power,
                'pre_active_power_w': r.get('watts_idle'),
                'no_stable_window': bool(r.get('flag_no_plateau_detected')),
                'trace_jitter_high': bool(r.get('flag_trace_jitter_high')),
                'extra_energy_unreliable': bool(r.get('flag_incremental_unreliable')),
                'stable_extra_energy_unreliable': bool(
                    r.get('flag_plateau_incremental_unreliable')),
                'extra_energy_basis': extra_energy_basis,
                'extra_energy_margin_w': r.get('idle_baseline_margin_w'),
                'stable_duration_s': r.get('plateau_duration_s'),
                'stable_sample_count': r.get('plateau_sample_count'),
            },
        }
        if q_rec:
            entry['Q_task_type'] = q_rec['task_type']
            entry['Q_metric'] = q_rec['metric_name']
            entry['Q_direction'] = q_rec['metric_direction']
            entry['Q_source'] = q_rec.get('source', '')

        (tensor
         .setdefault(r['device'], {})
         .setdefault(r['model'], {})
         .setdefault(str(r['dvfs_mode']), {})
         [str(r['batch_size'])]) = entry

    q_meta = {}
    for rec in quality_records:
        model = rec['model']
        source = rec.get('source', 'measured')
        q_meta.setdefault(model, {})[source] = {
            'q_value': rec['q_value'],
            'task_type': rec['task_type'],
            'metric_name': rec['metric_name'],
            'metric_direction': rec['metric_direction'],
            'dataset': rec.get('dataset', ''),
        }

    output = {
        'version': '1.0',
        # (E, Q, L, T): energy, quality, latency, throughput per (device, model, dvfs, batch).
        'description': 'EEP Tensor: pre-profiled (E, Q, L, T) per (device, model, dvfs, batch)',
        'n_entries': len(rows),
        'surface_semantics': {
            'idle_power_w': 'true idle power before remote bench launch when available',
            'avg_power_w': 'whole-run average power over the measured active window',
            'stable_power_w': 'stable active-segment power when detected; null if not detected',
            'stable_detected': 'true when a stable active segment was detected',
            'extra_energy_reliable': 'true when baseline-subtracted extra energy is reliable enough to interpret directly',
            'extra_energy_basis': 'which measured window produced the selected extra-energy signal: stable, avg, or none',
        },
        'quality_meta': q_meta,
        'tensor': tensor,
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    return len(rows)


def create_run(conn: sqlite3.Connection, run_id: str,
               operator: str = '', git_commit: str = '',
               config_snapshot: str = '{}', devices: str = '[]',
               notes: str = '') -> str:
    import time

    conn.execute(
        "INSERT INTO runs (run_id, started_at, operator, git_commit, "
        "config_snapshot, devices, status, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
        (
            run_id,
            time.strftime('%Y-%m-%dT%H:%M:%S'),
            operator,
            git_commit,
            config_snapshot,
            devices,
            notes,
        ),
    )
    conn.commit()
    return run_id


# Default age (hours) after which a status='running' row is presumed
# orphaned. Imported by run_sweep._orphan_threshold_hours and
# scripts/mark_orphan_runs.py so the three call sites agree.
DEFAULT_ORPHAN_THRESHOLD_HOURS = 6.0


def abort_run(conn: sqlite3.Connection, run_id: str, reason: str) -> None:
    """Mark a run 'aborted', set finished_at, append a timestamped note.

    Shared between the atexit finalizer, startup reconciliation, and the
    manual mark_orphan_runs CLI so the note format and CASE idiom live
    in one place.
    """
    now_iso = datetime.now().replace(microsecond=0).isoformat()
    note = f' [{reason} at {now_iso}]'
    conn.execute(
        "UPDATE runs SET status='aborted', finished_at=?, "
        "notes=CASE WHEN notes = '' THEN ltrim(?) ELSE notes || ? END "
        "WHERE run_id=?",
        (now_iso, note, note, run_id),
    )
    conn.commit()


def reconcile_stale_running_runs(
    conn: sqlite3.Connection,
    threshold_hours: float = DEFAULT_ORPHAN_THRESHOLD_HOURS,
    dry_run: bool = False,
) -> list[str]:
    now = datetime.now()
    reconciled = []
    rows = conn.execute(
        "SELECT run_id, started_at FROM runs WHERE status='running'"
    ).fetchall()
    for row in rows:
        try:
            started_at = datetime.fromisoformat(row['started_at'])
        except (TypeError, ValueError):
            continue
        age_hours = (now - started_at).total_seconds() / 3600.0
        if age_hours <= threshold_hours:
            continue
        if not dry_run:
            abort_run(conn, row['run_id'], f'startup-reconciled age={age_hours:.1f}h')
        reconciled.append(row['run_id'])
    return reconciled


def finish_run(conn: sqlite3.Connection, run_id: str,
               status: str = 'completed',
               n_trials_total: int = 0, n_trials_ok: int = 0):
    import time

    conn.execute(
        "UPDATE runs SET finished_at=?, status=?, "
        "n_trials_total=?, n_trials_ok=? WHERE run_id=?",
        (
            time.strftime('%Y-%m-%dT%H:%M:%S'),
            status,
            n_trials_total,
            n_trials_ok,
            run_id,
        ),
    )
    conn.commit()


def insert_trial(conn: sqlite3.Connection, trial: dict):
    conn.execute(_TRIAL_INSERT_SQL, _trial_tuple(trial))
    conn.commit()


def load_trials(path: str, run_id: str = None,
                device: str = None, model: str = None) -> list:
    conn = get_connection(path)
    query = "SELECT * FROM trials WHERE 1=1"
    params = []
    if run_id:
        query += " AND run_id = ?"
        params.append(run_id)
    if device:
        query += " AND device = ?"
        params.append(device)
    if model:
        query += " AND model = ?"
        params.append(model)
    query += " ORDER BY trial_id"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def load_runs(path: str) -> list:
    conn = get_connection(path)
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY started_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def migrate_from_csv(csv_path: str, db_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    conn = get_connection(db_path)
    init_db(conn)
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    conn.executemany(_INSERT_SQL, [_coerce(row) for row in rows])
    conn.commit()
    conn.close()
    print(f'Migrated {len(rows)} rows: {csv_path} → {db_path}')
    return len(rows)
