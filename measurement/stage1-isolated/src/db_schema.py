"""
EEP Profiler DB schema helpers.

Clean-cut canonical schema only.
- No in-place historical migration chain
- No legacy-version upgrade logic
- Existing pre-cutover DBs must be archived and replaced
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

TRIALS_DDL = """\
CREATE TABLE IF NOT EXISTS trials (
    trial_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    device          TEXT NOT NULL,
    model           TEXT NOT NULL,
    batch_size      INTEGER NOT NULL,
    dvfs_mode       INTEGER NOT NULL,
    trial_index     INTEGER NOT NULL,

    n_infer         INTEGER NOT NULL DEFAULT 0,
    duration_s      REAL    NOT NULL DEFAULT 0,
    ep              TEXT    NOT NULL DEFAULT '',
    elapsed_s       REAL    NOT NULL DEFAULT 0,
    t_start         REAL    NOT NULL DEFAULT 0,
    t_end           REAL    NOT NULL DEFAULT 0,
    active_start_ts REAL    NOT NULL DEFAULT 0,
    active_end_ts   REAL    NOT NULL DEFAULT 0,

    p_wall_avg_w         REAL,
    energy_total_j       REAL,
    energy_per_inf_j     REAL,
    energy_inc_j         REAL,
    energy_inc_per_inf_j REAL,
    n_power_samples      INTEGER NOT NULL DEFAULT 0,
    watts_idle           REAL,
    watts_idle_true      REAL,
    watts_active         REAL,
    idle_baseline_margin_w         REAL,
    energy_inc_canonical_j         REAL,
    energy_inc_canonical_per_inf_j REAL,
    incremental_source             TEXT    NOT NULL DEFAULT 'none',
    watts_idle_median              REAL,
    watts_idle_std                 REAL,
    watts_idle_true_median         REAL,
    watts_idle_true_std            REAL,
    p_plateau_w                    REAL,
    plateau_start_ts               REAL,
    plateau_end_ts                 REAL,
    plateau_duration_s             REAL,
    plateau_sample_count           INTEGER NOT NULL DEFAULT 0,
    energy_total_plateau_j         REAL,
    energy_inc_plateau_j           REAL,
    flag_no_plateau_detected       INTEGER NOT NULL DEFAULT 0,
    flag_trace_jitter_high         INTEGER NOT NULL DEFAULT 0,
    flag_incremental_unreliable    INTEGER NOT NULL DEFAULT 0,
    flag_plateau_incremental_unreliable INTEGER NOT NULL DEFAULT 0,

    latency_ms_mean REAL NOT NULL DEFAULT 0,
    latency_ms_p50  REAL NOT NULL DEFAULT 0,
    latency_ms_p95  REAL NOT NULL DEFAULT 0,
    latency_ms_p99  REAL NOT NULL DEFAULT 0,
    latency_ms_std  REAL NOT NULL DEFAULT 0,

    throughput_ips  REAL NOT NULL DEFAULT 0,

    temp_cpu_before_c REAL NOT NULL DEFAULT -1.0,
    temp_cpu_after_c  REAL NOT NULL DEFAULT -1.0,
    temp_gpu_before_c REAL NOT NULL DEFAULT -1.0,
    temp_gpu_after_c  REAL NOT NULL DEFAULT -1.0,

    bench_json           TEXT NOT NULL DEFAULT '',
    power_trace_json     TEXT NOT NULL DEFAULT '[]',
    power_windowing_json TEXT NOT NULL DEFAULT '{}',

    timestamp       TEXT NOT NULL DEFAULT '',

    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE (run_id, device, model, batch_size, dvfs_mode, trial_index)
);

CREATE INDEX IF NOT EXISTS idx_trials_run
    ON trials(run_id);
CREATE INDEX IF NOT EXISTS idx_trials_device_model
    ON trials(device, model, batch_size, dvfs_mode);
"""


DDL = """\
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    device          TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    batch_size      INTEGER NOT NULL,
    dvfs_mode       INTEGER NOT NULL,

    n_infer         INTEGER NOT NULL DEFAULT 0,
    duration_s      REAL    NOT NULL DEFAULT 0,
    ep              TEXT    NOT NULL DEFAULT '',

    p_wall_avg_w         REAL,
    energy_total_j       REAL,
    energy_per_inf_j     REAL,
    energy_inc_j         REAL,
    energy_inc_per_inf_j REAL,
    n_power_samples      INTEGER NOT NULL DEFAULT 0,
    watts_idle           REAL,
    watts_idle_true      REAL,
    watts_active         REAL,

    latency_ms_mean REAL NOT NULL DEFAULT 0,
    latency_ms_p50  REAL NOT NULL DEFAULT 0,
    latency_ms_p95  REAL NOT NULL DEFAULT 0,
    latency_ms_p99  REAL NOT NULL DEFAULT 0,
    latency_ms_std  REAL NOT NULL DEFAULT 0,

    throughput_ips  REAL NOT NULL DEFAULT 0,

    bench_elapsed_s REAL NOT NULL DEFAULT 0,
    wall_elapsed_s  REAL NOT NULL DEFAULT 0,

    n_trials                INTEGER NOT NULL DEFAULT 1,
    energy_per_inf_j_std    REAL,
    latency_ms_p95_std      REAL,
    throughput_ips_std      REAL,
    p_wall_avg_w_std        REAL,

    temp_cpu_before_c REAL NOT NULL DEFAULT -1.0,
    temp_cpu_after_c  REAL NOT NULL DEFAULT -1.0,
    temp_gpu_before_c REAL NOT NULL DEFAULT -1.0,
    temp_gpu_after_c  REAL NOT NULL DEFAULT -1.0,

    flag_negative_incremental_energy      INTEGER NOT NULL DEFAULT 0,
    flag_low_power_samples                INTEGER NOT NULL DEFAULT 0,
    flag_thermal_rise                     INTEGER NOT NULL DEFAULT 0,
    flag_bimodal_latency                  INTEGER NOT NULL DEFAULT 0,
    flag_partial_coverage                 INTEGER NOT NULL DEFAULT 0,
    coverage_gaps_json                    TEXT    NOT NULL DEFAULT '[]',
    idle_baseline_margin_w                REAL,
    watts_idle_median                     REAL,
    watts_idle_std                        REAL,
    watts_idle_true_median                REAL,
    watts_idle_true_std                   REAL,
    p_plateau_w                           REAL,
    plateau_duration_s                    REAL,
    plateau_sample_count                  INTEGER NOT NULL DEFAULT 0,
    energy_total_plateau_j                REAL,
    energy_inc_plateau_j                  REAL,
    flag_no_plateau_detected              INTEGER NOT NULL DEFAULT 0,
    flag_trace_jitter_high                INTEGER NOT NULL DEFAULT 0,
    flag_incremental_unreliable           INTEGER NOT NULL DEFAULT 0,
    flag_plateau_incremental_unreliable   INTEGER NOT NULL DEFAULT 0,
    energy_inc_canonical_j                REAL,
    energy_inc_canonical_per_inf_j        REAL,
    incremental_source                    TEXT    NOT NULL DEFAULT 'none',

    timestamp  TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    run_id     TEXT,

    PRIMARY KEY (device, model, batch_size, dvfs_mode)
);

CREATE INDEX IF NOT EXISTS idx_profiles_device
    ON profiles(device);
CREATE INDEX IF NOT EXISTS idx_profiles_model
    ON profiles(model);
CREATE INDEX IF NOT EXISTS idx_profiles_device_dvfs
    ON profiles(device, dvfs_mode);

CREATE TABLE IF NOT EXISTS quality (
    model             TEXT    NOT NULL,
    task_type         TEXT    NOT NULL DEFAULT 'classification',
    metric_name       TEXT    NOT NULL DEFAULT 'top1_accuracy',
    source            TEXT    NOT NULL DEFAULT 'measured',
    q_value           REAL    NOT NULL,
    metric_direction  TEXT    NOT NULL DEFAULT 'higher_is_better',
    dataset           TEXT    NOT NULL DEFAULT 'imagenet-1k-val',
    updated_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),

    PRIMARY KEY (model, task_type, metric_name, source)
);

-- Vocabulary note: "plateau" (internal column prefix p_plateau_*) and "stable"
-- (view alias stable_*) refer to the same detected stable active-segment window.
-- plateau = raw profiler term; stable = public-facing alias in this view.
CREATE VIEW eep_tensor_view AS
SELECT
    p.device,
    p.model,
    p.dvfs_mode,
    p.batch_size,
    p.energy_per_inf_j                 AS energy_per_inf_j,
    p.energy_inc_canonical_per_inf_j   AS extra_energy_per_inf_j,
    p.watts_idle_true                  AS idle_power_w,
    p.p_wall_avg_w                     AS avg_power_w,
    p.p_plateau_w                      AS stable_power_w,
    p.energy_total_j                   AS avg_energy_j,
    p.energy_total_plateau_j           AS stable_energy_j,
    p.energy_inc_j                     AS avg_extra_energy_j,
    p.energy_inc_plateau_j             AS stable_extra_energy_j,
    (NOT p.flag_no_plateau_detected AND p.p_plateau_w IS NOT NULL) AS stable_detected,
    (NOT p.flag_incremental_unreliable) AS extra_energy_reliable,
    p.incremental_source               AS extra_energy_basis,
    p.latency_ms_mean                  AS latency_ms_mean,
    p.latency_ms_p50                   AS latency_ms_p50,
    p.latency_ms_p95                   AS latency_ms_p95,
    p.latency_ms_p99                   AS latency_ms_p99,
    p.latency_ms_std                   AS latency_ms_std,
    p.throughput_ips                   AS throughput_ips,
    COALESCE(qm.q_value, qp.q_value)   AS q_value,
    COALESCE(qm.task_type, qp.task_type) AS q_task_type,
    COALESCE(qm.metric_name, qp.metric_name) AS q_metric_name,
    COALESCE(qm.metric_direction, qp.metric_direction) AS q_metric_direction,
    COALESCE(qm.source, qp.source)     AS q_source,
    p.n_infer,
    p.ep,
    p.timestamp
FROM profiles p
LEFT JOIN quality qm
    ON p.model = qm.model
    AND qm.task_type = 'classification'
    AND qm.source = 'measured'
LEFT JOIN quality qp
    ON p.model = qp.model
    AND qp.task_type = 'classification'
    AND qp.source = 'published'
WHERE p.energy_per_inf_j IS NOT NULL;

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    operator        TEXT NOT NULL DEFAULT '',
    git_commit      TEXT NOT NULL DEFAULT '',
    config_snapshot TEXT NOT NULL DEFAULT '{}',
    devices         TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'running',
    n_trials_total  INTEGER NOT NULL DEFAULT 0,
    n_trials_ok     INTEGER NOT NULL DEFAULT 0,
    notes           TEXT NOT NULL DEFAULT ''
);
"""


PROFILE_COLS = [
    'device', 'model', 'batch_size', 'dvfs_mode',
    'n_infer', 'duration_s', 'ep',
    'p_wall_avg_w', 'energy_total_j', 'energy_per_inf_j',
    'energy_inc_j', 'energy_inc_per_inf_j', 'n_power_samples',
    'watts_idle', 'watts_idle_true', 'watts_active',
    'latency_ms_mean', 'latency_ms_p50', 'latency_ms_p95', 'latency_ms_p99',
    'latency_ms_std', 'throughput_ips',
    'bench_elapsed_s', 'wall_elapsed_s',
    'n_trials', 'energy_per_inf_j_std', 'latency_ms_p95_std',
    'throughput_ips_std', 'p_wall_avg_w_std',
    'temp_cpu_before_c', 'temp_cpu_after_c',
    'temp_gpu_before_c', 'temp_gpu_after_c',
    'flag_negative_incremental_energy', 'flag_low_power_samples',
    'flag_thermal_rise', 'flag_bimodal_latency',
    'flag_partial_coverage', 'coverage_gaps_json',
    'idle_baseline_margin_w',
    'watts_idle_median', 'watts_idle_std',
    'watts_idle_true_median', 'watts_idle_true_std',
    'p_plateau_w', 'plateau_duration_s', 'plateau_sample_count',
    'energy_total_plateau_j', 'energy_inc_plateau_j',
    'flag_no_plateau_detected', 'flag_trace_jitter_high',
    'flag_incremental_unreliable', 'flag_plateau_incremental_unreliable',
    'energy_inc_canonical_j', 'energy_inc_canonical_per_inf_j',
    'incremental_source',
    'timestamp', 'run_id',
]

TRIAL_COLS = [
    'run_id', 'device', 'model', 'batch_size', 'dvfs_mode', 'trial_index',
    'n_infer', 'duration_s', 'ep', 'elapsed_s', 't_start', 't_end',
    'active_start_ts', 'active_end_ts',
    'p_wall_avg_w', 'energy_total_j', 'energy_per_inf_j',
    'energy_inc_j', 'energy_inc_per_inf_j', 'n_power_samples',
    'watts_idle', 'watts_idle_true', 'watts_active', 'idle_baseline_margin_w',
    'energy_inc_canonical_j', 'energy_inc_canonical_per_inf_j', 'incremental_source',
    'watts_idle_median', 'watts_idle_std',
    'watts_idle_true_median', 'watts_idle_true_std',
    'p_plateau_w', 'plateau_start_ts', 'plateau_end_ts', 'plateau_duration_s',
    'plateau_sample_count', 'energy_total_plateau_j', 'energy_inc_plateau_j',
    'flag_no_plateau_detected', 'flag_trace_jitter_high',
    'flag_incremental_unreliable', 'flag_plateau_incremental_unreliable',
    'latency_ms_mean', 'latency_ms_p50', 'latency_ms_p95',
    'latency_ms_p99', 'latency_ms_std', 'throughput_ips',
    'temp_cpu_before_c', 'temp_cpu_after_c',
    'temp_gpu_before_c', 'temp_gpu_after_c',
    'bench_json', 'power_trace_json', 'power_windowing_json', 'timestamp',
]

NULLABLE_COLS = frozenset({
    'p_wall_avg_w', 'energy_total_j', 'energy_per_inf_j',
    'energy_inc_j', 'energy_inc_per_inf_j',
    'watts_idle', 'watts_idle_true', 'watts_active',
    'idle_baseline_margin_w',
    'watts_idle_median', 'watts_idle_std',
    'watts_idle_true_median', 'watts_idle_true_std',
    'p_plateau_w', 'plateau_duration_s',
    'energy_total_plateau_j', 'energy_inc_plateau_j',
    'energy_inc_canonical_j', 'energy_inc_canonical_per_inf_j',
    'energy_per_inf_j_std', 'latency_ms_p95_std',
    'throughput_ips_std', 'p_wall_avg_w_std',
})

TEXT_NULLABLE_COLS = frozenset({'run_id'})

INT_COLS = frozenset({
    'batch_size', 'dvfs_mode', 'n_infer', 'n_power_samples', 'n_trials',
    'flag_negative_incremental_energy', 'flag_low_power_samples',
    'flag_thermal_rise', 'flag_bimodal_latency', 'flag_partial_coverage',
    'plateau_sample_count', 'flag_no_plateau_detected', 'flag_trace_jitter_high',
    'flag_incremental_unreliable', 'flag_plateau_incremental_unreliable',
})

FLOAT_COLS = frozenset({
    'duration_s', 'latency_ms_mean', 'latency_ms_p50', 'latency_ms_p95',
    'latency_ms_p99', 'latency_ms_std', 'throughput_ips',
    'bench_elapsed_s', 'wall_elapsed_s',
    'temp_cpu_before_c', 'temp_cpu_after_c',
    'temp_gpu_before_c', 'temp_gpu_after_c',
})

PROFILE_INSERT_SQL = (
    f"INSERT OR REPLACE INTO profiles ({', '.join(PROFILE_COLS)}, updated_at) "
    f"VALUES ({', '.join(['?'] * len(PROFILE_COLS))}, strftime('%Y-%m-%dT%H:%M:%S', 'now'))"
)

TRIAL_INSERT_SQL = (
    f"INSERT OR REPLACE INTO trials ({', '.join(TRIAL_COLS)}) "
    f"VALUES ({', '.join(['?'] * len(TRIAL_COLS))})"
)


def get_connection(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA cache_size = -8000")
    return conn


def _get_db_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0


def init_db(conn: sqlite3.Connection):
    db_ver = _get_db_version(conn)
    if db_ver not in (0, SCHEMA_VERSION):
        raise RuntimeError(
            f'Unsupported DB schema version {db_ver}. '
            'Archive the legacy DB and create a fresh canonical DB for this runtime.'
        )

    conn.execute("DROP VIEW IF EXISTS eep_tensor_view")
    conn.executescript(DDL)
    conn.executescript(TRIALS_DDL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),)
    )
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta VALUES "
        "('created_at', strftime('%Y-%m-%dT%H:%M:%S', 'now'))"
    )
    conn.commit()
