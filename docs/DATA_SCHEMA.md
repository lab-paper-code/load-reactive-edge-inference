# Data Schema

This document describes the column definitions for the three paper-facing derived CSVs, the raw JSON measurement record schema, the isolated-profile database columns used by the pipeline, and the pipeline execution order.

> **Terminology note.** Column, field, and file names retain their original tokens (`wall`, `capacity`, `dvfs`, `segment`) for compatibility with the shipped data and pipeline. The descriptions below use the terminology of the accompanying paper: *AC input* power/energy (measured at the wall plug) for `wall`, *maximum sustainable throughput (MST)* for `capacity`, *power mode* for `dvfs`, and *device type* for `segment`.

---

## Derived CSVs

All three files are in `data/derived/`.

### full_dvfs_lambda_cell_summary.csv

One row per (device, model, dvfs_mode, lambda_frac) cell. This is the primary measurement table from which the paper figures are generated.

| Column | Type | Description |
|---|---|---|
| device | string | Device codename, matches `devices.csv` |
| model | string | Model identifier |
| dvfs_mode | integer | Power-mode index |
| dvfs_mode_label | string | Human-readable power-mode label (e.g. "fixed", "max-n") |
| condition_tag | string | Internal condition tag from the run family |
| segment | string | Device type: Server, Jetson, or Small-board |
| capacity_ips | float | Confirmed maximum sustainable throughput (MST) in inferences per second |
| lambda_frac | float | Load fraction of confirmed MST (0.25, 0.50, 0.75, 1.00) |
| target_rps | float | Target request rate derived from MST and lambda_frac |
| n_runs | integer | Number of measurement runs in this cell |
| achieved_rps_median | float | Median achieved throughput across runs (inferences per second) |
| achieved_rps_min | float | Minimum achieved throughput across runs |
| achieved_rps_max | float | Maximum achieved throughput across runs |
| p95_latency_ms_median | float | Median 95th-percentile latency across runs (ms) |
| p95_latency_ms_max | float | Maximum 95th-percentile latency across runs (ms) |
| warm_idle_w_median | float | Median warm-idle AC input power across runs (W) |
| serving_w_median | float | Median serving AC input power across runs (W) |
| delta_w_median | float | Median incremental AC input power (serving minus idle) across runs (W) |
| marginal_wall_energy_j_per_inf_median | float | Median marginal AC input energy per inference (J) |
| marginal_wall_energy_j_per_inf_min | float | Minimum marginal AC input energy per inference across runs (J) |
| marginal_wall_energy_j_per_inf_max | float | Maximum marginal AC input energy per inference across runs (J) |
| batch_size | integer | Batch size used; canonical value is 1 for all shipped rows |

### scheduler_policy_eval.csv

One row per (device, model, demand level) scenario. Compares three power-mode selection policies: the load-reactive dynamic policy, the max-performance policy, and the capacity-only policy. Column groups are prefixed by `dynamic_`, `max_perf_`, and `capacity_only_` respectively.

| Column | Type | Description |
|---|---|---|
| device | string | Device codename |
| model | string | Model identifier |
| demand_frac_of_group_max | float | Demand as fraction of the group's maximum MST |
| demand_rps | float | Demand in requests per second |
| group_max_capacity_ips | float | Maximum MST across the device group |
| n_modes | integer | Total number of power modes profiled for this (device, model) |
| n_feasible_modes | integer | Modes with sufficient MST to serve the demand |
| n_sla_safe_modes | integer | Feasible modes where estimated p95 latency is within SLA |
| policy_status | string | Evaluation outcome: "ok", "no_feasible_mode", "no_sla_safe_mode", or similar |
| {policy}_dvfs_mode | integer | Selected power-mode index for this policy |
| {policy}_dvfs_mode_label | string | Human-readable label of the selected mode |
| {policy}_semantic_rank | integer | Rank of the selected mode by energy within feasible modes (1 = lowest energy) |
| {policy}_mode_category | string | Category of the selected mode (e.g. "low-power", "balanced", "max-perf") |
| {policy}_load_frac_on_mode | float | Load fraction of the selected mode's MST |
| {policy}_nearest_measured_lambda_frac | float | Nearest measured lambda_frac used for interpolation |
| {policy}_delta_w_est | float | Estimated incremental AC input power for the selected mode (W) |
| {policy}_serving_w_est | float | Estimated total serving AC input power for the selected mode (W) |
| {policy}_p95_latency_ms_est | float | Estimated p95 latency for the selected mode (ms) |
| {policy}_energy_j_per_inf_est | float | Estimated marginal AC input energy per inference for the selected mode (J) |
| {policy}_sla_safe | boolean | Whether the selected mode satisfies the p95 latency SLA |
| {policy}_interp_status | string | Interpolation method used: "exact", "interpolated", or "extrapolated" |
| dynamic_delta_w_saving_vs_max_perf_pct | float | Incremental power saving of dynamic over max-perf policy (%) |
| dynamic_energy_saving_vs_max_perf_pct | float | Energy-per-inference saving of dynamic over max-perf policy (%) |
| dynamic_delta_w_saving_vs_capacity_only_pct | float | Incremental power saving of dynamic over capacity-only policy (%) |
| decision_second_best_dvfs_mode | integer | Second-best power mode by energy for the dynamic decision |
| decision_second_best_dvfs_mode_label | string | Label of the second-best mode |
| decision_delta_w_margin | float | Absolute delta_w gap between best and second-best mode (W) |
| decision_delta_w_margin_pct | float | Relative delta_w gap (%) |
| decision_selected_delta_w_iqr_pct | float | IQR of delta_w for the selected mode as a fraction of its median (%) |
| decision_second_best_delta_w_iqr_pct | float | IQR of delta_w for the second-best mode as a fraction of its median (%) |
| decision_robustness | string | Qualitative robustness label for the dynamic selection |
| full_sweep_policy_relevant | boolean | True if this row is included in the full-sweep policy analysis |

### decision_flip.csv

One row per (device, model, policy) pair. Records whether the top-1 energy-optimal power mode differs between isolated profiling and serving-load measurement.

| Column | Type | Description |
|---|---|---|
| device | string | Device codename |
| model | string | Model identifier |
| policy | string | Power-mode policy / configuration label |
| iso_e_j_per_inf | float | Isolated-profile energy per inference (J) |
| srv_e_j_per_inf | float | Serving-load measured energy per inference (J) |
| iso_rank | integer | Energy rank under isolated condition (1 = lowest energy) |
| srv_rank | integer | Energy rank under serving condition (1 = lowest energy) |
| rank_delta | integer | Change in rank: srv_rank minus iso_rank |
| top1_different | boolean | True if the minimum-energy configuration differs between conditions |
| profile_reliable | boolean | True if the isolated profile is not flagged as unreliable |

---

## Raw JSON schema (data/stage2/raw/)

Each file records one measurement run. Files produced by the lambda sweep follow the naming pattern `{timestamp}_{run_id}_{device}_{model}_{dvfs_mode_label}_{ep}_l{load_frac}_r{repeat}_lsweep.json` (for example, `20260508T144012+0900_718_lattepanda_mob050_dvfs0_cpu_l025_r0_lsweep.json`). MST-confirmation files (null load fraction and repeat index) follow a similar pattern with `_v2.json`.

The fields below are the primary fields consumed by the reduction pipeline.

| Field | Type | Description |
|---|---|---|
| device | string | Device codename |
| model | string | Model identifier |
| dvfs_mode | integer | Power-mode index |
| dvfs_mode_label | string | Human-readable power-mode label |
| lambda_frac | float or null | Load fraction of confirmed MST; null in MST-confirmation files |
| run_idx | integer or null | Repeat index within this (device, model, dvfs_mode, lambda_frac) cell |
| batch_size | integer or null | Batch size; null encodes the canonical value of 1 in all shipped records |
| power_trace.status | string | "ok" if the power measurement completed without error; otherwise an error description |
| power_trace.summary.warm_idle_median_watts | float | Median AC input power during the warm idle window before load (W) |
| power_trace.summary.serving_median_watts | float | Median AC input power during the active serving window (W) |
| power_trace.summary.delta_median_watts | float | Median incremental AC input power: serving minus idle (W) |

---

## Isolated profile database (data/stage1/eep_profiler.db)

The SQLite database has one table, `profiles`, with one row per (device, model, batch_size, dvfs_mode). The columns used by the pipeline are:

| Column | Type | Description |
|---|---|---|
| device | string | Device codename (PK component) |
| model | string | Model identifier (PK component) |
| batch_size | integer | Batch size (PK component); pipeline filters to batch_size = 1 |
| dvfs_mode | integer | Power-mode index (PK component) |
| ep | string | Execution provider (e.g. "CPUExecutionProvider", "TensorrtExecutionProvider") |
| latency_ms_p95 | float | 95th-percentile single-inference latency (ms) |
| throughput_ips | float | Single-stream throughput in inferences per second |
| p_wall_avg_w | float | Mean AC input power during active inference (W) |
| watts_idle | float | Idle power estimate from the measurement run (W) |
| watts_idle_true | float | Idle power estimate with thermal drift correction (W); preferred over watts_idle when available |
| energy_inc_per_inf_j | float | Incremental energy per inference, older computation path (J) |
| energy_inc_canonical_per_inf_j | float | Canonical incremental energy per inference, preferred field (J) |
| incremental_source | string | How incremental energy was derived: "plateau", "window", or "none" |
| flag_incremental_unreliable | integer | 1 if the incremental energy measurement is flagged as unreliable; 0 otherwise |

The pipeline selects the best row per (device, model) by preferring rows with `flag_incremental_unreliable = 0` and then the lowest canonical incremental energy.

---

## Pipeline DAG

The two sub-pipelines run independently and produce complementary derived tables. The arrows below show the logical input and output of each step. The shipped reduction scripts use an internal `analysis/results/` working layout for their own input and output, not the `data/` paths shown here, so running them from raw requires adapting those paths. See `scripts/reproduce_from_raw.sh` and the Tier B section of the README. The guaranteed reproduction path (Tier A) does not run these scripts; it reads the frozen CSVs already in `data/derived/`.

### Serving-load sub-pipeline

```
data/stage2/raw/
  -> build_full_dvfs_capacity_artifact.py
  -> data/derived/full_dvfs_capacity.csv

data/stage2/raw/ + full_dvfs_capacity.csv
  -> build_full_dvfs_lambda_artifact.py
  -> data/derived/full_dvfs_lambda_sweep.csv
  -> data/derived/full_dvfs_lambda_cell_summary.csv

full_dvfs_lambda_cell_summary.csv
  -> analyze_full_dvfs.py
  -> data/derived/full_dvfs_analysis_cells.csv

full_dvfs_analysis_cells.csv
  -> analyze_scheduling_validity.py
  -> data/derived/scheduler_policy_eval.csv
```

### Isolated-vs-serving sub-pipeline

```
data/stage1/eep_profiler.db (isolated profiles)
  +
data/derived/serving_power_measured.csv (frozen input; produced by build_serving_power_csv.py
                                          from the serving-at-capacity campaign; raw records
                                          not redistributed)
  -> surrogate_contrast.py
  -> data/derived/surrogate_contrast.csv

surrogate_contrast.csv
  +
data/derived/t2_capacity_power.csv (frozen input; produced by build_t2_artifact.py
                                     from the serving-at-capacity campaign; raw records
                                     not redistributed)
  -> decision_flip.py
  -> data/derived/decision_flip.csv
```

Both `serving_power_measured.csv` and `t2_capacity_power.csv` are shipped as frozen inputs from the serving-at-capacity (policy-keyed) campaign; their raw run records are not redistributed in this repository. See `docs/PROVENANCE.md` for details.
