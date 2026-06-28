# Data Provenance

---

## Measurement stages

The dataset was produced in two separate measurement campaigns.

### Stage 1: isolated profiling (March and April 2026)

Isolated profiles were collected before the serving-load measurements. For each device, model, and DVFS configuration at batch size 1, the model was run in isolation as a sustained single-stream loop with no serving load and no request queue. Each run recorded energy, accuracy, latency, and throughput. Energy was measured at the AC input with a Shelly Plug S Gen3 smart meter (BL0942 metering IC), reported as wall energy per inference with idle power subtracted using an explicit idle-window median. The boundary convention for the idle window is the same one used in the stage 2 serving runs, so the two measurements differ only in operating condition (isolation versus Poisson serving load). This shared boundary is what makes the rank-flip result in the decision-flip figure an operating-condition effect rather than a measurement artifact.

The isolated profiles are stored in `data/stage1/eep_profiler.db`. This database has been pruned to include only the devices and models used in the paper.

### Stage 2: serving-load measurement (May 2026)

Serving-load runs were collected under Poisson request arrivals at four load fractions of the confirmed serving capacity (0.25, 0.50, 0.75, 1.00). Each (device, model, DVFS mode, load fraction) cell was repeated multiple times to estimate measurement variability. Wall power was recorded at the AC input with the same Shelly meter. Raw run records are in `data/stage2/raw/` as one JSON file per run.

---

## Frozen inputs

`data/derived/serving_power_measured.csv` and `data/derived/t2_capacity_power.csv` are both frozen inputs derived from the serving-at-capacity (policy-keyed) sub-campaign that measured power at a single load point (lambda = 1.0) across all (device, model, DVFS mode) configurations. `serving_power_measured.csv` was produced by `build_serving_power_csv.py` and `t2_capacity_power.csv` was produced by `build_t2_artifact.py`. The raw run records from that sub-campaign are not redistributed in this repository. Both CSVs are included as frozen artifacts so the isolated-vs-serving contrast pipeline (`surrogate_contrast.py`, `decision_flip.py`) can run without those raw records.

---

## Batch-size discipline

Only batch-size-1 runs are included in the shipped data. The derived grid in `data/derived/` is the authoritative batch-size-1 snapshot. The scheduling analysis scripts (`analyze_scheduling_validity.py`) are fed exclusively batch-size-1 data. This discipline is enforced at the raw-record level (the `canonical_bs1` subset selection in the build pipeline) and reflected in the `batch_size` column of `full_dvfs_lambda_cell_summary.csv`, which is 1 for all rows.
