# edge-inference-energy-grid

Measurement grid and analysis pipeline for the paper "Load Moves the Energy-Optimal Configuration of Real-Time Edge Inference" (2026).

The dataset covers nine heterogeneous edge devices running image-classification inference under Poisson serving load across four load levels and multiple DVFS configurations. The analysis identifies how the energy-optimal DVFS configuration shifts with load and quantifies the energy saving a load-reactive policy achieves over static baselines.

## What this repository contains

- **data/stage1/**: isolated first-stage profiles: a pruned SQLite database (`eep_profiler.db`) recording single-stream energy, latency, and throughput per (device, model, DVFS mode) at batch size 1.
- **data/stage2/raw/**: serving-load measurement records: 3960 load-sweep files (one per (device, model, DVFS mode, load fraction, repeat index)) plus 114 capacity-confirmation files (null load fraction and repeat index) from the May 2026 campaign.
- **data/derived/**: frozen derived CSVs produced by the reduction pipeline; these are the direct inputs to the paper figures and tables.
- **analysis/**: reduction scripts that build the derived CSVs from raw data.
- **paper/**: figure and table generation scripts that read from `data/derived/`.
- **scripts/**: entry-point shell scripts for the two reproducibility tiers.
- **measurement/**: serving and isolated measurement harness code (reference; not needed for reproduction). The harness is included to document how the measurements were taken. It references internal client and utility modules (for example `power_reader`, `power_trace`, `ssh`, `runtime_budget`) that are tied to our hardware and are not redistributed, so these files do not run as shipped.
- **models/**: model export utilities (ONNX, RKNN).
- **devices.csv**: hardware registry: codename, paper label, segment, accelerator, execution provider.
- **docs/**: data schema and provenance documentation.

## Scope

The released data and scripts reproduce every figure and number in the paper. Re-measurement requires the hardware listed in `devices.csv`.

The raw serving-load records in `data/stage2/raw/` are the canonical batch-size-1 subset. The raw runs for the serving-at-capacity campaign that produced `data/derived/serving_power_measured.csv` are not redistributed; that CSV is shipped as a frozen input.

## Reproducibility tiers

### Tier A: figures and tables from frozen derived CSVs (guaranteed)

This path uses only the Python standard library and requires no additional installation.

```bash
bash scripts/reproduce_figures.sh
```

The script calls `paper/gen_full_tables.py` and the figure emission scripts, all of which read from `data/derived/`. The derived CSVs are bit-for-bit identical to what the paper was computed from.

### Tier B: full raw-to-derived pipeline (transparency)

This path runs the complete upstream pipeline from raw measurement data to derived CSVs. It requires numpy, matplotlib, and PyYAML (see `requirements.txt`). The reduction scripts were written against an internal directory layout; their path configuration (constants in `analysis/constants.py`) may need adjusting to match this repo's `data/` layout before they execute cleanly.

```bash
# Optional: redirect output to avoid overwriting the shipped frozen CSVs.
export RELEASE_DERIVED_OUT="$PWD/data/derived_regen"
bash scripts/reproduce_from_raw.sh
```

After running, compare `data/derived_regen/` against `data/derived/` numerically to confirm agreement.

Do not use this path as the standard reproducibility check. Use Tier A instead.

## Environment

Python 3.9 or newer.

Tier A needs no third-party packages (standard library only):

```bash
bash scripts/reproduce_figures.sh
```

Tier B needs `pip install numpy matplotlib PyYAML`.

The `measurement/` harness is reference only; even importing it requires aiohttp, onnxruntime, PyYAML, and pandas, and it additionally requires the hardware fleet in `devices.csv` to run.

The model export script needs `pip install torch timm onnx`.

## Quickstart

```bash
git clone <repo-url>
cd edge-inference-energy-grid
bash scripts/reproduce_figures.sh
```

Generated files land in `paper/figures/` and alongside `paper/oracle_full_tables.tex`.

## Repository layout

```
edge-inference-energy-grid/
  analysis/                  reduction scripts (raw -> derived CSVs)
  data/
    derived/                 frozen derived CSVs (Tier A inputs)
    stage1/
      eep_profiler.db        isolated profile database
    stage2/
      raw/                   serving-load run records (JSON, one per cell/repeat)
  devices.csv                hardware registry
  docs/
    DATA_SCHEMA.md           column dictionary and pipeline DAG
    PROVENANCE.md            measurement provenance and canonical-data discipline
  measurement/               measurement harness (reference)
    stage1-isolated/
    stage2-serving/
  models/                    model export utilities
  paper/
    figures/                 figure generation scripts and output
    gen_full_tables.py       table generation script
  scripts/
    reproduce_figures.sh     Tier A entry point
    reproduce_from_raw.sh    Tier B entry point
  CITATION.cff
  LICENSE                    MIT (code)
  LICENSE-data               CC BY 4.0 (data)
  requirements.txt
```

## License

Source code is released under the MIT License (`LICENSE`).

Data files under `data/` are released under Creative Commons Attribution 4.0 International (`LICENSE-data`).

When using this dataset or code, cite the accompanying paper as described in `CITATION.cff`.

## Documentation

- Column dictionary and pipeline DAG: [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)
- Measurement provenance: [docs/PROVENANCE.md](docs/PROVENANCE.md)
