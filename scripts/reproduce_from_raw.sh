#!/usr/bin/env bash
# reproduce_from_raw.sh: full raw -> derived -> figures pipeline (documentation).
#
# IMPORTANT: The GUARANTEED-REPRODUCIBLE path is reproduce_figures.sh, which
# regenerates paper outputs from the shipped FROZEN DERIVED CSVs in data/derived/.
# Those CSVs are bit-for-bit identical to what the paper tables and figures were
# computed from.
#
# This script shows the full upstream pipeline that produced those CSVs from the
# raw measurement data in data/stage1/ and data/stage2/.  The reduction scripts
# were written against an internal results/ directory layout and may require
# adapting their input/output path configuration (constants.py DERIVED_DIR,
# RESULTS_DIR, etc.) to this repo's data/ layout before they will execute
# cleanly here.
#
# If you run this script, first redirect the derived output to a scratch
# directory (e.g. data/derived_regen/) so the shipped frozen data/derived/ is
# not overwritten:
#
#   export RELEASE_DERIVED_OUT="$PWD/data/derived_regen"
#
# Then diff derived_regen against derived numerically to confirm bitwise or
# float-tolerance agreement.
#
# DO NOT run this script as part of the standard reproducibility check.
# Use reproduce_figures.sh instead.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/analysis:$PWD/measurement/stage1-isolated/src:$PWD/measurement/stage2-serving"

# 1. serving grid: raw -> capacity -> lambda cell summary
python3 analysis/build_full_dvfs_capacity_artifact.py --apply
python3 analysis/build_full_dvfs_lambda_artifact.py --apply
python3 analysis/analyze_full_dvfs.py --apply
python3 analysis/analyze_scheduling_validity.py --apply

# 2. isolated vs serving flip: DB + frozen serving CSV -> surrogate -> decision flip
python3 analysis/surrogate_contrast.py --apply
python3 analysis/decision_flip.py --apply

echo "full pipeline regenerated; compare data/derived against frozen reference"
