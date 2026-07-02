#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 paper/gen_full_tables.py
python3 paper/figures/emit_fig1_data.py
python3 paper/figures/emit_extra_figures.py
python3 paper/figures/emit_reversal_pgfplots.py
echo "figures/tables regenerated from frozen derived CSVs"
