"""
Build EEP tensor / coverage / integrity artifacts.

Uses the DB (eep_profiler.db) as the source.
Phase 3 responsibilities:
- Apply measurement-quality annotations.
- Export coverage disclosure.
- Generate export integrity report.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import DB_PATH, MEASURED_RESULTS_DIR
from db import export_tensor_json
from measurement_signals import (write_coverage_report,
                                 write_export_integrity_report)


def print_summary(tensor: dict):
    print(f'\n{"device":>12s} | {"model":>20s} | {"dvfs":>5s} | {"bs":>4s} | '
          f'{"E(J/inf)":>10s} | {"L_p95(ms)":>10s} | {"T(ips)":>10s} | {"P_avg":>10s}')
    print('-' * 100)
    for dev in sorted(tensor):
        for model in sorted(tensor[dev]):
            for dvfs in sorted(tensor[dev][model]):
                for bs in sorted(tensor[dev][model][dvfs], key=int):
                    e = tensor[dev][model][dvfs][bs]
                    print(f'{dev:>12s} | {model:>20s} | {dvfs:>5s} | {bs:>4s} | '
                          f'{e["E_mean"]:>10.6f} | {e["L_p95"]:>10.2f} | '
                          f'{e["T_mean"]:>10.1f} | {e["avg_power_w"]:>10.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        default=os.path.join(MEASURED_RESULTS_DIR, 'eep_tensor.json'))
    args = parser.parse_args()

    n_entries = export_tensor_json(DB_PATH, args.output)
    coverage = write_coverage_report(DB_PATH)
    integrity = write_export_integrity_report(DB_PATH, export_path=args.output)

    with open(args.output) as f:
        tensor = json.load(f)['tensor']

    print(f'Loaded DB: {DB_PATH}')
    print_summary(tensor)
    print(f'\nTotal entries: {n_entries}')
    print(f'Coverage report: {coverage["summary"]["measured_profiles"]} measured / '
          f'{coverage["summary"]["configured_profiles"]} configured')
    print(f'Integrity status: {integrity["status"]}')
    print(f'Saved tensor to {args.output}')


if __name__ == '__main__':
    main()
