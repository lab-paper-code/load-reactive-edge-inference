#!/usr/bin/env python3
"""Export the plot-point data used by Figure 1.

Figure 1 is drawn from the batch-size-1 serving-load measurement grid. Each
output row is one plotted point, i.e., one
(device, model, DVFS mode, load fraction) cell.

Inputs
------
data/derived/full_dvfs_lambda_cell_summary.csv
    Median p95 latency and median AC-input energy per inference for the 396
    measured cells.

Outputs
-------
data/derived/fig1_fleet_cloud_data.csv
    A self-contained Figure 1 plot-point table.

Y-axis formula
--------------
The Figure 1 y value is deadline-normalized EDP efficiency:

    y = 1 / (AC input energy per inference * (p95 latency / 100 ms))
      = 100 / (energy_j_per_inf * p95_latency_ms)

Higher is better. Points with p95 latency above 100 ms are still exported, but
the figure renders them as open markers to indicate deadline violation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "derived" / "full_dvfs_lambda_cell_summary.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "derived" / "fig1_fleet_cloud_data.csv"
DEADLINE_MS = 100.0


SEG_OF = {
    "gpu-server": "Server",
    "orin": "Jetson",
    "orin-nano": "Jetson",
    "xavier": "Jetson",
    "jetson": "Jetson",
    "orangepi-npu": "SB-NPU",
    "orangepi": "SB-CPU",
    "rasp5": "SB-CPU",
    "lattepanda": "SB-CPU",
}
SEG_LABEL = {
    "Server": "Server",
    "Jetson": "Jetson",
    "SB-NPU": "Small-board NPU",
    "SB-CPU": "Small-board CPU",
}
SEG_HEX = {
    "Server": "D55E00",
    "Jetson": "0072B2",
    "SB-NPU": "009E73",
    "SB-CPU": "CC79A7",
}
MODEL_LABEL = {
    "mobilenet-v2-050": "MobileNet-V2 0.5",
    "mobilenet-v2-100": "MobileNet-V2 1.0",
    "efficientnet-b4": "EfficientNet-B4",
}


FIELDS = [
    "model",
    "model_label",
    "segment",
    "segment_label",
    "segment_hex",
    "device",
    "dvfs_mode",
    "dvfs_mode_label",
    "condition_tag",
    "capacity_ips",
    "lambda_frac",
    "load_group_pct",
    "target_rps",
    "n_runs",
    "p95_latency_ms",
    "ac_input_energy_j_per_inf",
    "p95_deadline_feasible",
    "deadline_normalized_ac_input_energy_delay_j_per_inf",
    "deadline_normalized_edp_efficiency_inf_per_j",
    "ac_input_energy_efficiency_inf_per_j",
]


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def export_rows(source: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with source.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if _int(row, "batch_size") != 1:
                continue

            device = row["device"]
            segment = SEG_OF[device]
            lambda_frac = round(_float(row, "lambda_frac"), 2)
            energy = _float(row, "marginal_wall_energy_j_per_inf_median")
            p95 = _float(row, "p95_latency_ms_median")

            rows.append(
                {
                    "model": row["model"],
                    "model_label": MODEL_LABEL.get(row["model"], row["model"]),
                    "segment": segment,
                    "segment_label": SEG_LABEL[segment],
                    "segment_hex": SEG_HEX[segment],
                    "device": device,
                    "dvfs_mode": _int(row, "dvfs_mode"),
                    "dvfs_mode_label": row["dvfs_mode_label"],
                    "condition_tag": row["condition_tag"],
                    "capacity_ips": _float(row, "capacity_ips"),
                    "lambda_frac": lambda_frac,
                    "load_group_pct": int(round(lambda_frac * 100)),
                    "target_rps": _float(row, "target_rps"),
                    "n_runs": _int(row, "n_runs"),
                    "p95_latency_ms": p95,
                    "ac_input_energy_j_per_inf": energy,
                    "p95_deadline_feasible": int(p95 <= DEADLINE_MS),
                    "deadline_normalized_ac_input_energy_delay_j_per_inf": (
                        energy * p95 / DEADLINE_MS
                    ),
                    "deadline_normalized_edp_efficiency_inf_per_j": (
                        DEADLINE_MS / (energy * p95)
                    ),
                    "ac_input_energy_efficiency_inf_per_j": 1.0 / energy,
                }
            )

    model_order = {
        "mobilenet-v2-050": 0,
        "mobilenet-v2-100": 1,
        "efficientnet-b4": 2,
    }
    segment_order = {"Server": 0, "Jetson": 1, "SB-NPU": 2, "SB-CPU": 3}
    rows.sort(
        key=lambda row: (
            model_order.get(str(row["model"]), 9),
            segment_order[str(row["segment"])],
            float(row["lambda_frac"]),
            str(row["device"]),
            int(row["dvfs_mode"]),
        )
    )
    return rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"source cell-summary CSV (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output Figure 1 CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = export_rows(args.source)
    if len(rows) != 396:
        raise SystemExit(f"expected 396 Figure 1 cells, got {len(rows)}")

    write_csv(rows, args.output)

    p95s = [float(row["p95_latency_ms"]) for row in rows]
    energies = [float(row["ac_input_energy_j_per_inf"]) for row in rows]
    scores = [
        float(row["deadline_normalized_edp_efficiency_inf_per_j"])
        for row in rows
    ]
    n_feasible = sum(int(row["p95_deadline_feasible"]) for row in rows)
    print(f"wrote {args.output}: {len(rows)} rows")
    print(f"  p95 latency: {min(p95s):.2f}-{max(p95s):.2f} ms")
    print(f"  AC input energy: {min(energies):.6f}-{max(energies):.6f} J/inf")
    print(f"  Fig.1 y value: {min(scores):.6f}-{max(scores):.6f} inf/J")
    print(f"  p95 <= {DEADLINE_MS:.0f} ms: {n_feasible}/{len(rows)} rows")


if __name__ == "__main__":
    main()
