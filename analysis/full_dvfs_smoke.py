"""Full-DVFS smoke-test helpers.

This module keeps the full-DVFS preflight path separate from the policy-keyed
admission/T2/T5 pipeline. It resolves one `(device, model,
dvfs_mode)` cell and validates the raw v4/v5 JSON shape produced by a smoke run.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from measurement_jobs import (MeasurementJob, condition_label, condition_tag,
                              iter_full_dvfs_jobs)
from measurement_runner import (FULL_DVFS_RAW_SCHEMA_MIN,
                                FULL_DVFS_RAW_SCHEMA_VERSION)

_CELL_RE = re.compile(r"^([^/]+)/([^/]+)/dvfs(\d+)$")
_GPU_FREQ_TOLERANCE_MHZ = 75


@dataclass
class FullDvfsCell:
    device: str
    model: str
    dvfs_mode: int


@dataclass
class RawValidationReport:
    path: Path
    ok: bool
    errors: list[str]
    warnings: list[str]


def parse_full_dvfs_cell(raw: str) -> FullDvfsCell:
    """Parse `device/model/dvfsN`."""
    match = _CELL_RE.match(raw)
    if not match:
        raise ValueError("cell must be formatted as device/model/dvfsN")
    device, model, mode = match.groups()
    return FullDvfsCell(device=device, model=model, dvfs_mode=int(mode))


def resolve_full_dvfs_job(raw_cell: str, batch_size: int = 1) -> MeasurementJob:
    """Return the planned full-DVFS job for one cell, optionally at batch_size N.

    batch_size != 1 enumerates only the requested width and is subject to the
    enumerator's server-only guard (raises ValueError for non-server devices).
    """
    cell = parse_full_dvfs_cell(raw_cell)
    bs_map = None if batch_size == 1 else {cell.device: (batch_size,)}
    try:
        jobs = iter_full_dvfs_jobs(devices={cell.device}, models={cell.model},
                                   batch_sizes=bs_map)
    except KeyError:
        raise ValueError(f"unknown full-DVFS device {cell.device!r}") from None
    for job in jobs:
        if job.dvfs_mode == cell.dvfs_mode and job.batch_size == batch_size:
            return job

    available = ", ".join(job_tag for job_tag in _job_tags(jobs)) or "none"
    raise ValueError(f"unknown full-DVFS cell {raw_cell!r} at bs={batch_size}; "
                     f"available: {available}")


def _job_tags(jobs: list[MeasurementJob]) -> list[str]:
    return [f"{j.device}/{j.model}/dvfs{j.dvfs_mode}" for j in jobs]


def _get_path(data: dict, dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _require(errors: list[str], data: dict, dotted: str, expected: Any = None) -> Any:
    actual = _get_path(data, dotted)
    if actual is None:
        errors.append(f"missing {dotted}")
    elif expected is not None and actual != expected:
        errors.append(f"{dotted}={actual!r}, expected {expected!r}")
    return actual


def _check_gpu_freq(report: RawValidationReport, status: dict, expected_mhz: int,
                    label: str) -> None:
    actual = status.get("gpu_clock_mhz")
    if actual is None and status.get("gpu_cur_freq_hz"):
        actual = float(status["gpu_cur_freq_hz"]) / 1_000_000
    if actual is None:
        report.errors.append(f"{label}.gpu frequency missing; expected ~{expected_mhz}MHz")
        return
    if abs(float(actual) - expected_mhz) > _GPU_FREQ_TOLERANCE_MHZ:
        report.errors.append(
            f"{label}.gpu frequency {actual}MHz, expected ~{expected_mhz}MHz"
        )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_full_dvfs_raw(path: Path, expected_job: MeasurementJob | None = None,
                           require_power: bool = True) -> RawValidationReport:
    """Validate that a raw JSON is a mode-keyed full-DVFS smoke artifact."""
    errors: list[str] = []
    warnings: list[str] = []
    report = RawValidationReport(path=path, ok=False, errors=errors, warnings=warnings)

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read JSON: {exc}")
        return report

    schema = _require(errors, data, "schema_version")
    batch_size = data.get("batch_size")
    try:
        schema_int = int(schema)
        if schema_int < FULL_DVFS_RAW_SCHEMA_MIN:
            errors.append(f"schema_version={schema!r}, expected >= {FULL_DVFS_RAW_SCHEMA_MIN}")
        elif schema_int >= FULL_DVFS_RAW_SCHEMA_VERSION and not (
                isinstance(batch_size, int) and batch_size >= 1):
            # v5 (batch-aware) raws must carry an explicit batch_size so the
            # per-image divisor is never silently defaulted to N=1.
            errors.append(f"schema_version={schema_int} (batch-aware) requires an "
                          f"integer batch_size>=1, got {batch_size!r}")
    except (TypeError, ValueError):
        errors.append(f"schema_version={schema!r} is not an integer")

    _require(errors, data, "run_family", "full_dvfs")
    _require(errors, data, "policy", "full_dvfs")
    _require(errors, data, "operating_condition.kind", "dvfs_mode")
    if data.get("run_type") == "lambda_sweep" or path.name.endswith("_lsweep.json"):
        errors.append("validator expects a capacity smoke raw JSON, not lambda_sweep raw")

    mode = _require(errors, data, "dvfs_mode")
    tag = _require(errors, data, "condition_tag")
    # Accept the legacy bs=1 tag `dvfs{mode}` and the batch-extended
    # `dvfs{mode}_bs{N}`; the `_bs{N}` suffix appears only when batch_size != 1
    # (v4/legacy raws carry no batch_size ⇒ treated as bs=1).
    bs_for_tag = batch_size if isinstance(batch_size, int) else 1
    if not isinstance(mode, int):
        errors.append(f"dvfs_mode={mode!r} is not an integer")
    else:
        expected_tag = f"dvfs{mode}" if bs_for_tag == 1 else f"dvfs{mode}_bs{bs_for_tag}"
        if tag != expected_tag:
            errors.append(f"condition_tag={tag!r}, expected {expected_tag!r}")
    if isinstance(tag, str) and tag not in path.name:
        errors.append(f"filename {path.name!r} does not contain condition_tag {tag!r}")

    device = _require(errors, data, "device")
    model = _require(errors, data, "model")
    if not isinstance(device, str) or not device:
        errors.append("device must be a non-empty string")
    if not isinstance(model, str) or not model:
        errors.append("model must be a non-empty string")
    label = _require(errors, data, "dvfs_mode_label")
    if not isinstance(label, str) or not label:
        errors.append("dvfs_mode_label must be a non-empty string")
    if isinstance(mode, int):
        _require(errors, data, "operating_condition.dvfs_mode", mode)
    if isinstance(tag, str):
        _require(errors, data, "operating_condition.condition_tag", tag)
    if isinstance(label, str) and label:
        _require(errors, data, "operating_condition.dvfs_mode_label", label)

    capacity = _require(errors, data, "capacity_ips")
    if not _is_finite_number(capacity):
        errors.append(f"capacity_ips={capacity!r} is not finite numeric")
    capacity_range = _require(errors, data, "capacity_range")
    if not (isinstance(capacity_range, list) and len(capacity_range) == 2
            and all(_is_finite_number(v) for v in capacity_range)):
        errors.append(f"capacity_range={capacity_range!r} is not a two-element list")
    elif float(capacity_range[0]) > float(capacity_range[1]):
        errors.append(f"capacity_range={capacity_range!r} has low > high")
    if not isinstance(data.get("capacity_confirmed"), bool):
        errors.append("capacity_confirmed must be a boolean")
    elif not data["capacity_confirmed"]:
        errors.append("capacity_confirmed is false")
    if data.get("l_max_ms") != 100.0:
        errors.append(f"l_max_ms={data.get('l_max_ms')!r}, expected 100.0")
    if not _is_finite_number(data.get("mu_hint")) or float(data.get("mu_hint")) <= 0:
        errors.append(f"mu_hint={data.get('mu_hint')!r} must be positive")
    if not isinstance(data.get("history"), list) or not data["history"]:
        errors.append("history must be a non-empty list")
    confirmation = data.get("confirmation") or {}
    rounds = confirmation.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        errors.append("confirmation.rounds must be a non-empty list")
    else:
        passing = [r for r in rounds if isinstance(r, dict) and r.get("pass")]
        if not passing:
            errors.append("confirmation.rounds has no passing round")
        else:
            runs = passing[-1].get("runs")
            if not isinstance(runs, list) or not runs:
                errors.append("passing confirmation round has no runs")
            else:
                for idx, run in enumerate(runs):
                    if not isinstance(run, dict):
                        errors.append(f"confirmation run {idx} is not an object")
                        continue
                    if not _is_finite_number(run.get("active_start_ts")):
                        errors.append(f"confirmation run {idx} missing active_start_ts")
                    if not _is_finite_number(run.get("active_end_ts")):
                        errors.append(f"confirmation run {idx} missing active_end_ts")
                    if (_is_finite_number(run.get("active_start_ts"))
                            and _is_finite_number(run.get("active_end_ts"))
                            and float(run["active_end_ts"]) <= float(run["active_start_ts"])):
                        errors.append(f"confirmation run {idx} active_end_ts <= active_start_ts")

    for key in ("device_status_before", "device_status_after"):
        if not isinstance(data.get(key), dict):
            errors.append(f"{key} must be present as an object")

    provenance = data.get("deploy_provenance")
    if not isinstance(provenance, dict):
        errors.append("deploy_provenance must be present as an object")
    else:
        for key in ("infer_server_sha256", "model_filename", "model_sha256"):
            if not provenance.get(key):
                errors.append(f"deploy_provenance.{key} missing")

    power_trace = data.get("power_trace")
    if not isinstance(power_trace, dict):
        msg = "power_trace missing"
        if require_power:
            errors.append(msg)
        else:
            warnings.append(msg)
    elif power_trace.get("status") != "ok":
        msg = f"power_trace.status={power_trace.get('status')!r}, expected 'ok'"
        if require_power:
            errors.append(msg)
        else:
            warnings.append(msg)
    else:
        warm_idle = power_trace.get("warm_idle") or {}
        sample_count = warm_idle.get("sample_count")
        if not isinstance(sample_count, int) or sample_count <= 0:
            errors.append(f"power_trace.warm_idle.sample_count={sample_count!r}")
        summary = power_trace.get("summary") or {}
        n_valid = summary.get("n_valid_runs")
        if not isinstance(n_valid, int) or n_valid < 2:
            errors.append(f"power_trace.summary.n_valid_runs={n_valid!r}, expected >= 2")

    if expected_job is not None:
        _require(errors, data, "device", expected_job.device)
        _require(errors, data, "model", expected_job.model)
        _require(errors, data, "dvfs_mode", expected_job.dvfs_mode)
        _require(errors, data, "dvfs_mode_label", condition_label(expected_job))
        _require(errors, data, "condition_tag", condition_tag(expected_job))
        _require(errors, data, "operating_condition.dvfs_mode", expected_job.dvfs_mode)
        _require(errors, data, "operating_condition.dvfs_mode_label",
                 condition_label(expected_job))
        _require(errors, data, "operating_condition.expected_ep", expected_job.expected_ep)
        _require(errors, data, "deploy_provenance.model_filename", expected_job.model_file)
        if expected_job.batch_size != 1:        # bs=1 (v4) raws carry no batch_size
            _require(errors, data, "batch_size", expected_job.batch_size)

        for status_key in ("device_status_before", "device_status_after"):
            status = data.get(status_key) or {}
            ep = status.get("ep")
            if ep != expected_job.expected_ep:
                errors.append(f"{status_key}.ep={ep!r}, expected {expected_job.expected_ep!r}")
            if expected_job.expected_nvpmodel:
                nvpmodel = status.get("nvpmodel", "")
                if expected_job.expected_nvpmodel not in nvpmodel:
                    errors.append(
                        f"{status_key}.nvpmodel={nvpmodel!r}, "
                        f"expected to contain {expected_job.expected_nvpmodel!r}"
                    )
            if expected_job.expected_gpu_freq_mhz:
                _check_gpu_freq(report, status, expected_job.expected_gpu_freq_mhz,
                                status_key)

    report.ok = not errors
    return report


def print_validation_report(report: RawValidationReport) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Full-DVFS raw validation: {status}")
    print(f"  path: {report.path}")
    if report.errors:
        print("  errors:")
        for err in report.errors:
            print(f"    - {err}")
    if report.warnings:
        print("  warnings:")
        for warn in report.warnings:
            print(f"    - {warn}")
