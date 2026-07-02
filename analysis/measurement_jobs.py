"""MeasurementJob spec, JOBS_ALL grid, path + tag helpers.

Single source of truth for the per-pair measurement plan and the raw-JSON
output file naming convention. Consumers: remeasure, lambda_sweep,
admit_new_runs, build_*_artifact.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from constants import (DB_PATH, DEVICES, EEP_ROOT, MODELS, MODELS_DIR, RKNN_MODELS_DIR,
                       POLICY_EINC, POLICY_CAPACITY, SERVER_BATCH_SIZES)

# --- Paths ---
_S1_DIR = Path(__file__).parent
RESULTS_RAW = _S1_DIR / "results" / "raw"
INFER_SERVER = _S1_DIR / "infer_server.py"
LOAD_GENERATOR = _S1_DIR / "load_generator.py"


@dataclass
class MeasurementJob:
    device: str
    model: str
    policy: str
    model_file: str
    mu_hint: float
    dvfs_setup: list[str]    # SSH commands to set DVFS before measurement
    expected_ep: str         # Expected EP to verify via /status
    dvfs_mode: int | None = None  # Full-DVFS jobs set this; policy-keyed jobs leave None.
    dvfs_label: str = ""
    needs_reboot: bool = False
    nvpmodel_mode: int = -1      # Jetson nvpmodel mode to set (requires reboot)
    expected_nvpmodel: str = ""
    expected_gpu_freq_mhz: int = 0
    batch_size: int = 1          # Server-side static batch width (N-wide forward); N=1 = legacy.


# --- Measurement plan ---
# Grouped by device; within each device, ordered to minimize DVFS changes.
JOBS_ALL: list[MeasurementJob] = [
    # orin: MODE_30W (mode 2) + jetson_clocks, Policy A pairs
    MeasurementJob("orin", "efficientnet-b4", POLICY_EINC, "efficientnet-b4.onnx",
                   40, [],
                   "TensorrtExecutionProvider", needs_reboot=True,
                   nvpmodel_mode=2, expected_nvpmodel="MODE_30W"),
    MeasurementJob("orin", "mobilenet-v2-050", POLICY_EINC, "mobilenet-v2-050.onnx",
                   250, [],
                   "TensorrtExecutionProvider",
                   expected_nvpmodel="MODE_30W"),
    MeasurementJob("orin", "mobilenet-v2-100", POLICY_EINC, "mobilenet-v2-100.onnx",
                   300, [],
                   "TensorrtExecutionProvider",
                   expected_nvpmodel="MODE_30W"),
    # orin: MAXN (mode 0), Policy B, requires reboot
    MeasurementJob("orin", "efficientnet-b4", POLICY_CAPACITY, "efficientnet-b4.onnx",
                   90, [],
                   "TensorrtExecutionProvider", needs_reboot=True,
                   nvpmodel_mode=0, expected_nvpmodel="MAXN"),

    # gpu-server: mode 4 (1005MHz), Policy A
    MeasurementJob("gpu-server", "efficientnet-b4", POLICY_EINC, "efficientnet-b4.onnx",
                   180, ["sudo nvidia-smi -lgc 1005,1005"],
                   "CUDAExecutionProvider",
                   expected_gpu_freq_mhz=1005),
    MeasurementJob("gpu-server", "mobilenet-v2-050", POLICY_EINC, "mobilenet-v2-050.onnx",
                   360, [],
                   "CUDAExecutionProvider",
                   expected_gpu_freq_mhz=1005),
    MeasurementJob("gpu-server", "mobilenet-v2-100", POLICY_EINC, "mobilenet-v2-100.onnx",
                   900, [],
                   "CUDAExecutionProvider",
                   expected_gpu_freq_mhz=1005),
    # gpu-server: mode 7 (1545MHz), Policy B
    MeasurementJob("gpu-server", "efficientnet-b4", POLICY_CAPACITY, "efficientnet-b4.onnx",
                   280, ["sudo nvidia-smi -lgc 1545,1545"],
                   "CUDAExecutionProvider",
                   expected_gpu_freq_mhz=1545),

    # xavier: mode 5, Policy A
    MeasurementJob("xavier", "efficientnet-b4", POLICY_EINC, "efficientnet-b4.onnx",
                   10, [],
                   "TensorrtExecutionProvider"),
    MeasurementJob("xavier", "mobilenet-v2-050", POLICY_EINC, "mobilenet-v2-050.onnx",
                   400, [],
                   "TensorrtExecutionProvider"),
    MeasurementJob("xavier", "mobilenet-v2-100", POLICY_EINC, "mobilenet-v2-100.onnx",
                   260, [],
                   "TensorrtExecutionProvider"),

    # jetson (Nano): MAXN (default), TensorRT, Policy A
    MeasurementJob("jetson", "mobilenet-v2-050", POLICY_EINC, "mobilenet-v2-050.onnx",
                   180, [],
                   "TensorrtExecutionProvider"),
    MeasurementJob("jetson", "mobilenet-v2-100", POLICY_EINC, "mobilenet-v2-100.onnx",
                   100, [],
                   "TensorrtExecutionProvider"),

    # lattepanda: CPU only, Policy A
    MeasurementJob("lattepanda", "mobilenet-v2-050", POLICY_EINC, "mobilenet-v2-050.onnx",
                   270, [],
                   "CPUExecutionProvider"),
    MeasurementJob("lattepanda", "mobilenet-v2-100", POLICY_EINC, "mobilenet-v2-100.onnx",
                   70, [],
                   "CPUExecutionProvider"),

    # orangepi: CPU only, Policy A
    MeasurementJob("orangepi", "mobilenet-v2-050", POLICY_EINC, "mobilenet-v2-050.onnx",
                   150, [],
                   "CPUExecutionProvider"),
    MeasurementJob("orangepi", "mobilenet-v2-100", POLICY_EINC, "mobilenet-v2-100.onnx",
                   40, [],
                   "CPUExecutionProvider"),

    # rasp5: CPU only, Policy A
    MeasurementJob("rasp5", "mobilenet-v2-050", POLICY_EINC, "mobilenet-v2-050.onnx",
                   95, [],
                   "CPUExecutionProvider"),
    MeasurementJob("rasp5", "mobilenet-v2-100", POLICY_EINC, "mobilenet-v2-100.onnx",
                   40, [],
                   "CPUExecutionProvider"),

    # orangepi-npu: Rockchip RK3588 NPU (rknnlite, bs=1), Policy A
    # Same physical host as orangepi; uses port 8091 to avoid conflict.
    MeasurementJob("orangepi-npu", "mobilenet-v2-050", POLICY_EINC,
                   "mobilenet-v2-050-rk3588-bs1.rknn",
                   250, [],
                   "RKNPUExecutionProvider"),
    MeasurementJob("orangepi-npu", "mobilenet-v2-100", POLICY_EINC,
                   "mobilenet-v2-100-rk3588-bs1.rknn",
                   170, [],
                   "RKNPUExecutionProvider"),
    # eep-profiler p95=108ms, borderline SLO; S1 load-test determines feasibility.
    MeasurementJob("orangepi-npu", "efficientnet-b4", POLICY_EINC,
                   "efficientnet-b4-rk3588-bs1.rknn",
                   8, [],
                   "RKNPUExecutionProvider"),
]


def model_src(job: MeasurementJob) -> Path:
    """Return the local path to the model file this job uses."""
    if job.model_file.endswith(".rknn"):
        return RKNN_MODELS_DIR / job.model_file
    return MODELS_DIR / job.model_file


def expected_model_name(job: MeasurementJob) -> str:
    """Return the model name the inference server is expected to report."""
    stem = Path(job.model_file).stem
    if job.model_file.endswith(".rknn"):
        return re.sub(r"-rk\d+-bs\d+$", "", stem)
    return stem


_EP_SHORT = {"TensorrtExecutionProvider": "trt",
             "CUDAExecutionProvider": "cuda",
             "CPUExecutionProvider": "cpu",
             "RKNPUExecutionProvider": "rknpu"}


def _ts_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%S%z") + f"_{int(time.time()*1000)%1000:03d}"


def _model_short(model: str) -> str:
    return model.replace("mobilenet-v2-", "mob").replace("efficientnet-", "eff")


def condition_tag(job: MeasurementJob) -> str:
    """Return the condition token used in raw artifact filenames.

    A `_bs{N}` suffix is appended only when batch_size != 1, so every bs=1
    tag (and therefore every legacy filename) is byte-identical to before.
    """
    base = f"dvfs{job.dvfs_mode}" if job.dvfs_mode is not None else job.policy
    return base if job.batch_size == 1 else f"{base}_bs{job.batch_size}"


def condition_label(job: MeasurementJob) -> str:
    """Human-readable operating-condition label for logs and raw metadata."""
    if job.dvfs_label:
        return job.dvfs_label
    if job.expected_nvpmodel:
        return job.expected_nvpmodel
    if job.expected_gpu_freq_mhz:
        return f"{job.expected_gpu_freq_mhz}MHz"
    if job.dvfs_mode is not None:
        return "fixed"
    return job.policy


def make_output_path(job: MeasurementJob) -> Path:
    """Build the raw JSON path for a maximum sustainable throughput (MST) finding run."""
    name = (f"{_ts_tag()}_{job.device}_{_model_short(job.model)}_{condition_tag(job)}_"
            f"{_EP_SHORT.get(job.expected_ep, 'unk')}_v2.json")
    return RESULTS_RAW / name


def make_lambda_output_path(job: MeasurementJob, lambda_frac: float,
                            run_idx: int) -> Path:
    """Build the raw JSON path for a lambda-sweep cell run."""
    frac_tag = f"l{int(lambda_frac * 100):03d}"
    name = (f"{_ts_tag()}_{job.device}_{_model_short(job.model)}_{condition_tag(job)}_"
            f"{_EP_SHORT.get(job.expected_ep, 'unk')}_"
            f"{frac_tag}_r{run_idx}_lsweep.json")
    return RESULTS_RAW / name


def job_tag(job: MeasurementJob) -> str:
    if job.dvfs_mode is not None:
        return f"{job.device}/{job.model}/dvfs{job.dvfs_mode}"
    return f"{job.device}/{job.model}/{job.policy}"


def job_key(job: MeasurementJob) -> tuple[str, str, str, int | None, int]:
    """Stable key preserving policy label, resolved power mode, and batch width."""
    return (job.device, job.model, job.policy, job.dvfs_mode, job.batch_size)


def job_err(job: MeasurementJob, reason: str,
            error_type: str = "guard", **extra) -> dict:
    return {"status": "error", "error_type": error_type, "reason": reason,
            "job": job_tag(job), **extra}


# --- lambda-sweep configuration ---
# lambda_frac: arrival rate as fraction of that mode's confirmed MST.
# run_family: measurement campaign lineage (e.g. "full_dvfs", "policy_keyed").
LAMBDA_FRACS_DEFAULT: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
N_RUNS_PER_CELL_DEFAULT: int = 3

# Representative sub-sample for the λ-sweep main claim (Stage 5 of the
# execution plan). Coverage: ≥ 2 pairs per device type spanning
# light / medium / heavy models.
REPRESENTATIVE_PAIRS: list[tuple[str, str, str]] = [
    # server-class
    ("gpu-server", "mobilenet-v2-050", POLICY_EINC),
    ("gpu-server", "mobilenet-v2-100", POLICY_EINC),
    # Jetson-class
    ("orin",   "mobilenet-v2-100", POLICY_EINC),
    ("xavier", "mobilenet-v2-050", POLICY_EINC),
    ("jetson", "mobilenet-v2-050", POLICY_EINC),
    # NPU+CPU-SBC-class
    ("orangepi-npu", "mobilenet-v2-050", POLICY_EINC),
    ("rasp5",        "mobilenet-v2-050", POLICY_EINC),
    ("lattepanda",   "mobilenet-v2-100", POLICY_EINC),
]


def find_job(device: str, model: str, policy: str) -> MeasurementJob | None:
    for j in JOBS_ALL:
        if j.device == device and j.model == model and j.policy == policy:
            return j
    return None


def _profiler_config():
    """Import stage-1 profiler config helpers for full-DVFS planning."""
    import sys
    from constants import EEP_ROOT
    _profiler_src = str(EEP_ROOT / "src")
    if _profiler_src not in sys.path:
        sys.path.insert(0, _profiler_src)
    from config import configured_dvfs_modes, compatible_models, dvfs_mode_label, get_device
    return configured_dvfs_modes, compatible_models, dvfs_mode_label, get_device


def _model_file_for(device: str, model: str, device_cfg: dict | None = None) -> str:
    if device == "orangepi-npu":
        target = (device_cfg or {}).get("target_platform", "rk3588")
        return f"{model}-{target}-bs1.rknn"
    return f"{model}.onnx"


def _existing_mu_hints() -> dict[tuple[str, str], float]:
    hints: dict[tuple[str, str], float] = {}
    for job in JOBS_ALL:
        key = (job.device, job.model)
        if key not in hints or job.policy == POLICY_EINC:
            hints[key] = job.mu_hint
    return hints


def _profile_mu_hints() -> dict[tuple[str, str, int, int], float]:
    """Isolated throughput hints (ips) from eep-profiler profiles, keyed by
    (device, model, dvfs_mode, batch_size).

    All measured batch sizes are returned (the WHERE batch_size=1 filter is
    dropped); the enumerator resolves per-(cell,bs) via `_resolve_mu_hint`
    with a nearest-bs fallback. A bs=1 lookup still hits the bs=1 row exactly,
    so bs=1 enumeration is unchanged.
    """
    if not DB_PATH.exists():
        return {}
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT device, model, dvfs_mode, batch_size, throughput_ips
            FROM profiles
            WHERE throughput_ips > 0
            """
        ).fetchall()
    finally:
        conn.close()
    return {
        (str(device), str(model), int(dvfs_mode), int(batch_size)): float(throughput_ips)
        for device, model, dvfs_mode, batch_size, throughput_ips in rows
    }


def _resolve_mu_hint(profile_hints: dict[tuple[str, str, int, int], float],
                     existing: dict[tuple[str, str], float],
                     device: str, model: str, dvfs_mode: int, bs: int) -> float:
    """Resolve a μ-hint (ips) for a (device, model, dvfs_mode, bs) cell.

    Fallback chain: exact (d,m,mode,bs) → nearest measured bs for the same
    (d,m,mode) → existing (d,m) JOBS_ALL hint → 10.0. The μ-hint is only a soft
    search start, so the nearest-bs substitution cannot clip feasible capacity.
    """
    exact = profile_hints.get((device, model, dvfs_mode, bs))
    if exact is not None:
        return exact
    candidates = [(b, v) for (d, m, md, b), v in profile_hints.items()
                  if d == device and m == model and md == dvfs_mode]
    if candidates:
        return min(candidates, key=lambda bv: (abs(bv[0] - bs), bv[0]))[1]
    return existing.get((device, model), 10.0)


def _full_dvfs_job(device: str, model: str, dvfs_mode: int,
                   mu_hint: float, batch_size: int = 1) -> MeasurementJob:
    _, _, dvfs_mode_label, get_device = _profiler_config()
    cfg = get_device(device)
    ep = (cfg.get("ep") or [""])[0]
    mode_label = dvfs_mode_label(device, dvfs_mode)
    dvfs_setup: list[str] = []
    expected_gpu_freq_mhz = 0
    nvpmodel_mode = -1
    expected_nvpmodel = ""

    if cfg.get("dvfs_type") == "nvidia-smi":
        clocks = cfg.get("dvfs_clocks", [])
        if dvfs_mode < len(clocks):
            expected_gpu_freq_mhz = int(clocks[dvfs_mode])
            dvfs_setup = [
                f"sudo nvidia-smi -lgc {expected_gpu_freq_mhz},{expected_gpu_freq_mhz}"
            ]
    elif cfg.get("gpu") and "dvfs_modes" in cfg:
        nvpmodel_mode = dvfs_mode
        expected_nvpmodel = mode_label

    return MeasurementJob(
        device=device,
        model=model,
        policy="full_dvfs",
        model_file=_model_file_for(device, model, cfg),
        mu_hint=mu_hint,
        dvfs_setup=dvfs_setup,
        expected_ep=ep,
        dvfs_mode=dvfs_mode,
        dvfs_label=mode_label,
        needs_reboot=bool(cfg.get("dvfs_reboot", False)),
        nvpmodel_mode=nvpmodel_mode,
        expected_nvpmodel=expected_nvpmodel,
        expected_gpu_freq_mhz=expected_gpu_freq_mhz,
        batch_size=batch_size,
    )


def iter_full_dvfs_jobs(devices: set[str] | None = None,
                        skip_devices: set[str] | None = None,
                        models: set[str] | None = None,
                        batch_sizes: dict[str, tuple[int, ...]] | None = None
                        ) -> list[MeasurementJob]:
    """Generate the planned full-DVFS MST matrix without mutating JOBS_ALL.

    `batch_sizes` is an optional device-scoped batch-width map. A device absent
    from the map (or `batch_sizes=None`) is enumerated at bs=1 only, so legacy
    callers are unchanged. Phase-1 passes `constants.SERVER_BATCH_SIZES`
    ({"gpu-server": (1,2,4,8)}). Server-only guard: only devices listed in
    SERVER_BATCH_SIZES may carry bs>1.
    """
    configured_dvfs_modes, compatible_models, _, _ = _profiler_config()
    include_devices = devices
    skip = skip_devices or set()
    include_models = models or set(MODELS)
    bs_map = batch_sizes or {}
    mu_hints = _existing_mu_hints()
    profile_hints = _profile_mu_hints()

    jobs: list[MeasurementJob] = []
    for device in sorted(set(DEVICES) if include_devices is None else include_devices):
        if device in skip:
            continue
        bss = tuple(bs_map.get(device, (1,)))
        if any(b != 1 for b in bss) and device not in SERVER_BATCH_SIZES:
            raise ValueError(
                f"batch extension is server-only: bs>1 requested for {device!r}, "
                f"but only {sorted(SERVER_BATCH_SIZES)} may carry bs>1.")
        for model in sorted(set(compatible_models(device)) & include_models):
            for mode in configured_dvfs_modes(device):
                for bs in bss:
                    mu_hint = _resolve_mu_hint(
                        profile_hints, mu_hints, device, model, int(mode), int(bs))
                    jobs.append(_full_dvfs_job(
                        device=device,
                        model=model,
                        dvfs_mode=int(mode),
                        mu_hint=mu_hint,
                        batch_size=int(bs),
                    ))
    return jobs


def server_batch_map(batch_sizes) -> dict[str, tuple[int, ...]] | None:
    """Map requested batch widths onto server-capable devices (SERVER_BATCH_SIZES).

    Returns None when `batch_sizes` is falsy → bs=1 enumeration everywhere,
    byte-identical to the pre-batch behaviour. Each server device's grid is
    intersected with the request, so a non-grid width is simply dropped and a
    device with no overlap is omitted (left at bs=1 by the enumerator).
    """
    if not batch_sizes:
        return None
    req = {int(b) for b in batch_sizes}
    return {dev: tuple(sorted(set(grid) & req))
            for dev, grid in SERVER_BATCH_SIZES.items()
            if set(grid) & req}
