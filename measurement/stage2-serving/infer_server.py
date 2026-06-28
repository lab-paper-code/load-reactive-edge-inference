"""
S1-2: HTTP Inference Server (aiohttp, single inference thread)

Deployed to remote devices via SSH and executed there. Loads ONNX (.onnx) or RKNN (.rknn) model once.

Architecture:
  - aiohttp async HTTP server: handles connections
  - Inference runs sequentially in a ThreadPoolExecutor(max_workers=1)
  → Measures queuing delay naturally without ORT mutex contention

Usage (on remote device):
    python3 infer_server.py <model.onnx|model.rknn> [--port 8090] [--ep auto]
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from aiohttp import web


# --- Constants ---
INFER_WORKERS = 1              # Single-thread executor (no ORT mutex contention)
DEFAULT_WARMUP = 50            # Warmup inference count before serving
SUBPROCESS_TIMEOUT = 5         # Timeout for nvidia-smi/nvpmodel subprocesses
DEFAULT_PORT = 8090

# --- Globals ---
sess = None          # ort.InferenceSession or rknnlite.RKNNLite instance
input_name = None    # ORT input name (unused for RKNN)
dummy_input = None
model_name = ""
ep_name = ""
server_start_ts = 0.0
executor = ThreadPoolExecutor(max_workers=INFER_WORKERS)
_infer_fn = None     # callable() → latency_ms; set by init_model

# Known Jetson GPU devfreq sysfs base paths (Orin, Xavier NX, Nano)
_JETSON_GPU_DEVFREQ_PATHS = [
    Path("/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu"),  # Orin (JetPack 6)
    Path("/sys/devices/17000000.ga10b/devfreq/17000000.ga10b"),  # Orin
    Path("/sys/devices/17000000.gv11b/devfreq/17000000.gv11b"),  # Xavier NX
    Path("/sys/devices/57000000.gpu/devfreq/57000000.gpu"),       # Nano
]


def _find_jetson_gpu_devfreq() -> Path | None:
    """Return the first existing Jetson GPU devfreq sysfs directory."""
    for p in _JETSON_GPU_DEVFREQ_PATHS:
        if p.is_dir():
            return p
    return None


def _read_sysfs_int(path: Path) -> int | None:
    """Read an integer value from a sysfs file, returning None on failure."""
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def _parse_numeric_token(value: str, cast):
    """Parse nvidia-smi numeric tokens, ignoring Jetson '[N/A]' values."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _run_inference():
    """Run a single inference in the single-thread executor. No contention."""
    t0 = time.perf_counter()
    _infer_fn()
    return (time.perf_counter() - t0) * 1000


async def handle_infer(request):
    loop = asyncio.get_running_loop()
    t_submit = time.perf_counter()
    infer_ms = await loop.run_in_executor(executor, _run_inference)
    total_ms = (time.perf_counter() - t_submit) * 1000
    return web.json_response({
        "latency_ms": round(total_ms, 4),
        "infer_ms": round(infer_ms, 4),
        "ok": True,
    })


async def handle_health(request):
    """Liveness probe."""
    return web.json_response({"status": "ok"})


async def handle_status(request):
    """Return hardware state for measurement provenance."""
    status = {
        "model": model_name,
        "ep": ep_name,
        "input_shape": list(dummy_input.shape),
        "batch_size": int(dummy_input.shape[0]),
        "hostname": platform.node(),
        "platform": platform.machine(),
        "server_start_ts": server_start_ts,
    }

    # CPU frequency (all Linux platforms)
    cpufreq = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    cur = _read_sysfs_int(cpufreq / "scaling_cur_freq")
    if cur is not None:
        status["cpu_cur_freq_khz"] = cur
    mx = _read_sysfs_int(cpufreq / "scaling_max_freq")
    if mx is not None:
        status["cpu_max_freq_khz"] = mx

    # Jetson: nvpmodel
    try:
        nv = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        if nv.returncode == 0:
            lines = nv.stdout.strip().split("\n")
            power_mode = next(
                (line for line in lines if line.startswith("NV Power Mode:")),
                None,
            )
            status["nvpmodel"] = power_mode or (lines[0] if lines else "unknown")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Jetson: GPU freq via devfreq sysfs
    gpu_devfreq = _find_jetson_gpu_devfreq()
    if gpu_devfreq is not None:
        cur = _read_sysfs_int(gpu_devfreq / "cur_freq")
        if cur is not None:
            status["gpu_cur_freq_hz"] = cur
        mx = _read_sysfs_int(gpu_devfreq / "max_freq")
        if mx is not None:
            status["gpu_max_freq_hz"] = mx

    # nvidia-smi (desktop GPU)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.current.graphics,clocks.max.graphics,"
             "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT,
        )
        if smi.returncode == 0:
            parts = [p.strip() for p in smi.stdout.strip().split(",")]
            if len(parts) >= 4:
                gpu_clock_mhz = _parse_numeric_token(parts[0], int)
                gpu_max_clock_mhz = _parse_numeric_token(parts[1], int)
                gpu_temp_c = _parse_numeric_token(parts[2], int)
                gpu_power_w = _parse_numeric_token(parts[3], float)
                if gpu_clock_mhz is not None:
                    status["gpu_clock_mhz"] = gpu_clock_mhz
                if gpu_max_clock_mhz is not None:
                    status["gpu_max_clock_mhz"] = gpu_max_clock_mhz
                if gpu_temp_c is not None:
                    status["gpu_temp_c"] = gpu_temp_c
                if gpu_power_w is not None:
                    status["gpu_power_w"] = gpu_power_w
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return web.json_response(status)


async def handle_info(request):
    """Return model metadata."""
    return web.json_response({
        "model": model_name,
        "ep": ep_name,
        "input_shape": list(dummy_input.shape),
    })


def init_model(model_path: str, ep_arg: str, batch_size: int, warmup: int):
    """Load model, select execution provider, and run warmup. Supports .onnx and .rknn."""
    if model_path.endswith(".rknn"):
        _init_rknn(model_path, warmup)
    else:
        _init_ort(model_path, ep_arg, batch_size, warmup)


def _init_ort(model_path: str, ep_arg: str, batch_size: int, warmup: int):
    """Load ONNX model via ONNX Runtime."""
    global sess, input_name, dummy_input, model_name, ep_name, server_start_ts, _infer_fn

    import onnxruntime as ort

    if ep_arg and ep_arg.lower() not in ("auto", "default", "none"):
        requested = ep_arg.split(",")
        available = ort.get_available_providers()
        providers = [ep for ep in requested if ep in available] or ["CPUExecutionProvider"]
    else:
        if platform.machine() == "aarch64":
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(model_path, providers=providers)
    ep_name = sess.get_providers()[0]

    meta = sess.get_inputs()[0]
    input_name = meta.name
    raw_shape = meta.shape
    shape = [batch_size] + [d if isinstance(d, int) else 1 for d in raw_shape[1:]]
    dummy_input = np.random.randn(*shape).astype(np.float32)
    model_name = Path(model_path).name.replace(".onnx", "")

    _infer_fn = lambda: sess.run(None, {input_name: dummy_input})

    for _ in range(warmup):
        _infer_fn()

    server_start_ts = time.time()
    print(f"Model: {model_name}, EP: {ep_name}, Shape: {shape}, Warmup: {warmup}")


# Input shapes for RKNN models keyed by canonical model name.
# Shape is (C, H, W); batch is always 1 for bs1 artifacts.
# Must be updated when a new model is exported to RKNN.
_RKNN_INPUT_SHAPES: dict[str, tuple[int, int, int]] = {
    "mobilenet-v2-050": (3, 224, 224),
    "mobilenet-v2-100": (3, 224, 224),
    "efficientnet-b4":  (3, 380, 380),
}


def _rknn_canonical_name(rknn_filename: str) -> str:
    """Extract canonical model name from RKNN artifact filename.

    e.g. 'mobilenet-v2-050-rk3588-bs1.rknn' → 'mobilenet-v2-050'
    """
    stem = Path(rknn_filename).stem   # e.g. mobilenet-v2-050-rk3588-bs1
    return re.sub(r"-rk\d+-bs\d+$", "", stem)


def _init_rknn(model_path: str, warmup: int):
    """Load RKNN model via rknnlite. Batch size is always 1 (bs1 artifact)."""
    global sess, input_name, dummy_input, model_name, ep_name, server_start_ts, _infer_fn

    canonical = _rknn_canonical_name(model_path)
    if canonical not in _RKNN_INPUT_SHAPES:
        raise ValueError(
            f"Unknown RKNN model '{canonical}'. "
            f"Add it to _RKNN_INPUT_SHAPES in infer_server.py."
        )
    C, H, W = _RKNN_INPUT_SHAPES[canonical]

    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: rc={ret}")
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: rc={ret}")

    # Clean NPU core release on termination.
    # Controlled experiments (2026-04-16) showed SIGKILL without release()
    # does NOT corrupt NPU state. This handler is defensive hygiene, not
    # a crash-safety requirement.
    def _rknn_shutdown(sig, frame):
        executor.shutdown(wait=True)   # wait for in-flight inference to finish
        try:
            rknn.release()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _rknn_shutdown)
    signal.signal(signal.SIGINT, _rknn_shutdown)

    dummy = np.random.randn(1, C, H, W).astype(np.float32)

    _infer_fn = lambda: rknn.inference(inputs=[dummy])

    for _ in range(warmup):
        _infer_fn()

    sess = rknn
    input_name = None
    dummy_input = dummy
    ep_name = "RKNPUExecutionProvider"
    model_name = canonical   # canonical name for /status model field

    server_start_ts = time.time()
    print(f"Model: {model_name}, EP: {ep_name}, Shape: {list(dummy.shape)}, Warmup: {warmup}")


def _acquire_server_lock(port: int):
    """Acquire board-level exclusive lock; only one infer_server per device.

    Uses a single lock path (not per-port) because different ports on the same
    board still compete for CPU/NPU/memory. Returns the open file handle.
    """
    lock_path = Path("/tmp/eep/infer_server.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Read PID from existing lock
        try:
            existing = open(lock_path).read().strip()
        except Exception:
            existing = "unknown"
        print(f"ABORT: another infer_server is already running on this device "
              f"(PID {existing}). Kill it first.",
              file=sys.stderr)
        sys.exit(1)
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def main():
    parser = argparse.ArgumentParser(description="S1-2: HTTP Inference Server")
    parser.add_argument("model", help="Path to model (.onnx or .rknn)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ep", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    args = parser.parse_args()

    _lock_fh = _acquire_server_lock(args.port)  # noqa: F841, must stay open

    init_model(args.model, args.ep, args.batch_size, args.warmup)

    app = web.Application()
    app.router.add_post("/infer", handle_infer)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/info", handle_info)

    print(f"Serving on 0.0.0.0:{args.port} (aiohttp, single inference thread)")
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
