"""
Inference benchmark script executed on the remote device.
Deployed via SSH from profiler.py / deploy_verify.py, then run there.

Usage:
    python3 bench.py <model.onnx> <batch_size> <duration_s> <warmup> [ep] [trials]
                     [cooldown] [idle_window_s]

Input tensor shape is auto-detected from the ONNX model; works for any model type (AE, classification, etc.).
When trials > 1, outputs NDJSON (one trial per line). ORT session is built only once.
"""
import os
import sys
import time
import json
import numpy as np
import onnxruntime as ort


def emit_event(name: str, **kwargs):
    payload = {'event': name, 'ts': round(time.time(), 3)}
    payload.update(kwargs)
    print(json.dumps(payload), flush=True)

model_path = sys.argv[1]
batch_size = int(sys.argv[2])
duration_s = float(sys.argv[3])
warmup = int(sys.argv[4])
ep_arg = sys.argv[5] if len(sys.argv) > 5 else None
trials = int(sys.argv[6]) if len(sys.argv) > 6 else 1
cooldown_s = float(sys.argv[7]) if len(sys.argv) > 7 else 0
idle_window_s = float(sys.argv[8]) if len(sys.argv) > 8 else 10.0

if ep_arg and ep_arg.lower() not in ('auto', 'default', 'none'):
    requested = ep_arg.split(',')
    available = ort.get_available_providers()
    supported = [ep for ep in requested if ep in available]
    if not supported:
        raise RuntimeError(f'None of requested EPs {requested} available. Have: {available}')
    providers = supported
else:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = int(os.environ.get('ORT_INTRA_OP', 2))
sess_opts.inter_op_num_threads = int(os.environ.get('ORT_INTER_OP', 1))
sess_opts.log_severity_level = 3

uses_trt = any(p == 'TensorrtExecutionProvider' for p in providers)
trt_cache_present = False
if uses_trt:
    cache_dir = os.environ.get('TRT_CACHE_DIR', '/tmp/eep_trt_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Heuristic probe: TRT cache presence is informational only.
    model_stem = os.path.basename(model_path).rsplit('.', 1)[0]
    try:
        with os.scandir(cache_dir) as it:
            trt_cache_present = any(e.name.startswith(model_stem) for e in it)
    except OSError:
        pass
    trt_opts = {
        'trt_engine_cache_enable': 'True',
        'trt_engine_cache_path': cache_dir,
    }
    providers = [
        (p, trt_opts) if p == 'TensorrtExecutionProvider' else p
        for p in providers
    ]

emit_event(
    'session_build_start',
    trt_cache_present=trt_cache_present,
    intra_op=sess_opts.intra_op_num_threads,
    inter_op=sess_opts.inter_op_num_threads,
)
sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
ep = sess.get_providers()[0]
emit_event('session_ready', ep=ep, providers=sess.get_providers(), batch_size=batch_size)

# Auto-detect input shape (batch dim overridden by CLI argument)
input_meta = sess.get_inputs()[0]
input_name = input_meta.name
raw_shape = input_meta.shape  # e.g. ['batch', 3, 224, 224] or [1, 60, 32]
shape = [batch_size] + [d if isinstance(d, int) else 1 for d in raw_shape[1:]]
dummy = np.random.randn(*shape).astype(np.float32)

emit_event('warmup_start', warmup=warmup)
for _ in range(warmup):
    sess.run(None, {input_name: dummy})
warmup_done_ts = time.time()
emit_event('warmup_done', warmup=warmup)

for trial_idx in range(trials):
    if trial_idx > 0 and cooldown_s > 0:
        time.sleep(cooldown_s)

    idle_start_ts = time.time()
    emit_event('idle_start', trial=trial_idx, idle_window_s=idle_window_s)
    time.sleep(idle_window_s)
    idle_end_ts = time.time()
    emit_event('idle_done', trial=trial_idx, idle_elapsed_s=round(idle_end_ts - idle_start_ts, 3))

    latencies = []
    t_start = time.time()
    perf_deadline = time.perf_counter() + duration_s
    first_infer_start_ts = t_start
    emit_event('trial_start', trial=trial_idx, duration_s=duration_s)
    while time.perf_counter() < perf_deadline:
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)
    t_end = time.time()
    last_infer_end_ts = t_end if latencies else None
    elapsed = t_end - t_start
    n_infer = len(latencies)
    emit_event('trial_done', trial=trial_idx, elapsed_s=round(elapsed, 3), n_infer=n_infer)

    lat = np.array(latencies)
    print(json.dumps({
        'trial': trial_idx,
        'warmup_done_ts': round(warmup_done_ts, 3),
        'idle_start_ts': round(idle_start_ts, 3),
        'idle_end_ts': round(idle_end_ts, 3),
        't_start': round(t_start, 3),
        't_end': round(t_end, 3),
        'active_start_ts': round(first_infer_start_ts or t_start, 3),
        'active_end_ts': round(last_infer_end_ts or t_end, 3),
        'ep': ep,
        'batch_size': batch_size,
        'n_infer': n_infer,
        'elapsed_s': round(elapsed, 3),
        'latency_ms_mean': round(float(lat.mean()), 4),
        'latency_ms_p50': round(float(np.percentile(lat, 50)), 4),
        'latency_ms_p95': round(float(np.percentile(lat, 95)), 4),
        'latency_ms_p99': round(float(np.percentile(lat, 99)), 4),
        'latency_ms_std': round(float(lat.std()), 4),
        'throughput_ips': round(n_infer * batch_size / elapsed, 2),
    }), flush=True)
emit_event('benchmark_done', trials=trials)
