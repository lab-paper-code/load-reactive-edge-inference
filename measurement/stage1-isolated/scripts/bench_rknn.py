"""
Remote RKNN inference benchmark.

Usage:
    python3 bench_rknn.py <model.rknn> <batch_size> <duration_s> <warmup>
                          <channels> <height> <width> [trials] [cooldown]
                          [idle_window_s]

Emits the same NDJSON event/result schema as scripts/bench.py so profiler.py
can reuse the existing measurement and aggregation pipeline.
"""

import json
import sys
import time
from dataclasses import dataclass

import numpy as np
from rknnlite.api import RKNNLite


@dataclass
class Args:
    model_path: str
    batch_size: int
    duration_s: float
    warmup: int
    channels: int
    height: int
    width: int
    trials: int = 1
    cooldown_s: float = 0
    idle_window_s: float = 10.0
    core_mask_name: str = 'NPU_CORE_0'
    runtime_backend: str = 'rknnlite'


def _resolve_core_mask(name: str) -> int:
    if not hasattr(RKNNLite, name):
        raise ValueError(f'Unknown RKNN core mask: {name}')
    return getattr(RKNNLite, name)


def emit_event(name: str, **kwargs):
    payload = {'event': name, 'ts': round(time.time(), 3)}
    payload.update(kwargs)
    print(json.dumps(payload), flush=True)


def _parse_args(argv) -> Args:
    return Args(
        model_path=argv[1],
        batch_size=int(argv[2]),
        duration_s=float(argv[3]),
        warmup=int(argv[4]),
        channels=int(argv[5]),
        height=int(argv[6]),
        width=int(argv[7]),
        trials=int(argv[8]) if len(argv) > 8 else 1,
        cooldown_s=float(argv[9]) if len(argv) > 9 else 0,
        idle_window_s=float(argv[10]) if len(argv) > 10 else 10.0,
        core_mask_name=argv[11] if len(argv) > 11 else 'NPU_CORE_0',
        runtime_backend=argv[12] if len(argv) > 12 else 'rknnlite',
    )


def main(argv):
    args = _parse_args(argv)
    rknn = RKNNLite()
    ret = rknn.load_rknn(args.model_path)
    if ret != 0:
        raise RuntimeError(f'load_rknn failed: rc={ret}')

    ret = rknn.init_runtime(core_mask=_resolve_core_mask(args.core_mask_name))
    if ret != 0:
        raise RuntimeError(f'init_runtime failed: rc={ret}')

    dummy = np.random.randn(
        args.batch_size, args.channels, args.height, args.width
    ).astype(np.float32)
    ep = 'RKNPUExecutionProvider'
    emit_event('session_ready', ep=ep, providers=[ep],
               batch_size=args.batch_size,
               runtime_class='rknn',
               runtime_backend=args.runtime_backend,
               core_mask=args.core_mask_name)

    emit_event('warmup_start', warmup=args.warmup)
    for _ in range(args.warmup):
        rknn.inference(inputs=[dummy])
    warmup_done_ts = time.time()
    emit_event('warmup_done', warmup=args.warmup)

    for trial_idx in range(args.trials):
        if trial_idx > 0 and args.cooldown_s > 0:
            time.sleep(args.cooldown_s)

        idle_start_ts = time.time()
        emit_event('idle_start', trial=trial_idx, idle_window_s=args.idle_window_s)
        time.sleep(args.idle_window_s)
        idle_end_ts = time.time()
        emit_event('idle_done', trial=trial_idx, idle_elapsed_s=round(idle_end_ts - idle_start_ts, 3))

        latencies = []
        t_start = time.time()
        deadline = t_start + args.duration_s
        first_infer_start_ts = None
        last_infer_end_ts = None
        emit_event('trial_start', trial=trial_idx, duration_s=args.duration_s)
        while time.time() < deadline:
            infer_start_wall = time.time()
            if first_infer_start_ts is None:
                first_infer_start_ts = infer_start_wall
            t0 = time.perf_counter()
            rknn.inference(inputs=[dummy])
            latencies.append((time.perf_counter() - t0) * 1000)
            last_infer_end_ts = time.time()
        t_end = time.time()
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
            'batch_size': args.batch_size,
            'n_infer': n_infer,
            'elapsed_s': round(elapsed, 3),
            'latency_ms_mean': round(float(lat.mean()), 4),
            'latency_ms_p50': round(float(np.percentile(lat, 50)), 4),
            'latency_ms_p95': round(float(np.percentile(lat, 95)), 4),
            'latency_ms_p99': round(float(np.percentile(lat, 99)), 4),
            'latency_ms_std': round(float(lat.std()), 4),
            'throughput_ips': round(n_infer * args.batch_size / elapsed, 2),
        }), flush=True)
    emit_event('benchmark_done', trials=args.trials)
    rknn.release()


if __name__ == '__main__':
    main(sys.argv)
