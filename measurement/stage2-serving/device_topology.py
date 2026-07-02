"""Network + physical-host topology for S1-capacity measurement runs.

Single source of truth for device IP map, inference-server ports, Shelly plug
hostnames, and physical-host groupings. Consumers: server_lifecycle,
measurement_runner, lambda_sweep, admit_new_runs.
"""
from __future__ import annotations

# --- Remote workspace (all SSH targets must have this path writable) ---
REMOTE_WORK_DIR = "/tmp/release-bench"

# --- Device network endpoints ---
# Device network endpoints are loaded from config.example.yaml in the public release.
# Real measurement requires populating ssh_aliases / IPs for your own fleet.
DEVICE_HOSTS = {d: "" for d in (
    "orin","orin-nano","xavier","gpu-server","lattepanda","jetson","orangepi","orangepi-npu","rasp5")}

# --- Inference-server ports ---
INFER_PORT = 8090
# Per-device overrides required when two logical devices share a physical host.
DEVICE_PORT: dict[str, int] = {
    "orangepi-npu": 8091,
}

# --- Physical-host groups ---
# Logical devices on the same board must NOT run in parallel. Devices sharing
# a physical host are serialized within that host's slot.
PHYSICAL_HOST: dict[str, str] = {
    "orangepi":     "orangepi-board",
    "orangepi-npu": "orangepi-board",
}


def device_port(device: str) -> int:
    """Return the inference-server port for a logical device."""
    return DEVICE_PORT.get(device, INFER_PORT)


# --- Shelly hostnames (AC input power plug sources) ---
# All devices with Shelly plugs (regardless of profiler include_in_all flag;
# power instrumentation eligibility is separate from profiler sweep scope).
# Shelly plug hostnames are populated from your fleet's devices.yaml at runtime.
SHELLY_HOSTNAMES: dict[str, str] = {}
