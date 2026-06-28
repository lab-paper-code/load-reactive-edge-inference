"""
Remote execution utilities for SSH/SCP operations.

Provides structured error handling and common operations for
interacting with remote devices in the measurement cluster.
"""
from __future__ import annotations

import hashlib
import subprocess


class RemoteError(Exception):
    """Structured error for SSH/SCP failures."""
    def __init__(self, error_type: str, device: str, command: str,
                 returncode: int, stderr: str):
        self.error_type = error_type
        self.device = device
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"[{error_type}] {device}: rc={returncode}, "
            f"cmd='{command[:80]}', stderr='{stderr[:200]}'")


def ssh(device: str, cmd: str, timeout: int = 30, check: bool = True) -> str:
    """Run command via SSH. Raises RemoteError on failure if check=True."""
    try:
        r = subprocess.run(["ssh", device, cmd],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RemoteError("ssh_timeout", device, cmd, -1,
                          f"timed out after {timeout}s")
    if check and r.returncode != 0:
        etype = "ssh_connect" if r.returncode == 255 else "remote_cmd"
        raise RemoteError(etype, device, cmd, r.returncode, r.stderr.strip())
    return r.stdout.strip()


def scp_to(local: str, device: str, remote: str):
    """Copy a local file to a remote device via SCP."""
    try:
        r = subprocess.run(["scp", local, f"{device}:{remote}"],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise RemoteError("scp_timeout", device, f"scp {local} → {remote}",
                          -1, "timed out after 120s")
    if r.returncode != 0:
        raise RemoteError("scp", device, f"scp {local} → {remote}",
                          r.returncode, r.stderr.strip())


def file_sha256(path: str) -> str:
    """Truncated SHA256 for deploy provenance (16 hex chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
