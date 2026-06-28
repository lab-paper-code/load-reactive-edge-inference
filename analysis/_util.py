"""Shared utilities for atomic writes and shared pipeline constants.

Provides interrupt-safe CSV/JSON writes (tmp file + os.replace) and the
POWER_EXCLUDED_DEVICES set used by the serving-power and T2 build scripts.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

# POWER_EXCLUDED_DEVICES: devices without a power meter (no Shelly Plug S attached).
# These devices cannot contribute wall-energy measurements and are excluded from
# the wall-energy computation in the serving-power and T2 build scripts.
# Transient operational issues must NOT be encoded here; only devices that are
# structurally unmetered belong in this set.
POWER_EXCLUDED_DEVICES: frozenset[str] = frozenset({"meterless-a", "meterless-b"})


def atomic_write(path: Path, render_fn) -> None:
    """Write via tmp file + os.replace so interrupts never leave a partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", newline="") as f:
            render_fn(f)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_csv_write(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    def _render(f):
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    atomic_write(path, _render)


def atomic_json_write(path: Path, payload) -> None:
    atomic_write(path, lambda f: json.dump(payload, f, indent=2))


def marginal_wall_energy_per_image(delta_w, rate, n):
    """Per-image marginal wall energy = ΔW / (rate × N), in joules per image.

    marginal_wall_energy: wall-plug energy per inference above idle.
    Single source of truth for the per-image energy FORMULA, shared by the
    λ-sweep builder, build_serving_power_csv, and the freshness gate. It is a
    PURE QUOTIENT: no rounding and no n==1 special-case. Rounding is each call
    site's responsibility (the λ builder rounds to 6 dp to match its frozen
    artifacts; build_serving_power_csv stores the raw quotient). A helper that
    rounded internally would break byte-identity at exactly one of the two
    sites, since they round differently. At N=1 it reduces to ΔW/rate, so bs=1
    values are unchanged. Returns None when rate is missing or non-positive;
    callers keep their own bad-rate handling (None-guard / row-skip)."""
    if rate is None or rate <= 0:
        return None
    return delta_w / (rate * n)
