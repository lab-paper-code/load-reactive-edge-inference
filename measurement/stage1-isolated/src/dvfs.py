"""Device DVFS control helpers for profiling sweeps."""

from __future__ import annotations

import time

from ssh import run

REBOOT_MAX_WAIT_S = 150
REBOOT_POLL_S = 5
POST_REBOOT_STABILIZE_S = 10
GPU_FREQ_TOLERANCE_MHZ = 75
DVFS_CMD_RETRIES = 6
DVFS_CMD_RETRY_SLEEP_S = 5


def _run_checked(ssh_alias: str, cmd: str, timeout: int = 30) -> tuple[str, str]:
    rc, out, err = run(ssh_alias, cmd, timeout=timeout)
    if rc != 0:
        raise RuntimeError(f'{ssh_alias}: rc={rc} cmd={cmd!r} err={err[:200]}')
    return out, err


def _wait_for_device_ready(ssh_alias: str, timeout_s: int = REBOOT_MAX_WAIT_S):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ready = True
        for cmd in ('echo up', 'python3 --version', 'nvpmodel -q'):
            rc, _, _ = run(ssh_alias, cmd, timeout=REBOOT_POLL_S)
            if rc != 0:
                ready = False
                break
        if ready:
            time.sleep(POST_REBOOT_STABILIZE_S)
            return
        time.sleep(REBOOT_POLL_S)
    raise RuntimeError(f'{ssh_alias}: device did not come back after reboot within {timeout_s}s')


def _run_checked_retry(ssh_alias: str, cmd: str, timeout: int = 30,
                       retries: int = DVFS_CMD_RETRIES,
                       sleep_s: int = DVFS_CMD_RETRY_SLEEP_S) -> tuple[str, str]:
    last_exc = None
    for attempt in range(retries):
        try:
            return _run_checked(ssh_alias, cmd, timeout=timeout)
        except RuntimeError as exc:
            last_exc = exc
            if attempt == retries - 1:
                break
            time.sleep(sleep_s)
    raise last_exc


def _verify_jetson_mode(ssh_alias: str, expected_label: str):
    out, _ = _run_checked(ssh_alias, 'nvpmodel -q', timeout=10)
    if expected_label not in out:
        raise RuntimeError(
            f"{ssh_alias}: DVFS mismatch, expected nvpmodel '{expected_label}', got '{out.strip()}'"
        )


def _verify_nvidia_clock(ssh_alias: str, expected_mhz: int):
    out, _ = _run_checked(
        ssh_alias,
        'nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits | head -1',
        timeout=10,
    )
    try:
        actual = int(float(out.strip()))
    except ValueError as exc:
        raise RuntimeError(f'{ssh_alias}: invalid nvidia-smi clock output: {out!r}') from exc
    if abs(actual - expected_mhz) > GPU_FREQ_TOLERANCE_MHZ:
        raise RuntimeError(
            f'{ssh_alias}: GPU clock mismatch, expected ~{expected_mhz}MHz, got {actual}MHz'
        )


def available_dvfs_modes(device_cfg: dict) -> list[int]:
    if 'dvfs_modes' in device_cfg:
        return sorted(int(k) for k in device_cfg['dvfs_modes'].keys())
    if 'dvfs_clocks' in device_cfg:
        return list(range(len(device_cfg['dvfs_clocks'])))
    return [int(device_cfg.get('current_mode', 0))]


def selected_dvfs_modes(device_cfg: dict,
                        dvfs_override: int | None = None,
                        all_dvfs: bool = False) -> list[int]:
    modes = available_dvfs_modes(device_cfg)
    if dvfs_override is not None:
        if int(dvfs_override) not in modes:
            raise ValueError(f'DVFS mode {dvfs_override} not valid for device; available={modes}')
        return [int(dvfs_override)]
    if all_dvfs:
        return modes
    return [int(device_cfg.get('current_mode', modes[0]))]


def apply_dvfs_mode(ssh_alias: str, device_name: str, device_cfg: dict, mode: int):
    """Apply and verify configured DVFS mode for the target device."""
    mode = int(mode)
    modes = available_dvfs_modes(device_cfg)
    if mode not in modes:
        raise ValueError(f'{device_name}: unsupported DVFS mode {mode}; available={modes}')

    if 'dvfs_modes' in device_cfg:
        label = str(device_cfg['dvfs_modes'][mode])
        if not device_cfg.get('gpu', False):
            return
        # Fixed single-mode devices still go through verification only.
        if len(modes) == 1 and mode == int(device_cfg.get('current_mode', mode)):
            _verify_jetson_mode(ssh_alias, label)
            return

        if device_cfg.get('dvfs_reboot'):
            # Orin mode changes can reboot the device.
            rc, _, err = run(
                ssh_alias,
                f"printf 'YES\\n' | sudo nvpmodel -m {mode}",
                timeout=15,
            )
            if rc not in (0, 255, -1):
                raise RuntimeError(
                    f'{device_name}: nvpmodel set failed rc={rc} err={err[:200]}'
                )
            _wait_for_device_ready(ssh_alias)
        else:
            _run_checked(ssh_alias, f'sudo nvpmodel -m {mode}', timeout=15)

        _run_checked_retry(ssh_alias, 'sudo jetson_clocks', timeout=20)
        _verify_jetson_mode(ssh_alias, label)
        return

    if device_cfg.get('dvfs_type') == 'nvidia-smi':
        target = int(device_cfg['dvfs_clocks'][mode])
        _run_checked(ssh_alias, f'sudo nvidia-smi -lgc {target},{target}', timeout=20)
        _verify_nvidia_clock(ssh_alias, target)
        return

    # Fixed / unsupported types: no-op.
    return
