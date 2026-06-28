"""
Central configuration loader.

Reads configs/devices.yaml + configs/profiling.yaml and provides global settings.
All hard-coded values are managed here.
"""

import os
from typing import Tuple
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RKNN_MODEL_DIR = os.path.join(MODEL_DIR, 'rknn')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MEASURED_RESULTS_DIR = os.path.join(RESULTS_DIR, 'measured')
ANALYSIS_RESULTS_DIR = os.path.join(RESULTS_DIR, 'analysis')
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, 'raw')
MANIFESTS_RESULTS_DIR = os.path.join(RESULTS_DIR, 'manifests')
DB_PATH = os.path.join(DATA_DIR, 'eep_profiler.db')
LEGACY_DB_ARCHIVE_PATH = os.path.join(DATA_DIR, 'eep_profiler_legacy_v9.db')

_DEVICES_YAML = os.path.join(BASE_DIR, 'configs', 'devices.yaml')
_PROFILING_YAML = os.path.join(BASE_DIR, 'configs', 'profiling.yaml')

_cache = {}


def _load_yaml(path: str) -> dict:
    if path not in _cache:
        with open(path) as f:
            _cache[path] = yaml.safe_load(f)
    return _cache[path]


def devices_cfg() -> dict:
    return _load_yaml(_DEVICES_YAML)


def profiling_cfg() -> dict:
    return _load_yaml(_PROFILING_YAML)


# --- Convenience accessors ---

def get_device(name: str) -> dict:
    return devices_cfg()['devices'][name]


def get_all_devices() -> dict:
    return devices_cfg()['devices']


def shelly_devices() -> dict:
    """Return only Shelly-connected devices. {name: device_cfg}"""
    return {
        name: dev for name, dev in get_all_devices().items()
        if dev.get('shelly_hostname') and dev.get('include_in_all', True)
    }


def shelly_hostname(device_name: str):
    return get_device(device_name).get('shelly_hostname')


def shelly_server_ips() -> dict:
    """Optional direct shelly_server IPs keyed by Shelly hostname."""
    ips = {}
    for dev in get_all_devices().values():
        hostname = dev.get('shelly_hostname')
        ip = dev.get('shelly_server_ip')
        if hostname and ip:
            ips[str(hostname)] = str(ip)
    return ips


def physical_host_id(device_name: str) -> str:
    dev = get_device(device_name)
    return physical_host_id_from_cfg(dev)


def physical_host_id_from_cfg(dev: dict) -> str:
    return (
        dev.get('physical_host_id')
        or dev.get('hostname')
        or dev.get('ssh')
        or ''
    )


def default_dvfs_mode(device_name: str) -> int:
    return get_device(device_name).get('current_mode', 0)


def dvfs_mode_label(device_name: str, mode: int) -> str:
    """Human-readable DVFS mode label (e.g. 'MODE_30W', '2100MHz')."""
    cfg = get_device(device_name)
    modes = cfg.get('dvfs_modes', {})
    if modes and mode in modes:
        return str(modes[mode])
    clocks = cfg.get('dvfs_clocks', [])
    if clocks and mode < len(clocks):
        return f"{clocks[mode]}MHz"
    return str(mode)


def compatible_models(device_name: str) -> list:
    return get_device(device_name)['compatible_models']


def model_names() -> list:
    return list(profiling_cfg()['models'].keys())


def configured_dvfs_modes(device_name: str) -> list:
    cfg = get_device(device_name)
    if 'dvfs_modes' in cfg:
        return sorted(int(k) for k in cfg['dvfs_modes'].keys())
    if 'dvfs_clocks' in cfg:
        return list(range(len(cfg['dvfs_clocks'])))
    return [int(cfg.get('current_mode', 0))]


# --- Profiling constants ---

def input_shape() -> tuple:
    inp = profiling_cfg()['input']
    return inp['seq_len'], inp['n_features']


def prometheus_url() -> str:
    return profiling_cfg()['infra']['prometheus_url']


def remote_dir() -> str:
    return profiling_cfg()['infra']['remote_dir']


def ssh_timeout() -> int:
    return profiling_cfg()['infra']['ssh_connect_timeout']


def profiling_defaults() -> dict:
    return profiling_cfg()['profiling']


def training_cfg() -> dict:
    return profiling_cfg()['training']


def model_cfg(name: str) -> dict:
    return profiling_cfg()['models'][name]


def model_input_shape(name: str) -> Tuple[int, int, int]:
    shape = model_cfg(name)['input_shape']
    if len(shape) != 3:
        raise ValueError(f'Expected 3D input shape for {name}, got {shape}')
    return tuple(int(x) for x in shape)


def batch_sizes() -> list:
    return profiling_cfg().get('batch_sizes', [1, 4, 16, 64])


def ensure_results_layout():
    """Ensure canonical results directory layout exists."""
    for path in (RESULTS_DIR, MEASURED_RESULTS_DIR, ANALYSIS_RESULTS_DIR,
                 RAW_RESULTS_DIR, MANIFESTS_RESULTS_DIR, MODEL_DIR,
                 RKNN_MODEL_DIR):
        os.makedirs(path, exist_ok=True)
