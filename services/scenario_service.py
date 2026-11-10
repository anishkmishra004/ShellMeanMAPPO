"""Scenario configuration presets and validation for ShellMean-MAPPO platform."""
from __future__ import annotations
import copy
import yaml
from typing import Dict, Any, Tuple, List
from pathlib import Path

BASE_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

SCENARIO_PRESETS = {
    "Normal Traffic": {
        "description": "Standard operating conditions. Balanced load across ground cells.",
        "traffic": {
            "hotspot_cell_ratio": 0.25,
            "hotspot_arrival_rate_mbps": 12.0,
            "regular_arrival_rate_mbps": 3.5,
            "burst_probability": 0.10,
            "burst_multiplier": 2.0,
            "initial_queue_max_mb": 20.0,
            "arrival_distribution": "poisson",
        },
    },
    "High Traffic": {
        "description": "Elevated demand. All cells experience higher arrival rates.",
        "traffic": {
            "hotspot_cell_ratio": 0.35,
            "hotspot_arrival_rate_mbps": 20.0,
            "regular_arrival_rate_mbps": 7.0,
            "burst_probability": 0.20,
            "burst_multiplier": 2.5,
            "initial_queue_max_mb": 50.0,
            "arrival_distribution": "poisson",
        },
    },
    "Congested Network": {
        "description": "Network near saturation. Large initial backlogs. Tests conflict resolution limits.",
        "traffic": {
            "hotspot_cell_ratio": 0.50,
            "hotspot_arrival_rate_mbps": 25.0,
            "regular_arrival_rate_mbps": 10.0,
            "burst_probability": 0.30,
            "burst_multiplier": 3.0,
            "initial_queue_max_mb": 100.0,
            "arrival_distribution": "bursty",
        },
    },
    "Uneven Traffic Distribution": {
        "description": "Few cells monopolize demand. Fairness stress test.",
        "traffic": {
            "hotspot_cell_ratio": 0.15,
            "hotspot_arrival_rate_mbps": 40.0,
            "regular_arrival_rate_mbps": 1.0,
            "burst_probability": 0.05,
            "burst_multiplier": 2.0,
            "initial_queue_max_mb": 80.0,
            "arrival_distribution": "poisson",
        },
    },
    "High Interference": {
        "description": "Dense spectrum reuse scenario. Many satellites contending on overlapping RBs.",
        "orbit": {
            "satellite_tx_power_w": 60.0,
        },
        "traffic": {
            "hotspot_cell_ratio": 0.40,
            "hotspot_arrival_rate_mbps": 18.0,
            "regular_arrival_rate_mbps": 6.0,
            "burst_probability": 0.20,
            "burst_multiplier": 3.0,
            "initial_queue_max_mb": 40.0,
            "arrival_distribution": "poisson",
        },
    },
    "Multi-Shell Load Imbalance": {
        "description": "Shell 0 (550km) overloaded; Shell 1 (1100km) underutilized. Tests MAPPO shell coordination.",
        "traffic": {
            "hotspot_cell_ratio": 0.40,
            "hotspot_arrival_rate_mbps": 18.0,
            "regular_arrival_rate_mbps": 6.0,
            "burst_probability": 0.25,
            "burst_multiplier": 3.0,
            "initial_queue_max_mb": 60.0,
            "arrival_distribution": "bursty",
        },
    },
}


def load_base_config() -> Dict[str, Any]:
    """Load the default YAML configuration."""
    with open(BASE_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def apply_preset(base_config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """
    Apply a scenario preset on top of the base configuration.

    Args:
        base_config: Base configuration dictionary.
        preset_name: Name of the preset from SCENARIO_PRESETS.

    Returns:
        New configuration dictionary with preset values merged.
    """
    if preset_name not in SCENARIO_PRESETS:
        return copy.deepcopy(base_config)

    cfg = copy.deepcopy(base_config)
    preset = SCENARIO_PRESETS[preset_name]

    for section_key, section_vals in preset.items():
        if section_key == "description":
            continue
        if section_key not in cfg:
            cfg[section_key] = {}
        cfg[section_key].update(section_vals)

    return cfg


def build_config_from_ui(
    base_config: Dict[str, Any],
    # Network
    num_satellites: int,
    num_shells: int,
    sats_per_shell: List[int],
    beams_per_satellite: int,
    num_ground_cells: int,
    num_resource_blocks: int,
    rb_bandwidth_mhz: float,
    episode_length: int,
    # Orbit
    shell_altitudes_km: List[float],
    inclination_deg: List[float],
    satellite_tx_power_w: float,
    satellite_antenna_gain_dbi: float,
    ground_antenna_gain_dbi: float,
    min_elevation_deg: float,
    noise_temperature_k: float,
    implementation_efficiency: float,
    # Traffic
    hotspot_cell_ratio: float,
    hotspot_arrival_rate_mbps: float,
    regular_arrival_rate_mbps: float,
    burst_probability: float,
    burst_multiplier: float,
    initial_queue_max_mb: float,
    max_queue_capacity_mb: float,
    arrival_distribution: str,
    # Conflict
    m_c_max: int,
    q_seg_mb: float,
    # MAPPO
    hidden_dim: int,
    actor_lr: float,
    critic_lr: float,
    gamma: float,
    gae_lambda: float,
    clip_param: float,
    ppo_epochs: int,
    rollout_steps: int,
    # Experiment
    seed: int,
    carrier_frequency_ghz: float,
) -> Dict[str, Any]:
    """
    Build a complete configuration dictionary from UI inputs.

    Returns:
        Merged configuration dictionary ready for MultiShellLEOEnv.
    """
    cfg = copy.deepcopy(base_config)

    cfg["network"].update({
        "num_satellites": num_satellites,
        "num_shells": num_shells,
        "satellites_per_shell": sats_per_shell,
        "beams_per_satellite": beams_per_satellite,
        "num_ground_cells": num_ground_cells,
        "num_resource_blocks": num_resource_blocks,
        "rb_bandwidth_mhz": rb_bandwidth_mhz,
        "carrier_frequency_ghz": carrier_frequency_ghz,
        "episode_length": episode_length,
    })

    cfg["orbit"].update({
        "shell_altitudes_km": shell_altitudes_km,
        "inclination_deg": inclination_deg,
        "satellite_tx_power_w": satellite_tx_power_w,
        "satellite_antenna_gain_dbi": satellite_antenna_gain_dbi,
        "ground_antenna_gain_dbi": ground_antenna_gain_dbi,
        "min_elevation_deg": min_elevation_deg,
        "noise_temperature_k": noise_temperature_k,
        "implementation_efficiency": implementation_efficiency,
    })

    cfg["traffic"].update({
        "hotspot_cell_ratio": hotspot_cell_ratio,
        "hotspot_arrival_rate_mbps": hotspot_arrival_rate_mbps,
        "regular_arrival_rate_mbps": regular_arrival_rate_mbps,
        "burst_probability": burst_probability,
        "burst_multiplier": burst_multiplier,
        "initial_queue_max_mb": initial_queue_max_mb,
        "max_queue_capacity_mb": max_queue_capacity_mb,
        "arrival_distribution": arrival_distribution,
    })

    cfg["conflict_resolution"].update({
        "m_c_max": m_c_max,
        "q_seg_mb": q_seg_mb,
    })

    cfg["mappo"].update({
        "hidden_dim": hidden_dim,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_param": clip_param,
        "ppo_epochs": ppo_epochs,
        "rollout_steps": rollout_steps,
    })

    cfg["experiment"]["seed"] = seed

    return cfg


def validate_config(cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a configuration dictionary for logical consistency.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors = []
    net = cfg.get("network", {})
    orb = cfg.get("orbit", {})
    traf = cfg.get("traffic", {})

    num_shells = net.get("num_shells", 2)
    sats_per_shell = net.get("satellites_per_shell", [4, 4])
    num_satellites = net.get("num_satellites", 8)

    if len(sats_per_shell) != num_shells:
        errors.append(f"satellites_per_shell length ({len(sats_per_shell)}) must match num_shells ({num_shells}).")

    if sum(sats_per_shell) != num_satellites:
        errors.append(
            f"Sum of satellites_per_shell ({sum(sats_per_shell)}) must equal num_satellites ({num_satellites})."
        )

    shell_altitudes = orb.get("shell_altitudes_km", [550.0, 1100.0])
    if len(shell_altitudes) != num_shells:
        errors.append(f"shell_altitudes_km length ({len(shell_altitudes)}) must match num_shells ({num_shells}).")

    inclinations = orb.get("inclination_deg", [53.0, 70.0])
    if len(inclinations) != num_shells:
        errors.append(f"inclination_deg length ({len(inclinations)}) must match num_shells ({num_shells}).")

    num_rbs = net.get("num_resource_blocks", 8)
    if num_rbs < 1:
        errors.append("num_resource_blocks must be >= 1.")

    if net.get("beams_per_satellite", 2) < 1:
        errors.append("beams_per_satellite must be >= 1.")

    if traf.get("hotspot_cell_ratio", 0.25) < 0 or traf.get("hotspot_cell_ratio", 0.25) > 1:
        errors.append("hotspot_cell_ratio must be in [0, 1].")

    if orb.get("min_elevation_deg", 25.0) < 0 or orb.get("min_elevation_deg", 25.0) > 90:
        errors.append("min_elevation_deg must be in [0, 90].")

    return len(errors) == 0, errors


def get_scenario_description(preset_name: str) -> str:
    """Return description of a scenario preset."""
    return SCENARIO_PRESETS.get(preset_name, {}).get("description", "Custom configuration.")
