"""Comprehensive integration tests for platform services, telemetry, and visualization."""
import os
import pytest
import numpy as np
import pandas as pd
import yaml

from services.scenario_service import (
    load_base_config,
    apply_preset,
    build_config_from_ui,
    validate_config,
    SCENARIO_PRESETS,
)
from services.simulation_service import (
    SimulationSession,
    build_policy,
    POLICY_NAMES,
    _ecef_to_latlon,
)
from services.telemetry_service import TelemetryStore
from services.training_service import TrainingManager, TrainingState, get_training_manager
from visualization.constellation_view import create_3d_constellation_fig, create_2d_ground_track_fig
from visualization.performance_charts import (
    create_slot_dynamics_fig,
    create_baseline_comparison_fig,
    create_traffic_heatmap_fig,
    create_rf_link_fig,
)


def test_scenario_presets_and_validation():
    base_cfg = load_base_config()
    assert "network" in base_cfg
    assert "orbit" in base_cfg
    assert "traffic" in base_cfg

    # Test all presets apply cleanly
    for preset_name in SCENARIO_PRESETS:
        preset_cfg = apply_preset(base_cfg, preset_name)
        is_valid, errors = validate_config(preset_cfg)
        assert is_valid, f"Preset '{preset_name}' validation failed: {errors}"

    # Test invalid config detection
    invalid_cfg = dict(base_cfg)
    invalid_cfg["network"] = dict(base_cfg["network"])
    invalid_cfg["network"]["num_shells"] = 3
    invalid_cfg["network"]["satellites_per_shell"] = [4, 4]  # Mismatch: length 2 != 3
    is_valid, errors = validate_config(invalid_cfg)
    assert not is_valid
    assert any("satellites_per_shell length" in e for e in errors)


def test_ecef_to_latlon():
    # North pole: [0, 0, 6371] -> lat=90, lon=0
    lat, lon = _ecef_to_latlon(np.array([0.0, 0.0, 6371.0]))
    assert np.isclose(lat, 90.0, atol=1e-2)

    # Equator on Greenwich: [6371, 0, 0] -> lat=0, lon=0
    lat, lon = _ecef_to_latlon(np.array([6371.0, 0.0, 0.0]))
    assert np.isclose(lat, 0.0, atol=1e-2)
    assert np.isclose(lon, 0.0, atol=1e-2)


def test_simulation_session_lifecycle():
    cfg = load_base_config()
    cfg["network"]["episode_length"] = 5

    session = SimulationSession(cfg, policy_name="Rule-Based Conflict-Aware", seed=42)
    init_info = session.initialize()

    assert session.is_initialized
    assert not session.is_done
    assert session.current_step == 0
    assert "queues" in init_info
    assert len(session.event_log) > 0

    # Execute one step
    tel = session.step()
    assert tel is not None
    assert session.current_step == 1
    assert "served_traffic_mb" in tel
    assert "num_retained" in tel
    assert "num_rejected" in tel
    assert "beam_data" in tel
    assert "satellites" in tel
    assert "cells" in tel

    # Snapshots
    sat_snap = session.get_satellite_snapshot()
    cell_snap = session.get_cell_snapshot()
    assert len(sat_snap) == session.env.num_satellites
    assert len(cell_snap) == session.env.num_ground_cells

    # Step remaining slots until completion
    session.step_n(10)
    assert session.is_done
    assert len(session.telemetry) == 5


def test_all_policy_instantiation():
    cfg = load_base_config()
    for pname in POLICY_NAMES:
        policy = build_policy(pname, cfg, seed=42)
        assert policy is not None


def test_telemetry_store_and_export():
    store = TelemetryStore()
    cfg = load_base_config()

    sample_rec = {
        "slot": 1,
        "team_reward": 5.2,
        "served_traffic_mb": 12.5,
        "arrivals_mb": 10.0,
        "remaining_backlog_mb": 35.0,
        "num_retained": 14,
        "num_rejected": 2,
        "interference_watts": 1.2e-12,
        "jains_fairness": 0.88,
        "switching_count": 1,
        "mean_sinr_db": 15.4,
        "policy_time_ms": 3.2,
    }
    store.add(sample_rec)

    df = store.to_dataframe()
    assert len(df) == 1
    assert df["served_traffic_mb"].iloc[0] == 12.5

    csv_str = store.export_csv()
    assert "served_traffic_mb" in csv_str

    yaml_str = store.export_config_yaml(cfg)
    assert "network:" in yaml_str

    report_md = store.generate_report(cfg, policy_name="Rule-Based Conflict-Aware")
    assert "# ShellMean-MAPPO Scientific Experiment Report" in report_md
    assert "Total Delivered Data" in report_md


def test_visualization_generators():
    cfg = load_base_config()
    session = SimulationSession(cfg, policy_name="Rule-Based Conflict-Aware", seed=42)
    session.initialize()
    tel = session.step()

    sat_snap = session.get_satellite_snapshot()
    cell_snap = session.get_cell_snapshot()
    beam_data = tel["beam_data"]

    # 3D Constellation
    fig_3d = create_3d_constellation_fig(sat_snap, cell_snap, active_beams=beam_data)
    assert fig_3d is not None

    # 2D Ground Track
    fig_2d = create_2d_ground_track_fig(sat_snap, cell_snap, active_beams=beam_data)
    assert fig_2d is not None

    # Performance Dynamics
    df_tel = pd.DataFrame([tel])
    fig_dyn = create_slot_dynamics_fig(df_tel)
    assert fig_dyn is not None

    # Traffic Heatmap
    q_mat = np.random.uniform(5, 20, size=(5, 20))
    fig_heat = create_traffic_heatmap_fig(q_mat)
    assert fig_heat is not None

    # RF Link Fig
    fig_rf = create_rf_link_fig(beam_data)
    assert fig_rf is not None


def test_training_manager_interface():
    mgr = get_training_manager()
    status = mgr.get_status()
    assert "state" in status
    assert "slots_trained" in status
    df_hist = mgr.get_history_df()
    assert isinstance(df_hist, pd.DataFrame)

