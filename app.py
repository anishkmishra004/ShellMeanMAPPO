"""
OrbitTrack | Autonomous Satellite Operations & Downlink Scheduling Platform
Clean, unified mission control interface powered by ShellMean-MAPPO.
"""
from __future__ import annotations
import os
import sys
import time
import datetime
import tempfile
from pathlib import Path

# Silence Matplotlib cache directory warning on Windows
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from services.scenario_service import load_base_config, apply_preset, SCENARIO_PRESETS
from services.simulation_service import SimulationSession, POLICY_NAMES, build_policy
from services.telemetry_service import TelemetryStore
from training.train import MAPPOTrainer
from training.checkpoint import load_checkpoint
from evaluation.evaluate import evaluate_policy
from visualization.constellation_view import create_3d_constellation_fig, create_2d_ground_track_fig
from visualization.performance_charts import (
    create_slot_dynamics_fig,
    create_baseline_comparison_fig,
    create_rf_link_fig,
)
from visualization.imagery_view import (
    render_photorealistic_earth_center_panel,
    render_live_imagery_section,
)

st.set_page_config(
    page_title="OrbitTrack | Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Aerospace Mission Control Stylesheet ──────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg-dark: #050811;
    --bg-panel: #0a0f1d;
    --bg-card: #0d1424;
    --border: #141f36;
    --accent-cyan: #00d4ff;
    --accent-green: #00ff88;
    --accent-blue: #0066ff;
    --accent-yellow: #ffaa00;
    --accent-red: #ff3366;
    --text-primary: #f0f4fc;
    --text-muted: #7388a8;
}

.stApp {
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] {
    background-color: #070b16 !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 10px;
}

/* Header Bar */
.mc-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #090e1c;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 18px;
    margin-bottom: 12px;
}
.mc-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.mc-brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.05em;
}
.mc-pill-online {
    background: rgba(0, 255, 136, 0.12);
    border: 1px solid var(--accent-green);
    color: var(--accent-green);
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Clean Panels */
.orbit-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.orbit-panel-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 8px;
}

/* Satellite Banner */
.sat-banner {
    background: #0d1527;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 18px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sat-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.04em;
}
.sat-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 2px;
}

/* Telemetry Key-Value Row */
.telem-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(20, 31, 54, 0.6);
    font-size: 0.82rem;
}
.telem-label {
    color: var(--text-muted);
}
.telem-val {
    color: var(--accent-cyan);
    font-family: monospace;
    font-weight: 600;
}
.telem-badge-locked {
    background: rgba(0, 255, 136, 0.15);
    color: var(--accent-green);
    border: 1px solid var(--accent-green);
    font-size: 0.68rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 3px;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(180deg, #0d3880 0%, #082354 100%) !important;
    border: 1px solid #1a4f9e !important;
    color: #ffffff !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 8px rgba(0, 212, 255, 0.3) !important;
}

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div {
    background-color: #0c1322 !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────────────
if "sim_session" not in st.session_state:
    base_cfg = load_base_config()
    st.session_state["config"] = base_cfg
    st.session_state["sim_session"] = SimulationSession(base_cfg, policy_name="Rule-Based Conflict-Aware", seed=42)
    st.session_state["sim_session"].initialize()

session: SimulationSession = st.session_state["sim_session"]


# ── Left Navigation Sidebar (Sleek, matching OrbitTrack image) ─────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
        <span style="font-size:1.8rem;">🌐</span>
        <div>
            <div style="font-size:1.15rem; font-weight:800; color:#fff; letter-spacing:0.04em;">OrbitTrack</div>
            <div style="font-size:0.65rem; color:#00d4ff; letter-spacing:0.12em; text-transform:uppercase;">SATELLITE OPERATIONS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_view = st.radio(
        "Navigation",
        options=[
            "📊 Dashboard",
            "🛰️ Satellites",
            "🌐 Tracking & Digital Twin",
            "📡 Telemetry & Link Budget",
            "⚔️ Conflict & Spectrum",
            "🧠 Live MAPPO Training",
            "🔀 Policy Benchmarking",
            "📑 Reports & Export",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem; color:#7388a8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;'>Mission Control Actions</p>", unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ Step 1", use_container_width=True):
            session.step()
            st.rerun()
    with col_btn2:
        if st.button("⏩ Step 10", width="stretch"):
            session.step_n(10)
            st.rerun()

    active_policy = st.selectbox(
        "Downlink Policy",
        options=POLICY_NAMES,
        index=POLICY_NAMES.index(session.policy_name) if session.policy_name in POLICY_NAMES else 0,
    )
    if active_policy != session.policy_name:
        session.set_policy(active_policy)
        st.rerun()

    active_preset = st.selectbox(
        "Scenario Preset",
        options=list(SCENARIO_PRESETS.keys()),
        index=0,
    )
    if st.button("🚀 Apply Preset & Run Scenario", width="stretch"):
        new_cfg = apply_preset(st.session_state["config"], active_preset)
        st.session_state["config"] = new_cfg
        new_session = SimulationSession(new_cfg, policy_name=session.policy_name, seed=session.seed)
        init_info = new_session.initialize()

        # Execute 3 initial slots immediately (Operation 3)
        initial_records = new_session.step_n(3)

        # Record the 3 operations report for the user's presentation
        net_cfg = new_cfg.get("network", {})
        orb_cfg = new_cfg.get("orbit", {})
        traf_cfg = new_cfg.get("traffic", {})
        total_backlog_init = float(np.sum(init_info.get("queues", [])))

        tot_served = sum(r.get("served_traffic_mb", 0.0) for r in initial_records)
        tot_retained = sum(r.get("num_retained", 0) for r in initial_records)
        tot_rejected = sum(r.get("num_rejected", 0) for r in initial_records)
        mean_sinr = np.mean([r.get("mean_sinr_db", 0.0) for r in initial_records]) if initial_records else 0.0

        st.session_state["preset_execution_report"] = {
            "preset_name": active_preset,
            "op1_constellation": f"{net_cfg.get('num_satellites', 8)} Sats across {net_cfg.get('num_shells', 2)} Shells ({orb_cfg.get('shell_altitudes_km', [550, 1100])} km)",
            "op2_demand": f"{net_cfg.get('num_ground_cells', 20)} Ground Cells | Initial Backlog: {total_backlog_init:.1f} Mb | Distribution: {traf_cfg.get('arrival_distribution', 'poisson').capitalize()}",
            "op3_execution": f"3 Initial Slots Executed | Delivered: {tot_served:.2f} Mb | Retained Beams: {tot_retained} | Contention Rejections: {tot_rejected} | Mean SINR: {mean_sinr:.1f} dB",
        }

        st.session_state["sim_session"] = new_session
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="background:#090e1c; border:1px solid #141f36; border-radius:6px; padding:10px 12px;">
        <div style="font-size:0.75rem; color:#00ff88; font-weight:700;">● Network Connected</div>
        <div style="font-size:0.68rem; color:#7388a8; margin-top:3px;">
            Ground Station: <b>Hartebeesthoek (HBX)</b><br>
            Coordinates: 25.891° S, 27.680° E<br>
            Constellation: 8 Sats (Shell 0 & 1)
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Top Bar (Matching Reference Image) ────────────────────────────────────────
utc_string = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class="mc-topbar">
    <div class="mc-brand">
        <span style="font-size:1.1rem; font-weight:700;">🛡️ Mission Control</span>
        <span class="mc-pill-online">● ONLINE</span>
        <span style="color:#7388a8; font-size:0.8rem; font-family:monospace; margin-left:10px;">{utc_string}</span>
    </div>
    <div style="display:flex; align-items:center; gap:18px;">
        <span style="font-size:0.8rem; color:#7388a8;">🌐 Global View</span>
        <span style="background:rgba(255, 51, 102, 0.15); color:#ff3366; border:1px solid #ff3366; font-size:0.72rem; padding:2px 8px; border-radius:4px; font-weight:700;">🔔 3 Alerts</span>
        <span style="font-size:0.82rem; color:#ffffff;">👤 <b>Alex Carter</b> <span style="color:#7388a8;">(Operator)</span></span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Helper Data Retrieval ─────────────────────────────────────────────────────
sat_snap = session.get_satellite_snapshot()
cell_snap = session.get_cell_snapshot()
last_tel = session.telemetry[-1] if session.telemetry else {}
active_beams = last_tel.get("beam_data", [])
df_tel = pd.DataFrame(session.telemetry)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 1: DASHBOARD (MAIN ORBITTRACK COCKPIT)
# ═════════════════════════════════════════════════════════════════════════════
if nav_view == "📊 Dashboard":
    # 0. Preset Execution 3-Operations Report (if recently triggered)
    if "preset_execution_report" in st.session_state:
        rep = st.session_state["preset_execution_report"]
        st.markdown(f"""
        <div style="background:#091224; border:1px solid var(--accent-cyan); border-radius:6px; padding:12px 18px; margin-bottom:14px; box-shadow:0 0 12px rgba(0,212,255,0.15);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:700; color:#00d4ff; font-size:0.95rem;">🎯 Scenario Successfully Applied & Initialized: {rep['preset_name']}</span>
                <span style="background:rgba(0,255,136,0.15); color:#00ff88; border:1px solid #00ff88; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:4px;">3 Operations Executed Live</span>
            </div>
            <div style="display:grid; grid-template-columns: 1fr 1fr 1.3fr; gap:12px; font-size:0.78rem;">
                <div style="background:#060b17; border:1px solid #14223d; border-radius:4px; padding:8px 10px;">
                    <b style="color:#00ff88;">1. Constellation Sizing:</b><br>
                    <span style="color:#cbd5e1;">{rep['op1_constellation']}</span>
                </div>
                <div style="background:#060b17; border:1px solid #14223d; border-radius:4px; padding:8px 10px;">
                    <b style="color:#ffaa00;">2. Ground Demand & Queues:</b><br>
                    <span style="color:#cbd5e1;">{rep['op2_demand']}</span>
                </div>
                <div style="background:#060b17; border:1px solid #14223d; border-radius:4px; padding:8px 10px;">
                    <b style="color:#00d4ff;">3. Downlink Execution:</b><br>
                    <span style="color:#cbd5e1;">{rep['op3_execution']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 1. Satellite Banner
    selected_sat = sat_snap[1] if len(sat_snap) > 1 else sat_snap[0]
    st.markdown(f"""
    <div class="sat-banner">
        <div style="display:flex; align-items:center; gap:14px;">
            <div style="font-size:2.2rem;">🛰️</div>
            <div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <span class="sat-title">ORBIT-{selected_sat['sat_id']+1:02d}</span>
                    <span class="mc-pill-online">OPERATIONAL</span>
                </div>
                <div class="sat-sub">Shell {selected_sat['shell_id']} LEO · Earth Observation & Downlink · {selected_sat['altitude_km']:.0f} km Altitude · 53.0° Inclination</div>
            </div>
        </div>
        <div style="display:flex; gap:24px; font-size:0.75rem; color:#7388a8;">
            <div>NORAD ID: <b style="color:#fff;">43218</b><br>COSPAR: <b style="color:#fff;">2022-056A</b></div>
            <div>Next Pass: <b style="color:#00ff88;">in 12m 43s</b><br>Duration: <b style="color:#fff;">10m 43s</b></div>
            <div>AOS: <b style="color:#00d4ff;">08:37:19 UTC</b><br>Max Elev: <b style="color:#fff;">78.3°</b></div>
            <div>Sub-Satellite Point:<br><b style="color:#00d4ff;">Lat {selected_sat['lat']:.3f}° N, Lon {selected_sat['lon']:.3f}° E</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Main Center + Right Column
    col_center, col_right = st.columns([7, 3])

    with col_center:
        c_head_left, c_head_right = st.columns([5, 5])
        with c_head_left:
            st.markdown("<div class='orbit-panel-title'>ORBITAL TRACKING & REAL-TIME 3D DIGITAL TWIN</div>", unsafe_allow_html=True)
        with c_head_right:
            view_mode = st.radio(
                "Display Engine",
                ["🌍 Photorealistic 3D Globe (OrbitTrack)", "🛰️ Plotly 3D Constellation Mesh", "🗺️ 2D Ground Track"],
                horizontal=True,
                label_visibility="collapsed",
            )

        if view_mode == "🌍 Photorealistic 3D Globe (OrbitTrack)":
            render_photorealistic_earth_center_panel(
                selected_sat=selected_sat,
                last_telemetry=last_tel,
                active_beams=active_beams,
                satellites=sat_snap,
                cells=cell_snap,
            )
        elif view_mode == "🛰️ Plotly 3D Constellation Mesh":
            fig_3d = create_3d_constellation_fig(
                satellites=sat_snap,
                cells=cell_snap,
                active_beams=active_beams,
                show_visibility=True,
                show_beams=True,
            )
            st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
        else:
            fig_2d = create_2d_ground_track_fig(
                satellites=sat_snap,
                cells=cell_snap,
                active_beams=active_beams,
            )
            st.plotly_chart(fig_2d, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown("<div class='orbit-panel-title'>PAYLOAD & TELEMETRY SUBSYSTEMS</div>", unsafe_allow_html=True)
        tab_t1, tab_t2, tab_t3, tab_t4 = st.tabs(["Telemetry", "Attitude", "Power", "Comm"])

        with tab_t1:
            st.markdown(f"""
            <div class="telem-row"><span class="telem-label">Altitude (mean)</span><span class="telem-val">{selected_sat['altitude_km']:.1f} km</span></div>
            <div class="telem-row"><span class="telem-label">Velocity</span><span class="telem-val">7.61 km/s</span></div>
            <div class="telem-row"><span class="telem-label">Inclination</span><span class="telem-val">53.0°</span></div>
            <div class="telem-row"><span class="telem-label">Eccentricity</span><span class="telem-val">0.0012</span></div>
            <div class="telem-row"><span class="telem-label">Period</span><span class="telem-val">95.4 min</span></div>
            <div class="telem-row"><span class="telem-label">Latitude</span><span class="telem-val">{selected_sat['lat']:.3f}° N</span></div>
            <div class="telem-row"><span class="telem-label">Longitude</span><span class="telem-val">{selected_sat['lon']:.3f}° E</span></div>
            <div class="telem-row"><span class="telem-label">Ground Track Rate</span><span class="telem-val">7.42 km/s</span></div>
            <div class="telem-row"><span class="telem-label">Visible Cells</span><span class="telem-val">{len(selected_sat.get('visible_cells', []))} cells in LOS</span></div>
            """, unsafe_allow_html=True)

        with tab_t2:
            st.markdown("""
            <div class="telem-row"><span class="telem-label">Roll (ECI)</span><span class="telem-val">0.12°</span></div>
            <div class="telem-row"><span class="telem-label">Pitch</span><span class="telem-val">-0.08°</span></div>
            <div class="telem-row"><span class="telem-label">Yaw</span><span class="telem-val">0.03°</span></div>
            <div class="telem-row"><span class="telem-label">Angular Rate (Roll)</span><span class="telem-val">0.0002 °/s</span></div>
            <div class="telem-row"><span class="telem-label">Angular Rate (Pitch)</span><span class="telem-val">-0.0001 °/s</span></div>
            <div class="telem-row"><span class="telem-label">Angular Rate (Yaw)</span><span class="telem-val">0.0003 °/s</span></div>
            """, unsafe_allow_html=True)

        with tab_t3:
            st.markdown("""
            <div class="telem-row"><span class="telem-label">Solar Array Output</span><span class="telem-val" style="color:#00ff88;">98%</span></div>
            <div class="telem-row"><span class="telem-label">Battery Level</span><span class="telem-val" style="color:#00ff88;">86%</span></div>
            <div class="telem-row"><span class="telem-label">Bus Voltage</span><span class="telem-val">28.4 V</span></div>
            <div class="telem-row"><span class="telem-label">Bus Current</span><span class="telem-val">4.2 A</span></div>
            <div class="telem-row"><span class="telem-label">Payload Thermal</span><span class="telem-val">+18.5 °C (Nominal)</span></div>
            """, unsafe_allow_html=True)

        with tab_t4:
            st.markdown(f"""
            <div class="telem-row"><span class="telem-label">Uplink (S-band)</span><span class="telem-val">8.402 GHz <span class="telem-badge-locked">LOCKED</span></span></div>
            <div class="telem-row"><span class="telem-label">Downlink (Ka-band)</span><span class="telem-val">20.0 GHz <span class="telem-badge-locked">LOCKED</span></span></div>
            <div class="telem-row"><span class="telem-label">Signal Strength</span><span class="telem-val">45.2 dBHz</span></div>
            <div class="telem-row"><span class="telem-label">Mean Downlink SINR</span><span class="telem-val">{last_tel.get('mean_sinr_db', 16.4):.1f} dB</span></div>
            <div class="telem-row"><span class="telem-label">Delivered This Slot</span><span class="telem-val">{last_tel.get('served_traffic_mb', 0.0):.2f} Mb</span></div>
            <div class="telem-row"><span class="telem-label">Active Transmissions</span><span class="telem-val">{last_tel.get('num_retained', 0)} Spot Beams</span></div>
            """, unsafe_allow_html=True)

    # 3. Bottom 3-Card Deck (Exact Match to Aerospace Console Screenshot)
    st.markdown("---")
    b_col1, b_col2, b_col3 = st.columns(3)

    with b_col1:
        st.markdown("<div class='orbit-panel-title'>UPCOMING PASSES & EPHEMERIS</div>", unsafe_allow_html=True)
        passes_df = pd.DataFrame([
            {"AOS (UTC)": "08:37:19", "LOS": "08:48:02", "Duration": "10m 43s", "Max Elev": "78.3°", "Status": "● Upcoming"},
            {"AOS (UTC)": "10:12:06", "LOS": "10:22:41", "Duration": "10m 35s", "Max Elev": "64.7°", "Status": "● Upcoming"},
            {"AOS (UTC)": "12:45:33", "LOS": "12:59:17", "Duration": "13m 44s", "Max Elev": "82.1°", "Status": "● Scheduled"},
            {"AOS (UTC)": "15:20:11", "LOS": "15:31:02", "Duration": "10m 51s", "Max Elev": "70.4°", "Status": "● Scheduled"},
        ])
        st.dataframe(passes_df, hide_index=True, use_container_width=True)
        st.markdown("""
        <div style="font-size:0.75rem; color:#7388a8; margin-top:6px;">
            Semi-major axis: <b style="color:#fff;">6,878.1 km</b> | Apogee: <b style="color:#fff;">548.7 km</b> | Perigee: <b style="color:#fff;">536.4 km</b> | Period: <b style="color:#fff;">94.7 min</b>
        </div>
        """, unsafe_allow_html=True)

    with b_col2:
        st.markdown("<div class='orbit-panel-title'>LIVE IMAGERY & SATELLITE CAPTURES</div>", unsafe_allow_html=True)
        render_live_imagery_section(selected_sat, last_tel, active_beams, getattr(session, "current_step", 0))

    with b_col3:
        st.markdown("<div class='orbit-panel-title'>MISSION STATUS & RECENT EVENTS</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; justify-content:space-between; font-size:0.71rem; margin-bottom:8px; border-bottom:1px solid #141f36; padding-bottom:6px;">
            <span style="color:#00ff88;">✓ Acquisition</span>
            <span style="color:#00ff88;">✓ Tracking</span>
            <span style="color:#00d4ff;">● Imaging Active</span>
            <span style="color:#7388a8;">○ Downlink</span>
        </div>
        <div style="font-size:0.72rem; color:#cbd5e1; margin-bottom:6px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span>Optical Camera (PAN) 1.2m</span> <span style="color:#00ff88; font-weight:700;">ACTIVE</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span>Multispectral (MS) 4.0m</span> <span style="color:#00ff88; font-weight:700;">ACTIVE</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Synthetic Aperture Radar</span> <span style="color:#7388a8;">STANDBY</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        log_box = st.container(height=110)
        with log_box:
            if session.event_log:
                for entry in reversed(session.event_log[-6:]):
                    st.markdown(f"<div style='font-size:0.72rem; color:#cbd5e1; font-family:monospace; margin-bottom:2px;'>{entry}</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='font-size:0.72rem; color:#7388a8; font-family:monospace;'>
                08:21:14 Image capture completed (Mission: EO-2025-0417)<br>
                08:19:03 Attitude adjustment (ΔV: 0.002 m/s)<br>
                08:15:22 Downlink session started (20.0 GHz Ka-band)
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 2: LIVE MAPPO TRAINING (SOLVES USER ISSUE: "no records found")
# ═════════════════════════════════════════════════════════════════════════════
elif nav_view == "🧠 Live MAPPO Training":
    st.markdown("### 🧠 Live MAPPO Policy Training & Convergence Monitoring")
    st.markdown("<p style='color:#7388a8;'>Multi-Agent Proximal Policy Optimization with GAE Advantage Estimation & Autoregressive Allocation</p>", unsafe_allow_html=True)

    log_path = os.path.join(ROOT_DIR, "results", "training_log.csv")
    chk_path = os.path.join(ROOT_DIR, "checkpoints", "best_model.pt")

    # Interactive Training Trigger
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 3])
    with ctrl_col1:
        slots_to_train = st.selectbox("Training Batch Slots", options=[128, 256, 512, 1024], index=0)
    with ctrl_col2:
        if st.button("🚀 Run Live Training Iteration", type="primary", width="stretch"):
            with st.spinner(f"Executing MAPPO training ({slots_to_train} slots with rollout & PPO update)..."):
                trainer = MAPPOTrainer(session.config)
                df_new = trainer.train(
                    total_slots=slots_to_train,
                    eval_interval=128,
                    checkpoint_path=chk_path,
                    use_counterfactual_credit=session.config.get("rewards", {}).get("use_counterfactual_credit", True),
                )
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                if os.path.exists(log_path):
                    df_existing = pd.read_csv(log_path)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    df_combined = df_new
                df_combined.to_csv(log_path, index=False)
                st.success(f"Trained {slots_to_train} slots! Best model updated.")
                st.rerun()

    # Load and Render Records
    if os.path.exists(log_path):
        df_train = pd.read_csv(log_path)
        st.markdown(f"**Training Log:** `{len(df_train)} iterations recorded` | Checkpoint: `{chk_path}`")

        # 4-Panel Training Convergence Chart
        fig_train = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Team Rollout Reward", "Actor & Critic Losses", "Policy Entropy", "Evaluation Throughput (Mbps)"),
            vertical_spacing=0.18,
            horizontal_spacing=0.10,
        )

        iters = df_train["iteration"] if "iteration" in df_train else list(range(1, len(df_train) + 1))

        # Reward
        fig_train.add_trace(go.Scatter(x=iters, y=df_train["rollout_reward"], mode="lines+markers", name="Rollout Reward", line=dict(color="#00ff88")), row=1, col=1)
        if "eval_reward" in df_train and df_train["eval_reward"].notna().any():
            ev = df_train.dropna(subset=["eval_reward"])
            fig_train.add_trace(go.Scatter(x=ev["iteration"], y=ev["eval_reward"], mode="markers", name="Eval Reward", marker=dict(size=8, color="#00d4ff")), row=1, col=1)

        # Losses
        fig_train.add_trace(go.Scatter(x=iters, y=df_train["actor_loss"], mode="lines", name="Actor Loss", line=dict(color="#ffaa00")), row=1, col=2)
        fig_train.add_trace(go.Scatter(x=iters, y=df_train["critic_loss"], mode="lines", name="Critic Loss", line=dict(color="#ff4466")), row=1, col=2)

        # Entropy
        fig_train.add_trace(go.Scatter(x=iters, y=df_train["entropy"], mode="lines", name="Entropy", line=dict(color="#d946ef")), row=2, col=1)

        # Throughput
        if "eval_throughput" in df_train and df_train["eval_throughput"].notna().any():
            ev_tp = df_train.dropna(subset=["eval_throughput"])
            fig_train.add_trace(go.Scatter(x=ev_tp["iteration"], y=ev_tp["eval_throughput"], mode="lines+markers", name="Eval Throughput", line=dict(color="#00d4ff", width=2)), row=2, col=2)

        fig_train.update_layout(
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0d1117",
            font=dict(color="#e8eaf0"),
            height=460,
        )
        for r in [1, 2]:
            for c in [1, 2]:
                fig_train.update_xaxes(gridcolor="#141f36", title_text="Iteration", row=r, col=c)
                fig_train.update_yaxes(gridcolor="#141f36", row=r, col=c)

        st.plotly_chart(fig_train, use_container_width=True)
        st.dataframe(df_train.tail(8), use_container_width=True, hide_index=True)
    else:
        st.info("No training records file found yet. Click 'Run Live Training Iteration' above to run the MAPPO trainer.")


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 3: TRACKING & DIGITAL TWIN
# ═════════════════════════════════════════════════════════════════════════════
elif nav_view == "🌐 Tracking & Digital Twin":
    st.markdown("### 🌐 Global Constellation Tracking & Ground Track")
    t_tab1, t_tab2, t_tab3 = st.tabs([
        "🌍 Photorealistic 3D Globe (Interactive Instances)",
        "🛰️ 3D Constellation Mesh (ECEF)",
        "🗺️ 2D Equirectangular Ground Track",
    ])
    with t_tab1:
        render_photorealistic_earth_center_panel(selected_sat, last_tel, active_beams, sat_snap, cell_snap)
    with t_tab2:
        fig_3d = create_3d_constellation_fig(sat_snap, cell_snap, active_beams, show_visibility=True, show_beams=True)
        st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
    with t_tab3:
        fig_2d = create_2d_ground_track_fig(sat_snap, cell_snap, active_beams)
        st.plotly_chart(fig_2d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 4: SATELLITES INSPECTOR
# ═════════════════════════════════════════════════════════════════════════════
elif nav_view == "🛰️ Satellites":
    st.markdown("### 🛰️ Multi-Shell Constellation Fleet Inspector")
    sat_list = []
    for s in sat_snap:
        sat_list.append({
            "Sat ID": f"S{s['sat_id']:02d}",
            "Shell": f"Shell {s['shell_id']}",
            "Altitude (km)": f"{s['altitude_km']:.0f}",
            "Sub-Sat Lat": f"{s['lat']:.2f}° N",
            "Sub-Sat Lon": f"{s['lon']:.2f}° E",
            "LOS Visible Cells": len(s.get("visible_cells", [])),
            "Operational": "OPERATIONAL" if s.get("is_operational", True) else "OFFLINE",
        })
    st.dataframe(pd.DataFrame(sat_list), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 5: TELEMETRY & LINK BUDGET
# ═════════════════════════════════════════════════════════════════════════════
elif nav_view == "📡 Telemetry & Link Budget":
    st.markdown("### 📡 Physical RF Downlink Budget & Shannon Capacity")
    if not df_tel.empty:
        fig_dyn = create_slot_dynamics_fig(df_tel)
        st.plotly_chart(fig_dyn, use_container_width=True)
        fig_rf = create_rf_link_fig(active_beams)
        st.plotly_chart(fig_rf, use_container_width=True)
    else:
        st.info("Execute slots using 'Step 1' in the sidebar to populate physical telemetry.")


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 6: CONFLICT & SPECTRUM
# ═════════════════════════════════════════════════════════════════════════════
elif nav_view == "⚔️ Conflict & Spectrum":
    st.markdown("### ⚔️ Deterministic Conflict Resolution Inspector")
    outcome = session.env.last_resolution_outcome
    if outcome:
        st.markdown(f"**Retained Requests:** `{len(outcome.retained_requests)}` | **Rejected Collisions:** `{len(outcome.rejected_requests)}`")
        if outcome.logs:
            st.code("\n".join(outcome.logs[:12]))
    else:
        st.info("Execute a slot to view resolver decisions.")


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 7: POLICY BENCHMARKING
# ═════════════════════════════════════════════════════════════════════════════
elif nav_view == "🔀 Policy Benchmarking":
    st.markdown("### 🔀 Multi-Algorithm Benchmark Comparison")
    st.markdown("<p style='color:#7388a8;'>Empirical comparison across identical initial environments, orbital trajectories, and evaluation seeds.</p>", unsafe_allow_html=True)

    if st.button("🚀 Run Controlled Multi-Seed Benchmark (Seeds: 42, 101)", type="primary"):
        with st.spinner("Evaluating algorithms across seeds..."):
            summary_list = []
            for p in POLICY_NAMES:
                pol = build_policy(p, session.config, seed=42)
                res = evaluate_policy(session.env, pol, p, seeds=[42, 101])
                means = res.get("means", {})
                deliv = float(means.get("total_served_mb", means.get("total_delivered_mb", 0.0)))
                summary_list.append({
                    "Algorithm": p,
                    "Delivered Traffic (Mb)": deliv,
                    "Throughput (Mbps)": float(means.get("mean_throughput_mbps", 0.0)),
                    "Retention Rate (%)": float(means.get("mean_retention_rate", 0.0)),
                    "Jain's Fairness": float(means.get("mean_jains_fairness", 0.0)),
                    "Terminal Backlog (Mb)": float(means.get("final_backlog_mb", 0.0)),
                    "Mean SINR (dB)": float(means.get("mean_sinr_db", 0.0)),
                    "total_delivered_mb": deliv,
                    "mean_throughput_mbps": float(means.get("mean_throughput_mbps", 0.0)),
                    "mean_retention_rate": float(means.get("mean_retention_rate", 0.0)),
                    "mean_jains_fairness": float(means.get("mean_jains_fairness", 0.0)),
                })
            st.session_state["benchmark_df"] = pd.DataFrame(summary_list)

    if "benchmark_df" in st.session_state:
        df_bench = st.session_state["benchmark_df"]
        display_cols = ["Algorithm", "Delivered Traffic (Mb)", "Throughput (Mbps)", "Retention Rate (%)", "Jain's Fairness", "Terminal Backlog (Mb)", "Mean SINR (dB)"]
        st.dataframe(df_bench[display_cols], use_container_width=True, hide_index=True)
        fig_b = create_baseline_comparison_fig(df_bench)
        st.plotly_chart(fig_b, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 8: REPORTS & EXPORT
# ═════════════════════════════════════════════════════════════════════════════
elif nav_view == "📑 Reports & Export":
    st.markdown("### 📑 Scientific Experiment Report & Telemetry Export")
    store = TelemetryStore()
    for rec in session.telemetry:
        store.add(rec)
    rep = store.generate_report(session.config, session.policy_name)
    st.download_button("📥 Download Full Report (.md)", data=rep, file_name="shellmean_mappo_report.md", mime="text/markdown", type="primary")
    with st.expander("Preview Generated Report", expanded=True):
        st.markdown(rep)
