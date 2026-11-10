"""
Imagery & Photorealistic Earth Digital Twin Module for OrbitTrack Mission Control.
Provides high-resolution satellite imagery feeds, target terrain inspection,
and a true interactive 3D WebGL Earth Globe Digital Twin in Three.js showing real instances.
"""
from __future__ import annotations
import os
import json
import base64
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).parent.parent.resolve()
ASSETS_DIR = ROOT_DIR / "assets"

_CACHE: Dict[str, str] = {}


def get_asset_text(filename: str) -> str:
    """Read a text asset and cache in memory."""
    if filename not in _CACHE:
        filepath = ASSETS_DIR / filename
        if filepath.exists():
            _CACHE[filename] = filepath.read_text(encoding="utf-8")
        else:
            _CACHE[filename] = ""
    return _CACHE[filename]


def get_asset_base64(filename: str) -> str:
    """Read a binary asset, return base64 data URI, and cache in memory."""
    key = f"b64_{filename}"
    if key not in _CACHE:
        filepath = ASSETS_DIR / filename
        if filepath.exists():
            with open(filepath, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = filepath.suffix.lower().lstrip(".")
                mime = "jpeg" if ext in ("jpg", "jpeg") else ("png" if ext == "png" else ext)
                _CACHE[key] = f"data:image/{mime};base64,{encoded}"
        else:
            _CACHE[key] = ""
    return _CACHE[key]


def render_live_imagery_section(
    selected_sat: Dict[str, Any],
    last_telemetry: Dict[str, Any],
    active_beams: List[Dict[str, Any]],
    current_slot: int = 0,
):
    """
    Renders the middle bottom card: 'Live Imagery | Recent Images | Analysis'
    matching the OrbitTrack aerospace mission control console.
    """
    img_tab1, img_tab2, img_tab3 = st.tabs([
        "📸 Live Imagery",
        "🖼️ Recent Images",
        "📡 Downlink Spectrum & Analysis",
    ])

    # 1. LIVE IMAGERY TAB
    with img_tab1:
        col_sel, col_stat = st.columns([6, 4])
        with col_sel:
            target_cell_options = {
                "Cell C02 (Coastal Maritime Gateway & Port)": "live_imagery_terrain_sharp.jpg",
                "Cell C01 (Urban Metropolitan Hub)": "recent_urban_metro.jpg",
                "Cell C04 (Sahara Desert Gateway Station)": "recent_desert_terminal.jpg",
                "Cell C06 (Alpine Mountain Telemetry Gateway)": "recent_mountain_ridge.jpg",
            }
            selected_target = st.selectbox(
                "Ground Target LOS Footprint",
                options=list(target_cell_options.keys()),
                index=0,
                label_visibility="collapsed",
            )
        with col_stat:
            st.markdown(
                textwrap.dedent("""
                <div style="font-size:0.75rem; color:#00ff88; text-align:right; padding-top:6px; font-weight:700;">
                ● DOWNLINK BEAM #02 LOCKED (20.0 GHz)
                </div>
                """).strip(),
                unsafe_allow_html=True,
            )

        img_filename = target_cell_options[selected_target]
        b64_img = get_asset_base64(img_filename)

        base_seconds = 14 + current_slot * 3
        sec_str = f"{base_seconds % 60:02d}"
        min_str = f"{(21 + base_seconds // 60) % 60:02d}"
        capture_time = f"08:{min_str}:{sec_str} UTC"

        if b64_img:
            target_label = selected_target.split(' ')[1]
            card_html = textwrap.dedent(f"""
            <div style="background:#060a15; border:1px solid #14223d; border-radius:6px; overflow:hidden; position:relative; margin-bottom:8px;">
                <img src="{b64_img}" style="width:100%; height:200px; object-fit:cover; display:block;" alt="Satellite Earth Capture">
                <div style="position:absolute; top:8px; left:8px; background:rgba(6,10,21,0.85); border:1px solid rgba(0,212,255,0.4); border-radius:3px; padding:2px 7px; font-size:0.7rem; color:#00d4ff; font-family:monospace;">
                    TARGET: {target_label} | FOOTPRINT: 12.8 km²
                </div>
                <div style="position:absolute; top:8px; right:8px; background:rgba(0,255,136,0.2); border:1px solid #00ff88; border-radius:3px; padding:2px 7px; font-size:0.68rem; color:#00ff88; font-weight:700;">
                    LIVE DOWNLINK
                </div>
                <div style="background:#090f1f; border-top:1px solid #14223d; padding:6px 12px; display:flex; justify-content:space-between; align-items:center; font-size:0.74rem;">
                    <div><span style="color:#7388a8;">Captured:</span> <b style="color:#ffffff; font-family:monospace;">{capture_time}</b></div>
                    <div><span style="color:#7388a8;">Resolution:</span> <b style="color:#ffffff;">1.2 m/px</b></div>
                    <div><span style="color:#7388a8;">Swath:</span> <b style="color:#ffffff;">12.8 km</b></div>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="background:rgba(0,102,255,0.25); border:1px solid #0066ff; color:#00d4ff; font-weight:700; font-size:0.68rem; padding:1px 6px; border-radius:3px;">RGB</span>
                        <span style="color:#7388a8; font-size:0.85rem;" title="Full Field of View">⛶</span>
                    </div>
                </div>
            </div>
            """).strip()
            st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.warning("Satellite imagery asset loading...")

        deliv_mb = last_telemetry.get("served_traffic_mb", 18.4)
        sinr = last_telemetry.get("mean_sinr_db", 17.8)
        meta_html = textwrap.dedent(f"""
        <div style="font-size:0.73rem; color:#7388a8; display:flex; justify-content:space-between; margin-top:2px;">
            <span>Sensor: <b style="color:#cbd5e1;">HRMI-2 Optical Multi-Spectral</b></span>
            <span>Downlink: <b style="color:#00ff88;">Ka-Band 20.0 GHz</b></span>
            <span>Capacity: <b style="color:#00d4ff;">{deliv_mb*10:.1f} Mbps (SINR: {sinr:.1f} dB)</b></span>
        </div>
        """).strip()
        st.markdown(meta_html, unsafe_allow_html=True)

    # 2. RECENT IMAGES TAB
    with img_tab2:
        st.caption("Recent Earth Observation Passes & Downlinked Terrain Captures:")
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            b64_sf = get_asset_base64("live_imagery_terrain_sharp.jpg")
            h1 = textwrap.dedent(f"""
            <div style="background:#090f1f; border:1px solid #14223d; border-radius:5px; padding:6px; margin-bottom:8px;">
                <img src="{b64_sf}" style="width:100%; height:90px; object-fit:cover; border-radius:3px; display:block;">
                <div style="font-size:0.72rem; font-weight:700; color:#ffffff; margin-top:4px;">San Francisco Bay Corridor</div>
                <div style="font-size:0.66rem; color:#7388a8;">Pass 08:21 UTC · 1.2 m/px · Cloud 4%</div>
            </div>
            """).strip()
            st.markdown(h1, unsafe_allow_html=True)

            b64_des = get_asset_base64("recent_desert_terminal.jpg")
            h2 = textwrap.dedent(f"""
            <div style="background:#090f1f; border:1px solid #14223d; border-radius:5px; padding:6px;">
                <img src="{b64_des}" style="width:100%; height:90px; object-fit:cover; border-radius:3px; display:block;">
                <div style="font-size:0.72rem; font-weight:700; color:#ffffff; margin-top:4px;">Sahara Ground Gateway C04</div>
                <div style="font-size:0.66rem; color:#7388a8;">Pass 05:12 UTC · 2.0 m/px · Cloud 0%</div>
            </div>
            """).strip()
            st.markdown(h2, unsafe_allow_html=True)

        with r_col2:
            b64_urb = get_asset_base64("recent_urban_metro.jpg")
            h3 = textwrap.dedent(f"""
            <div style="background:#090f1f; border:1px solid #14223d; border-radius:5px; padding:6px; margin-bottom:8px;">
                <img src="{b64_urb}" style="width:100%; height:90px; object-fit:cover; border-radius:3px; display:block;">
                <div style="font-size:0.72rem; font-weight:700; color:#ffffff; margin-top:4px;">Tokyo Metropolitan Basin C01</div>
                <div style="font-size:0.66rem; color:#7388a8;">Pass 06:45 UTC · 0.8 m/px · Cloud 8%</div>
            </div>
            """).strip()
            st.markdown(h3, unsafe_allow_html=True)

            b64_mnt = get_asset_base64("recent_mountain_ridge.jpg")
            h4 = textwrap.dedent(f"""
            <div style="background:#090f1f; border:1px solid #14223d; border-radius:5px; padding:6px;">
                <img src="{b64_mnt}" style="width:100%; height:90px; object-fit:cover; border-radius:3px; display:block;">
                <div style="font-size:0.72rem; font-weight:700; color:#ffffff; margin-top:4px;">Alpine Relay Station C06</div>
                <div style="font-size:0.66rem; color:#7388a8;">Pass 03:38 UTC · 1.5 m/px · Cloud 12%</div>
            </div>
            """).strip()
            st.markdown(h4, unsafe_allow_html=True)

    # 3. ANALYSIS & SPECTRUM TAB
    with img_tab3:
        if active_beams:
            import pandas as pd
            df_ab = pd.DataFrame(active_beams)
            cols = [c for c in ["sat_id", "beam_id", "cell_id", "rate_mbps", "sinr_db", "tx_power_w"] if c in df_ab.columns]
            st.dataframe(df_ab[cols].head(5), hide_index=True, width="stretch")
        else:
            st.info("Execute simulation slot via 'Step 1' in the sidebar to view active beam allocations.")

        s_html = textwrap.dedent(f"""
        <div style="font-size:0.75rem; color:#7388a8; margin-top:8px; line-height:1.6;">
            • Carrier: <b style="color:#ffffff;">20.0 GHz Ka-band</b> | Bandwidth: <b style="color:#ffffff;">200 MHz</b><br>
            • Resource Blocks: <b style="color:#00d4ff;">8 RBs @ 25 MHz/block</b><br>
            • Total Remaining Queue Backlog: <b style="color:#00ff88;">{last_telemetry.get('remaining_backlog_mb', 0.0):.1f} Mb</b>
        </div>
        """).strip()
        st.markdown(s_html, unsafe_allow_html=True)


def render_photorealistic_earth_center_panel(
    selected_sat: Dict[str, Any],
    last_telemetry: Dict[str, Any],
    active_beams: List[Dict[str, Any]],
    satellites: Optional[List[Dict[str, Any]]] = None,
    cells: Optional[List[Dict[str, Any]]] = None,
):
    """
    Renders a true interactive 3D WebGL Earth Globe Digital Twin in Three.js
    showing real constellation instances:
      - 3D NASA Blue Marble textured spherical Earth with rotating clouds & atmospheric rim.
      - 3D Satellite instances in orbit with deployed solar array wings.
      - 3D Ground station instances on the Earth surface (green normal, red hotspots).
      - Active 3D Laser downlink beam instances connecting satellites to ground stations.
      - 3D Inclined orbit trajectory rings for Shell 0 and Shell 1.
      - Full 360-degree click-and-drag rotation, mouse scroll zoom, and pan controls.
    """
    # 1. Prepare Base64 Textures and Libraries
    three_code = get_asset_text("three.min.js")
    controls_code = get_asset_text("OrbitControls.js")
    earth_b64 = get_asset_base64("earth_surface_texture.jpg")
    clouds_b64 = get_asset_base64("earth_clouds_texture.jpg")

    sat_id = selected_sat.get("sat_id", 0)
    alt = selected_sat.get("altitude_km", 542.6)
    lat = selected_sat.get("lat", 12.634)
    lon = selected_sat.get("lon", 45.218)
    incl = selected_sat.get("inclination", 53.0)
    active_beam_cnt = len(active_beams) if active_beams else 0

    # Ensure satellite and cell lists are valid
    sats_data = satellites if satellites else [selected_sat]
    cells_data = cells if cells else []

    instances_payload = {
        "satellites": [
            {
                "id": s.get("sat_id", 0),
                "shell_id": s.get("shell_id", 0),
                "alt": float(s.get("altitude_km", 550.0)),
                "lat": float(s.get("lat", 0.0)),
                "lon": float(s.get("lon", 0.0)),
                "visible_count": len(s.get("visible_cells", [])),
            }
            for s in sats_data
        ],
        "cells": [
            {
                "id": c.get("cell_id", 0),
                "lat": float(c.get("lat", 0.0)),
                "lon": float(c.get("lon", 0.0)),
                "queue": float(c.get("queue_mb", 0.0)),
                "is_hotspot": bool(c.get("is_hotspot", False)),
            }
            for c in cells_data
        ],
        "active_beams": [
            {
                "sat_id": int(b.get("sat_id", 0)),
                "cell_id": int(b.get("cell_id", 1)) - 1 if b.get("cell_id", 1) > 0 else 0,
                "rate": float(b.get("rate_mbps", 0.0)),
                "sinr": float(b.get("sinr_db", 0.0)),
            }
            for b in active_beams
        ],
        "selected_id": sat_id,
    }

    instances_json = json.dumps(instances_payload)

    html_payload = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    user-select: none;
}}
body {{
    background-color: #050811;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    overflow: hidden;
    width: 100%;
    height: 100%;
}}
#canvas-container {{
    position: relative;
    width: 100%;
    height: 500px;
    background: #050811;
    border: 1px solid #14223d;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 0 24px rgba(0,0,0,0.85);
}}
#three-canvas {{
    width: 100%;
    height: 100%;
    display: block;
}}
/* Satellite Callout HUD */
.sat-tag {{
    position: absolute;
    top: 20px;
    right: 20px;
    background: rgba(9, 15, 31, 0.88);
    border: 1px solid #00d4ff;
    border-radius: 6px;
    padding: 10px 16px;
    box-shadow: 0 0 16px rgba(0, 212, 255, 0.3);
    backdrop-filter: blur(6px);
    z-index: 10;
    pointer-events: none;
}}
.sat-tag-title {{
    font-weight: 800;
    font-size: 0.95rem;
    color: #ffffff;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.sat-tag-body {{
    font-size: 0.74rem;
    color: #7388a8;
    font-family: monospace;
    margin-top: 4px;
    line-height: 1.5;
}}
/* Bottom Telemetry HUD */
.hud-card {{
    position: absolute;
    bottom: 16px;
    left: 16px;
    background: rgba(7, 12, 24, 0.88);
    border: 1px solid #14223d;
    border-radius: 6px;
    padding: 9px 14px;
    font-size: 0.74rem;
    backdrop-filter: blur(6px);
    z-index: 10;
    pointer-events: none;
}}
.hud-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 3px;
}}
/* Controls Bar */
.ctrl-bar {{
    position: absolute;
    bottom: 20px;
    right: 20px;
    display: flex;
    gap: 6px;
    z-index: 10;
}}
.btn-action {{
    background: #090f1f;
    border: 1px solid #1f3354;
    color: #ffffff;
    padding: 6px 12px;
    font-size: 0.72rem;
    font-weight: 700;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 4px;
}}
.btn-action:hover {{
    background: #14223d;
    border-color: #00d4ff;
    color: #00d4ff;
}}
.btn-action.active {{
    background: rgba(0, 212, 255, 0.2);
    border-color: #00d4ff;
    color: #00d4ff;
}}
.legend-pill {{
    position: absolute;
    top: 20px;
    left: 20px;
    background: rgba(7, 12, 24, 0.85);
    border: 1px solid #14223d;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.72rem;
    display: flex;
    gap: 14px;
    z-index: 10;
    backdrop-filter: blur(4px);
}}
</style>
</head>
<body>
<div id="canvas-container">
    <div id="three-canvas"></div>

    <div class="legend-pill">
        <span style="color:#00d4ff;">● Shell 0 (550 km)</span>
        <span style="color:#ffaa00;">● Shell 1 (1100 km)</span>
        <span style="color:#00ff88;">■ Ground Cells</span>
        <span style="color:#00ffff;">⚡ Active Lasers</span>
    </div>

    <div class="sat-tag">
        <div class="sat-tag-title">
            <span>ORBIT-{sat_id+1:02d}</span>
            <span style="background:rgba(0,255,136,0.15); color:#00ff88; border:1px solid #00ff88; font-size:0.65rem; padding:1px 6px; border-radius:3px;">ACTIVE TRACK</span>
        </div>
        <div class="sat-tag-body">
            Alt <b style="color:#00ff88;">{alt:.1f} km</b><br>
            Vel <b style="color:#ffffff;">7.61 km/s</b><br>
            Incl <b style="color:#00d4ff;">{incl:.1f}°</b>
        </div>
    </div>

    <div class="hud-card">
        <div class="hud-row">
            <span style="color:#00ff88; font-size:0.95rem;">●</span>
            <span style="color:#ffffff;">Sub-Satellite: <b style="color:#00d4ff; font-family:monospace;">Lat {lat:.3f}° N, Lon {lon:.3f}° E</b></span>
        </div>
        <div class="hud-row">
            <span style="color:#0066ff; font-size:1.1rem; line-height:0.8;">—</span>
            <span style="color:#7388a8;">Constellation Fleet: <b style="color:#ffffff;">{len(sats_data)} Satellites in 3D Orbit</b></span>
        </div>
        <div class="hud-row" style="margin-bottom:0;">
            <span style="color:#00ffff;">⚡</span>
            <span style="color:#7388a8;">Active Downlinks: <b style="color:#00ff88;">{active_beam_cnt} Laser Beams Connected</b></span>
        </div>
    </div>

    <div class="ctrl-bar">
        <div class="btn-action active" id="btnRotate" onclick="toggleAutoRotate()">⟳ Auto-Rotate</div>
        <div class="btn-action" onclick="focusSelectedSat()">🛰️ Focus Target</div>
        <div class="btn-action" onclick="resetCamera()">🔍 Reset View</div>
    </div>
</div>

<script>
{three_code}
</script>
<script>
{controls_code}
</script>

<script>
const DATA = {instances_json};
const container = document.getElementById('canvas-container');
const width = container.clientWidth || 800;
const height = container.clientHeight || 500;

// 1. Scene, Camera, Renderer
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050811);

const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
camera.position.set(24, 14, 26);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false, powerPreference: 'high-performance' }});
renderer.setSize(width, height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.getElementById('three-canvas').appendChild(renderer.domElement);

// 2. Interactive OrbitControls (Mouse drag rotate, scroll zoom)
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.rotateSpeed = 0.8;
controls.zoomSpeed = 1.2;
controls.minDistance = 12.0;
controls.maxDistance = 85.0;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;

// 3. Space Lighting
const sunLight = new THREE.DirectionalLight(0xffffff, 1.35);
sunLight.position.set(45, 25, 35);
scene.add(sunLight);

const ambientLight = new THREE.AmbientLight(0x182438, 0.55);
scene.add(ambientLight);

// 4. Background Starfield
const starGeo = new THREE.BufferGeometry();
const starCount = 1200;
const starPos = new Float32Array(starCount * 3);
for (let i = 0; i < starCount * 3; i += 3) {{
    const r = 250 + Math.random() * 400;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    starPos[i] = r * Math.sin(phi) * Math.cos(theta);
    starPos[i+1] = r * Math.sin(phi) * Math.sin(theta);
    starPos[i+2] = r * Math.cos(phi);
}}
starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
const starMat = new THREE.PointsMaterial({{ color: 0x99bbdd, size: 1.2, transparent: true, opacity: 0.8 }});
const starField = new THREE.Points(starGeo, starMat);
scene.add(starField);

// 5. Earth Globe with NASA Blue Marble Texture
const EARTH_RADIUS = 10.0;
const textureLoader = new THREE.TextureLoader();
const earthTex = textureLoader.load('{earth_b64}');
earthTex.generateMipmaps = true;

const earthGeo = new THREE.SphereGeometry(EARTH_RADIUS, 64, 64);
const earthMat = new THREE.MeshPhongMaterial({{
    map: earthTex,
    specular: new THREE.Color(0x224466),
    shininess: 18,
}});
const earthMesh = new THREE.Mesh(earthGeo, earthMat);
earthMesh.rotation.y = -Math.PI / 2;
scene.add(earthMesh);

// 6. Earth Rotating Clouds & Atmosphere Layer
const cloudsTex = textureLoader.load('{clouds_b64}');
const cloudsGeo = new THREE.SphereGeometry(EARTH_RADIUS + 0.08, 64, 64);
const cloudsMat = new THREE.MeshLambertMaterial({{
    map: cloudsTex,
    transparent: true,
    opacity: 0.42,
    blending: THREE.AdditiveBlending,
}});
const cloudsMesh = new THREE.Mesh(cloudsGeo, cloudsMat);
cloudsMesh.rotation.y = -Math.PI / 2;
scene.add(cloudsMesh);

// 7. Atmosphere Rim Glow
const atmosGeo = new THREE.SphereGeometry(EARTH_RADIUS + 0.22, 48, 48);
const atmosMat = new THREE.MeshBasicMaterial({{
    color: 0x00d4ff,
    transparent: true,
    opacity: 0.10,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
}});
const atmosMesh = new THREE.Mesh(atmosGeo, atmosMat);
scene.add(atmosMesh);

// Helper: Convert (lat, lon, altitude_km) to Three.js 3D Vector
function geoToVec3(latDeg, lonDeg, altKm) {{
    const r = EARTH_RADIUS + (altKm / 6371.0) * EARTH_RADIUS;
    const latRad = latDeg * (Math.PI / 180);
    const lonRad = (lonDeg + 90) * (Math.PI / 180);
    const y = r * Math.sin(latRad);
    const x = r * Math.cos(latRad) * Math.sin(lonRad);
    const z = r * Math.cos(latRad) * Math.cos(lonRad);
    return new THREE.Vector3(x, y, z);
}}

// 8. 3D Orbital Rings (Shell 0 & Shell 1)
const shellColors = {{ 0: 0x00d4ff, 1: 0xffaa00, 2: 0xd946ef }};
const shellAlts = {{ 0: 550.0, 1: 1100.0 }};
[0, 1].forEach(sId => {{
    const alt = shellAlts[sId] || 550.0;
    const r = EARTH_RADIUS + (alt / 6371.0) * EARTH_RADIUS;
    const inc = (sId === 0 ? 53.0 : 98.7) * (Math.PI / 180);

    const orbitPoints = [];
    const segments = 120;
    for (let i = 0; i <= segments; i++) {{
        const theta = (i / segments) * Math.PI * 2;
        const x = r * Math.cos(theta);
        const y = r * Math.sin(theta) * Math.sin(inc);
        const z = r * Math.sin(theta) * Math.cos(inc);
        orbitPoints.push(new THREE.Vector3(x, y, z));
    }}
    const orbitGeo = new THREE.BufferGeometry().setFromPoints(orbitPoints);
    const orbitMat = new THREE.LineDashedMaterial({{
        color: shellColors[sId],
        linewidth: 1.5,
        scale: 1,
        dashSize: 0.5,
        gapSize: 0.25,
        transparent: true,
        opacity: 0.65,
    }});
    const orbitLine = new THREE.Line(orbitGeo, orbitMat);
    orbitLine.computeLineDistances();
    scene.add(orbitLine);
}});

// 9. Ground Cell Instances on Earth Surface
const cellPositions = {{}};
DATA.cells.forEach(c => {{
    const pos = geoToVec3(c.lat, c.lon, 0.0);
    cellPositions[c.id] = pos;

    // Glowing base ring
    const ringGeo = new THREE.RingGeometry(0.12, 0.22, 16);
    const ringMat = new THREE.MeshBasicMaterial({{
        color: c.is_hotspot ? 0xff3366 : 0x00ff88,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.9,
    }});
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.position.copy(pos);
    ringMesh.lookAt(0, 0, 0);
    scene.add(ringMesh);

    // Vertical beacon pole
    const poleGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.45, 8);
    const poleMat = new THREE.MeshBasicMaterial({{ color: c.is_hotspot ? 0xff3366 : 0x00ff88 }});
    const poleMesh = new THREE.Mesh(poleGeo, poleMat);
    poleMesh.position.copy(pos.clone().multiplyScalar(1.02));
    poleMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pos.clone().normalize());
    scene.add(poleMesh);
}});

// 10. Satellite Instances in Orbit
const satObjects = {{}};
let targetSatPos = null;

DATA.satellites.forEach(s => {{
    const pos = geoToVec3(s.lat, s.lon, s.alt);
    const isSelected = (s.id === DATA.selected_id);
    if (isSelected) targetSatPos = pos;

    const satGroup = new THREE.Group();
    satGroup.position.copy(pos);

    // A. Satellite Body (Metallic cuboid)
    const bodyGeo = new THREE.BoxGeometry(0.38, 0.24, 0.26);
    const bodyMat = new THREE.MeshStandardMaterial({{
        color: 0xcccccc,
        metalness: 0.85,
        roughness: 0.2,
    }});
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    satGroup.add(body);

    // B. Solar Array Wings (Photovoltaic blue panels)
    const wingGeo = new THREE.BoxGeometry(1.05, 0.04, 0.36);
    const wingMat = new THREE.MeshStandardMaterial({{
        color: 0x0055ff,
        metalness: 0.4,
        roughness: 0.15,
        emissive: 0x002288,
        emissiveIntensity: 0.6,
    }});
    const wings = new THREE.Mesh(wingGeo, wingMat);
    satGroup.add(wings);

    // C. Pointing towards Earth center
    satGroup.lookAt(0, 0, 0);

    // D. Beacon glow light
    const beaconGeo = new THREE.SphereGeometry(0.12, 12, 12);
    const beaconMat = new THREE.MeshBasicMaterial({{
        color: isSelected ? 0x00ffff : (shellColors[s.shell_id] || 0x00d4ff),
    }});
    const beacon = new THREE.Mesh(beaconGeo, beaconMat);
    beacon.position.set(0, 0.22, 0);
    satGroup.add(beacon);

    // E. Target Reticle if selected
    if (isSelected) {{
        const targetRingGeo = new THREE.RingGeometry(0.7, 0.82, 32);
        const targetRingMat = new THREE.MeshBasicMaterial({{
            color: 0x00ffff,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.85,
        }});
        const targetRing = new THREE.Mesh(targetRingGeo, targetRingMat);
        targetRing.lookAt(0, 0, 0);
        satGroup.add(targetRing);
    }}

    scene.add(satGroup);
    satObjects[s.id] = satGroup;
}});

// 11. Active Downlink Laser Beams
const beamParticles = [];
DATA.active_beams.forEach(b => {{
    const sPos = satObjects[b.sat_id] ? satObjects[b.sat_id].position : null;
    const cPos = cellPositions[b.cell_id] || null;

    if (sPos && cPos) {{
        // Laser Line
        const laserPoints = [sPos.clone(), cPos.clone()];
        const laserGeo = new THREE.BufferGeometry().setFromPoints(laserPoints);
        const laserMat = new THREE.LineBasicMaterial({{
            color: 0x00ffff,
            linewidth: 3,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending,
        }});
        const laser = new THREE.Line(laserGeo, laserMat);
        scene.add(laser);

        // Moving photon pulse packet along the beam
        const photonGeo = new THREE.SphereGeometry(0.12, 12, 12);
        const photonMat = new THREE.MeshBasicMaterial({{ color: 0xffffff, blending: THREE.AdditiveBlending }});
        const photon = new THREE.Mesh(photonGeo, photonMat);
        scene.add(photon);
        beamParticles.push({{ mesh: photon, start: sPos, end: cPos, progress: Math.random() }});
    }}
}});

// 12. Camera Controls & Buttons
let autoRotateEnabled = true;

function toggleAutoRotate() {{
    autoRotateEnabled = !autoRotateEnabled;
    controls.autoRotate = autoRotateEnabled;
    const btn = document.getElementById('btnRotate');
    if (autoRotateEnabled) {{
        btn.classList.add('active');
    }} else {{
        btn.classList.remove('active');
    }}
}}

function resetCamera() {{
    camera.position.set(24, 14, 26);
    controls.target.set(0, 0, 0);
    controls.update();
}}

function focusSelectedSat() {{
    if (targetSatPos) {{
        const offset = targetSatPos.clone().normalize().multiplyScalar(18.0);
        camera.position.copy(offset);
        controls.target.copy(targetSatPos);
        controls.update();
    }}
}}

// Responsive resize
window.addEventListener('resize', () => {{
    const w = container.clientWidth || 800;
    const h = container.clientHeight || 500;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}});

// 13. High-Performance 60 FPS Render Loop
function animate() {{
    requestAnimationFrame(animate);

    // Slowly rotate clouds independently for atmospheric realism
    cloudsMesh.rotation.y += 0.0002;

    // Animate photon packets moving along laser downlinks
    beamParticles.forEach(p => {{
        p.progress += 0.025;
        if (p.progress > 1.0) p.progress = 0.0;
        p.mesh.position.lerpVectors(p.start, p.end, p.progress);
    }});

    controls.update();
    renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""

    components.html(html_payload, height=515, scrolling=False)
