"""Interactive 3D and 2D Plotly visualizations with realistic Earth geography and satellite tracking."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go

DARK_LAYOUT_TEMPLATE = {
    "paper_bgcolor": "#050811",
    "plot_bgcolor": "#050811",
    "font": {"color": "#e8eaf0", "family": "Inter, sans-serif"},
    "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
}


def create_realistic_earth_globe_fig(
    satellites: List[Dict[str, Any]],
    cells: List[Dict[str, Any]],
    active_beams: Optional[List[Dict[str, Any]]] = None,
    center_lat: float = 35.0,
    center_lon: float = 15.0,
    show_visibility: bool = True,
    show_beams: bool = True,
) -> go.Figure:
    """
    Generate a photorealistic 3D orthographic spherical Earth globe with continents,
    country borders, oceans, satellite orbits, sub-satellite nodes, and active downlink beams.
    """
    fig = go.Figure()

    # 1. Active Downlink Beams (Connecting Satellites to Target Terrestrial Cells)
    if show_beams and active_beams:
        beam_lats = []
        beam_lons = []
        for b in active_beams:
            s_id = b["sat_id"]
            c_id = b["cell_id"] - 1 if b["cell_id"] > 0 else None
            if c_id is not None and c_id < len(cells) and s_id < len(satellites):
                sat = satellites[s_id]
                cell = cells[c_id]
                beam_lats.extend([sat["lat"], cell["lat"], None])
                beam_lons.extend([sat["lon"], cell["lon"], None])

        if beam_lats:
            fig.add_trace(
                go.Scattergeo(
                    lat=beam_lats,
                    lon=beam_lons,
                    mode="lines",
                    line=dict(color="#00ffff", width=3.5),
                    name="Active Downlink Beams",
                    hoverinfo="none",
                )
            )

    # 2. LOS Visibility Links (faint)
    if show_visibility and not active_beams:
        vis_lats = []
        vis_lons = []
        for s in satellites:
            for c_id in s.get("visible_cells", []):
                if c_id < len(cells):
                    cell = cells[c_id]
                    vis_lats.extend([s["lat"], cell["lat"], None])
                    vis_lons.extend([s["lon"], cell["lon"], None])

        if vis_lats:
            fig.add_trace(
                go.Scattergeo(
                    lat=vis_lats,
                    lon=vis_lons,
                    mode="lines",
                    line=dict(color="rgba(0, 212, 255, 0.2)", width=1),
                    name="LOS Visibility Links",
                    hoverinfo="none",
                )
            )

    # 3. Ground Cells on Earth Surface
    cell_lats = [c["lat"] for c in cells]
    cell_lons = [c["lon"] for c in cells]
    cell_colors = ["#ff3366" if c.get("is_hotspot", False) else "#00ff88" for c in cells]
    cell_hover = [
        f"<b>Ground Cell C{c['cell_id']:02d}</b><br>Type: {'🔥 Hotspot' if c.get('is_hotspot') else 'Regular'}<br>Lat: {c['lat']:.2f}°, Lon: {c['lon']:.2f}°<br>Queue Backlog: {c.get('queue_mb', 0):.1f} Mb"
        for c in cells
    ]

    fig.add_trace(
        go.Scattergeo(
            lat=cell_lats,
            lon=cell_lons,
            mode="markers+text",
            marker=dict(
                size=8,
                color=cell_colors,
                symbol="square",
                line=dict(color="#ffffff", width=1),
                opacity=0.95,
            ),
            text=[f"C{c['cell_id']:02d}" for c in cells],
            textposition="top center",
            textfont=dict(color="#00ff88", size=9, family="monospace"),
            hovertext=cell_hover,
            hoverinfo="text",
            name="Ground Cells",
        )
    )

    # 4. Satellites by Shell (Distinct diamond markers with orbital colors)
    shell_palette = {0: "#00d4ff", 1: "#ffaa00", 2: "#d946ef"}
    unique_shells = sorted(list(set(s.get("shell_id", 0) for s in satellites)))

    for shell_id in unique_shells:
        shell_sats = [s for s in satellites if s.get("shell_id", 0) == shell_id]
        alt = shell_sats[0].get("altitude_km", 550.0) if shell_sats else 550.0

        s_lats = [s["lat"] for s in shell_sats]
        s_lons = [s["lon"] for s in shell_sats]
        s_hover = [
            f"<b>🛰️ Sat S{s['sat_id']:02d} (Shell {shell_id})</b><br>Altitude: {s.get('altitude_km', alt):.0f} km<br>Sub-Sat: {s['lat']:.2f}° N, {s['lon']:.2f}° E<br>LOS Visible Cells: {len(s.get('visible_cells', []))}"
            for s in shell_sats
        ]

        # Orbital Ground Track representation for this shell
        if len(s_lats) >= 2:
            track_lats = np.linspace(min(s_lats) - 15, max(s_lats) + 15, 30)
            track_lons = np.linspace(min(s_lons) - 25, max(s_lons) + 25, 30)
            fig.add_trace(
                go.Scattergeo(
                    lat=track_lats,
                    lon=track_lons,
                    mode="lines",
                    line=dict(
                        color=shell_palette.get(shell_id, "#00d4ff"),
                        width=1.5,
                        dash="dash",
                    ),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scattergeo(
                lat=s_lats,
                lon=s_lons,
                mode="markers+text",
                marker=dict(
                    size=13,
                    color=shell_palette.get(shell_id, "#00d4ff"),
                    symbol="diamond",
                    line=dict(color="#ffffff", width=1.8),
                ),
                text=[f"S{s['sat_id']:02d}" for s in shell_sats],
                textposition="bottom center",
                textfont=dict(color=shell_palette.get(shell_id, "#00d4ff"), size=10, family="monospace"),
                hovertext=s_hover,
                hoverinfo="text",
                name=f"Shell {shell_id} ({alt:.0f} km)",
            )
        )

    # 5. Photorealistic Orthographic Earth Sphere Configuration
    # Uses realistic Earth landmasses, lush green terrain, deep blue oceans, and glowing coastlines
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=center_lon, lat=center_lat, roll=0),
        showocean=True,
        oceancolor="#0b2239",
        showland=True,
        landcolor="#1c3e27",
        showlakes=True,
        lakecolor="#0b2239",
        showcountries=True,
        countrycolor="#2f5a3b",
        countrywidth=0.8,
        showcoastlines=True,
        coastlinecolor="#00d4ff",
        coastlinewidth=1.3,
        bgcolor="#050811",
        resolution=50,
    )

    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        height=540,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(10, 15, 29, 0.85)",
            bordercolor="#141f36",
            borderwidth=1,
            x=0.02,
            y=0.98,
            font=dict(size=10, color="#cbd5e1"),
        ),
    )
    return fig


def create_3d_constellation_fig(
    satellites: List[Dict[str, Any]],
    cells: List[Dict[str, Any]],
    active_beams: Optional[List[Dict[str, Any]]] = None,
    show_visibility: bool = True,
    show_beams: bool = True,
) -> go.Figure:
    """
    Ultra-responsive GPU-accelerated WebGL 3D Constellation Mesh in ECEF space.
    Provides 60 FPS lag-free 3D mouse rotation, orbital shells, satellite nodes,
    Earth sphere surface, and active downlink laser beams.
    """
    R_E = 6371.0  # Earth mean radius in km
    fig = go.Figure()

    # 1. 3D Earth Spherical Surface Mesh (Optimized 20x40 grid for instant 60 FPS WebGL)
    n_phi, n_theta = 20, 40
    phi = np.linspace(0, np.pi, n_phi)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    x_earth = R_E * np.sin(phi_grid) * np.cos(theta_grid)
    y_earth = R_E * np.sin(phi_grid) * np.sin(theta_grid)
    z_earth = R_E * np.cos(phi_grid)

    # Continent/ocean gradient pattern
    topo_pattern = np.sin(2.5 * phi_grid) * np.cos(3.5 * theta_grid)

    fig.add_trace(
        go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            surfacecolor=topo_pattern,
            colorscale=[
                [0.0, "#061324"],
                [0.35, "#0b233a"],
                [0.50, "#133a20"],
                [0.75, "#1e5230"],
                [1.0, "#326e44"],
            ],
            showscale=False,
            opacity=0.94,
            hoverinfo="none",
            name="Earth Globe",
        )
    )

    # 2. Equator & Meridian Guidance Rings
    angles = np.linspace(0, 2 * np.pi, 60)
    fig.add_trace(
        go.Scatter3d(
            x=R_E * np.cos(angles),
            y=R_E * np.sin(angles),
            z=np.zeros_like(angles),
            mode="lines",
            line=dict(color="rgba(0, 212, 255, 0.25)", width=1.5),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # 3. 3D Inclined Orbital Shell Rings
    shell_colors = {0: "#00d4ff", 1: "#ffaa00", 2: "#d946ef"}
    unique_shells = sorted(list(set(s.get("shell_id", 0) for s in satellites)))

    for shell_id in unique_shells:
        shell_sats = [s for s in satellites if s.get("shell_id", 0) == shell_id]
        alt = shell_sats[0].get("altitude_km", 550.0) if shell_sats else 550.0
        r_orbit = R_E + alt
        inc = np.radians(shell_sats[0].get("inclination", 53.0)) if shell_sats else np.radians(53.0)

        # 3D inclined circular orbit trajectory
        x_orb = r_orbit * np.cos(angles)
        y_orb = r_orbit * np.sin(angles) * np.cos(inc)
        z_orb = r_orbit * np.sin(angles) * np.sin(inc)

        fig.add_trace(
            go.Scatter3d(
                x=x_orb,
                y=y_orb,
                z=z_orb,
                mode="lines",
                line=dict(
                    color=shell_colors.get(shell_id, "#00d4ff"),
                    width=2,
                    dash="dash",
                ),
                name=f"Shell {shell_id} Orbit ({alt:.0f} km)",
                hoverinfo="none",
            )
        )

    # 4. Active Downlink Laser Beams (Satellite (X,Y,Z) to Ground Cell (X,Y,Z))
    if show_beams and active_beams:
        beam_x, beam_y, beam_z = [], [], []
        for b in active_beams:
            s_id = b["sat_id"]
            c_id = b["cell_id"] - 1 if b["cell_id"] > 0 else 0
            if s_id < len(satellites) and c_id < len(cells):
                sat = satellites[s_id]
                cell = cells[c_id]
                sx = sat.get("x_km", (R_E + sat.get("altitude_km", 550.0)) * np.cos(np.radians(sat["lat"])) * np.cos(np.radians(sat["lon"])))
                sy = sat.get("y_km", (R_E + sat.get("altitude_km", 550.0)) * np.cos(np.radians(sat["lat"])) * np.sin(np.radians(sat["lon"])))
                sz = sat.get("z_km", (R_E + sat.get("altitude_km", 550.0)) * np.sin(np.radians(sat["lat"])))

                cx = cell.get("x_km", R_E * np.cos(np.radians(cell["lat"])) * np.cos(np.radians(cell["lon"])))
                cy = cell.get("y_km", R_E * np.cos(np.radians(cell["lat"])) * np.sin(np.radians(cell["lon"])))
                cz = cell.get("z_km", R_E * np.sin(np.radians(cell["lat"])))

                beam_x.extend([sx, cx, None])
                beam_y.extend([sy, cy, None])
                beam_z.extend([sz, cz, None])

        if beam_x:
            fig.add_trace(
                go.Scatter3d(
                    x=beam_x,
                    y=beam_y,
                    z=beam_z,
                    mode="lines",
                    line=dict(color="#00ffff", width=5.0),
                    name="Active Laser Downlink",
                    hoverinfo="none",
                )
            )

    # 5. Ground Cells on Earth Surface (3D ECEF)
    cell_x, cell_y, cell_z = [], [], []
    cell_colors, cell_hover, cell_labels = [], [], []
    for c in cells:
        lat_r, lon_r = np.radians(c["lat"]), np.radians(c["lon"])
        cx = c.get("x_km", R_E * np.cos(lat_r) * np.cos(lon_r))
        cy = c.get("y_km", R_E * np.cos(lat_r) * np.sin(lon_r))
        cz = c.get("z_km", R_E * np.sin(lat_r))

        cell_x.append(cx)
        cell_y.append(cy)
        cell_z.append(cz)
        cell_colors.append("#ff3366" if c.get("is_hotspot", False) else "#00ff88")
        cell_labels.append(f"C{c['cell_id']:02d}")
        cell_hover.append(
            f"<b>Ground Cell C{c['cell_id']:02d}</b><br>"
            f"Lat: {c['lat']:.2f}°, Lon: {c['lon']:.2f}°<br>"
            f"Queue: {c.get('queue_mb', 0):.1f} Mb<br>"
            f"Type: {'🔥 Hotspot' if c.get('is_hotspot') else 'Regular'}"
        )

    fig.add_trace(
        go.Scatter3d(
            x=cell_x,
            y=cell_y,
            z=cell_z,
            mode="markers+text",
            marker=dict(
                size=6,
                color=cell_colors,
                symbol="circle",
                line=dict(color="#ffffff", width=1),
                opacity=0.95,
            ),
            text=cell_labels,
            textposition="top center",
            textfont=dict(color="#00ff88", size=9, family="monospace"),
            hovertext=cell_hover,
            hoverinfo="text",
            name="Ground Stations",
        )
    )

    # 6. Satellites in Orbit (3D ECEF)
    for shell_id in unique_shells:
        shell_sats = [s for s in satellites if s.get("shell_id", 0) == shell_id]
        sx, sy, sz = [], [], []
        s_hover, s_text = [], []

        for s in shell_sats:
            alt = s.get("altitude_km", 550.0)
            r_sat = R_E + alt
            lat_r, lon_r = np.radians(s["lat"]), np.radians(s["lon"])
            x = s.get("x_km", r_sat * np.cos(lat_r) * np.cos(lon_r))
            y = s.get("y_km", r_sat * np.cos(lat_r) * np.sin(lon_r))
            z = s.get("z_km", r_sat * np.sin(lat_r))

            sx.append(x)
            sy.append(y)
            sz.append(z)
            s_text.append(f"S{s['sat_id']:02d}")
            s_hover.append(
                f"<b>🛰️ Satellite S{s['sat_id']:02d} (Shell {shell_id})</b><br>"
                f"Altitude: {alt:.0f} km<br>"
                f"Position: ({x:.0f}, {y:.0f}, {z:.0f}) km<br>"
                f"Sub-Sat: {s['lat']:.2f}° N, {s['lon']:.2f}° E<br>"
                f"LOS Visible: {len(s.get('visible_cells', []))} cells"
            )

        fig.add_trace(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="markers+text",
                marker=dict(
                    size=8,
                    color=shell_colors.get(shell_id, "#00d4ff"),
                    symbol="diamond",
                    line=dict(color="#ffffff", width=1.5),
                ),
                text=s_text,
                textposition="bottom center",
                textfont=dict(color=shell_colors.get(shell_id, "#00d4ff"), size=10, family="monospace"),
                hovertext=s_hover,
                hoverinfo="text",
                name=f"Shell {shell_id} Satellites",
            )
        )

    # 7. Layout & GPU-accelerated 3D WebGL Camera Configuration
    fig.update_layout(
        paper_bgcolor="#050811",
        plot_bgcolor="#050811",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        height=480,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="constellation_3d_state",  # Preserves user's 3D zoom & rotation between simulation steps!
        showlegend=True,
        legend=dict(
            bgcolor="rgba(9, 15, 31, 0.85)",
            bordercolor="#14223d",
            borderwidth=1,
            x=0.02,
            y=0.98,
            font=dict(size=10, color="#cbd5e1"),
        ),
        scene=dict(
            aspectmode="data",
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            bgcolor="#050811",
            camera=dict(
                eye=dict(x=1.35, y=1.35, z=0.95),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
    )
    return fig


def create_2d_ground_track_fig(
    satellites: List[Dict[str, Any]],
    cells: List[Dict[str, Any]],
    active_beams: Optional[List[Dict[str, Any]]] = None,
) -> go.Figure:
    """
    Generate 2D equirectangular ground track map with continental landmasses.
    """
    fig = go.Figure()

    # Ground Cells
    c_lats = [c["lat"] for c in cells]
    c_lons = [c["lon"] for c in cells]
    c_colors = ["#ff3366" if c.get("is_hotspot") else "#00ff88" for c in cells]
    c_hover = [
        f"<b>Cell C{c['cell_id']:02d}</b><br>Lat: {c['lat']:.2f}°, Lon: {c['lon']:.2f}°<br>Queue: {c.get('queue_mb', 0):.1f} Mb"
        for c in cells
    ]

    fig.add_trace(
        go.Scattergeo(
            lat=c_lats,
            lon=c_lons,
            mode="markers+text",
            marker=dict(size=8, color=c_colors, symbol="square", line=dict(color="#ffffff", width=1)),
            text=[f"C{c['cell_id']:02d}" for c in cells],
            textposition="top center",
            textfont=dict(color="#00ff88", size=9),
            hovertext=c_hover,
            hoverinfo="text",
            name="Ground Cells",
        )
    )

    # Satellites
    shell_colors = {0: "#00d4ff", 1: "#ffaa00"}
    for shell_id in sorted(list(set(s.get("shell_id", 0) for s in satellites))):
        s_list = [s for s in satellites if s.get("shell_id", 0) == shell_id]
        s_lats = [s["lat"] for s in s_list]
        s_lons = [s["lon"] for s in s_list]
        s_hover = [
            f"<b>Sat S{s['sat_id']:02d} (Shell {shell_id})</b><br>Lat: {s['lat']:.2f}°, Lon: {s['lon']:.2f}°<br>Visible: {len(s.get('visible_cells', []))} cells"
            for s in s_list
        ]

        fig.add_trace(
            go.Scattergeo(
                lat=s_lats,
                lon=s_lons,
                mode="markers+text",
                marker=dict(
                    size=13,
                    color=shell_colors.get(shell_id, "#00d4ff"),
                    symbol="diamond",
                    line=dict(color="#ffffff", width=1.5),
                ),
                text=[f"S{s['sat_id']:02d}" for s in s_list],
                textposition="bottom center",
                textfont=dict(color=shell_colors.get(shell_id, "#00d4ff"), size=10, family="monospace"),
                hovertext=s_hover,
                hoverinfo="text",
                name=f"Satellites (Shell {shell_id})",
            )
        )

    # Active Beams
    if active_beams:
        beam_lats, beam_lons = [], []
        for b in active_beams:
            s_id = b["sat_id"]
            c_id = b["cell_id"] - 1 if b["cell_id"] > 0 else None
            if c_id is not None and c_id < len(cells) and s_id < len(satellites):
                sat = satellites[s_id]
                cell = cells[c_id]
                beam_lats.extend([sat["lat"], cell["lat"], None])
                beam_lons.extend([sat["lon"], cell["lon"], None])

        if beam_lats:
            fig.add_trace(
                go.Scattergeo(
                    lat=beam_lats,
                    lon=beam_lons,
                    mode="lines",
                    line=dict(color="#00ffff", width=2.5),
                    name="Active Beams",
                    hoverinfo="none",
                )
            )

    fig.update_geos(
        projection_type="equirectangular",
        showocean=True,
        oceancolor="#08182b",
        showland=True,
        landcolor="#182c3f",
        showcountries=True,
        countrycolor="#2d4863",
        showcoastlines=True,
        coastlinecolor="#4a7094",
        bgcolor="#050811",
        resolution=50,
    )

    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        height=480,
        legend=dict(
            bgcolor="rgba(10, 15, 29, 0.85)",
            bordercolor="#141f36",
            borderwidth=1,
            x=0.01,
            y=0.99,
        ),
    )
    return fig
