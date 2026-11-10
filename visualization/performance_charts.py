"""Interactive Plotly scientific charts for simulation telemetry, baselines, and RF analysis."""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK_LAYOUT = {
    "paper_bgcolor": "#0a0e1a",
    "plot_bgcolor": "#0d1117",
    "font": {"color": "#e8eaf0", "family": "Inter, sans-serif"},
    "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
}


def create_slot_dynamics_fig(df_telemetry: pd.DataFrame) -> go.Figure:
    """
    Generate 2-panel live slot execution dynamics:
    Panel 1: Served Traffic (Mb) vs Backlog (Mb)
    Panel 2: Retained Beams vs Rejected Requests + SINR (dB)
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Traffic Delivery & Backlog Dynamics", "Conflict Resolution & RF Quality"),
        horizontal_spacing=0.12,
    )

    if df_telemetry.empty:
        fig.update_layout(**DARK_LAYOUT, height=350)
        return fig

    slots = df_telemetry["slot"]

    # Panel 1: Served vs Backlog
    fig.add_trace(
        go.Scatter(
            x=slots, y=df_telemetry["served_traffic_mb"],
            mode="lines+markers", name="Served Traffic (Mb)",
            line=dict(color="#00ff88", width=2),
            marker=dict(size=5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=slots, y=df_telemetry["remaining_backlog_mb"],
            mode="lines+markers", name="Pending Backlog (Mb)",
            line=dict(color="#ff4466", width=2, dash="dash"),
            marker=dict(size=5),
        ),
        row=1, col=1,
    )

    # Panel 2: Retained, Rejected, SINR
    fig.add_trace(
        go.Bar(
            x=slots, y=df_telemetry["num_retained"],
            name="Retained Beams",
            marker=dict(color="#00d4ff", opacity=0.7),
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=slots, y=df_telemetry["num_rejected"],
            name="Rejected Requests",
            marker=dict(color="#ffaa00", opacity=0.8),
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=slots, y=df_telemetry["mean_sinr_db"],
            mode="lines+markers", name="Mean SINR (dB)",
            line=dict(color="#d946ef", width=2),
            yaxis="y2" if False else None,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        **DARK_LAYOUT,
        height=380,
        barmode="group",
        legend=dict(
            bgcolor="rgba(13, 17, 23, 0.8)",
            bordercolor="#1e2d3d",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_xaxes(title_text="Scheduling Slot (t)", gridcolor="#1e2d3d", zerolinecolor="#1e2d3d")
    fig.update_yaxes(title_text="Traffic (Mb)", gridcolor="#1e2d3d", zerolinecolor="#1e2d3d", row=1, col=1)
    fig.update_yaxes(title_text="Beams / dB", gridcolor="#1e2d3d", zerolinecolor="#1e2d3d", row=1, col=2)
    return fig


def create_baseline_comparison_fig(df_comparison_summary: pd.DataFrame) -> go.Figure:
    """
    Generate 4-panel comparison bar charts for multi-policy evaluation across seeds.
    Metrics: Delivered Traffic (Mb), Throughput (Mbps), Jain's Fairness, Retention Rate (%).
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Total Delivered Traffic (Mb)",
            "Mean Throughput (Mbps)",
            "Jain's Fairness Index",
            "Beam Request Retention Rate (%)",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    if df_comparison_summary.empty:
        fig.update_layout(**DARK_LAYOUT, height=500)
        return fig

    policies = df_comparison_summary["policy"] if "policy" in df_comparison_summary else df_comparison_summary.index
    palette = ["#ff4466", "#ffaa00", "#00d4ff", "#00ff88", "#d946ef"]

    # 1. Delivered / Served
    val_deliv = df_comparison_summary["total_served_mb"] if "total_served_mb" in df_comparison_summary else df_comparison_summary.get("total_delivered_mb", [0] * len(policies))
    fig.add_trace(
        go.Bar(x=policies, y=val_deliv, marker_color=palette[:len(policies)], showlegend=False),
        row=1, col=1,
    )

    # 2. Throughput
    val_tp = df_comparison_summary.get("mean_throughput_mbps", [0] * len(policies))
    fig.add_trace(
        go.Bar(x=policies, y=val_tp, marker_color=palette[:len(policies)], showlegend=False),
        row=1, col=2,
    )

    # 3. Fairness
    val_fair = df_comparison_summary.get("mean_jains_fairness", [0] * len(policies))
    fig.add_trace(
        go.Bar(x=policies, y=val_fair, marker_color=palette[:len(policies)], showlegend=False),
        row=2, col=1,
    )

    # 4. Retention
    val_ret = df_comparison_summary.get("mean_retention_rate", [0] * len(policies))
    fig.add_trace(
        go.Bar(x=policies, y=val_ret, marker_color=palette[:len(policies)], showlegend=False),
        row=2, col=2,
    )

    fig.update_layout(
        **DARK_LAYOUT,
        height=520,
    )
    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(gridcolor="#1e2d3d", zerolinecolor="#1e2d3d", row=r, col=c)
            fig.update_yaxes(gridcolor="#1e2d3d", zerolinecolor="#1e2d3d", row=r, col=c)

    return fig


def create_traffic_heatmap_fig(queue_matrix: np.ndarray) -> go.Figure:
    """
    Generate Ground Cell Queue Backlog Heatmap over time.

    Args:
        queue_matrix: Shape [num_slots, num_cells].
    """
    num_slots, num_cells = queue_matrix.shape
    fig = go.Figure(
        data=go.Heatmap(
            z=queue_matrix.T,
            x=[f"Slot {s+1}" for s in range(num_slots)],
            y=[f"Cell {c:02d}" for c in range(num_cells)],
            colorscale="Viridis",
            colorbar=dict(title="Backlog (Mb)", tickfont=dict(color="#e8eaf0")),
        )
    )
    fig.update_layout(
        **DARK_LAYOUT,
        height=450,
        title=dict(text="Terrestrial Cell Queue Evolution (Temporal Heatmap)", font=dict(size=14, color="#e8eaf0")),
        xaxis=dict(title="Scheduling Slot", gridcolor="#1e2d3d"),
        yaxis=dict(title="Ground Cell Index", gridcolor="#1e2d3d"),
    )
    return fig


def create_rf_link_fig(beam_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Generate SINR vs Throughput scatter plot for all active beams in a slot.
    """
    fig = go.Figure()

    if not beam_data:
        fig.update_layout(**DARK_LAYOUT, height=350)
        return fig

    active_beams = [b for b in beam_data if b.get("is_active", False) and b.get("is_retained", False)]

    if not active_beams:
        fig.update_layout(**DARK_LAYOUT, height=350, title="No Active Beams In Current Slot")
        return fig

    sinrs = [b["sinr_db"] for b in active_beams]
    rates = [b["rate_mbps"] for b in active_beams]
    powers = [b["tx_power_w"] for b in active_beams]
    labels = [f"Sat S{b['sat_id']} Beam B{b['beam_id']}" for b in active_beams]

    fig.add_trace(
        go.Scatter(
            x=sinrs,
            y=rates,
            mode="markers+text",
            marker=dict(
                size=[max(8, p * 1.5) for p in powers],
                color=powers,
                colorscale="teal",
                showscale=True,
                colorbar=dict(title="Tx Power (W)", tickfont=dict(color="#e8eaf0")),
                line=dict(color="#ffffff", width=1),
            ),
            text=labels,
            textposition="top center",
            textfont=dict(color="#00d4ff", size=10),
            hovertext=[
                f"{l}<br>SINR: {s:.2f} dB<br>Rate: {r:.2f} Mbps<br>Power: {p:.1f} W"
                for l, s, r, p in zip(labels, sinrs, rates, powers)
            ],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        height=380,
        title=dict(text="Physical Downlink: Shannon Achievable Rate vs Receiver SINR", font=dict(size=14, color="#e8eaf0")),
        xaxis=dict(title="Signal-to-Interference-plus-Noise Ratio (dB)", gridcolor="#1e2d3d", zerolinecolor="#1e2d3d"),
        yaxis=dict(title="Achievable Physical Rate (Mbps)", gridcolor="#1e2d3d", zerolinecolor="#1e2d3d"),
    )
    return fig
