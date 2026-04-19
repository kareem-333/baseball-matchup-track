"""
Visualization helpers used by the Matchup "Dig Deeper" expanders.

All functions return Plotly figures or Streamlit component calls.
They are intentionally kept separate from scoring logic.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mlb_season.pipeline import PITCH_LABELS, _season_start_str, _today_str


# ── colour palette ────────────────────────────────────────────────────────────

DARK_BG = "#0e1117"
CARD_BG = "#1a1a2e"
ACCENT = "#EB6E1F"
TEXT = "#fafafa"
SUBTLE = "#888888"

HEAT_COLORS = [
    [0.0, "#1a1a2e"],
    [0.25, "#1e3a5f"],
    [0.5, "#EB6E1F"],
    [0.75, "#f5a623"],
    [1.0, "#ffffff"],
]


# ── per-pitch breakdown table ─────────────────────────────────────────────────

def plot_per_pitch_breakdown(per_pitch_df: pd.DataFrame) -> go.Figure:
    """Styled Plotly table of the per-pitch MAS breakdown."""
    if per_pitch_df.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, height=100
        )
        return fig

    df = per_pitch_df.copy()
    df["pitch_label"] = df["pitch_type"].map(PITCH_LABELS).fillna(df["pitch_type"])
    df["usage_pct_fmt"] = (df["usage_pct"] * 100).round(1).astype(str) + "%"
    df["barrel_overlap_fmt"] = (df["barrel_overlap"] * 100).round(1).astype(str)
    df["whiff_overlap_fmt"] = (df["whiff_overlap"] * 100).round(1).astype(str)
    df["z_barrel_fmt"] = df["z_barrel"].round(2).astype(str)
    df["z_whiff_fmt"] = df["z_whiff"].round(2).astype(str)
    df["sample_size"] = df["sample_size"].astype(int).astype(str)

    header_vals = [
        "Pitch", "Usage", "Sample", "zBarrel", "zWhiff",
        "Barrel Score", "Whiff Score",
    ]
    cell_vals = [
        df["pitch_label"],
        df["usage_pct_fmt"],
        df["sample_size"],
        df["z_barrel_fmt"],
        df["z_whiff_fmt"],
        df["barrel_overlap_fmt"],
        df["whiff_overlap_fmt"],
    ]

    fig = go.Figure(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in header_vals],
                fill_color="#16213e",
                font=dict(color=TEXT, size=13),
                align="center",
                height=34,
            ),
            cells=dict(
                values=cell_vals,
                fill_color=CARD_BG,
                font=dict(color=TEXT, size=12),
                align="center",
                height=30,
            ),
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=DARK_BG,
        height=max(120, len(df) * 32 + 50),
    )
    return fig


# ── hot zone grid ─────────────────────────────────────────────────────────────

def plot_hot_zone_grid(batter_id: int, season: int | None = None) -> go.Figure:
    """
    3×3 heat-map of batter's barrel rate by strike zone location.
    Uses Statcast plate_x / plate_z divided into 3 horizontal × 3 vertical zones.

    Zones (column = left→right from catcher's view, row = low→high):
        (0,2) (1,2) (2,2)   ← top
        (0,1) (1,1) (2,1)   ← middle
        (0,0) (1,0) (2,0)   ← bottom
    """
    from pybaseball import statcast_batter

    season = season or date.today().year
    start = _season_start_str(season)
    end = _today_str()

    try:
        df = statcast_batter(start, end, batter_id)
    except Exception:
        df = pd.DataFrame()

    zone_labels = [
        ["Inside\nLow", "Middle\nLow", "Outside\nLow"],
        ["Inside\nMid", "Middle\nMid", "Outside\nMid"],
        ["Inside\nHigh", "Middle\nHigh", "Outside\nHigh"],
    ]

    if df is None or df.empty or "plate_x" not in df.columns:
        # Return empty grid
        z_data = np.zeros((3, 3))
        text_data = [["N/A"] * 3 for _ in range(3)]
    else:
        df = df.dropna(subset=["plate_x", "plate_z"])

        # Standard strike zone bounds (feet)
        x_cuts = [-0.83, 0.0, 0.83]
        z_cuts = [1.5, 2.5, 3.5]

        def bin_x(val):
            if val < x_cuts[0]:
                return 0
            elif val < x_cuts[1]:
                return 1
            elif val < x_cuts[2]:
                return 2
            else:
                return 2

        def bin_z(val):
            if val < z_cuts[0]:
                return 0
            elif val < z_cuts[1]:
                return 1
            elif val < z_cuts[2]:
                return 2
            else:
                return 2

        df["zone_x"] = df["plate_x"].apply(bin_x)
        df["zone_z"] = df["plate_z"].apply(bin_z)

        if "launch_speed_angle" in df.columns:
            df["is_barrel"] = (df["launch_speed_angle"] == 6).astype(int)
        else:
            has_ev = df["launch_speed"].notna() & df["launch_angle"].notna()
            df["is_barrel"] = (
                has_ev
                & (df["launch_speed"] >= 98)
                & (df["launch_angle"].between(26, 30))
            ).astype(int)

        z_data = np.zeros((3, 3))
        text_data = [[""] * 3 for _ in range(3)]
        for row in range(3):
            for col in range(3):
                cell = df[(df["zone_x"] == col) & (df["zone_z"] == row)]
                n = len(cell)
                rate = cell["is_barrel"].mean() if n > 0 else 0.0
                z_data[row][col] = rate
                text_data[row][col] = f"{rate:.1%}<br>n={n}"

    fig = go.Figure(
        go.Heatmap(
            z=z_data,
            text=text_data,
            texttemplate="%{text}",
            colorscale=HEAT_COLORS,
            showscale=True,
            colorbar=dict(
                title="Barrel %",
                tickformat=".1%",
                tickfont=dict(color=TEXT),
                titlefont=dict(color=TEXT),
            ),
            zmin=0,
            zmax=0.15,
        )
    )

    fig.update_layout(
        title=dict(text="Hot Zone (Barrel Rate by Location)", font=dict(color=TEXT)),
        xaxis=dict(
            tickvals=[0, 1, 2],
            ticktext=["Inside", "Middle", "Outside"],
            tickfont=dict(color=TEXT),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickvals=[0, 1, 2],
            ticktext=["Low", "Mid", "High"],
            tickfont=dict(color=TEXT),
            showgrid=False,
            zeroline=False,
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        margin=dict(l=40, r=20, t=40, b=40),
        height=320,
    )
    return fig


# ── career pitch splits ───────────────────────────────────────────────────────

def plot_career_pitch_splits(batter_id: int, season: int | None = None) -> go.Figure:
    """
    Grouped bar chart: barrel rate and whiff rate per pitch type for the
    current season, split by pitcher handedness (vs LHP / vs RHP).
    """
    from mlb_season.pipeline import get_batter_pitch_splits

    season = season or date.today().year
    df = get_batter_pitch_splits(batter_id, season)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=SUBTLE, size=14),
        )
        fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, height=200)
        return fig

    df["pitch_label"] = df["pitch_type"].map(PITCH_LABELS).fillna(df["pitch_type"])
    pitch_labels = df["pitch_label"].unique().tolist()

    traces = []
    hand_colors = {"L": ("#3498db", "#1a6fa3"), "R": (ACCENT, "#a34d14")}
    for hand in ["L", "R"]:
        sub = df[df["vs_hand"] == hand]
        if sub.empty:
            continue

        # Barrel rate bars
        barrel_vals = [
            sub[sub["pitch_label"] == pl]["barrel_rate"].values[0] * 100
            if pl in sub["pitch_label"].values else 0.0
            for pl in pitch_labels
        ]
        traces.append(
            go.Bar(
                name=f"vs {'LHP' if hand == 'L' else 'RHP'} — Barrel%",
                x=pitch_labels,
                y=barrel_vals,
                marker_color=hand_colors[hand][0],
                offsetgroup=hand,
                legendgroup=hand,
            )
        )
        # Whiff rate bars (dashed / lighter)
        whiff_vals = [
            sub[sub["pitch_label"] == pl]["whiff_rate"].values[0] * 100
            if pl in sub["pitch_label"].values else 0.0
            for pl in pitch_labels
        ]
        traces.append(
            go.Bar(
                name=f"vs {'LHP' if hand == 'L' else 'RHP'} — Whiff%",
                x=pitch_labels,
                y=whiff_vals,
                marker_color=hand_colors[hand][1],
                offsetgroup=f"{hand}_whiff",
                legendgroup=f"{hand}_whiff",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="group",
        title=dict(text="Barrel % & Whiff % by Pitch Type (split by pitcher hand)", font=dict(color=TEXT)),
        xaxis=dict(tickfont=dict(color=TEXT), title="Pitch Type"),
        yaxis=dict(tickfont=dict(color=TEXT), title="Rate (%)", ticksuffix="%"),
        legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        margin=dict(l=40, r=20, t=50, b=40),
        height=360,
    )
    return fig


# ── pitch mix simulator ───────────────────────────────────────────────────────

def show_pitch_mix_simulator(
    pitcher_id: int,
    batter_id: int,
    season: int | None = None,
) -> None:
    """
    Interactive Streamlit component. User slides pitcher's pitch mix usage,
    and MAS updates in real time showing how the matchup changes.
    """
    from core.matchup_score import (
        aggregate_to_score,
        compute_batter_z_scores,
        compute_per_pitch_overlap,
        compute_pitcher_z_scores,
        mas_color,
    )
    from core.player_lookup import get_pitcher_handedness
    from mlb_season.pipeline import (
        apply_pitcher_decay_to_arsenal,
        get_batter_pitch_splits,
        get_league_pitch_baselines,
        get_pitcher_arsenal,
    )

    season = season or date.today().year
    pitcher_hand = get_pitcher_handedness(pitcher_id)
    arsenal = get_pitcher_arsenal(pitcher_id, season)
    batter_splits = get_batter_pitch_splits(batter_id, season)
    league_baselines = get_league_pitch_baselines(season)

    if arsenal.empty or batter_splits.empty:
        st.info("Insufficient data for pitch mix simulator.")
        return

    arsenal = apply_pitcher_decay_to_arsenal(arsenal)

    st.markdown("**Adjust pitch mix below to simulate how MAS changes:**")

    pitch_types = arsenal["pitch_type"].tolist()
    default_usage = {
        row["pitch_type"]: float(row["usage_pct"])
        for _, row in arsenal.iterrows()
    }

    cols = st.columns(len(pitch_types))
    simulated_usage: dict[str, float] = {}
    for i, pt in enumerate(pitch_types):
        with cols[i]:
            label = PITCH_LABELS.get(pt, pt)
            simulated_usage[pt] = (
                cols[i].slider(
                    label,
                    min_value=0,
                    max_value=100,
                    value=int(default_usage.get(pt, 0.2) * 100),
                    step=1,
                    key=f"sim_slider_{pitcher_id}_{batter_id}_{pt}",
                )
                / 100.0
            )

    total = sum(simulated_usage.values())
    if total == 0:
        st.warning("All sliders at 0 — adjust the pitch mix.")
        return

    # Normalize to sum to 1
    normalized = {pt: v / total for pt, v in simulated_usage.items()}

    sim_arsenal = arsenal.copy()
    sim_arsenal["usage_pct"] = sim_arsenal["pitch_type"].map(normalized).fillna(0.0)

    batter_z = compute_batter_z_scores(batter_splits, league_baselines)
    pitcher_z = compute_pitcher_z_scores(sim_arsenal, league_baselines)

    if batter_z.empty or pitcher_z.empty:
        st.info("Not enough data to simulate.")
        return

    per_pitch = compute_per_pitch_overlap(batter_z, pitcher_z, pitcher_hand)
    sim_mas = aggregate_to_score(per_pitch, "barrel_overlap")
    sim_whiff = aggregate_to_score(per_pitch, "whiff_overlap")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Simulated MAS", f"{sim_mas:.1f}")
    col_b.metric("Simulated Whiff Score", f"{sim_whiff:.1f}")
    col_c.metric("Mix sums to", f"{total * 100:.0f}%")

    st.markdown(
        f"<div style='width:{sim_mas}%;height:8px;"
        f"background:{mas_color(sim_mas)};border-radius:4px'></div>",
        unsafe_allow_html=True,
    )
