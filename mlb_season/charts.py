"""
mlb_season/charts.py — Season-level Plotly chart builders.

Pitch matchup heatmap, strike-zone hot zone grid, rolling OPS,
K-rate dual chart with league-average reference lines, barrel rate trend,
and the interactive pitch mix simulator widget.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core.config import (
    PITCH_NAMES, PITCH_COLORS,
    SZ_L, SZ_R, SZ_B, SZ_T,
    ZONE_BOXES, ZONE_CENTERS,
)

_DARK = dict(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"))


def _layout(**kwargs):
    base = dict(**_DARK, margin=dict(l=10, r=10, t=50, b=10))
    base.update(kwargs)
    return base


# ── 1. Pitch matchup heatmap ──────────────────────────────────────────────────

def plot_matchup_heatmap(
    batters_splits: dict[str, pd.DataFrame],
    pitcher_arsenal: pd.DataFrame,
    danger_barrel_thresh: float = 12.0,
    danger_usage_thresh:  float = 10.0,
) -> go.Figure:
    """
    Heatmap: rows = batters, cols = pitch types.
    Color = barrel_rate. Text = xBA. Yellow border = dangerous matchup.
    Pitch types the starter doesn't throw are dimmed.
    """
    pitcher_pitches: dict[str, float] = {}
    if not pitcher_arsenal.empty:
        pitcher_pitches = pitcher_arsenal.set_index("pitch_type")["usage_pct"].to_dict()

    # Collect all pitch types — pitcher's first, then others
    all_pts: set[str] = set(pitcher_pitches.keys())
    for df in batters_splits.values():
        if not df.empty and "pitch_type" in df.columns:
            all_pts.update(df["pitch_type"].tolist())

    ordered_pts = (
        sorted(pitcher_pitches, key=lambda x: -pitcher_pitches[x])
        + sorted(p for p in all_pts if p not in pitcher_pitches)
    )

    batter_names = list(batters_splits.keys())
    n_b, n_p = len(batter_names), len(ordered_pts)

    barrel_mat = np.full((n_b, n_p), np.nan)
    xba_mat    = np.full((n_b, n_p), np.nan)
    text_mat   = [[""] * n_p for _ in range(n_b)]

    for i, name in enumerate(batter_names):
        df = batters_splits[name]
        if df.empty:
            continue
        sp = df.set_index("pitch_type").to_dict("index")
        for j, pt in enumerate(ordered_pts):
            if pt not in sp:
                continue
            br = sp[pt].get("barrel_rate", np.nan)
            xb = sp[pt].get("xba", np.nan)
            barrel_mat[i][j] = br
            xba_mat[i][j]    = xb
            usage = pitcher_pitches.get(pt, 0)
            danger = (not np.isnan(br)) and br >= danger_barrel_thresh and usage >= danger_usage_thresh
            xba_str = f".{int(xb*1000):03d}" if not np.isnan(xb) else "—"
            br_str  = f"{br:.1f}%" if not np.isnan(br) else "—"
            text_mat[i][j] = f"xBA {xba_str}<br>Brl {br_str}{'  ⚡' if danger else ''}"

    # X-axis labels: pitch name + usage%
    x_labels = []
    for pt in ordered_pts:
        name = PITCH_NAMES.get(pt, pt)
        usage = pitcher_pitches.get(pt, 0)
        if usage >= 1:
            x_labels.append(f"{name}<br><b>{usage:.0f}%</b>")
        else:
            x_labels.append(f"<span style='opacity:0.4'>{name}</span>")

    fig = go.Figure(go.Heatmap(
        z=barrel_mat,
        x=x_labels,
        y=batter_names,
        text=text_mat,
        texttemplate="%{text}",
        textfont={"size": 9},
        colorscale=[[0, "#1a4d8f"], [0.25, "#5495c4"], [0.5, "#f5f5a0"],
                    [0.75, "#e07020"], [1.0, "#b30000"]],
        zmin=0, zmax=25,
        colorbar=dict(title="Barrel %", len=0.8, thickness=12),
        hovertemplate="<b>%{y}</b> vs %{x}<br>Barrel: %{z:.1f}%<extra></extra>",
    ))

    # Danger borders
    shapes = []
    for i, name in enumerate(batter_names):
        df = batters_splits[name]
        if df.empty:
            continue
        sp = df.set_index("pitch_type").to_dict("index")
        for j, pt in enumerate(ordered_pts):
            if pt not in sp:
                continue
            br = sp[pt].get("barrel_rate", 0) or 0
            if br >= danger_barrel_thresh and pitcher_pitches.get(pt, 0) >= danger_usage_thresh:
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=j - 0.49, x1=j + 0.49,
                    y0=i - 0.49, y1=i + 0.49,
                    line=dict(color="#FFD700", width=2.5),
                ))

    fig.update_layout(
        shapes=shapes,
        title="Pitch Matchup Heatmap — Barrel Rate & xBA by Batter × Pitch Type",
        xaxis=dict(title="Pitch Type (starter usage %)", tickangle=-15, showgrid=False),
        yaxis=dict(title="", showgrid=False),
        height=max(280, 52 * n_b + 120),
        **_layout(margin=dict(l=140, r=60, t=60, b=80)),
    )
    return fig


# ── 2. Strike zone hot zones + pitch location clusters ───────────────────────

def plot_hot_zone_grid(
    hot_zones: dict[int, float],
    pitcher_locations: pd.DataFrame | None = None,
    title: str = "Strike Zone Hot Zones + Pitch Locations",
) -> go.Figure:
    fig = go.Figure()

    # Color zones by xBA (blue cold → white avg → red hot)
    vals = [v for v in hot_zones.values() if v and v > 0]
    z_min = min(vals) if vals else 0.150
    z_max = max(vals) if vals else 0.450

    def xba_color(xba: float) -> str:
        if not xba:
            return "rgba(80,80,80,0.35)"
        t = (xba - z_min) / max(z_max - z_min, 0.001)
        t = max(0.0, min(1.0, t))
        if t >= 0.5:
            s = (t - 0.5) * 2
            r, g, b = int(255), int(255 * (1-s)), int(255 * (1-s))
        else:
            s = (0.5 - t) * 2
            r, g, b = int(255*(1-s)), int(255*(1-s)), 255
        return f"rgba({r},{g},{b},0.75)"

    for z, (xl, xr, zb, zt) in ZONE_BOXES.items():
        xba_val = hot_zones.get(z, 0.0)
        fig.add_shape(type="rect", x0=xl, x1=xr, y0=zb, y1=zt,
                      fillcolor=xba_color(xba_val),
                      line=dict(color="rgba(255,255,255,0.5)", width=0.8))
        if xba_val:
            fig.add_annotation(x=(xl+xr)/2, y=(zb+zt)/2,
                                text=f"<b>.{int(xba_val*1000):03d}</b>",
                                showarrow=False, font=dict(size=13, color="white"))

    # Strike zone outline
    fig.add_shape(type="rect", x0=SZ_L, x1=SZ_R, y0=SZ_B, y1=SZ_T,
                  line=dict(color="white", width=1.8), fillcolor="rgba(0,0,0,0)")

    # Pitcher pitch location clusters
    if pitcher_locations is not None and not pitcher_locations.empty:
        for _, row in pitcher_locations.iterrows():
            pt = row.get("pitch_type", "")
            color = PITCH_COLORS.get(pt, "#aaaaaa")
            usage = row.get("usage_pct", 0)
            name  = PITCH_NAMES.get(pt, row.get("pitch_name", pt))
            fig.add_trace(go.Scatter(
                x=[row["avg_x"]], y=[row["avg_z"]],
                mode="markers",
                marker=dict(size=14 + usage * 0.4, color=color,
                            line=dict(color="white", width=2), opacity=0.9),
                name=f"{name} ({usage:.0f}%)",
                hovertemplate=f"<b>{name}</b><br>Avg loc: ({row['avg_x']:.2f}, {row['avg_z']:.2f})<br>Usage: {usage:.0f}%<extra></extra>",
            ))

    # Home plate marker
    fig.add_shape(type="path",
        path="M -0.708 0.15 L 0 -0.1 L 0.708 0.15 L 0.708 0.35 L -0.708 0.35 Z",
        fillcolor="white", line=dict(color="white"))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[-2.2, 2.2], showgrid=False, zeroline=False,
                   title="Horizontal (ft, catcher's view)"),
        yaxis=dict(range=[-0.3, 4.8], showgrid=False, zeroline=False,
                   title="Height (ft)", scaleanchor="x", scaleratio=1),
        showlegend=True,
        legend=dict(x=1.02, y=0.95, font=dict(size=10)),
        height=520,
        **_layout(margin=dict(l=50, r=160, t=55, b=40)),
    )
    return fig


# ── 3. Rolling OPS line chart ─────────────────────────────────────────────────

def plot_rolling_ops(game_logs: list[dict], batter_name: str, n: int = 15) -> go.Figure:
    """game_logs: list of per-game stat dicts from MLB Stats API game log."""
    if not game_logs:
        return go.Figure(layout=go.Layout(title=f"No game log for {batter_name}", **_layout()))

    df = pd.DataFrame(game_logs).tail(n).reset_index(drop=True)
    if "ops" not in df.columns:
        return go.Figure(layout=go.Layout(title="OPS data unavailable", **_layout()))

    df["ops"] = pd.to_numeric(df["ops"], errors="coerce")
    df["game_num"] = range(1, len(df) + 1)
    roll_avg = df["ops"].rolling(5, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["game_num"], y=df["ops"],
        marker_color=["#e74c3c" if v >= 0.900 else "#e67e22" if v >= 0.750 else "#3498db"
                      for v in df["ops"].fillna(0)],
        name="Game OPS", opacity=0.7,
        hovertemplate="Game %{x}<br>OPS: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["game_num"], y=roll_avg,
        mode="lines", line=dict(color="white", width=2),
        name="5-game avg",
    ))
    fig.update_layout(
        title=f"OPS — Last {n} Games — {batter_name}",
        xaxis=dict(title="Game", showgrid=False),
        yaxis=dict(title="OPS"),
        height=300,
        **_layout(margin=dict(l=60, r=20, t=55, b=40)),
    )
    return fig


# ── 4. K-rate dual line chart ─────────────────────────────────────────────────

def plot_krate_chart(
    batter_logs: list[dict],
    pitcher_logs: list[dict],
    batter_name: str,
    pitcher_name: str,
    n: int = 15,
    league_avg_kpct: float | None = None,   # MLB avg batter K% (e.g. 22.5)
    league_avg_k9: float | None = None,     # MLB avg pitcher K/9 (e.g. 8.5)
) -> go.Figure:
    fig = go.Figure()

    n_games_plotted = 1  # track x range for reference lines

    if batter_logs:
        df_b = pd.DataFrame(batter_logs).tail(n).reset_index(drop=True)
        if "strikeOuts" in df_b.columns and "plateAppearances" in df_b.columns:
            df_b["k_rate"] = (
                pd.to_numeric(df_b["strikeOuts"], errors="coerce")
                / pd.to_numeric(df_b["plateAppearances"], errors="coerce").replace(0, np.nan)
                * 100
            )
            n_games_plotted = max(n_games_plotted, len(df_b))
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df_b) + 1)), y=df_b["k_rate"],
                mode="lines+markers", line=dict(color="#e74c3c", width=2),
                name=f"{batter_name} K%",
                hovertemplate="Game %{x}<br>K%%: %{y:.1f}%%<extra></extra>",
            ))

    if pitcher_logs:
        df_p = pd.DataFrame(pitcher_logs).tail(n).reset_index(drop=True)
        if "strikeOuts" in df_p.columns and "inningsPitched" in df_p.columns:
            ip = pd.to_numeric(df_p["inningsPitched"], errors="coerce").replace(0, np.nan)
            k9 = pd.to_numeric(df_p["strikeOuts"], errors="coerce") / ip * 9
            n_games_plotted = max(n_games_plotted, len(df_p))
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df_p) + 1)), y=k9,
                mode="lines+markers", line=dict(color="#3498db", width=2),
                name=f"{pitcher_name} K/9", yaxis="y2",
                hovertemplate="Start %{x}<br>K/9: %{y:.1f}<extra></extra>",
            ))

    # League average reference lines
    x_range = list(range(1, n_games_plotted + 1))
    if league_avg_kpct is not None and batter_logs:
        fig.add_trace(go.Scatter(
            x=x_range, y=[league_avg_kpct] * len(x_range),
            mode="lines",
            line=dict(color="#e74c3c", width=1.5, dash="dot"),
            name=f"Lg Avg K% ({league_avg_kpct:.1f}%)",
            opacity=0.55,
            hovertemplate=f"MLB Avg K%: {league_avg_kpct:.1f}%<extra></extra>",
        ))
    if league_avg_k9 is not None and pitcher_logs:
        fig.add_trace(go.Scatter(
            x=x_range, y=[league_avg_k9] * len(x_range),
            mode="lines",
            line=dict(color="#3498db", width=1.5, dash="dot"),
            name=f"Lg Avg K/9 ({league_avg_k9:.1f})",
            yaxis="y2",
            opacity=0.55,
            hovertemplate=f"MLB Avg K/9: {league_avg_k9:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title=f"K-Rate — {batter_name} K% vs {pitcher_name} K/9",
        xaxis=dict(title="Game #", showgrid=False),
        yaxis=dict(title="Batter K%", side="left"),
        yaxis2=dict(title="Pitcher K/9", side="right", overlaying="y"),
        height=300,
        legend=dict(x=0, y=1.12, orientation="h"),
        **_layout(margin=dict(l=60, r=60, t=60, b=40)),
    )
    return fig


# ── 5. Barrel rate trend ──────────────────────────────────────────────────────

def plot_barrel_trend(barrel_df: pd.DataFrame, batter_name: str) -> go.Figure:
    if barrel_df.empty:
        return go.Figure(layout=go.Layout(title=f"No barrel data for {batter_name}", **_layout()))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=barrel_df["date"].dt.strftime("%m/%d"),
        y=barrel_df["barrel_rate"],
        marker_color=["#e74c3c" if v > 15 else "#e67e22" if v > 8 else "#3498db"
                      for v in barrel_df["barrel_rate"]],
        name="Barrel %",
        hovertemplate="%{x}<br>Barrel%: %{y:.1f}%<br>(%{customdata} BIP)<extra></extra>",
        customdata=barrel_df["bip"],
    ))
    roll = barrel_df["barrel_rate"].rolling(3, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=barrel_df["date"].dt.strftime("%m/%d"), y=roll,
        mode="lines", line=dict(color="white", width=2), name="3-game avg",
    ))
    fig.update_layout(
        title=f"Barrel Rate Trend — {batter_name}",
        xaxis=dict(title="Date", showgrid=False, tickangle=-40),
        yaxis=dict(title="Barrel %"),
        height=280,
        **_layout(margin=dict(l=60, r=20, t=55, b=60)),
    )
    return fig


# ── 6. Pitch mix simulator (Streamlit widget) ─────────────────────────────────

def show_pitch_mix_simulator(
    batters_splits: dict[str, pd.DataFrame],
    pitcher_arsenal: pd.DataFrame,
    key_prefix: str = "mixer",
):
    if pitcher_arsenal.empty:
        st.info("No pitcher arsenal data available for simulation.")
        return

    pitcher_pitches = pitcher_arsenal.set_index("pitch_type")["usage_pct"].to_dict()
    ordered = sorted(pitcher_pitches, key=lambda x: -pitcher_pitches[x])

    st.markdown("**Adjust Pitch Mix** (drag sliders — total auto-normalizes)")
    n_cols = min(len(ordered), 5)
    cols = st.columns(n_cols)
    raw: dict[str, float] = {}
    for i, pt in enumerate(ordered):
        nm = PITCH_NAMES.get(pt, pt)
        with cols[i % n_cols]:
            raw[pt] = st.slider(nm, 0.0, 100.0, float(pitcher_pitches[pt]),
                                 step=1.0, key=f"{key_prefix}_{pt}")

    total = sum(raw.values()) or 1
    adj_mix = {pt: v / total * 100 for pt, v in raw.items()}

    # Bar chart of adjusted mix
    mix_fig = go.Figure(go.Bar(
        x=[PITCH_NAMES.get(pt, pt) for pt in adj_mix],
        y=list(adj_mix.values()),
        marker_color=[PITCH_COLORS.get(pt, "#aaa") for pt in adj_mix],
        text=[f"{v:.1f}%" for v in adj_mix.values()],
        textposition="outside",
    ))
    mix_fig.update_layout(
        title="Adjusted Mix", yaxis_title="%", height=260,
        **_layout(margin=dict(l=40, r=20, t=45, b=40)),
    )
    st.plotly_chart(mix_fig, use_container_width=True)

    # Expected barrel probability per batter
    results = []
    for batter, df in batters_splits.items():
        if df.empty:
            continue
        sp = df.set_index("pitch_type")[["barrel_rate"]].to_dict("index")
        adj_br   = sum((adj_mix.get(pt, 0)/100) * (sp.get(pt, {}).get("barrel_rate") or 0)
                       for pt in adj_mix)
        base_br  = sum((pitcher_pitches.get(pt, 0)/100) * (sp.get(pt, {}).get("barrel_rate") or 0)
                       for pt in pitcher_pitches)
        results.append({
            "Batter": batter,
            "Adj Barrel%": round(adj_br, 2),
            "Base Barrel%": round(base_br, 2),
            "Δ": round(adj_br - base_br, 2),
        })

    if not results:
        st.caption("Select batters in the Scout tab to see projections.")
        return

    res_df = pd.DataFrame(results).sort_values("Adj Barrel%", ascending=False)

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        name="Baseline", x=res_df["Batter"], y=res_df["Base Barrel%"],
        marker_color="#4a8fc4", opacity=0.65,
    ))
    bar_fig.add_trace(go.Bar(
        name="Adjusted", x=res_df["Batter"], y=res_df["Adj Barrel%"],
        marker_color="#e05030",
        text=[f"{'+' if d>0 else ''}{d:.1f}%" for d in res_df["Δ"]],
        textposition="outside",
    ))
    bar_fig.update_layout(
        title="Expected Barrel Probability — Baseline vs Adjusted Mix",
        yaxis_title="Expected Barrel %",
        barmode="group",
        height=360,
        legend=dict(x=0, y=1.1, orientation="h"),
        **_layout(margin=dict(l=60, r=20, t=70, b=120)),
        xaxis=dict(tickangle=-30, showgrid=False),
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # Flag dangerous matchups
    danger = res_df[res_df["Adj Barrel%"] > 8.0]
    if not danger.empty:
        st.markdown("**⚡ High-Risk Matchups** (adj barrel% > 8%)")
        st.dataframe(danger.reset_index(drop=True), use_container_width=True, hide_index=True)


# ── 7. Career pitch-type split overview ──────────────────────────────────────

_STAT_META: list[dict] = [
    {"col": "whiff_rate",    "label": "Whiff%",     "color": "#e74c3c", "hi_bad": True,  "fmt": ".1f", "suffix": "%"},
    {"col": "k_rate",        "label": "K%",          "color": "#e67e22", "hi_bad": True,  "fmt": ".1f", "suffix": "%"},
    {"col": "barrel_rate",   "label": "Barrel%",     "color": "#f1c40f", "hi_bad": False, "fmt": ".1f", "suffix": "%"},
    {"col": "hard_hit_rate", "label": "Hard Hit%",   "color": "#2ecc71", "hi_bad": False, "fmt": ".1f", "suffix": "%"},
    {"col": "avg_ev",        "label": "Avg EV",      "color": "#3498db", "hi_bad": False, "fmt": ".1f", "suffix": " mph"},
    {"col": "xba",           "label": "xBA",         "color": "#9b59b6", "hi_bad": False, "fmt": ".3f", "suffix": ""},
    {"col": "xwoba",         "label": "xwOBA",       "color": "#1abc9c", "hi_bad": False, "fmt": ".3f", "suffix": ""},
]


def plot_career_pitch_splits(career_df: pd.DataFrame, batter_name: str) -> go.Figure:
    """
    Grouped bar chart — one group per pitch type, one bar per metric.
    Shows career aggregated pitch-type split stats for a batter.
    All metrics are normalized to [0,1] across pitch types for comparability,
    with the actual value shown in the hover tooltip.
    """
    if career_df.empty:
        return go.Figure(layout=go.Layout(title=f"No career data for {batter_name}", **_layout()))

    df = career_df.copy()
    pitch_types = df["pitch_name"].tolist()
    n_groups    = len(pitch_types)

    fig = go.Figure()

    for meta in _STAT_META:
        col = meta["col"]
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().all():
            continue

        # Normalize 0→1 for visual comparison; keep raw for tooltip
        v_min, v_max = vals.min(), vals.max()
        norm = (vals - v_min) / (v_max - v_min + 1e-9)

        # Invert bad-is-high metrics so green = good across the board
        if meta["hi_bad"]:
            norm = 1 - norm

        hover = [
            f"<b>{pitch_types[i]}</b><br>{meta['label']}: "
            + (f"{v:{meta['fmt']}}{meta['suffix']}" if not np.isnan(v) else "N/A")
            for i, v in enumerate(vals)
        ]

        fig.add_trace(go.Bar(
            name=meta["label"],
            x=pitch_types,
            y=norm.fillna(0),
            marker_color=meta["color"],
            opacity=0.82,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
            text=[
                f"{v:{meta['fmt']}}{meta['suffix']}" if not np.isnan(v) else ""
                for v in vals
            ],
            textposition="outside",
            textfont=dict(size=8),
        ))

    fig.update_layout(
        barmode="group",
        title=f"Career Pitch-Type Profile — {batter_name}",
        xaxis=dict(title="Pitch Type", showgrid=False),
        yaxis=dict(
            title="Relative Score (normalized per metric)",
            showgrid=False, showticklabels=False,
            range=[0, 1.25],
        ),
        height=400,
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=9)),
        **_layout(margin=dict(l=50, r=20, t=75, b=60)),
    )
    return fig


def plot_pitch_type_season_trend(
    by_season_df: pd.DataFrame,
    pitch_type_name: str,
    batter_name: str,
) -> go.Figure:
    """
    Year-over-year line chart for a specific pitch type.
    Shows whiff%, barrel%, xBA, and (if available) xwOBA across seasons.
    """
    if by_season_df.empty:
        return go.Figure(layout=go.Layout(
            title=f"No season data for {batter_name} vs {pitch_type_name}", **_layout()))

    df = by_season_df[by_season_df["pitch_name"] == pitch_type_name].copy()
    if df.empty or df["season"].nunique() < 2:
        return go.Figure(layout=go.Layout(
            title=f"Insufficient seasons for {pitch_type_name} trend", **_layout()))

    df = df.sort_values("season")
    seasons = df["season"].astype(str).tolist()

    fig = go.Figure()

    trend_metrics = [
        ("whiff_rate",  "Whiff%",   "#e74c3c", "y",  ".1f", "%"),
        ("barrel_rate", "Barrel%",  "#f1c40f", "y",  ".1f", "%"),
        ("xba",         "xBA",      "#9b59b6", "y2", ".3f", ""),
        ("xwoba",       "xwOBA",    "#1abc9c", "y2", ".3f", ""),
    ]

    for col, label, color, yax, fmt, suf in trend_metrics:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().all():
            continue

        hover = [
            f"<b>{s}</b><br>{label}: {f'{v:{fmt}}{suf}' if not np.isnan(v) else 'N/A'}"
            for s, v in zip(seasons, vals)
        ]

        fig.add_trace(go.Scatter(
            x=seasons, y=vals,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=8, symbol="circle"),
            name=label,
            yaxis=yax,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
            connectgaps=True,
        ))

    # Pitches-seen bar (secondary context)
    if "pitches" in df.columns:
        fig.add_trace(go.Bar(
            x=seasons, y=df["pitches"],
            name="Pitches seen",
            marker_color="rgba(180,180,180,0.18)",
            yaxis="y3",
            hovertemplate="<b>%{x}</b><br>Pitches: %{y}<extra></extra>",
        ))

    fig.update_layout(
        title=f"{batter_name} vs {pitch_type_name} — Season-by-Season Trend",
        xaxis=dict(title="Season", showgrid=False),
        yaxis=dict(title="Rate (%)", side="left",  showgrid=False),
        yaxis2=dict(title="Expected Stat", side="right", overlaying="y",
                    showgrid=False, tickformat=".3f"),
        yaxis3=dict(overlaying="y", side="right", showgrid=False,
                    showticklabels=False, range=[0, df["pitches"].max() * 5]
                    if "pitches" in df.columns else None),
        legend=dict(orientation="h", x=0, y=1.1, font=dict(size=9)),
        height=340,
        **_layout(margin=dict(l=60, r=70, t=70, b=50)),
    )
    return fig


def plot_career_splits_table_fig(career_df: pd.DataFrame, batter_name: str) -> go.Figure:
    """
    Colour-coded table of career pitch-type stats.
    Each cell is coloured by percentile within its column (green = good for batter).
    """
    if career_df.empty:
        return go.Figure(layout=go.Layout(title=f"No data for {batter_name}", **_layout()))

    display_cols = ["pitch_name", "pitches", "whiff_rate", "k_rate",
                    "barrel_rate", "hard_hit_rate", "avg_ev", "xba", "xwoba"]
    col_labels   = ["Pitch", "Pitches", "Whiff%", "K%",
                    "Barrel%", "Hard Hit%", "Avg EV", "xBA", "xwOBA"]
    hi_bad_cols  = {"whiff_rate", "k_rate"}

    df = career_df[[c for c in display_cols if c in career_df.columns]].copy()

    cell_colors: list[list[str]] = []
    cell_texts:  list[list[str]] = []

    for i, col in enumerate(df.columns):
        col_texts  = []
        col_colors = []

        if col == "pitch_name":
            for v in df[col]:
                col_texts.append(str(v))
                col_colors.append("rgba(20,30,50,0.9)")
            cell_texts.append(col_texts)
            cell_colors.append(col_colors)
            continue

        vals = pd.to_numeric(df[col], errors="coerce")
        v_min, v_max = vals.min(), vals.max()

        for v in vals:
            if np.isnan(v):
                col_texts.append("—")
                col_colors.append("rgba(20,30,50,0.9)")
                continue

            # Format
            meta_match = next((m for m in _STAT_META if m["col"] == col), None)
            if meta_match:
                fmt, suf = meta_match["fmt"], meta_match["suffix"]
                col_texts.append(f"{v:{fmt}}{suf}")
            elif col in ("pitches", "bip"):
                col_texts.append(str(int(v)))
            else:
                col_texts.append(f"{v:.2f}")

            # Color: percentile-based (0=worst,1=best for batter)
            pct = (v - v_min) / (v_max - v_min + 1e-9)
            if col in hi_bad_cols:
                pct = 1 - pct   # invert: lower = better
            r = int(255 * (1 - pct) * 0.6 + 10)
            g_val = int(255 * pct * 0.7 + 20)
            b_val = int(40)
            col_colors.append(f"rgba({r},{g_val},{b_val},0.75)")

        cell_texts.append(col_texts)
        cell_colors.append(col_colors)

    actual_labels = col_labels[:len(df.columns)]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{l}</b>" for l in actual_labels],
            fill_color="rgba(30,50,80,0.95)",
            align="center",
            font=dict(color="white", size=11),
            line_color="rgba(255,255,255,0.1)",
            height=32,
        ),
        cells=dict(
            values=cell_texts,
            fill_color=cell_colors,
            align=["left"] + ["center"] * (len(df.columns) - 1),
            font=dict(color="white", size=10),
            line_color="rgba(255,255,255,0.06)",
            height=28,
        ),
    ))
    fig.update_layout(
        title=f"Career Pitch-Type Stats — {batter_name}  "
              f"<span style='font-size:10px;color:#888'>"
              f"(green = favorable for batter)</span>",
        height=max(200, 32 + 28 * len(df) + 60),
        **_layout(margin=dict(l=10, r=10, t=55, b=10)),
    )
    return fig
