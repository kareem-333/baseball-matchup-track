"""
nba/charts.py — NBA Defensive Analytics chart builders.

Dragon Index & Fortress Rating visualizations.

Column conventions (matching nba/pipeline.py):
  User-facing: DRAGON_INDEX, FORTRESS_RATING, COMBINED_SCORE, POSITION, TEAM_COLOR, …
  Components:  d_defl_n, d_charges_n, d_steals_n, d_perim_n, d_loose_n
               f_rim_inv_n, f_reb_n, f_blocks_n, f_rim_rate_n, f_putback_n
"""

import base64
import io
import requests

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw

from nba.pipeline import nba_headshot_url, team_color, get_play_sequence_stats

# ── Shared dark theme ─────────────────────────────────────────────────────────

_DARK = dict(
    plot_bgcolor  = "#0e1117",
    paper_bgcolor = "#0e1117",
    font          = dict(color="#e0e0e0", family="Inter, Arial, sans-serif"),
)

def _layout(**kw):
    base = dict(**_DARK, margin=dict(l=20, r=20, t=55, b=20))
    base.update(kw)
    return base


_DRAGON_COMP_COLORS   = ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#3498db"]
_FORTRESS_COMP_COLORS = ["#9b59b6", "#1abc9c", "#2980b9", "#8e44ad", "#16a085"]

DRAGON_COMP_LABELS   = ["Deflections", "Charges", "Steals", "Perimeter Contests", "Loose Balls"]
FORTRESS_COMP_LABELS = ["Rim FG% Inv", "Box-Out Rate", "Blocks", "Rim Contests", "Putbacks"]


# ── Headshot helpers ──────────────────────────────────────────────────────────

_headshot_cache: dict[int, str] = {}


def _circular_headshot_b64(player_id: int, border_hex: str = "#888888", size: int = 80) -> str | None:
    """Fetch headshot from CDN, clip to circle with team-colour ring, return base64 data URI."""
    if player_id in _headshot_cache:
        return _headshot_cache[player_id]
    try:
        url = nba_headshot_url(player_id)
        r   = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None

        img = Image.open(io.BytesIO(r.content)).convert("RGBA").resize(
            (size, size), Image.LANCZOS
        )

        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, size, size], fill=255)

        canvas = Image.new("RGBA", (size + 6, size + 6), (0, 0, 0, 0))
        ring   = Image.new("RGBA", (size + 6, size + 6), (0, 0, 0, 0))
        hex_   = border_hex.lstrip("#")
        rc, gc, bc = int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)
        ImageDraw.Draw(ring).ellipse([0, 0, size + 5, size + 5], fill=(rc, gc, bc, 220))
        canvas.paste(ring, (0, 0))
        canvas.paste(img, (3, 3), mask)

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        _headshot_cache[player_id] = b64
        return b64
    except Exception:
        return None


# ── 1. Bubble Scatter ─────────────────────────────────────────────────────────

def plot_bubble_scatter(
    df: pd.DataFrame,
    position_filter: str | None = None,
    highlight_ids: list[int] | None = None,
) -> go.Figure:
    """
    Dragon Index (X) vs Fortress Rating (Y).
    Bubble size ∝ MIN. Team-colour rings. Headshots for top-10 by COMBINED_SCORE.
    """
    if df.empty or "DRAGON_INDEX" not in df.columns:
        return go.Figure(layout=go.Layout(title="No data available", **_layout()))

    filt = df.copy()
    if position_filter and "POSITION" in filt.columns:
        filt = filt[filt["POSITION"] == position_filter]
        if filt.empty:
            return go.Figure(layout=go.Layout(
                title=f"No data for position: {position_filter}", **_layout()))

    x_mid = filt["DRAGON_INDEX"].median()
    y_mid = filt["FORTRESS_RATING"].median()

    fig = go.Figure()

    x_max = filt["DRAGON_INDEX"].max() + 3
    y_max = filt["FORTRESS_RATING"].max() + 3
    quads = [
        (x_mid, x_max, y_mid, y_max, "Complete Defender",    "rgba(46,204,113,0.07)"),
        (0,     x_mid, y_mid, y_max, "Anchor",               "rgba(52,152,219,0.07)"),
        (x_mid, x_max, 0,     y_mid, "Disruptor",            "rgba(231,76,60,0.07)"),
        (0,     x_mid, 0,     y_mid, "Offensive Specialist", "rgba(255,255,255,0.03)"),
    ]
    for x0, x1, y0, y1, label, color in quads:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=color, line_width=0, layer="below")
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=f"<b>{label}</b>",
                           showarrow=False, font=dict(size=11, color="rgba(220,220,220,0.30)"))

    fig.add_hline(y=y_mid, line_dash="dot", line_color="rgba(255,255,255,0.15)")
    fig.add_vline(x=x_mid, line_dash="dot", line_color="rgba(255,255,255,0.15)")

    highlight_ids = set(highlight_ids or [])
    top10_ids     = set(filt.nlargest(10, "COMBINED_SCORE")["PLAYER_ID"].tolist())

    groups = {
        "highlighted": filt[filt["PLAYER_ID"].isin(highlight_ids)],
        "top10":       filt[filt["PLAYER_ID"].isin(top10_ids) & ~filt["PLAYER_ID"].isin(highlight_ids)],
        "rest":        filt[~filt["PLAYER_ID"].isin(top10_ids) & ~filt["PLAYER_ID"].isin(highlight_ids)],
    }
    group_styles = {
        "highlighted": dict(opacity=1.0,  line_width=3, line_color="white"),
        "top10":       dict(opacity=0.90, line_width=1, line_color=None),
        "rest":        dict(opacity=0.55, line_width=0, line_color=None),
    }

    for gname, gdf in groups.items():
        if gdf.empty:
            continue
        style = group_styles[gname]
        customdata = gdf[[
            "PLAYER_NAME", "TEAM_ABBREVIATION",
            "DRAGON_INDEX", "FORTRESS_RATING",
            "MIN", "POSITION", "PLAYER_ID",
        ]].values.tolist()
        sizes  = [max(8, min(40, float(r.get("MIN", 20)) * 0.55)) for _, r in gdf.iterrows()]
        colors = gdf["TEAM_COLOR"].tolist()
        lcolors = ["white"] * len(gdf) if style["line_color"] == "white" else colors
        fig.add_trace(go.Scatter(
            x=gdf["DRAGON_INDEX"].tolist(), y=gdf["FORTRESS_RATING"].tolist(),
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=style["opacity"],
                        line=dict(color=lcolors, width=style["line_width"])),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                "Dragon: %{customdata[2]:.1f}  |  Fortress: %{customdata[3]:.1f}<br>"
                "MPG: %{customdata[4]:.1f}  |  %{customdata[5]}"
                "<extra></extra>"
            ),
            showlegend=False, name=gname,
        ))

    top10   = filt.nlargest(10, "COMBINED_SCORE")
    x_range = filt["DRAGON_INDEX"].max()    - filt["DRAGON_INDEX"].min()
    y_range = filt["FORTRESS_RATING"].max() - filt["FORTRESS_RATING"].min()

    for _, row in top10.iterrows():
        tcolor  = row.get("TEAM_COLOR", "#888888")
        img_b64 = _circular_headshot_b64(int(row["PLAYER_ID"]), border_hex=tcolor, size=60)
        if img_b64:
            fig.add_layout_image(dict(
                source=img_b64, xref="x", yref="y",
                x=row["DRAGON_INDEX"], y=row["FORTRESS_RATING"],
                sizex=max(x_range * 0.07, 5), sizey=max(y_range * 0.11, 7),
                xanchor="center", yanchor="middle", layer="above",
            ))
        else:
            fig.add_annotation(
                x=row["DRAGON_INDEX"], y=row["FORTRESS_RATING"] + y_range * 0.03,
                text=row["PLAYER_NAME"].split()[-1], showarrow=False,
                font=dict(size=8, color="white"), bgcolor=tcolor, opacity=0.85,
            )

    fig.update_layout(
        title="Dragon Index vs Fortress Rating",
        xaxis=dict(title="Dragon Index  (Active Disruption)", showgrid=False,
                   range=[filt["DRAGON_INDEX"].min() - 3, x_max]),
        yaxis=dict(title="Fortress Rating  (Interior Anchor)", showgrid=False,
                   range=[filt["FORTRESS_RATING"].min() - 3, y_max]),
        height=640,
        **_layout(margin=dict(l=60, r=20, t=60, b=60)),
    )
    return fig


# ── 2 & 3. Leaderboard stacked bars ──────────────────────────────────────────

def plot_leaderboard(
    df: pd.DataFrame,
    metric: str = "DRAGON_INDEX",
    n: int = 20,
) -> go.Figure:
    """Horizontal stacked bar leaderboard (top-N). Each segment = one normalised component."""
    if df.empty:
        return go.Figure(layout=go.Layout(**_layout()))

    is_dragon   = metric.upper() in ("DRAGON_INDEX", "DRAGON")
    score_col   = "DRAGON_INDEX"   if is_dragon else "FORTRESS_RATING"
    comp_cols   = (["d_defl_n","d_charges_n","d_steals_n","d_perim_n","d_loose_n"]
                   if is_dragon else
                   ["f_rim_inv_n","f_reb_n","f_blocks_n","f_rim_rate_n","f_putback_n"])
    comp_labels = DRAGON_COMP_LABELS if is_dragon else FORTRESS_COMP_LABELS
    colors      = _DRAGON_COMP_COLORS if is_dragon else _FORTRESS_COMP_COLORS

    if score_col not in df.columns:
        return go.Figure(layout=go.Layout(title=f"{score_col} not found", **_layout()))

    avail_comps = [c for c in comp_cols if c in df.columns]
    keep_cols   = [c for c in ["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","TEAM_COLOR",score_col]
                   + avail_comps if c in df.columns]

    top = (df.nlargest(n, score_col)[keep_cols]
             .sort_values(score_col).reset_index(drop=True))
    y_labels = [f"{r['PLAYER_NAME']} ({r['TEAM_ABBREVIATION']})" for _, r in top.iterrows()]

    fig = go.Figure()
    for col, label, color in zip(avail_comps, comp_labels, colors):
        vals = top[col] * 20
        fig.add_trace(go.Bar(
            orientation="h", x=vals, y=y_labels,
            name=label, marker_color=color, opacity=0.85,
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}<extra></extra>",
        ))

    for i, (_, row) in enumerate(top.iterrows()):
        seg_total = top[[c for c in avail_comps]].iloc[i].sum() * 20
        fig.add_annotation(x=seg_total + 0.5, y=y_labels[i],
                           text=f"<b>{row[score_col]:.0f}</b>",
                           showarrow=False, font=dict(size=9, color="white"))

    title = "Dragon Index — Top 20 Disruptors" if is_dragon else "Fortress Rating — Top 20 Interior Anchors"
    fig.update_layout(
        barmode="stack", title=title,
        xaxis=dict(title="Component Score (normalised)", showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        height=max(420, 28 * n + 80),
        legend=dict(orientation="h", x=0, y=1.04, font=dict(size=9)),
        **_layout(margin=dict(l=210, r=80, t=80, b=30)),
    )
    return fig


# ── 4a. Comparison radar ──────────────────────────────────────────────────────

_RADAR_CATS = [
    "Deflections", "Charges/Steals", "Perimeter Contests",
    "Blocks", "Box-Out Rate", "Putbacks",
]


def _radar_values(row: pd.Series) -> list[float]:
    return [
        float(row.get("d_defl_n",    0) * 100),
        float((row.get("d_charges_n", 0) + row.get("d_steals_n", 0)) / 2 * 100),
        float(row.get("d_perim_n",   0) * 100),
        float(row.get("f_blocks_n",  0) * 100),
        float(row.get("f_reb_n",     0) * 100),
        float(row.get("f_putback_n", 0) * 100),
    ]


def plot_comparison_radar(players_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing up to 3 players across all metric components."""
    if players_df.empty:
        return go.Figure(layout=go.Layout(title="No players selected", **_layout()))

    categories = _RADAR_CATS + [_RADAR_CATS[0]]
    fig = go.Figure()
    fallback_colors = ["#EB6E1F", "#3498db", "#2ecc71"]

    for i, (_, row) in enumerate(players_df.head(3).iterrows()):
        vals   = _radar_values(row) + [_radar_values(row)[0]]
        color  = row.get("TEAM_COLOR", fallback_colors[i % 3])
        if color.startswith("#") and len(color) == 7:
            r2, g2, b2 = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fill = f"rgba({r2},{g2},{b2},0.15)"
        else:
            fill = "rgba(128,128,128,0.15)"

        name = row.get("PLAYER_NAME", f"Player {i+1}")
        team = row.get("TEAM_ABBREVIATION", "")
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories,
            fill="toself", fillcolor=fill,
            line=dict(color=color, width=2),
            name=f"{name} ({team})" if team else name,
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=8, color="#aaa"),
                            gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="#0e1117",
        ),
        title="Defensive Component Comparison",
        showlegend=True, height=440,
        **_layout(margin=dict(l=50, r=50, t=70, b=50)),
    )
    return fig


# ── 4b. Comparison rolling trend ─────────────────────────────────────────────

def plot_comparison_rolling(rolling_data: dict[str, pd.DataFrame], metric: str) -> go.Figure:
    """Dual-line rolling trend for Dragon or Fortress across last 15 games."""
    fig = go.Figure()
    fallback_colors = ["#EB6E1F", "#3498db", "#2ecc71"]
    is_dragon = metric.upper() in ("DRAGON_INDEX", "DRAGON")
    label     = "Dragon Index" if is_dragon else "Fortress Rating"

    for i, (name, df) in enumerate(rolling_data.items()):
        if df.empty or "score" not in df.columns:
            continue
        color = fallback_colors[i % 3]
        x     = df.get("game_num", range(len(df)))
        fig.add_trace(go.Scatter(
            x=x, y=df["score"], mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=5),
            name=name,
            hovertemplate=f"<b>{name}</b><br>Game %{{x}}: %{{y:.1f}}<extra></extra>",
        ))
        roll = df["score"].rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=x, y=roll, mode="lines",
            line=dict(color=color, width=1.5, dash="dash"),
            showlegend=False, opacity=0.55,
        ))

    fig.update_layout(
        title=f"Rolling {label} — Last 15 Games",
        xaxis=dict(title="Game #", showgrid=False),
        yaxis=dict(title=label),
        height=320,
        **_layout(margin=dict(l=60, r=20, t=60, b=40)),
    )
    return fig


# ── 5. Team bubble view ───────────────────────────────────────────────────────

def plot_team_bubbles(team_df: pd.DataFrame) -> go.Figure:
    """Team aggregate bubble: Dragon avg (X) vs Fortress avg (Y)."""
    if team_df.empty or "DRAGON_INDEX" not in team_df.columns:
        return go.Figure(layout=go.Layout(title="No team data", **_layout()))

    x_mid = team_df["DRAGON_INDEX"].median()
    y_mid = team_df["FORTRESS_RATING"].median()
    x_max = team_df["DRAGON_INDEX"].max() + 3
    y_max = team_df["FORTRESS_RATING"].max() + 3

    fig = go.Figure()
    quads = [
        (x_mid, x_max, y_mid, y_max, "Complete Defense", "rgba(46,204,113,0.07)"),
        (0,     x_mid, y_mid, y_max, "Fortress System",  "rgba(52,152,219,0.07)"),
        (x_mid, x_max, 0,     y_mid, "Dragon System",    "rgba(231,76,60,0.07)"),
        (0,     x_mid, 0,     y_mid, "Rebuilding",       "rgba(255,255,255,0.03)"),
    ]
    for x0, x1, y0, y1, label, color in quads:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=color, line_width=0, layer="below")
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=f"<b>{label}</b>",
                           showarrow=False, font=dict(size=11, color="rgba(220,220,220,0.30)"))

    fig.add_hline(y=y_mid, line_dash="dot", line_color="rgba(255,255,255,0.15)")
    fig.add_vline(x=x_mid, line_dash="dot", line_color="rgba(255,255,255,0.15)")

    for _, row in team_df.iterrows():
        avg_min = float(row.get("AVG_MIN", 20))
        tcolor  = row.get("TEAM_COLOR", "#888888")
        abbr    = row.get("TEAM_ABBREVIATION", "?")
        size    = max(20, min(55, avg_min * 1.8))
        fig.add_trace(go.Scatter(
            x=[row["DRAGON_INDEX"]], y=[row["FORTRESS_RATING"]],
            mode="markers+text",
            marker=dict(size=size, color=tcolor, opacity=0.85,
                        line=dict(color="white", width=1.5)),
            text=[abbr], textposition="middle center",
            textfont=dict(size=9, color="white", family="Arial Black"),
            name=row.get("TEAM_NAME", abbr),
            hovertemplate=(
                f"<b>{row.get('TEAM_NAME', abbr)}</b><br>"
                f"Dragon: {row['DRAGON_INDEX']:.1f}<br>"
                f"Fortress: {row['FORTRESS_RATING']:.1f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title="Team Defensive Identity — Dragon vs Fortress",
        xaxis=dict(title="Dragon Index (Team Avg)", showgrid=False,
                   range=[team_df["DRAGON_INDEX"].min() - 3, x_max]),
        yaxis=dict(title="Fortress Rating (Team Avg)", showgrid=False,
                   range=[team_df["FORTRESS_RATING"].min() - 3, y_max]),
        height=580,
        **_layout(margin=dict(l=60, r=20, t=60, b=60)),
    )
    return fig


# ── 6. Steal → Points chain Sankey ───────────────────────────────────────────

def plot_steal_chain_sankey(seq_stats: dict, player_name: str) -> go.Figure:
    """Sankey: Steals → scored/not → fast break / half-court."""
    if not seq_stats or seq_stats.get("total_steals", 0) == 0:
        return go.Figure(layout=go.Layout(
            title=f"{player_name} — No steal data available", **_layout()))

    n          = seq_stats["total_steals"]
    scored     = seq_stats["steals_with_score"]
    no_score   = n - scored
    fast_break = seq_stats["fast_break_steals"]
    half_court = scored - fast_break

    labels  = [f"Steals<br>({n})", f"Scored<br>({scored})", f"No Score<br>({no_score})",
               f"Fast Break<br>({fast_break})", f"Half-Court<br>({half_court})"]
    sources = [0, 0, 1, 1]
    targets = [1, 2, 3, 4]
    values  = [scored, max(no_score, 0), max(fast_break, 0), max(half_court, 0)]
    colors  = ["rgba(46,204,113,0.6)", "rgba(231,76,60,0.5)",
               "rgba(241,196,15,0.7)", "rgba(52,152,219,0.6)"]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=22,
                  line=dict(color="#222", width=0.5), label=labels,
                  color=["#888888","#2ecc71","#e74c3c","#f1c40f","#3498db"]),
        link=dict(source=sources, target=targets, value=values, color=colors),
    ))
    fig.update_layout(
        title=f"{player_name} — Steal → Points Chain  (last {seq_stats['games_analyzed']} games)",
        height=380,
        **_layout(margin=dict(l=20, r=20, t=65, b=20)),
    )
    return fig


# ── 7. Multi-player steal-impact comparison ───────────────────────────────────

def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_sequence_comparison(stats_dict: dict[str, dict], colors: list[str] | None = None) -> go.Figure:
    """Side-by-side grouped bar chart comparing steal-chain stats across players."""
    if not stats_dict:
        return go.Figure(layout=go.Layout(title="No data", **_layout()))

    fallback = ["#EB6E1F", "#3498db", "#2ecc71", "#e74c3c"]
    colors   = (colors or fallback)[:4]
    names    = list(stats_dict.keys())

    conv_pcts   = [stats_dict[n].get("conversion_pct", 0) for n in names]
    fb_pcts     = [stats_dict[n].get("fast_break_pct", 0) for n in names]
    pts_per_stl = [stats_dict[n].get("pts_per_steal",  0) for n in names]

    solid = [_hex_rgba(colors[i % len(colors)], 0.85) for i in range(len(names))]
    faded = [_hex_rgba(colors[i % len(colors)], 0.45) for i in range(len(names))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Conversion % (steal → any score)", x=names, y=conv_pcts,
        marker_color=solid, text=[f"{v:.0f}%" for v in conv_pcts], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Fast Break % (score within 2 plays)", x=names, y=fb_pcts,
        marker_color=faded, marker_pattern_shape="/",
        text=[f"{v:.0f}%" for v in fb_pcts], textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        name="Pts per Steal", x=names, y=pts_per_stl,
        mode="markers+text",
        marker=dict(size=14, color="white", symbol="diamond",
                    line=dict(color="#EB6E1F", width=2)),
        text=[f"{v:.2f}" for v in pts_per_stl], textposition="top center",
        yaxis="y2",
    ))

    fig.update_layout(
        barmode="group",
        title="Steal Impact Comparison — Conversion, Fast Break Rate & Pts Generated",
        yaxis=dict(title="Rate (%)", range=[0, 105], showgrid=False),
        yaxis2=dict(title="Pts per Steal", overlaying="y", side="right",
                    range=[0, max(pts_per_stl or [3]) * 1.6],
                    showgrid=False, tickformat=".1f"),
        height=420,
        legend=dict(orientation="h", x=0, y=1.07, font=dict(size=9)),
        **_layout(margin=dict(l=60, r=70, t=80, b=60)),
    )
    return fig
