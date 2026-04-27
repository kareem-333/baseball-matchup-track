"""
dashboard/components/live_charts.py — Real-time Plotly chart builders for live game data.

Pitch movement scatter, velocity fade with game-break lines,
fatigue gauge, and win-probability area chart.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.config import PITCH_NAMES, PITCH_COLORS

_DARK = dict(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"))


def _layout(**kwargs):
    base = dict(**_DARK, margin=dict(l=10, r=10, t=50, b=10))
    base.update(kwargs)
    return base


def plot_pitch_movement(pitches: list[dict], pitcher_name: str = "") -> go.Figure:
    """
    Horizontal (pfx_x) vs vertical induced break (pfx_z), colored by pitch type.
    One dot per pitch. Uses live game feed data.

    Strike zone reference indicators:
    • Vertical dashed lines at ±8.5" — half of the 17" wide plate.
      Pitches that move more than 8.5" horizontally have shifted by a full
      half-plate from their release point.
    • Horizontal shaded band at +3" to +15" — typical range where fastball
      rise keeps the ball in the top of the strike zone.
    • Horizontal shaded band at -5" to -18" — typical range for breaking
      balls to land at or below the bottom of the zone.
    • Height reference note: exact batter-height-dependent boundaries are not
      shown because vertical break is measured relative to spin-free trajectory,
      not absolute pitch location.
    """
    if not pitches:
        return go.Figure(layout=go.Layout(title="No pitch data yet", **_layout()))

    df = pd.DataFrame(pitches)
    df = df[df["pfx_x"].notna() & df["pfx_z"].notna()] if "pfx_x" in df.columns else df

    fig = go.Figure()

    # ── Strike zone reference overlays ───────────────────────────────────────
    HALF_PLATE = 8.5   # inches

    fig.add_shape(
        type="rect",
        x0=-25, x1=25, y0=3, y1=15,
        fillcolor="rgba(46,204,113,0.06)",
        line=dict(width=0),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=-25, x1=25, y0=-18, y1=-5,
        fillcolor="rgba(52,152,219,0.06)",
        line=dict(width=0),
        layer="below",
    )

    for sign, label in [(-HALF_PLATE, "← Plate Edge"), (HALF_PLATE, "Plate Edge →")]:
        fig.add_shape(
            type="line",
            xref="x", yref="paper",
            x0=sign, x1=sign, y0=0, y1=1,
            line=dict(color="rgba(255,255,255,0.22)", width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=sign, y=1.01,
            xref="x", yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=8, color="rgba(180,180,180,0.7)"),
            xanchor="center",
        )

    fig.add_annotation(
        x=-24, y=9,
        text="FB rise zone",
        showarrow=False,
        font=dict(size=8, color="rgba(46,204,113,0.55)"),
        xanchor="left",
    )
    fig.add_annotation(
        x=-24, y=-11.5,
        text="Breaking zone",
        showarrow=False,
        font=dict(size=8, color="rgba(52,152,219,0.55)"),
        xanchor="left",
    )

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")

    # ── One trace per pitch type ──────────────────────────────────────────────
    for pt, g in df.groupby("pitch_type"):
        color = PITCH_COLORS.get(pt, "#cccccc")
        name  = PITCH_NAMES.get(pt, pt)
        px_vals = g["pfx_x"] * (1 if g["pfx_x"].abs().max() > 5 else 12)
        pz_vals = g["pfx_z"] * (1 if g["pfx_z"].abs().max() > 5 else 12)
        speeds  = g["speed"].fillna(0) if "speed" in g.columns else pd.Series([0]*len(g))
        fig.add_trace(go.Scatter(
            x=px_vals, y=pz_vals,
            mode="markers",
            marker=dict(size=7, color=color,
                        line=dict(color="white", width=0.4), opacity=0.85),
            name=f"{name} (n={len(g)})",
            hovertemplate=(
                f"<b>{name}</b><br>"
                "HBreak: %{x:.1f}″<br>VBreak: %{y:.1f}″<br>"
                "Velo: %{customdata:.1f} mph<extra></extra>"
            ),
            customdata=speeds,
        ))

    fig.update_layout(
        title=f"Pitch Movement — {pitcher_name}  "
              f"<span style='font-size:11px;color:#888'>"
              f"Break vs spin-free trajectory (dashed = ½ plate = 8.5″)</span>",
        xaxis=dict(
            title="Horizontal Break (in, catcher's view +right)",
            showgrid=False, range=[-25, 25],
        ),
        yaxis=dict(
            title="Vertical Induced Break (in, +up)",
            showgrid=False, range=[-22, 22],
            scaleanchor="x", scaleratio=1,
        ),
        height=480,
        **_layout(margin=dict(l=60, r=160, t=70, b=50)),
    )
    return fig


def plot_velocity_fade(pitches: list[dict], pitcher_name: str = "") -> go.Figure:
    """
    Velocity by pitch number, one line per pitch type. Shows fade across game(s).

    If pitches include 'game_pk' / 'game_date' fields (Statcast season data),
    vertical dashed lines are drawn at every game boundary with date labels.
    """
    if not pitches:
        return go.Figure(layout=go.Layout(title="No pitch data yet", **_layout()))

    df = pd.DataFrame(pitches)
    df = df[df["speed"].notna()] if "speed" in df.columns else df
    if df.empty:
        return go.Figure(layout=go.Layout(title="No velocity data yet", **_layout()))

    has_game_info = "game_pk" in df.columns and df["game_pk"].notna().any()
    if has_game_info:
        for col in ("at_bat_number", "pitch_number"):
            if col not in df.columns:
                df[col] = 0
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.sort_values(["game_date", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    df["pitch_num"] = range(1, len(df) + 1)

    game_breaks: list[dict] = []
    if has_game_info:
        prev_pk = None
        for _, row in df.iterrows():
            pk = row.get("game_pk")
            if prev_pk is not None and pk != prev_pk:
                game_breaks.append({
                    "x":     row["pitch_num"] - 0.5,
                    "label": str(row.get("game_date", ""))[:10],
                })
            prev_pk = pk

    fig = go.Figure()

    for pt, g in df.groupby("pitch_type"):
        g     = g.sort_values("pitch_num")
        color = PITCH_COLORS.get(pt, "#cccccc")
        name  = PITCH_NAMES.get(pt, pt)
        hover_text = []
        for _, row in g.iterrows():
            gdate = str(row.get("game_date", ""))[:10] if has_game_info else ""
            extra = f"<br>{gdate}" if gdate else ""
            hover_text.append(f"<b>{name}</b><br>Pitch #{int(row['pitch_num'])}{extra}<br>{row['speed']:.1f} mph")
        fig.add_trace(go.Scatter(
            x=g["pitch_num"], y=g["speed"],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            name=name,
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
        ))

    if len(df) >= 5:
        roll_avg = df.sort_values("pitch_num")["speed"].rolling(5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df.sort_values("pitch_num")["pitch_num"], y=roll_avg,
            mode="lines",
            line=dict(color="white", width=1.5, dash="dash"),
            name="5-pitch avg", opacity=0.6,
        ))

    shapes = []
    annotations = []
    for brk in game_breaks:
        shapes.append(dict(
            type="line",
            xref="x", yref="paper",
            x0=brk["x"], x1=brk["x"],
            y0=0, y1=1,
            line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dot"),
        ))
        if brk["label"]:
            annotations.append(dict(
                x=brk["x"], y=1.02,
                xref="x", yref="paper",
                text=brk["label"],
                showarrow=False,
                font=dict(size=9, color="#aaa"),
                xanchor="center",
            ))

    title_suffix = f" ({len(game_breaks) + 1} games)" if game_breaks else ""
    fig.update_layout(
        title=f"Velocity Fade — {pitcher_name}{title_suffix}",
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(title="Cumulative Pitch #", showgrid=False),
        yaxis=dict(title="Velocity (mph)"),
        height=350,
        **_layout(margin=dict(l=60, r=160, t=55, b=50)),
    )
    return fig


def plot_fatigue_gauge(
    pitch_count: int,
    vel_drop_pct: float,
    spin_drop_pct: float,
    pitcher_name: str = "",
) -> go.Figure:
    """
    Composite fatigue score gauge (0 = fresh, 100 = exhausted).
    vel_drop_pct / spin_drop_pct: drop relative to game-start baseline (%).
    """
    pc_score    = min(100, pitch_count / 110 * 100)
    vel_score   = max(0, min(100, vel_drop_pct * 15))
    spin_score  = max(0, min(100, spin_drop_pct * 8))
    fatigue     = round(0.55 * pc_score + 0.28 * vel_score + 0.17 * spin_score, 1)

    bar_color = (
        "#2ecc71" if fatigue < 35
        else "#f39c12" if fatigue < 65
        else "#e74c3c"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fatigue,
        title=dict(text=f"Fatigue — {pitcher_name}", font=dict(size=14, color="#e0e0e0")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#e0e0e0", tickfont=dict(color="#e0e0e0")),
            bar=dict(color=bar_color),
            steps=[
                dict(range=[0,  35], color="#1a2a1a"),
                dict(range=[35, 65], color="#2a2a1a"),
                dict(range=[65,100], color="#2a1a1a"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=fatigue),
        ),
        delta=dict(reference=0, valueformat=".1f"),
        number=dict(suffix=" / 100", font=dict(color=bar_color, size=26)),
    ))
    fig.update_layout(
        height=260,
        **_layout(margin=dict(l=30, r=30, t=10, b=10)),
    )
    return fig


def plot_win_probability(wp_data: list[dict], home_team: str = "Home") -> go.Figure:
    """
    wp_data: list of {inning, half, atbat, home_win_exp, away_score, home_score}
    """
    if not wp_data:
        return go.Figure(layout=go.Layout(title="Win probability not yet available", **_layout()))

    df = pd.DataFrame(wp_data)
    df["x"] = range(len(df))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["x"], y=df["home_win_exp"] * 100,
        mode="lines",
        line=dict(color="#EB6E1F", width=2),
        fill="tozeroy",
        fillcolor="rgba(235,110,31,0.18)",
        name=f"{home_team} Win%",
        hovertemplate="<b>%{text}</b><br>Win%: %{y:.1f}%<extra></extra>",
        text=[f"{r.get('inning_label','')} — {r.get('away_score',0)}-{r.get('home_score',0)}"
              for _, r in df.iterrows()],
    ))
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        title=f"Win Probability — {home_team}",
        xaxis=dict(title="At-Bat #", showgrid=False, showticklabels=False),
        yaxis=dict(title="Win Probability (%)", range=[0, 100]),
        height=300,
        **_layout(margin=dict(l=60, r=20, t=55, b=40)),
    )
    return fig
