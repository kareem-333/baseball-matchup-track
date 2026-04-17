"""
MLB Scouting Module
All Statcast-based analytics + Plotly chart builders.
"""

import requests
import statsapi
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
from datetime import date
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

PITCH_NAMES = {
    "FF": "4-Seam FB", "SI": "Sinker",   "FC": "Cutter",
    "SL": "Slider",    "ST": "Sweeper",  "CH": "Changeup",
    "CU": "Curveball", "KC": "K-Curve",  "FS": "Splitter",
    "SV": "Slurve",    "KN": "Knuckler", "CS": "Slow Curve",
    "FO": "Forkball",  "EP": "Eephus",   "SC": "Screwball",
}

PITCH_COLORS = {
    "FF": "#e41a1c", "SI": "#ff7f00", "FC": "#984ea3",
    "SL": "#377eb8", "ST": "#4daf4a", "CH": "#f781bf",
    "CU": "#a65628", "KC": "#999999", "FS": "#ffff33",
    "SV": "#17becf", "KN": "#bcbd22", "CS": "#7f7f7f",
}

STAR_BATTERS = {
    "Shohei Ohtani":        660271,
    "Aaron Judge":          592450,
    "Bobby Witt Jr.":       677951,
    "Yordan Alvarez":       670541,
    "Juan Soto":            665742,
    "Fernando Tatis Jr.":   665487,
    "Mookie Betts":         605141,
    "Freddie Freeman":      518692,
    "Gunnar Henderson":     683002,
    "Ronald Acuña Jr.":     660670,
    "Pete Alonso":          624413,
    "Jose Ramirez":         608070,
    "Corey Seager":         608369,
    "Rafael Devers":        646240,
    "Matt Olson":           621566,
    "Kyle Tucker":          663656,
    "Vladimir Guerrero Jr.":665489,
    "Jose Altuve":          514888,
    "Carlos Correa":        621043,
    "Isaac Paredes":        670623,
    "Christian Walker":     572233,
}

# Strike zone boundaries (feet, catcher's view)
SZ_L, SZ_R = -0.831, 0.831
SZ_B, SZ_T = 1.50, 3.50
_cw = (SZ_R - SZ_L) / 3
_ch = (SZ_T - SZ_B) / 3

# Zone 1-9 → (x_left, x_right, z_bot, z_top)
ZONE_BOXES = {}
ZONE_CENTERS = {}
_ZONE_LAYOUT = {
    1:(0,2), 2:(0,1), 3:(0,0),   # top row  (left=inside for LHB)
    4:(1,2), 5:(1,1), 6:(1,0),
    7:(2,2), 8:(2,1), 9:(2,0),   # bot row
}
for _z, (_r, _c) in _ZONE_LAYOUT.items():
    _xl = SZ_L + _c * _cw
    _xr = SZ_L + (_c+1) * _cw
    _zb = SZ_T - (_r+1) * _ch
    _zt = SZ_T - _r * _ch
    ZONE_BOXES[_z]   = (_xl, _xr, _zb, _zt)
    ZONE_CENTERS[_z] = ((_xl+_xr)/2, (_zb+_zt)/2)

_SAVANT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
}


# ── Statcast data fetching ────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_statcast_csv(player_id: int, player_type: str, season: int) -> pd.DataFrame:
    """
    Download one season of Statcast pitch-level data for a player.
    player_type: 'batter' or 'pitcher'
    """
    lookup_param = (
        f"batters_lookup%5B%5D={player_id}" if player_type == "batter"
        else f"pitchers_lookup%5B%5D={player_id}"
    )
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv?"
        f"all=true&hfGT=R%7C&hfSea={season}%7C&player_type={player_type}"
        f"&{lookup_param}&min_pitches=0&min_results=0"
        "&group_by=name&sort_col=pitches&sort_order=desc&type=details"
    )
    try:
        r = requests.get(url, headers=_SAVANT_HEADERS, timeout=30)
        if r.status_code != 200 or len(r.content) < 200:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(r.text), low_memory=False)
        # Fix BOM and quoted column name on first column
        df.columns = (df.columns.str.lstrip("\ufeff")
                                 .str.strip('"')
                                 .str.strip("'"))
        # Normalize numeric columns
        for col in ["plate_x", "plate_z", "pfx_x", "pfx_z",
                    "release_speed", "release_spin_rate",
                    "estimated_ba_using_speedangle", "launch_speed",
                    "launch_angle", "api_break_x_arm",
                    "api_break_z_with_gravity", "delta_home_win_exp",
                    "home_win_exp", "bat_speed", "swing_length"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "zone" in df.columns:
            df["zone"] = pd.to_numeric(df["zone"], errors="coerce")
        if "launch_speed_angle" in df.columns:
            df["launch_speed_angle"] = pd.to_numeric(df["launch_speed_angle"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _best_season_df(player_id: int, player_type: str) -> pd.DataFrame:
    """Try current season first, fall back to previous if < 30 rows."""
    cur = date.today().year
    df = fetch_statcast_csv(player_id, player_type, cur)
    if len(df) >= 30:
        return df
    prev = fetch_statcast_csv(player_id, player_type, cur - 1)
    if len(prev) > len(df):
        return prev
    return df


# ── Pitcher arsenal ───────────────────────────────────────────────────────────

def get_pitcher_arsenal(pitcher_id: int) -> pd.DataFrame:
    """
    Returns pitch arsenal: pitch_type, pitch_name, usage_pct, avg_velo,
    avg_spin, avg_h_break, avg_v_break, avg_x, avg_z.
    """
    df = _best_season_df(pitcher_id, "pitcher")
    if df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()

    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]
    rows = []
    total = len(df)
    for pt, g in df.groupby("pitch_type"):
        rows.append({
            "pitch_type":  pt,
            "pitch_name":  PITCH_NAMES.get(pt, g["pitch_name"].iloc[0] if "pitch_name" in g.columns else pt),
            "count":       len(g),
            "usage_pct":   len(g) / total * 100,
            "avg_velo":    g["release_speed"].mean() if "release_speed" in g.columns else 0,
            "avg_spin":    g["release_spin_rate"].mean() if "release_spin_rate" in g.columns else 0,
            "avg_h_break": g["api_break_x_arm"].mean() if "api_break_x_arm" in g.columns else 0,
            "avg_v_break": g["api_break_z_with_gravity"].mean() if "api_break_z_with_gravity" in g.columns else 0,
            "avg_x":       g["plate_x"].mean() if "plate_x" in g.columns else 0,
            "avg_z":       g["plate_z"].mean() if "plate_z" in g.columns else 0,
        })
    return pd.DataFrame(rows).sort_values("usage_pct", ascending=False)


# ── Batter pitch splits ───────────────────────────────────────────────────────

def get_batter_pitch_splits(batter_id: int) -> pd.DataFrame:
    """
    Returns per-pitch-type split: pitch_type, pitch_name, pa, barrel_rate, xba.
    """
    df = _best_season_df(batter_id, "batter")
    if df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()

    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]
    rows = []
    for pt, g in df.groupby("pitch_type"):
        bip = g[g["type"] == "X"] if "type" in g.columns else g
        n_bip = len(bip)
        # Barrel: launch_speed_angle == 6
        barrels = (bip["launch_speed_angle"] == 6).sum() if "launch_speed_angle" in bip.columns else 0
        xba_vals = bip["estimated_ba_using_speedangle"].dropna() if "estimated_ba_using_speedangle" in bip.columns else pd.Series(dtype=float)
        rows.append({
            "pitch_type":  pt,
            "pitch_name":  PITCH_NAMES.get(pt, pt),
            "pitches":     len(g),
            "bip":         n_bip,
            "barrel_rate": barrels / n_bip * 100 if n_bip >= 5 else np.nan,
            "xba":         float(xba_vals.mean()) if len(xba_vals) >= 3 else np.nan,
        })
    df_out = pd.DataFrame(rows)
    return df_out[df_out["pitches"] >= 10].sort_values("pitches", ascending=False)


# ── Batter hot zones ──────────────────────────────────────────────────────────

def get_batter_hot_zones(batter_id: int) -> dict[int, float]:
    """
    Returns {zone_num: xba} for zones 1-9.
    """
    df = _best_season_df(batter_id, "batter")
    if df.empty or "zone" not in df.columns:
        return {}

    bip = df[(df["type"] == "X") & df["zone"].between(1, 9)] if "type" in df.columns else df[df["zone"].between(1, 9)]
    if bip.empty or "estimated_ba_using_speedangle" not in bip.columns:
        return {}

    result = {}
    for z in range(1, 10):
        z_bip = bip[bip["zone"] == z]["estimated_ba_using_speedangle"].dropna()
        result[z] = float(z_bip.mean()) if len(z_bip) >= 3 else 0.0
    return result


# ── Barrel rate by game (trend) ───────────────────────────────────────────────

def get_barrel_trend(batter_id: int, n_games: int = 15) -> pd.DataFrame:
    """Rolling barrel rate per game over last n games."""
    df = _best_season_df(batter_id, "batter")
    if df.empty or "game_date" not in df.columns or "launch_speed_angle" not in df.columns:
        return pd.DataFrame()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    bip = df[df["type"] == "X"].copy() if "type" in df.columns else df.copy()
    bip = bip[bip["game_date"].notna()]

    rows = []
    for gdate, g in bip.groupby("game_date"):
        n_bip = len(g)
        barrels = (g["launch_speed_angle"] == 6).sum()
        rows.append({
            "date":        gdate,
            "bip":         n_bip,
            "barrels":     int(barrels),
            "barrel_rate": barrels / n_bip * 100 if n_bip > 0 else 0,
        })

    result = pd.DataFrame(rows).sort_values("date")
    return result.tail(n_games)


# ══════════════════════════════════════════════════════════════════════════════
# Plotly Chart Builders
# ══════════════════════════════════════════════════════════════════════════════

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


# ── 3. Pitch movement scatter ─────────────────────────────────────────────────

def plot_pitch_movement(pitches: list[dict], pitcher_name: str = "") -> go.Figure:
    """
    Horizontal (pfx_x) vs vertical (pfx_z) movement, colored by pitch type.
    One dot per pitch. Uses live game feed data.
    """
    if not pitches:
        return go.Figure(layout=go.Layout(title="No pitch data yet", **_layout()))

    df = pd.DataFrame(pitches)
    df = df[df["pfx_x"].notna() & df["pfx_z"].notna()] if "pfx_x" in df.columns else df

    fig = go.Figure()
    for pt, g in df.groupby("pitch_type"):
        color = PITCH_COLORS.get(pt, "#cccccc")
        name  = PITCH_NAMES.get(pt, pt)
        # pfx_x / pfx_z come from live API in feet * 12 (inches in some cases)
        # Normalize: if max abs > 5, they're in inches → divide by 12
        px_vals = g["pfx_x"] * (1 if g["pfx_x"].abs().max() <= 5 else 1/12)
        pz_vals = g["pfx_z"] * (1 if g["pfx_z"].abs().max() <= 5 else 1/12)
        speeds  = g["speed"].fillna(0) if "speed" in g.columns else pd.Series([0]*len(g))
        fig.add_trace(go.Scatter(
            x=px_vals, y=pz_vals,
            mode="markers",
            marker=dict(size=7, color=color,
                        line=dict(color="white", width=0.4), opacity=0.85),
            name=f"{name} (n={len(g)})",
            hovertemplate=f"<b>{name}</b><br>HB: %{{x:.1f}} in<br>VB: %{{y:.1f}} in<br>Velo: %{{customdata:.1f}}<extra></extra>",
            customdata=speeds,
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        title=f"Pitch Movement — {pitcher_name}",
        xaxis=dict(title="Horizontal Break (in, catcher's view)", showgrid=False),
        yaxis=dict(title="Vertical Break (in)", showgrid=False, scaleanchor="x", scaleratio=1),
        height=430,
        **_layout(margin=dict(l=60, r=160, t=55, b=50)),
    )
    return fig


# ── 4. Velocity fade chart ────────────────────────────────────────────────────

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

    # ── Sort & assign cumulative pitch numbers ────────────────────────────────
    has_game_info = "game_pk" in df.columns and df["game_pk"].notna().any()
    if has_game_info:
        # Sort: oldest game first, then by at_bat_number, then pitch_number
        for col in ("at_bat_number", "pitch_number"):
            if col not in df.columns:
                df[col] = 0
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.sort_values(["game_date", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    df["pitch_num"] = range(1, len(df) + 1)

    # ── Compute game boundary positions ──────────────────────────────────────
    game_breaks: list[dict] = []   # {x: last pitch # of prev game, label: date string}
    if has_game_info:
        prev_pk = None
        for _, row in df.iterrows():
            pk = row.get("game_pk")
            if prev_pk is not None and pk != prev_pk:
                # First pitch of new game → boundary at pitch_num - 0.5
                game_breaks.append({
                    "x":     row["pitch_num"] - 0.5,
                    "label": str(row.get("game_date", ""))[:10],
                })
            prev_pk = pk

    fig = go.Figure()

    # ── One trace per pitch type ──────────────────────────────────────────────
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

    # ── 5-pitch rolling average ───────────────────────────────────────────────
    if len(df) >= 5:
        roll_avg = df.sort_values("pitch_num")["speed"].rolling(5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df.sort_values("pitch_num")["pitch_num"], y=roll_avg,
            mode="lines",
            line=dict(color="white", width=1.5, dash="dash"),
            name="5-pitch avg", opacity=0.6,
        ))

    # ── Game-break vertical lines + date labels ───────────────────────────────
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


# ── 5. Pitcher fatigue gauge ──────────────────────────────────────────────────

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


# ── 6. Win probability chart ──────────────────────────────────────────────────

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


# ── 7. Rolling OPS line chart ─────────────────────────────────────────────────

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


# ── 8. K-rate dual line chart ─────────────────────────────────────────────────

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


# ── 9. Barrel rate trend ──────────────────────────────────────────────────────

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


# ── 10. Pitch mix simulator (Streamlit widget) ────────────────────────────────

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


# ── Headshot URL helper ───────────────────────────────────────────────────────

_mlb_hs_cache: dict[int, str | None] = {}   # player_id → base64 data-URI or None


def headshot_url(player_id: int) -> str:
    """Working MLB CDN URL for player headshots (securea.mlb.com)."""
    return f"https://securea.mlb.com/mlb/images/players/head_shot/{player_id}.jpg"


def headshot_b64(player_id: int, size: int = 80) -> str | None:
    """
    Fetch an MLB headshot and return a base64 data-URI so Streamlit can
    display it without relying on CDN hot-link permissions.
    Returns None if the image cannot be fetched.
    """
    import requests, base64, io
    from PIL import Image

    if player_id in _mlb_hs_cache:
        return _mlb_hs_cache[player_id]

    try:
        url = headshot_url(player_id)
        r   = requests.get(
            url, timeout=6,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.mlb.com/"},
        )
        if r.status_code != 200:
            _mlb_hs_cache[player_id] = None
            return None

        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.thumbnail((size * 3, size * 3), Image.LANCZOS)   # keep aspect, limit size

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        _mlb_hs_cache[player_id] = b64
        return b64
    except Exception:
        _mlb_hs_cache[player_id] = None
        return None
