from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import statsapi
import streamlit as st
from pybaseball import statcast_batter, statcast_pitcher, statcast

# ── constants ────────────────────────────────────────────────────────────────

SEASON_START = {
    2024: date(2024, 3, 20),
    2025: date(2025, 3, 27),
    2026: date(2026, 4, 3),
}

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "hit_into_play",
    "foul_bunt", "missed_bunt",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked"}

PITCH_LABELS = {
    "FF": "4-Seam FB", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "CU": "Curveball",
    "KC": "Knuckle Curve", "CH": "Changeup", "FS": "Splitter",
    "SV": "Slurve", "FO": "Forkball", "KN": "Knuckleball",
    "EP": "Eephus", "SC": "Screwball",
}


def _season_start(season: int) -> date:
    return SEASON_START.get(season, date(season, 4, 1))


def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def _season_start_str(season: int) -> str:
    return _season_start(season).strftime("%Y-%m-%d")


# ── live / schedule ──────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def get_todays_games() -> list[dict]:
    """All games scheduled today from the MLB StatsAPI."""
    raw = statsapi.schedule(date=_today_str())
    games = []
    for g in raw:
        games.append({
            "game_pk": g["game_id"],
            "status": g["status"],
            "away_team": g["away_name"],
            "home_team": g["home_name"],
            "away_id": g["away_id"],
            "home_id": g["home_id"],
            "away_score": g.get("away_score", 0),
            "home_score": g.get("home_score", 0),
            "game_datetime": g.get("game_datetime", ""),
            "venue": g.get("venue_name", ""),
            "inning": g.get("current_inning", 0),
            "inning_state": g.get("inning_state", ""),
            "away_probable_pitcher_id": g.get("away_probable_pitcher_id"),
            "home_probable_pitcher_id": g.get("home_probable_pitcher_id"),
            "away_probable_pitcher": g.get("away_probable_pitcher", "TBD"),
            "home_probable_pitcher": g.get("home_probable_pitcher", "TBD"),
        })
    return games


@st.cache_data(ttl=30)
def get_live_game_data(game_pk: int) -> dict:
    """Full live game feed for a given game_pk."""
    try:
        feed = statsapi.get("game", {"gamePk": game_pk})
        gd = feed.get("gameData", {})
        ld = feed.get("liveData", {})
        linescore = ld.get("linescore", {})
        boxscore = ld.get("boxscore", {})
        plays = ld.get("plays", {})

        current_play = plays.get("currentPlay", {})
        matchup = current_play.get("matchup", {})
        batter = matchup.get("batter", {})
        pitcher = matchup.get("pitcher", {})

        return {
            "game_pk": game_pk,
            "status": gd.get("status", {}).get("detailedState", "Unknown"),
            "inning": linescore.get("currentInning", 0),
            "inning_state": linescore.get("inningState", ""),
            "half": linescore.get("inningHalf", ""),
            "outs": linescore.get("outs", 0),
            "balls": linescore.get("balls", 0),
            "strikes": linescore.get("strikes", 0),
            "home_team": gd.get("teams", {}).get("home", {}).get("name", ""),
            "away_team": gd.get("teams", {}).get("away", {}).get("name", ""),
            "home_id": gd.get("teams", {}).get("home", {}).get("id"),
            "away_id": gd.get("teams", {}).get("away", {}).get("id"),
            "home_score": linescore.get("teams", {}).get("home", {}).get("runs", 0),
            "away_score": linescore.get("teams", {}).get("away", {}).get("runs", 0),
            "current_batter_id": batter.get("id"),
            "current_batter_name": batter.get("fullName", ""),
            "current_pitcher_id": pitcher.get("id"),
            "current_pitcher_name": pitcher.get("fullName", ""),
            "linescore": linescore,
            "boxscore": boxscore,
        }
    except Exception:
        return {}


@st.cache_data(ttl=60)
def get_game_lineup(game_pk: int, team_id: int) -> list[dict]:
    """Batting order for a team in a given game."""
    try:
        feed = statsapi.get("game", {"gamePk": game_pk})
        ld = feed.get("liveData", {})
        boxscore = ld.get("boxscore", {})

        side = None
        gd = feed.get("gameData", {})
        if gd.get("teams", {}).get("home", {}).get("id") == team_id:
            side = "home"
        elif gd.get("teams", {}).get("away", {}).get("id") == team_id:
            side = "away"
        if side is None:
            return []

        players_dict = boxscore.get("teams", {}).get(side, {}).get("players", {})
        batting_order = boxscore.get("teams", {}).get(side, {}).get("battingOrder", [])

        lineup = []
        for pid in batting_order:
            key = f"ID{pid}"
            p = players_dict.get(key, {})
            info = p.get("person", {})
            pos = p.get("position", {}).get("abbreviation", "")
            lineup.append({
                "player_id": pid,
                "name": info.get("fullName", str(pid)),
                "position": pos,
            })
        return lineup
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_probable_lineups(game_pk: int) -> dict:
    """Try to fetch probable lineups for a pre-game matchup."""
    try:
        feed = statsapi.get("game", {"gamePk": game_pk})
        gd = feed.get("gameData", {})
        pitchers = {
            "home": gd.get("probablePitchers", {}).get("home", {}),
            "away": gd.get("probablePitchers", {}).get("away", {}),
        }
        return pitchers
    except Exception:
        return {}


# ── pitcher arsenal ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_pitcher_arsenal(pitcher_id: int, season: int | None = None) -> pd.DataFrame:
    """
    Pitcher's pitch mix for the given season from Statcast.

    Returns columns: pitch_type, pitch_label, usage_pct, count,
                     avg_velocity, last_appearance_date, appearances_used
    """
    season = season or date.today().year
    start = _season_start_str(season)
    end = _today_str()

    try:
        df = statcast_pitcher(start, end, pitcher_id)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["pitch_type"])
    df = df[df["pitch_type"] != ""]

    total = len(df)
    if total == 0:
        return pd.DataFrame()

    # Per-appearance dates for decay support
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        appearances = sorted(df["game_date"].unique(), reverse=True)
        last_date = appearances[0] if appearances else None
        n_appearances = len(appearances)
    else:
        last_date = None
        n_appearances = 0

    pitch_groups = df.groupby("pitch_type")
    rows = []
    for pt, grp in pitch_groups:
        velocity = (
            grp["release_speed"].dropna().mean()
            if "release_speed" in grp.columns
            else None
        )
        rows.append({
            "pitch_type": pt,
            "pitch_label": PITCH_LABELS.get(pt, pt),
            "count": len(grp),
            "usage_pct": round(len(grp) / total, 4),
            "avg_velocity": round(velocity, 1) if velocity is not None else None,
            "last_appearance_date": last_date,
            "appearances_used": n_appearances,
        })

    result = pd.DataFrame(rows).sort_values("usage_pct", ascending=False).reset_index(drop=True)
    return result


# ── batter pitch splits ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_batter_pitch_splits(batter_id: int, season: int | None = None) -> pd.DataFrame:
    """
    Batter's performance vs each pitch type, split by pitcher handedness.

    Returns columns: pitch_type, pitch_label, vs_hand, barrel_rate,
                     whiff_rate, sample_size, last_event_date
    """
    season = season or date.today().year
    start = _season_start_str(season)
    end = _today_str()

    try:
        df = statcast_batter(start, end, batter_id)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["pitch_type", "p_throws"])
    df = df[df["pitch_type"] != ""]

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    rows = []
    for (pt, hand), grp in df.groupby(["pitch_type", "p_throws"]):
        n = len(grp)

        # barrel rate: barrels per pitch seen
        if "launch_speed_angle" in grp.columns:
            barrels = (grp["launch_speed_angle"] == 6).sum()
        else:
            # manual barrel: EV >= 98 and 26 <= LA <= 30
            has_ev = grp["launch_speed"].notna() & grp["launch_angle"].notna()
            barrels = (
                has_ev
                & (grp["launch_speed"] >= 98)
                & (grp["launch_angle"].between(26, 30))
            ).sum()
        barrel_rate = barrels / n if n > 0 else 0.0

        # whiff rate: swinging strikes per swing
        if "description" in grp.columns:
            swings = grp["description"].isin(SWING_DESCS).sum()
            whiffs = grp["description"].isin(WHIFF_DESCS).sum()
            whiff_rate = whiffs / swings if swings > 0 else 0.0
        else:
            whiff_rate = 0.0

        last_date = grp["game_date"].max() if "game_date" in grp.columns else None

        rows.append({
            "pitch_type": pt,
            "pitch_label": PITCH_LABELS.get(pt, pt),
            "vs_hand": hand,
            "barrel_rate": round(barrel_rate, 4),
            "whiff_rate": round(whiff_rate, 4),
            "sample_size": n,
            "last_event_date": last_date,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["pitch_type", "vs_hand"]
    ).reset_index(drop=True)


# ── league baselines ─────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_league_pitch_baselines(season: int) -> dict:
    """
    League-wide mean and std for barrel_rate, whiff_rate, usage_pct per pitch type.
    Pulls last 30 days of Statcast to keep the call manageable.
    Returns: {pitch_type: {barrel_rate_mean, barrel_rate_std,
                            whiff_rate_mean, whiff_rate_std,
                            usage_pct_mean, usage_pct_std}}
    """
    end = date.today()
    start = end - timedelta(days=30)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    try:
        df = statcast(start_str, end_str)
    except Exception:
        return _fallback_baselines()

    if df is None or df.empty:
        return _fallback_baselines()

    df = df.dropna(subset=["pitch_type"])
    df = df[df["pitch_type"] != ""]

    # barrel flag
    if "launch_speed_angle" in df.columns:
        df["is_barrel"] = (df["launch_speed_angle"] == 6).astype(int)
    else:
        has_ev = df["launch_speed"].notna() & df["launch_angle"].notna()
        df["is_barrel"] = (
            has_ev
            & (df["launch_speed"] >= 98)
            & (df["launch_angle"].between(26, 30))
        ).astype(int)

    # swing / whiff flags
    df["is_swing"] = df["description"].isin(SWING_DESCS).astype(int)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCS).astype(int)

    # per-pitcher usage: group by (pitcher, game_date, pitch_type) to compute usage_pct
    total_per_game = (
        df.groupby(["pitcher", "game_date"])
        .size()
        .reset_index(name="total_pitches")
    )
    pitch_per_game = (
        df.groupby(["pitcher", "game_date", "pitch_type"])
        .size()
        .reset_index(name="pitch_count")
    )
    usage_df = pitch_per_game.merge(total_per_game, on=["pitcher", "game_date"])
    usage_df["usage_pct"] = usage_df["pitch_count"] / usage_df["total_pitches"]

    # per-batter barrel/whiff: group by (batter, pitch_type)
    batter_pt = df.groupby(["batter", "pitch_type"]).agg(
        n=("pitch_type", "count"),
        barrels=("is_barrel", "sum"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
    ).reset_index()
    batter_pt = batter_pt[batter_pt["n"] >= 10]
    batter_pt["barrel_rate"] = batter_pt["barrels"] / batter_pt["n"]
    batter_pt["whiff_rate"] = np.where(
        batter_pt["swings"] > 0,
        batter_pt["whiffs"] / batter_pt["swings"],
        0.0,
    )

    baselines = {}
    for pt in df["pitch_type"].unique():
        br = batter_pt[batter_pt["pitch_type"] == pt]["barrel_rate"]
        wr = batter_pt[batter_pt["pitch_type"] == pt]["whiff_rate"]
        up = usage_df[usage_df["pitch_type"] == pt]["usage_pct"]

        baselines[pt] = {
            "barrel_rate_mean": float(br.mean()) if len(br) else 0.05,
            "barrel_rate_std": float(br.std()) if len(br) > 1 else 0.04,
            "whiff_rate_mean": float(wr.mean()) if len(wr) else 0.25,
            "whiff_rate_std": float(wr.std()) if len(wr) > 1 else 0.15,
            "usage_pct_mean": float(up.mean()) if len(up) else 0.20,
            "usage_pct_std": float(up.std()) if len(up) > 1 else 0.10,
        }

    return baselines


def _fallback_baselines() -> dict:
    """Hardcoded league averages when Statcast is unavailable."""
    defaults = {
        "FF": (0.055, 0.040, 0.220, 0.130, 0.340, 0.120),
        "SI": (0.045, 0.038, 0.180, 0.115, 0.220, 0.100),
        "FC": (0.040, 0.035, 0.230, 0.135, 0.130, 0.080),
        "SL": (0.035, 0.030, 0.330, 0.150, 0.190, 0.100),
        "ST": (0.030, 0.028, 0.360, 0.155, 0.110, 0.075),
        "CU": (0.030, 0.028, 0.310, 0.145, 0.130, 0.090),
        "KC": (0.028, 0.026, 0.290, 0.140, 0.080, 0.060),
        "CH": (0.038, 0.032, 0.330, 0.155, 0.140, 0.090),
        "FS": (0.032, 0.030, 0.350, 0.160, 0.060, 0.050),
    }
    out = {}
    for pt, vals in defaults.items():
        out[pt] = {
            "barrel_rate_mean": vals[0],
            "barrel_rate_std": vals[1],
            "whiff_rate_mean": vals[2],
            "whiff_rate_std": vals[3],
            "usage_pct_mean": vals[4],
            "usage_pct_std": vals[5],
        }
    return out


# ── player headshot ──────────────────────────────────────────────────────────

def get_player_headshot_url(player_id: int) -> str:
    return (
        f"https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"d_people:generic:headshot:67:current.png/w_213,q_auto:best/"
        f"v1/people/{player_id}/headshot/67/current"
    )
