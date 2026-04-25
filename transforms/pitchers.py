# Layer 3: Pitcher transforms.
# Pure functions: DataFrame in, DataFrame out. No I/O, no API calls, no Streamlit.
# Input is the raw Statcast CSV DataFrame returned by MLBStatsSource.get_pitcher_stats().

from __future__ import annotations

from datetime import date

import pandas as pd

from core.config import PITCH_NAMES

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "hit_into_play", "foul_bunt", "missed_bunt",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked"}

PITCH_USAGE_FLOOR = 0.05  # 5% — pitches below this are unqualified


def parse_pitcher_arsenal(df: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """
    Compute pitch arsenal from raw Statcast pitch-level data.

    Input:  raw Statcast DataFrame for a pitcher (one row per pitch)
    Output: one row per pitch type with:
        pitch_type, pitch_label, pitch_name, count, usage_pct (0-1),
        avg_velocity, avg_velo, avg_spin, avg_h_break, avg_v_break,
        avg_x, avg_z, last_appearance_date, appearances_used

    Expects columns: pitch_type, release_speed, release_spin_rate,
                     api_break_x_arm, api_break_z_with_gravity,
                     plate_x, plate_z, game_date
    """
    if df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()

    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")].copy()
    total = len(df)
    if total == 0:
        return pd.DataFrame()

    appearances = (
        sorted(df["game_date"].dropna().unique(), reverse=True)
        if "game_date" in df.columns else []
    )

    rows = []
    for pt, g in df.groupby("pitch_type"):
        rows.append({
            "pitch_type":           pt,
            "pitch_label":          PITCH_NAMES.get(pt, pt),
            "pitch_name":           PITCH_NAMES.get(pt, pt),
            "count":                len(g),
            "usage_pct":            len(g) / total,
            "avg_velocity":         round(g["release_speed"].mean(), 1) if "release_speed" in g.columns else None,
            "avg_velo":             g["release_speed"].mean() if "release_speed" in g.columns else 0,
            "avg_spin":             g["release_spin_rate"].mean() if "release_spin_rate" in g.columns else 0,
            "avg_h_break":          g["api_break_x_arm"].mean() if "api_break_x_arm" in g.columns else 0,
            "avg_v_break":          g["api_break_z_with_gravity"].mean() if "api_break_z_with_gravity" in g.columns else 0,
            "avg_x":                g["plate_x"].mean() if "plate_x" in g.columns else 0,
            "avg_z":                g["plate_z"].mean() if "plate_z" in g.columns else 0,
            "last_appearance_date": appearances[0] if appearances else None,
            "appearances_used":     len(appearances),
        })

    return pd.DataFrame(rows).sort_values("usage_pct", ascending=False).reset_index(drop=True)


def parse_qualified_arsenal(arsenal: pd.DataFrame) -> pd.DataFrame:
    """Filter arsenal to pitches meeting the 5% usage floor."""
    if arsenal.empty:
        return pd.DataFrame()
    return arsenal[arsenal["usage_pct"] >= PITCH_USAGE_FLOOR].reset_index(drop=True)


def parse_pitcher_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-game log from raw Statcast pitcher data.

    Input:  raw Statcast DataFrame for a pitcher
    Output: one row per game_date with:
        game_date, pitches, strikeouts, whiffs, whiff_rate

    Expects columns: game_date, pitch_type, events, description
    """
    if df.empty or "game_date" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    if "events" in df.columns:
        df["is_k"] = df["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    else:
        df["is_k"] = 0

    if "description" in df.columns:
        df["is_whiff"] = df["description"].isin(WHIFF_DESCS).astype(int)
    else:
        df["is_whiff"] = 0

    by_game = (
        df.groupby("game_date")
        .agg(pitches=("pitch_type", "count"), strikeouts=("is_k", "sum"), whiffs=("is_whiff", "sum"))
        .reset_index()
        .sort_values("game_date", ascending=False)
    )
    by_game["whiff_rate"] = (by_game["whiffs"] / by_game["pitches"].clip(lower=1)).round(3)
    return by_game


def parse_pitcher_sample_flag(df: pd.DataFrame) -> dict:
    """
    Return sample size diagnostic for a pitcher.

    Input:  raw Statcast DataFrame for a pitcher
    Output: {"total_pitches": int, "is_low_sample": bool, "reason": str}

    Threshold: under 200 total pitches = low sample.
    """
    total = len(df) if not df.empty else 0
    is_low = total < 200
    if is_low:
        reason = "No pitches thrown this season yet" if total == 0 else f"Only {total} pitches thrown — results may be unstable"
    else:
        reason = ""
    return {"total_pitches": total, "is_low_sample": is_low, "reason": reason}
