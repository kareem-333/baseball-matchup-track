# Layer 3: Batter transforms.
# Pure functions: DataFrame in, DataFrame out. No I/O, no API calls, no Streamlit.
# Input is the raw Statcast CSV DataFrame returned by MLBStatsSource.get_batter_stats().

from __future__ import annotations

from datetime import date

import pandas as pd

from core.config import PITCH_NAMES

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "hit_into_play", "foul_bunt", "missed_bunt",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked"}

_HIT_EVENTS = {"single", "double", "triple", "home_run"}
_AB_EVENTS = _HIT_EVENTS | {
    "strikeout", "field_out", "grounded_into_double_play",
    "force_out", "double_play", "fielders_choice",
    "fielders_choice_out", "strikeout_double_play",
}


def parse_batter_pitch_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-pitch-type performance split by pitcher handedness.

    Input:  raw Statcast DataFrame for a batter (one row per pitch)
    Output: one row per (pitch_type, vs_hand) with:
        pitch_type, pitch_label, pitch_name, vs_hand, barrel_rate (0-1),
        whiff_rate (0-1), sample_size, last_event_date, bip, xba, pitches

    Expects columns: pitch_type, p_throws, type, launch_speed_angle,
                     launch_speed, launch_angle, description, game_date,
                     estimated_ba_using_speedangle
    """
    if df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()

    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")].copy()
    if "p_throws" not in df.columns:
        return pd.DataFrame()

    rows = []
    for (pt, hand), grp in df.groupby(["pitch_type", "p_throws"]):
        n = len(grp)
        bip = grp[grp["type"] == "X"] if "type" in grp.columns else grp
        n_bip = len(bip)

        if "launch_speed_angle" in bip.columns:
            barrels = (bip["launch_speed_angle"] == 6).sum()
        else:
            has_ev = bip["launch_speed"].notna() & bip["launch_angle"].notna()
            barrels = (
                has_ev
                & (bip["launch_speed"] >= 98)
                & (bip["launch_angle"].between(26, 30))
            ).sum()

        swings = grp["description"].isin(SWING_DESCS).sum() if "description" in grp.columns else 0
        whiffs = grp["description"].isin(WHIFF_DESCS).sum() if "description" in grp.columns else 0
        xba_vals = (
            bip["estimated_ba_using_speedangle"].dropna()
            if "estimated_ba_using_speedangle" in bip.columns
            else pd.Series(dtype=float)
        )
        last_date = grp["game_date"].max() if "game_date" in grp.columns else None

        rows.append({
            "pitch_type":      pt,
            "pitch_label":     PITCH_NAMES.get(pt, pt),
            "pitch_name":      PITCH_NAMES.get(pt, pt),
            "vs_hand":         hand,
            "barrel_rate":     round(float(barrels / n_bip) if n_bip > 0 else 0.0, 4),
            "whiff_rate":      round(float(whiffs / swings) if swings > 0 else 0.0, 4),
            "sample_size":     n,
            "last_event_date": last_date,
            "bip":             n_bip,
            "xba":             float(xba_vals.mean()) if len(xba_vals) >= 3 else float("nan"),
            "pitches":         n,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["pitch_type", "vs_hand"]).reset_index(drop=True)


def parse_batter_career_splits(frames: list[tuple[int, pd.DataFrame]]) -> pd.DataFrame:
    """
    Compute multi-season pitch-type breakdown with vs_hand split.

    Input:  list of (season, raw_df) tuples
    Output: one row per (season, pitch_type, vs_hand) with:
        season, pitch_type, pitch_label, vs_hand, barrel_rate, whiff_rate, sample_size

    Minimum 10 pitches per (pitch_type, hand) group to be included.
    """
    result_rows = []
    for season, df in frames:
        if df.empty or "pitch_type" not in df.columns or "p_throws" not in df.columns:
            continue
        df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")].copy()
        for (pt, hand), grp in df.groupby(["pitch_type", "p_throws"]):
            n = len(grp)
            if n < 10:
                continue
            bip = grp[grp["type"] == "X"] if "type" in grp.columns else grp
            barrels = (bip["launch_speed_angle"] == 6).sum() if "launch_speed_angle" in bip.columns else 0
            swings = grp["description"].isin(SWING_DESCS).sum() if "description" in grp.columns else 0
            whiffs = grp["description"].isin(WHIFF_DESCS).sum() if "description" in grp.columns else 0
            result_rows.append({
                "season":      season,
                "pitch_type":  pt,
                "pitch_label": PITCH_NAMES.get(pt, pt),
                "vs_hand":     hand,
                "barrel_rate": float(barrels / len(bip)) if len(bip) > 0 else 0.0,
                "whiff_rate":  float(whiffs / swings) if swings > 0 else 0.0,
                "sample_size": n,
            })

    if not result_rows:
        return pd.DataFrame()
    return pd.DataFrame(result_rows).sort_values(["season", "pitch_type", "vs_hand"])


def parse_batter_hot_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 9-zone hot zone grid.

    Input:  raw Statcast DataFrame for a batter
    Output: 9-row DataFrame with columns: zone (1-9), barrel_rate, xba, n

    Expects columns: zone, type, launch_speed_angle, estimated_ba_using_speedangle
    """
    if df.empty or "zone" not in df.columns:
        return pd.DataFrame()

    df = df[df["zone"].between(1, 9)].copy()
    rows = []
    for zone in range(1, 10):
        grp = df[df["zone"] == zone]
        n = len(grp)
        if n == 0:
            rows.append({"zone": zone, "barrel_rate": 0.0, "xba": float("nan"), "n": 0})
            continue
        bip = grp[grp["type"] == "X"] if "type" in grp.columns else grp
        barrels = (bip["launch_speed_angle"] == 6).sum() if "launch_speed_angle" in bip.columns else 0
        xba_vals = (
            bip["estimated_ba_using_speedangle"].dropna()
            if "estimated_ba_using_speedangle" in bip.columns
            else pd.Series(dtype=float)
        )
        rows.append({
            "zone":        zone,
            "barrel_rate": float(barrels / len(bip)) if len(bip) > 0 else 0.0,
            "xba":         float(xba_vals.mean()) if len(xba_vals) >= 3 else float("nan"),
            "n":           n,
        })
    return pd.DataFrame(rows)


def parse_batter_barrel_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling 15-game barrel rate trend.

    Input:  raw Statcast DataFrame for a batter
    Output: one row per game_date with:
        game_date, bip_count, barrels, barrel_rate, rolling_barrel_rate

    Expects columns: game_date, type, launch_speed_angle
    """
    if df.empty or "game_date" not in df.columns:
        return pd.DataFrame()

    bip = df[df["type"] == "X"].copy() if "type" in df.columns else df.copy()
    bip["is_barrel"] = (
        (bip["launch_speed_angle"] == 6).astype(int)
        if "launch_speed_angle" in bip.columns
        else 0
    )
    by_game = (
        bip.groupby("game_date")
        .agg(bip_count=("is_barrel", "count"), barrels=("is_barrel", "sum"))
        .reset_index()
        .sort_values("game_date")
    )
    by_game["barrel_rate"] = by_game["barrels"] / by_game["bip_count"].clip(lower=1)
    by_game["rolling_barrel_rate"] = by_game["barrel_rate"].rolling(15, min_periods=3).mean()
    return by_game


def parse_batter_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-game batting stats.

    Input:  raw Statcast DataFrame for a batter
    Output: one row per game_date (descending) with:
        game_date, ab, hits, hr, k, pitches, avg

    Expects columns: game_date, events, pitch_type
    """
    if df.empty or "game_date" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    if "events" in df.columns:
        df["is_hit"] = df["events"].isin(_HIT_EVENTS).astype(int)
        df["is_ab"] = df["events"].isin(_AB_EVENTS).astype(int)
        df["is_hr"] = (df["events"] == "home_run").astype(int)
        df["is_k"] = df["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    else:
        for c in ["is_hit", "is_ab", "is_hr", "is_k"]:
            df[c] = 0

    by_game = (
        df.groupby("game_date")
        .agg(
            ab=("is_ab", "sum"), hits=("is_hit", "sum"),
            hr=("is_hr", "sum"), k=("is_k", "sum"), pitches=("pitch_type", "count"),
        )
        .reset_index()
        .sort_values("game_date", ascending=False)
    )
    by_game["avg"] = (by_game["hits"] / by_game["ab"].clip(lower=1)).round(3)
    return by_game


def parse_league_avg_krate(splits: list[dict]) -> float:
    """
    Compute league-wide K rate from a list of stat splits.

    Input:  splits from statsapi stats endpoint
            (each dict has {"stat": {"strikeOuts": int, "plateAppearances": int}})
    Output: float K/PA rate, defaults to 0.235 on failure
    """
    try:
        total_k = sum(int(s.get("stat", {}).get("strikeOuts", 0)) for s in splits)
        total_pa = sum(int(s.get("stat", {}).get("plateAppearances", 0)) for s in splits)
        return total_k / total_pa if total_pa else 0.235
    except Exception:
        return 0.235
