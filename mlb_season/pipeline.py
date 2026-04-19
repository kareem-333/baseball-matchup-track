"""
mlb_season/pipeline.py — Season-level MLB data pipeline.

All Statcast data is fetched directly from Baseball Savant CSV exports
(no pybaseball dependency). Live game data uses mlb-statsapi.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests
import statsapi
import streamlit as st

# ── constants ────────────────────────────────────────────────────────────────

PITCH_NAMES = {
    "FF": "4-Seam FB", "SI": "Sinker",   "FC": "Cutter",
    "SL": "Slider",    "ST": "Sweeper",  "CH": "Changeup",
    "CU": "Curveball", "KC": "K-Curve",  "FS": "Splitter",
    "SV": "Slurve",    "KN": "Knuckler", "CS": "Slow Curve",
    "FO": "Forkball",  "EP": "Eephus",   "SC": "Screwball",
}
PITCH_LABELS = PITCH_NAMES  # alias

_SAVANT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    )
}

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "hit_into_play", "foul_bunt", "missed_bunt",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked"}

SEASON_START = {
    2024: date(2024, 3, 20),
    2025: date(2025, 3, 27),
    2026: date(2026, 4, 3),
}


def _season_start(season: int) -> date:
    return SEASON_START.get(season, date(season, 4, 1))

def _season_start_str(season: int) -> str:
    return _season_start(season).strftime("%Y-%m-%d")

def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


# ── Baseball Savant CSV fetch ─────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_statcast_csv(player_id: int, player_type: str, season: int) -> pd.DataFrame:
    """Download one season of Statcast pitch-level data from Baseball Savant."""
    lookup = (
        f"batters_lookup%5B%5D={player_id}"
        if player_type == "batter"
        else f"pitchers_lookup%5B%5D={player_id}"
    )
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv?"
        f"all=true&hfGT=R%7C&hfSea={season}%7C&player_type={player_type}"
        f"&{lookup}&min_pitches=0&min_results=0"
        "&group_by=name&sort_col=pitches&sort_order=desc&type=details"
    )
    try:
        r = requests.get(url, headers=_SAVANT_HEADERS, timeout=30)
        if r.status_code != 200 or len(r.content) < 200:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(r.text), low_memory=False)
        df.columns = df.columns.str.lstrip("\ufeff").str.strip('"').str.strip("'")
        for col in [
            "plate_x", "plate_z", "pfx_x", "pfx_z",
            "release_speed", "release_spin_rate",
            "estimated_ba_using_speedangle", "launch_speed",
            "launch_angle", "api_break_x_arm",
            "api_break_z_with_gravity", "delta_home_win_exp",
            "home_win_exp", "bat_speed", "swing_length",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["zone", "launch_speed_angle"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        return df
    except Exception:
        return pd.DataFrame()


def _best_season_df(player_id: int, player_type: str) -> pd.DataFrame:
    cur = date.today().year
    df = fetch_statcast_csv(player_id, player_type, cur)
    if len(df) >= 30:
        return df
    prev = fetch_statcast_csv(player_id, player_type, cur - 1)
    return prev if len(prev) > len(df) else df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_statcast_multi_season(
    player_id: int,
    player_type: str,
    seasons: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Fetch and concatenate Statcast data across multiple seasons."""
    if seasons is None:
        cur = date.today().year
        seasons = tuple(range(cur - 3, cur + 1))
    frames = []
    for yr in seasons:
        df = fetch_statcast_csv(player_id, player_type, yr)
        if not df.empty:
            df = df.copy()
            df["season"] = yr
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["season"] = combined["season"].astype(int)
    return combined


# ── pitcher arsenal ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_pitcher_arsenal(pitcher_id: int, season: int | None = None) -> pd.DataFrame:
    """
    Pitcher pitch mix for the season.
    Returns: pitch_type, pitch_label, count, usage_pct (0-1), avg_velocity,
             avg_velo, avg_spin, avg_h_break, avg_v_break, avg_x, avg_z,
             last_appearance_date, appearances_used
    """
    season = season or date.today().year
    df = fetch_statcast_csv(pitcher_id, "pitcher", season)
    if df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()
    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]
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
            "pitch_type":         pt,
            "pitch_label":        PITCH_NAMES.get(pt, pt),
            "pitch_name":         PITCH_NAMES.get(pt, pt),
            "count":              len(g),
            "usage_pct":          len(g) / total,
            "avg_velocity":       round(g["release_speed"].mean(), 1) if "release_speed" in g.columns else None,
            "avg_velo":           g["release_speed"].mean() if "release_speed" in g.columns else 0,
            "avg_spin":           g["release_spin_rate"].mean() if "release_spin_rate" in g.columns else 0,
            "avg_h_break":        g["api_break_x_arm"].mean() if "api_break_x_arm" in g.columns else 0,
            "avg_v_break":        g["api_break_z_with_gravity"].mean() if "api_break_z_with_gravity" in g.columns else 0,
            "avg_x":              g["plate_x"].mean() if "plate_x" in g.columns else 0,
            "avg_z":              g["plate_z"].mean() if "plate_z" in g.columns else 0,
            "last_appearance_date": appearances[0] if appearances else None,
            "appearances_used":     len(appearances),
        })
    return pd.DataFrame(rows).sort_values("usage_pct", ascending=False).reset_index(drop=True)


# ── batter pitch splits ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_batter_pitch_splits(batter_id: int, season: int | None = None) -> pd.DataFrame:
    """
    Batter performance vs each pitch type, split by pitcher handedness.
    Returns: pitch_type, pitch_label, vs_hand, barrel_rate (0-1),
             whiff_rate (0-1), sample_size, last_event_date, bip, xba, pitches
    """
    season = season or date.today().year
    df = _best_season_df(batter_id, "batter")
    if df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()
    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]
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
            barrels = (has_ev & (bip["launch_speed"] >= 98) & (bip["launch_angle"].between(26, 30))).sum()
        swings = grp["description"].isin(SWING_DESCS).sum() if "description" in grp.columns else 0
        whiffs = grp["description"].isin(WHIFF_DESCS).sum() if "description" in grp.columns else 0
        xba_vals = bip["estimated_ba_using_speedangle"].dropna() if "estimated_ba_using_speedangle" in bip.columns else pd.Series(dtype=float)
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


# ── career / multi-season batter splits ──────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def get_batter_career_pitch_splits(batter_id: int) -> pd.DataFrame:
    """Multi-season pitch-type breakdown with vs_hand split."""
    cur = date.today().year
    frames = []
    for yr in range(cur - 3, cur + 1):
        df = fetch_statcast_csv(batter_id, "batter", yr)
        if df.empty or "pitch_type" not in df.columns or "p_throws" not in df.columns:
            continue
        df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]
        for (pt, hand), grp in df.groupby(["pitch_type", "p_throws"]):
            n = len(grp)
            if n < 10:
                continue
            bip = grp[grp["type"] == "X"] if "type" in grp.columns else grp
            barrels = (bip["launch_speed_angle"] == 6).sum() if "launch_speed_angle" in bip.columns else 0
            swings = grp["description"].isin(SWING_DESCS).sum() if "description" in grp.columns else 0
            whiffs = grp["description"].isin(WHIFF_DESCS).sum() if "description" in grp.columns else 0
            frames.append({
                "season":      yr,
                "pitch_type":  pt,
                "pitch_label": PITCH_NAMES.get(pt, pt),
                "vs_hand":     hand,
                "barrel_rate": float(barrels / len(bip)) if len(bip) > 0 else 0.0,
                "whiff_rate":  float(whiffs / swings) if swings > 0 else 0.0,
                "sample_size": n,
            })
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_values(["season", "pitch_type", "vs_hand"])


# ── hot zones ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_batter_hot_zones(batter_id: int, season: int | None = None) -> pd.DataFrame:
    """9-row DataFrame (Statcast zones 1-9) with barrel_rate, xba, n."""
    season = season or date.today().year
    df = fetch_statcast_csv(batter_id, "batter", season)
    if df.empty or "zone" not in df.columns:
        return pd.DataFrame()
    df = df[df["zone"].between(1, 9)]
    rows = []
    for zone in range(1, 10):
        grp = df[df["zone"] == zone]
        n = len(grp)
        if n == 0:
            rows.append({"zone": zone, "barrel_rate": 0.0, "xba": float("nan"), "n": 0})
            continue
        bip = grp[grp["type"] == "X"] if "type" in grp.columns else grp
        barrels = (bip["launch_speed_angle"] == 6).sum() if "launch_speed_angle" in bip.columns else 0
        xba_vals = bip["estimated_ba_using_speedangle"].dropna() if "estimated_ba_using_speedangle" in bip.columns else pd.Series(dtype=float)
        rows.append({
            "zone":        zone,
            "barrel_rate": float(barrels / len(bip)) if len(bip) > 0 else 0.0,
            "xba":         float(xba_vals.mean()) if len(xba_vals) >= 3 else float("nan"),
            "n":           n,
        })
    return pd.DataFrame(rows)


# ── barrel trend ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_barrel_trend(batter_id: int, season: int | None = None) -> pd.DataFrame:
    """Rolling 15-game barrel rate."""
    season = season or date.today().year
    df = fetch_statcast_csv(batter_id, "batter", season)
    if df.empty or "game_date" not in df.columns:
        return pd.DataFrame()
    bip = df[df["type"] == "X"].copy() if "type" in df.columns else df.copy()
    bip["is_barrel"] = (bip["launch_speed_angle"] == 6).astype(int) if "launch_speed_angle" in bip.columns else 0
    by_game = (
        bip.groupby("game_date")
        .agg(bip_count=("is_barrel", "count"), barrels=("is_barrel", "sum"))
        .reset_index().sort_values("game_date")
    )
    by_game["barrel_rate"] = by_game["barrels"] / by_game["bip_count"].clip(lower=1)
    by_game["rolling_barrel_rate"] = by_game["barrel_rate"].rolling(15, min_periods=3).mean()
    return by_game


# ── game logs ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_batter_game_log(batter_id: int, season: int | None = None) -> pd.DataFrame:
    season = season or date.today().year
    df = fetch_statcast_csv(batter_id, "batter", season)
    if df.empty or "game_date" not in df.columns:
        return pd.DataFrame()
    hit_events = {"single", "double", "triple", "home_run"}
    ab_events  = hit_events | {"strikeout", "field_out", "grounded_into_double_play",
                                "force_out", "double_play", "fielders_choice",
                                "fielders_choice_out", "strikeout_double_play"}
    if "events" in df.columns:
        df["is_hit"] = df["events"].isin(hit_events).astype(int)
        df["is_ab"]  = df["events"].isin(ab_events).astype(int)
        df["is_hr"]  = (df["events"] == "home_run").astype(int)
        df["is_k"]   = df["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    else:
        for c in ["is_hit","is_ab","is_hr","is_k"]: df[c] = 0
    by_game = (
        df.groupby("game_date")
        .agg(ab=("is_ab","sum"), hits=("is_hit","sum"),
             hr=("is_hr","sum"), k=("is_k","sum"), pitches=("pitch_type","count"))
        .reset_index().sort_values("game_date", ascending=False)
    )
    by_game["avg"] = (by_game["hits"] / by_game["ab"].clip(lower=1)).round(3)
    return by_game


@st.cache_data(ttl=3600, show_spinner=False)
def get_pitcher_game_log(pitcher_id: int, season: int | None = None) -> pd.DataFrame:
    season = season or date.today().year
    df = fetch_statcast_csv(pitcher_id, "pitcher", season)
    if df.empty or "game_date" not in df.columns:
        return pd.DataFrame()
    if "events" in df.columns:
        df["is_k"] = df["events"].isin({"strikeout","strikeout_double_play"}).astype(int)
    else:
        df["is_k"] = 0
    if "description" in df.columns:
        df["is_whiff"] = df["description"].isin(WHIFF_DESCS).astype(int)
    else:
        df["is_whiff"] = 0
    by_game = (
        df.groupby("game_date")
        .agg(pitches=("pitch_type","count"), strikeouts=("is_k","sum"), whiffs=("is_whiff","sum"))
        .reset_index().sort_values("game_date", ascending=False)
    )
    by_game["whiff_rate"] = (by_game["whiffs"] / by_game["pitches"].clip(lower=1)).round(3)
    return by_game


# ── team / roster helpers ─────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_last_n_completed_games(team_id: int, n: int = 3) -> list[dict]:
    today = date.today()
    start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    try:
        games = statsapi.schedule(teamId=team_id, start_date=start, end_date=_today_str(), sportId=1)
        completed = [g for g in games if g.get("status") in ("Final","Game Over")]
        return list(reversed(completed))[:n]
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def aggregate_batting_stats(game_ids: list[int], team_id: int) -> pd.DataFrame:
    rows = []
    for gid in game_ids:
        try:
            data = statsapi.get("game", {"gamePk": gid})
            gd = data.get("gameData", {})
            box = data.get("liveData", {}).get("boxscore", {})
            side = "home" if gd.get("teams",{}).get("home",{}).get("id") == team_id else "away"
            for key, p in box.get("teams",{}).get(side,{}).get("players",{}).items():
                s = p.get("stats",{}).get("batting",{})
                if not s: continue
                rows.append({"player_id": p.get("person",{}).get("id"),
                              "name": p.get("person",{}).get("fullName",""),
                              "game_pk": gid, "ab": s.get("atBats",0),
                              "hits": s.get("hits",0), "hr": s.get("homeRuns",0),
                              "rbi": s.get("rbi",0), "k": s.get("strikeOuts",0)})
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby(["player_id","name"]).agg(
        ab=("ab","sum"), hits=("hits","sum"), hr=("hr","sum"), rbi=("rbi","sum"), k=("k","sum")
    ).reset_index()
    agg["avg"] = (agg["hits"] / agg["ab"].clip(lower=1)).round(3)
    return agg.sort_values("ab", ascending=False)


@st.cache_data(ttl=300, show_spinner=False)
def aggregate_pitching_stats(game_ids: list[int], team_id: int) -> pd.DataFrame:
    rows = []
    for gid in game_ids:
        try:
            data = statsapi.get("game", {"gamePk": gid})
            gd = data.get("gameData", {})
            box = data.get("liveData", {}).get("boxscore", {})
            side = "home" if gd.get("teams",{}).get("home",{}).get("id") == team_id else "away"
            for key, p in box.get("teams",{}).get(side,{}).get("players",{}).items():
                s = p.get("stats",{}).get("pitching",{})
                if not s or s.get("inningsPitched","0.0") == "0.0": continue
                rows.append({"player_id": p.get("person",{}).get("id"),
                              "name": p.get("person",{}).get("fullName",""),
                              "game_pk": gid, "ip": s.get("inningsPitched","0.0"),
                              "k": s.get("strikeOuts",0), "bb": s.get("baseOnBalls",0),
                              "er": s.get("earnedRuns",0), "hits": s.get("hits",0)})
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("game_pk")


@st.cache_data(ttl=3600, show_spinner=False)
def predict_lineup(team_id: int, n_games: int = 5) -> pd.DataFrame:
    games = get_last_n_completed_games(team_id, n=n_games)
    pos_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    name_map: dict[int, str] = {}
    for g in games:
        gid = g.get("game_id")
        if not gid: continue
        try:
            data = statsapi.get("game", {"gamePk": gid})
            gd = data.get("gameData", {})
            box = data.get("liveData", {}).get("boxscore", {})
            side = "home" if gd.get("teams",{}).get("home",{}).get("id") == team_id else "away"
            order = box.get("teams",{}).get(side,{}).get("battingOrder", [])
            players = box.get("teams",{}).get(side,{}).get("players", {})
            for slot, pid in enumerate(order[:9], 1):
                pos_counts[pid][slot] += 1
                name_map[pid] = players.get(f"ID{pid}",{}).get("person",{}).get("fullName", str(pid))
        except Exception:
            continue
    if not pos_counts: return pd.DataFrame()
    rows = [{"order": max(slots, key=slots.get), "player_id": pid, "name": name_map.get(pid, str(pid))}
            for pid, slots in pos_counts.items()]
    return (pd.DataFrame(rows).sort_values("order").drop_duplicates("order").head(9).reset_index(drop=True))


@st.cache_data(ttl=3600, show_spinner=False)
def get_lineup_with_ids(team_id: int) -> pd.DataFrame:
    return predict_lineup(team_id)


@st.cache_data(ttl=3600, show_spinner=False)
def get_team_pitching_staff(team_id: int) -> pd.DataFrame:
    try:
        data = statsapi.get("roster", {"teamId": team_id, "rosterType": "active"})
        return pd.DataFrame([
            {"player_id": p["person"]["id"], "name": p["person"]["fullName"],
             "position": p.get("position",{}).get("abbreviation","")}
            for p in data.get("roster",[])
            if p.get("position",{}).get("abbreviation","") in ("P","SP","RP")
        ])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_team_batting_leaders(team_id: int, season: int | None = None) -> pd.DataFrame:
    season = season or date.today().year
    try:
        data = statsapi.get("stats", {"stats":"season","group":"hitting",
                                       "teamId":team_id,"season":season,"playerPool":"All"})
        splits = data.get("stats",[{}])[0].get("splits",[])
        rows = [{"player_id": s.get("player",{}).get("id"),
                 "name": s.get("player",{}).get("fullName",""),
                 "pa": s.get("stat",{}).get("plateAppearances",0),
                 "avg": s.get("stat",{}).get("avg",".000"),
                 "obp": s.get("stat",{}).get("obp",".000"),
                 "slg": s.get("stat",{}).get("slg",".000"),
                 "ops": s.get("stat",{}).get("ops",".000"),
                 "hr":  s.get("stat",{}).get("homeRuns",0),
                 "rbi": s.get("stat",{}).get("rbi",0)} for s in splits]
        df = pd.DataFrame(rows)
        if df.empty: return df
        df["pa"] = pd.to_numeric(df["pa"], errors="coerce").fillna(0).astype(int)
        return df[df["pa"] >= 10].sort_values("ops", ascending=False).head(15)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def get_league_avg_krate(season: int | None = None) -> float:
    season = season or date.today().year
    try:
        data = statsapi.get("stats", {"stats":"season","group":"hitting","season":season,"playerPool":"All"})
        splits = data.get("stats",[{}])[0].get("splits",[])
        total_k  = sum(int(s.get("stat",{}).get("strikeOuts",0))  for s in splits)
        total_pa = sum(int(s.get("stat",{}).get("plateAppearances",0)) for s in splits)
        return total_k / total_pa if total_pa else 0.235
    except Exception:
        return 0.235


# ── league pitch baselines (for MAS z-scores) ────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_league_pitch_baselines(season: int | None = None) -> dict:
    """
    League-wide mean/std for barrel_rate, whiff_rate, usage_pct per pitch type.
    Sampled from STAR_BATTERS in core.config. Falls back to hardcoded values.
    """
    season = season or date.today().year
    try:
        from core.config import STAR_BATTERS
        sample_ids = list(STAR_BATTERS.values())[:12]
        batter_rows = []
        for bid in sample_ids:
            df = fetch_statcast_csv(bid, "batter", season)
            if df.empty or "pitch_type" not in df.columns:
                continue
            df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]
            for pt, grp in df.groupby("pitch_type"):
                n = len(grp)
                if n < 5: continue
                bip = grp[grp["type"] == "X"] if "type" in grp.columns else grp
                barrels = (bip["launch_speed_angle"] == 6).sum() if "launch_speed_angle" in bip.columns else 0
                swings = grp["description"].isin(SWING_DESCS).sum() if "description" in grp.columns else 0
                whiffs = grp["description"].isin(WHIFF_DESCS).sum() if "description" in grp.columns else 0
                batter_rows.append({
                    "pitch_type":  pt,
                    "barrel_rate": float(barrels / len(bip)) if len(bip) > 0 else 0.0,
                    "whiff_rate":  float(whiffs / swings) if swings > 0 else 0.0,
                })
        if not batter_rows:
            return _fallback_baselines()
        bdf = pd.DataFrame(batter_rows)
        fb  = _fallback_baselines()
        baselines: dict = {}
        for pt in bdf["pitch_type"].unique():
            sub  = bdf[bdf["pitch_type"] == pt]
            fb_pt = fb.get(pt, fb["FF"])
            br_std = float(sub["barrel_rate"].std()) if len(sub) >= 3 else fb_pt["barrel_rate_std"]
            wr_std = float(sub["whiff_rate"].std())  if len(sub) >= 3 else fb_pt["whiff_rate_std"]
            baselines[pt] = {
                "barrel_rate_mean": float(sub["barrel_rate"].mean()) if len(sub) >= 3 else fb_pt["barrel_rate_mean"],
                "barrel_rate_std":  br_std if br_std > 0 else fb_pt["barrel_rate_std"],
                "whiff_rate_mean":  float(sub["whiff_rate"].mean())  if len(sub) >= 3 else fb_pt["whiff_rate_mean"],
                "whiff_rate_std":   wr_std if wr_std > 0 else fb_pt["whiff_rate_std"],
                "usage_pct_mean":   fb_pt["usage_pct_mean"],
                "usage_pct_std":    fb_pt["usage_pct_std"],
            }
        for pt, vals in fb.items():
            if pt not in baselines:
                baselines[pt] = vals
        return baselines
    except Exception:
        return _fallback_baselines()


def _fallback_baselines() -> dict:
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
        "CS": (0.025, 0.022, 0.280, 0.135, 0.030, 0.025),
    }
    return {pt: {"barrel_rate_mean": v[0], "barrel_rate_std": v[1],
                  "whiff_rate_mean": v[2],  "whiff_rate_std": v[3],
                  "usage_pct_mean":  v[4],  "usage_pct_std":  v[5]}
            for pt, v in defaults.items()}


def get_player_headshot_url(player_id: int) -> str:
    return (
        f"https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"d_people:generic:headshot:67:current.png/w_213,q_auto:best/"
        f"v1/people/{player_id}/headshot/67/current"
    )
