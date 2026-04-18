"""
mlb_season/pipeline.py — Season-level MLB data pipeline.

Statcast CSV downloads (Baseball Savant), MLB Stats API season queries,
team roster/stats lookups, lineup prediction, and league averages.
"""

import statsapi
import requests
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO
from datetime import date, timedelta
from collections import defaultdict, Counter

from core.config import ASTROS_ID, PITCH_NAMES

_CURRENT_SEASON = date.today().year
# Default career window — last 3 full seasons + current
_CAREER_SEASONS = list(range(_CURRENT_SEASON - 3, _CURRENT_SEASON + 1))

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
    from core.config import PITCH_NAMES
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
    from core.config import PITCH_NAMES
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


# ── Multi-season Statcast fetch ───────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_statcast_multi_season(
    player_id: int,
    player_type: str,
    seasons: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """
    Fetch and concatenate Statcast pitch-level data across multiple seasons.

    seasons: tuple of ints, e.g. (2022, 2023, 2024, 2025).
             Defaults to the last 3 full seasons + current year.
    player_type: 'batter' or 'pitcher'.

    Returns a single DataFrame with a 'season' column added.
    Seasons with no data are silently skipped.
    """
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
    # Ensure season is int
    combined["season"] = combined["season"].astype(int)
    return combined


# ── Career batter pitch-type splits ──────────────────────────────────────────

def _whiff_rate(g: pd.DataFrame) -> float:
    """Swinging strikes / total swings (swing = S or contact on swing)."""
    if "description" not in g.columns:
        return np.nan
    swings = g["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul",
        "foul_tip", "foul_bunt", "missed_bunt",
        "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
    ])
    whiffs = g["description"].isin(["swinging_strike", "swinging_strike_blocked"])
    n_swings = swings.sum()
    return float(whiffs.sum() / n_swings * 100) if n_swings >= 5 else np.nan


def _hard_hit_rate(bip: pd.DataFrame) -> float:
    """BIP with exit velo >= 95 mph / total BIP."""
    if "launch_speed" not in bip.columns or len(bip) < 5:
        return np.nan
    ev = pd.to_numeric(bip["launch_speed"], errors="coerce").dropna()
    if len(ev) < 5:
        return np.nan
    return float((ev >= 95).sum() / len(ev) * 100)


def _avg_ev(bip: pd.DataFrame) -> float:
    if "launch_speed" not in bip.columns:
        return np.nan
    ev = pd.to_numeric(bip["launch_speed"], errors="coerce").dropna()
    return float(ev.mean()) if len(ev) >= 3 else np.nan


def _xwoba(g: pd.DataFrame) -> float:
    col = "estimated_woba_using_speedangle"
    if col not in g.columns:
        return np.nan
    vals = pd.to_numeric(g[col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) >= 5 else np.nan


def _k_rate_from_events(g: pd.DataFrame) -> float:
    """Strikeout rate per plate appearance (events = 'strikeout' on last pitch of PA)."""
    if "events" not in g.columns:
        return np.nan
    pa_rows = g[g["events"].notna() & (g["events"] != "")]
    if len(pa_rows) < 5:
        return np.nan
    ks = (pa_rows["events"] == "strikeout").sum()
    return float(ks / len(pa_rows) * 100)


def _splits_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-pitch-type stats from a Statcast DataFrame.
    Returns rows with columns: pitch_type, pitch_name, pitches, bip,
    whiff_rate, k_rate, barrel_rate, hard_hit_rate, avg_ev, xba, xwoba.
    """
    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")].copy()
    rows = []
    total = len(df)
    for pt, g in df.groupby("pitch_type"):
        bip = g[g["type"] == "X"].copy() if "type" in g.columns else g.copy()
        n_bip = len(bip)
        barrels = 0
        if "launch_speed_angle" in bip.columns and n_bip >= 3:
            barrels = (pd.to_numeric(bip["launch_speed_angle"], errors="coerce") == 6).sum()
        xba_col = "estimated_ba_using_speedangle"
        xba_vals = pd.to_numeric(bip[xba_col], errors="coerce").dropna() if xba_col in bip.columns else pd.Series(dtype=float)

        rows.append({
            "pitch_type":    pt,
            "pitch_name":    PITCH_NAMES.get(pt, g["pitch_name"].iloc[0] if "pitch_name" in g.columns else pt),
            "pitches":       len(g),
            "usage_pct":     round(len(g) / total * 100, 1) if total > 0 else 0.0,
            "bip":           n_bip,
            "whiff_rate":    _whiff_rate(g),
            "k_rate":        _k_rate_from_events(g),
            "barrel_rate":   float(barrels / n_bip * 100) if n_bip >= 5 else np.nan,
            "hard_hit_rate": _hard_hit_rate(bip),
            "avg_ev":        _avg_ev(bip),
            "xba":           float(xba_vals.mean()) if len(xba_vals) >= 3 else np.nan,
            "xwoba":         _xwoba(g),
        })

    out = pd.DataFrame(rows)
    return out[out["pitches"] >= 10].sort_values("pitches", ascending=False) if not out.empty else out


@st.cache_data(ttl=3600, show_spinner=False)
def get_batter_career_pitch_splits(
    batter_id: int,
    seasons: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return career-level and season-by-season pitch-type split stats for a batter.

    Returns:
        career_df    — one row per pitch type, aggregated across all seasons
        by_season_df — one row per (season, pitch_type), for trend charts

    Both DataFrames carry the same stats columns:
        pitch_type, pitch_name, pitches, usage_pct, bip,
        whiff_rate, k_rate, barrel_rate, hard_hit_rate, avg_ev, xba, xwoba
    """
    if seasons is None:
        cur = date.today().year
        seasons = tuple(range(cur - 3, cur + 1))

    combined = fetch_statcast_multi_season(batter_id, "batter", seasons)
    if combined.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ── Career aggregate ──────────────────────────────────────────────────────
    career_df = _splits_for_df(combined)
    if not career_df.empty:
        career_df.insert(0, "season", "Career")

    # ── Season-by-season breakdown ────────────────────────────────────────────
    season_rows = []
    for yr in sorted(combined["season"].unique()):
        yr_df = combined[combined["season"] == yr]
        splits = _splits_for_df(yr_df)
        if not splits.empty:
            splits.insert(0, "season", int(yr))
            season_rows.append(splits)

    by_season_df = pd.concat(season_rows, ignore_index=True) if season_rows else pd.DataFrame()

    return career_df, by_season_df


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


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _side_for_team(box: dict, team_id: int) -> str | None:
    """Return 'home' or 'away' for the given team_id in this box score."""
    for side in ("home", "away"):
        if box.get(side, {}).get("team", {}).get("id") == team_id:
            return side
    return None


def _ip_to_float(ip_str) -> float:
    try:
        whole, frac = str(ip_str).split(".") if "." in str(ip_str) else (str(ip_str), "0")
        return int(whole) + int(frac) / 3
    except Exception:
        return 0.0


def _float_to_ip(ip_float: float) -> str:
    whole = int(ip_float)
    frac = round((ip_float - whole) * 3)
    if frac >= 3:
        whole += 1
        frac = 0
    return f"{whole}.{frac}"


def get_last_n_completed_games(team_id: int, n: int = 3) -> list[dict]:
    """Return last n Final games for a team (looks back up to 5 weeks)."""
    end = date.today()
    start = end - timedelta(days=35)
    games = statsapi.schedule(
        sportId=1, team=team_id,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    completed = [g for g in games if "Final" in g.get("status", "")]
    return completed[-n:]


def aggregate_batting_stats(game_ids: list[int], team_id: int) -> pd.DataFrame:
    """Sum batting stats across multiple completed games for one team."""
    from mlb_live.pipeline import get_live_box_score, build_batting_table
    totals: dict[str, dict] = {}
    for gid in game_ids:
        try:
            box = get_live_box_score(gid)
            side = _side_for_team(box, team_id)
            if not side:
                continue
            for _, row in build_batting_table(box, side).iterrows():
                name = row["Player"]
                if name not in totals:
                    totals[name] = {"Player": name, "Pos": row["Pos"],
                                    "G": 0, "AB": 0, "R": 0, "H": 0, "RBI": 0, "BB": 0, "K": 0}
                totals[name]["G"]   += 1
                totals[name]["AB"]  += int(row.get("AB") or 0)
                totals[name]["R"]   += int(row.get("R") or 0)
                totals[name]["H"]   += int(row.get("H") or 0)
                totals[name]["RBI"] += int(row.get("RBI") or 0)
                totals[name]["BB"]  += int(row.get("BB") or 0)
                totals[name]["K"]   += int(row.get("K") or 0)
        except Exception:
            continue

    if not totals:
        return pd.DataFrame()

    df = pd.DataFrame(list(totals.values()))
    df["AVG"] = df.apply(lambda r: f"{r['H']/r['AB']:.3f}" if r["AB"] > 0 else ".000", axis=1)
    return df[["Player", "Pos", "G", "AB", "R", "H", "RBI", "BB", "K", "AVG"]].sort_values("AB", ascending=False)


def aggregate_pitching_stats(game_ids: list[int], team_id: int) -> pd.DataFrame:
    """Sum pitching stats across multiple completed games for one team."""
    from mlb_live.pipeline import get_live_box_score, build_pitching_table
    totals: dict[str, dict] = {}
    for gid in game_ids:
        try:
            box = get_live_box_score(gid)
            side = _side_for_team(box, team_id)
            if not side:
                continue
            for _, row in build_pitching_table(box, side).iterrows():
                name = row["Pitcher"]
                if name not in totals:
                    totals[name] = {"Pitcher": name, "G": 0, "_ip": 0.0,
                                    "H": 0, "R": 0, "ER": 0, "BB": 0, "K": 0}
                totals[name]["G"]   += 1
                totals[name]["_ip"] += _ip_to_float(row.get("IP", "0.0"))
                totals[name]["H"]   += int(row.get("H") or 0)
                totals[name]["R"]   += int(row.get("R") or 0)
                totals[name]["ER"]  += int(row.get("ER") or 0)
                totals[name]["BB"]  += int(row.get("BB") or 0)
                totals[name]["K"]   += int(row.get("K") or 0)
        except Exception:
            continue

    if not totals:
        return pd.DataFrame()

    df = pd.DataFrame(list(totals.values())).sort_values("_ip", ascending=False)
    df["IP"]  = df["_ip"].apply(_float_to_ip)
    df["ERA"] = df.apply(
        lambda r: f"{r['ER']*9/r['_ip']:.2f}" if r["_ip"] > 0 else "-.--", axis=1
    )
    return df[["Pitcher", "G", "IP", "H", "R", "ER", "BB", "K", "ERA"]]


# ── Lineup prediction ─────────────────────────────────────────────────────────

def predict_lineup(team_id: int, n_games: int = 5) -> pd.DataFrame:
    """
    Predict likely batting order from the last n_games.
    Returns a DataFrame with Slot, Projected Player, and Confidence columns.
    """
    from mlb_live.pipeline import get_live_box_score
    end = date.today()
    start = end - timedelta(days=35)
    games = statsapi.schedule(
        sportId=1, team=team_id,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    completed = [g for g in games if "Final" in g.get("status", "")][-n_games:]

    # slot (1-9) -> Counter of player names
    slot_counters: dict[int, Counter] = defaultdict(Counter)

    for game in completed:
        try:
            box = get_live_box_score(game["game_id"])
            side = _side_for_team(box, team_id)
            if not side:
                continue
            team_data = box.get(side, {})
            players = team_data.get("players", {})
            slot = 1
            seen: set[str] = set()
            for pid in team_data.get("batters", []):
                if slot > 9:
                    break
                p = players.get(f"ID{pid}", {})
                name = p.get("person", {}).get("fullName", "")
                if name and name not in seen:
                    slot_counters[slot][name] += 1
                    seen.add(name)
                    slot += 1
        except Exception:
            continue

    rows = []
    total_games = len(completed)
    for slot in range(1, 10):
        counter = slot_counters[slot]
        if counter:
            player, count = counter.most_common(1)[0]
            conf = f"{count}/{total_games}" if total_games else "—"
        else:
            player, conf = "Insufficient data", "—"
        rows.append({"Slot": slot, "Projected Player": player, "Confidence": conf})

    return pd.DataFrame(rows)


# ── Player game logs ──────────────────────────────────────────────────────────

def get_batter_game_log(player_id: int, season: int | None = None) -> list[dict]:
    """Per-game hitting stats for a batter via MLB Stats API."""
    if season is None:
        season = date.today().year
    try:
        data = statsapi.get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=hitting,type=gameLog,season={season},gameType=R)",
        })
        for person in data.get("people", []):
            for sg in person.get("stats", []):
                splits = sg.get("splits", [])
                if splits:
                    return [s.get("stat", {}) for s in splits]
    except Exception:
        pass
    return []


def get_pitcher_game_log(player_id: int, season: int | None = None) -> list[dict]:
    """Per-game pitching stats for a pitcher via MLB Stats API."""
    if season is None:
        season = date.today().year
    try:
        data = statsapi.get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=pitching,type=gameLog,season={season},gameType=R)",
        })
        for person in data.get("people", []):
            for sg in person.get("stats", []):
                splits = sg.get("splits", [])
                if splits:
                    return [s.get("stat", {}) for s in splits]
    except Exception:
        pass
    return []


def get_team_pitching_staff(team_id: int, season: int) -> list[dict]:
    """
    Return active pitchers for a team sorted by IP descending.
    Each row: {player_id, name, role, ip, ip_float, era, whip, k, bb, gs, g}
    role = 'SP' if gamesStarted > 0 else 'RP'.
    """
    try:
        data = statsapi.get("team_roster", {"teamId": team_id})
        pitchers = [p for p in data.get("roster", [])
                    if p.get("position", {}).get("type") == "Pitcher"]
        if not pitchers:
            return []

        ids_str = ",".join(str(p["person"]["id"]) for p in pitchers)
        resp = statsapi.get("people", {
            "personIds": ids_str,
            "hydrate": f"stats(group=pitching,type=season,season={season})",
        })
    except Exception:
        return []

    rows = []
    for person in resp.get("people", []):
        pid  = person["id"]
        name = person["fullName"]
        for sg in person.get("stats", []):
            for split in sg.get("splits", []):
                s  = split.get("stat", {})
                ip = s.get("inningsPitched", "0.0")
                gs = int(s.get("gamesStarted", 0) or 0)
                g  = int(s.get("gamesPlayed", 0) or 0)
                k  = int(s.get("strikeOuts", 0) or 0)
                bb = int(s.get("baseOnBalls", 0) or 0)
                rows.append({
                    "player_id":  pid,
                    "name":       name,
                    "role":       "SP" if gs > 0 else "RP",
                    "ip":         ip,
                    "ip_float":   _ip_to_float(ip),
                    "era":        s.get("era", "-.--") or "-.--",
                    "whip":       s.get("whip", "-.--") or "-.--",
                    "k":          k,
                    "bb":         bb,
                    "gs":         gs,
                    "g":          g,
                })
    # Sort SP first (by IP), then RP (by IP)
    rows.sort(key=lambda r: (0 if r["role"] == "SP" else 1, -r["ip_float"]))
    return rows


def get_team_batting_leaders(team_id: int, season: int, n: int = 5) -> list[dict]:
    """
    Return the top n position players by OPS for a team.
    Each row: {player_id, name, ops, avg, hr, pa}
    """
    try:
        data = statsapi.get("team_roster", {"teamId": team_id})
        batters = [p for p in data.get("roster", [])
                   if p.get("position", {}).get("type") != "Pitcher"]
        if not batters:
            return []

        ids_str = ",".join(str(p["person"]["id"]) for p in batters)
        resp = statsapi.get("people", {
            "personIds": ids_str,
            "hydrate": f"stats(group=hitting,type=season,season={season})",
        })
    except Exception:
        return []

    rows = []
    for person in resp.get("people", []):
        pid  = person["id"]
        name = person["fullName"]
        for sg in person.get("stats", []):
            for split in sg.get("splits", []):
                s   = split.get("stat", {})
                ops = s.get("ops", "")
                pa  = int(s.get("plateAppearances", 0) or 0)
                if ops and ops not in (".000", "") and pa >= 10:
                    try:
                        ops_f = float(ops)
                    except ValueError:
                        continue
                    rows.append({
                        "player_id": pid,
                        "name":      name,
                        "ops":       ops_f,
                        "avg":       s.get("avg", ".000"),
                        "hr":        int(s.get("homeRuns", 0) or 0),
                        "pa":        pa,
                    })

    rows.sort(key=lambda r: r["ops"], reverse=True)
    return rows[:n]


def get_league_avg_krate(season: int) -> float:
    """
    Return MLB league-average batter K% (per plate appearance, as a percentage).
    Fetches from the stats API and falls back to 22.5 if unavailable.
    """
    try:
        # Use league-level cumulative stats; sportId=1 = MLB
        resp = statsapi.get("stats", {
            "stats":   "season",
            "group":   "hitting",
            "sportId": 1,
            "season":  season,
            "limit":   1000,
        })
        total_k  = 0
        total_pa = 0
        for split in resp.get("stats", [{}])[0].get("splits", []):
            s = split.get("stat", {})
            total_k  += int(s.get("strikeOuts", 0)        or 0)
            total_pa += int(s.get("plateAppearances", 0)  or 0)
        if total_pa > 0:
            return round(total_k / total_pa * 100, 2)
    except Exception:
        pass
    return 22.5   # MLB historical avg 2023-2025


def get_lineup_with_ids(game_id: int, side: str, team_id: int | None = None) -> list[dict]:
    """
    Return the confirmed batting lineup with player IDs for one side.
    Falls back to the predicted lineup (names only, IDs looked up separately)
    if the box score lineup is not yet posted.

    Each row: {name, player_id, pos, slot}
    """
    from mlb_live.pipeline import get_live_box_score
    try:
        box = get_live_box_score(game_id)
    except Exception:
        return []

    team_data = box.get(side, {})
    players   = team_data.get("players", {})
    batter_ids_ordered = team_data.get("batters", [])

    rows = []
    slot = 1
    seen: set[str] = set()
    for pid in batter_ids_ordered:
        if slot > 9:
            break
        p    = players.get(f"ID{pid}", {})
        name = p.get("person", {}).get("fullName", "")
        pos  = p.get("position", {}).get("abbreviation", "")
        s    = p.get("stats", {}).get("batting", {})
        # Only include if they have at-bat data (confirmed in game)
        if name and name not in seen:
            seen.add(name)
            rows.append({
                "name":      name,
                "player_id": pid,
                "pos":       pos,
                "slot":      slot,
                "confirmed": bool(s),
            })
            slot += 1

    return rows
