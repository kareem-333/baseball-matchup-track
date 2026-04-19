"""
mlb_live/pipeline.py — Live game data pipeline.

Covers: schedule lookups, live box scores, pitch-by-pitch feed,
win-probability calculation, fatigue metrics, and active-pitcher tracking.
All functions call the MLB Stats API directly; no long-term caching.
"""

import statsapi
import pandas as pd
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict, Counter
import math

from core.config import ASTROS_ID

CT = ZoneInfo("America/Chicago")


def today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


# ── All games (multi-team schedule) ──────────────────────────────────────────

def get_all_todays_games() -> list[dict]:
    """Return all MLB games scheduled today (every team, not just one)."""
    return statsapi.schedule(sportId=1, start_date=today_str(), end_date=today_str())


def get_game_lineup(game_id: int, team_id: int) -> list[dict]:
    """
    Return the batting order for a team in a given game.
    Each entry: {player_id, name, position}
    """
    try:
        box = get_live_box_score(game_id)
        side = _side_for_team(box, team_id)
        if side is None:
            return []
        team_data = box.get(side, {})
        players   = team_data.get("players", {})
        order     = team_data.get("batters", [])
        return [
            {
                "player_id": pid,
                "name":      players.get(f"ID{pid}", {}).get("person", {}).get("fullName", str(pid)),
                "position":  players.get(f"ID{pid}", {}).get("position", {}).get("abbreviation", ""),
            }
            for pid in order
        ]
    except Exception:
        return []


# ── Teams ─────────────────────────────────────────────────────────────────────

def get_all_teams() -> list[dict]:
    """Return all active MLB teams sorted alphabetically."""
    data = statsapi.get("teams", {"sportId": 1, "activeStatus": "Y"})
    teams = []
    for t in data.get("teams", []):
        if t.get("sport", {}).get("id") == 1:
            teams.append({
                "id": t["id"],
                "name": t["name"],
                "abbreviation": t.get("abbreviation", ""),
                "division": t.get("division", {}).get("name", ""),
                "league": t.get("league", {}).get("name", ""),
            })
    return sorted(teams, key=lambda x: x["name"])


# ── Schedule ──────────────────────────────────────────────────────────────────

def get_todays_game(team_id: int = ASTROS_ID) -> dict | None:
    """Return the first game scheduled today for a team, or None."""
    games = statsapi.schedule(sportId=1, team=team_id, start_date=today_str(), end_date=today_str())
    return games[0] if games else None


def get_upcoming_game(team_id: int) -> dict | None:
    """Return next scheduled game for a team (next 7 days)."""
    start = date.today()
    end = start + timedelta(days=7)
    games = statsapi.schedule(
        sportId=1, team=team_id,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    upcoming = [g for g in games if g.get("status") in
                ("Scheduled", "Pre-Game", "Preview", "Warmup")]
    return upcoming[0] if upcoming else None


def get_game_summary(game: dict) -> dict:
    return {
        "game_id":               game.get("game_id"),
        "status":                game.get("status"),
        "game_date":             game.get("game_date"),
        "game_datetime":         game.get("game_datetime"),
        "away_name":             game.get("away_name"),
        "home_name":             game.get("home_name"),
        "away_id":               game.get("away_id"),
        "home_id":               game.get("home_id"),
        "away_score":            game.get("away_score", 0),
        "home_score":            game.get("home_score", 0),
        "inning":                game.get("current_inning", "-"),
        "inning_state":          game.get("inning_state", ""),
        "venue":                 game.get("venue_name"),
        "away_probable_pitcher": game.get("away_probable_pitcher") or "TBD",
        "home_probable_pitcher": game.get("home_probable_pitcher") or "TBD",
    }


# ── Live game data ────────────────────────────────────────────────────────────

def get_live_box_score(game_id: int) -> dict:
    return statsapi.boxscore_data(game_id)


def get_linescore(game_id: int) -> dict:
    data = statsapi.get("game", {"gamePk": game_id})
    return data.get("liveData", {}).get("linescore", {})


def get_current_play(game_id: int) -> dict:
    data = statsapi.get("game", {"gamePk": game_id})
    return data.get("liveData", {}).get("plays", {}).get("currentPlay", {})


# ── Box score table builders ──────────────────────────────────────────────────

def build_inning_table(linescore: dict) -> pd.DataFrame:
    innings = linescore.get("innings", [])
    rows = [{"Inning": i.get("num"),
             "Away": i.get("away", {}).get("runs", "-"),
             "Home": i.get("home", {}).get("runs", "-")} for i in innings]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Inning", "Away", "Home"])


def build_batting_table(box: dict, side: str) -> pd.DataFrame:
    team_data = box.get(side, {})
    players = team_data.get("players", {})
    rows = []
    for pid in team_data.get("batters", []):
        p = players.get(f"ID{pid}", {})
        s = p.get("stats", {}).get("batting", {})
        if not s:
            continue
        rows.append({
            "Player": p.get("person", {}).get("fullName", ""),
            "Pos":    p.get("position", {}).get("abbreviation", ""),
            "AB":     s.get("atBats", 0),
            "R":      s.get("runs", 0),
            "H":      s.get("hits", 0),
            "RBI":    s.get("rbi", 0),
            "BB":     s.get("baseOnBalls", 0),
            "K":      s.get("strikeOuts", 0),
            "AVG":    s.get("avg", ".---"),
        })
    return pd.DataFrame(rows)


def build_pitching_table(box: dict, side: str) -> pd.DataFrame:
    team_data = box.get(side, {})
    players = team_data.get("players", {})
    rows = []
    for pid in team_data.get("pitchers", []):
        p = players.get(f"ID{pid}", {})
        s = p.get("stats", {}).get("pitching", {})
        if not s:
            continue
        rows.append({
            "Pitcher": p.get("person", {}).get("fullName", ""),
            "IP":      s.get("inningsPitched", "0.0"),
            "H":       s.get("hits", 0),
            "R":       s.get("runs", 0),
            "ER":      s.get("earnedRuns", 0),
            "BB":      s.get("baseOnBalls", 0),
            "K":       s.get("strikeOuts", 0),
            "ERA":     s.get("era", "-.--"),
        })
    return pd.DataFrame(rows)


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


# ── Live pitch feed ───────────────────────────────────────────────────────────

def get_all_live_pitches(game_id: int) -> list[dict]:
    """
    Extract every pitch from the live game feed with movement, location,
    velocity, spin, count, batter/pitcher identities, and inning.
    """
    try:
        data = statsapi.get("game", {"gamePk": game_id})
    except Exception:
        return []

    all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    pitches = []
    for play in all_plays:
        about   = play.get("about", {})
        matchup = play.get("matchup", {})
        for event in play.get("playEvents", []):
            if not event.get("isPitch", False):
                continue
            pd_data = event.get("pitchData", {})
            coords  = pd_data.get("coordinates", {})
            brk     = pd_data.get("breaks", {})
            details = event.get("details", {})
            ptype   = details.get("type", {})
            pitches.append({
                "inning":       about.get("inning"),
                "half":         about.get("halfInning", ""),
                "pitch_type":   ptype.get("code", ""),
                "pitch_name":   ptype.get("description", ""),
                "speed":        pd_data.get("startSpeed"),
                "end_speed":    pd_data.get("endSpeed"),
                "plate_x":      coords.get("pX"),
                "plate_z":      coords.get("pZ"),
                "pfx_x":        coords.get("pfxX"),
                "pfx_z":        coords.get("pfxZ"),
                "break_h":      brk.get("breakHorizontal"),
                "break_v":      brk.get("breakVertical"),
                "break_v_ind":  brk.get("breakVerticalInduced"),
                "spin_rate":    brk.get("spinRate"),
                "spin_dir":     brk.get("spinDirection"),
                "pitch_number": event.get("pitchNumber"),
                "balls":        event.get("count", {}).get("balls"),
                "strikes":      event.get("count", {}).get("strikes"),
                "outs":         event.get("count", {}).get("outs"),
                "description":  details.get("description", ""),
                "result_code":  details.get("code", ""),
                "batter_id":    matchup.get("batter", {}).get("id"),
                "batter_name":  matchup.get("batter", {}).get("fullName", ""),
                "pitcher_id":   matchup.get("pitcher", {}).get("id"),
                "pitcher_name": matchup.get("pitcher", {}).get("fullName", ""),
                "away_score":   play.get("result", {}).get("awayScore"),
                "home_score":   play.get("result", {}).get("homeScore"),
            })
    return pitches


def get_pitcher_live_pitches(game_id: int, pitcher_id: int) -> list[dict]:
    """Filter live pitches to a specific pitcher."""
    return [p for p in get_all_live_pitches(game_id) if p.get("pitcher_id") == pitcher_id]


def compute_pitcher_fatigue(pitches: list[dict]) -> dict:
    """
    Returns pitch_count, vel_drop_pct, spin_drop_pct for fatigue gauge.
    Baseline = first 15 pitches; current = last 15 pitches.
    """
    if not pitches:
        return {"pitch_count": 0, "vel_drop_pct": 0.0, "spin_drop_pct": 0.0}

    def _avg(vals):
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else 0.0

    n = len(pitches)
    baseline = pitches[:min(15, n)]
    recent   = pitches[max(0, n - 15):]

    base_velo = _avg([p["speed"]     for p in baseline])
    base_spin = _avg([p["spin_rate"] for p in baseline])
    cur_velo  = _avg([p["speed"]     for p in recent])
    cur_spin  = _avg([p["spin_rate"] for p in recent])

    vel_drop  = (base_velo - cur_velo) / base_velo * 100 if base_velo else 0.0
    spin_drop = (base_spin - cur_spin) / base_spin * 100 if base_spin else 0.0

    return {
        "pitch_count":  n,
        "vel_drop_pct": max(0.0, vel_drop),
        "spin_drop_pct":max(0.0, spin_drop),
    }


def get_win_probability_from_plays(game_id: int, home_team: str = "") -> list[dict]:
    """
    Build win-probability timeline from at-bat results.
    Uses score + inning to estimate WP when Statcast not available.
    """
    try:
        data = statsapi.get("game", {"gamePk": game_id})
    except Exception:
        return []

    all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    timeline  = []
    for play in all_plays:
        about  = play.get("about", {})
        result = play.get("result", {})
        if not about.get("isComplete", False):
            continue
        inning = about.get("inning", 1)
        half   = "Bot" if not about.get("isTopInning", True) else "Top"

        away_score = result.get("awayScore", 0) or 0
        home_score = result.get("homeScore", 0) or 0
        diff = home_score - away_score

        # Simple win-expectancy model (8.5-inning scale)
        innings_left = max(0, 9 - inning + (0 if half == "Bot" else 0.5))
        # Each inning = ~0.5 runs of uncertainty; roughly sigmoid by run diff
        try:
            wp = 1 / (1 + math.exp(-0.45 * diff * (1 + (9 - innings_left) / 9)))
        except Exception:
            wp = 0.5

        timeline.append({
            "inning_label": f"{half} {inning}",
            "inning":       inning,
            "half":         half,
            "away_score":   away_score,
            "home_score":   home_score,
            "home_win_exp": round(wp, 4),
        })
    return timeline


def get_game_pitchers(game_id: int, side: str) -> list[dict]:
    """
    Return every pitcher who has appeared in the game for one side,
    in order of appearance.  First pitcher = SP, rest = RP.

    Fields per row:
        pitcher_id, pitcher_name, hand, role ('SP'/'RP'),
        ip, pitches_thrown, k, bb, er, h
    """
    try:
        box = get_live_box_score(game_id)
    except Exception:
        return []

    team_data = box.get(side, {})
    players   = team_data.get("players", {})
    pid_order = team_data.get("pitchers", [])   # ordered list of player IDs

    rows = []
    for i, pid in enumerate(pid_order):
        p = players.get(f"ID{pid}", {})
        s = p.get("stats", {}).get("pitching", {})
        if not s:
            continue
        name = p.get("person", {}).get("fullName", "")
        rows.append({
            "pitcher_id":      pid,
            "pitcher_name":    name,
            "hand":            "?",          # filled lazily on demand
            "role":            "SP" if i == 0 else "RP",
            "ip":              s.get("inningsPitched", "0.0"),
            "pitches_thrown":  s.get("numberOfPitches", 0),
            "k":               s.get("strikeOuts", 0),
            "bb":              s.get("baseOnBalls", 0),
            "er":              s.get("earnedRuns", 0),
            "h":               s.get("hits", 0),
        })
    return rows


def get_active_pitcher(game_id: int) -> dict | None:
    """
    Return {pitcher_id, pitcher_name, half} for the pitcher currently on the mound.
    half = 'top' or 'bottom' — lets the caller know which team is pitching.
    Returns None if the game has not started or data is unavailable.
    """
    try:
        data = statsapi.get("game", {"gamePk": game_id})
        cp   = data.get("liveData", {}).get("plays", {}).get("currentPlay", {})
        if not cp:
            return None
        matchup = cp.get("matchup", {})
        pitcher = matchup.get("pitcher", {})
        half    = cp.get("about", {}).get("halfInning", "")
        pid     = pitcher.get("id")
        if not pid:
            return None
        return {
            "pitcher_id":   pid,
            "pitcher_name": pitcher.get("fullName", ""),
            "half":         half,   # 'top' or 'bottom'
        }
    except Exception:
        return None
