"""
MLB Live Tracker — Data Pipeline
Fetches live, historical, and predictive game data via the MLB Stats API.
"""

import statsapi
import pandas as pd
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict, Counter

CT = ZoneInfo("America/Chicago")
ASTROS_ID = 117


def today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


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


def aggregate_batting_stats(game_ids: list[int], team_id: int) -> pd.DataFrame:
    """Sum batting stats across multiple completed games for one team."""
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


def get_pitcher_handedness(player_id: int) -> str:
    """
    Return 'R', 'L', or 'S' for the pitcher's throwing hand.
    Fetches from MLB people endpoint — not a manual input.
    """
    try:
        data = statsapi.get("people", {"personIds": player_id})
        for person in data.get("people", []):
            code = person.get("pitchHand", {}).get("code", "")
            if code:
                return code
    except Exception:
        pass
    return "?"


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
        import math
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


def lookup_player_id(name: str) -> int | None:
    """Return MLB player ID for a given name string."""
    try:
        results = statsapi.lookup_player(name)
        return results[0]["id"] if results else None
    except Exception:
        return None


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
