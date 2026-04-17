"""
NBA Defensive Analytics Pipeline
Fetches hustle, tracking, and advanced stats via nba_api.
Computes Dragon Index and Fortress Rating.

Column naming convention
------------------------
All user-facing metric columns use SCREAMING_SNAKE_CASE (DRAGON_INDEX, FORTRESS_RATING…).
Internal normalised component columns use lowercase (d_rim_n, f_blocks_n…).
"""

import time
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nba_api.stats.endpoints import (
    LeagueHustleStatsPlayer,
    LeagueDashPlayerStats,
    LeagueDashPlayerBioStats,
    LeagueDashPtStats,
    PlayerIndex,
)
from nba_api.stats.static import teams as nba_teams_static

log = logging.getLogger(__name__)

CACHE_DIR   = Path(__file__).parent / "cache" / "nba"
CACHE_TTL_S = 86_400          # 24 hours
SEASON      = "2024-25"
RATE_DELAY  = 0.75            # seconds between nba_api calls

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Team colors ───────────────────────────────────────────────────────────────

NBA_TEAM_COLORS: dict[int, str] = {
    1610612737: "#E03A3E",  # ATL
    1610612738: "#007A33",  # BOS
    1610612751: "#777D84",  # BKN
    1610612766: "#1D1160",  # CHA
    1610612741: "#CE1141",  # CHI
    1610612739: "#860038",  # CLE
    1610612742: "#00538C",  # DAL
    1610612743: "#0E2240",  # DEN
    1610612765: "#C8102E",  # DET
    1610612744: "#1D428A",  # GSW
    1610612745: "#CE1141",  # HOU
    1610612754: "#002D62",  # IND
    1610612746: "#C8102E",  # LAC
    1610612747: "#552583",  # LAL
    1610612763: "#5D76A9",  # MEM
    1610612748: "#98002E",  # MIA
    1610612749: "#00471B",  # MIL
    1610612750: "#236192",  # MIN
    1610612740: "#0C2340",  # NOP
    1610612752: "#006BB6",  # NYK
    1610612760: "#007AC1",  # OKC
    1610612753: "#007DC5",  # ORL
    1610612755: "#006BB6",  # PHI
    1610612756: "#1D1160",  # PHX
    1610612757: "#E03A3E",  # POR
    1610612758: "#5A2D81",  # SAC
    1610612759: "#C4CED4",  # SAS
    1610612761: "#CE1141",  # TOR
    1610612762: "#002B5C",  # UTA
    1610612764: "#002B5C",  # WAS
}

def team_color(team_id: int) -> str:
    return NBA_TEAM_COLORS.get(int(team_id), "#888888")

def nba_logo_url(team_id: int) -> str:
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.png"

def nba_headshot_url(player_id: int) -> str:
    # 1040x760 is the highest-quality CDN size
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"


# ── Disk cache helpers ────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"

def _is_cache_fresh(name: str) -> bool:
    p = _cache_path(name)
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) < CACHE_TTL_S

def _save(name: str, obj):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(obj, f)

def _load(name: str):
    with open(_cache_path(name), "rb") as f:
        return pickle.load(f)


# ── Rate-limited nba_api caller ───────────────────────────────────────────────

def _call(fn, *args, retries=3, **kwargs):
    """Call an nba_api endpoint with retry and rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(RATE_DELAY)
            return fn(*args, **kwargs).get_data_frames()[0]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RATE_DELAY * 2 ** attempt)
            else:
                log.warning(f"nba_api call failed after {retries} tries: {e}")
                return pd.DataFrame()


# ── Individual endpoint fetchers ──────────────────────────────────────────────

def _fetch_hustle() -> pd.DataFrame:
    # Correct param name is per_mode_time (not per_mode_time_frame)
    return _call(LeagueHustleStatsPlayer, season=SEASON, per_mode_time="PerGame")


def _fetch_advanced() -> pd.DataFrame:
    return _call(
        LeagueDashPlayerStats,
        season=SEASON,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )


def _fetch_opponent_stats() -> pd.DataFrame:
    """
    On-court opponent stats (MeasureType=Defense).
    Key cols used: OPP_PTS_OFF_TOV, OPP_PTS_FB, OPP_PTS_PAINT.
    """
    return _call(
        LeagueDashPlayerStats,
        season=SEASON,
        measure_type_detailed_defense="Defense",
        per_mode_detailed="PerGame",
    )


def _fetch_base() -> pd.DataFrame:
    return _call(
        LeagueDashPlayerStats,
        season=SEASON,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
    )


def _fetch_defense_tracking() -> pd.DataFrame:
    """Rim protection tracking — columns: DEF_RIM_FGM, DEF_RIM_FGA, DEF_RIM_FG_PCT."""
    return _call(
        LeagueDashPtStats,
        season=SEASON,
        pt_measure_type="Defense",
        player_or_team="Player",
        per_mode_simple="PerGame",
    )


def _fetch_bio() -> pd.DataFrame:
    """
    Fetch player positions via PlayerIndex (no season filter required).
    Returns DataFrame with PLAYER_ID and PLAYER_POSITION columns.
    """
    try:
        time.sleep(RATE_DELAY)
        df = PlayerIndex(league_id="00").get_data_frames()[0]
        # Rename PERSON_ID → PLAYER_ID to match other endpoints
        if "PERSON_ID" in df.columns:
            df = df.rename(columns={"PERSON_ID": "PLAYER_ID", "POSITION": "PLAYER_POSITION"})
        return df[["PLAYER_ID", "PLAYER_POSITION"]]
    except Exception as e:
        log.warning(f"PlayerIndex fetch failed: {e}")
        return pd.DataFrame()


# ── Utility helpers ───────────────────────────────────────────────────────────

def _safe_get(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _norm(s: pd.Series, clip_low: float = 0.0) -> pd.Series:
    """Min-max normalise to [0, 1]."""
    s = s.clip(lower=clip_low)
    r = s.max() - s.min()
    return (s - s.min()) / (r if r > 1e-9 else 1.0)


# ── Data merge ────────────────────────────────────────────────────────────────

def _merge_all(
    base: pd.DataFrame,
    advanced: pd.DataFrame,
    hustle: pd.DataFrame,
    defense: pd.DataFrame,
    bio: pd.DataFrame,
    opp_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()

    df = base[[c for c in
               ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
                "GP", "MIN", "STL", "BLK", "DREB", "OREB", "PTS", "AST"]
               if c in base.columns]].copy()

    # Advanced stats
    adv_cols = ["PLAYER_ID", "USG_PCT", "DEF_RATING", "OFF_RATING", "NET_RATING"]
    avail = [c for c in adv_cols if c in advanced.columns]
    if avail:
        df = df.merge(advanced[avail], on="PLAYER_ID", how="left")

    # Opponent on-court stats (Defense measure type)
    if opp_stats is not None and not opp_stats.empty:
        opp_cols = ["PLAYER_ID", "OPP_PTS_OFF_TOV", "OPP_PTS_FB", "OPP_PTS_PAINT"]
        avail_opp = [c for c in opp_cols if c in opp_stats.columns]
        if avail_opp:
            df = df.merge(opp_stats[avail_opp], on="PLAYER_ID", how="left")

    # Hustle stats  (BOX_OUT_PLAYER_REBS is the correct column name, not DEF_BOXOUT_PLAYER_REBS)
    hustle_want = ["PLAYER_ID", "CONTESTED_SHOTS", "CONTESTED_SHOTS_2PT",
                   "CONTESTED_SHOTS_3PT", "DEFLECTIONS", "CHARGES_DRAWN",
                   "DEF_LOOSE_BALLS_RECOVERED", "DEF_BOXOUTS", "BOX_OUT_PLAYER_REBS"]
    hustle_avail = [c for c in hustle_want if c in hustle.columns]
    if hustle_avail:
        df = df.merge(hustle[hustle_avail], on="PLAYER_ID", how="left")

    # Defense tracking (rim) — prefix TRK_ to avoid collision with base STL/BLK
    def_want = ["PLAYER_ID", "DEF_RIM_FGM", "DEF_RIM_FGA", "DEF_RIM_FG_PCT"]
    def_avail = [c for c in def_want if c in defense.columns]
    if def_avail:
        rename = {c: f"TRK_{c}" for c in def_avail if c != "PLAYER_ID"}
        df = df.merge(defense[def_avail].rename(columns=rename), on="PLAYER_ID", how="left")

    # Bio (position)
    if "PLAYER_ID" in bio.columns and "PLAYER_POSITION" in bio.columns:
        df = df.merge(bio[["PLAYER_ID", "PLAYER_POSITION"]], on="PLAYER_ID", how="left")

    return df.fillna(0.0)


# ── Dragon Index ──────────────────────────────────────────────────────────────
#
# Design principle: Dragon measures ACTIVE DISRUPTION on the perimeter and in
# transition.  Rim contests are intentionally excluded — contesting shots inside
# the paint is an interior/Fortress skill.  "Closest defender within a foot"
# only enters Dragon for shots OUTSIDE the paint (mid-range + 3pt).
#
#   Perimeter contests = CONTESTED_SHOTS_TOTAL − TRK_DEF_RIM_FGA
#   (hustle total contested minus rim-tracked contested shots)

DRAGON_WEIGHTS = {
    "deflections_pg":          0.25,   # deflections (forces turnovers / bad passes)
    "charges_drawn_pg":        0.20,   # charges drawn (elite anticipation)
    "steals_pg":               0.25,   # steals (hands/reads in passing lanes)
    "perimeter_contests_pg":   0.20,   # closest-defender contests OUTSIDE the paint
    "loose_balls_pg":          0.10,   # defensive loose balls recovered
}

def _compute_dragon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Perimeter contests = total hustle contested shots − rim tracked shots
    total_contested = _safe_get(df, "CONTESTED_SHOTS")
    rim_contested   = _safe_get(df, "TRK_DEF_RIM_FGA")   # shots at the rim
    df["d_perimeter_contests"] = (total_contested - rim_contested).clip(lower=0)

    df["d_deflections"] = _safe_get(df, "DEFLECTIONS")
    df["d_charges"]     = _safe_get(df, "CHARGES_DRAWN")
    df["d_steals"]      = _safe_get(df, "STL")
    df["d_loose_balls"] = _safe_get(df, "DEF_LOOSE_BALLS_RECOVERED")

    df["d_defl_n"]       = _norm(df["d_deflections"])
    df["d_charges_n"]    = _norm(df["d_charges"])
    df["d_steals_n"]     = _norm(df["d_steals"])
    df["d_perim_n"]      = _norm(df["d_perimeter_contests"])
    df["d_loose_n"]      = _norm(df["d_loose_balls"])

    usg = _norm(_safe_get(df, "USG_PCT"), clip_low=0.05)

    # OPP_PTS_OFF_TOV: how many pts the opponent scores off the player's team's turnovers
    # when this player is on court. High disruptors keep this LOW → invert it.
    opp_tov_pts = _safe_get(df, "OPP_PTS_OFF_TOV").replace(0, np.nan).fillna(
        _safe_get(df, "OPP_PTS_OFF_TOV").median() if "OPP_PTS_OFF_TOV" in df.columns else 7.0
    )
    tov_impact  = _norm(opp_tov_pts.max() - opp_tov_pts, clip_low=0)   # inverted: lower = better

    # Fall back to DEF_RATING if opponent stats unavailable
    if tov_impact.sum() == 0:
        tov_impact = _norm(100 - _safe_get(df, "DEF_RATING").replace(0, 100), clip_low=0)

    weight = 0.6 + 0.4 * (0.5 * usg + 0.5 * tov_impact)

    raw = (
        DRAGON_WEIGHTS["deflections_pg"]        * df["d_defl_n"]    +
        DRAGON_WEIGHTS["charges_drawn_pg"]       * df["d_charges_n"] +
        DRAGON_WEIGHTS["steals_pg"]              * df["d_steals_n"]  +
        DRAGON_WEIGHTS["perimeter_contests_pg"]  * df["d_perim_n"]   +
        DRAGON_WEIGHTS["loose_balls_pg"]         * df["d_loose_n"]
    ) * weight

    df["DRAGON_INDEX"] = (_norm(raw) * 100).round(1)
    return df


# ── Fortress Rating ───────────────────────────────────────────────────────────

FORTRESS_WEIGHTS = {
    "rim_fg_pct_inv":       0.28,
    "contested_reb_rate":   0.22,
    "blocks_pg":            0.22,
    "rim_contest_rate":     0.18,
    "putback_contribution": 0.10,
}

def _compute_fortress(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    minutes = _safe_get(df, "MIN").replace(0, np.nan)

    rim_fgpct = _safe_get(df, "TRK_DEF_RIM_FG_PCT").replace(0, np.nan).fillna(0.6)
    df["f_rim_inv"] = (1.0 - rim_fgpct).clip(lower=0)

    def_boxouts = _safe_get(df, "DEF_BOXOUTS")
    dreb        = _safe_get(df, "DREB")
    df["f_reb_contest"] = def_boxouts / (dreb + def_boxouts + 1e-6)

    df["f_blocks"]   = _safe_get(df, "BLK")
    df["f_rim_rate"] = (_safe_get(df, "TRK_DEF_RIM_FGA") / minutes * 36).fillna(0)
    df["f_putback"]  = _safe_get(df, "OREB") * 0.7

    df["f_rim_inv_n"]  = _norm(df["f_rim_inv"])
    df["f_reb_n"]      = _norm(df["f_reb_contest"])
    df["f_blocks_n"]   = _norm(df["f_blocks"])
    df["f_rim_rate_n"] = _norm(df["f_rim_rate"])
    df["f_putback_n"]  = _norm(df["f_putback"])

    paint_weight = 0.6 + 0.4 * _norm(_safe_get(df, "BLK") + _safe_get(df, "DREB"))

    raw = (
        FORTRESS_WEIGHTS["rim_fg_pct_inv"]      * df["f_rim_inv_n"]  +
        FORTRESS_WEIGHTS["contested_reb_rate"]   * df["f_reb_n"]      +
        FORTRESS_WEIGHTS["blocks_pg"]            * df["f_blocks_n"]   +
        FORTRESS_WEIGHTS["rim_contest_rate"]     * df["f_rim_rate_n"] +
        FORTRESS_WEIGHTS["putback_contribution"] * df["f_putback_n"]
    ) * paint_weight

    df["FORTRESS_RATING"] = (_norm(raw) * 100).round(1)
    return df


# ── Position classification ───────────────────────────────────────────────────

def _classify_position(pos: str) -> str:
    if not isinstance(pos, str) or not pos.strip():
        return "Unknown"
    pos = pos.upper()
    if any(p in pos for p in ["PG", "SG", "G-F"]):
        return "Guard"
    if "G" in pos and "F" not in pos:
        return "Guard"
    if any(p in pos for p in ["SF", "F-C"]):
        return "Wing"
    if any(p in pos for p in ["PF", "C-F", "F-G"]):
        return "Big"
    if "F" in pos:
        return "Wing"
    if "C" in pos:
        return "Big"
    return "Unknown"


# ── Rolling trend per player ──────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def get_player_rolling_trend(player_id: int, metric: str = "DRAGON_INDEX") -> pd.DataFrame:
    """
    Approximate per-game Dragon or Fortress score from box-score game log.
    `metric` accepts "DRAGON_INDEX" or "FORTRESS_RATING".
    Returns DataFrame: GAME_DATE, game_num, score, MATCHUP (last 15 games).
    """
    try:
        from nba_api.stats.endpoints import PlayerGameLog
        time.sleep(RATE_DELAY)
        gl = PlayerGameLog(player_id=player_id, season=SEASON).get_data_frames()[0]
        if gl.empty:
            return pd.DataFrame()

        gl = gl.sort_values("GAME_DATE").reset_index(drop=True)
        gl["game_num"] = range(1, len(gl) + 1)

        stl  = pd.to_numeric(gl.get("STL",  0), errors="coerce").fillna(0)
        blk  = pd.to_numeric(gl.get("BLK",  0), errors="coerce").fillna(0)
        dreb = pd.to_numeric(gl.get("DREB", 0), errors="coerce").fillna(0)
        oreb = pd.to_numeric(gl.get("OREB", 0), errors="coerce").fillna(0)

        use_dragon = metric.upper() in ("DRAGON_INDEX", "DRAGON")
        if use_dragon:
            raw = stl * 2.5 + blk * 0.8
        else:
            raw = blk * 2.0 + dreb * 0.8 + oreb * 1.0

        r = raw.max() - raw.min()
        gl["score"] = ((raw - raw.min()) / (r if r > 1e-9 else 1.0) * 100).round(1)

        cols = [c for c in ["GAME_DATE", "game_num", "score", "MATCHUP"] if c in gl.columns]
        return gl[cols].tail(15)
    except Exception as e:
        log.warning(f"Rolling trend error for {player_id}: {e}")
        return pd.DataFrame()


# ── Play-sequence analysis (steal → points chain) ────────────────────────────

def _clock_to_seconds(clock_str: str) -> float:
    """Convert ISO-8601 period clock 'PT10M35.00S' → remaining seconds."""
    try:
        s = str(clock_str).replace("PT", "").replace("S", "")
        if "M" in s:
            m, sec = s.split("M")
            return float(m) * 60 + float(sec)
        return float(s)
    except Exception:
        return 0.0


def _trace_steal_chains(pbp: pd.DataFrame, player_id: int, team_id) -> list[dict]:
    """
    Scan a single-game PBP DataFrame for steal events by player_id.
    For each steal, trace forward up to 8 plays and record:
      - whether the team scored
      - how many points
      - plays elapsed until score
      - whether it was a fast break (scored within 2 plays)
    """
    results = []
    steal_rows = pbp[
        pbp["description"].str.contains("STEAL", na=False) &
        (pbp["personId"].astype(str) == str(player_id))
    ]

    for _, steal_row in steal_rows.iterrows():
        steal_action = steal_row["actionNumber"]
        future = pbp[pbp["actionNumber"] > steal_action].head(12)

        pts           = 0
        plays_to_score = 0
        is_fast_break  = False
        scored         = False

        for i, (_, play) in enumerate(future.iterrows()):
            atype = play.get("actionType", "")

            # Period/game end → stop
            if atype == "period":
                break

            # Turnover or out-of-bounds by OUR team → possession lost, stop
            if (atype == "Turnover" and
                    str(play.get("teamId", "")) == str(team_id)):
                break

            # Made shot by OUR team → score, record and stop
            if (atype == "Made Shot" and
                    str(play.get("teamId", "")) == str(team_id)):
                pts            = int(play.get("shotValue") or 2)
                plays_to_score = i + 1
                is_fast_break  = (plays_to_score <= 2)
                scored         = True
                break

            # Free throw made by OUR team
            if (atype == "Free Throw" and
                    str(play.get("teamId", "")) == str(team_id)):
                desc = str(play.get("description", ""))
                if "MISS" not in desc.upper():
                    pts           += 1
                    plays_to_score = i + 1
                # Keep scanning for additional FTs; end after FT sequence
                if "1 of 1" in desc or "2 of 2" in desc or "3 of 3" in desc:
                    scored = pts > 0
                    is_fast_break = (plays_to_score <= 3)
                    break

        results.append({
            "scored":         scored,
            "pts":            pts,
            "plays_to_score": plays_to_score,
            "is_fast_break":  is_fast_break,
        })

    return results


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def get_play_sequence_stats(player_id: int, n_games: int = 10) -> dict:
    """
    For the player's last n_games, fetch play-by-play and trace every steal
    to measure points generated in the ensuing possession.

    Returns a dict with aggregate stats and a per-chain list for visualisation.
    """
    from nba_api.stats.endpoints import PlayerGameLog, PlayByPlayV3

    empty: dict = {
        "games_analyzed":    0,
        "total_steals":      0,
        "steals_with_score": 0,
        "total_pts":         0,
        "fast_break_steals": 0,
        "pts_per_steal":     0.0,
        "conversion_pct":    0.0,
        "fast_break_pct":    0.0,
        "chains":            [],   # list[dict] for per-steal details
    }

    try:
        time.sleep(RATE_DELAY)
        gl = PlayerGameLog(player_id=player_id, season=SEASON).get_data_frames()[0]
        if gl.empty:
            return empty

        recent       = gl.head(n_games)
        result       = {**empty}
        all_chains   = []
        player_tid   = None  # learned from first steal found

        for _, game_row in recent.iterrows():
            game_id = str(game_row.get("Game_ID", ""))
            if not game_id:
                continue

            time.sleep(RATE_DELAY)
            try:
                pbp = PlayByPlayV3(game_id=game_id).get_data_frames()[0]
            except Exception:
                continue

            if pbp.empty:
                continue

            result["games_analyzed"] += 1

            # Determine player's team for this game
            my_steals = pbp[
                pbp["description"].str.contains("STEAL", na=False) &
                (pbp["personId"].astype(str) == str(player_id))
            ]
            if my_steals.empty:
                continue

            team_id = my_steals.iloc[0]["teamId"]
            if player_tid is None:
                player_tid = team_id

            chains = _trace_steal_chains(pbp, player_id, team_id)
            for c in chains:
                c["game_id"] = game_id
            all_chains.extend(chains)

        if not all_chains:
            return result

        result["total_steals"]      = len(all_chains)
        result["steals_with_score"] = sum(c["scored"]        for c in all_chains)
        result["total_pts"]         = sum(c["pts"]           for c in all_chains)
        result["fast_break_steals"] = sum(c["is_fast_break"] for c in all_chains)

        n = result["total_steals"]
        result["pts_per_steal"]  = round(result["total_pts"] / n, 2)
        result["conversion_pct"] = round(result["steals_with_score"] / n * 100, 1)
        result["fast_break_pct"] = round(result["fast_break_steals"]  / n * 100, 1)
        result["chains"]         = all_chains
        return result

    except Exception as e:
        log.warning(f"get_play_sequence_stats error for {player_id}: {e}")
        return empty


# ── Master fetch (cached to disk daily) ──────────────────────────────────────

def _fetch_all_fresh() -> pd.DataFrame:
    """Fetch all endpoints, merge, compute metrics. Shows step progress in Streamlit."""
    ph = st.empty()

    def step(msg: str):
        ph.caption(f"⏳ {msg}")

    step("Fetching base stats…")
    base = _fetch_base()

    step("Fetching advanced stats…")
    advanced = _fetch_advanced()

    step("Fetching hustle stats…")
    hustle = _fetch_hustle()

    step("Fetching defense tracking…")
    defense = _fetch_defense_tracking()

    step("Fetching player bios…")
    bio = _fetch_bio()

    step("Fetching opponent on-court stats…")
    opp_stats = _fetch_opponent_stats()

    step("Merging datasets…")
    df = _merge_all(base, advanced, hustle, defense, bio, opp_stats=opp_stats)

    if df.empty:
        ph.empty()
        return df

    step("Computing Dragon Index…")
    df = _compute_dragon(df)

    step("Computing Fortress Rating…")
    df = _compute_fortress(df)

    # Position
    if "PLAYER_POSITION" in df.columns:
        df["POSITION"] = df["PLAYER_POSITION"].apply(_classify_position)
    else:
        df["POSITION"] = "Unknown"

    # Derived columns
    df["TEAM_COLOR"]    = df["TEAM_ID"].apply(team_color)
    df["HEADSHOT_URL"]  = df["PLAYER_ID"].apply(nba_headshot_url)
    df["LOGO_URL"]      = df["TEAM_ID"].apply(nba_logo_url)
    df["COMBINED_SCORE"] = ((df["DRAGON_INDEX"] + df["FORTRESS_RATING"]) / 2).round(1)

    # Minimum games filter
    df = df[df["GP"] >= 10].reset_index(drop=True)

    ph.empty()
    return df


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def get_all_player_metrics() -> pd.DataFrame:
    """Load from disk cache if fresh, otherwise re-fetch. TTL: 24 h."""
    if _is_cache_fresh("player_metrics"):
        try:
            df = _load("player_metrics")
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Validate it has the expected columns (cache may be from old format)
                if "DRAGON_INDEX" in df.columns and "FORTRESS_RATING" in df.columns:
                    return df
        except Exception:
            pass

    df = _fetch_all_fresh()
    if not df.empty:
        _save("player_metrics", df)
    return df


# ── Team aggregate view ───────────────────────────────────────────────────────

def get_team_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate DRAGON_INDEX and FORTRESS_RATING to team level."""
    if df.empty or "DRAGON_INDEX" not in df.columns:
        return pd.DataFrame()

    static_teams = nba_teams_static.get_teams()
    id_to_name = {t["id"]: t["full_name"]   for t in static_teams}
    id_to_abbr = {t["id"]: t["abbreviation"] for t in static_teams}

    agg = df.groupby("TEAM_ID").agg(
        DRAGON_INDEX   =("DRAGON_INDEX",   "mean"),
        FORTRESS_RATING=("FORTRESS_RATING", "mean"),
        COMBINED_SCORE =("COMBINED_SCORE",  "mean"),
        AVG_MIN        =("MIN",             "mean"),
        PLAYER_COUNT   =("PLAYER_ID",       "count"),
    ).reset_index()

    agg["TEAM_NAME"]         = agg["TEAM_ID"].map(id_to_name).fillna("Unknown")
    agg["TEAM_ABBREVIATION"] = agg["TEAM_ID"].map(id_to_abbr).fillna("?")
    agg["TEAM_COLOR"]        = agg["TEAM_ID"].apply(team_color)
    agg["LOGO_URL"]          = agg["TEAM_ID"].apply(nba_logo_url)

    return agg.round(1)
