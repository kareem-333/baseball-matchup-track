# Layer 1: MLB Stats API source.
# Wraps every external network call the app makes: MLB Stats API (via statsapi)
# and Baseball Savant CSV exports (via requests).
# Each method: check cache → if stale/missing, fetch from API → write to cache → return.
# No transformation, no field renaming, no metric math.

from __future__ import annotations

import json
from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests
import statsapi

from sources.base import Source
from storage.base import Storage

# ── TTL constants ─────────────────────────────────────────────────────────────

_TTL_LIVE = timedelta(seconds=60)       # schedule and live game data
_TTL_STATCAST = timedelta(hours=24)     # season pitch data
_TTL_PLAYER = timedelta(days=7)         # player metadata (handedness, position)
_TTL_TEAMS = timedelta(hours=24)

_SAVANT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    )
}

_NUMERIC_COLS = [
    "plate_x", "plate_z", "pfx_x", "pfx_z", "release_speed", "release_spin_rate",
    "estimated_ba_using_speedangle", "launch_speed", "launch_angle",
    "api_break_x_arm", "api_break_z_with_gravity", "delta_home_win_exp",
    "home_win_exp", "bat_speed", "swing_length",
]
_INT_COLS = ["zone", "launch_speed_angle"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_parquet(data) -> pd.DataFrame:
    """Serialize a JSON-serializable value (dict or list) into a single-row DataFrame."""
    return pd.DataFrame({"_json": [json.dumps(data)]})


def _from_parquet(df: pd.DataFrame):
    """Deserialize the JSON stored by _to_parquet."""
    return json.loads(df["_json"].iloc[0])


def _is_fresh(storage: Storage, key: str, ttl: timedelta | None) -> bool:
    """Return True if the cached value exists and has not exceeded TTL."""
    if not storage.exists(key):
        return False
    if ttl is None:
        return True  # never expires
    age = storage.age(key)
    return age is not None and age <= ttl


# ── source class ──────────────────────────────────────────────────────────────

class MLBStatsSource(Source):
    """
    Wraps all MLB Stats API and Baseball Savant calls.
    Injects a Storage instance for transparent caching with TTL.
    """

    def __init__(self, storage: Storage):
        self._storage = storage

    @property
    def name(self) -> str:
        return "mlb_stats"

    def fetch(self, endpoint: str, params: dict):
        """Low-level pass-through to statsapi.get(). Callers use typed methods."""
        self.logger.debug("statsapi.get(%s, %s)", endpoint, params)
        return statsapi.get(endpoint, params)

    # ── Schedule ──────────────────────────────────────────────────────────────

    def get_schedule(self, game_date: str | date) -> list[dict]:
        """Return all MLB games for a given date (YYYY-MM-DD or date object)."""
        date_str = game_date.strftime("%Y-%m-%d") if isinstance(game_date, date) else game_date
        key = f"{self.name}/schedule/{date_str}"
        if _is_fresh(self._storage, key, _TTL_LIVE):
            return _from_parquet(self._storage.read(key))
        self.logger.info("fetch schedule %s", date_str)
        data = statsapi.schedule(sportId=1, start_date=date_str, end_date=date_str)
        self._storage.write(key, _to_parquet(data))
        return data

    def get_team_schedule(self, team_id: int, game_date: str | date) -> list[dict]:
        """Return games for a single team on a given date."""
        date_str = game_date.strftime("%Y-%m-%d") if isinstance(game_date, date) else game_date
        key = f"{self.name}/schedule/team/{team_id}/{date_str}"
        if _is_fresh(self._storage, key, _TTL_LIVE):
            return _from_parquet(self._storage.read(key))
        self.logger.info("fetch team schedule team=%s date=%s", team_id, date_str)
        data = statsapi.schedule(sportId=1, team=team_id, start_date=date_str, end_date=date_str)
        self._storage.write(key, _to_parquet(data))
        return data

    def get_schedule_range(self, team_id: int, start: str, end: str) -> list[dict]:
        """Return games for a team over a date range."""
        key = f"{self.name}/schedule/range/{team_id}/{start}/{end}"
        if _is_fresh(self._storage, key, _TTL_LIVE):
            return _from_parquet(self._storage.read(key))
        self.logger.info("fetch schedule range team=%s %s to %s", team_id, start, end)
        data = statsapi.schedule(sportId=1, team=team_id, start_date=start, end_date=end)
        self._storage.write(key, _to_parquet(data))
        return data

    # ── Live feed / boxscore ─────────────────────────────────────────────────

    def get_live_feed(self, game_pk: int) -> dict:
        """Return full live game feed (liveData + gameData) from MLB Stats API."""
        key = f"{self.name}/live_feed/{game_pk}"
        if _is_fresh(self._storage, key, _TTL_LIVE):
            return _from_parquet(self._storage.read(key))
        self.logger.info("fetch live feed game_pk=%s", game_pk)
        data = statsapi.get("game", {"gamePk": game_pk})
        self._storage.write(key, _to_parquet(data))
        return data

    def get_boxscore(self, game_pk: int, is_final: bool = False) -> dict:
        """Return boxscore data. Final games are cached indefinitely."""
        ttl = None if is_final else _TTL_LIVE
        key = f"{self.name}/boxscore/{game_pk}"
        if _is_fresh(self._storage, key, ttl):
            return _from_parquet(self._storage.read(key))
        self.logger.info("fetch boxscore game_pk=%s final=%s", game_pk, is_final)
        data = statsapi.boxscore_data(game_pk)
        self._storage.write(key, _to_parquet(data))
        return data

    # ── Teams ────────────────────────────────────────────────────────────────

    def get_teams(self) -> list[dict]:
        """Return all active MLB teams."""
        key = f"{self.name}/teams"
        if _is_fresh(self._storage, key, _TTL_TEAMS):
            return _from_parquet(self._storage.read(key))
        self.logger.info("fetch teams")
        data = statsapi.get("teams", {"sportId": 1, "activeStatus": "Y"})
        self._storage.write(key, _to_parquet(data))
        return data

    def get_roster(self, team_id: int, roster_type: str = "active") -> dict:
        """Return team roster."""
        key = f"{self.name}/roster/{team_id}/{roster_type}"
        if _is_fresh(self._storage, key, _TTL_TEAMS):
            return _from_parquet(self._storage.read(key))
        self.logger.info("fetch roster team=%s type=%s", team_id, roster_type)
        data = statsapi.get("roster", {"teamId": team_id, "rosterType": roster_type})
        self._storage.write(key, _to_parquet(data))
        return data

    # ── Player metadata ───────────────────────────────────────────────────────

    def get_person(self, player_id: int, hydrate: str | None = None) -> dict:
        """Return player metadata (handedness, position, team). Cached 7 days."""
        key = f"{self.name}/person/{player_id}"
        if _is_fresh(self._storage, key, _TTL_PLAYER):
            return _from_parquet(self._storage.read(key))
        self.logger.info("fetch person player_id=%s", player_id)
        params: dict = {"personId": player_id}
        if hydrate:
            params["hydrate"] = hydrate
        data = statsapi.get("person", params)
        self._storage.write(key, _to_parquet(data))
        return data

    def lookup_player(self, query: str) -> list[dict]:
        """Search players by name. Not cached (lightweight, real-time search)."""
        self.logger.info("lookup_player query=%r", query)
        return statsapi.lookup_player(query)

    def get_stats(self, params: dict) -> dict:
        """Return league or team stats. Cached 24h."""
        # Build a stable key from sorted params
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        key = f"{self.name}/stats/{param_str}"
        if _is_fresh(self._storage, key, _TTL_STATCAST):
            return _from_parquet(self._storage.read(key))
        self.logger.info("fetch stats params=%s", params)
        data = statsapi.get("stats", params)
        self._storage.write(key, _to_parquet(data))
        return data

    # ── Statcast (Baseball Savant CSV) ────────────────────────────────────────

    def get_pitcher_stats(self, player_id: int, season: int) -> pd.DataFrame:
        """Return one season of Statcast pitch-level data for a pitcher."""
        return self._get_statcast_csv(player_id, "pitcher", season)

    def get_batter_stats(self, player_id: int, season: int) -> pd.DataFrame:
        """Return one season of Statcast pitch-level data for a batter."""
        return self._get_statcast_csv(player_id, "batter", season)

    def get_pitch_data(self, player_id: int, season: int, player_type: str = "pitcher") -> pd.DataFrame:
        """Generic Statcast CSV fetch keyed by player_id, season, and player_type."""
        return self._get_statcast_csv(player_id, player_type, season)

    def _get_statcast_csv(self, player_id: int, player_type: str, season: int) -> pd.DataFrame:
        key = f"{self.name}/statcast/{player_type}/{player_id}/{season}"
        if _is_fresh(self._storage, key, _TTL_STATCAST):
            return self._storage.read(key)
        self.logger.info("fetch statcast %s player_id=%s season=%s", player_type, player_id, season)
        df = self._fetch_savant_csv(player_id, player_type, season)
        if not df.empty:
            self._storage.write(key, df)
        return df

    def _fetch_savant_csv(self, player_id: int, player_type: str, season: int) -> pd.DataFrame:
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
            for col in _NUMERIC_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            for col in _INT_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "game_date" in df.columns:
                df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
            return df
        except Exception:
            self.logger.exception("Savant fetch failed player_id=%s season=%s", player_id, season)
            return pd.DataFrame()
