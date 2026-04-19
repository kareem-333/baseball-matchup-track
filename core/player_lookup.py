from __future__ import annotations

import functools

import statsapi

# ── internal cache (avoid repeated StatsAPI round-trips) ─────────────────────

@functools.lru_cache(maxsize=512)
def get_player_info(player_id: int) -> dict:
    """
    Full player record from MLB StatsAPI.
    Returns a flat dict with name, position, team, throws, bats, headshot_url.
    Result is process-cached so repeated calls are free.
    """
    try:
        data = statsapi.get(
            "person",
            {"personId": player_id, "hydrate": "currentTeam"},
        )
        p = data.get("people", [{}])[0]
        team = p.get("currentTeam", {})
        return {
            "player_id": player_id,
            "full_name": p.get("fullName", str(player_id)),
            "first_name": p.get("firstName", ""),
            "last_name": p.get("lastName", ""),
            "throws": p.get("pitchHand", {}).get("code", "R"),
            "bats": p.get("batSide", {}).get("code", "R"),
            "position": p.get("primaryPosition", {}).get("abbreviation", ""),
            "team_id": team.get("id"),
            "team_name": team.get("name", ""),
            "team_abbr": team.get("abbreviation", ""),
            "jersey_number": p.get("primaryNumber", ""),
            "headshot_url": (
                f"https://img.mlbstatic.com/mlb-photos/image/upload/"
                f"d_people:generic:headshot:67:current.png/w_213,q_auto:best/"
                f"v1/people/{player_id}/headshot/67/current"
            ),
        }
    except Exception:
        return {
            "player_id": player_id,
            "full_name": str(player_id),
            "first_name": "",
            "last_name": str(player_id),
            "throws": "R",
            "bats": "R",
            "position": "",
            "team_id": None,
            "team_name": "",
            "team_abbr": "",
            "jersey_number": "",
            "headshot_url": (
                "https://img.mlbstatic.com/mlb-photos/image/upload/"
                "d_people:generic:headshot:67:current.png/w_213,q_auto:best/"
                "v1/people/0/headshot/67/current"
            ),
        }


def get_player_name(player_id: int) -> str:
    return get_player_info(player_id)["full_name"]


def get_pitcher_handedness(pitcher_id: int) -> str:
    """Returns 'L' or 'R'."""
    return get_player_info(pitcher_id).get("throws", "R")


def get_batter_handedness(batter_id: int) -> str:
    """Returns 'L', 'R', or 'S' (switch)."""
    return get_player_info(batter_id).get("bats", "R")


@functools.lru_cache(maxsize=256)
def lookup_player_id(first: str, last: str) -> int | None:
    """
    Look up MLBAM player_id by first + last name via StatsAPI search.
    Returns None when no match is found.
    """
    try:
        results = statsapi.lookup_player(f"{first} {last}")
        if not results:
            return None
        # Prefer active players; otherwise take the first result
        active = [r for r in results if r.get("active")]
        chosen = active[0] if active else results[0]
        return chosen["id"]
    except Exception:
        return None


@functools.lru_cache(maxsize=256)
def search_players(query: str) -> list[dict]:
    """
    Free-text player search. Returns list of {player_id, full_name, position, team}.
    Useful for the dashboard's search box.
    """
    try:
        results = statsapi.lookup_player(query)
        out = []
        for r in results:
            out.append({
                "player_id": r["id"],
                "full_name": r.get("fullName", ""),
                "position": r.get("primaryPosition", {}).get("abbreviation", ""),
                "team": r.get("currentTeam", {}).get("name", ""),
                "active": r.get("active", False),
            })
        return sorted(out, key=lambda x: (not x["active"], x["full_name"]))
    except Exception:
        return []


@functools.lru_cache(maxsize=64)
def get_team_roster(team_id: int, roster_type: str = "active") -> list[dict]:
    """
    Returns active roster for a team as list of {player_id, full_name, position}.
    """
    try:
        data = statsapi.get(
            "roster",
            {"teamId": team_id, "rosterType": roster_type},
        )
        players = data.get("roster", [])
        return [
            {
                "player_id": p["person"]["id"],
                "full_name": p["person"]["fullName"],
                "position": p.get("position", {}).get("abbreviation", ""),
                "jersey_number": p.get("jerseyNumber", ""),
            }
            for p in players
        ]
    except Exception:
        return []


# ── known player IDs (used by test_mas.py) ───────────────────────────────────

KNOWN_IDS = {
    # Batters
    "Aaron Judge": 592450,
    "Shohei Ohtani": 660271,
    "Luis Arraez": 650333,
    "Joey Gallo": 608566,
    "Yordan Alvarez": 670541,
    "Jose Altuve": 514888,
    "Freddie Freeman": 518692,
    "Manny Machado": 592518,
    # Pitchers
    "Justin Verlander": 434378,
    "Gerrit Cole": 543037,
    "Spencer Strider": 675911,
    "Zack Wheeler": 554430,
    "Sandy Alcantara": 645261,
    "Max Scherzer": 453286,
}
