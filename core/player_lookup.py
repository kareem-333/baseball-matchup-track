"""
core/player_lookup.py — Player ID resolution and bio lookups.

Handles both MLB (statsapi) and NBA (nba_api) player lookups.
Results are not cached here; call-sites should apply st.cache_data
or functools.lru_cache as appropriate.
"""

from __future__ import annotations
import statsapi


def lookup_player_id(name: str) -> int | None:
    """Return the MLB Stats API player ID for a given name string."""
    try:
        results = statsapi.lookup_player(name)
        return results[0]["id"] if results else None
    except Exception:
        return None


def get_pitcher_handedness(player_id: int) -> str:
    """
    Return 'R', 'L', or 'S' for the pitcher's throwing hand.
    Fetches from the MLB people endpoint — not a manual input.
    Returns '?' if the API call fails.
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


def get_batter_hand(player_id: int) -> str:
    """
    Return 'R', 'L', or 'S' for the batter's hitting hand.
    Returns '?' if the API call fails.
    """
    try:
        data = statsapi.get("people", {"personIds": player_id})
        for person in data.get("people", []):
            code = person.get("batSide", {}).get("code", "")
            if code:
                return code
    except Exception:
        pass
    return "?"
