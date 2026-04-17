"""
core — shared utilities, constants, and helpers.

Imported by every other package.  Nothing in core/ imports from
mlb_live/, mlb_season/, or nba/ to avoid circular dependencies.
"""

from core.config import (
    ASTROS_ID,
    PITCH_NAMES, PITCH_COLORS,
    STAR_BATTERS,
    SZ_L, SZ_R, SZ_B, SZ_T,
    ZONE_BOXES, ZONE_CENTERS,
    MLB_TEAM_COLORS, NBA_TEAM_COLORS,
)
from core.headshots import headshot_url, headshot_b64
from core.player_lookup import lookup_player_id, get_pitcher_handedness

__all__ = [
    "ASTROS_ID", "PITCH_NAMES", "PITCH_COLORS", "STAR_BATTERS",
    "SZ_L", "SZ_R", "SZ_B", "SZ_T", "ZONE_BOXES", "ZONE_CENTERS",
    "MLB_TEAM_COLORS", "NBA_TEAM_COLORS",
    "headshot_url", "headshot_b64",
    "lookup_player_id", "get_pitcher_handedness",
]
