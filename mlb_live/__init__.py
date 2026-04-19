"""
mlb_live — Live game data pipeline and real-time chart builders.

Refreshes every time it's called (no long-term caching here).
Callers should apply st.cache_data(ttl=30) for the live endpoints.
"""

from mlb_live.pipeline import (
    get_todays_game,
    get_game_summary,
    get_live_box_score,
    get_linescore,
    get_current_play,
    build_inning_table,
    build_batting_table,
    build_pitching_table,
    get_all_live_pitches,
    get_pitcher_live_pitches,
    compute_pitcher_fatigue,
    get_win_probability_from_plays,
    get_game_pitchers,
    get_active_pitcher,
)
from mlb_live.charts import (
    plot_pitch_movement,
    plot_velocity_fade,
    plot_fatigue_gauge,
    plot_win_probability,
)

__all__ = [
    "get_todays_game", "get_game_summary", "get_live_box_score",
    "get_linescore", "get_current_play",
    "build_inning_table", "build_batting_table", "build_pitching_table",
    "get_all_live_pitches", "get_pitcher_live_pitches",
    "compute_pitcher_fatigue", "get_win_probability_from_plays",
    "get_game_pitchers", "get_active_pitcher",
    "plot_pitch_movement", "plot_velocity_fade",
    "plot_fatigue_gauge", "plot_win_probability",
]
