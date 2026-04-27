"""
mlb_season — Season-level MLB stats pipeline.

Covers: Statcast CSV pulls, pitch arsenal, batter splits, hot zones,
barrel trends, OPS/K-rate game logs, lineup prediction, team rosters,
pitching staff rankings, batting leaders, league averages, and
multi-season career pitch-type split analysis.

Charts have moved to dashboard/components/season_charts.py.
"""

from mlb_season.pipeline import (
    fetch_statcast_csv,
    fetch_statcast_multi_season,
    get_pitcher_arsenal,
    get_batter_pitch_splits,
    get_batter_career_pitch_splits,
    get_batter_hot_zones,
    get_barrel_trend,
    get_batter_game_log,
    get_pitcher_game_log,
    get_last_n_completed_games,
    predict_lineup,
    aggregate_batting_stats,
    aggregate_pitching_stats,
    get_team_pitching_staff,
    get_team_batting_leaders,
    get_league_avg_krate,
    get_lineup_with_ids,
)

__all__ = [
    "fetch_statcast_csv", "fetch_statcast_multi_season",
    "get_pitcher_arsenal", "get_batter_pitch_splits",
    "get_batter_career_pitch_splits",
    "get_batter_hot_zones", "get_barrel_trend",
    "get_batter_game_log", "get_pitcher_game_log",
    "get_last_n_completed_games", "predict_lineup",
    "aggregate_batting_stats", "aggregate_pitching_stats",
    "get_team_pitching_staff", "get_team_batting_leaders",
    "get_league_avg_krate", "get_lineup_with_ids",
]
