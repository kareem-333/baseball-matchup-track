"""
mlb_season — Season-level MLB stats pipeline and chart builders.

Covers: Statcast CSV pulls, pitch arsenal, batter splits, hot zones,
barrel trends, OPS/K-rate game logs, lineup prediction, team rosters,
pitching staff rankings, batting leaders, and league averages.
"""

from mlb_season.pipeline import (
    fetch_statcast_csv,
    get_pitcher_arsenal,
    get_batter_pitch_splits,
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
from mlb_season.charts import (
    plot_matchup_heatmap,
    plot_hot_zone_grid,
    plot_rolling_ops,
    plot_krate_chart,
    plot_barrel_trend,
    show_pitch_mix_simulator,
)

__all__ = [
    "fetch_statcast_csv", "get_pitcher_arsenal", "get_batter_pitch_splits",
    "get_batter_hot_zones", "get_barrel_trend",
    "get_batter_game_log", "get_pitcher_game_log",
    "get_last_n_completed_games", "predict_lineup",
    "aggregate_batting_stats", "aggregate_pitching_stats",
    "get_team_pitching_staff", "get_team_batting_leaders",
    "get_league_avg_krate", "get_lineup_with_ids",
    "plot_matchup_heatmap", "plot_hot_zone_grid",
    "plot_rolling_ops", "plot_krate_chart",
    "plot_barrel_trend", "show_pitch_mix_simulator",
]
