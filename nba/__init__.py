"""
nba — NBA Defensive Analytics pipeline and charts.

Dragon Index: active perimeter disruption metric.
Fortress Rating: interior anchor / paint protection metric.

All data fetched via nba_api. Disk-cached 24 h.
"""

from nba.pipeline import (
    get_all_player_metrics,
    get_team_aggregates,
    get_player_rolling_trend,
    get_play_sequence_stats,
    team_color,
    nba_logo_url,
    nba_headshot_url,
    DRAGON_WEIGHTS,
    FORTRESS_WEIGHTS,
    SEASON,
)
from nba.charts import (
    plot_bubble_scatter,
    plot_leaderboard,
    plot_comparison_radar,
    plot_comparison_rolling,
    plot_team_bubbles,
    plot_steal_chain_sankey,
    plot_sequence_comparison,
)

__all__ = [
    "get_all_player_metrics", "get_team_aggregates",
    "get_player_rolling_trend", "get_play_sequence_stats",
    "team_color", "nba_logo_url", "nba_headshot_url",
    "DRAGON_WEIGHTS", "FORTRESS_WEIGHTS", "SEASON",
    "plot_bubble_scatter", "plot_leaderboard",
    "plot_comparison_radar", "plot_comparison_rolling",
    "plot_team_bubbles", "plot_steal_chain_sankey",
    "plot_sequence_comparison",
]
