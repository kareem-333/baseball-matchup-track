# Layer 4: Arsenal+ — pitch mix quality vs league norms, weighted by Stuff+.
# Normalized to 100 = league average; higher = better.
#
# Formula: arsenal_plus = sum(stuff_plus[p] * usage[p]) / league_avg_weighted * 100
#
# Requires ≥200 total pitches. Individual pitches below 5% usage are excluded.

from __future__ import annotations

import pandas as pd

_PITCH_USAGE_FLOOR = 0.05
_MIN_TOTAL_PITCHES = 200

# Fallback league-average weighted Stuff+ contribution per pitch type (usage * 100).
# Replace with actual league baselines once computed from real data.
_FALLBACK_LEAGUE_AVG_WEIGHTED = 100.0


def compute_arsenal_plus(
    arsenal_df: pd.DataFrame,
    stuff_scores: dict[str, float],
    league_avg_weighted: float = _FALLBACK_LEAGUE_AVG_WEIGHTED,
) -> dict:
    """
    Compute Arsenal+ from a pitcher's parsed arsenal and per-pitch Stuff+ scores.

    Args:
        arsenal_df:          Output of transforms.pitchers.parse_pitcher_arsenal().
                             Must have columns: pitch_type, usage_pct, count.
        stuff_scores:        {pitch_type: stuff_plus_score} from metrics.stuff.compute_stuff_plus()
                             Use the "per_pitch" dict. Missing types are skipped.
        league_avg_weighted: Sum(league_usage[p] * 100) across all pitch types for average pitcher.
                             Default 100 until computed from real league data.

    Returns: {
        "overall":      float,   # Arsenal+ (100 = league average)
        "qualified":    list,    # pitch types meeting usage floor and scored
        "n_total":      int,     # total pitches
        "is_qualified": bool,    # True if ≥200 total pitches
        "breakdown":    DataFrame,  # per-pitch contribution
    }
    """
    if arsenal_df.empty or "pitch_type" not in arsenal_df.columns:
        return {
            "overall": 100.0, "qualified": [], "n_total": 0,
            "is_qualified": False, "breakdown": pd.DataFrame(),
        }

    n_total = int(arsenal_df["count"].sum()) if "count" in arsenal_df.columns else len(arsenal_df)
    is_qualified = n_total >= _MIN_TOTAL_PITCHES

    qualified = arsenal_df[arsenal_df["usage_pct"] >= _PITCH_USAGE_FLOOR].copy()

    rows = []
    for _, row in qualified.iterrows():
        pt = row["pitch_type"]
        if pt not in stuff_scores:
            continue
        usage = float(row["usage_pct"])
        stuff = float(stuff_scores[pt])
        rows.append({
            "pitch_type":        pt,
            "usage_pct":         usage,
            "stuff_plus":        stuff,
            "weighted_contrib":  stuff * usage,
        })

    if not rows:
        return {
            "overall": 100.0, "qualified": [], "n_total": n_total,
            "is_qualified": is_qualified, "breakdown": pd.DataFrame(),
        }

    breakdown = pd.DataFrame(rows)
    total_weight = breakdown["usage_pct"].sum()
    weighted_sum = breakdown["weighted_contrib"].sum()

    if total_weight == 0 or league_avg_weighted == 0:
        overall = 100.0
    else:
        # Normalize so that a pitcher with all-100 Stuff+ pitches at league-average
        # usage distribution scores exactly 100
        overall = (weighted_sum / total_weight) / (league_avg_weighted / 100.0)

    return {
        "overall":      round(overall, 1),
        "qualified":    breakdown["pitch_type"].tolist(),
        "n_total":      n_total,
        "is_qualified": is_qualified,
        "breakdown":    breakdown,
    }
