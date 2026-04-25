# Layer 4: MISS — Whiff Score (swing-and-miss probability).
# Pure function: takes already-transformed DataFrames, returns float 0-100.
# Higher = more swing-and-miss expected (worse for batter, better for pitcher).
#
# Same architecture as MASH but uses z_whiff × z_freq overlap instead of z_barrel.

from __future__ import annotations

import pandas as pd

from metrics.mash import sigmoid, _pitcher_z_scores, _batter_z_scores, _FALLBACK_BASELINES, _PITCH_USAGE_FLOOR


def compute_miss(
    pitcher_arsenal: pd.DataFrame,
    batter_profile: pd.DataFrame,
    *,
    pitcher_handedness: str,
    batter_handedness: str,
    league_baselines: dict | None = None,
) -> float:
    """
    Whiff Score — swing-and-miss probability for this matchup.

    Args:
        pitcher_arsenal:   Output of transforms.pitchers.parse_pitcher_arsenal()
        batter_profile:    Output of transforms.batters.parse_batter_pitch_splits()
        pitcher_handedness: 'R' or 'L'
        batter_handedness:  'R', 'L', or 'S'
        league_baselines:   dict of per-pitch-type mean/std

    Returns float 0-100. 50 = neutral. Higher = more whiffs expected.
    """
    baselines = league_baselines or _FALLBACK_BASELINES
    hand = pitcher_handedness

    pitcher_z = _pitcher_z_scores(pitcher_arsenal, baselines)
    batter_z = _batter_z_scores(batter_profile, hand, baselines)

    if pitcher_z.empty or batter_z.empty:
        return 50.0

    merged = pitcher_z.merge(batter_z, on="pitch_type", how="inner")
    if merged.empty:
        return 50.0

    merged["whiff_overlap"] = merged.apply(
        lambda r: sigmoid(r["z_whiff"] * r["z_freq"]), axis=1
    )

    total_weight = merged["usage_pct"].sum()
    if total_weight <= 0:
        return 50.0

    return round((merged["whiff_overlap"] * merged["usage_pct"]).sum() / total_weight * 100, 1)
