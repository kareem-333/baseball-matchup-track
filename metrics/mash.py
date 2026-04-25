# Layer 4: MASH — Matchup Advantage Score (contact quality).
# Pure function: takes already-transformed DataFrames, returns float 0-100.
# Higher = batter advantage on contact quality (barrel/xwOBA potential).
#
# Depends on: transforms.pitchers.parse_pitcher_arsenal()
#             transforms.batters.parse_batter_pitch_splits()

from __future__ import annotations

import math

import pandas as pd

_PITCH_USAGE_FLOOR = 0.05


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


_FALLBACK_BASELINES: dict[str, dict] = {
    "FF": {"barrel_rate_mean": 0.055, "barrel_rate_std": 0.040,
           "whiff_rate_mean": 0.220, "whiff_rate_std": 0.130,
           "usage_pct_mean": 0.340, "usage_pct_std": 0.120},
    "SI": {"barrel_rate_mean": 0.045, "barrel_rate_std": 0.038,
           "whiff_rate_mean": 0.180, "whiff_rate_std": 0.115,
           "usage_pct_mean": 0.220, "usage_pct_std": 0.100},
    "FC": {"barrel_rate_mean": 0.040, "barrel_rate_std": 0.035,
           "whiff_rate_mean": 0.230, "whiff_rate_std": 0.135,
           "usage_pct_mean": 0.130, "usage_pct_std": 0.080},
    "SL": {"barrel_rate_mean": 0.035, "barrel_rate_std": 0.030,
           "whiff_rate_mean": 0.330, "whiff_rate_std": 0.150,
           "usage_pct_mean": 0.190, "usage_pct_std": 0.100},
    "ST": {"barrel_rate_mean": 0.030, "barrel_rate_std": 0.028,
           "whiff_rate_mean": 0.360, "whiff_rate_std": 0.155,
           "usage_pct_mean": 0.110, "usage_pct_std": 0.075},
    "CU": {"barrel_rate_mean": 0.030, "barrel_rate_std": 0.028,
           "whiff_rate_mean": 0.310, "whiff_rate_std": 0.145,
           "usage_pct_mean": 0.130, "usage_pct_std": 0.090},
    "KC": {"barrel_rate_mean": 0.028, "barrel_rate_std": 0.026,
           "whiff_rate_mean": 0.290, "whiff_rate_std": 0.140,
           "usage_pct_mean": 0.080, "usage_pct_std": 0.060},
    "CH": {"barrel_rate_mean": 0.038, "barrel_rate_std": 0.032,
           "whiff_rate_mean": 0.330, "whiff_rate_std": 0.155,
           "usage_pct_mean": 0.140, "usage_pct_std": 0.090},
    "FS": {"barrel_rate_mean": 0.032, "barrel_rate_std": 0.030,
           "whiff_rate_mean": 0.350, "whiff_rate_std": 0.160,
           "usage_pct_mean": 0.060, "usage_pct_std": 0.050},
    "CS": {"barrel_rate_mean": 0.025, "barrel_rate_std": 0.022,
           "whiff_rate_mean": 0.280, "whiff_rate_std": 0.135,
           "usage_pct_mean": 0.030, "usage_pct_std": 0.025},
}


def _pitcher_z_scores(arsenal: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """Compute z_freq per qualified pitch type from pitcher arsenal."""
    if arsenal.empty or "usage_pct" not in arsenal.columns:
        return pd.DataFrame()
    qualified = arsenal[arsenal["usage_pct"] >= _PITCH_USAGE_FLOOR].copy()
    rows = []
    for _, r in qualified.iterrows():
        pt = r["pitch_type"]
        bl = baselines.get(pt)
        if not bl or bl.get("usage_pct_std", 0) <= 0:
            z_freq = 0.0
        else:
            z_freq = (r["usage_pct"] - bl["usage_pct_mean"]) / bl["usage_pct_std"]
        rows.append({
            "pitch_type":  pt,
            "pitch_label": r.get("pitch_label", pt),
            "usage_pct":   r["usage_pct"],
            "z_freq":      z_freq,
        })
    return pd.DataFrame(rows)


def _batter_z_scores(splits: pd.DataFrame, pitcher_hand: str, baselines: dict) -> pd.DataFrame:
    """Compute z_barrel and z_whiff for the relevant handedness split."""
    if splits.empty or "vs_hand" not in splits.columns:
        return pd.DataFrame()
    relevant = splits[splits["vs_hand"] == pitcher_hand].copy()
    rows = []
    for _, r in relevant.iterrows():
        pt = r["pitch_type"]
        bl = baselines.get(pt)
        if not bl:
            z_barrel = z_whiff = 0.0
        else:
            br_std = bl.get("barrel_rate_std", 0)
            wr_std = bl.get("whiff_rate_std", 0)
            z_barrel = (r["barrel_rate"] - bl["barrel_rate_mean"]) / br_std if br_std > 0 else 0.0
            z_whiff = (r["whiff_rate"] - bl["whiff_rate_mean"]) / wr_std if wr_std > 0 else 0.0
        rows.append({
            "pitch_type":  pt,
            "vs_hand":     pitcher_hand,
            "barrel_rate": r["barrel_rate"],
            "whiff_rate":  r["whiff_rate"],
            "z_barrel":    z_barrel,
            "z_whiff":     z_whiff,
            "sample_size": r.get("sample_size", 0),
        })
    return pd.DataFrame(rows)


def compute_mash(
    pitcher_arsenal: pd.DataFrame,
    batter_profile: pd.DataFrame,
    *,
    pitcher_handedness: str,
    batter_handedness: str,
    league_baselines: dict | None = None,
) -> float:
    """
    Matchup Advantage Score — batter contact quality vs this pitcher.

    Args:
        pitcher_arsenal:   Output of transforms.pitchers.parse_pitcher_arsenal()
        batter_profile:    Output of transforms.batters.parse_batter_pitch_splits()
        pitcher_handedness: 'R' or 'L'
        batter_handedness:  'R', 'L', or 'S' (switch — treated as matching hand)
        league_baselines:   dict of per-pitch-type mean/std (falls back to hardcoded values)

    Returns float 0-100. 50 = neutral. Higher = batter barrel advantage.
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

    merged["barrel_overlap"] = merged.apply(
        lambda r: sigmoid(r["z_barrel"] * r["z_freq"]), axis=1
    )

    total_weight = merged["usage_pct"].sum()
    if total_weight <= 0:
        return 50.0

    return round((merged["barrel_overlap"] * merged["usage_pct"]).sum() / total_weight * 100, 1)


def compute_mash_full(
    pitcher_arsenal: pd.DataFrame,
    batter_profile: pd.DataFrame,
    *,
    pitcher_handedness: str,
    batter_handedness: str,
    league_baselines: dict | None = None,
) -> dict:
    """
    Full MASH result with per-pitch breakdown and primary driver.

    Returns: {
        "mash": float,
        "primary_driver": dict,
        "qualified_pitches": list,
        "per_pitch_breakdown": DataFrame,
    }
    """
    baselines = league_baselines or _FALLBACK_BASELINES
    hand = pitcher_handedness

    pitcher_z = _pitcher_z_scores(pitcher_arsenal, baselines)
    batter_z = _batter_z_scores(batter_profile, hand, baselines)

    _empty = {
        "mash": 50.0, "primary_driver": {}, "qualified_pitches": [],
        "per_pitch_breakdown": pd.DataFrame(),
    }

    if pitcher_z.empty or batter_z.empty:
        return _empty

    merged = pitcher_z.merge(batter_z, on="pitch_type", how="inner")
    if merged.empty:
        return _empty

    merged["barrel_overlap"] = merged.apply(
        lambda r: sigmoid(r["z_barrel"] * r["z_freq"]), axis=1
    )
    merged["whiff_overlap"] = merged.apply(
        lambda r: sigmoid(r["z_whiff"] * r["z_freq"]), axis=1
    )

    total_weight = merged["usage_pct"].sum()
    if total_weight <= 0:
        return _empty

    mash = round((merged["barrel_overlap"] * merged["usage_pct"]).sum() / total_weight * 100, 1)

    merged["contribution"] = merged["barrel_overlap"] * merged["usage_pct"]
    top = merged.nlargest(1, "contribution")
    if not top.empty:
        tr = top.iloc[0]
        total_contrib = merged["contribution"].sum()
        driver = {
            "pitch_type":       tr["pitch_type"],
            "pitch_label":      tr.get("pitch_label", tr["pitch_type"]),
            "usage_pct":        round(tr["usage_pct"] * 100, 1),
            "contribution_pct": round((tr["contribution"] / total_contrib) * 100, 1) if total_contrib > 0 else 0.0,
        }
    else:
        driver = {}

    return {
        "mash":               mash,
        "primary_driver":     driver,
        "qualified_pitches":  merged["pitch_type"].tolist(),
        "per_pitch_breakdown": merged,
    }
