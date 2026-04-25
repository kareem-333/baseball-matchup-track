# Layer 4: Stuff+ — per-pitch physical quality score.
# Normalized to 100 = league average; higher = better.
# Formula: stuff_plus = 100 + (-1 * (predicted_xwoba - league_avg_xwoba) / league_xwoba_std) * 10
#
# The ML model (LightGBM) is trained offline. This module handles:
#   1. Feature extraction from raw Statcast data
#   2. Score normalization (given model predictions)
#   3. Per-pitcher rollup
#   4. Shrinkage toward prior when sample is small
#
# When no model is provided, returns 100 (neutral) with is_modeled=False.

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.features import compute_vaa_series
from metrics.shrinkage import shrink

# Statcast columns needed for Stuff+ feature extraction
_FEATURE_COLS = [
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "spin_axis",
]

_PITCH_USAGE_FLOOR = 0.05

# Fallback league xwOBA constants (calibrate against real data when model is trained)
_FALLBACK_LEAGUE_XWOBA = 0.320
_FALLBACK_LEAGUE_STD = 0.035


def extract_stuff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for the Stuff+ model from raw Statcast data.

    Input:  raw Statcast DataFrame (one row per pitch)
    Output: feature DataFrame with columns matching _FEATURE_COLS + vaa_residual + pitch_type

    Missing feature columns are filled with NaN (model handles imputation).
    vaa_residual requires vy0, vz0, ay, az to be present; otherwise NaN.
    """
    features = pd.DataFrame(index=df.index)

    for col in _FEATURE_COLS:
        features[col] = pd.to_numeric(df.get(col), errors="coerce")

    # VAA residual: actual VAA - expected VAA
    # Expected VAA proxy: league mean VAA for that pitch type at that release height.
    # Without a trained VAA model, use the raw VAA as a proxy (residual = 0).
    if {"vy0", "vz0", "ay", "az"}.issubset(df.columns):
        features["vaa"] = compute_vaa_series(df)
        # Residual requires a league VAA model; set to 0 until trained
        features["vaa_residual"] = 0.0
    else:
        features["vaa"] = np.nan
        features["vaa_residual"] = np.nan

    features["pitch_type"] = df.get("pitch_type", pd.Series(np.nan, index=df.index))
    features["p_throws"] = df.get("p_throws", pd.Series(np.nan, index=df.index))

    return features


def score_pitches(
    features: pd.DataFrame,
    model,
    league_xwoba_mean: float = _FALLBACK_LEAGUE_XWOBA,
    league_xwoba_std: float = _FALLBACK_LEAGUE_STD,
) -> pd.Series:
    """
    Apply a trained model to feature matrix and normalize to Stuff+ scale.

    model must implement .predict(X) → array of xwOBA predictions.
    Returns per-pitch Stuff+ as a Series.
    """
    numeric_cols = [c for c in features.columns if c not in ("pitch_type", "p_throws")]
    X = features[numeric_cols].fillna(features[numeric_cols].median())
    predicted_xwoba = model.predict(X)
    # Higher predicted xwOBA = worse for pitcher → negate
    return pd.Series(
        100.0 + (-1.0 * (predicted_xwoba - league_xwoba_mean) / league_xwoba_std) * 10.0,
        index=features.index,
        name="stuff_plus",
    )


def compute_stuff_plus(
    df: pd.DataFrame,
    model=None,
    league_xwoba_mean: float = _FALLBACK_LEAGUE_XWOBA,
    league_xwoba_std: float = _FALLBACK_LEAGUE_STD,
    shrinkage_k: int = 100,
) -> dict:
    """
    Compute Stuff+ for a pitcher from raw Statcast pitch-level data.

    Input:  df — raw Statcast DataFrame for one pitcher (one row per pitch)
    Output: {
        "overall":   float,             # usage-weighted Stuff+ across all pitches
        "per_pitch": dict[str, float],  # Stuff+ per pitch type
        "n_pitches": int,
        "is_modeled": bool,             # False when no model provided (returns 100)
        "qualified": list[str],         # pitch types meeting usage floor
    }

    When model=None, per_pitch scores are all 100 (neutral placeholder).
    Shrinkage is applied toward 100 (league average) when sample < shrinkage_k * 5.
    """
    if df.empty or "pitch_type" not in df.columns:
        return {"overall": 100.0, "per_pitch": {}, "n_pitches": 0, "is_modeled": False, "qualified": []}

    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")].copy()
    n_total = len(df)

    if model is None:
        # No model available — return neutral scores
        usage = df.groupby("pitch_type").size() / n_total
        qualified = usage[usage >= _PITCH_USAGE_FLOOR].index.tolist()
        return {
            "overall": 100.0,
            "per_pitch": {pt: 100.0 for pt in qualified},
            "n_pitches": n_total,
            "is_modeled": False,
            "qualified": qualified,
        }

    features = extract_stuff_features(df)
    per_pitch_raw = score_pitches(features, model, league_xwoba_mean, league_xwoba_std)
    df = df.copy()
    df["stuff_plus_raw"] = per_pitch_raw

    usage = df.groupby("pitch_type").size() / n_total
    qualified = usage[usage >= _PITCH_USAGE_FLOOR].index.tolist()

    per_pitch: dict[str, float] = {}
    for pt in qualified:
        mask = df["pitch_type"] == pt
        n_pt = mask.sum()
        obs = float(df.loc[mask, "stuff_plus_raw"].mean())
        per_pitch[pt] = shrink(obs, prior=100.0, n=n_pt, k=shrinkage_k)

    if not per_pitch:
        return {"overall": 100.0, "per_pitch": {}, "n_pitches": n_total, "is_modeled": True, "qualified": []}

    usage_q = usage[qualified]
    overall = float(
        sum(per_pitch[pt] * usage_q[pt] for pt in qualified) / usage_q[qualified].sum()
    )
    return {
        "overall": round(overall, 1),
        "per_pitch": {pt: round(v, 1) for pt, v in per_pitch.items()},
        "n_pitches": n_total,
        "is_modeled": True,
        "qualified": qualified,
    }
