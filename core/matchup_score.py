"""
Matchup Advantage Score (MAS) — core scoring module.

Public API: compute_matchup_score(batter_id, pitcher_id) -> dict

Returns:
    mas              float  0-100  higher = batter advantage on contact quality
    whiff_score      float  0-100  higher = expect strikeouts
    volatility       float  0-100  higher = boom-or-bust / less confident
    primary_driver   dict   {pitch_type, usage_pct, contribution_pct}
    pitcher_hand     str    'L' or 'R'
    sample_warnings  list[str]
    per_pitch_breakdown  DataFrame

Known v1 limitations (documented, not bugs):
    - Pitch location not modeled (same pitch type in any zone treated identically).
    - Count situation not modeled (aggregate pitch mix used).
    - Batter decay is approximate (season aggregate, not per-event timestamps).
    - Weights are theoretically motivated, not yet calibrated against outcomes.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd


# ── half-life decay ───────────────────────────────────────────────────────────

def pitcher_appearance_decay(appearances_ago: int, half_life: int = 3) -> float:
    """
    Pitcher data decays by appearance count, not calendar time.
    Half-life of 3 appearances means data from 3 starts ago has half the weight.
    """
    return 0.5 ** (appearances_ago / half_life)


def batter_calendar_decay(days_ago: int, half_life: int = 14) -> float:
    """
    Batter data decays by calendar days because hitters get reps daily.
    Half-life of 14 days means at-bats from 2 weeks ago have half the weight.
    """
    return 0.5 ** (days_ago / half_life)


def apply_pitcher_decay_to_arsenal(
    arsenal_df: pd.DataFrame, today: date | None = None
) -> pd.DataFrame:
    """
    Apply appearance-based decay to pitcher's per-appearance pitch mix data.
    If only season aggregate is available (no 'appearance_date' column),
    returns input unchanged with decay_weight=1.0 and decay_applied=False.
    """
    today = today or date.today()
    if arsenal_df.empty:
        return arsenal_df

    if "appearance_date" not in arsenal_df.columns:
        out = arsenal_df.copy()
        out["decay_weight"] = 1.0
        out["decay_applied"] = False
        return out

    out = arsenal_df.sort_values("appearance_date", ascending=False).copy()
    out["appearances_ago"] = range(len(out))
    out["decay_weight"] = out["appearances_ago"].apply(pitcher_appearance_decay)
    out["decay_applied"] = True
    return out


def apply_batter_decay_to_events(
    event_df: pd.DataFrame, today: date | None = None
) -> pd.DataFrame:
    """
    Apply calendar-day decay to per-event batter outcome data.
    event_df must have a 'game_date' column (date objects).
    """
    today = today or date.today()
    out = event_df.copy()
    out["days_ago"] = out["game_date"].apply(
        lambda d: (today - d).days if d else 999
    )
    out["decay_weight"] = out["days_ago"].apply(batter_calendar_decay)
    return out


# ── z-score computation ───────────────────────────────────────────────────────

def compute_batter_z_scores(
    batter_profile_df: pd.DataFrame, league_baselines: dict
) -> pd.DataFrame:
    """
    Returns a DataFrame with z_barrel and z_whiff per (pitch_type, vs_hand).
    Rows where the baseline is missing or std=0 are silently dropped.
    """
    rows = []
    for _, r in batter_profile_df.iterrows():
        pt = r["pitch_type"]
        baseline = league_baselines.get(pt)
        if not baseline:
            continue
        if baseline["barrel_rate_std"] == 0 or baseline["whiff_rate_std"] == 0:
            continue
        z_barrel = (
            (r["barrel_rate"] - baseline["barrel_rate_mean"])
            / baseline["barrel_rate_std"]
        )
        z_whiff = (
            (r["whiff_rate"] - baseline["whiff_rate_mean"])
            / baseline["whiff_rate_std"]
        )
        rows.append(
            {
                "pitch_type": pt,
                "vs_hand": r["vs_hand"],
                "z_barrel": z_barrel,
                "z_whiff": z_whiff,
                "sample_size": int(r.get("sample_size", 0)),
            }
        )
    return pd.DataFrame(rows)


def compute_pitcher_z_scores(
    pitcher_arsenal_df: pd.DataFrame, league_baselines: dict
) -> pd.DataFrame:
    """
    Returns a DataFrame with z_freq per pitch_type.
    Also carries usage_pct (raw) and decay_weight forward for downstream weighting.
    """
    rows = []
    for _, r in pitcher_arsenal_df.iterrows():
        pt = r["pitch_type"]
        baseline = league_baselines.get(pt)
        if not baseline:
            continue
        if baseline["usage_pct_std"] == 0:
            continue
        z_freq = (
            (r["usage_pct"] - baseline["usage_pct_mean"])
            / baseline["usage_pct_std"]
        )
        rows.append(
            {
                "pitch_type": pt,
                "usage_pct": float(r["usage_pct"]),
                "z_freq": z_freq,
                "decay_weight": float(r.get("decay_weight", 1.0)),
            }
        )
    return pd.DataFrame(rows)


# ── core scoring functions ────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    """Bound a z-score product to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def compute_per_pitch_overlap(
    batter_z_df: pd.DataFrame,
    pitcher_z_df: pd.DataFrame,
    pitcher_hand: str,
) -> pd.DataFrame:
    """
    For each pitch type the pitcher throws, compute overlap with the batter's
    tendency on that pitch type for the relevant handedness split.

    Missing batter data on a pitch type is skipped (flagged upstream as a warning).
    """
    batter_relevant = batter_z_df[batter_z_df["vs_hand"] == pitcher_hand]

    rows = []
    for _, p_row in pitcher_z_df.iterrows():
        pt = p_row["pitch_type"]
        b_match = batter_relevant[batter_relevant["pitch_type"] == pt]
        if b_match.empty:
            continue
        b_row = b_match.iloc[0]

        barrel_overlap = sigmoid(b_row["z_barrel"] * p_row["z_freq"])
        whiff_overlap = sigmoid(b_row["z_whiff"] * p_row["z_freq"])

        rows.append(
            {
                "pitch_type": pt,
                "usage_pct": p_row["usage_pct"],
                "decay_weight": p_row["decay_weight"],
                "barrel_overlap": barrel_overlap,
                "whiff_overlap": whiff_overlap,
                "z_barrel": b_row["z_barrel"],
                "z_whiff": b_row["z_whiff"],
                "z_freq": p_row["z_freq"],
                "sample_size": b_row["sample_size"],
            }
        )
    return pd.DataFrame(rows)


def aggregate_to_score(per_pitch_df: pd.DataFrame, score_col: str) -> float:
    """
    Weighted aggregation: pitcher's pitch frequency × decay × per-pitch overlap.
    Returns a 0-100 scaled score (50 = neutral).
    """
    if per_pitch_df.empty:
        return 50.0

    weights = per_pitch_df["usage_pct"] * per_pitch_df["decay_weight"]
    if weights.sum() == 0:
        return 50.0

    weighted_avg = (per_pitch_df[score_col] * weights).sum() / weights.sum()
    return round(weighted_avg * 100, 1)


def compute_volatility(per_pitch_df: pd.DataFrame) -> float:
    """
    Weighted standard deviation of per-pitch barrel overlap.
    High = boom-or-bust matchup; low = consistent prediction.
    Scaled to 0-100.
    """
    if per_pitch_df.empty or len(per_pitch_df) < 2:
        return 0.0

    weights = per_pitch_df["usage_pct"] * per_pitch_df["decay_weight"]
    if weights.sum() == 0:
        return 0.0

    mean = (per_pitch_df["barrel_overlap"] * weights).sum() / weights.sum()
    variance = (
        ((per_pitch_df["barrel_overlap"] - mean) ** 2) * weights
    ).sum() / weights.sum()
    return round(math.sqrt(variance) * 100, 1)


def identify_primary_driver(per_pitch_df: pd.DataFrame) -> dict:
    """
    Returns the pitch type that contributes most to the matchup advantage.
    This is the 'why' behind the MAS number.
    """
    if per_pitch_df.empty:
        return {}

    df = per_pitch_df.copy()
    df["contribution"] = df["barrel_overlap"] * df["usage_pct"]
    total_contribution = df["contribution"].sum()
    top = df.nlargest(1, "contribution").iloc[0]

    return {
        "pitch_type": top["pitch_type"],
        "usage_pct": round(float(top["usage_pct"]) * 100, 1),
        "contribution_pct": round(
            (top["contribution"] / total_contribution) * 100, 1
        ) if total_contribution > 0 else 0.0,
        "z_barrel": round(float(top["z_barrel"]), 2),
        "z_whiff": round(float(top["z_whiff"]), 2),
    }


# ── public API ────────────────────────────────────────────────────────────────

def compute_matchup_score(
    batter_id: int,
    pitcher_id: int,
    season: int | None = None,
) -> dict:
    """
    Primary entry point. Returns everything the dashboard needs for the headline UI.

    Return shape:
    {
        'mas': float,               # 0-100, higher = batter advantage on contact
        'whiff_score': float,       # 0-100, higher = expect strikeouts
        'volatility': float,        # 0-100, higher = less confident prediction
        'primary_driver': dict,     # {pitch_type, usage_pct, contribution_pct, z_barrel, z_whiff}
        'pitcher_hand': str,        # 'L' or 'R'
        'sample_warnings': list,    # data quality flags
        'per_pitch_breakdown': DataFrame,
    }
    """
    from core.player_lookup import get_pitcher_handedness
    from mlb_season.pipeline import (
        get_batter_pitch_splits,
        get_league_pitch_baselines,
        get_pitcher_arsenal,
    )

    season = season or date.today().year

    # 1. Pitcher handedness — most important confounder
    pitcher_hand = get_pitcher_handedness(pitcher_id)

    # 2. Raw profiles
    pitcher_arsenal = get_pitcher_arsenal(pitcher_id, season)
    batter_splits = get_batter_pitch_splits(batter_id, season)

    warnings: list[str] = []

    if pitcher_arsenal.empty:
        warnings.append("No Statcast data found for pitcher this season.")
        return _empty_result(pitcher_hand, warnings)

    if batter_splits.empty:
        warnings.append("No Statcast data found for batter this season.")
        return _empty_result(pitcher_hand, warnings)

    # 3. Apply pitcher decay (season aggregate → decay_weight=1.0, flagged)
    pitcher_arsenal = apply_pitcher_decay_to_arsenal(pitcher_arsenal)
    if not pitcher_arsenal.get("decay_applied", pd.Series([True])).all():
        warnings.append(
            "Pitcher decay not applied — only season-aggregate arsenal available (v1 limitation)."
        )

    # 4. League baselines
    league_baselines = get_league_pitch_baselines(season)
    if not league_baselines:
        warnings.append("League baselines unavailable; using hardcoded fallback values.")
        from mlb_season.pipeline import _fallback_baselines
        league_baselines = _fallback_baselines()

    # 5. Z-scores
    batter_z = compute_batter_z_scores(batter_splits, league_baselines)
    pitcher_z = compute_pitcher_z_scores(pitcher_arsenal, league_baselines)

    if batter_z.empty:
        warnings.append(
            "No z-scores computable for batter — pitch types may not match league baselines."
        )
        return _empty_result(pitcher_hand, warnings)

    if pitcher_z.empty:
        warnings.append(
            "No z-scores computable for pitcher — pitch types may not match league baselines."
        )
        return _empty_result(pitcher_hand, warnings)

    # 6. Per-pitch overlap
    per_pitch = compute_per_pitch_overlap(batter_z, pitcher_z, pitcher_hand)

    if per_pitch.empty:
        warnings.append(
            f"No overlapping pitch-type data for batter vs {pitcher_hand}HP pitcher. "
            "The batter may lack vs-hand splits for any pitch the pitcher throws."
        )
        return _empty_result(pitcher_hand, warnings)

    # 7. Aggregate
    mas = aggregate_to_score(per_pitch, "barrel_overlap")
    whiff = aggregate_to_score(per_pitch, "whiff_overlap")
    vol = compute_volatility(per_pitch)
    driver = identify_primary_driver(per_pitch)

    # 8. Sample warnings
    low_sample = per_pitch[per_pitch["sample_size"] < 30]["pitch_type"].tolist()
    if low_sample:
        warnings.append(
            f"Low sample size (<30 pitches seen) for: {', '.join(low_sample)}. "
            "Barrel rate on these pitch types is unstable."
        )

    return {
        "mas": mas,
        "whiff_score": whiff,
        "volatility": vol,
        "primary_driver": driver,
        "pitcher_hand": pitcher_hand,
        "sample_warnings": warnings,
        "per_pitch_breakdown": per_pitch,
    }


def _empty_result(pitcher_hand: str, warnings: list[str]) -> dict:
    return {
        "mas": 50.0,
        "whiff_score": 50.0,
        "volatility": 0.0,
        "primary_driver": {},
        "pitcher_hand": pitcher_hand,
        "sample_warnings": warnings,
        "per_pitch_breakdown": pd.DataFrame(),
    }


# ── MAS label helpers (used by both dashboard and tests) ─────────────────────

def mas_label(mas: float) -> str:
    if mas >= 70:
        return "Strong Advantage"
    if mas >= 50:
        return "Slight Advantage"
    if mas >= 30:
        return "Slight Disadvantage"
    return "Strong Disadvantage"


def mas_color(mas: float) -> str:
    if mas >= 70:
        return "#2ecc71"
    if mas >= 50:
        return "#95d35d"
    if mas >= 30:
        return "#f39c12"
    return "#e74c3c"


def mas_css_class(mas: float) -> str:
    if mas >= 70:
        return "mas-strong-advantage"
    if mas >= 50:
        return "mas-slight-advantage"
    if mas >= 30:
        return "mas-slight-disadvantage"
    return "mas-strong-disadvantage"
