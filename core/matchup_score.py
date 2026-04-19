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


# ── MASH / MISS scoring (game-first flow) ────────────────────────────────────

PITCH_USAGE_FLOOR = 0.05  # 5% — pitches below this are ignored


def _compute_pitcher_z_scores_qualified(
    arsenal: pd.DataFrame,
    baselines: dict,
) -> pd.DataFrame:
    """Filter arsenal to qualified pitches (>= 5% usage) and compute z_freq."""
    if arsenal.empty:
        return pd.DataFrame()

    qualified = arsenal[arsenal["usage_pct"] >= PITCH_USAGE_FLOOR].copy()
    if qualified.empty:
        return pd.DataFrame()

    rows = []
    for _, r in qualified.iterrows():
        pt = r["pitch_type"]
        baseline = baselines.get(pt)
        if not baseline or baseline.get("usage_pct_std", 0) <= 0:
            z_freq = 0.0
        else:
            z_freq = (r["usage_pct"] - baseline["usage_pct_mean"]) / baseline["usage_pct_std"]
        rows.append({
            "pitch_type": pt,
            "pitch_label": r.get("pitch_label", pt),
            "usage_pct": r["usage_pct"],
            "z_freq": z_freq,
        })
    return pd.DataFrame(rows)


def _compute_batter_z_scores_for_hand(
    splits: pd.DataFrame,
    pitcher_hand: str,
    baselines: dict,
) -> pd.DataFrame:
    """Filter splits to the matching handedness and compute z_barrel and z_whiff."""
    if splits.empty or "vs_hand" not in splits.columns:
        return pd.DataFrame()

    relevant = splits[splits["vs_hand"] == pitcher_hand].copy()
    if relevant.empty:
        return pd.DataFrame()

    rows = []
    for _, r in relevant.iterrows():
        pt = r["pitch_type"]
        baseline = baselines.get(pt)
        if not baseline:
            z_barrel = z_whiff = 0.0
        else:
            br_std = baseline.get("barrel_rate_std", 0)
            wr_std = baseline.get("whiff_rate_std", 0)
            z_barrel = (
                (r["barrel_rate"] - baseline["barrel_rate_mean"]) / br_std
                if br_std > 0 else 0.0
            )
            z_whiff = (
                (r["whiff_rate"] - baseline["whiff_rate_mean"]) / wr_std
                if wr_std > 0 else 0.0
            )
        rows.append({
            "pitch_type": pt,
            "vs_hand": pitcher_hand,
            "barrel_rate": r["barrel_rate"],
            "whiff_rate": r["whiff_rate"],
            "z_barrel": z_barrel,
            "z_whiff": z_whiff,
            "sample_size": r.get("sample_size", 0),
        })
    return pd.DataFrame(rows)


def compute_mash_and_miss(
    batter_id: int,
    pitcher_id: int,
    season: int | None = None,
) -> dict:
    """
    Game-first matchup scoring.

    Returns:
        mash: float (0-100) — higher = batter advantage on contact quality
        miss: float (0-100) — higher = expect strikeouts / swing-and-miss
        primary_driver: dict {pitch_type, pitch_label, usage_pct, contribution_pct}
        pitcher_hand: str
        qualified_pitches: list[str]
        low_sample_warning: str
        per_pitch_breakdown: DataFrame
    """
    from core.handedness import get_pitcher_handedness
    from mlb_season.pipeline import (
        get_pitcher_arsenal,
        get_batter_pitch_splits,
        get_league_pitch_baselines,
        get_pitcher_sample_flag,
    )
    from datetime import date as _date

    season = season or _date.today().year

    pitcher_hand = get_pitcher_handedness(pitcher_id)
    sample_flag = get_pitcher_sample_flag(pitcher_id, season)

    arsenal = get_pitcher_arsenal(pitcher_id, season)
    splits = get_batter_pitch_splits(batter_id, season)
    baselines = get_league_pitch_baselines(season)

    _empty = {
        "mash": 50.0,
        "miss": 50.0,
        "primary_driver": {},
        "pitcher_hand": pitcher_hand,
        "qualified_pitches": [],
        "low_sample_warning": sample_flag["reason"] or "Insufficient data",
        "per_pitch_breakdown": pd.DataFrame(),
    }

    pitcher_z = _compute_pitcher_z_scores_qualified(arsenal, baselines)
    batter_z = _compute_batter_z_scores_for_hand(splits, pitcher_hand, baselines)

    if pitcher_z.empty or batter_z.empty:
        return _empty

    merged = pitcher_z.merge(batter_z, on="pitch_type", how="inner")
    if merged.empty:
        _empty["qualified_pitches"] = pitcher_z["pitch_type"].tolist()
        _empty["low_sample_warning"] = (
            sample_flag["reason"] or
            f"Batter has no recorded data vs {pitcher_hand}HP on pitcher's qualified pitches"
        )
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

    mash_raw = (merged["barrel_overlap"] * merged["usage_pct"]).sum() / total_weight
    miss_raw = (merged["whiff_overlap"] * merged["usage_pct"]).sum() / total_weight
    mash = round(mash_raw * 100, 1)
    miss = round(miss_raw * 100, 1)

    merged["contribution"] = merged["barrel_overlap"] * merged["usage_pct"]
    top = merged.nlargest(1, "contribution")
    if not top.empty:
        top_row = top.iloc[0]
        total_contrib = merged["contribution"].sum()
        primary_driver = {
            "pitch_type": top_row["pitch_type"],
            "pitch_label": top_row.get("pitch_label", top_row["pitch_type"]),
            "usage_pct": round(top_row["usage_pct"] * 100, 1),
            "contribution_pct": round(
                (top_row["contribution"] / total_contrib) * 100, 1
            ) if total_contrib > 0 else 0.0,
        }
    else:
        primary_driver = {}

    return {
        "mash": mash,
        "miss": miss,
        "primary_driver": primary_driver,
        "pitcher_hand": pitcher_hand,
        "qualified_pitches": merged["pitch_type"].tolist(),
        "low_sample_warning": sample_flag["reason"],
        "per_pitch_breakdown": merged,
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
