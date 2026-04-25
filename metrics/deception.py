# Layer 4: Deception+ — hitter-decision difficulty.
# Normalized to 100 = league average; higher = better.
#
# Three components:
#   1. Release consistency — how similarly the pitcher releases each pitch type
#   2. Velocity separation — fastball vs offspeed gap
#   3. Late-break share — % breaking balls with break-point distance < 25 ft
#
# Weights: 0.3 / 0.3 / 0.4 (tunable).

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from metrics.features import compute_break_point_distances, normalize_to_100

# Pitch type classifications
_FASTBALL_TYPES = {"FF", "SI", "FC"}
_OFFSPEED_TYPES = {"CH", "FS", "CU", "KC", "SL", "ST", "CS", "SV"}
_BREAKING_TYPES = {"SL", "ST", "CU", "KC", "CS", "SV", "FS"}

_W_RELEASE = 0.3
_W_VELO = 0.3
_W_LATE_BREAK = 0.4

_LATE_BREAK_THRESHOLD_FT = 25.0   # pitches breaking within 25 ft = "late break"
_VELO_NORM_MEAN = 12.0             # approximate league-avg velo gap (mph); tune with data
_VELO_NORM_STD = 4.0


def compute_release_consistency(df: pd.DataFrame) -> float:
    """
    Release consistency score (0-1, higher = more consistent across pitch types).

    Measures mean pairwise Euclidean distance between per-pitch-type release centroids.
    Pitchers who release everything from the same point score highest.

        score = 1 / (1 + mean_pairwise_separation)  where separation is in feet

    Requires: release_pos_x, release_pos_z, pitch_type columns.
    Returns nan if fewer than 2 pitch types with sufficient data.
    """
    required = {"release_pos_x", "release_pos_z", "pitch_type"}
    if not required.issubset(df.columns) or df.empty:
        return float("nan")

    centroids = (
        df.dropna(subset=["release_pos_x", "release_pos_z"])
        .groupby("pitch_type")[["release_pos_x", "release_pos_z"]]
        .mean()
    )
    if len(centroids) < 2:
        return 1.0  # only one pitch type → perfect consistency by definition

    # Mean pairwise distance across all (pitch_type_1, pitch_type_2) pairs
    pts = centroids.values
    n = len(pts)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((pts[i, 0] - pts[j, 0]) ** 2 + (pts[i, 1] - pts[j, 1]) ** 2)
            distances.append(d)

    mean_sep = float(np.mean(distances))
    return 1.0 / (1.0 + mean_sep)


def compute_velo_separation(df: pd.DataFrame) -> float:
    """
    Fastball-vs-offspeed velocity gap in mph.

    Returns fastball_mean_velo - offspeed_mean_velo.
    Returns nan if pitcher doesn't throw both a fastball and an offspeed pitch.

    Requires: release_speed, pitch_type columns.
    """
    if "release_speed" not in df.columns or "pitch_type" not in df.columns or df.empty:
        return float("nan")

    fb = df[df["pitch_type"].isin(_FASTBALL_TYPES)]["release_speed"].dropna()
    os = df[df["pitch_type"].isin(_OFFSPEED_TYPES)]["release_speed"].dropna()

    if fb.empty or os.empty:
        return float("nan")

    return float(fb.mean() - os.mean())


def compute_late_break_share(
    df: pd.DataFrame,
    threshold_ft: float = _LATE_BREAK_THRESHOLD_FT,
) -> float:
    """
    % of breaking balls with break-point distance < threshold_ft from the plate.

    Pitches that deviate 4 inches from a straight line within 25 ft of the plate
    are "late-breaking" — past the hitter's decision window (~150-200ms before contact).

    Requires: ax, az, vy0, ay, pitch_type columns.
    Returns nan if no breaking balls found or physics columns missing.
    """
    breaking = df[df["pitch_type"].isin(_BREAKING_TYPES)].copy()
    if breaking.empty:
        return float("nan")

    required = {"ax", "az", "vy0", "ay"}
    if not required.issubset(breaking.columns):
        return float("nan")

    dists = compute_break_point_distances(breaking)
    valid = dists.dropna()
    if valid.empty:
        return float("nan")

    return float((valid < threshold_ft).mean())


def compute_deception_plus(
    df: pd.DataFrame,
    league_deception_mean: float | None = None,
    league_deception_std: float | None = None,
) -> dict:
    """
    Compute Deception+ for a pitcher from raw Statcast pitch-level data.

    Input:  df — raw Statcast DataFrame for one pitcher
    Output: {
        "overall":             float,   # normalized Deception+ (100 = league avg)
        "release_consistency": float,   # 0-1 score (higher = more consistent release)
        "velo_gap_mph":        float,   # fastball - offspeed mph
        "late_break_share":    float,   # 0-1 fraction of breaking balls with late break
        "deception_raw":       float,   # unnormalized composite
        "n_pitches":           int,
        "is_normalized":       bool,
    }
    """
    n = len(df)
    release_cons = compute_release_consistency(df)
    velo_gap = compute_velo_separation(df)
    late_break = compute_late_break_share(df)

    # Normalize velo gap to 0-1 scale for compositing
    if not math.isnan(velo_gap):
        velo_norm = max(0.0, min(1.0, velo_gap / (_VELO_NORM_MEAN * 2)))
    else:
        velo_norm = float("nan")

    # Composite — skip components that are nan
    components = [
        (_W_RELEASE, release_cons),
        (_W_VELO, velo_norm),
        (_W_LATE_BREAK, late_break),
    ]
    valid = [(w, v) for w, v in components if not math.isnan(v)]
    if not valid:
        deception_raw = float("nan")
        overall = 100.0
        is_normalized = False
    else:
        total_w = sum(w for w, _ in valid)
        deception_raw = sum(w * v for w, v in valid) / total_w

        if league_deception_mean is not None and league_deception_std is not None and league_deception_std > 0:
            overall = normalize_to_100(deception_raw, league_deception_mean, league_deception_std)
            is_normalized = True
        else:
            overall = 100.0
            is_normalized = False

    return {
        "overall":             round(overall, 1),
        "release_consistency": round(release_cons, 4) if not math.isnan(release_cons) else float("nan"),
        "velo_gap_mph":        round(velo_gap, 1) if not math.isnan(velo_gap) else float("nan"),
        "late_break_share":    round(late_break, 4) if not math.isnan(late_break) else float("nan"),
        "deception_raw":       round(deception_raw, 4) if not math.isnan(deception_raw) else float("nan"),
        "n_pitches":           n,
        "is_normalized":       is_normalized,
    }
