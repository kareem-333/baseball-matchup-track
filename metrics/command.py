# Layer 4: Command+ — location precision and consistency.
# Normalized to 100 = league average; higher = better.
#
# Core metric: signed edge distance (negative inside zone, positive outside).
# Two components: accuracy (mean distance from cluster center) + consistency (−std).
# edge_pct (% within 3 inches of edge) is the human-readable display number.
# Command+ is the normalized composite that feeds the model.

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.features import compute_edge_distances, normalize_to_100

_W_ACCURACY = 0.6
_W_CONSISTENCY = 0.4
_EDGE_PCT_THRESHOLD_IN = 3.0   # inches from edge for edge_pct display metric
_MIN_PITCHES_PER_CELL = 10     # minimum pitches to compute a cluster


def compute_edge_pct(df: pd.DataFrame, threshold_inches: float = _EDGE_PCT_THRESHOLD_IN) -> float:
    """
    % of pitches within threshold_inches of the nearest strike-zone edge.
    Human-readable display metric — Command+ is what feeds the model.

    Input:  raw Statcast DataFrame with plate_x, plate_z columns
    Output: float 0-1
    """
    if df.empty or "plate_x" not in df.columns:
        return float("nan")
    dists = compute_edge_distances(df).dropna()
    if dists.empty:
        return float("nan")
    return float((dists.abs() <= threshold_inches).mean())


def compute_command_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy and consistency per (pitch_type, batter_hand, count_bucket).

    Input:  raw Statcast DataFrame
    Output: DataFrame with columns:
        pitch_type, batter_hand, count_bucket,
        n_pitches, accuracy_in, consistency_in, command_raw

    accuracy    = mean absolute deviation from cluster center (inches)
    consistency = −std of miss distances (negated so higher = better)
    command_raw = w_acc * (−accuracy) + w_cons * consistency  [higher = better]

    Cluster center = empirical mean location for (pitch_type, batter_hand, count_bucket).
    """
    if df.empty or "plate_x" not in df.columns or "plate_z" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["edge_dist_in"] = compute_edge_distances(df)

    # Count bucket: early (0-1 strikes/0-1 balls), two-strike, three-ball
    if "strikes" in df.columns and "balls" in df.columns:
        df["count_bucket"] = pd.cut(
            df["strikes"] * 10 + df["balls"],
            bins=[-1, 10, 20, 99],
            labels=["early", "two_strike", "three_ball"],
        ).astype(str)
    else:
        df["count_bucket"] = "all"

    batter_hand_col = "stand" if "stand" in df.columns else None
    if batter_hand_col is None:
        df["batter_hand"] = "?"
    else:
        df["batter_hand"] = df[batter_hand_col]

    group_cols = ["pitch_type", "batter_hand", "count_bucket"]
    rows = []
    for keys, grp in df.groupby(group_cols):
        if len(grp) < _MIN_PITCHES_PER_CELL:
            continue
        cx = grp["plate_x"].mean()
        cz = grp["plate_z"].mean()
        miss = np.sqrt((grp["plate_x"] - cx) ** 2 + (grp["plate_z"] - cz) ** 2) * 12.0  # inches
        accuracy = float(miss.mean())
        consistency = float(-miss.std()) if len(miss) > 1 else 0.0
        command_raw = _W_ACCURACY * (-accuracy) + _W_CONSISTENCY * consistency

        rows.append({
            "pitch_type":    keys[0],
            "batter_hand":   keys[1],
            "count_bucket":  keys[2],
            "n_pitches":     len(grp),
            "accuracy_in":   round(accuracy, 3),
            "consistency_in": round(-consistency, 3),  # store as positive std for display
            "command_raw":   round(command_raw, 4),
        })

    return pd.DataFrame(rows)


def compute_command_plus(
    df: pd.DataFrame,
    league_command_mean: float | None = None,
    league_command_std: float | None = None,
) -> dict:
    """
    Compute Command+ for a pitcher from raw Statcast pitch-level data.

    Input:  df — raw Statcast DataFrame for one pitcher
    Output: {
        "overall":       float,   # normalized Command+ (100 = league avg)
        "edge_pct":      float,   # % pitches within 3 inches of edge
        "command_raw":   float,   # unnormalized composite
        "n_pitches":     int,
        "by_pitch_type": DataFrame,   # command_raw per pitch_type
        "is_normalized": bool,   # False if no league baseline provided
    }

    If league_command_mean/std are None, overall is returned on raw scale (not 100-normalized).
    """
    if df.empty:
        return {"overall": 100.0, "edge_pct": float("nan"), "command_raw": float("nan"),
                "n_pitches": 0, "by_pitch_type": pd.DataFrame(), "is_normalized": False}

    components = compute_command_components(df)
    edge_pct = compute_edge_pct(df)
    n = len(df)

    if components.empty:
        return {"overall": 100.0, "edge_pct": edge_pct, "command_raw": float("nan"),
                "n_pitches": n, "by_pitch_type": pd.DataFrame(), "is_normalized": False}

    command_raw = float(components["command_raw"].mean())

    if league_command_mean is not None and league_command_std is not None and league_command_std > 0:
        overall = normalize_to_100(command_raw, league_command_mean, league_command_std, higher_is_better=True)
        is_normalized = True
    else:
        overall = 100.0
        is_normalized = False

    by_pitch = (
        components.groupby("pitch_type")["command_raw"]
        .mean()
        .reset_index()
        .rename(columns={"command_raw": "command_raw_mean"})
    )

    return {
        "overall":       round(overall, 1),
        "edge_pct":      round(edge_pct, 4) if not pd.isna(edge_pct) else float("nan"),
        "command_raw":   round(command_raw, 4),
        "n_pitches":     n,
        "by_pitch_type": by_pitch,
        "is_normalized": is_normalized,
    }
