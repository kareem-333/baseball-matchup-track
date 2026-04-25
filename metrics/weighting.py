# Layer 4: Multi-season sample weighting (Marcel-style) and dynamic current-season weight.
# Pure functions — no I/O.

from __future__ import annotations


def dynamic_current_weight(n_current: int) -> float:
    """
    Effective weight for the current season given sample size.
    Scales from 0 at 0 pitches to 0.70 cap as n → ∞.

    At 100 pitches:  ~17% weight
    At 500 pitches:  50% weight
    At 2000 pitches: ~80% (capped at 70%)
    """
    return min(0.70, n_current / (n_current + 500))


def marcel_pitch_counts(
    n_current: int,
    n_prior1: int = 0,
    n_prior2: int = 0,
    n_prior3: int = 0,
) -> float:
    """
    Marcel-style effective pitch count combining up to four seasons.

    Weights:  current × w_current  +  prior1 × 0.50  +  prior2 × 0.25  +  prior3 × 0.125
    where w_current = dynamic_current_weight(n_current) and the past-season weights
    are redistributed so that total weight sums to 1.

    Returns the effective sample size for downstream shrinkage calculations.
    """
    w_current = dynamic_current_weight(n_current)
    remaining = 1.0 - w_current

    # Distribute remaining weight using the Marcel 0.5/0.25/0.125 ratios
    raw_past = 0.50 * n_prior1 + 0.25 * n_prior2 + 0.125 * n_prior3
    total_past_raw = 0.50 + 0.25 + 0.125  # max possible if all three seasons present

    # Scale past weights so they sum to `remaining` if all data present
    scale = remaining / total_past_raw if total_past_raw > 0 else 0.0

    return (
        w_current * n_current
        + scale * 0.50 * n_prior1
        + scale * 0.25 * n_prior2
        + scale * 0.125 * n_prior3
    )


def marcel_weighted_mean(
    values: list[float],
    counts: list[int],
) -> float:
    """
    Compute Marcel-weighted mean across up to 4 seasons.

    Args:
        values: [current_mean, prior1_mean, prior2_mean, prior3_mean] (trailing seasons optional)
        counts: [current_n, prior1_n, prior2_n, prior3_n]

    Returns weighted mean, or 0.0 if total weight is 0.
    """
    assert len(values) == len(counts), "values and counts must have the same length"
    n = len(values)
    season_weights = [1.0, 0.5, 0.25, 0.125][:n]

    # Apply dynamic current-season weight
    if n > 0 and counts[0] > 0:
        season_weights[0] = dynamic_current_weight(counts[0])

    weighted_sum = sum(w * v * c for w, v, c in zip(season_weights, values, counts))
    total_weight = sum(w * c for w, c in zip(season_weights, counts))
    return weighted_sum / total_weight if total_weight > 0 else 0.0
