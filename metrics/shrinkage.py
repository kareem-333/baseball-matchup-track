# Layer 4: Empirical Bayes shrinkage for pitcher scores.
# Prevents noisy small-sample scores from dominating. Pure functions — no I/O.

from __future__ import annotations


def shrink(observed: float, prior: float, n: int, k: int = 100) -> float:
    """
    Shrink an observed score toward a prior using empirical Bayes.

        shrunk = (n / (n + k)) * observed + (k / (n + k)) * prior

    Args:
        observed: the raw measured score for this pitcher/pitch
        prior:    the model-predicted or league-average score
        n:        sample size (pitches thrown)
        k:        shrinkage constant (default 100 pitches — tune empirically)

    At n=0:   fully at prior
    At n=100: 50% observed, 50% prior
    At n=500: 83% observed, 17% prior
    At n=∞:   fully observed
    """
    if n + k == 0:
        return prior
    alpha = n / (n + k)
    return alpha * observed + (1 - alpha) * prior


def shrink_series(
    observed_scores: list[float],
    prior_scores: list[float],
    sample_sizes: list[int],
    k: int = 100,
) -> list[float]:
    """Vectorized shrinkage for a list of (observed, prior, n) triples."""
    return [
        shrink(obs, pri, n, k)
        for obs, pri, n in zip(observed_scores, prior_scores, sample_sizes)
    ]
