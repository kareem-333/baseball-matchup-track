import math

import pandas as pd
import pytest

from metrics.features import (
    edge_distance_inches,
    compute_edge_distances,
    compute_vaa,
    compute_break_point_distance,
)
from metrics.weighting import dynamic_current_weight, marcel_pitch_counts, marcel_weighted_mean
from metrics.shrinkage import shrink, shrink_series
from metrics.mash import compute_mash, compute_mash_full, _FALLBACK_BASELINES
from metrics.miss import compute_miss
from metrics.command import compute_edge_pct, compute_command_components, compute_command_plus
from metrics.deception import (
    compute_release_consistency,
    compute_velo_separation,
    compute_late_break_share,
    compute_deception_plus,
)
from metrics.arsenal import compute_arsenal_plus


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeDistance:
    def test_dead_center_is_negative(self):
        # Center of zone is negative (inside)
        d = edge_distance_inches(0.0, 2.5)
        assert d < 0

    def test_outside_is_positive(self):
        # Far outside the zone
        d = edge_distance_inches(2.0, 2.5)
        assert d > 0

    def test_on_edge_is_near_zero(self):
        # Exactly on the right edge
        d = edge_distance_inches(0.831, 2.5)
        assert abs(d) < 0.1   # very close to 0

    def test_nan_propagates(self):
        assert math.isnan(edge_distance_inches(float("nan"), 2.5))

    def test_vectorized_matches_scalar(self):
        df = pd.DataFrame({"plate_x": [0.0, 2.0, 0.831], "plate_z": [2.5, 2.5, 2.5]})
        series = compute_edge_distances(df)
        assert len(series) == 3
        assert series.iloc[0] < 0    # inside
        assert series.iloc[1] > 0    # outside
        assert abs(series.iloc[2]) < 0.1   # near edge


class TestVAA:
    def test_fastball_vaa_is_negative(self):
        # Typical 4-seam FB params — ball drops, angle should be negative
        vaa = compute_vaa(vy0=-130.0, vz0=5.0, ay=-17.0, az=-20.0)
        assert vaa < 0

    def test_vaa_nan_on_bad_input(self):
        assert math.isnan(compute_vaa(float("nan"), 5.0, -17.0, -20.0))


class TestBreakPointDistance:
    def test_straight_pitch_returns_inf(self):
        d = compute_break_point_distance(ax=0.0, az=0.0, vy0=-130.0, ay=-17.0)
        assert d == float("inf")

    def test_high_break_pitch_has_finite_distance(self):
        # Large horizontal + vertical break (like a sweeper)
        d = compute_break_point_distance(ax=15.0, az=-25.0, vy0=-130.0, ay=-17.0)
        assert 0 <= d < float("inf")

    def test_late_break_is_closer_to_plate(self):
        # High acceleration → ball deviates earlier → LARGER break-point distance
        # Low acceleration → deviates late → SMALLER break-point distance
        d_early = compute_break_point_distance(ax=20.0, az=-30.0, vy0=-130.0, ay=-17.0)
        d_late = compute_break_point_distance(ax=5.0, az=-8.0, vy0=-130.0, ay=-17.0)
        assert d_late < d_early


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

class TestWeighting:
    def test_dynamic_weight_caps_at_70pct(self):
        assert dynamic_current_weight(100_000) <= 0.70

    def test_dynamic_weight_zero_at_zero(self):
        assert dynamic_current_weight(0) == 0.0

    def test_dynamic_weight_50pct_at_500(self):
        assert abs(dynamic_current_weight(500) - 0.5) < 0.001

    def test_marcel_counts_increases_with_prior_data(self):
        n1 = marcel_pitch_counts(500, 0, 0, 0)
        n2 = marcel_pitch_counts(500, 1000, 800, 500)
        assert n2 > n1

    def test_marcel_weighted_mean_single_season(self):
        result = marcel_weighted_mean([0.320], [1000])
        assert abs(result - 0.320) < 0.001

    def test_marcel_weighted_mean_two_seasons(self):
        result = marcel_weighted_mean([0.320, 0.300], [1000, 1000])
        # Should be between 0.300 and 0.320
        assert 0.300 < result < 0.320


# ─────────────────────────────────────────────────────────────────────────────
# SHRINKAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestShrinkage:
    def test_zero_sample_returns_prior(self):
        assert shrink(120.0, prior=100.0, n=0) == 100.0

    def test_large_sample_near_observed(self):
        result = shrink(120.0, prior=100.0, n=10_000)
        assert result > 119.0   # very close to 120

    def test_k100_at_100_pitches_is_50_50(self):
        result = shrink(120.0, prior=100.0, n=100, k=100)
        assert abs(result - 110.0) < 0.01

    def test_series_length(self):
        results = shrink_series([110.0, 90.0], [100.0, 100.0], [200, 200])
        assert len(results) == 2


# ─────────────────────────────────────────────────────────────────────────────
# MASH
# ─────────────────────────────────────────────────────────────────────────────

def _arsenal(pitch_types, usages):
    return pd.DataFrame({
        "pitch_type":  pitch_types,
        "pitch_label": pitch_types,
        "usage_pct":   usages,
    })


def _splits(pitch_types, vs_hand, barrel_rates, whiff_rates, n=50):
    return pd.DataFrame({
        "pitch_type":  pitch_types,
        "vs_hand":     vs_hand,
        "barrel_rate": barrel_rates,
        "whiff_rate":  whiff_rates,
        "sample_size": [n] * len(pitch_types),
    })


class TestMASH:
    def test_known_advantage_above_50(self):
        # Batter has elite barrel rate vs FF; pitcher throws 100% FF
        arsenal = _arsenal(["FF"], [1.0])
        splits = _splits(["FF"], ["R"], [0.200], [0.200])
        score = compute_mash(arsenal, splits, pitcher_handedness="R", batter_handedness="R")
        assert score > 50

    def test_known_disadvantage_below_50(self):
        # Batter can't barrel anything
        arsenal = _arsenal(["FF"], [1.0])
        splits = _splits(["FF"], ["R"], [0.000], [0.500])
        score = compute_mash(arsenal, splits, pitcher_handedness="R", batter_handedness="R")
        assert score < 50

    def test_handedness_flip_changes_score(self):
        arsenal = _arsenal(["FF", "SL"], [0.6, 0.4])
        splits_r = _splits(["FF", "SL"], ["R", "R"], [0.10, 0.02], [0.30, 0.40])
        splits_l = _splits(["FF", "SL"], ["L", "L"], [0.02, 0.10], [0.40, 0.30])
        full_splits = pd.concat([splits_r, splits_l], ignore_index=True)

        score_vs_r = compute_mash(arsenal, full_splits, pitcher_handedness="R", batter_handedness="L")
        score_vs_l = compute_mash(arsenal, full_splits, pitcher_handedness="L", batter_handedness="L")
        assert score_vs_r != score_vs_l

    def test_empty_arsenal_returns_50(self):
        score = compute_mash(pd.DataFrame(), pd.DataFrame(), pitcher_handedness="R", batter_handedness="R")
        assert score == 50.0

    def test_no_overlap_returns_50(self):
        # Pitcher only throws SL; batter has no SL data
        arsenal = _arsenal(["SL"], [1.0])
        splits = _splits(["FF"], ["R"], [0.10], [0.30])
        score = compute_mash(arsenal, splits, pitcher_handedness="R", batter_handedness="R")
        assert score == 50.0

    def test_usage_floor_respected(self):
        # Add a 2% pitch — should be excluded from scoring
        arsenal = _arsenal(["FF", "EP"], [0.98, 0.02])
        splits = _splits(["FF", "EP"], ["R", "R"], [0.10, 0.50], [0.20, 0.60])
        result = compute_mash_full(arsenal, splits, pitcher_handedness="R", batter_handedness="R")
        assert "EP" not in result["qualified_pitches"]


# ─────────────────────────────────────────────────────────────────────────────
# MISS
# ─────────────────────────────────────────────────────────────────────────────

class TestMISS:
    def test_high_whiff_rate_above_50(self):
        arsenal = _arsenal(["FF"], [1.0])
        splits = _splits(["FF"], ["R"], [0.05], [0.500])
        score = compute_miss(arsenal, splits, pitcher_handedness="R", batter_handedness="R")
        assert score > 50

    def test_low_whiff_rate_below_50(self):
        arsenal = _arsenal(["FF"], [1.0])
        splits = _splits(["FF"], ["R"], [0.10], [0.050])
        score = compute_miss(arsenal, splits, pitcher_handedness="R", batter_handedness="R")
        assert score < 50

    def test_mash_miss_diverge(self):
        # High barrel + low whiff → high MASH, low MISS
        arsenal = _arsenal(["FF"], [1.0])
        splits = _splits(["FF"], ["R"], [0.200], [0.050])
        mash = compute_mash(arsenal, splits, pitcher_handedness="R", batter_handedness="R")
        miss = compute_miss(arsenal, splits, pitcher_handedness="R", batter_handedness="R")
        assert mash > miss


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND+
# ─────────────────────────────────────────────────────────────────────────────

def _pitch_df_at(plate_x, plate_z, pitch_type="FF", stand="R"):
    return pd.DataFrame({
        "plate_x":    plate_x,
        "plate_z":    plate_z,
        "pitch_type": pitch_type,
        "stand":      stand,
    })


class TestCommand:
    def test_edge_pct_all_on_edge(self):
        df = _pitch_df_at([0.831] * 20, [2.5] * 20)
        pct = compute_edge_pct(df, threshold_inches=3.0)
        assert pct > 0.9

    def test_edge_pct_all_in_center(self):
        df = _pitch_df_at([0.0] * 20, [2.5] * 20)
        pct = compute_edge_pct(df, threshold_inches=3.0)
        # Center of zone is about 10 inches from edge → all fail 3-inch threshold
        assert pct == 0.0

    def test_command_components_shape(self):
        n = 30
        df = _pitch_df_at([0.5] * n + [-0.5] * n, [2.0] * n + [3.0] * n)
        result = compute_command_components(df)
        assert "accuracy_in" in result.columns
        assert "command_raw" in result.columns

    def test_compute_command_plus_returns_dict(self):
        df = _pitch_df_at([0.5] * 50, [2.5] * 50)
        result = compute_command_plus(df)
        assert "overall" in result
        assert "edge_pct" in result
        assert "n_pitches" in result


# ─────────────────────────────────────────────────────────────────────────────
# DECEPTION+
# ─────────────────────────────────────────────────────────────────────────────

def _release_df():
    return pd.DataFrame({
        "pitch_type":    ["FF", "FF", "FF", "SL", "SL", "SL"],
        "release_pos_x": [1.5,  1.6,  1.5,  1.5,  1.6,  1.5],
        "release_pos_z": [5.8,  5.9,  5.8,  5.7,  5.8,  5.7],
        "release_speed": [96.0, 97.0, 96.0, 84.0, 85.0, 84.0],
        "ax":  [-3.0, -3.1, -3.0, 8.0, 8.1, 8.0],
        "az":  [-15.0, -15.5, -15.0, -30.0, -29.0, -30.0],
        "vy0": [-130.0] * 6,
        "ay":  [-17.0] * 6,
    })


class TestDeception:
    def test_release_consistency_value_range(self):
        df = _release_df()
        score = compute_release_consistency(df)
        assert 0.0 <= score <= 1.0

    def test_perfect_consistency_when_one_pitch(self):
        df = _release_df()[_release_df()["pitch_type"] == "FF"]
        score = compute_release_consistency(df)
        assert score == 1.0

    def test_velo_separation_positive(self):
        df = _release_df()
        gap = compute_velo_separation(df)
        assert gap > 0   # FB is faster than SL

    def test_velo_separation_nan_no_offspeed(self):
        df = _release_df()[_release_df()["pitch_type"] == "FF"]
        assert math.isnan(compute_velo_separation(df))

    def test_late_break_share_value_range(self):
        df = pd.DataFrame({
            "pitch_type": ["SL"] * 20,
            "ax": [8.0] * 20,
            "az": [-10.0] * 20,
            "vy0": [-130.0] * 20,
            "ay": [-17.0] * 20,
        })
        share = compute_late_break_share(df)
        assert 0.0 <= share <= 1.0

    def test_deception_plus_returns_dict(self):
        df = _release_df()
        result = compute_deception_plus(df)
        assert "overall" in result
        assert "release_consistency" in result
        assert "velo_gap_mph" in result
        assert "late_break_share" in result


# ─────────────────────────────────────────────────────────────────────────────
# ARSENAL+
# ─────────────────────────────────────────────────────────────────────────────

class TestArsenal:
    def _arsenal_df(self):
        return pd.DataFrame({
            "pitch_type": ["FF", "SL"],
            "usage_pct":  [0.60, 0.40],
            "count":      [600,  400],
        })

    def test_all_100_stuff_gives_100_arsenal(self):
        arsenal = self._arsenal_df()
        stuff = {"FF": 100.0, "SL": 100.0}
        result = compute_arsenal_plus(arsenal, stuff, league_avg_weighted=100.0)
        assert abs(result["overall"] - 100.0) < 0.1

    def test_above_average_stuff_gives_above_100_arsenal(self):
        arsenal = self._arsenal_df()
        stuff = {"FF": 115.0, "SL": 110.0}
        result = compute_arsenal_plus(arsenal, stuff, league_avg_weighted=100.0)
        assert result["overall"] > 100

    def test_below_min_pitches_is_not_qualified(self):
        arsenal = pd.DataFrame({
            "pitch_type": ["FF"],
            "usage_pct":  [1.0],
            "count":      [50],   # below 200 threshold
        })
        result = compute_arsenal_plus(arsenal, {"FF": 110.0})
        assert result["is_qualified"] is False

    def test_below_floor_pitch_excluded(self):
        arsenal = pd.DataFrame({
            "pitch_type": ["FF", "EP"],
            "usage_pct":  [0.98, 0.02],  # EP below 5%
            "count":      [980, 20],
        })
        result = compute_arsenal_plus(arsenal, {"FF": 110.0, "EP": 90.0})
        assert "EP" not in result["qualified"]

    def test_empty_arsenal_returns_100(self):
        result = compute_arsenal_plus(pd.DataFrame(), {})
        assert result["overall"] == 100.0
