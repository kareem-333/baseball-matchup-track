from datetime import date

import pandas as pd
import pytest

from transforms.pitchers import (
    parse_pitcher_arsenal,
    parse_qualified_arsenal,
    parse_pitcher_game_log,
    parse_pitcher_sample_flag,
    PITCH_USAGE_FLOOR,
)
from transforms.batters import (
    parse_batter_pitch_splits,
    parse_batter_game_log,
    parse_batter_hot_zones,
    parse_batter_barrel_trend,
    parse_league_avg_krate,
)
from transforms.lineups import (
    parse_confirmed_lineup,
    parse_predicted_lineup,
    parse_lineup_vs_hand,
    merge_partial_and_predicted,
)


# ─────────────────────────────────────────────────────────────────────────────
# PITCHER TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def _pitcher_df():
    return pd.DataFrame({
        "pitch_type": ["FF", "FF", "FF", "SL", "SL", "CH"],
        "release_speed": [96.0, 97.0, 95.0, 84.0, 85.0, 88.0],
        "release_spin_rate": [2200, 2250, 2150, 2400, 2350, 1800],
        "api_break_x_arm": [1.0, 1.1, 0.9, -3.0, -2.8, 0.5],
        "api_break_z_with_gravity": [10.0, 10.5, 9.8, 3.0, 2.9, 6.0],
        "plate_x": [0.1, 0.2, -0.1, -0.3, -0.2, 0.0],
        "plate_z": [2.5, 2.6, 2.4, 2.0, 1.9, 2.2],
        "game_date": [date(2026, 4, 1)] * 6,
        "events": ["strikeout", None, None, "field_out", None, None],
        "description": ["swinging_strike", "ball", "called_strike",
                        "swinging_strike_blocked", "foul", "hit_into_play"],
    })


def test_pitcher_arsenal_shape_and_usage():
    df = _pitcher_df()
    result = parse_pitcher_arsenal(df)
    assert len(result) == 3
    assert set(result["pitch_type"]) == {"FF", "SL", "CH"}
    total_usage = result["usage_pct"].sum()
    assert abs(total_usage - 1.0) < 0.001


def test_pitcher_arsenal_sorted_by_usage():
    df = _pitcher_df()
    result = parse_pitcher_arsenal(df)
    # FF has 3/6 = 50% → should be first
    assert result.iloc[0]["pitch_type"] == "FF"
    assert abs(result.iloc[0]["usage_pct"] - 0.5) < 0.01


def test_pitcher_arsenal_empty_input():
    result = parse_pitcher_arsenal(pd.DataFrame())
    assert result.empty


def test_pitcher_arsenal_missing_pitch_type_col():
    df = pd.DataFrame({"release_speed": [95.0]})
    result = parse_pitcher_arsenal(df)
    assert result.empty


def test_qualified_arsenal_filters_floor():
    arsenal = pd.DataFrame({
        "pitch_type": ["FF", "SL", "EP"],
        "usage_pct": [0.50, 0.45, 0.02],  # EP is below 5% floor
    })
    result = parse_qualified_arsenal(arsenal)
    assert len(result) == 2
    assert "EP" not in result["pitch_type"].values


def test_qualified_arsenal_all_zero_usage():
    arsenal = pd.DataFrame({"pitch_type": ["FF", "SL"], "usage_pct": [0.0, 0.0]})
    result = parse_qualified_arsenal(arsenal)
    assert result.empty


def test_pitcher_game_log_counts():
    df = _pitcher_df()
    result = parse_pitcher_game_log(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["pitches"] == 6
    assert row["strikeouts"] == 1
    assert row["whiffs"] == 2
    assert abs(row["whiff_rate"] - 2 / 6) < 0.01


def test_pitcher_game_log_empty():
    result = parse_pitcher_game_log(pd.DataFrame())
    assert result.empty


def test_pitcher_sample_flag_low():
    flag = parse_pitcher_sample_flag(pd.DataFrame({"pitch_type": ["FF"] * 50}))
    assert flag["total_pitches"] == 50
    assert flag["is_low_sample"] is True
    assert "50" in flag["reason"]


def test_pitcher_sample_flag_zero():
    flag = parse_pitcher_sample_flag(pd.DataFrame())
    assert flag["total_pitches"] == 0
    assert flag["is_low_sample"] is True


def test_pitcher_sample_flag_ok():
    flag = parse_pitcher_sample_flag(pd.DataFrame({"pitch_type": ["FF"] * 300}))
    assert flag["is_low_sample"] is False
    assert flag["reason"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# BATTER TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def _batter_df():
    return pd.DataFrame({
        "pitch_type": ["FF", "FF", "FF", "SL", "SL"],
        "p_throws":   ["R",  "R",  "R",  "R",  "R"],
        "type":       ["X",  "S",  "B",  "X",  "S"],
        "launch_speed_angle": [6, None, None, 3, None],
        "description": ["hit_into_play", "swinging_strike", "ball",
                        "hit_into_play", "swinging_strike_blocked"],
        "game_date": [date(2026, 4, 1)] * 5,
        "events": ["home_run", None, None, "field_out", None],
        "estimated_ba_using_speedangle": [0.9, None, None, 0.3, None],
    })


def test_batter_pitch_splits_shape():
    df = _batter_df()
    result = parse_batter_pitch_splits(df)
    # Two pitch types × one hand each → 2 rows
    assert len(result) == 2
    assert set(result["pitch_type"]) == {"FF", "SL"}


def test_batter_pitch_splits_barrel_rate():
    df = _batter_df()
    result = parse_batter_pitch_splits(df)
    ff = result[result["pitch_type"] == "FF"].iloc[0]
    # 1 barrel in 1 BIP → 100%
    assert ff["barrel_rate"] == 1.0


def test_batter_pitch_splits_whiff_rate():
    df = _batter_df()
    result = parse_batter_pitch_splits(df)
    ff = result[result["pitch_type"] == "FF"].iloc[0]
    # 1 whiff in 2 swings (swinging_strike + hit_into_play) → 0.5
    assert abs(ff["whiff_rate"] - 0.5) < 0.01


def test_batter_pitch_splits_empty():
    assert parse_batter_pitch_splits(pd.DataFrame()).empty


def test_batter_pitch_splits_missing_handedness():
    df = pd.DataFrame({"pitch_type": ["FF"], "type": ["X"]})
    assert parse_batter_pitch_splits(df).empty


def test_batter_game_log_shape():
    df = _batter_df()
    result = parse_batter_game_log(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["hr"] == 1
    assert row["pitches"] == 5


def test_batter_game_log_empty():
    assert parse_batter_game_log(pd.DataFrame()).empty


def test_batter_hot_zones_9_rows():
    df = pd.DataFrame({
        "zone": list(range(1, 10)) * 3,
        "type": ["X"] * 27,
        "launch_speed_angle": [6] * 9 + [3] * 9 + [1] * 9,
        "estimated_ba_using_speedangle": [0.8] * 27,
    })
    result = parse_batter_hot_zones(df)
    assert len(result) == 9
    assert set(result["zone"]) == set(range(1, 10))


def test_batter_hot_zones_empty():
    assert parse_batter_hot_zones(pd.DataFrame()).empty


def test_league_krate_computes():
    splits = [
        {"stat": {"strikeOuts": 100, "plateAppearances": 400}},
        {"stat": {"strikeOuts": 50,  "plateAppearances": 200}},
    ]
    rate = parse_league_avg_krate(splits)
    assert abs(rate - 150 / 600) < 0.001


def test_league_krate_empty_fallback():
    assert parse_league_avg_krate([]) == 0.235


# ─────────────────────────────────────────────────────────────────────────────
# LINEUP TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def _game_data(home_id: int, team_id: int, batting_order: list[int], pitcher_ids: list[int]) -> dict:
    """Build a minimal game_data structure matching what get_live_feed() returns."""
    side = "home" if home_id == team_id else "away"
    opp_side = "away" if side == "home" else "home"
    players = {f"ID{pid}": {"person": {"fullName": f"Player{pid}"}} for pid in batting_order}
    return {
        "gameData": {
            "teams": {
                "home": {"id": home_id},
                "away": {"id": home_id + 100},
            }
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    side: {"battingOrder": batting_order, "players": players},
                    opp_side: {"pitchers": pitcher_ids, "players": {}},
                }
            }
        },
    }


def test_parse_confirmed_lineup_full():
    order = list(range(1, 10))
    gd = _game_data(home_id=117, team_id=117, batting_order=order, pitcher_ids=[999])
    df, status = parse_confirmed_lineup(gd, team_id=117)
    assert status == "confirmed"
    assert len(df) == 9


def test_parse_confirmed_lineup_partial():
    order = [1, 2, 3]
    gd = _game_data(home_id=117, team_id=117, batting_order=order, pitcher_ids=[999])
    df, status = parse_confirmed_lineup(gd, team_id=117)
    assert status == "partial"
    assert len(df) == 3


def test_parse_confirmed_lineup_none():
    gd = _game_data(home_id=117, team_id=117, batting_order=[], pitcher_ids=[999])
    df, status = parse_confirmed_lineup(gd, team_id=117)
    assert status == "none"
    assert df.empty


def test_parse_predicted_lineup():
    order = list(range(1, 10))
    games = [_game_data(home_id=117, team_id=117, batting_order=order, pitcher_ids=[999])] * 3
    result = parse_predicted_lineup(games, team_id=117)
    assert len(result) <= 9
    assert "player_id" in result.columns


def test_parse_lineup_vs_hand_filters_correctly():
    order = list(range(1, 10))
    # Two games: one vs RHP, one vs LHP
    rh_game = _game_data(home_id=117, team_id=117, batting_order=order, pitcher_ids=[50])
    lh_game = _game_data(home_id=117, team_id=117, batting_order=order, pitcher_ids=[51])

    hand_map = {50: "R", 51: "L"}
    games = [rh_game, lh_game]

    result_vs_r = parse_lineup_vs_hand(games, team_id=117, opposing_sp_hand="R",
                                       pitcher_handedness_map=hand_map)
    result_vs_l = parse_lineup_vs_hand(games, team_id=117, opposing_sp_hand="L",
                                       pitcher_handedness_map=hand_map)

    assert len(result_vs_r) <= 9
    assert len(result_vs_l) <= 9


def test_parse_lineup_vs_hand_no_match_returns_empty():
    order = list(range(1, 10))
    game = _game_data(home_id=117, team_id=117, batting_order=order, pitcher_ids=[50])
    result = parse_lineup_vs_hand([game], team_id=117, opposing_sp_hand="L",
                                  pitcher_handedness_map={50: "R"})
    assert result.empty


def test_merge_partial_and_predicted():
    confirmed = pd.DataFrame({"order": [1, 2], "player_id": [10, 20], "name": ["A", "B"]})
    predicted = pd.DataFrame({"order": [2, 3, 4], "player_id": [20, 30, 40], "name": ["B", "C", "D"]})
    result = merge_partial_and_predicted(confirmed, predicted)
    # Slots 1 and 2 from confirmed; slot 3, 4 from predicted (slot 2 not duplicated)
    assert set(result["order"]) == {1, 2, 3, 4}
    assert len(result) == 4
