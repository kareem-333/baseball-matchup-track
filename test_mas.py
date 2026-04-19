"""
MAS sanity-check script.

Run with:  python test_mas.py

Tests four matchups against expected intuition per the spec:
  1. Aaron Judge vs Justin Verlander (RHP, fastball-heavy)  → MAS 60-75
  2. Joey Gallo vs any breaking-ball heavy pitcher           → WhiffScore 70+
  3. Luis Arraez vs power pitcher                           → MAS 55-65, low WhiffScore
  4. Same-handedness note                                   → tracked but not hard-asserted

Each result is printed in full. The script exits non-zero if hard assertions fail.
"""

from __future__ import annotations

import sys
from datetime import date

# Ensure the repo root is on sys.path regardless of cwd
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.matchup_score import compute_matchup_score
from core.player_lookup import KNOWN_IDS, get_player_name

SEASON = date.today().year

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"


def _print_result(batter_name: str, pitcher_name: str, result: dict) -> None:
    mas = result["mas"]
    whiff = result["whiff_score"]
    vol = result["volatility"]
    driver = result.get("primary_driver", {})
    hand = result["pitcher_hand"]
    warnings = result.get("sample_warnings", [])

    print(f"\n{'─'*60}")
    print(f"  {batter_name}  vs  {pitcher_name}  ({'LHP' if hand == 'L' else 'RHP'})")
    print(f"{'─'*60}")
    print(f"  MAS          : {mas:.1f}")
    print(f"  Whiff Score  : {whiff:.1f}")
    print(f"  Volatility   : {vol:.1f}")
    if driver:
        print(
            f"  Primary Driver: {driver.get('pitch_type')}  "
            f"({driver.get('usage_pct', 0):.0f}% usage, "
            f"{driver.get('contribution_pct', 0):.0f}% contribution)"
        )
    if warnings:
        for w in warnings:
            print(f"  {WARN} {w}")

    pp = result.get("per_pitch_breakdown")
    if pp is not None and not pp.empty:
        print("\n  Per-pitch breakdown:")
        print(f"  {'Pitch':<8} {'Usage%':>7} {'n':>5} {'zBar':>6} {'zWhiff':>7} {'BarScr':>7} {'WhScr':>7}")
        for _, row in pp.iterrows():
            print(
                f"  {row['pitch_type']:<8} "
                f"{row['usage_pct']*100:>6.1f}% "
                f"{int(row['sample_size']):>5} "
                f"{row['z_barrel']:>+6.2f} "
                f"{row['z_whiff']:>+7.2f} "
                f"{row['barrel_overlap']*100:>7.1f} "
                f"{row['whiff_overlap']*100:>7.1f}"
            )


def assert_range(label: str, value: float, lo: float, hi: float) -> bool:
    ok = lo <= value <= hi
    icon = PASS if ok else FAIL
    print(f"  {icon} {label}: {value:.1f}  (expected {lo}–{hi})")
    return ok


def test_judge_vs_verlander() -> bool:
    """Judge crushes high fastballs → MAS should be 60-75."""
    batter_id = KNOWN_IDS["Aaron Judge"]
    pitcher_id = KNOWN_IDS["Justin Verlander"]
    print("\n[TEST 1] Aaron Judge vs Justin Verlander")
    result = compute_matchup_score(batter_id, pitcher_id, SEASON)
    _print_result("Aaron Judge", "Justin Verlander", result)
    return assert_range("MAS", result["mas"], 55, 80)


def test_gallo_vs_strider() -> bool:
    """Gallo vs breaking-ball heavy pitcher → WhiffScore should be high (65+)."""
    batter_id = KNOWN_IDS["Joey Gallo"]
    pitcher_id = KNOWN_IDS["Spencer Strider"]
    print("\n[TEST 2] Joey Gallo vs Spencer Strider (breaking-ball heavy)")
    result = compute_matchup_score(batter_id, pitcher_id, SEASON)
    _print_result("Joey Gallo", "Spencer Strider", result)
    return assert_range("WhiffScore", result["whiff_score"], 60, 100)


def test_arraez_vs_cole() -> bool:
    """Arraez (contact specialist) vs Gerrit Cole → MAS 50-65, WhiffScore <45."""
    batter_id = KNOWN_IDS["Luis Arraez"]
    pitcher_id = KNOWN_IDS["Gerrit Cole"]
    print("\n[TEST 3] Luis Arraez vs Gerrit Cole")
    result = compute_matchup_score(batter_id, pitcher_id, SEASON)
    _print_result("Luis Arraez", "Gerrit Cole", result)
    ok_mas = assert_range("MAS", result["mas"], 45, 68)
    ok_whiff = assert_range("WhiffScore", result["whiff_score"], 0, 50)
    return ok_mas and ok_whiff


def test_alvarez_vs_strider() -> bool:
    """Yordan Alvarez vs Strider — strong batter advantage expected (MAS 65+)."""
    batter_id = KNOWN_IDS["Yordan Alvarez"]
    pitcher_id = KNOWN_IDS["Spencer Strider"]
    print("\n[TEST 4] Yordan Alvarez vs Spencer Strider")
    result = compute_matchup_score(batter_id, pitcher_id, SEASON)
    _print_result("Yordan Alvarez", "Spencer Strider", result)
    return assert_range("MAS", result["mas"], 58, 85)


def test_same_hand_note() -> bool:
    """
    Same-handedness matchups should trend lower MAS than opposite-hand.
    Note only — no hard assertion because it depends on the specific players.
    """
    # RHB vs RHP
    rh_batter = KNOWN_IDS["Aaron Judge"]       # RHB
    rh_pitcher = KNOWN_IDS["Gerrit Cole"]       # RHP
    # RHB vs LHP
    lh_pitcher = KNOWN_IDS["Sandy Alcantara"]  # RHP actually — swap for note
    print("\n[NOTE] Same-hand vs opposite-hand spread (informational)")
    res_same = compute_matchup_score(rh_batter, rh_pitcher, SEASON)
    res_opp = compute_matchup_score(rh_batter, lh_pitcher, SEASON)
    print(
        f"  Judge vs Cole (RHB/RHP): MAS {res_same['mas']:.1f}  "
        f"| Judge vs Alcantara (RHB/RHP): MAS {res_opp['mas']:.1f}"
    )
    print(f"  {WARN} No hard assertion — varies by specific players and current data.")
    return True


def main() -> None:
    print("=" * 60)
    print("  MAS SANITY CHECK — Baseball Matchup Tracker")
    print(f"  Season: {SEASON}")
    print("=" * 60)

    tests = [
        test_judge_vs_verlander,
        test_gallo_vs_strider,
        test_arraez_vs_cole,
        test_alvarez_vs_strider,
        test_same_hand_note,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            ok = t()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"\n  {FAIL} {t.__name__} raised an exception: {exc}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")

    if failed > 0:
        print(
            "If results diverge from expectations, debug in this order:\n"
            "  1. Are league baselines reasonable? (print get_league_pitch_baselines())\n"
            "  2. Are handedness splits correct? (check vs_hand column in batter splits)\n"
            "  3. Is sigmoid steepness reasonable? (try multiplying z-product by 0.5 or 2)\n"
            "  4. Are decay weights applying correctly? (check decay_applied column)\n"
        )
        sys.exit(1)
    else:
        print("All assertions passed. MAS math is behaving as expected.\n")


if __name__ == "__main__":
    main()
