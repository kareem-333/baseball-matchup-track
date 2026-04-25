# Layer 3: Lineup transforms.
# Pure functions: dict/list in, DataFrame out. No I/O, no API calls, no Streamlit.
# Input is raw boxscore/game data returned by MLBStatsSource.get_live_feed() or get_boxscore().

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def _side_for_team(game_data: dict, team_id: int) -> str:
    """Return 'home' or 'away' string for the given team in this game."""
    gd = game_data.get("gameData", {})
    teams = gd.get("teams", {})
    if teams.get("home", {}).get("id") == team_id:
        return "home"
    return "away"


def parse_confirmed_lineup(game_data: dict, team_id: int) -> tuple[pd.DataFrame, str]:
    """
    Extract confirmed lineup from a live game feed.

    Input:  game_data dict from MLBStatsSource.get_live_feed()
            team_id — the team whose lineup to extract
    Output: (DataFrame, status_str) where status is one of:
                'confirmed'  — full 9-player lineup posted
                'partial'    — some players listed but not 9
                'none'       — no lineup posted yet
            DataFrame columns: order, player_id, name
            (handedness is attached by the dashboard via get_batter_handedness)

    Expects: game_data["liveData"]["boxscore"]["teams"][side]["battingOrder"]
    """
    try:
        box = game_data.get("liveData", {}).get("boxscore", {})
        gd = game_data.get("gameData", {})
        home_id = gd.get("teams", {}).get("home", {}).get("id")
        side = "home" if home_id == team_id else "away"

        order = box.get("teams", {}).get(side, {}).get("battingOrder", [])
        players = box.get("teams", {}).get(side, {}).get("players", {})

        if not order:
            return pd.DataFrame(), "none"

        rows = [
            {
                "order": slot,
                "player_id": pid,
                "name": players.get(f"ID{pid}", {}).get("person", {}).get("fullName", str(pid)),
            }
            for slot, pid in enumerate(order[:9], 1)
        ]
        df = pd.DataFrame(rows)
        status = "confirmed" if len(df) >= 9 else "partial"
        return df, status
    except Exception:
        return pd.DataFrame(), "none"


def parse_predicted_lineup(completed_games: list[dict], team_id: int) -> pd.DataFrame:
    """
    Predict lineup from historical boxscore data.

    Input:  list of game_data dicts from MLBStatsSource.get_live_feed()
            team_id — the team to predict a lineup for
    Output: DataFrame with columns: order, player_id, name
            Rows are sorted by most common batting slot across input games.

    Assigns each player their most frequent batting slot. Deduplicates by slot.
    """
    pos_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    name_map: dict[int, str] = {}

    for game_data in completed_games:
        try:
            my_side = _side_for_team(game_data, team_id)
            box = game_data.get("liveData", {}).get("boxscore", {})
            order = box.get("teams", {}).get(my_side, {}).get("battingOrder", [])
            players = box.get("teams", {}).get(my_side, {}).get("players", {})
            for slot, pid in enumerate(order[:9], 1):
                pos_counts[pid][slot] += 1
                name_map[pid] = players.get(f"ID{pid}", {}).get("person", {}).get("fullName", str(pid))
        except Exception:
            continue

    if not pos_counts:
        return pd.DataFrame()

    rows = [
        {"order": max(slots, key=slots.get), "player_id": pid, "name": name_map.get(pid, str(pid))}
        for pid, slots in pos_counts.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("order")
        .drop_duplicates("order")
        .head(9)
        .reset_index(drop=True)
    )


def parse_lineup_vs_hand(
    completed_games: list[dict],
    team_id: int,
    opposing_sp_hand: str,
    pitcher_handedness_map: dict[int, str],
    batter_handedness_map: dict[int, str] | None = None,
    n_games: int = 5,
) -> pd.DataFrame:
    """
    Predict lineup from games played against a pitcher of a given handedness.

    Input:
        completed_games      — list of game_data dicts (must include liveData)
        team_id              — the team to predict for
        opposing_sp_hand     — 'R' or 'L' — filters games to matching SP handedness
        pitcher_handedness_map — {pitcher_player_id: hand} for all starting pitchers seen
        batter_handedness_map  — optional {player_id: hand} used to fill 'handedness' column
        n_games              — max matching games to use

    Output: DataFrame with columns:
        order, player_id, name, appearances, games_sampled, confidence_pct, handedness

    Confidence is appearances / n_matching (how often the player appeared in matching games).
    """
    matching_games = []
    for game_data in completed_games:
        if len(matching_games) >= n_games:
            break
        try:
            gd = game_data.get("gameData", {})
            home_id = gd.get("teams", {}).get("home", {}).get("id")
            my_side = "home" if home_id == team_id else "away"
            opp_side = "away" if my_side == "home" else "home"

            box = game_data.get("liveData", {}).get("boxscore", {})
            pitchers = box.get("teams", {}).get(opp_side, {}).get("pitchers", [])
            if not pitchers:
                continue

            opp_sp_id = pitchers[0]
            opp_sp_hand = pitcher_handedness_map.get(opp_sp_id)
            if opp_sp_hand != opposing_sp_hand:
                continue

            matching_games.append((game_data, my_side))
        except Exception:
            continue

    if not matching_games:
        return pd.DataFrame()

    pos_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    appearance_counts: dict[int, int] = defaultdict(int)
    name_map: dict[int, str] = {}

    for game_data, my_side in matching_games:
        box = game_data.get("liveData", {}).get("boxscore", {})
        order = box.get("teams", {}).get(my_side, {}).get("battingOrder", [])
        players = box.get("teams", {}).get(my_side, {}).get("players", {})
        for slot, pid in enumerate(order[:9], 1):
            pos_counts[pid][slot] += 1
            appearance_counts[pid] += 1
            name_map[pid] = players.get(f"ID{pid}", {}).get("person", {}).get("fullName", str(pid))

    if not pos_counts:
        return pd.DataFrame()

    n_matching = len(matching_games)
    batter_handedness_map = batter_handedness_map or {}

    rows = []
    for pid, slots in pos_counts.items():
        best_slot = max(slots, key=slots.get)
        appearances = appearance_counts[pid]
        rows.append({
            "order":          best_slot,
            "player_id":      pid,
            "name":           name_map.get(pid, str(pid)),
            "appearances":    appearances,
            "games_sampled":  n_matching,
            "confidence_pct": round((appearances / n_matching) * 100),
            "handedness":     batter_handedness_map.get(pid, "?"),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["order", "confidence_pct"], ascending=[True, False])
        .drop_duplicates("order")
        .head(9)
        .reset_index(drop=True)
    )


def merge_partial_and_predicted(confirmed: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Fill gaps in a partial confirmed lineup with predicted players.

    Input:  confirmed — partial confirmed lineup DataFrame
            predicted — predicted lineup DataFrame
    Output: merged DataFrame sorted by order, max 9 rows
    """
    filled_slots = set(confirmed["order"].tolist()) if not confirmed.empty else set()
    gap_predictions = predicted[~predicted["order"].isin(filled_slots)]
    combined = pd.concat([confirmed, gap_predictions], ignore_index=True)
    return combined.sort_values("order").head(9).reset_index(drop=True)
