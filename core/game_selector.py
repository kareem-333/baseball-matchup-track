"""
core/game_selector.py
Day picker (At Bat / On Deck / In the Hole) and game selection logic.
Single source of truth for which game the user is analyzing.
"""

import statsapi
import streamlit as st
from datetime import date, timedelta
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# DAY PICKER
# ─────────────────────────────────────────────────────────────────────────────

DAY_LABELS = {
    0: "⚾ At Bat (Today)",
    1: "🧤 On Deck (Tomorrow)",
    2: "👀 In the Hole (Day After)",
}


def get_day_from_offset(offset: int) -> date:
    """offset 0 = today, 1 = tomorrow, 2 = day after tomorrow."""
    return date.today() + timedelta(days=offset)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULE FETCH
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def get_games_for_date(target_date: date) -> list[dict]:
    """
    Return all MLB games scheduled for a given date.
    Each entry has: game_id, away_name, home_name, away_id, home_id,
    game_datetime, status, away_probable_pitcher, home_probable_pitcher,
    away_probable_pitcher_id, home_probable_pitcher_id.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        games = statsapi.schedule(sportId=1, start_date=date_str, end_date=date_str)
    except Exception:
        return []

    # statsapi schedule() returns probable pitcher NAMES but not IDs.
    # Fetch IDs from the game feed for each game — needed downstream for arsenal lookup.
    enriched = []
    for g in games:
        game_id = g.get("game_id")
        away_sp_id = None
        home_sp_id = None
        try:
            data = statsapi.get("game", {"gamePk": game_id})
            probables = data.get("gameData", {}).get("probablePitchers", {})
            away_sp_id = probables.get("away", {}).get("id")
            home_sp_id = probables.get("home", {}).get("id")
        except Exception:
            pass

        enriched.append({
            **g,
            "away_probable_pitcher_id": away_sp_id,
            "home_probable_pitcher_id": home_sp_id,
        })
    return enriched


def format_game_label(game: dict) -> str:
    """Used in the selectbox. Shows teams, SPs, and start time."""
    away = game.get("away_name", "?")
    home = game.get("home_name", "?")
    away_sp = game.get("away_probable_pitcher") or "TBD"
    home_sp = game.get("home_probable_pitcher") or "TBD"
    time = game.get("game_datetime", "")[:16].replace("T", " ")
    return f"{away} @ {home}  —  {away_sp} vs {home_sp}  ({time})"


# ─────────────────────────────────────────────────────────────────────────────
# SELECTED GAME STATE
# ─────────────────────────────────────────────────────────────────────────────

def render_day_and_game_selector():
    """
    Renders the day picker + game selector at the top of the app.
    Writes the selected game dict to st.session_state['selected_game'].
    Returns the selected game dict or None.
    """
    col_day, col_game = st.columns([1, 3])

    with col_day:
        offset = st.selectbox(
            "Day",
            options=[0, 1, 2],
            format_func=lambda x: DAY_LABELS[x],
            key="day_offset",
        )

    target_date = get_day_from_offset(offset)
    games = get_games_for_date(target_date)

    if not games:
        with col_game:
            st.info(f"No games scheduled for {target_date.strftime('%A, %B %d')}")
        st.session_state["selected_game"] = None
        return None

    with col_game:
        labels = [format_game_label(g) for g in games]
        sel_idx = st.selectbox(
            "Game",
            options=range(len(games)),
            format_func=lambda i: labels[i],
            key="game_select",
        )

    selected = games[sel_idx]
    st.session_state["selected_game"] = selected
    return selected
