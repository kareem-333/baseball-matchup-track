"""
core/handedness.py
Player handedness lookups. Cached aggressively since it almost never changes.
"""

import statsapi
import streamlit as st


@st.cache_data(ttl=86400, show_spinner=False)
def get_pitcher_handedness(pitcher_id: int) -> str:
    """Returns 'R', 'L', or '?' for a pitcher's throwing hand."""
    try:
        data = statsapi.get("person", {"personId": pitcher_id})
        people = data.get("people", [])
        if not people:
            return "?"
        hand = people[0].get("pitchHand", {}).get("code", "?")
        return hand if hand in ("R", "L") else "?"
    except Exception:
        return "?"


@st.cache_data(ttl=86400, show_spinner=False)
def get_batter_handedness(batter_id: int) -> str:
    """Returns 'R', 'L', 'S' (switch), or '?' for a batter's hitting hand."""
    try:
        data = statsapi.get("person", {"personId": batter_id})
        people = data.get("people", [])
        if not people:
            return "?"
        hand = people[0].get("batSide", {}).get("code", "?")
        return hand if hand in ("R", "L", "S") else "?"
    except Exception:
        return "?"


def handedness_badge_html(hand: str, role: str = "throws") -> str:
    """
    Returns HTML snippet for a colored handedness badge.
    role = 'throws' (pitcher) or 'hits' (batter).
    """
    colors = {"R": "#e74c3c", "L": "#3498db", "S": "#9b59b6", "?": "#888"}
    color = colors.get(hand, "#888")
    label = f"{role.capitalize()} {hand}"
    return (
        f'<span style="background:{color};color:white;border-radius:6px;'
        f'padding:2px 10px;font-size:0.75rem;font-weight:700">{label}</span>'
    )
