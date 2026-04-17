"""
core/headshots.py — Headshot fetching for both MLB and NBA players.

MLB: securea.mlb.com CDN, fetched server-side as base64 to avoid
     hot-link blocks in Streamlit.
NBA: cdn.nba.com, high-res (1040×760) direct URL.
"""

from __future__ import annotations
import base64
import io
import requests

# In-memory cache: player_id → base64 data-URI | None
_mlb_hs_cache: dict[int, str | None] = {}


def headshot_url(player_id: int) -> str:
    """Working MLB CDN URL for a player headshot (securea.mlb.com)."""
    return f"https://securea.mlb.com/mlb/images/players/head_shot/{player_id}.jpg"


def headshot_b64(player_id: int, size: int = 80) -> str | None:
    """
    Fetch an MLB headshot and return a base64 data-URI so Streamlit
    can display it without relying on CDN hot-link permissions.
    Returns None if the image cannot be fetched.
    """
    from PIL import Image  # lazy import — not needed at module load time

    if player_id in _mlb_hs_cache:
        return _mlb_hs_cache[player_id]

    try:
        url = headshot_url(player_id)
        r   = requests.get(
            url, timeout=6,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.mlb.com/"},
        )
        if r.status_code != 200:
            _mlb_hs_cache[player_id] = None
            return None

        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.thumbnail((size * 3, size * 3), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        _mlb_hs_cache[player_id] = b64
        return b64
    except Exception:
        _mlb_hs_cache[player_id] = None
        return None


def nba_headshot_url(player_id: int) -> str:
    """High-resolution NBA headshot URL (1040×760) from cdn.nba.com."""
    return (
        f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    )
