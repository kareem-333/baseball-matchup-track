"""
dashboard/components/matchup_cards.py — Reusable render helpers for the Matchup tab.

Functions here are pure Streamlit renderers: they receive data, emit HTML/widgets.
No data fetching or business logic.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mlb_season.pipeline import get_player_headshot_url, get_pitcher_sample_flag
from core.handedness import get_pitcher_handedness, handedness_badge_html


def _show_mlb_headshot(player_id: int, width: int = 56):
    url = get_player_headshot_url(player_id)
    st.markdown(
        f'<img src="{url}" width="{width}" height="{width}" '
        f'style="border-radius:50%;object-fit:cover;border:2px solid #2a2a2a">',
        unsafe_allow_html=True,
    )


def render_pitcher_header(pitcher_id: int, pitcher_name: str):
    """Pitcher headshot, name, handedness badge, sample flag."""
    hand = get_pitcher_handedness(pitcher_id)
    flag = get_pitcher_sample_flag(pitcher_id)

    col_hs, col_info = st.columns([1, 5])
    with col_hs:
        _show_mlb_headshot(pitcher_id, width=90)
    with col_info:
        badge_html = handedness_badge_html(hand, role="throws")
        warn_html = ""
        if flag["is_low_sample"]:
            warn_html = (
                f' <span style="background:#F39C12;color:white;border-radius:6px;'
                f'padding:2px 10px;font-size:0.75rem;font-weight:700">'
                f'⚠️ Low Sample: {flag["total_pitches"]} pitches</span>'
            )
        st.markdown(
            f"### {pitcher_name} &nbsp; {badge_html}{warn_html}",
            unsafe_allow_html=True,
        )
        if flag["is_low_sample"]:
            st.caption(f"⚠️ {flag['reason']}")


def render_leverage_summary(scored_lineup: list[dict], pitcher_name: str):
    """Headline insight: best and worst matchups vs this pitcher."""
    ranked = sorted(scored_lineup, key=lambda x: x["mash"], reverse=True)
    top3 = ranked[:3]
    bottom3 = ranked[-3:]

    top_str = " · ".join(
        [f"{p['name'].split()[-1]} ({p['mash']:.0f})" for p in top3]
    )
    bot_str = " · ".join(
        [f"{p['name'].split()[-1]} ({p['mash']:.0f})" for p in bottom3]
    )

    st.markdown(
        f"""
        <div style="background:linear-gradient(90deg,#1a2a4a 0%,#2a1a4a 100%);
                    border-left:4px solid #EB6E1F;padding:12px 18px;
                    border-radius:6px;margin-bottom:16px">
            <div style="font-weight:700;font-size:0.85rem;color:#EB6E1F;
                        margin-bottom:6px">⚡ LINEUP LEVERAGE vs {pitcher_name}</div>
            <div style="font-size:0.82rem;color:#e0e0e0">
                <b>Top MASH:</b> {top_str}<br>
                <b>Most at-risk:</b> {bot_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_matchup_cards(scored_lineup: list[dict], show_confidence: bool):
    """One card per hitter with MASH, MISS, driver pill, per-pitch breakdown."""
    for hitter in scored_lineup:
        mash = hitter["mash"]
        miss = hitter["miss"]
        name = hitter["name"]
        pid = hitter["player_id"]
        confidence = hitter.get("confidence_pct", 100)
        hitter_hand = hitter.get("handedness", "?")
        driver = hitter.get("primary_driver", {})

        if mash >= 65:
            mash_color = "#2ecc71"
            mash_tag = "Strong edge"
            mash_css = "mas-strong-advantage"
        elif mash >= 55:
            mash_color = "#95d35d"
            mash_tag = "Slight edge"
            mash_css = "mas-slight-advantage"
        elif mash >= 45:
            mash_color = "#f39c12"
            mash_tag = "Neutral"
            mash_css = "mas-slight-disadvantage"
        else:
            mash_color = "#e74c3c"
            mash_tag = "Disadvantage"
            mash_css = "mas-strong-disadvantage"

        miss_severity = "#e74c3c" if miss >= 60 else "#f39c12" if miss >= 45 else "#95d35d"

        conf_badge = ""
        if show_confidence and confidence < 100:
            conf_badge = (
                f' <span style="background:#2a2a4a;color:#ccc;border-radius:4px;'
                f'padding:1px 6px;font-size:0.7rem">🔮 {confidence}%</span>'
            )

        hand_badge = ""
        if hitter_hand in ("R", "L", "S"):
            hand_colors = {"R": "#e74c3c", "L": "#3498db", "S": "#9b59b6"}
            hand_badge = (
                f' <span style="background:{hand_colors[hitter_hand]};'
                f'color:white;border-radius:4px;padding:1px 6px;'
                f'font-size:0.7rem;font-weight:700">{hitter_hand}HB</span>'
            )

        driver_line = ""
        if driver:
            driver_line = (
                f'<span class="driver-pill">Driver: {driver.get("pitch_label", "?")} '
                f'({driver.get("usage_pct", 0):.0f}% usage · '
                f'{driver.get("contribution_pct", 0):.0f}% of MASH)</span>'
            )

        headshot = get_player_headshot_url(pid)

        st.markdown(f"""
<div class="mas-card">
  <img src="{headshot}" width="56" height="56"
       style="border-radius:50%;object-fit:cover;border:2px solid #2a2a2a">
  <div class="mas-card-body">
    <p class="mas-player-name">
      {hitter.get('order', '?')}. {name}{hand_badge}{conf_badge}
    </p>
    <div class="mas-bar-row">
      <div class="mas-bar-container">
        <div class="mas-bar-fill {mash_css}" style="width:{mash}%"></div>
      </div>
      <span class="mas-number" style="color:{mash_color}">{mash:.0f}</span>
      <span style="font-size:0.75rem;color:{miss_severity};font-weight:700">
        MISS {miss:.0f}
      </span>
    </div>
    <div class="mas-meta-row">
      <span style="font-size:0.72rem;color:#aaa">{mash_tag}</span>
      {driver_line}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        with st.expander(f"↳ {name} — Details", expanded=False):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.markdown(
                    f"""<div style="border:2px solid {mash_color};border-radius:8px;
                                   padding:8px;text-align:center">
                        <div style="font-size:0.72rem;color:#888">MASH</div>
                        <div style="font-size:2rem;font-weight:800;color:{mash_color}">
                            {mash:.0f}</div>
                        <div style="font-size:0.7rem;color:#aaa">{mash_tag}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div style="border:2px solid {miss_severity};border-radius:8px;
                                   padding:8px;text-align:center">
                        <div style="font-size:0.72rem;color:#888">MISS</div>
                        <div style="font-size:2rem;font-weight:800;color:{miss_severity}">
                            {miss:.0f}</div>
                        <div style="font-size:0.7rem;color:#aaa">Whiff risk</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                if driver:
                    st.markdown(
                        f"**Primary driver:** {driver.get('pitch_label', '?')}"
                        f" — {driver.get('usage_pct', 0):.0f}% usage, "
                        f"{driver.get('contribution_pct', 0):.0f}% of MASH"
                    )
                if hitter.get("low_sample_warning"):
                    st.caption(f"⚠️ {hitter['low_sample_warning']}")

            breakdown = hitter.get("per_pitch_breakdown", pd.DataFrame())
            if not breakdown.empty:
                cols_needed = [
                    c for c in [
                        "pitch_label", "usage_pct", "barrel_rate", "whiff_rate",
                        "z_barrel", "z_whiff", "z_freq", "barrel_overlap", "whiff_overlap",
                    ] if c in breakdown.columns
                ]
                if cols_needed:
                    st.markdown("**Per-pitch breakdown**")
                    show_df = breakdown[cols_needed].rename(columns={
                        "pitch_label": "Pitch",
                        "usage_pct": "Usage",
                        "barrel_rate": "Barrel%",
                        "whiff_rate": "Whiff%",
                        "z_barrel": "z(B)",
                        "z_whiff": "z(W)",
                        "z_freq": "z(Freq)",
                        "barrel_overlap": "MASH contrib",
                        "whiff_overlap": "MISS contrib",
                    }).round(3)
                    st.dataframe(show_df, use_container_width=True, hide_index=True)


def merge_partial_and_predicted(
    confirmed: pd.DataFrame,
    predicted: pd.DataFrame,
) -> pd.DataFrame:
    """Fills gaps in a partial confirmed lineup with predicted players."""
    filled_slots = set(confirmed["order"].tolist()) if not confirmed.empty else set()
    gap_predictions = predicted[~predicted["order"].isin(filled_slots)]
    combined = pd.concat([confirmed, gap_predictions], ignore_index=True)
    return combined.sort_values("order").head(9).reset_index(drop=True)
