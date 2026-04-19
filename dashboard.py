"""
Baseball Matchup Tracker — Streamlit Dashboard
Four tabs: Today · Matchup · Trends · History
MAS (Matchup Advantage Score) is the headline metric.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Baseball Matchup Tracker",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── consolidated CSS ──────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── base ── */
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stHeader"] { background: #0e1117; }
h1, h2, h3 { color: #fafafa; }
p, li { color: #ccc; }

/* ── MAS card ── */
.mas-card {
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    background: #0e1117;
    display: flex;
    align-items: flex-start;
    gap: 14px;
}
.mas-card-body { flex: 1; }
.mas-player-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #fafafa;
    margin: 0;
}
.mas-matchup-line {
    font-size: 0.82rem;
    color: #888;
    margin: 2px 0 6px;
}
.mas-bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.mas-bar-container {
    flex: 1;
    height: 10px;
    background: #2a2a2a;
    border-radius: 6px;
    overflow: hidden;
}
.mas-bar-fill { height: 100%; transition: width 0.3s ease; }
.mas-strong-advantage    { background: #2ecc71; }
.mas-slight-advantage    { background: #95d35d; }
.mas-slight-disadvantage { background: #f39c12; }
.mas-strong-disadvantage { background: #e74c3c; }
.mas-number {
    font-size: 1.1rem;
    font-weight: 800;
    min-width: 38px;
    text-align: right;
    color: #fafafa;
}
.mas-meta-row {
    display: flex;
    gap: 16px;
    margin-top: 4px;
}
.mas-whiff-badge {
    font-size: 0.78rem;
    color: #aaa;
}
.driver-pill {
    font-size: 0.78rem;
    color: #EB6E1F;
}
.volatile-tag {
    font-size: 0.7rem;
    background: #3a2a1a;
    color: #f39c12;
    padding: 1px 6px;
    border-radius: 4px;
    margin-left: 6px;
}

/* ── Lineup Leverage banner ── */
.leverage-banner {
    background: linear-gradient(90deg, #1a2a4a 0%, #2a1a4a 100%);
    border-left: 4px solid #EB6E1F;
    padding: 12px 18px;
    border-radius: 6px;
    margin-bottom: 16px;
}
.leverage-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #EB6E1F;
    margin-bottom: 4px;
    letter-spacing: 0.06em;
}
.leverage-line {
    font-size: 0.88rem;
    color: #ddd;
    margin: 2px 0;
}

/* ── score header (live) ── */
.score-header {
    text-align: center;
    padding: 16px 0 8px;
}
.score-teams {
    font-size: 1.4rem;
    font-weight: 700;
    color: #fafafa;
    letter-spacing: 0.04em;
}
.score-numbers {
    font-size: 2.8rem;
    font-weight: 900;
    color: #EB6E1F;
    line-height: 1.1;
}
.score-situation {
    font-size: 0.88rem;
    color: #888;
    margin-top: 4px;
}

/* ── current batter MAS chip (live tab) ── */
.live-mas-chip {
    background: #16213e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.live-mas-label {
    font-size: 0.78rem;
    color: #888;
    margin-bottom: 2px;
}
.live-mas-value {
    font-size: 1.6rem;
    font-weight: 900;
}

/* ── tab styling ── */
[data-testid="stTabs"] button {
    font-size: 0.95rem;
    font-weight: 600;
    color: #aaa;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #EB6E1F;
    border-bottom-color: #EB6E1F;
}

/* ── generic divider ── */
hr { border-color: #2a2a2a; margin: 12px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ── imports (after page config) ───────────────────────────────────────────────

from core.matchup_score import (
    compute_matchup_score,
    mas_color,
    mas_css_class,
    mas_label,
)
from core.player_lookup import (
    get_player_info,
    get_player_name,
    get_pitcher_handedness,
    get_team_roster,
)
from core.visualizations import (
    plot_career_pitch_splits,
    plot_hot_zone_grid,
    plot_per_pitch_breakdown,
    show_pitch_mix_simulator,
)
from mlb_season.pipeline import (
    get_game_lineup,
    get_live_game_data,
    get_pitcher_arsenal,
    get_player_headshot_url,
    get_probable_lineups,
    get_todays_games,
)


# ── session state defaults ────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "selected_game_pk": None,
        "selected_pitcher_id": None,
        "selected_team_id": None,
        "active_tab": "Today",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='margin-bottom:0'>⚾ Baseball Matchup Tracker</h1>"
    f"<p style='color:#888;margin-top:2px'>{date.today().strftime('%A, %B %-d, %Y')}</p>",
    unsafe_allow_html=True,
)

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_today, tab_matchup, tab_trends, tab_history = st.tabs(
    ["🔴 Today", "🎯 Matchup", "📈 Trends", "📋 History"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TODAY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_today:
    games = get_todays_games()

    if not games:
        st.info("No games scheduled today.")
    else:
        # ── game selector ─────────────────────────────────────────────────────
        def _game_label(g: dict) -> str:
            score = ""
            if g["status"] in ("In Progress", "Final", "Game Over"):
                score = f"  {g['away_score']}–{g['home_score']}"
            status_icon = {
                "In Progress": "🔴",
                "Final": "✅",
                "Game Over": "✅",
                "Pre-Game": "🕐",
                "Warmup": "🕐",
                "Preview": "📅",
            }.get(g["status"], "📅")
            return f"{status_icon}  {g['away_team']}  @  {g['home_team']}{score}"

        game_labels = [_game_label(g) for g in games]
        sel_idx = 0
        if st.session_state.selected_game_pk:
            for i, g in enumerate(games):
                if g["game_pk"] == st.session_state.selected_game_pk:
                    sel_idx = i
                    break

        chosen_idx = st.selectbox(
            "Select game", range(len(games)),
            format_func=lambda i: game_labels[i],
            index=sel_idx,
            key="today_game_sel",
        )
        game = games[chosen_idx]
        st.session_state.selected_game_pk = game["game_pk"]

        is_live = game["status"] in ("In Progress", "Warmup", "Pre-Game")
        is_final = "Final" in game["status"] or "Game Over" in game["status"]

        # ── score header ──────────────────────────────────────────────────────
        if is_live or is_final:
            live = get_live_game_data(game["game_pk"])
            away_score = live.get("away_score", game["away_score"])
            home_score = live.get("home_score", game["home_score"])
            inning = live.get("inning", game.get("inning", 0))
            inning_state = live.get("inning_state", game.get("inning_state", ""))
            situation = (
                f"{inning_state} {inning}" if inning else game["status"]
            )
        else:
            away_score = home_score = "–"
            situation = game.get("game_datetime", game["status"])
            live = {}

        st.markdown(
            f"""
<div class="score-header">
  <div class="score-teams">{game['away_team']} @ {game['home_team']}</div>
  <div class="score-numbers">{away_score} – {home_score}</div>
  <div class="score-situation">{situation}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        # ── current at-bat MAS chip (live only) ───────────────────────────────
        if is_live and live.get("current_batter_id") and live.get("current_pitcher_id"):
            bat_id = live["current_batter_id"]
            pit_id = live["current_pitcher_id"]
            bat_name = live.get("current_batter_name", get_player_name(bat_id))
            pit_name = live.get("current_pitcher_name", get_player_name(pit_id))

            with st.spinner("Computing live MAS…"):
                result = compute_matchup_score(bat_id, pit_id)

            mas = result["mas"]
            whiff = result["whiff_score"]
            css_cls = mas_css_class(mas)
            color = mas_color(mas)

            st.markdown(
                f"""
<div class="live-mas-chip">
  <div>
    <div class="live-mas-label">Current at-bat</div>
    <div style="color:#fafafa;font-weight:700">{bat_name} vs {pit_name}</div>
  </div>
  <div>
    <div class="live-mas-label">MAS</div>
    <div class="live-mas-value" style="color:{color}">{mas}</div>
  </div>
  <div>
    <div class="live-mas-label">Whiff Risk</div>
    <div class="live-mas-value" style="color:#888">{whiff}</div>
  </div>
  <div style="flex:1">
    <div class="mas-bar-container">
      <div class="mas-bar-fill {css_cls}" style="width:{mas}%"></div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        # ── box score detail (expandable) ──────────────────────────────────────
        with st.expander("📊 Box Score Detail", expanded=False):
            if is_live or is_final:
                linescore = live.get("linescore", {})
                innings_data = linescore.get("innings", [])
                if innings_data:
                    innings_cols = [str(i["num"]) for i in innings_data]
                    away_runs = [i.get("away", {}).get("runs", "") for i in innings_data]
                    home_runs = [i.get("home", {}).get("runs", "") for i in innings_data]
                    ls_df = pd.DataFrame(
                        {
                            "Team": [game["away_team"], game["home_team"]],
                            **{
                                col: [a, h]
                                for col, a, h in zip(innings_cols, away_runs, home_runs)
                            },
                            "R": [
                                linescore.get("teams", {}).get("away", {}).get("runs", ""),
                                linescore.get("teams", {}).get("home", {}).get("runs", ""),
                            ],
                            "H": [
                                linescore.get("teams", {}).get("away", {}).get("hits", ""),
                                linescore.get("teams", {}).get("home", {}).get("hits", ""),
                            ],
                            "E": [
                                linescore.get("teams", {}).get("away", {}).get("errors", ""),
                                linescore.get("teams", {}).get("home", {}).get("errors", ""),
                            ],
                        }
                    )
                    st.dataframe(ls_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Linescore not yet available.")
            else:
                st.info("Box score will be available once the game starts.")

        # ── probable pitchers / navigate to matchup ──────────────────────────
        st.markdown("---")
        col_away, col_home = st.columns(2)

        with col_away:
            st.markdown(f"**{game['away_team']}**")
            away_pitcher = game.get("away_probable_pitcher", "TBD")
            away_pitcher_id = game.get("away_probable_pitcher_id")
            st.markdown(f"🔵 SP: {away_pitcher}")
            if away_pitcher_id and st.button(
                f"Analyze vs {away_pitcher}", key="btn_away_pitcher"
            ):
                st.session_state.selected_pitcher_id = away_pitcher_id
                st.session_state.selected_team_id = game["home_id"]

        with col_home:
            st.markdown(f"**{game['home_team']}**")
            home_pitcher = game.get("home_probable_pitcher", "TBD")
            home_pitcher_id = game.get("home_probable_pitcher_id")
            st.markdown(f"🟠 SP: {home_pitcher}")
            if home_pitcher_id and st.button(
                f"Analyze vs {home_pitcher}", key="btn_home_pitcher"
            ):
                st.session_state.selected_pitcher_id = home_pitcher_id
                st.session_state.selected_team_id = game["away_id"]

        if st.session_state.selected_pitcher_id:
            st.success(
                f"Pitcher selected — switch to the **🎯 Matchup** tab to see MAS scores."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MATCHUP
# ═══════════════════════════════════════════════════════════════════════════════

with tab_matchup:

    # ── pitcher selector (compact) ────────────────────────────────────────────
    st.markdown("#### Select Pitcher")

    pitcher_search = st.text_input(
        "Search pitcher by name",
        placeholder="e.g. Spencer Strider",
        key="pitcher_search",
    )

    selected_pitcher_id: int | None = st.session_state.selected_pitcher_id

    if pitcher_search:
        from core.player_lookup import search_players
        results = search_players(pitcher_search)
        pitchers = [r for r in results if r["position"] in ("P", "SP", "RP")]
        if pitchers:
            labels = [
                f"{r['full_name']} — {r['team']}"
                for r in pitchers
            ]
            chosen = st.selectbox("Match", labels, key="pitcher_search_sel")
            chosen_idx = labels.index(chosen)
            selected_pitcher_id = pitchers[chosen_idx]["player_id"]
            st.session_state.selected_pitcher_id = selected_pitcher_id
        else:
            st.warning("No pitchers found. Try a different spelling.")

    if selected_pitcher_id:
        pitcher_info = get_player_info(selected_pitcher_id)
        pitcher_hand = get_pitcher_handedness(selected_pitcher_id)

        st.markdown(
            f"**{pitcher_info['full_name']}** &nbsp;·&nbsp; {pitcher_info['team_name']} "
            f"&nbsp;·&nbsp; {'LHP' if pitcher_hand == 'L' else 'RHP'}",
        )

        # ── pitcher arsenal detail (collapsible) ──────────────────────────────
        with st.expander("📋 Pitcher Arsenal Detail", expanded=False):
            arsenal = get_pitcher_arsenal(selected_pitcher_id)
            if not arsenal.empty:
                disp = arsenal[
                    ["pitch_label", "usage_pct", "count", "avg_velocity"]
                ].copy()
                disp["usage_pct"] = (disp["usage_pct"] * 100).round(1).astype(str) + "%"
                disp.columns = ["Pitch", "Usage %", "Count", "Avg Velo (mph)"]
                st.dataframe(disp, use_container_width=True, hide_index=True)
            else:
                st.info("No arsenal data available.")

        st.markdown("---")

        # ── batter lineup source ──────────────────────────────────────────────
        st.markdown("#### Lineup")
        lineup_source = st.radio(
            "Load lineup from",
            ["Today's game", "Manual team roster", "Enter player IDs"],
            horizontal=True,
            key="lineup_source",
        )

        batters: list[dict] = []

        if lineup_source == "Today's game":
            if st.session_state.selected_game_pk and st.session_state.selected_team_id:
                batters = get_game_lineup(
                    st.session_state.selected_game_pk,
                    st.session_state.selected_team_id,
                )
                if not batters:
                    st.info("Lineup not yet posted — try 'Manual team roster'.")
            else:
                st.info("Select a game and team in the **Today** tab first.")

        elif lineup_source == "Manual team roster":
            team_search = st.text_input("Team name or abbreviation", key="team_search_matchup")
            if team_search:
                from core.player_lookup import search_players
                team_results = search_players(team_search)
                if team_results:
                    team_ids = list({r.get("player_id") for r in team_results})
                    # Get team via first result's team_id
                    first_info = get_player_info(team_results[0]["player_id"])
                    if first_info.get("team_id"):
                        roster = get_team_roster(first_info["team_id"])
                        batters = [
                            r for r in roster
                            if r["position"] not in ("P", "SP", "RP")
                        ]

        else:  # Manual entry
            id_input = st.text_input(
                "Enter MLBAM player IDs, comma-separated",
                placeholder="592450, 670541, 514888",
                key="manual_ids_input",
            )
            if id_input:
                try:
                    ids = [int(x.strip()) for x in id_input.split(",") if x.strip()]
                    batters = [
                        {
                            "player_id": pid,
                            "name": get_player_name(pid),
                            "position": get_player_info(pid).get("position", ""),
                        }
                        for pid in ids
                    ]
                except ValueError:
                    st.error("Invalid player IDs — enter comma-separated integers.")

        # ── compute MAS for all batters ───────────────────────────────────────
        if batters:
            season = date.today().year

            with st.spinner(f"Computing MAS for {len(batters)} batters…"):
                results: list[dict] = []
                for b in batters:
                    r = compute_matchup_score(b["player_id"], selected_pitcher_id, season)
                    results.append(
                        {
                            "player_id": b["player_id"],
                            "name": b.get("name", get_player_name(b["player_id"])),
                            "position": b.get("position", ""),
                            **r,
                        }
                    )

            # ── Lineup Leverage banner ────────────────────────────────────────
            sorted_results = sorted(results, key=lambda x: x["mas"], reverse=True)
            top3 = sorted_results[:3]
            bot3 = sorted_results[-3:][::-1]

            top_str = " · ".join(
                f"{r['name'].split()[-1]} ({r['mas']:.0f})" for r in top3
            )
            bot_str = " · ".join(
                f"{r['name'].split()[-1]} ({r['mas']:.0f})" for r in bot3
            )

            st.markdown(
                f"""
<div class="leverage-banner">
  <div class="leverage-title">⚡ LINEUP LEVERAGE</div>
  <div class="leverage-line">🟢 Top advantages tonight: {top_str}</div>
  <div class="leverage-line">🔴 Top risks: {bot_str}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            # ── headline MAS grid ─────────────────────────────────────────────
            for r in results:
                pid = r["player_id"]
                mas = r["mas"]
                whiff = r["whiff_score"]
                vol = r["volatility"]
                driver = r.get("primary_driver", {})
                css_cls = mas_css_class(mas)
                color = mas_color(mas)
                headshot_url = get_player_headshot_url(pid)

                volatile_tag = (
                    '<span class="volatile-tag">Boom/Bust</span>'
                    if vol > 20 else ""
                )

                driver_line = ""
                if driver:
                    pt_label = driver.get("pitch_type", "")
                    usage = driver.get("usage_pct", 0)
                    contrib = driver.get("contribution_pct", 0)
                    driver_line = (
                        f'<span class="driver-pill">'
                        f"Driver: {pt_label} ({usage:.0f}% usage · "
                        f"{contrib:.0f}% contribution)"
                        f"</span>"
                    )

                pos_badge = (
                    f'<span style="font-size:0.75rem;color:#666">'
                    f"{r.get('position', '')} · {r['name']}</span>"
                )

                st.markdown(
                    f"""
<div class="mas-card">
  <img src="{headshot_url}" width="56" height="56"
       style="border-radius:50%;object-fit:cover;border:2px solid #2a2a2a">
  <div class="mas-card-body">
    <p class="mas-player-name">{r['name']}</p>
    <p class="mas-matchup-line">
      vs {pitcher_info['full_name']} &nbsp;({'LHP' if pitcher_hand == 'L' else 'RHP'})
    </p>
    <div class="mas-bar-row">
      <div class="mas-bar-container">
        <div class="mas-bar-fill {css_cls}" style="width:{mas}%"></div>
      </div>
      <span class="mas-number" style="color:{color}">{mas:.0f}</span>
      {volatile_tag}
    </div>
    <div class="mas-meta-row">
      <span class="mas-whiff-badge">Whiff Risk: {whiff:.0f}</span>
      {driver_line}
    </div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

                # ── dig deeper expander ───────────────────────────────────────
                with st.expander(f"↳ Dig Deeper — {r['name']}", expanded=False):
                    warn = r.get("sample_warnings", [])
                    if warn:
                        for w in warn:
                            st.warning(w)

                    dig_tab1, dig_tab2, dig_tab3, dig_tab4 = st.tabs(
                        ["Per-Pitch Breakdown", "Hot Zone", "Career Splits", "Pitch Mix Simulator"]
                    )

                    with dig_tab1:
                        per_pitch_df = r.get("per_pitch_breakdown", pd.DataFrame())
                        if not per_pitch_df.empty:
                            st.plotly_chart(
                                plot_per_pitch_breakdown(per_pitch_df),
                                use_container_width=True,
                                key=f"breakdown_{pid}",
                            )
                        else:
                            st.info("No per-pitch data.")

                    with dig_tab2:
                        st.plotly_chart(
                            plot_hot_zone_grid(pid, season),
                            use_container_width=True,
                            key=f"hotzone_{pid}",
                        )

                    with dig_tab3:
                        st.plotly_chart(
                            plot_career_pitch_splits(pid, season),
                            use_container_width=True,
                            key=f"splits_{pid}",
                        )

                    with dig_tab4:
                        show_pitch_mix_simulator(
                            selected_pitcher_id, pid, season
                        )

        elif lineup_source != "Today's game":
            st.info("Add batters above to compute MAS scores.")

    else:
        st.info(
            "Search for a pitcher above, or select one from the **Today** tab."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRENDS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_trends:
    st.markdown("### MAS Trend — Batter over Time")
    st.info(
        "Coming soon: track how a batter's MAS vs a specific pitcher evolves across "
        "a rolling 30-day window as the pitcher's arsenal usage shifts."
    )

    st.markdown("---")
    st.markdown("### League Pitch Baseline Trends")
    st.info(
        "Coming soon: chart how barrel_rate_mean and whiff_rate_mean for each "
        "pitch type move across the season — contextualizes whether a MAS of 65 "
        "means the same thing in April vs September."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown("### Last 3 Games — Batter Performance")

    batter_hist_search = st.text_input(
        "Search batter", placeholder="e.g. Yordan Alvarez", key="hist_batter_search"
    )

    if batter_hist_search:
        from core.player_lookup import search_players
        hist_results = search_players(batter_hist_search)
        non_pitchers = [r for r in hist_results if r["position"] not in ("P", "SP", "RP")]

        if non_pitchers:
            hist_labels = [f"{r['full_name']} — {r['team']}" for r in non_pitchers]
            hist_chosen = st.selectbox("Select batter", hist_labels, key="hist_batter_sel")
            hist_idx = hist_labels.index(hist_chosen)
            hist_batter_id = non_pitchers[hist_idx]["player_id"]

            season = date.today().year
            splits = None
            try:
                from mlb_season.pipeline import get_batter_pitch_splits
                splits = get_batter_pitch_splits(hist_batter_id, season)
            except Exception:
                pass

            if splits is not None and not splits.empty:
                st.markdown("#### Season Pitch-Type Splits (vs LHP / RHP)")
                disp = splits[[
                    "pitch_label", "vs_hand", "barrel_rate", "whiff_rate", "sample_size"
                ]].copy()
                disp["barrel_rate"] = (disp["barrel_rate"] * 100).round(2).astype(str) + "%"
                disp["whiff_rate"] = (disp["whiff_rate"] * 100).round(2).astype(str) + "%"
                disp.columns = ["Pitch", "vs Hand", "Barrel %", "Whiff %", "Pitches Seen"]
                st.dataframe(disp, use_container_width=True, hide_index=True)

                st.markdown("#### Career Pitch Splits Chart")
                st.plotly_chart(
                    plot_career_pitch_splits(hist_batter_id, season),
                    use_container_width=True,
                    key="hist_career_splits",
                )
            else:
                st.info("No split data found for this batter this season.")
        else:
            st.warning("No position players found. Try a different name.")

    st.markdown("---")
    st.markdown("### Last 3 Starts — Pitcher Breakdown")
    pitcher_hist_search = st.text_input(
        "Search pitcher", placeholder="e.g. Gerrit Cole", key="hist_pitcher_search"
    )
    if pitcher_hist_search:
        from core.player_lookup import search_players
        pit_results = search_players(pitcher_hist_search)
        pitchers = [r for r in pit_results if r["position"] in ("P", "SP", "RP")]

        if pitchers:
            pit_labels = [f"{r['full_name']} — {r['team']}" for r in pitchers]
            pit_chosen = st.selectbox("Select pitcher", pit_labels, key="hist_pitcher_sel")
            pit_idx = pit_labels.index(pit_chosen)
            hist_pitcher_id = pitchers[pit_idx]["player_id"]

            season = date.today().year
            arsenal = get_pitcher_arsenal(hist_pitcher_id, season)
            if not arsenal.empty:
                st.markdown("#### Season Arsenal")
                disp = arsenal[["pitch_label", "usage_pct", "count", "avg_velocity"]].copy()
                disp["usage_pct"] = (disp["usage_pct"] * 100).round(1).astype(str) + "%"
                disp.columns = ["Pitch", "Usage %", "Count", "Avg Velo (mph)"]
                st.dataframe(disp, use_container_width=True, hide_index=True)
            else:
                st.info("No arsenal data found.")
        else:
            st.warning("No pitchers found.")
