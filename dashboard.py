"""
Baseball Matchup Tracker — Streamlit Dashboard
Four tabs: Today · Matchup · Trends · History

Live game data  →  mlb_live/pipeline.py  (statsapi, no long-term cache)
Statcast data   →  mlb_season/pipeline.py (Baseball Savant CSV, cached)
"""

from __future__ import annotations

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

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stHeader"] { background: #0e1117; }
h1, h2, h3 { color: #fafafa; }
p, li { color: #ccc; }

.mas-card {
    border: 1px solid #2a2a2a; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
    background: #0e1117; display: flex; align-items: flex-start; gap: 14px;
}
.mas-card-body { flex: 1; }
.mas-player-name { font-size: 1.05rem; font-weight: 700; color: #fafafa; margin: 0; }
.mas-matchup-line { font-size: 0.82rem; color: #888; margin: 2px 0 6px; }
.mas-bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.mas-bar-container { flex: 1; height: 10px; background: #2a2a2a; border-radius: 6px; overflow: hidden; }
.mas-bar-fill { height: 100%; transition: width 0.3s ease; }
.mas-strong-advantage    { background: #2ecc71; }
.mas-slight-advantage    { background: #95d35d; }
.mas-slight-disadvantage { background: #f39c12; }
.mas-strong-disadvantage { background: #e74c3c; }
.mas-number { font-size: 1.1rem; font-weight: 800; min-width: 38px; text-align: right; color: #fafafa; }
.mas-meta-row { display: flex; gap: 16px; margin-top: 4px; }
.mas-whiff-badge { font-size: 0.78rem; color: #aaa; }
.driver-pill { font-size: 0.78rem; color: #EB6E1F; }
.volatile-tag { font-size: 0.7rem; background: #3a2a1a; color: #f39c12; padding: 1px 6px; border-radius: 4px; margin-left: 6px; }

.leverage-banner {
    background: linear-gradient(90deg, #1a2a4a 0%, #2a1a4a 100%);
    border-left: 4px solid #EB6E1F; padding: 12px 18px;
    border-radius: 6px; margin-bottom: 16px;
}
.leverage-title { font-size: 0.85rem; font-weight: 700; color: #EB6E1F; margin-bottom: 4px; letter-spacing: 0.06em; }
.leverage-line { font-size: 0.88rem; color: #ddd; margin: 2px 0; }

.score-header { text-align: center; padding: 16px 0 8px; }
.score-teams { font-size: 1.4rem; font-weight: 700; color: #fafafa; letter-spacing: 0.04em; }
.score-numbers { font-size: 2.8rem; font-weight: 900; color: #EB6E1F; line-height: 1.1; }
.score-situation { font-size: 0.88rem; color: #888; margin-top: 4px; }

.live-mas-chip {
    background: #16213e; border: 1px solid #2a2a4a;
    border-radius: 8px; padding: 10px 14px;
    display: flex; align-items: center; gap: 12px;
}
.live-mas-label { font-size: 0.78rem; color: #888; margin-bottom: 2px; }
.live-mas-value { font-size: 1.6rem; font-weight: 900; }

[data-testid="stTabs"] button { font-size: 0.95rem; font-weight: 600; color: #aaa; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #EB6E1F; border-bottom-color: #EB6E1F; }
hr { border-color: #2a2a2a; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

# ── imports ───────────────────────────────────────────────────────────────────

# Live game data — original mlb_live pipeline (statsapi, no long-term cache)
from mlb_live.pipeline import (
    get_all_teams,
    get_all_todays_games,
    get_todays_game,
    get_upcoming_game,
    get_game_summary,
    get_live_box_score,
    get_linescore,
    get_current_play,
    build_inning_table,
    build_batting_table,
    build_pitching_table,
    get_game_lineup,
    get_pitcher_live_pitches,
    compute_pitcher_fatigue,
    get_win_probability_from_plays,
    get_game_pitchers,
    get_active_pitcher,
)

# Season / Statcast data — mlb_season pipeline (Baseball Savant CSV, cached)
from mlb_season.pipeline import (
    get_pitcher_arsenal,
    get_batter_pitch_splits,
    get_batter_career_pitch_splits,
    get_team_batting_leaders,
    get_player_headshot_url,
)

# MAS scoring
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
    search_players,
)
from core.visualizations import (
    plot_career_pitch_splits,
    plot_hot_zone_grid,
    plot_per_pitch_breakdown,
    show_pitch_mix_simulator,
)


# ── session state ─────────────────────────────────────────────────────────────

def _init_state():
    for k, v in {
        "selected_game_id": None,
        "selected_team_id": None,
        "selected_pitcher_id": None,
    }.items():
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
# TAB 1 — TODAY  (uses mlb_live.pipeline — statsapi, no cache)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_today:
    all_games = get_all_todays_games()

    if not all_games:
        st.info("No games scheduled today.")
    else:
        def _game_label(g: dict) -> str:
            status = g.get("status", "")
            icon = {"In Progress": "🔴", "Final": "✅", "Game Over": "✅"}.get(status, "📅")
            score = ""
            if status in ("In Progress", "Final", "Game Over"):
                score = f"  {g.get('away_score', 0)}–{g.get('home_score', 0)}"
            return f"{icon}  {g['away_name']}  @  {g['home_name']}{score}"

        # Default selection to previously chosen game
        default_idx = 0
        if st.session_state.selected_game_id:
            for i, g in enumerate(all_games):
                if g.get("game_id") == st.session_state.selected_game_id:
                    default_idx = i
                    break

        chosen_idx = st.selectbox(
            "Select game",
            range(len(all_games)),
            format_func=lambda i: _game_label(all_games[i]),
            index=default_idx,
            key="today_game_sel",
        )
        game = all_games[chosen_idx]
        game_id = game["game_id"]
        st.session_state.selected_game_id = game_id
        s = get_game_summary(game)

        status   = s["status"]
        is_live  = status in ("In Progress", "Manager challenge", "Warmup")
        is_final = "Final" in status or "Game Over" in status

        # ── score header ──────────────────────────────────────────────────────
        if is_live or is_final:
            ls = get_linescore(game_id)
            inning = ls.get("currentInning", s.get("inning", ""))
            half   = ls.get("inningState", s.get("inning_state", ""))
            situation = f"{half} {inning}" if inning else status
            away_score = ls.get("teams", {}).get("away", {}).get("runs", s["away_score"])
            home_score = ls.get("teams", {}).get("home", {}).get("runs", s["home_score"])
        else:
            situation  = s.get("game_datetime", status)
            away_score = s["away_score"]
            home_score = s["home_score"]

        st.markdown(f"""
<div class="score-header">
  <div class="score-teams">{s['away_name']} @ {s['home_name']}</div>
  <div class="score-numbers">{away_score} – {home_score}</div>
  <div class="score-situation">{situation}</div>
</div>""", unsafe_allow_html=True)

        # ── current at-bat MAS chip (live only) ───────────────────────────────
        if is_live:
            cp = get_current_play(game_id)
            matchup = cp.get("matchup", {})
            bat_id  = matchup.get("batter", {}).get("id")
            pit_id  = matchup.get("pitcher", {}).get("id")
            bat_name = matchup.get("batter", {}).get("fullName", "")
            pit_name = matchup.get("pitcher", {}).get("fullName", "")

            if bat_id and pit_id:
                with st.spinner("Computing live MAS…"):
                    result = compute_matchup_score(bat_id, pit_id)
                mas   = result["mas"]
                whiff = result["whiff_score"]
                color = mas_color(mas)
                css   = mas_css_class(mas)
                st.markdown(f"""
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
      <div class="mas-bar-fill {css}" style="width:{mas}%"></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── box score detail ──────────────────────────────────────────────────
        with st.expander("📊 Box Score Detail", expanded=False):
            if is_live or is_final:
                box = get_live_box_score(game_id)
                ls  = get_linescore(game_id)
                col_a, col_h = st.columns(2)
                with col_a:
                    st.markdown(f"**{s['away_name']}**")
                    df_away = build_batting_table(box, "away")
                    if not df_away.empty:
                        st.dataframe(df_away, use_container_width=True, hide_index=True)
                with col_h:
                    st.markdown(f"**{s['home_name']}**")
                    df_home = build_batting_table(box, "home")
                    if not df_home.empty:
                        st.dataframe(df_home, use_container_width=True, hide_index=True)

                inning_df = build_inning_table(ls)
                if not inning_df.empty:
                    st.markdown("**Linescore**")
                    st.dataframe(inning_df, use_container_width=True, hide_index=True)
            else:
                st.info("Box score will be available once the game starts.")

        # ── probable pitchers / navigate to Matchup ────────────────────────────
        st.markdown("---")
        col_away, col_home = st.columns(2)
        with col_away:
            st.markdown(f"**{s['away_name']}**")
            away_pitcher = s.get("away_probable_pitcher", "TBD")
            st.markdown(f"🔵 SP: {away_pitcher}")
            away_pitcher_id = game.get("away_probable_pitcher_id")
            if away_pitcher_id and st.button(f"Analyze vs {away_pitcher}", key="btn_away"):
                st.session_state.selected_pitcher_id = away_pitcher_id
                st.session_state.selected_team_id    = game["home_id"]
                st.success("Switch to the **🎯 Matchup** tab.")
        with col_home:
            st.markdown(f"**{s['home_name']}**")
            home_pitcher = s.get("home_probable_pitcher", "TBD")
            st.markdown(f"🟠 SP: {home_pitcher}")
            home_pitcher_id = game.get("home_probable_pitcher_id")
            if home_pitcher_id and st.button(f"Analyze vs {home_pitcher}", key="btn_home"):
                st.session_state.selected_pitcher_id = home_pitcher_id
                st.session_state.selected_team_id    = game["away_id"]
                st.success("Switch to the **🎯 Matchup** tab.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MATCHUP  (MAS — uses mlb_season.pipeline for Statcast)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_matchup:
    st.markdown("#### Select Pitcher")
    pitcher_search = st.text_input(
        "Search pitcher by name", placeholder="e.g. Spencer Strider", key="pitcher_search"
    )

    selected_pitcher_id: int | None = st.session_state.selected_pitcher_id

    if pitcher_search:
        results = search_players(pitcher_search)
        pitchers = [r for r in results if r["position"] in ("P", "SP", "RP")]
        if pitchers:
            labels  = [f"{r['full_name']} — {r['team']}" for r in pitchers]
            chosen  = st.selectbox("Match", labels, key="pitcher_search_sel")
            chosen_idx = labels.index(chosen)
            selected_pitcher_id = pitchers[chosen_idx]["player_id"]
            st.session_state.selected_pitcher_id = selected_pitcher_id
        else:
            st.warning("No pitchers found.")

    if selected_pitcher_id:
        pitcher_info = get_player_info(selected_pitcher_id)
        pitcher_hand = get_pitcher_handedness(selected_pitcher_id)
        st.markdown(
            f"**{pitcher_info['full_name']}** &nbsp;·&nbsp; {pitcher_info['team_name']} "
            f"&nbsp;·&nbsp; {'LHP' if pitcher_hand == 'L' else 'RHP'}"
        )

        with st.expander("📋 Pitcher Arsenal Detail", expanded=False):
            arsenal = get_pitcher_arsenal(selected_pitcher_id)
            if not arsenal.empty:
                disp = arsenal[["pitch_label", "usage_pct", "count", "avg_velo"]].copy()
                disp["usage_pct"] = (disp["usage_pct"] * 100).round(1).astype(str) + "%"
                disp.columns = ["Pitch", "Usage %", "Count", "Avg Velo (mph)"]
                st.dataframe(disp, use_container_width=True, hide_index=True)
            else:
                st.info("No arsenal data available for this season.")

        st.markdown("---")
        st.markdown("#### Lineup")
        lineup_source = st.radio(
            "Load lineup from",
            ["Today's game", "Manual team roster", "Enter player IDs"],
            horizontal=True, key="lineup_source",
        )

        batters: list[dict] = []

        if lineup_source == "Today's game":
            if st.session_state.selected_game_id and st.session_state.selected_team_id:
                # Use mlb_live.pipeline.get_game_lineup — original statsapi model
                batters = get_game_lineup(
                    st.session_state.selected_game_id,
                    st.session_state.selected_team_id,
                )
                if not batters:
                    st.info("Lineup not yet posted — try 'Manual team roster'.")
            else:
                st.info("Select a game in the **Today** tab first, then click 'Analyze vs [Pitcher]'.")

        elif lineup_source == "Manual team roster":
            team_search = st.text_input("Search team name", key="team_search_matchup")
            if team_search:
                team_results = search_players(team_search)
                if team_results:
                    info = get_player_info(team_results[0]["player_id"])
                    if info.get("team_id"):
                        roster  = get_team_roster(info["team_id"])
                        batters = [r for r in roster if r["position"] not in ("P", "SP", "RP")]

        else:
            id_input = st.text_input(
                "Enter MLBAM player IDs, comma-separated",
                placeholder="592450, 670541, 514888", key="manual_ids",
            )
            if id_input:
                try:
                    ids = [int(x.strip()) for x in id_input.split(",") if x.strip()]
                    batters = [
                        {"player_id": pid, "name": get_player_name(pid),
                         "position": get_player_info(pid).get("position", "")}
                        for pid in ids
                    ]
                except ValueError:
                    st.error("Invalid IDs — enter comma-separated integers.")

        # ── compute MAS ───────────────────────────────────────────────────────
        if batters:
            season = date.today().year

            with st.spinner(f"Computing MAS for {len(batters)} batters…"):
                results: list[dict] = []
                for b in batters:
                    r = compute_matchup_score(b["player_id"], selected_pitcher_id, season)
                    results.append({
                        "player_id": b["player_id"],
                        "name":      b.get("name", get_player_name(b["player_id"])),
                        "position":  b.get("position", ""),
                        **r,
                    })

            # Lineup Leverage banner
            sorted_r = sorted(results, key=lambda x: x["mas"], reverse=True)
            top3 = sorted_r[:3]
            bot3 = sorted_r[-3:][::-1]
            top_str = " · ".join(f"{r['name'].split()[-1]} ({r['mas']:.0f})" for r in top3)
            bot_str = " · ".join(f"{r['name'].split()[-1]} ({r['mas']:.0f})" for r in bot3)
            st.markdown(f"""
<div class="leverage-banner">
  <div class="leverage-title">⚡ LINEUP LEVERAGE</div>
  <div class="leverage-line">🟢 Top advantages: {top_str}</div>
  <div class="leverage-line">🔴 Top risks: {bot_str}</div>
</div>""", unsafe_allow_html=True)

            # Headline MAS grid
            for r in results:
                pid   = r["player_id"]
                mas   = r["mas"]
                whiff = r["whiff_score"]
                vol   = r["volatility"]
                driver = r.get("primary_driver", {})
                color  = mas_color(mas)
                css    = mas_css_class(mas)
                headshot = get_player_headshot_url(pid)

                volatile_tag = '<span class="volatile-tag">Boom/Bust</span>' if vol > 20 else ""
                driver_line = ""
                if driver:
                    driver_line = (
                        f'<span class="driver-pill">Driver: {driver.get("pitch_type")} '
                        f'({driver.get("usage_pct", 0):.0f}% usage · '
                        f'{driver.get("contribution_pct", 0):.0f}% contribution)</span>'
                    )

                st.markdown(f"""
<div class="mas-card">
  <img src="{headshot}" width="56" height="56"
       style="border-radius:50%;object-fit:cover;border:2px solid #2a2a2a">
  <div class="mas-card-body">
    <p class="mas-player-name">{r['name']}</p>
    <p class="mas-matchup-line">vs {pitcher_info['full_name']} ({'LHP' if pitcher_hand=='L' else 'RHP'})</p>
    <div class="mas-bar-row">
      <div class="mas-bar-container">
        <div class="mas-bar-fill {css}" style="width:{mas}%"></div>
      </div>
      <span class="mas-number" style="color:{color}">{mas:.0f}</span>
      {volatile_tag}
    </div>
    <div class="mas-meta-row">
      <span class="mas-whiff-badge">Whiff Risk: {whiff:.0f}</span>
      {driver_line}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

                with st.expander(f"↳ Dig Deeper — {r['name']}", expanded=False):
                    for w in r.get("sample_warnings", []):
                        st.warning(w)

                    d1, d2, d3, d4 = st.tabs(
                        ["Per-Pitch Breakdown", "Hot Zone", "Career Splits", "Pitch Mix Simulator"]
                    )
                    with d1:
                        pp = r.get("per_pitch_breakdown", pd.DataFrame())
                        if not pp.empty:
                            st.plotly_chart(plot_per_pitch_breakdown(pp),
                                            use_container_width=True, key=f"bd_{pid}")
                        else:
                            st.info("No per-pitch data.")
                    with d2:
                        st.plotly_chart(plot_hot_zone_grid(pid, season),
                                        use_container_width=True, key=f"hz_{pid}")
                    with d3:
                        st.plotly_chart(plot_career_pitch_splits(pid, season),
                                        use_container_width=True, key=f"cs_{pid}")
                    with d4:
                        show_pitch_mix_simulator(selected_pitcher_id, pid, season)

        elif lineup_source != "Today's game":
            st.info("Add batters above to compute MAS scores.")
    else:
        st.info("Search for a pitcher above, or click 'Analyze vs [Pitcher]' in the Today tab.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRENDS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_trends:
    st.markdown("### MAS Trend — Batter over Time")
    st.info(
        "Coming soon: track how a batter's MAS vs a specific pitcher evolves "
        "across a rolling 30-day window as the pitcher's arsenal usage shifts."
    )
    st.markdown("---")
    st.markdown("### League Pitch Baseline Trends")
    st.info(
        "Coming soon: chart how barrel_rate_mean and whiff_rate_mean for each "
        "pitch type move across the season."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown("### Season Splits — Batter")
    batter_search = st.text_input(
        "Search batter", placeholder="e.g. Yordan Alvarez", key="hist_batter"
    )
    if batter_search:
        hist_results = search_players(batter_search)
        non_p = [r for r in hist_results if r["position"] not in ("P", "SP", "RP")]
        if non_p:
            labels  = [f"{r['full_name']} — {r['team']}" for r in non_p]
            chosen  = st.selectbox("Select batter", labels, key="hist_batter_sel")
            bid     = non_p[labels.index(chosen)]["player_id"]
            splits  = get_batter_pitch_splits(bid)
            if not splits.empty:
                disp = splits[["pitch_label","vs_hand","barrel_rate","whiff_rate","sample_size"]].copy()
                disp["barrel_rate"] = (disp["barrel_rate"]*100).round(2).astype(str)+"%"
                disp["whiff_rate"]  = (disp["whiff_rate"]*100).round(2).astype(str)+"%"
                disp.columns = ["Pitch","vs Hand","Barrel %","Whiff %","Pitches Seen"]
                st.dataframe(disp, use_container_width=True, hide_index=True)
                st.plotly_chart(plot_career_pitch_splits(bid), use_container_width=True, key="hist_cs")
            else:
                st.info("No split data found for this batter this season.")
        else:
            st.warning("No position players found.")

    st.markdown("---")
    st.markdown("### Season Arsenal — Pitcher")
    pitcher_hist = st.text_input(
        "Search pitcher", placeholder="e.g. Gerrit Cole", key="hist_pitcher"
    )
    if pitcher_hist:
        pit_results = search_players(pitcher_hist)
        pitchers    = [r for r in pit_results if r["position"] in ("P","SP","RP")]
        if pitchers:
            labels   = [f"{r['full_name']} — {r['team']}" for r in pitchers]
            chosen   = st.selectbox("Select pitcher", labels, key="hist_pitcher_sel")
            pid      = pitchers[labels.index(chosen)]["player_id"]
            arsenal  = get_pitcher_arsenal(pid)
            if not arsenal.empty:
                disp = arsenal[["pitch_label","usage_pct","count","avg_velo"]].copy()
                disp["usage_pct"] = (disp["usage_pct"]*100).round(1).astype(str)+"%"
                disp.columns = ["Pitch","Usage %","Count","Avg Velo (mph)"]
                st.dataframe(disp, use_container_width=True, hide_index=True)
            else:
                st.info("No arsenal data found.")
        else:
            st.warning("No pitchers found.")
