"""
Baseball Matchup Tracker — Streamlit Dashboard
Five tabs: Live · Matchup · Pitching · Trends · Last 3

Game selector at top drives the Matchup tab automatically.
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

from mlb_season.pipeline import (
    get_pitcher_arsenal,
    get_batter_pitch_splits,
    get_batter_career_pitch_splits,
    get_batter_game_log,
    get_pitcher_game_log,
    get_last_n_completed_games,
    aggregate_batting_stats,
    aggregate_pitching_stats,
    get_team_batting_leaders,
    get_player_headshot_url,
    predict_lineup_vs_hand,
    get_confirmed_lineup,
    get_pitcher_sample_flag,
)

from core.matchup_score import (
    compute_matchup_score,
    compute_mash_and_miss,
    mas_color,
    mas_css_class,
    mas_label,
)
from core.player_lookup import (
    get_player_info,
    get_player_name,
    get_team_roster,
    search_players,
)
from core.handedness import (
    get_pitcher_handedness,
    get_batter_handedness,
    handedness_badge_html,
)
from core.visualizations import (
    plot_career_pitch_splits,
    plot_hot_zone_grid,
    plot_per_pitch_breakdown,
    show_pitch_mix_simulator,
)
from core.game_selector import render_day_and_game_selector

from dashboard.components.matchup_cards import (
    render_pitcher_header,
    render_leverage_summary,
    render_matchup_cards,
    merge_partial_and_predicted,
)

# ── session state ─────────────────────────────────────────────────────────────

def _init_state():
    for k, v in {
        "selected_game_id": None,
        "selected_game": None,
        "selected_pitcher_id": None,
        "selected_team_id": None,
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

# ── day + game selector (global) ──────────────────────────────────────────────

st.markdown("### 📅 Game Selection")
render_day_and_game_selector()
st.markdown("---")

# ── tabs ──────────────────────────────────────────────────────────────────────

(tab_live, tab_matchup, tab_pitching,
 tab_trends, tab_l3) = st.tabs([
    "🔴 Live", "🎯 Matchup", "📡 Pitching",
    "📈 Trends", "📋 Last 3",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_live:
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
# TAB 2 — MATCHUP
# ═══════════════════════════════════════════════════════════════════════════════

with tab_matchup:
    selected_game = st.session_state.get("selected_game")

    if not selected_game:
        st.info("Pick a day and game at the top to begin analysis.")
        st.stop()

    away_name     = selected_game["away_name"]
    home_name     = selected_game["home_name"]
    away_sp_name  = selected_game.get("away_probable_pitcher") or "TBD"
    home_sp_name  = selected_game.get("home_probable_pitcher") or "TBD"
    away_sp_id    = selected_game.get("away_probable_pitcher_id")
    home_sp_id    = selected_game.get("home_probable_pitcher_id")

    side_choice = st.radio(
        "Analyze pitcher from:",
        [f"{away_name} — {away_sp_name}", f"{home_name} — {home_sp_name}"],
        horizontal=True,
        key="matchup_side_choice",
    )

    is_away_pitcher    = side_choice.startswith(away_name)
    pitcher_id         = away_sp_id if is_away_pitcher else home_sp_id
    pitcher_name       = away_sp_name if is_away_pitcher else home_sp_name
    opposing_team_id   = selected_game["home_id"] if is_away_pitcher else selected_game["away_id"]
    opposing_team_name = home_name if is_away_pitcher else away_name

    if not pitcher_id or pitcher_name == "TBD":
        st.warning("Probable pitcher not yet announced for this side.")
        st.stop()

    render_pitcher_header(pitcher_id, pitcher_name)

    pitcher_hand = get_pitcher_handedness(pitcher_id)
    confirmed_df, lineup_status = get_confirmed_lineup(
        selected_game["game_id"], opposing_team_id
    )

    if lineup_status == "confirmed":
        lineup_df = confirmed_df
        lineup_badge = "✅ Confirmed"
        show_confidence = False
    elif lineup_status == "partial":
        predicted_df = predict_lineup_vs_hand(opposing_team_id, pitcher_hand)
        lineup_df = merge_partial_and_predicted(confirmed_df, predicted_df)
        lineup_badge = "⚠️ Partial"
        show_confidence = True
    else:
        lineup_df = predict_lineup_vs_hand(opposing_team_id, pitcher_hand)
        lineup_badge = f"🔮 Predicted (vs {pitcher_hand}HP last 5 games)"
        show_confidence = True

    if lineup_df.empty:
        st.warning(f"Could not determine lineup for {opposing_team_name}.")
        st.stop()

    st.markdown(
        f"**Opposing Lineup — {opposing_team_name}** &nbsp; "
        f'<span style="color:#EB6E1F;font-size:0.85rem">{lineup_badge}</span>',
        unsafe_allow_html=True,
    )

    with st.spinner(f"Computing MASH & MISS scores vs {pitcher_name}…"):
        scored_lineup = []
        for _, row in lineup_df.iterrows():
            result = compute_mash_and_miss(row["player_id"], pitcher_id)
            scored_lineup.append({**row.to_dict(), **result})

    render_leverage_summary(scored_lineup, pitcher_name)
    render_matchup_cards(scored_lineup, show_confidence=show_confidence)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PITCHING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_pitching:
    st.markdown("### Pitcher History")
    pitcher_hist = st.text_input(
        "Search pitcher", placeholder="e.g. Lance McCullers Jr.", key="hist_pitcher"
    )
    if pitcher_hist:
        pit_results = search_players(pitcher_hist)
        pitchers    = [r for r in pit_results if r["position"] in ("P", "SP", "RP")]
        if pitchers:
            labels  = [f"{r['full_name']} — {r['team']}" for r in pitchers]
            chosen  = st.selectbox("Select pitcher", labels, key="hist_pitcher_sel")
            pid     = pitchers[labels.index(chosen)]["player_id"]

            ptab1, ptab2 = st.tabs(["Game Log", "Season Arsenal"])

            with ptab1:
                st.markdown("#### Start-by-Start Log")
                plog = get_pitcher_game_log(pid)
                if not plog.empty:
                    disp = plog[["game_date", "pitches", "strikeouts", "whiffs", "whiff_rate"]].copy()
                    disp.columns = ["Date", "Pitches", "K", "Whiffs", "Whiff%"]
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.info("No Statcast data found for this pitcher this season.")

            with ptab2:
                arsenal = get_pitcher_arsenal(pid)
                if not arsenal.empty:
                    disp = arsenal[["pitch_label", "usage_pct", "count", "avg_velo"]].copy()
                    disp["usage_pct"] = (disp["usage_pct"] * 100).round(1).astype(str) + "%"
                    disp.columns = ["Pitch", "Usage %", "Count", "Avg Velo (mph)"]
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.info("No arsenal data found for this season.")
        else:
            st.warning("No pitchers found.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRENDS
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
# TAB 5 — LAST 3
# ═══════════════════════════════════════════════════════════════════════════════

with tab_l3:

    # ── Batter section ────────────────────────────────────────────────────────
    st.markdown("### Batter History")
    batter_search = st.text_input(
        "Search batter", placeholder="e.g. Yordan Alvarez", key="hist_batter"
    )
    if batter_search:
        hist_results = search_players(batter_search)
        non_p = [r for r in hist_results if r["position"] not in ("P", "SP", "RP")]
        if non_p:
            labels = [f"{r['full_name']} — {r['team']}" for r in non_p]
            chosen = st.selectbox("Select batter", labels, key="hist_batter_sel")
            bid    = non_p[labels.index(chosen)]["player_id"]

            htab1, htab2, htab3 = st.tabs(["Game Log", "Pitch Splits", "Split Chart"])

            with htab1:
                st.markdown("#### Game-by-Game Log (current season)")
                log = get_batter_game_log(bid)
                if not log.empty:
                    disp = log[["game_date", "ab", "hits", "hr", "k", "pitches", "avg"]].copy()
                    disp.columns = ["Date", "AB", "H", "HR", "K", "Pitches", "AVG"]
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.info("No game log data available for this season.")

            with htab2:
                st.markdown("#### Season Pitch-Type Splits (vs LHP / RHP)")
                splits = get_batter_pitch_splits(bid)
                if not splits.empty:
                    disp = splits[["pitch_label", "vs_hand", "barrel_rate", "whiff_rate", "sample_size"]].copy()
                    disp["barrel_rate"] = (disp["barrel_rate"] * 100).round(2).astype(str) + "%"
                    disp["whiff_rate"]  = (disp["whiff_rate"]  * 100).round(2).astype(str) + "%"
                    disp.columns = ["Pitch", "vs Hand", "Barrel %", "Whiff %", "Pitches Seen"]
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.info("No split data found for this batter this season.")

            with htab3:
                st.plotly_chart(
                    plot_career_pitch_splits(bid),
                    use_container_width=True, key="hist_cs"
                )
        else:
            st.warning("No position players found.")

    st.markdown("---")

    # ── Team Last-3 section ───────────────────────────────────────────────────
    st.markdown("### Last 3 Games — Team Stats")
    team_search_l3 = st.text_input(
        "Search team", placeholder="e.g. Houston Astros", key="hist_team"
    )
    if team_search_l3:
        try:
            all_teams = get_all_teams()
            matched = [t for t in all_teams if team_search_l3.lower() in t["name"].lower()]
            if matched:
                team_labels = [t["name"] for t in matched]
                chosen_team = st.selectbox("Select team", team_labels, key="hist_team_sel")
                team_id = next(t["id"] for t in matched if t["name"] == chosen_team)

                with st.spinner("Loading last 3 games…"):
                    recent_games = get_last_n_completed_games(team_id, n=3)
                    game_ids     = [g["game_id"] for g in recent_games]

                if not recent_games:
                    st.info("No completed games found in the last 30 days.")
                else:
                    for g in recent_games:
                        st.markdown(
                            f"✅ **{g['game_date']}** — {g['away_name']} @ {g['home_name']}  "
                            f"({g.get('away_score', '')}–{g.get('home_score', '')})"
                        )

                    col_bat, col_pit = st.columns(2)
                    with col_bat:
                        st.markdown("#### Batting (L3 combined)")
                        bat_stats = aggregate_batting_stats(game_ids, team_id)
                        if not bat_stats.empty:
                            disp = bat_stats[["name", "ab", "hits", "hr", "rbi", "k", "avg"]].copy()
                            disp.columns = ["Player", "AB", "H", "HR", "RBI", "K", "AVG"]
                            st.dataframe(disp, use_container_width=True, hide_index=True)
                        else:
                            st.info("No batting data.")

                    with col_pit:
                        st.markdown("#### Pitching (L3 combined)")
                        pit_stats = aggregate_pitching_stats(game_ids, team_id)
                        if not pit_stats.empty:
                            disp = pit_stats[["name", "ip", "k", "bb", "er", "hits"]].copy()
                            disp.columns = ["Pitcher", "IP", "K", "BB", "ER", "H"]
                            st.dataframe(disp, use_container_width=True, hide_index=True)
                        else:
                            st.info("No pitching data.")
            else:
                st.warning("No team found. Try the full team name, e.g. 'Houston Astros'.")
        except Exception as e:
            st.error(f"Error loading team data: {e}")
