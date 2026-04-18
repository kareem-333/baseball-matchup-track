"""
MLB Live Tracker Dashboard — Full Edition
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo

# ── Core shared utilities ─────────────────────────────────────────────────────
from core.config import ASTROS_ID, STAR_BATTERS, PITCH_NAMES
from core.headshots import headshot_b64
from core.player_lookup import lookup_player_id, get_pitcher_handedness

# ── MLB Live game pipeline ────────────────────────────────────────────────────
from mlb_live.pipeline import (
    get_all_teams, get_todays_game, get_upcoming_game,
    get_live_box_score, get_linescore, get_current_play, get_game_summary,
    build_inning_table, build_batting_table, build_pitching_table,
    get_pitcher_live_pitches, compute_pitcher_fatigue,
    get_win_probability_from_plays,
    get_game_pitchers, get_active_pitcher,
)
from mlb_live.charts import (
    plot_pitch_movement, plot_velocity_fade,
    plot_fatigue_gauge, plot_win_probability,
)

# ── MLB Season stats pipeline ─────────────────────────────────────────────────
from mlb_season.pipeline import (
    get_last_n_completed_games, aggregate_batting_stats, aggregate_pitching_stats,
    predict_lineup, get_batter_game_log, get_pitcher_game_log,
    fetch_statcast_csv, get_pitcher_arsenal, get_batter_pitch_splits,
    get_batter_career_pitch_splits,
    get_batter_hot_zones, get_barrel_trend,
    get_lineup_with_ids, get_team_pitching_staff, get_team_batting_leaders,
    get_league_avg_krate,
)
from mlb_season.charts import (
    plot_matchup_heatmap, plot_hot_zone_grid,
    plot_rolling_ops, plot_krate_chart, plot_barrel_trend,
    show_pitch_mix_simulator,
    plot_career_pitch_splits, plot_pitch_type_season_trend,
    plot_career_splits_table_fig,
)

CT = ZoneInfo("America/Chicago")


def _show_mlb_headshot(player_id: int, width: int = 80):
    """Display an MLB player headshot fetched as base64 (avoids CDN hot-link blocks)."""
    b64 = headshot_b64(player_id, size=width)
    if b64:
        st.image(b64, width=width)
    else:
        st.markdown(f'<div style="width:{width}px;height:{width}px;'
                    f'background:#1a2a4a;border-radius:50%;display:flex;'
                    f'align-items:center;justify-content:center;'
                    f'color:#aaa;font-size:1.6rem">⚾</div>',
                    unsafe_allow_html=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Live Tracker",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .score-box { border:2px solid #EB6E1F; border-radius:10px;
               padding:0.7rem 1.4rem; text-align:center; }
  .score-num { font-size:2.8rem; font-weight:800; color:#EB6E1F; }
  .team-name { font-size:0.95rem; font-weight:600; }
  .status-badge { background:#EB6E1F; color:white; border-radius:20px;
                  padding:3px 12px; font-weight:700; font-size:0.82rem; display:inline-block; }
  .sec-hdr { font-size:0.95rem; font-weight:700; color:#EB6E1F;
             border-bottom:1px solid #EB6E1F44; margin-bottom:0.4rem; padding-bottom:2px; }
  .base-on  { color:#EB6E1F; font-size:1.5rem; }
  .base-off { color:#333;    font-size:1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ MLB Live Tracker")
    st.markdown("---")

    @st.cache_data(ttl=3600)
    def load_teams():
        return get_all_teams()

    teams     = load_teams()
    team_map  = {t["name"]: t["id"] for t in teams}
    default_i = next((i for i, t in enumerate(teams) if t["id"] == ASTROS_ID), 0)
    sel_team  = st.selectbox("Team", list(team_map.keys()), index=default_i)
    team_id   = team_map[sel_team]

    st.markdown("---")
    st.caption(f"Last refresh: {datetime.now(CT).strftime('%-I:%M %p CT')}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Helper: formatted score header ───────────────────────────────────────────
def _score_header(s: dict, team_id: int):
    ca, cm, ch = st.columns([3, 2, 3])
    status = s["status"]
    is_live  = status in ("In Progress", "Manager challenge")
    is_final = "Final" in status

    with ca:
        st.markdown(f"""<div class="score-box">
            <div class="team-name">{s['away_name']}</div>
            <div class="score-num">{s['away_score']}</div>
        </div>""", unsafe_allow_html=True)
    with cm:
        st.markdown("<br>", unsafe_allow_html=True)
        if is_live:
            badge = f"{s['inning_state']} {s['inning']}"
        elif is_final:
            badge = "Final"
        else:
            try:
                dt = datetime.fromisoformat(s["game_datetime"].replace("Z", "+00:00")).astimezone(CT)
                badge = dt.strftime("%-I:%M %p CT")
            except Exception:
                badge = "Today"
        st.markdown(f'<div style="text-align:center"><span class="status-badge">{badge}</span></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center;color:#aaa;font-size:0.78rem;margin-top:4px">'
                    f'{s["venue"]}</p>', unsafe_allow_html=True)
    with ch:
        st.markdown(f"""<div class="score-box">
            <div class="team-name">{s['home_name']}</div>
            <div class="score-num">{s['home_score']}</div>
        </div>""", unsafe_allow_html=True)


# ── Helper: lineup-or-prediction column ──────────────────────────────────────
def _lineup_col(box, side, tid, label):
    batters = box.get(side, {}).get("batters", []) if box else []
    if len(batters) >= 9:
        st.markdown(f'<div class="sec-hdr">{label} — Confirmed</div>', unsafe_allow_html=True)
        df = build_batting_table(box, side)
        st.dataframe(df.head(9), use_container_width=True, hide_index=True)
    else:
        st.markdown(f'<div class="sec-hdr">{label} — Projected</div>', unsafe_allow_html=True)
        st.caption("Lineup not posted — projecting from last 5 games.")
        pred = predict_lineup(tid, n_games=5)
        if not pred.empty:
            st.dataframe(pred, use_container_width=True, hide_index=True)
        else:
            st.caption("Insufficient history.")


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"## {sel_team}")

# ── Main tabs ─────────────────────────────────────────────────────────────────
(tab_live, tab_scout, tab_pitching,
 tab_trends, tab_l3, tab_stats, tab_next) = st.tabs([
    "🔴 Live", "🎯 Scout", "📡 Pitching",
    "📈 Trends", "📋 Last 3", "📊 L3 Stats", "🔮 Up Next",
])

# ── Safe defaults — overwritten in tab_live if a game exists ─────────────────
# Defining here ensures downstream tabs never hit NameError when there's no game.
game    = None
s       = None
game_id = None
status  = "None"
is_live = is_final = is_pre = False

# ═══════════════════════════════════════════════════════════════════════════════
# 🔴 LIVE TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    with st.spinner("Loading today's game…"):
        game = get_todays_game(team_id)

    if not game:
        st.info("No game today. Check **Up Next** tab.")
    else:
        s         = get_game_summary(game)
        game_id   = s["game_id"]
        status    = s["status"]
        is_live   = status in ("In Progress", "Manager challenge")
        is_final  = "Final" in status
        is_pre    = not is_live and not is_final

        _score_header(s, team_id)
        st.markdown("---")

        team_side = "home" if s["home_id"] == team_id else "away"
        opp_side  = "away" if team_side == "home" else "home"
        opp_id    = s["away_id"] if team_side == "home" else s["home_id"]
        opp_name  = s["away_name"] if team_side == "home" else s["home_name"]

        # ── Pre-game ──────────────────────────────────────────────────────────
        if is_pre:
            pc1, pc2 = st.columns(2)
            pc1.metric(f"{s['away_name']} SP", s["away_probable_pitcher"])
            pc2.metric(f"{s['home_name']} SP", s["home_probable_pitcher"])
            st.markdown("---")
            try:
                pre_box = get_live_box_score(game_id)
                lc1, lc2 = st.columns(2)
                with lc1:
                    _lineup_col(pre_box, team_side, team_id,   sel_team)
                with lc2:
                    _lineup_col(pre_box, opp_side,  opp_id,    opp_name)
            except Exception as e:
                st.caption(f"Pre-game lineups unavailable: {e}")

        # ── In-progress / Final ───────────────────────────────────────────────
        else:
            try:
                box       = get_live_box_score(game_id)
                linescore = get_linescore(game_id)

                # Current at-bat + baserunners (live only)
                if is_live:
                    cp = get_current_play(game_id)
                    if cp:
                        mi  = cp.get("matchup", {})
                        cnt = cp.get("count", {})
                        r1  = bool(mi.get("postOnFirst"))
                        r2  = bool(mi.get("postOnSecond"))
                        r3  = bool(mi.get("postOnThird"))

                        col_ab, col_bases = st.columns([3, 1])
                        with col_ab:
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Batter",  mi.get("batter",  {}).get("fullName", "—"))
                            c2.metric("Pitcher", mi.get("pitcher", {}).get("fullName", "—"))
                            c3.metric("Count",   f"{cnt.get('balls',0)}-{cnt.get('strikes',0)}")
                            c4.metric("Outs",    cnt.get("outs", 0))
                            desc = cp.get("result", {}).get("description", "")
                            if desc:
                                st.caption(desc)
                        with col_bases:
                            # Diamond visual
                            st.markdown(f"""
                            <div style="text-align:center;line-height:1.2">
                              <span class="{'base-on' if r2 else 'base-off'}">◆</span><br>
                              <span class="{'base-on' if r3 else 'base-off'}">◆</span>
                              &nbsp;&nbsp;
                              <span class="{'base-on' if r1 else 'base-off'}">◆</span><br>
                              <span style="color:#555">◆</span>
                            </div>""", unsafe_allow_html=True)
                        st.markdown("---")

                # Win probability
                wp_data = get_win_probability_from_plays(game_id, home_team=s["home_name"])
                if wp_data:
                    wp_fig = plot_win_probability(wp_data, home_team=s["home_name"])
                    st.plotly_chart(wp_fig, use_container_width=True)
                    st.markdown("---")

                # Linescore
                st.markdown('<div class="sec-hdr">Linescore</div>', unsafe_allow_html=True)
                inn_df = build_inning_table(linescore)
                if not inn_df.empty:
                    ta = linescore.get("teams", {}).get("away", {})
                    th = linescore.get("teams", {}).get("home", {})
                    hdrs   = ["Team"] + [str(r) for r in inn_df["Inning"]] + ["R", "H", "E"]
                    away_r = [s["away_name"]] + list(inn_df["Away"]) + [
                        ta.get("runs",0), ta.get("hits",0), ta.get("errors",0)]
                    home_r = [s["home_name"]] + list(inn_df["Home"]) + [
                        th.get("runs",0), th.get("hits",0), th.get("errors",0)]
                    st.dataframe(pd.DataFrame([away_r, home_r], columns=hdrs),
                                 use_container_width=True, hide_index=True)
                st.markdown("---")

                # Batting / Pitching
                st.markdown('<div class="sec-hdr">Batting</div>', unsafe_allow_html=True)
                bt1, bt2 = st.tabs([sel_team, opp_name])
                with bt1:
                    df = build_batting_table(box, team_side)
                    if not df.empty:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No at-bats yet.")
                with bt2:
                    df = build_batting_table(box, opp_side)
                    if not df.empty:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Lineup not in box score — showing projection.")
                        pred = predict_lineup(opp_id, n_games=5)
                        if not pred.empty:
                            st.dataframe(pred, use_container_width=True, hide_index=True)
                        else:
                            st.caption("Insufficient history.")

                st.markdown('<div class="sec-hdr">Pitching</div>', unsafe_allow_html=True)
                pt1, pt2 = st.tabs([sel_team, opp_name])
                with pt1:
                    df = build_pitching_table(box, team_side)
                    if not df.empty:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No pitching data.")
                with pt2:
                    df = build_pitching_table(box, opp_side)
                    if not df.empty:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No pitching data.")

            except Exception as e:
                st.error(f"Error loading game details: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 SCOUT TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scout:
    st.markdown("### Pitch Matchup Scouting")

    # ── Pitcher selection ─────────────────────────────────────────────────────
    with st.expander("Pitcher Selection", expanded=True):
        col_ps, col_pm = st.columns([2, 1])
        with col_ps:
            pitcher_input = st.text_input(
                "Opposing Pitcher Name",
                value=(s["away_probable_pitcher"] if game and s["home_id"] == team_id
                       else s["home_probable_pitcher"] if game else ""),
                key=f"scout_pitcher_name_{team_id}",
            )

        pitcher_id   = None
        pitcher_hand = "?"
        if pitcher_input and pitcher_input not in ("TBD", ""):
            with st.spinner(f"Looking up {pitcher_input}…"):
                pitcher_id = lookup_player_id(pitcher_input)
            if pitcher_id:
                pitcher_hand = get_pitcher_handedness(pitcher_id)
                hand_color   = "#e74c3c" if pitcher_hand == "R" else "#3498db" if pitcher_hand == "L" else "#9b59b6"
                col_hs, col_pi = st.columns([1, 3])
                with col_hs:
                    _show_mlb_headshot(pitcher_id, width=80)
                with col_pi:
                    st.markdown(
                        f"**{pitcher_input}** &nbsp;"
                        f'<span style="background:{hand_color};color:white;border-radius:6px;'
                        f'padding:2px 10px;font-size:0.78rem;font-weight:700">Throws {pitcher_hand}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning(f"Could not find '{pitcher_input}' in MLB database.")

    # ── Load pitcher arsenal ──────────────────────────────────────────────────
    pitcher_arsenal = pd.DataFrame()
    if pitcher_id:
        with st.spinner("Loading pitcher arsenal…"):
            pitcher_arsenal = get_pitcher_arsenal(pitcher_id)
        if not pitcher_arsenal.empty:
            st.markdown('<div class="sec-hdr">Pitcher Arsenal</div>', unsafe_allow_html=True)
            disp_cols = ["pitch_name", "usage_pct", "avg_velo", "avg_spin", "avg_h_break", "avg_v_break"]
            disp_cols = [c for c in disp_cols if c in pitcher_arsenal.columns]
            st.dataframe(
                pitcher_arsenal[disp_cols].rename(columns={
                    "pitch_name": "Pitch", "usage_pct": "Usage%",
                    "avg_velo": "Velo", "avg_spin": "Spin",
                    "avg_h_break": "H Break", "avg_v_break": "V Break",
                }).round(1),
                use_container_width=True, hide_index=True,
            )

    st.markdown("---")

    # ── Batter selection ──────────────────────────────────────────────────────
    # Priority: confirmed game lineup → predicted lineup → star batters fallback
    # pre_batter_ids maps name → player_id for names sourced from the box score
    pre_batter_ids: dict[str, int] = {}

    lineup_rows: list[dict] = []
    if game:
        ts = "home" if s["home_id"] == team_id else "away"
        try:
            lineup_rows = get_lineup_with_ids(game_id, ts)
        except Exception:
            lineup_rows = []

    if lineup_rows:
        # Confirmed or partial confirmed lineup from box score
        lineup_source = "Confirmed" if any(r["confirmed"] for r in lineup_rows) else "Partial"
        for r in lineup_rows:
            pre_batter_ids[r["name"]] = r["player_id"]
        default_batters = [r["name"] for r in lineup_rows]
        source_label    = f"Today's Lineup ({lineup_source})"
    else:
        # No box score yet — fall back to predict_lineup
        try:
            pred_df = predict_lineup(team_id, n_games=5)
            predicted_names = pred_df["Projected Player"].tolist() if not pred_df.empty else []
        except Exception:
            predicted_names = []
        default_batters = [n for n in predicted_names if n and n != "Insufficient data"][:9]
        source_label    = "Projected Lineup"

    # Build options: box-score lineup first, then projected names, then star batters
    all_batter_options = list(pre_batter_ids.keys())
    for name in default_batters:                   # add projected names not already present
        if name not in all_batter_options:
            all_batter_options.append(name)
    for name in STAR_BATTERS:                      # star batters as extra options
        if name not in all_batter_options:
            all_batter_options.append(name)

    if default_batters:
        st.caption(f"Auto-populated from: **{source_label}**. Adjust below as needed.")

    sel_batters = st.multiselect(
        "Batters (for matchup analysis)",
        all_batter_options,
        default=[b for b in default_batters if b in all_batter_options],
        key=f"scout_batters_{team_id}",
    )

    if not sel_batters:
        st.info("Select at least one batter above.")
    else:
        # ── Load batter splits ────────────────────────────────────────────────
        batters_splits: dict[str, pd.DataFrame] = {}
        batter_ids: dict[str, int] = {}

        with st.spinner("Loading batter pitch splits…"):
            for name in sel_batters:
                # Box-score ID → STAR_BATTERS dict → name lookup
                bid = pre_batter_ids.get(name) or STAR_BATTERS.get(name) or lookup_player_id(name)
                if bid:
                    batter_ids[name]     = bid
                    batters_splits[name] = get_batter_pitch_splits(bid)

        if not any(not df.empty for df in batters_splits.values()):
            st.warning("No pitch split data found for selected batters.")
        else:
            # ── Matchup heatmap ───────────────────────────────────────────────
            hm_fig = plot_matchup_heatmap(batters_splits, pitcher_arsenal)
            st.plotly_chart(hm_fig, use_container_width=True)

            # ── Hot zone grid per batter ──────────────────────────────────────
            st.markdown("---")
            st.markdown("### Strike Zone Hot Zones")
            hz_batter = st.selectbox("Select Batter for Hot Zone", list(batter_ids.keys()),
                                      key=f"hz_batter_{team_id}")
            if hz_batter and hz_batter in batter_ids:
                with st.spinner(f"Loading hot zones for {hz_batter}…"):
                    hot_zones = get_batter_hot_zones(batter_ids[hz_batter])
                    pitcher_locs = (
                        get_pitcher_arsenal(pitcher_id)[["pitch_type", "avg_x", "avg_z", "usage_pct", "pitch_name"]]
                        if pitcher_id and not pitcher_arsenal.empty
                        else None
                    )

                hc1, hc2 = st.columns([1, 5])
                with hc1:
                    _show_mlb_headshot(batter_ids[hz_batter], width=80)
                with hc2:
                    st.markdown(f"**{hz_batter}** — xBA by Zone")

                hz_fig = plot_hot_zone_grid(
                    hot_zones, pitcher_locs,
                    title=f"{hz_batter} — Hot Zones + {pitcher_input or 'Pitcher'} Pitch Locations",
                )
                st.plotly_chart(hz_fig, use_container_width=True)

            # ── Pitch mix simulator ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("### Pitch Mix Simulator")
            st.caption(
                "Adjust the sliders to change the pitcher's pitch mix. "
                "Adj Barrel% and Base Barrel% are identical until you move a slider — "
                "that is expected, as sliders start at the pitcher's actual usage rates."
            )
            show_pitch_mix_simulator(batters_splits, pitcher_arsenal, key_prefix="sim")

        # ── Career / multi-season pitch-type splits ───────────────────────────
        st.markdown("---")
        with st.expander("📊 Career Pitch-Type Splits — larger sample size", expanded=False):
            st.markdown(
                "Aggregates Statcast data across multiple seasons so you can see how "
                "a batter *actually* performs against each pitch type over a bigger sample. "
                "Great for spotting weaknesses like **Ohtani vs sliders** or "
                "**Freeman vs breaking balls**."
            )

            import datetime as _dt_career
            _cur_yr = _dt_career.date.today().year

            # Season range selector
            career_col1, career_col2 = st.columns([2, 1])
            with career_col1:
                avail_seasons = list(range(_cur_yr - 4, _cur_yr + 1))
                sel_seasons   = st.multiselect(
                    "Seasons to include",
                    avail_seasons,
                    default=[_cur_yr - 2, _cur_yr - 1, _cur_yr],
                    key=f"career_seasons_{team_id}",
                )
            with career_col2:
                career_batter = st.selectbox(
                    "Batter to analyse",
                    list(batter_ids.keys()) if batter_ids else list(STAR_BATTERS.keys()),
                    key=f"career_batter_{team_id}",
                )

            if not sel_seasons:
                st.warning("Select at least one season.")
            else:
                # Look up ID — box score first, then STAR_BATTERS, then name lookup
                career_bid = (
                    batter_ids.get(career_batter)
                    or STAR_BATTERS.get(career_batter)
                    or lookup_player_id(career_batter)
                )
                if not career_bid:
                    st.warning(f"Could not resolve player ID for **{career_batter}**.")
                else:
                    seasons_tuple = tuple(sorted(sel_seasons))
                    season_label  = (
                        f"{seasons_tuple[0]}–{seasons_tuple[-1]}"
                        if len(seasons_tuple) > 1 else str(seasons_tuple[0])
                    )
                    with st.spinner(
                        f"Fetching {season_label} Statcast data for {career_batter}…"
                    ):
                        career_df, by_season_df = get_batter_career_pitch_splits(
                            career_bid, seasons=seasons_tuple
                        )

                    if career_df.empty:
                        st.info(
                            f"No Statcast data found for {career_batter} "
                            f"in seasons {seasons_tuple}."
                        )
                    else:
                        total_pitches = career_df["pitches"].sum() if "pitches" in career_df.columns else 0
                        st.caption(
                            f"**{career_batter}** · {season_label} · "
                            f"{total_pitches:,} total pitches across "
                            f"{len(career_df)} pitch types"
                        )

                        # ── Colour-coded stats table ──────────────────────────
                        tbl_fig = plot_career_splits_table_fig(career_df, career_batter)
                        st.plotly_chart(tbl_fig, use_container_width=True)

                        # ── Grouped bar overview ──────────────────────────────
                        bar_fig = plot_career_pitch_splits(career_df, career_batter)
                        st.plotly_chart(bar_fig, use_container_width=True)

                        # ── Per-pitch-type deep dive (season-by-season) ───────
                        if not by_season_df.empty and by_season_df["season"].nunique() >= 2:
                            st.markdown("#### Year-over-Year Trend by Pitch Type")
                            pitch_name_options = career_df["pitch_name"].tolist()
                            dive_pitch = st.selectbox(
                                "Select pitch type for season-by-season breakdown",
                                pitch_name_options,
                                key=f"career_dive_pitch_{team_id}_{career_batter}",
                            )
                            if dive_pitch:
                                trend_fig = plot_pitch_type_season_trend(
                                    by_season_df, dive_pitch, career_batter
                                )
                                st.plotly_chart(trend_fig, use_container_width=True)

                                # Raw numbers table
                                dive_rows = by_season_df[
                                    by_season_df["pitch_name"] == dive_pitch
                                ].sort_values("season")
                                if not dive_rows.empty:
                                    show_cols = [c for c in [
                                        "season", "pitches", "whiff_rate", "k_rate",
                                        "barrel_rate", "hard_hit_rate", "avg_ev", "xba", "xwoba"
                                    ] if c in dive_rows.columns]
                                    st.dataframe(
                                        dive_rows[show_cols].rename(columns={
                                            "season": "Season",
                                            "pitches": "Pitches",
                                            "whiff_rate": "Whiff%",
                                            "k_rate": "K%",
                                            "barrel_rate": "Barrel%",
                                            "hard_hit_rate": "Hard Hit%",
                                            "avg_ev": "Avg EV",
                                            "xba": "xBA",
                                            "xwoba": "xwOBA",
                                        }).round(3),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                        elif not by_season_df.empty:
                            st.caption(
                                "Select 2+ seasons above to enable year-over-year trend view."
                            )


# ═══════════════════════════════════════════════════════════════════════════════
# 📡 PITCHING TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pitching:
    st.markdown("### Live Pitching Analysis")

    live_game   = get_todays_game(team_id)
    live_s      = get_game_summary(live_game) if live_game else None
    live_id     = live_s["game_id"] if live_s else None
    live_status = live_s["status"] if live_s else "None"
    is_active   = live_status in ("In Progress", "Manager challenge", "Final")

    import datetime as _dt_pit
    _pit_season = _dt_pit.date.today().year

    # ── Determine team sides ──────────────────────────────────────────────────
    if live_s:
        t_side        = "home" if live_s["home_id"] == team_id else "away"
        opp_s         = "away" if t_side == "home" else "home"
        opp_name_pit  = live_s["away_name"] if t_side == "home" else live_s["home_name"]
        opp_team_id   = live_s["away_id"]   if t_side == "home" else live_s["home_id"]
    else:
        t_side = opp_s = opp_name_pit = None
        opp_team_id = None

    # ── Team selector ─────────────────────────────────────────────────────────
    team_choices = [sel_team]
    if opp_name_pit:
        team_choices.append(opp_name_pit)
    pit_team_label = st.radio("Team", team_choices, horizontal=True, key=f"pit_team_radio_{team_id}")
    analyze_team_id = team_id if pit_team_label == sel_team else (opp_team_id or team_id)
    analyze_side    = (t_side if pit_team_label == sel_team else opp_s) if live_s else None

    # ── Load season pitching staff (always) ───────────────────────────────────
    with st.spinner(f"Loading {pit_team_label} pitching staff…"):
        season_staff: list[dict] = get_team_pitching_staff(analyze_team_id, _pit_season)

    # ── Also load today's game pitchers (for in-progress / final games) ───────
    game_pitchers: list[dict] = []
    active_pit: dict | None   = None
    if live_id and is_active and analyze_side:
        game_pitchers = get_game_pitchers(live_id, analyze_side)
        active_pit    = get_active_pitcher(live_id)

    # Build quick lookup: player_id → today's box-score row
    today_pit_map = {p["pitcher_id"]: p for p in game_pitchers}

    # Active pitcher on mound right now
    half_for_side = "top" if analyze_side == "away" else "bottom"
    active_id = (
        active_pit["pitcher_id"]
        if active_pit and active_pit.get("half") == half_for_side
        else None
    )

    # ── Pitcher selector dropdown ─────────────────────────────────────────────
    focus_pid:        int | None = None
    focus_sp:         str        = ""
    focus_role:       str        = "?"
    focus_season_row: dict       = {}
    focus_today_row:  dict       = {}

    if season_staff:
        def _staff_label(p: dict) -> str:
            active_marker = "🟢 NOW  " if p["player_id"] == active_id else ""
            today         = today_pit_map.get(p["player_id"])
            today_tag     = f"  |  Today: {today['ip']} IP {today['pitches_thrown']}P" if today else ""
            return (
                f"{active_marker}{p['name']} [{p['role']}]"
                f"  —  {p['ip']} IP  {p['era']} ERA{today_tag}"
            )

        label_map = {_staff_label(p): p for p in season_staff}

        # Default: active pitcher → else probable starter → else first entry
        probable_name = ""
        if live_s and analyze_side:
            probable_name = (
                live_s.get("home_probable_pitcher", "")
                if analyze_side == "home"
                else live_s.get("away_probable_pitcher", "")
            ) or ""

        default_idx = 0
        for i, (lbl, p) in enumerate(label_map.items()):
            if p["player_id"] == active_id:
                default_idx = i; break
            if probable_name and probable_name.lower() in p["name"].lower():
                default_idx = i

        sel_label    = st.selectbox(
            "Pitcher (season staff, sorted by IP)", list(label_map.keys()),
            index=default_idx, key=f"pit_select_{team_id}",
        )
        sel_row       = label_map[sel_label]
        focus_pid     = sel_row["player_id"]
        focus_sp      = sel_row["name"]
        focus_role    = sel_row["role"]
        focus_season_row = sel_row
        focus_today_row  = today_pit_map.get(focus_pid, {})

    else:
        # Roster fetch failed — manual fallback
        fallback_name = ""
        if live_s and analyze_side:
            fallback_name = (
                live_s.get("home_probable_pitcher", "")
                if analyze_side == "home"
                else live_s.get("away_probable_pitcher", "")
            ) or ""
        focus_sp = st.text_input("Pitcher name", value=fallback_name, key=f"pit_manual_{team_id}")
        if focus_sp and focus_sp not in ("TBD", ""):
            with st.spinner(f"Looking up {focus_sp}…"):
                focus_pid = lookup_player_id(focus_sp)
        focus_role = "SP"

    # ── Analysis panel ────────────────────────────────────────────────────────
    if not focus_pid:
        st.info("No pitcher selected. Enter a name above or wait for the game to start.")
    else:
        # Fetch handedness from MLB API (not a manual dropdown)
        with st.spinner("Fetching pitcher info…"):
            hand = get_pitcher_handedness(focus_pid)

        # Role / hand pill styles
        role_color = "#2196F3" if focus_role == "SP" else "#FF9800"
        hand_color = "#e74c3c" if hand == "R" else "#3498db" if hand == "L" else "#9b59b6"

        hs_col, info_col = st.columns([1, 5])
        with hs_col:
            _show_mlb_headshot(focus_pid, width=90)
        with info_col:
            st.markdown(
                f"### {focus_sp} &nbsp;"
                f'<span style="background:{role_color};color:white;border-radius:6px;'
                f'padding:2px 10px;font-size:0.78rem;font-weight:700">{focus_role}</span>'
                f'&nbsp;<span style="background:{hand_color};color:white;border-radius:6px;'
                f'padding:2px 10px;font-size:0.78rem;font-weight:700">Throws {hand}</span>',
                unsafe_allow_html=True,
            )
            if focus_season_row:
                st.caption(
                    f"Season — IP: **{focus_season_row['ip']}** | "
                    f"ERA: **{focus_season_row['era']}** | "
                    f"WHIP: **{focus_season_row['whip']}** | "
                    f"K: **{focus_season_row['k']}** | BB: **{focus_season_row['bb']}**"
                )
            if focus_today_row:
                st.caption(
                    f"Today — IP: **{focus_today_row['ip']}** | "
                    f"Pitches: **{focus_today_row['pitches_thrown']}** | "
                    f"K: **{focus_today_row['k']}** | BB: **{focus_today_row['bb']}** | "
                    f"ER: **{focus_today_row['er']}** | H: **{focus_today_row['h']}**"
                )
            if active_pit and active_pit.get("pitcher_id") == focus_pid:
                st.markdown(
                    '<span style="background:#2ecc71;color:black;border-radius:6px;'
                    'padding:2px 10px;font-size:0.78rem;font-weight:700">🟢 Currently on mound</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Live pitch movement ───────────────────────────────────────────────
        # Use pitcher_id from box score (not name-lookup) to guarantee correct filter
        if live_id and is_active:
            with st.spinner(f"Loading live pitches for {focus_sp}…"):
                live_pitches = get_pitcher_live_pitches(live_id, focus_pid)
        else:
            live_pitches = []

        if live_pitches:
            st.markdown(
                f'<div class="sec-hdr">Pitch Movement — Live ({len(live_pitches)} pitches)</div>',
                unsafe_allow_html=True,
            )
            mv_fig = plot_pitch_movement(live_pitches, focus_sp)
            st.plotly_chart(mv_fig, use_container_width=True)

            vf_fig = plot_velocity_fade(live_pitches, focus_sp)
            st.plotly_chart(vf_fig, use_container_width=True)

            fat    = compute_pitcher_fatigue(live_pitches)
            fg_fig = plot_fatigue_gauge(
                fat["pitch_count"], fat["vel_drop_pct"], fat["spin_drop_pct"],
                pitcher_name=focus_sp,
            )
            st.markdown('<div class="sec-hdr">Fatigue Indicator</div>', unsafe_allow_html=True)
            fg_col, fm_col = st.columns([2, 1])
            with fg_col:
                st.plotly_chart(fg_fig, use_container_width=True)
            with fm_col:
                st.metric("Pitch Count", fat["pitch_count"])
                st.metric("Velo Drop",   f"{fat['vel_drop_pct']:.1f}%")
                st.metric("Spin Drop",   f"{fat['spin_drop_pct']:.1f}%")

        else:
            # No live pitches for this pitcher → show their season Statcast data
            st.markdown(
                '<div class="sec-hdr">Pitch Mix (Season Data)</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "No live pitches recorded for this pitcher in today's game yet. "
                "Showing season Statcast data."
            )
            with st.spinner("Loading Statcast data…"):
                import datetime as _dt
                sc_df = fetch_statcast_csv(
                    focus_pid, "pitcher", season=_dt.date.today().year
                )

            if not sc_df.empty and "pfx_x" in sc_df.columns:
                season_pitches = (
                    sc_df[sc_df["pitch_type"].notna() & sc_df["pfx_x"].notna()]
                    .apply(lambda r: {
                        "pitch_type":    r.get("pitch_type", ""),
                        "pitch_name":    r.get("pitch_name", ""),
                        "speed":         r.get("release_speed"),
                        "pfx_x":         r.get("pfx_x"),
                        "pfx_z":         r.get("pfx_z"),
                        "plate_x":       r.get("plate_x"),
                        "plate_z":       r.get("plate_z"),
                        "spin_rate":     r.get("release_spin_rate"),
                        "inning":        r.get("inning"),
                        # game-break fields for velocity fade
                        "game_pk":       r.get("game_pk"),
                        "game_date":     r.get("game_date"),
                        "at_bat_number": r.get("at_bat_number"),
                        "pitch_number":  r.get("pitch_number"),
                    }, axis=1)
                    .tolist()
                )
                mv_fig = plot_pitch_movement(season_pitches, f"{focus_sp} (Season)")
                st.plotly_chart(mv_fig, use_container_width=True)

                vf_fig = plot_velocity_fade(season_pitches, f"{focus_sp} (Season)")
                st.plotly_chart(vf_fig, use_container_width=True)
            else:
                st.info("No Statcast pitch data available for this pitcher yet this season.")


# ═══════════════════════════════════════════════════════════════════════════════
# 📈 TRENDS TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    import datetime as _dt_tr
    _tr_season = _dt_tr.date.today().year

    st.markdown(f"### Rolling Stat Trends — {sel_team}")

    # ── Load team's top-5 OPS batters (season leaders) ────────────────────────
    with st.spinner(f"Loading {sel_team} batting leaders…"):
        batting_leaders = get_team_batting_leaders(team_id, _tr_season, n=5)

    # Build name→id map from leaders + STAR_BATTERS
    trend_id_map: dict[str, int] = {r["name"]: r["player_id"] for r in batting_leaders}
    for name, bid in STAR_BATTERS.items():
        if name not in trend_id_map:
            trend_id_map[name] = bid

    default_trend_batters = [r["name"] for r in batting_leaders]
    all_trend_options     = list(trend_id_map.keys())

    if batting_leaders:
        tops = ", ".join(
            f"{r['name']} ({r['ops']:.3f} OPS)" for r in batting_leaders
        )
        st.caption(f"Auto-selected top-5 OPS leaders: {tops}")

    trend_batters = st.multiselect(
        "Select Batters",
        all_trend_options,
        default=default_trend_batters,
        key=f"trend_batters_{team_id}",
    )

    # ── Opposing pitcher selector ──────────────────────────────────────────────
    opp_prob = ""
    if live_s:
        opp_prob = (
            live_s.get("away_probable_pitcher", "")
            if live_s["home_id"] == team_id
            else live_s.get("home_probable_pitcher", "")
        ) or ""

    trend_pitcher    = st.text_input(
        "Opposing Pitcher (for K-Rate comparison)",
        value=opp_prob,
        key=f"trend_pitcher_{team_id}",
    )
    trend_pitcher_id = lookup_player_id(trend_pitcher) if trend_pitcher else None

    # ── League avg K% (fetched once per season) ───────────────────────────────
    @st.cache_data(ttl=86400, show_spinner=False)
    def _lg_krate(season: int) -> float:
        return get_league_avg_krate(season)

    lg_kpct = _lg_krate(_tr_season)
    # MLB avg pitcher K/9 (2023-2025 ~8.5); use as reference on pitcher axis
    LG_AVG_K9 = 8.5

    # ── Per-batter expanders ───────────────────────────────────────────────────
    for bname in trend_batters:
        bid = trend_id_map.get(bname) or lookup_player_id(bname)
        if not bid:
            continue

        with st.expander(bname, expanded=(bname == trend_batters[0])):
            hs_c, nm_c = st.columns([1, 6])
            with hs_c:
                _show_mlb_headshot(bid, width=65)
            with nm_c:
                leader_row = next((r for r in batting_leaders if r["player_id"] == bid), None)
                if leader_row:
                    st.markdown(
                        f"**{bname}** — OPS: **{leader_row['ops']:.3f}** | "
                        f"AVG: **{leader_row['avg']}** | HR: **{leader_row['hr']}**"
                    )
                else:
                    st.markdown(f"**{bname}**")

            with st.spinner(f"Loading stats for {bname}…"):
                b_log     = get_batter_game_log(bid)
                barrel_df = get_barrel_trend(bid, n_games=15)

            c1, c2 = st.columns(2)
            with c1:
                ops_fig = plot_rolling_ops(b_log, bname)
                st.plotly_chart(ops_fig, use_container_width=True)
            with c2:
                brl_fig = plot_barrel_trend(barrel_df, bname)
                st.plotly_chart(brl_fig, use_container_width=True)

            # K-rate chart with league averages
            p_log = get_pitcher_game_log(trend_pitcher_id) if trend_pitcher_id else []
            krate_fig = plot_krate_chart(
                b_log, p_log, bname, trend_pitcher or "—",
                league_avg_kpct=lg_kpct,
                league_avg_k9=LG_AVG_K9 if p_log else None,
            )
            st.plotly_chart(krate_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 📋 LAST 3 GAMES TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_l3:
    with st.spinner("Loading last 3 games…"):
        past_games = get_last_n_completed_games(team_id, n=3)

    if not past_games:
        st.info("No completed games found.")
    else:
        for g_past in reversed(past_games):
            sp = get_game_summary(g_past)
            won = ((sp["home_id"] == team_id and sp["home_score"] > sp["away_score"]) or
                   (sp["away_id"] == team_id and sp["away_score"] > sp["home_score"]))
            result = "W" if won else "L"
            with st.expander(
                f"**{result}**  {sp['away_name']} {sp['away_score']} — "
                f"{sp['home_score']} {sp['home_name']}  ·  {sp.get('game_date','')}",
                expanded=False,
            ):
                try:
                    bx = get_live_box_score(sp["game_id"])
                    ls = get_linescore(sp["game_id"])
                    inn_df = build_inning_table(ls)
                    if not inn_df.empty:
                        ta = ls.get("teams",{}).get("away",{})
                        th = ls.get("teams",{}).get("home",{})
                        hdrs   = ["Team"] + [str(r) for r in inn_df["Inning"]] + ["R","H","E"]
                        away_r = [sp["away_name"]] + list(inn_df["Away"]) + [
                            ta.get("runs",0),ta.get("hits",0),ta.get("errors",0)]
                        home_r = [sp["home_name"]] + list(inn_df["Home"]) + [
                            th.get("runs",0),th.get("hits",0),th.get("errors",0)]
                        st.dataframe(pd.DataFrame([away_r,home_r],columns=hdrs),
                                     use_container_width=True,hide_index=True)

                    ts2  = "home" if sp["home_id"] == team_id else "away"
                    os2  = "away" if ts2 == "home" else "home"
                    on2  = sp["away_name"] if ts2 == "home" else sp["home_name"]

                    bb1,bb2 = st.tabs([f"{sel_team} Batting", f"{on2} Batting"])
                    with bb1:
                        df = build_batting_table(bx, ts2)
                        if not df.empty:
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.caption("No data.")
                    with bb2:
                        df = build_batting_table(bx, os2)
                        if not df.empty:
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.caption("No data.")

                    pb1, pb2 = st.tabs([f"{sel_team} Pitching", f"{on2} Pitching"])
                    with pb1:
                        df = build_pitching_table(bx, ts2)
                        if not df.empty:
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.caption("No data.")
                    with pb2:
                        df = build_pitching_table(bx, os2)
                        if not df.empty:
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.caption("No data.")
                except Exception as e:
                    st.warning(f"Could not load box score: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 L3 STATS TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    with st.spinner("Aggregating last 3 games…"):
        past3 = get_last_n_completed_games(team_id, n=3)

    if not past3:
        st.info("No completed games to aggregate.")
    else:
        gids  = [g["game_id"] for g in past3]
        dates = " · ".join(g.get("game_date","") for g in past3)
        st.caption(f"Games: {dates}")

        c_bat, c_pit = st.columns(2)
        with c_bat:
            st.markdown('<div class="sec-hdr">Batting — L3</div>', unsafe_allow_html=True)
            with st.spinner():
                bat_df = aggregate_batting_stats(gids, team_id)
            if not bat_df.empty:
                st.dataframe(bat_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No data.")
        with c_pit:
            st.markdown('<div class="sec-hdr">Pitching — L3</div>', unsafe_allow_html=True)
            with st.spinner():
                pit_df = aggregate_pitching_stats(gids, team_id)
            if not pit_df.empty:
                st.dataframe(pit_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No data.")


# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 UP NEXT TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_next:
    with st.spinner("Loading upcoming game…"):
        next_game = get_upcoming_game(team_id)

    if not next_game:
        st.info("No upcoming game found in next 7 days.")
    else:
        ns = get_game_summary(next_game)
        try:
            dt  = datetime.fromisoformat(ns["game_datetime"].replace("Z","+00:00")).astimezone(CT)
            gts = dt.strftime("%A, %B %-d  ·  %-I:%M %p CT")
        except Exception:
            gts = ns.get("game_date","")

        st.markdown(f"### {ns['away_name']} @ {ns['home_name']}")
        st.markdown(f"**{gts}**  ·  {ns['venue']}")
        st.markdown("---")

        sp1, sp2 = st.columns(2)
        sp1.metric(f"{ns['away_name']} SP", ns["away_probable_pitcher"])
        sp2.metric(f"{ns['home_name']} SP", ns["home_probable_pitcher"])

        st.markdown("---")

        try:
            ub = get_live_box_score(ns["game_id"])
        except Exception:
            ub = {}

        nc1, nc2 = st.columns(2)
        with nc1:
            _lineup_col(ub, "away", ns["away_id"], ns["away_name"])
        with nc2:
            _lineup_col(ub, "home", ns["home_id"], ns["home_name"])

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: MLB Stats API · Baseball Savant Statcast · "
           "Use 🔄 Refresh to update · Lineup projections from last 5 games")
