"""Local Yahoo fantasy league history dashboard. Run: streamlit run fantasy_history_app.py"""
from pathlib import Path
import csv
import json
import pandas as pd
import streamlit as st
from ask_league import answer_question, EXAMPLES
from league_honors import calculate_honors, honor_totals
from current_season_all_play import render_current_season
from league_awards import render_league_awards

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='41st St Fellas', page_icon='🏈', layout='wide', initial_sidebar_state='expanded')

@st.cache_data(show_spinner=False)
def load_data(root_str):
    root = Path(root_str)
    mapping_file = root / 'manager_mapping.csv'
    if not mapping_file.exists():
        raise FileNotFoundError(f'Missing {mapping_file}')
    mapping = pd.read_csv(mapping_file, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required = {'season','team_id','team_name','manager_id','manager_name'}
    if not required.issubset(mapping.columns):
        raise ValueError(f'Mapping missing columns: {sorted(required - set(mapping.columns))}')
    mapping['season'] = pd.to_numeric(mapping['season'], errors='raise').astype(int)
    for col in ['team_id','manager_id','manager_name','team_name']:
        mapping[col] = mapping[col].str.strip()
    if mapping.duplicated(['season','team_id']).any():
        raise ValueError('Duplicate season/team_id in manager_mapping.csv')
    incomplete = mapping[(mapping.manager_id == '') | (mapping.manager_name == '')]
    if len(incomplete):
        raise ValueError(f'{len(incomplete)} team-season rows lack manager names or IDs')
    names = mapping[['manager_id','manager_name']].drop_duplicates()
    if names.duplicated('manager_id').any() or names.duplicated('manager_name').any():
        raise ValueError('Manager ID/name inconsistency in mapping. Check spelling and IDs.')
    lookups = mapping.set_index(['season','team_id'])
    games = []
    for file in sorted((root/'raw_history').glob('*/*_games.csv')):
        try:
            directory_year = int(file.parent.name)
        except ValueError:
            continue
        if not 2007 <= directory_year <= 2026:
            continue
        # Only read season aggregates, not week_01_games.csv etc.
        if file.name != f'{file.parent.name}_games.csv':
            continue
        try:
            season = int(file.parent.name)
        except ValueError:
            continue
        with file.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        maxweek = max(int(r['week']) for r in rows)
        expected_final_week = 17 if season >= 2021 else 16
        if maxweek > expected_final_week:
            raise ValueError(f'{file}: week {maxweek} exceeds season maximum {expected_final_week}')
        # An incomplete current season is valid; historical seasons should be complete.
        if season < 2026 and maxweek != expected_final_week:
            raise ValueError(f'{file}: expected final week {expected_final_week}, found {maxweek}')
        for r in rows:
            week = int(r['week'])
            t1, t2 = str(r['team1_id']).strip(), str(r['team2_id']).strip()
            try:
                a, b = lookups.loc[(season,t1)], lookups.loc[(season,t2)]
            except KeyError:
                raise ValueError(f'Missing manager mapping: {season} week {week}, {t1} vs {t2}')
            phase = str(r.get('phase','')).strip().lower()
            if phase in ('regular','regular_season','regular season'):
                phase = 'Regular season'
            elif phase in ('postseason','playoffs','playoff'):
                phase = 'Postseason / consolation'
            elif not phase:
                phase = 'Regular season' if week <= maxweek - 3 else 'Postseason / consolation'
            else:
                raise ValueError(f'Unknown phase {phase!r}: {season} week {week}')
            games.append(dict(season=season, week=week, phase=phase,
                              team1_id=t1, team2_id=t2,
                              manager1_id=a.manager_id, manager2_id=b.manager_id,
                              manager1=a.manager_name, manager2=b.manager_name,
                              team1=a.team_name, team2=b.team_name,
                              score1=float(r['team1_score']), score2=float(r['team2_score'])))
    df = pd.DataFrame(games)
    if df.empty:
        raise ValueError('No game files found under raw_history/<year>/<year>_games.csv')
    df['pair'] = df.apply(lambda r: '|'.join(sorted([r.team1_id,r.team2_id])), axis=1)
    if df.duplicated(['season','week','pair']).any():
        raise ValueError('Duplicate games found (same season/week/team pair)')
    if (df.manager1_id == df.manager2_id).any():
        raise ValueError('A game maps both sides to the same manager')
    champs_path = root/'yahoo_championships.csv'
    champs = pd.read_csv(champs_path, dtype=str, keep_default_na=False, encoding='utf-8-sig') if champs_path.exists() else pd.DataFrame()
    if not champs.empty:
        champs['season'] = pd.to_numeric(champs['season']).astype(int)
        champs = champs[champs.verification_status.str.upper().isin(['VERIFIED','VERIFIED_BRACKET_AND_MATCHUP'])].copy()
        champs = champs.merge(mapping[['season','team_id','manager_id','manager_name']], left_on=['season','champion_team_id'],right_on=['season','team_id'],how='left').rename(columns={'manager_id':'champion_id','manager_name':'champion_manager'}).drop(columns=['team_id'])
        champs = champs.merge(mapping[['season','team_id','manager_id','manager_name']], left_on=['season','runner_up_team_id'],right_on=['season','team_id'],how='left').rename(columns={'manager_id':'runner_up_id','manager_name':'runner_up_manager'}).drop(columns=['team_id'])
    return mapping, df.drop(columns='pair'), champs


def directional(g):
    cols = ['season','week','phase','manager_id','manager','opponent_id','opponent','team','opponent_team','points_for','points_against']
    a = g.rename(columns={'manager1_id':'manager_id','manager1':'manager','manager2_id':'opponent_id','manager2':'opponent','team1':'team','team2':'opponent_team','score1':'points_for','score2':'points_against'})[cols]
    b = g.rename(columns={'manager2_id':'manager_id','manager2':'manager','manager1_id':'opponent_id','manager1':'opponent','team2':'team','team1':'opponent_team','score2':'points_for','score1':'points_against'})[cols]
    out = pd.concat([a,b],ignore_index=True)
    out['W'] = (out.points_for > out.points_against).astype(int)
    out['L'] = (out.points_for < out.points_against).astype(int)
    out['T'] = (out.points_for == out.points_against).astype(int)
    out['margin'] = out.points_for - out.points_against
    return out


def matchup_streaks(rows):
    """Current consecutive W/L/T for each directional manager-opponent pairing.

    Chronology uses season and week; streaks span seasons, and only games
    included by the selected season-range and game-type filters count.
    """
    streaks = {}
    ordered = rows.sort_values(['season', 'week'], kind='stable')
    for (manager, opponent), group in ordered.groupby(['manager', 'opponent'], sort=False):
        current_type, count = None, 0
        for game in group.itertuples():
            outcome = 'W' if game.W else ('L' if game.L else 'T')
            if outcome == current_type:
                count += 1
            else:
                current_type, count = outcome, 1
        if current_type is not None:
            streaks[(manager, opponent)] = f'{current_type}{count}'
    return streaks


def summarize(rows, by):
    if rows.empty:
        return pd.DataFrame()
    s = rows.groupby(by, as_index=False).agg(Games=('W','size'), Wins=('W','sum'), Losses=('L','sum'), Ties=('T','sum'), PF=('points_for','sum'), PA=('points_against','sum'))
    s['Win %'] = ((s.Wins + .5*s.Ties) / s.Games * 100).round(1)
    s['PF'] = s.PF.round(2)
    s['PA'] = s.PA.round(2)
    s['Diff'] = (s.PF-s.PA).round(2)
    return s

st.title('🏈 41st St Fellas')
st.caption('Local, read-only dashboard • Historical and in-progress Yahoo fantasy football results')
with st.sidebar:
    st.header('Explore')
    page = st.radio('Page', ['Overview','Current Season','League Awards','All-time standings','Championships','Regular-season records','Head-to-head matrix','Manager profile','Season explorer','Game finder','Record book','Deep records','Rivalry lab','Ask the League'])
    if st.button('↻ Reload CSV files'):
        st.cache_data.clear()
        st.rerun()
try:
    mapping, games, champs = load_data(str(ROOT))
except Exception as e:
    st.error(str(e))
    st.info('Put this script in the project root next to manager_mapping.csv, yahoo_championships.csv, and raw_history/.')
    st.stop()

all_seasons = sorted(games.season.unique().tolist())
if min(all_seasons) > 2007:
    st.warning(f'Historical games begin in {min(all_seasons)}, not 2007. Install raw_history/2007–2013 and complete manager_mapping.csv to include older seasons.')
all_managers = dict(sorted(mapping[['manager_name','manager_id']].drop_duplicates().values.tolist()))
with st.sidebar:
    st.divider()
    years = st.slider('Season range', min(all_seasons), max(all_seasons), (min(all_seasons), max(all_seasons)))
    phase_choice = st.selectbox('Games to include', ['Regular season','All games','Postseason / consolation'])
    st.caption('Postseason includes consolation matchups. Championship results are verified separately.')
    config_file = ROOT / 'league_config.json'
    league_config = json.loads(config_file.read_text(encoding='utf-8')) if config_file.exists() else {}
    inactive_names = list(dict.fromkeys(league_config.get('inactive_managers', [])))
    show_inactive = st.checkbox('Include inactive managers (' + ', '.join(inactive_names) + ')', value=False, help='Hides these managers from leaderboards and selectors. Their games against active managers still count toward active managers’ records.')

subset = games[games.season.between(*years)].copy()
if phase_choice != 'All games':
    subset = subset[subset.phase == phase_choice]
d = directional(subset)
INACTIVE = set(inactive_names)
visible_names = sorted(n for n in mapping.manager_name.unique() if show_inactive or n not in INACTIVE)
visible_d = d[d.manager.isin(visible_names)].copy()
visible_mapping = mapping[mapping.manager_name.isin(visible_names)].copy()
visible_champs = champs if show_inactive or champs.empty else champs[champs.champion_manager.isin(visible_names)]
# Honors always use complete regular seasons, independent of sidebar game-type selection.
all_regular_rows = directional(games[games.season.between(*years) & games.phase.eq('Regular season')])
honors = calculate_honors(all_regular_rows, champs, playoff_slots_by_year=league_config.get('playoff_slots_by_year', {}))
visible_honors = honors[honors.manager.isin(visible_names)]
honors_summary = honor_totals(visible_honors)

def show_table(data, filename=None, height=480):
    st.dataframe(data, use_container_width=True, hide_index=True, height=height)
    if filename is not None:
        st.download_button('Download CSV', data.to_csv(index=False).encode('utf-8-sig'), file_name=filename, mime='text/csv')

if page == 'Overview':
    st.subheader('League at a glance')
    st.caption('Metrics respect season range and game-type filters. Inactive managers are hidden from rankings, but games against them still count.')
    career = summarize(visible_d, ['manager_id','manager'])
    title_counts = (champs[champs.season.between(*years)].groupby('champion_manager').size() if not champs.empty else pd.Series(dtype=int))
    active_games = subset[subset.manager1.isin(visible_names) | subset.manager2.isin(visible_names)]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric('Seasons selected', len(set(subset.season)))
    c2.metric('Managers displayed', len(visible_names))
    c3.metric('Matchups involving displayed managers', len(active_games))
    c4.metric('Average team score', f'{visible_d.points_for.mean():.1f}' if not visible_d.empty else '—')
    c5.metric('Verified championships', len(champs[champs.season.between(*years)]) if not champs.empty else 0)

    if not career.empty:
        career['Titles'] = career.manager.map(title_counts).fillna(0).astype(int)
        career['PPG'] = (career.PF / career.Games).round(2)
        career['Avg margin'] = (career.Diff / career.Games).round(2)
        st.subheader('Career leaders')
        col1,col2,col3,col4 = st.columns(4)
        for col, label, metric, format_value in [
            (col1,'Most wins','Wins',lambda v: str(int(v))),
            (col2,'Best win rate (30+ games)','Win %',lambda v: f'{v:.1f}%'),
            (col3,'Most points scored','PF',lambda v: f'{v:,.1f}'),
            (col4,'Most titles','Titles',lambda v: str(int(v)))]:
            candidates = career[career.Games >= 30] if metric == 'Win %' else career
            if not candidates.empty:
                leader = candidates.sort_values([metric,'Wins'], ascending=False).iloc[0]
                col.metric(label, leader.manager, format_value(leader[metric]))
        st.subheader('All-time leaderboard')
        if not honors_summary.empty:
            career = career.merge(honors_summary[['manager_id','Regular-season titles','Playoff appearances','Playoff runner-ups']], on='manager_id',how='left')
            for c in ['Regular-season titles','Playoff appearances','Playoff runner-ups']:
                career[c] = career[c].fillna(0).astype(int)
        board = career.sort_values(['Wins','Win %'],ascending=False)[['manager','Games','Wins','Losses','Ties','Win %','PPG','PF','PA','Diff','Titles'] + ([ 'Regular-season titles','Playoff appearances','Playoff runner-ups'] if not honors_summary.empty else [])]
        show_table(board,'dashboard_leaderboard.csv',400)
        c1,c2 = st.columns(2)
        with c1:
            st.subheader('Wins by manager')
            st.bar_chart(career.sort_values('Wins',ascending=False).set_index('manager')['Wins'])
        with c2:
            st.subheader('Points per game')
            st.bar_chart(career.sort_values('PPG',ascending=False).set_index('manager')['PPG'])

    st.subheader('Scoring trends by season')
    if not visible_d.empty:
        trend = visible_d.groupby('season').agg(Avg_score=('points_for','mean'),High_score=('points_for','max'),Games=('points_for','size')).reset_index()
        trend['Avg_score'] = trend.Avg_score.round(2)
        trend['High_score'] = trend.High_score.round(2)
        st.line_chart(trend.set_index('season')[['Avg_score','High_score']])
        show_table(trend.rename(columns={'season':'Season','Avg_score':'Average score','High_score':'Highest score','Games':'Team performances'}),'scoring_trends.csv',280)

    st.subheader('Biggest single-game scores')
    if not visible_d.empty:
        best = visible_d.sort_values('points_for',ascending=False).head(10)
        show_table(best[['season','week','manager','opponent','points_for','points_against','phase']].rename(columns={'points_for':'Score','points_against':'Opponent score'}),'highest_scores.csv',300)

elif page == 'Current Season':
    render_current_season(games, mapping, visible_names, show_table)

elif page == 'League Awards':
    render_league_awards(games, mapping, champs, visible_names, show_table)

elif page == 'All-time standings':
    st.subheader('Career standings')
    st.caption('Choose one or both game types for this table. This filter is independent of the sidebar game filter, so other pages stay unchanged.')
    standings_phases = st.multiselect(
        'Standings game types',
        options=['Regular season', 'Postseason / consolation'],
        default=['Regular season'],
        help='Postseason / consolation includes playoff and consolation matchups; the source files do not reliably distinguish those brackets.'
    )
    if not standings_phases:
        st.info('Select at least one game type to display standings.')
    else:
        standings_games = games[
            games.season.between(*years) & games.phase.isin(standings_phases)
        ].copy()
        standings_rows = directional(standings_games)
        standings_rows = standings_rows[standings_rows.manager.isin(visible_names)]
        summary = summarize(standings_rows, ['manager_id', 'manager'])
        if summary.empty:
            st.info('No games match these filters.')
        else:
            seasons_played = standings_rows.groupby('manager_id').season.nunique()
            summary['Seasons'] = summary.manager_id.map(seasons_played).fillna(0).astype(int)
            summary['PPG'] = (summary.PF / summary.Games).round(2)
            summary['PA/G'] = (summary.PA / summary.Games).round(2)
            summary['Diff/G'] = (summary.Diff / summary.Games).round(2)
            # Championship titles are season-level achievements, not game-type totals.
            # Show them only for a view that includes postseason results.
            if 'Postseason / consolation' in standings_phases and not champs.empty:
                eligible = champs[champs.season.between(*years)]
                cc = eligible.groupby('champion_id').size()
                rr = eligible.groupby('runner_up_id').size()
                summary['Titles'] = summary.manager_id.map(cc).fillna(0).astype(int)
                summary['Runner-ups'] = summary.manager_id.map(rr).fillna(0).astype(int)
                st.caption('Titles and runner-up finishes are verified season-level achievements, shown when postseason is selected.')
            # Honors are season-level and remain visible regardless of selected game type.
            if not honors_summary.empty:
                hcols = ['manager_id','Regular-season titles','Playoff appearances','Playoff rate','Playoff runner-ups']
                summary = summary.merge(honors_summary[hcols], on='manager_id', how='left')
                for col in hcols[1:]: summary[col] = summary[col].fillna(0)
            summary = summary.sort_values(['Wins', 'Win %', 'PF'], ascending=False).reset_index(drop=True)
            summary.insert(0, 'Rank', range(1, len(summary) + 1))
            st.caption(f"Showing {', '.join(standings_phases)} • {years[0]}–{years[1]} • {len(standings_games):,} matchups")
            show_table(summary, 'all_time_standings_filtered.csv')

elif page == 'Regular-season records':
    st.subheader('🏆 Best regular seasons in league history')
    st.caption('Ranks individual manager-seasons by regular-season record, using points scored as the tiebreaker. Postseason and consolation games are excluded. The 13-game and 14-game eras can be compared separately.')
    regular = games[games.season.between(*years) & games.phase.eq('Regular season')].copy()
    regular_rows = directional(regular)
    regular_rows = regular_rows[regular_rows.manager.isin(visible_names)].copy()
    if regular_rows.empty:
        st.info('No regular-season games in this range.')
    else:
        seasons_table = summarize(regular_rows, ['season', 'manager_id', 'manager'])
        seasons_table['Expected games'] = seasons_table['season'].apply(lambda y: 14 if y >= 2021 else 13)
        seasons_table['PPG'] = (seasons_table.PF / seasons_table.Games).round(2)
        seasons_table['Point diff/G'] = (seasons_table.Diff / seasons_table.Games).round(2)
        seasons_table['Complete'] = seasons_table.Games.eq(seasons_table['Expected games'])
        era_choice = st.radio('Regular-season length', ['Both eras', '13 games (through 2020)', '14 games (2021 onward)'], horizontal=True)
        records = seasons_table.copy()
        include_incomplete = st.checkbox('Include in-progress seasons in all-time record rankings', value=False, help='Current-year partial seasons should not compete with completed 13- or 14-game seasons.')
        if not include_incomplete:
            records = records[records.Complete]
        if era_choice.startswith('13 games'):
            records = records[records['Expected games'] == 13]
        elif era_choice.startswith('14 games'):
            records = records[records['Expected games'] == 14]
        complete_only = st.checkbox('Only show seasons with all scheduled regular-season games', value=True,
                                    help='Avoids ranking an incomplete season against a complete season. Bye weeks may affect this check in older formats.')
        if complete_only:
            records = records[records.Complete]
        st.caption('Rank: win percentage first, then total points scored (PF). Equal records are broken by PF. Compare PPG for scoring across the two schedule lengths.')
        records = records.sort_values(['Win %', 'PF', 'Diff'], ascending=[False, False, False]).reset_index(drop=True)
        if records.empty:
            st.warning('No manager-seasons match these settings. Uncheck the completeness filter if historical schedules included byes.')
        else:
            records.insert(0, 'Rank', range(1, len(records) + 1))
            display_cols = ['Rank', 'season', 'manager', 'Wins', 'Losses', 'Ties', 'Games', 'Win %', 'PF', 'PA', 'Diff', 'PPG', 'Point diff/G', 'Expected games', 'Complete']
            show_table(records[display_cols].rename(columns={'season':'Season', 'manager':'Manager'}), 'best_regular_seasons.csv', 550)
            leader = records.iloc[0]
            a,b,c = st.columns(3)
            a.metric('Best record', f"{int(leader.Wins)}-{int(leader.Losses)}-{int(leader.Ties)}")
            b.metric('Top-ranked season', f"{leader['manager']} ({int(leader.season)})")
            b.caption('Ranked by win % then points')
            c.metric('Points scored', f"{leader.PF:,.2f}")
        st.divider()
        st.subheader('Standings for a single season')
        seasons_available = sorted(regular_rows.season.unique().tolist(), reverse=True)
        selected_year = st.selectbox('Season', seasons_available, key='regular_standings_year')
        one = seasons_table[seasons_table.season == selected_year].copy()
        one = one.sort_values(['Win %', 'PF', 'Diff'], ascending=[False, False, False]).reset_index(drop=True)
        one.insert(0, 'Rank', range(1, len(one) + 1))
        st.caption(f'{selected_year}: {14 if selected_year >= 2021 else 13} scheduled regular-season weeks • standings exclude playoff and consolation scores' + (' • SEASON IN PROGRESS' if not one.Complete.all() else ''))
        if selected_year in set(honors.season):
            season_honors = honors[honors.season.eq(selected_year)][['manager_id','Finish','Regular-season champion','Made playoffs']]
            one = one.merge(season_honors, on='manager_id', how='left')
            one['Status'] = one.apply(lambda r: 'Regular-season champion' if r['Regular-season champion'] else ('Playoffs' if r['Made playoffs'] else 'Missed playoffs'),axis=1)
        else:
            one['Status'] = 'In progress'
        show_table(one[['Rank','manager','Wins','Losses','Ties','Games','Win %','PF','PA','Diff','PPG','Status']].rename(columns={'manager':'Manager'}), f'regular_season_{selected_year}_standings.csv', 420)

elif page == 'Head-to-head matrix':
    st.subheader('Head-to-head records')
    st.caption('Each cell shows W-L-T and the current matchup streak (W3 = row manager won the last 3 meetings; L2 = lost the last 2; T1 = most recent meeting tied). Streaks use selected seasons and game type, and span seasons. Hidden managers still count in career standings.')
    summary = summarize(visible_d,['manager_id','manager','opponent_id','opponent'])
    streaks = matchup_streaks(d)
    names = visible_names
    if not summary.empty:
        record = {(r.manager,r.opponent): f'{int(r.Wins)}-{int(r.Losses)}-{int(r.Ties)}' for r in summary.itertuples()}
        diff = {(r.manager,r.opponent): int(r.Wins)-int(r.Losses) for r in summary.itertuples()}
        matrix = pd.DataFrame(index=names,columns=names)
        for name in names:
            for opp in names:
                matrix.loc[name,opp] = ('—' if name == opp else
                    (f'{record[(name,opp)]}  ({streaks[(name,opp)]})' if (name,opp) in record else '—'))
        def shade(val, name, opp):
            if name == opp or (name,opp) not in diff:
                return 'background-color: #64748b20; color: inherit'
            n = diff[(name,opp)]
            return 'background-color: #16a34a35' if n > 0 else ('background-color: #dc262635' if n < 0 else 'background-color: #64748b35')
        style = pd.DataFrame([[shade(matrix.loc[n,o],n,o) for o in names] for n in names],index=names,columns=names)
        st.dataframe(matrix.style.apply(lambda _: style, axis=None),use_container_width=True,height=520)
        st.download_button('Download matrix (CSV; import as text in Excel)',matrix.reset_index(names='Manager').to_csv(index=False).encode('utf-8-sig'),file_name='h2h_matrix.csv')
        st.subheader('Matchup detail')
        c1,c2 = st.columns(2)
        first = c1.selectbox('Manager',names,index=0)
        second = c2.selectbox('Opponent',[n for n in names if n != first])
        detail = d[(d.manager == first)&(d.opponent == second)].sort_values(['season','week'],ascending=False)
        st.write(f'**{first} vs {second}:** {record.get((first,second),"No games")} · **Current streak:** {streaks.get((first,second),"—")}')
        show_table(detail[['season','week','phase','team','opponent_team','points_for','points_against','margin','W','L','T']], 'matchup_detail.csv',350)

elif page == 'Manager profile':
    manager = st.selectbox('Choose manager',visible_names)
    person = visible_d[visible_d.manager == manager]
    s = summarize(person,['manager'])
    if not s.empty:
        r = s.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Record',f'{r.Wins}-{r.Losses}-{r.Ties}')
        c2.metric('Win rate',f'{r["Win %"]}%')
        c3.metric('Points scored',f'{r.PF:,.2f}')
        c4.metric('Point differential',f'{r.Diff:+,.2f}')
        person_honors = visible_honors[visible_honors.manager.eq(manager)]
        st.subheader('Season honors')
        if not person_honors.empty:
            h1,h2,h3,h4 = st.columns(4)
            h1.metric('Regular-season titles', int(person_honors['Regular-season champion'].sum()))
            h2.metric('Playoff appearances', int(person_honors['Made playoffs'].sum()))
            h3.metric('Playoff runner-ups', int(person_honors['Playoff runner-up'].sum()))
            h4.metric('League titles', int(person_honors['League champion'].sum()))
            show_table(person_honors[['season','Finish','Wins','Losses','PF','Regular-season champion','Made playoffs','Playoff runner-up','League champion']], 'manager_honors.csv', 300)
        st.subheader('Performance by season')
        byseason = summarize(person,['season'])
        show_table(byseason,'manager_seasons.csv',340)
        st.subheader('Record against opponents')
        vs = summarize(person,['opponent']).sort_values('Wins',ascending=False)
        current_streaks = matchup_streaks(d)
        vs['Current streak'] = vs.opponent.map(lambda opponent: current_streaks.get((manager, opponent), '—'))
        show_table(vs,'manager_opponents.csv',340)
    st.subheader('Teams managed')
    teams = mapping[(mapping.manager_name == manager)&mapping.season.between(*years)][['season','team_name']].sort_values('season')
    show_table(teams,'manager_teams.csv',260)
    if not champs.empty:
        won = champs[(champs.champion_manager == manager)&champs.season.between(*years)]
        st.write('**Championship seasons:**', ', '.join(map(str,won.season.tolist())) or 'None')

elif page == 'Season explorer':
    season = st.selectbox('Season',sorted(set(subset.season),reverse=True) if not subset.empty else all_seasons[::-1])
    season_games = d[d.season == season]
    st.subheader(f'{season} standings (selected game type)')
    standings = summarize(season_games[season_games.manager.isin(visible_names)],['manager']).sort_values(['Wins','Win %'],ascending=False) if not season_games.empty else pd.DataFrame()
    show_table(standings,'season_standings.csv',350)
    st.subheader('Week-by-week results')
    g = subset[(subset.season == season) & (subset.manager1.isin(visible_names) | subset.manager2.isin(visible_names))]
    show_table(g[['week','phase','manager1','score1','manager2','score2']].sort_values('week'),'season_games.csv')
    if not champs.empty:
        winner = champs[champs.season == season]
        if not winner.empty:
            r = winner.iloc[0]
            st.success(f'Champion: {r.champion_manager} ({r.champion_team_name}) defeated {r.runner_up_manager} ({r.runner_up_team_name}), {r.champion_score}–{r.runner_up_score}')

elif page == 'Game finder':
    st.subheader('🔎 Game Finder')
    st.caption('Search actual weekly matchups. Choose a manager and season to see a complete schedule, or combine filters to find specific games. Filters here are independent of the sidebar season and game-type settings.')
    full = directional(games)
    # Show each matchup once when no manager is selected, but orient every
    # matchup from the chosen manager's perspective when one is selected.
    c1, c2, c3 = st.columns(3)
    manager_choice = c1.selectbox('Manager / schedule', ['All managers'] + visible_names, key='finder_manager')
    season_options = sorted(games.season.unique().tolist(), reverse=True)
    season_choice = c2.selectbox('Season', ['All seasons'] + season_options, key='finder_season')
    opponent_options = [n for n in visible_names if n != manager_choice]
    opponent_choice = c3.selectbox('Opponent', ['Any opponent'] + opponent_options, key='finder_opponent')
    c4, c5, c6 = st.columns(3)
    phase = c4.selectbox('Game type', ['All games', 'Regular season', 'Postseason / consolation'], key='finder_phase')
    result_choice = c5.selectbox('Result', ['All results', 'Wins', 'Losses', 'Ties'], key='finder_result')
    preset = c6.selectbox('Quick search', ['Custom filters', 'Highest-scoring games', 'Closest games', 'Biggest blowouts', '120+ point losses'], key='finder_preset')
    with st.expander('Advanced filters — weeks, points and margin', expanded=False):
        a, b = st.columns(2)
        week_range = a.slider('Week range', int(games.week.min()), int(games.week.max()),
                              (int(games.week.min()), int(games.week.max())), key='finder_weeks')
        season_range = b.slider('Season range', int(games.season.min()), int(games.season.max()),
                                (int(games.season.min()), int(games.season.max())), key='finder_years')
        p1, p2 = st.columns(2)
        points_min = p1.number_input('Minimum manager points', min_value=0.0, value=0.0, step=5.0, key='finder_min_pf')
        points_max = p2.number_input('Maximum manager points (0 = no limit)', min_value=0.0, value=0.0, step=5.0, key='finder_max_pf')
        m1, m2 = st.columns(2)
        margin_min = m1.number_input('Minimum absolute margin', min_value=0.0, value=0.0, step=5.0, key='finder_min_margin')
        margin_max = m2.number_input('Maximum absolute margin (0 = no limit)', min_value=0.0, value=0.0, step=5.0, key='finder_max_margin')
    if manager_choice == 'All managers':
        # Stable orientation for unfiltered matchups; one row per actual game.
        found = full[(full.manager_id == full.manager_id)].copy()
        found = found.sort_values(['season','week','manager_id']).drop_duplicates(['season','week','manager_id','opponent_id'])
        found = found[found.manager_id < found.opponent_id]
        found = found[found.manager.isin(visible_names) | found.opponent.isin(visible_names)]
    else:
        found = full[full.manager.eq(manager_choice)].copy()
    if season_choice != 'All seasons':
        found = found[found.season.eq(season_choice)]
    found = found[found.season.between(*season_range) & found.week.between(*week_range)]
    if opponent_choice != 'Any opponent':
        if manager_choice == 'All managers':
            found = found[found.manager.eq(opponent_choice) | found.opponent.eq(opponent_choice)]
        else:
            found = found[found.opponent.eq(opponent_choice)]
    if phase != 'All games':
        found = found[found.phase.eq(phase)]
    if result_choice != 'All results':
        field = {'Wins':'W','Losses':'L','Ties':'T'}[result_choice]
        found = found[found[field].eq(1)]
    found = found[found.points_for.ge(points_min)]
    if points_max > 0:
        found = found[found.points_for.le(points_max)]
    found = found[found.margin.abs().ge(margin_min)]
    if margin_max > 0:
        found = found[found.margin.abs().le(margin_max)]
    if preset == '120+ point losses':
        found = found[found.L.eq(1) & found.points_for.gt(120)]
    elif preset == 'Highest-scoring games':
        found = found.sort_values(['points_for','season','week'], ascending=[False,False,False])
    elif preset == 'Closest games':
        found = found.assign(_sort=found.margin.abs()).sort_values(['_sort','season','week'], ascending=[True,False,False])
    elif preset == 'Biggest blowouts':
        found = found.assign(_sort=found.margin.abs()).sort_values(['_sort','season','week'], ascending=[False,False,False])
    else:
        found = found.sort_values(['season','week'], ascending=[False,True] if manager_choice != 'All managers' and season_choice != 'All seasons' else [False,False])
    if found.empty:
        st.info('No matchups match these filters. Try broadening your search.')
    else:
        w, l, t = int(found.W.sum()), int(found.L.sum()), int(found['T'].sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Matching games', f'{len(found):,}')
        k2.metric('Record' if manager_choice != 'All managers' else 'Displayed-side record', f'{w}-{l}-{t}')
        k3.metric('Average points', f'{found.points_for.mean():.2f}')
        k4.metric('Average margin', f'{found.margin.mean():+.2f}')
        shown = found[['season','week','phase','manager','team','opponent','opponent_team','points_for','points_against','margin','W','L','T']].copy()
        shown['Result'] = shown.apply(lambda r: 'W' if r.W else ('L' if r.L else 'T'), axis=1)
        shown = shown.drop(columns=['W','L','T']).rename(columns={
            'season':'Season','week':'Week','phase':'Game type','manager':'Manager',
            'team':'Team','opponent':'Opponent','opponent_team':'Opponent team',
            'points_for':'Score','points_against':'Opponent score','margin':'Margin'})
        st.subheader('Matching games')
        show_table(shown, 'game_finder_results.csv', 580)
        if manager_choice == 'All managers':
            st.caption('When no manager is selected, each matchup appears once. Select a manager for their complete W-L-T schedule and perspective-specific filters.')
elif page == 'Championships':
    st.subheader('Regular-season champions, playoff qualifiers and championship finalists')
    st.caption('Top six by regular-season win percentage, then points scored. Incomplete seasons excluded. Championship finalists are verified from Yahoo results.')
    if not visible_honors.empty:
        st.subheader('All-time honors by manager')
        show_table(honors_summary.sort_values(['Regular-season titles','Playoff appearances'],ascending=False), 'league_honors.csv')
        st.subheader('Season-by-season qualification')
        season_view = visible_honors[['season','Finish','manager','Wins','Losses','Ties','PF','Regular-season champion','Made playoffs','Playoff runner-up','League champion']]
        show_table(season_view.sort_values(['season','Finish'],ascending=[False,True]),'playoff_qualifications.csv')
    st.subheader('Verified championship history')
    if champs.empty:
        st.warning('yahoo_championships.csv not found')
    else:
        filtered = visible_champs[visible_champs.season.between(*years)].sort_values('season',ascending=False)
        cols = ['season','champion_manager','champion_team_name','champion_score','runner_up_manager','runner_up_team_name','runner_up_score','margin']
        show_table(filtered[cols],'championship_history.csv')
        st.subheader('Titles by manager')
        totals = filtered.groupby('champion_manager').size().sort_values(ascending=False)
        st.bar_chart(totals)


elif page == 'Record book':
    st.subheader('League record book')
    st.caption('Top 50 per category • Records respect the selected seasons, game type, and visible managers. Games against hidden managers remain eligible.')
    records = visible_d.copy()
    if records.empty:
        st.info('No games match these filters.')
    else:
        base_cols = ['season', 'week', 'manager', 'opponent', 'points_for', 'points_against', 'margin', 'phase']

        def record_table(title, frame, filename, columns=None, height=480):
            st.subheader(title)
            if frame.empty:
                st.info('No qualifying records.')
                return
            result = frame.head(50).copy()
            result.insert(0, 'Rank', range(1, len(result) + 1))
            result.insert(1, 'Record', title)
            chosen = ['Rank', 'Record'] + (columns or base_cols)
            show_table(result[chosen].rename(columns={
                'season': 'Season', 'week': 'Week', 'manager': 'Manager',
                'opponent': 'Opponent', 'points_for': 'Score',
                'points_against': 'Opponent score', 'margin': 'Margin',
                'phase': 'Game type'
            }), filename, height)

        record_table('Highest weekly score', records.sort_values(['points_for', 'season', 'week'], ascending=[False, True, True]), 'record_high_scores.csv')
        record_table('Lowest weekly score', records.sort_values(['points_for', 'season', 'week'], ascending=[True, True, True]), 'record_low_scores.csv')
        record_table('Largest victory', records[records.W.eq(1)].sort_values(['margin', 'season', 'week'], ascending=[False, True, True]), 'record_biggest_wins.csv')
        record_table('Closest victory', records[records.W.eq(1)].sort_values(['margin', 'season', 'week'], ascending=[True, True, True]), 'record_closest_wins.csv')
        record_table('Highest score in a loss', records[records.L.eq(1)].sort_values(['points_for', 'season', 'week'], ascending=[False, True, True]), 'record_highest_losing_scores.csv')

        seasons = summarize(records, ['manager', 'season'])
        if not seasons.empty:
            seasons['PPG'] = (seasons.PF / seasons.Games).round(2)
            record_table('Best single season',
                         seasons.sort_values(['Win %', 'Wins', 'PF'], ascending=False),
                         'record_best_seasons.csv',
                         ['manager', 'season', 'Games', 'Wins', 'Losses', 'Ties', 'Win %', 'PF', 'PA', 'Diff', 'PPG'])

        # Keep each completed winning run as its own result, rather than only
        # retaining the single longest streak for each manager.
        streaks = []
        for name, group in records.sort_values(['season', 'week'], kind='stable').groupby('manager'):
            length = 0
            start = end = None
            for r in group.itertuples():
                if r.W:
                    if length == 0:
                        start = f'{r.season} W{r.week}'
                    length += 1
                    end = f'{r.season} W{r.week}'
                elif length:
                    streaks.append({'Manager': name, 'Consecutive wins': length, 'From': start, 'Through': end})
                    length = 0
            if length:
                streaks.append({'Manager': name, 'Consecutive wins': length, 'From': start, 'Through': end})
        if streaks:
            runs = pd.DataFrame(streaks).sort_values(['Consecutive wins', 'Manager'], ascending=[False, True])
            record_table('Longest winning streak', runs, 'winning_streaks.csv',
                         ['Manager', 'Consecutive wins', 'From', 'Through'])
        else:
            st.subheader('Longest winning streak')
            st.info('No qualifying winning streaks.')
        st.caption('Streaks span season boundaries and use the selected game type; season-range filters can truncate streaks. In-progress seasons can appear in single-game records.')


elif page == 'Deep records':
    st.subheader('🏆 Deep league records')
    st.caption('All records respect the selected years, game type, and visible managers. Opponents may be inactive. Tied games are excluded from win/loss-specific records. Season rankings use the same selected game type.')
    if visible_d.empty:
        st.info('No matching results.')
    else:
        base = visible_d.copy()
        base['Outcome'] = base.apply(lambda r: 'W' if r.W else ('L' if r.L else 'T'), axis=1)
        base['Matchup total'] = (base.points_for + base.points_against).round(2)
        base['Abs margin'] = base.margin.abs().round(2)
        base['Score'] = base.points_for.round(2)
        base['Opponent score'] = base.points_against.round(2)
        display = ['season','week','manager','opponent','Score','Opponent score','margin','phase']

        def leaderboard(title, frame, filename, note=None, columns=None):
            st.markdown(f'#### {title}')
            if note:
                st.caption(note)
            if frame.empty:
                st.info('No qualifying games.')
            else:
                show_table(frame[columns or display].head(20), filename, 340)

        tabs = st.tabs(['Wild games','Luck & pain','Season superlatives','Consistency','Streaks & runs'])
        with tabs[0]:
            leaderboard('Highest combined-score games',base.sort_values(['Matchup total','season'],ascending=False).drop_duplicates(['season','week', 'Matchup total', 'Abs margin']), 'highest_combined_games.csv', 'A matchup appears once in this table when both managers are visible.',display+['Matchup total'])
            leaderboard('Lowest combined-score games',base.sort_values('Matchup total').drop_duplicates(['season','week','Matchup total','Abs margin']), 'lowest_combined_games.csv',columns=display+['Matchup total'])
            leaderboard('Lowest winning scores',base[base.W.eq(1)].sort_values('Score'), 'lowest_winning_scores.csv')
            leaderboard('Largest losing margins',base[base.L.eq(1)].sort_values('margin'), 'worst_losses.csv')
            leaderboard('Narrowest losses',base[base.L.eq(1)].sort_values('Abs margin'), 'narrow_losses.csv')
        with tabs[1]:
            leaderboard('Highest-scoring losses',base[base.L.eq(1)].sort_values('Score',ascending=False),'unlucky_losses.csv')
            leaderboard('Lowest-scoring wins',base[base.W.eq(1)].sort_values('Score'),'fortunate_wins.csv')
            leaderboard('Most points allowed in a win',base[base.W.eq(1)].sort_values('Opponent score',ascending=False),'shootout_wins.csv')
            st.markdown('#### Win rate versus points per game')
            career = summarize(base,['manager'])
            career['PPG'] = (career.PF / career.Games).round(2)
            career['PA/G'] = (career.PA / career.Games).round(2)
            career['Expected wins (50% baseline)'] = (career.Games * .5).round(1)
            show_table(career[['manager','Games','Wins','Losses','Ties','Win %','PPG','PA/G','Diff']].sort_values('PPG',ascending=False),'luck_context.csv',350)
            st.caption('Scoring versus wins is descriptive, not a strength-of-schedule-adjusted luck metric.')
        with tabs[2]:
            season = summarize(base,['manager','season'])
            season['PPG'] = (season.PF / season.Games).round(2)
            season['PA/G'] = (season.PA / season.Games).round(2)
            season['Diff/G'] = (season.Diff / season.Games).round(2)
            season = season[season.Games >= 5].copy()
            if season.empty:
                st.info('Not enough games for season comparisons.')
            else:
                st.markdown('#### Most dominant seasons (point differential per game)')
                show_table(season.sort_values(['Diff/G','Win %'],ascending=False).head(20),'dominant_seasons.csv',340)
                st.markdown('#### Highest-scoring seasons (points per game)')
                show_table(season.sort_values('PPG',ascending=False).head(20),'scoring_seasons.csv',340)
                st.markdown('#### Toughest seasons (points allowed per game)')
                show_table(season.sort_values('PA/G',ascending=False).head(20),'toughest_seasons.csv',340)
                st.markdown('#### Most wins in a season')
                show_table(season.sort_values(['Wins','Win %'],ascending=False).head(20),'most_season_wins.csv',340)
                st.caption('Minimum five qualifying games; cross-era scoring rules may differ.')
        with tabs[3]:
            consistency = base.groupby('manager',as_index=False).agg(Games=('Score','size'),Average=('Score','mean'),Median=('Score','median'),StdDev=('Score','std'),Lowest=('Score','min'),Highest=('Score','max'),Above_100=('Score',lambda x: int((x >= 100).sum())),Above_150=('Score',lambda x: int((x >= 150).sum())))
            consistency = consistency[consistency.Games >= 10].copy()
            if not consistency.empty:
                consistency['100+ %'] = (100*consistency.Above_100/consistency.Games).round(1)
                consistency['150+ %'] = (100*consistency.Above_150/consistency.Games).round(1)
                consistency['StdDev'] = consistency.StdDev.round(2)
                for c in ['Average','Median','Lowest','Highest']:
                    consistency[c] = consistency[c].round(2)
                st.markdown('#### Most consistent scorers')
                st.caption('Lowest standard deviation of weekly scores; minimum 10 games. A low deviation does not necessarily mean strong scoring.')
                show_table(consistency.sort_values('StdDev'), 'consistency.csv', 450)
                st.markdown('#### Most 150+ point performances')
                show_table(consistency.sort_values(['Above_150','150+ %'],ascending=False), '150_point_club.csv',350)
            st.caption('Absolute scoring thresholds are sensitive to historical Yahoo scoring-setting changes.')
        with tabs[4]:
            st.markdown('#### Longest winning and losing streaks')
            runs = []
            for manager, group in base.sort_values(['season','week'],kind='stable').groupby('manager'):
                current = None; length = 0; start = None
                for r in group.itertuples():
                    result = 'W' if r.W else ('L' if r.L else 'T')
                    if result == current:
                        length += 1
                    else:
                        current, length, start = result, 1, f'{r.season} W{r.week}'
                    runs.append({'Manager':manager,'Type':result,'Length':length,'From':start,'Through':f'{r.season} W{r.week}'})
            runs = pd.DataFrame(runs)
            for kind,label in [('W','Winning runs'),('L','Losing runs')]:
                st.markdown(f'**{label}**')
                show_table(runs[runs.Type.eq(kind)].sort_values('Length',ascending=False).drop_duplicates(['Manager','Length','From']).head(15),f'{kind.lower()}_runs.csv',300)
            st.caption('Streaks use selected game type and seasons; playoff/consolation inclusion changes streak definitions.')

elif page == 'Rivalry lab':
    st.subheader('⚔️ Rivalry lab')
    st.caption('Head-to-head streaks and rivalry milestones; results use selected seasons and game type. Only visible managers can be selected.')
    if len(visible_names) < 2:
        st.info('At least two managers must be visible.')
    else:
        a_col,b_col = st.columns(2)
        a = a_col.selectbox('Manager A',visible_names,key='rivalry_a')
        b = b_col.selectbox('Manager B',[n for n in visible_names if n != a],key='rivalry_b')
        h = d[(d.manager == a) & (d.opponent == b)].sort_values(['season','week'],kind='stable').copy()
        if h.empty:
            st.info('No matchups for these filters.')
        else:
            h['Outcome'] = h.apply(lambda r: 'W' if r.W else ('L' if r.L else 'T'),axis=1)
            wins, losses, ties = int(h['W'].sum()), int(h['L'].sum()), int(h['T'].sum())
            st.metric(f'{a} vs {b}',f'{wins}-{losses}-{ties}',f'{len(h)} meetings',delta_color='off')
            streak = matchup_streaks(h).get((a,b),'—')
            best = {'W':(0,None,None),'L':(0,None,None)}
            last = None; count = 0; start = None
            for r in h.itertuples():
                if r.Outcome == last:
                    count += 1
                else:
                    last, count, start = r.Outcome, 1, f'{r.season} W{r.week}'
                if last in best and count > best[last][0]:
                    best[last] = (count,start,f'{r.season} W{r.week}')
            c1,c2,c3,c4 = st.columns(4)
            c1.metric('Current streak',streak)
            c2.metric(f'{a} longest run',f'W{best["W"][0]}')
            c3.metric(f'{b} longest run',f'W{best["L"][0]}')
            c4.metric('Average point margin',f'{h.margin.mean():+.2f}')
            st.caption(f'{a} best run: {best["W"][1] or "—"} → {best["W"][2] or "—"} · {b} best run: {best["L"][1] or "—"} → {best["L"][2] or "—"}')
            h['A cumulative wins'] = h.W.cumsum()
            h['B cumulative wins'] = h.L.cumsum()
            h['Meeting'] = range(1,len(h)+1)
            st.subheader('Rivalry momentum over time')
            st.line_chart(h.set_index('Meeting')[['A cumulative wins','B cumulative wins']])
            st.subheader('Closest meeting')
            show_table(h.assign(AbsMargin=h.margin.abs()).sort_values('AbsMargin').head(3)[['season','week','phase','manager','opponent','points_for','points_against','margin']],'closest_rivalry_games.csv',180)
            st.subheader('Biggest victory by each manager')
            for label, frame in [
                (a, h[h['W'].eq(1)].sort_values('margin', ascending=False)),
                (b, h[h['L'].eq(1)].sort_values('margin', ascending=True)),
            ]:
                st.markdown(f'**{label}**')
                if frame.empty:
                    st.write('No wins in selected games.')
                else:
                    result = frame.head(1).copy()
                    if label == b:
                        result['points_for'] = frame.head(1)['points_against'].to_numpy()
                        result['points_against'] = frame.head(1)['points_for'].to_numpy()
                        result['margin'] = -result['margin']
                    show_table(result[['season', 'week', 'phase', 'points_for', 'points_against', 'margin']], None, 90)
            st.subheader('Every meeting (newest first)')
            show_table(h.sort_values(['season','week'],ascending=False)[['season','week','phase','team','opponent_team','points_for','points_against','margin','Outcome']],'rivalry_history.csv',460)



elif page == 'Ask the League':
    st.subheader('🏈 Ask the League')
    st.caption('Free, verified answers from recorded league matchups. No AI API or usage charges.')
    st.info('Your sidebar season range, game type, and inactive-manager settings apply. 2026 currently includes only the completed weeks in your CSV.')
    example = st.selectbox('Or choose an example question', ['Choose an example…'] + EXAMPLES, key='ask_example')
    question = st.text_input('Ask a stats question', value='', placeholder='Who has the most wins all time?', key='ask_question')
    asked = question.strip() or (example if example != 'Choose an example…' else '')
    if asked:
        # Hide inactive managers from rankings, but retain their games as opponents.
        # H2H and other questions about an excluded manager require enabling them in sidebar.
        result = answer_question(asked, visible_d, visible_champs, visible_names, honors=honors, regular_rows=all_regular_rows)
        if result is None:
            st.warning('I cannot reliably answer that wording yet. Try one of the supported examples below.')
            st.write('This feature uses verified Python calculations rather than guessing.')
        else:
            st.success(result['title'])
            st.write(result['detail'])
            if not result['table'].empty:
                st.dataframe(result['table'], use_container_width=True, hide_index=True)
            if result['note']:
                st.caption(result['note'])
            if result.get('diagnostic'):
                with st.expander('How this question was interpreted'):
                    st.code(result['diagnostic'])
    else:
        st.markdown('**Start with an example, or type your own question.**')
    st.caption('Supported questions are intentionally limited. If a question is not recognized, no statistic is invented.')
