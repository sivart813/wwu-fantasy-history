"""Current-season all-play analysis for the 41st St Fellas Streamlit dashboard."""
import pandas as pd
import streamlit as st


def calculate_all_play(games, mapping, season):
    """Return standings, pairwise hypothetical records, weekly results, and skipped weeks.

    A week is eligible only when every rostered team has one recorded regular-season
    matchup and every recorded score is present and nonzero (to reject 0-0 placeholders).
    """
    roster = mapping.loc[mapping.season.eq(season), ['manager_id', 'manager_name']].drop_duplicates()
    names = dict(zip(roster.manager_id, roster.manager_name))
    ids = set(names)
    regular = games.loc[games.season.eq(season) & games.phase.eq('Regular season')].copy()
    weekly = []
    skipped = []
    actual = {i: {'W': 0, 'L': 0, 'T': 0, 'PF': 0.0, 'PA': 0.0} for i in ids}
    pairwise = {(i, j): [0, 0, 0] for i in ids for j in ids if i != j}
    all_play = {i: {'W': 0, 'L': 0, 'T': 0, 'expected': 0.0} for i in ids}

    for week, group in regular.groupby('week', sort=True):
        participants = pd.concat([group.manager1_id, group.manager2_id], ignore_index=True)
        scores = pd.concat([group.score1, group.score2], ignore_index=True)
        if (not ids or len(participants) != len(ids) or set(participants) != ids
                or participants.duplicated().any() or scores.isna().any()
                or (scores <= 0).any()):
            skipped.append(int(week))
            continue
        points = {}
        for r in group.itertuples():
            points[r.manager1_id] = float(r.score1)
            points[r.manager2_id] = float(r.score2)
            for manager, own, other in [(r.manager1_id, r.score1, r.score2),
                                        (r.manager2_id, r.score2, r.score1)]:
                a = actual[manager]
                a['PF'] += float(own)
                a['PA'] += float(other)
                a['W' if own > other else ('L' if own < other else 'T')] += 1
        n = len(points)
        for i, score in points.items():
            wins = sum(score > v for j, v in points.items() if j != i)
            losses = sum(score < v for j, v in points.items() if j != i)
            ties = n - 1 - wins - losses
            ap = all_play[i]
            ap['W'] += wins
            ap['L'] += losses
            ap['T'] += ties
            ap['expected'] += (wins + .5 * ties) / (n - 1) if n > 1 else 0
            weekly.append({'Week': int(week), 'Manager': names[i], 'Score': round(score, 2),
                           'Actual': 'W' if actual_week_outcome(group, i) == 1 else ('L' if actual_week_outcome(group, i) == -1 else 'T'),
                           'All-play W': wins, 'All-play L': losses, 'All-play T': ties,
                           'Expected wins': round((wins + .5 * ties) / (n - 1), 3) if n > 1 else 0})
            for j, opponent_score in points.items():
                if i == j:
                    continue
                pairwise[(i, j)][0 if score > opponent_score else (1 if score < opponent_score else 2)] += 1

    weeks = len(set(r['Week'] for r in weekly))
    records = []
    for i in sorted(ids, key=lambda key: names[key].casefold()):
        a, ap = actual[i], all_play[i]
        eq = ap['expected']
        records.append({'Manager': names[i], 'Actual W': a['W'], 'Actual L': a['L'],
                        'Actual T': a['T'], 'Actual record': f"{a['W']}-{a['L']}" + (f"-{a['T']}" if a['T'] else ''),
                        'All-play W': ap['W'], 'All-play L': ap['L'], 'All-play T': ap['T'],
                        'All-play record': f"{ap['W']}-{ap['L']}" + (f"-{ap['T']}" if ap['T'] else ''),
                        'All-play %': round(100 * (ap['W'] + .5 * ap['T']) / (weeks * (len(ids) - 1)), 1) if weeks and len(ids) > 1 else 0,
                        'Equivalent wins': round(eq, 2),
                        'Luck': round(a['W'] + .5 * a['T'] - eq, 2),
                        'PF': round(a['PF'], 2), 'PA': round(a['PA'], 2)})
    standings = pd.DataFrame(records)
    if not standings.empty:
        standings = standings.sort_values(['Equivalent wins', 'PF'], ascending=False).reset_index(drop=True)
        standings.insert(0, 'All-play rank', range(1, len(standings) + 1))
    matrix = pd.DataFrame(index=sorted(names.values()), columns=sorted(names.values()), dtype=object)
    for (i, j), (w, l, t) in pairwise.items():
        matrix.loc[names[i], names[j]] = f'{w}-{l}' + (f'-{t}' if t else '')
    for name in names.values():
        matrix.loc[name, name] = '—'
    return standings, matrix, pd.DataFrame(weekly), skipped, weeks



def calculate_schedule_swaps(games, mapping, season):
    """Hypothetical records: row manager's scores against column manager's opponents.

    The row manager's own schedule reproduces their actual record. When a borrowed
    schedule includes a game against the row manager, use the schedule owner
    as the opponent instead, so every valid week produces a result.
    Only complete, nonzero regular-season weeks are included.
    """
    roster = mapping.loc[mapping.season.eq(season), ['manager_id', 'manager_name']].drop_duplicates()
    names = dict(zip(roster.manager_id, roster.manager_name))
    ids = set(names)
    regular = games.loc[games.season.eq(season) & games.phase.eq('Regular season')]
    totals = {(row, col): [0, 0, 0] for row in ids for col in ids}
    valid_weeks = 0

    for week, group in regular.groupby('week', sort=True):
        participants = pd.concat([group.manager1_id, group.manager2_id], ignore_index=True)
        scores = pd.concat([group.score1, group.score2], ignore_index=True)
        if (not ids or len(participants) != len(ids) or set(participants) != ids
                or participants.duplicated().any() or scores.isna().any()
                or (scores <= 0).any()):
            continue
        valid_weeks += 1
        points, opponents = {}, {}
        for r in group.itertuples():
            points[r.manager1_id], points[r.manager2_id] = float(r.score1), float(r.score2)
            opponents[r.manager1_id], opponents[r.manager2_id] = r.manager2_id, r.manager1_id

        for row in ids:
            for col in ids:
                opponent = opponents[col]
                result = totals[(row, col)]
                if opponent == row and row != col:
                    # The schedule owner faced the row manager this week.
                    # Reverse that matchup: the row manager faces the schedule owner.
                    opponent = col
                a, b = points[row], points[opponent]
                result[0 if a > b else (1 if a < b else 2)] += 1

    ordered = sorted(ids, key=lambda i: names[i].casefold())
    matrix = pd.DataFrame(index=[names[i] for i in ordered],
                          columns=[names[i] for i in ordered], dtype=object)
    for row in ordered:
        for col in ordered:
            w, l, t = totals[(row, col)]
            matrix.loc[names[row], names[col]] = f'{w}-{l}' + (f'-{t}' if t else '')
    return matrix, valid_weeks


def actual_week_outcome(group, manager_id):
    for r in group.itertuples():
        if r.manager1_id == manager_id:
            return (r.score1 > r.score2) - (r.score1 < r.score2)
        if r.manager2_id == manager_id:
            return (r.score2 > r.score1) - (r.score2 < r.score1)
    return 0


def render_current_season(games, mapping, visible_names, show_table):
    season = int(games.season.max())
    st.subheader(f'🏈 {season} Current Season — All-Play')
    st.caption('Regular season only • Uses the latest season in your CSV files, independent of the historical sidebar season/game-type filters.')
    standings, matrix, weekly, skipped, weeks = calculate_all_play(games, mapping, season)
    if skipped:
        st.warning('Excluded incomplete or invalid weeks: ' + ', '.join(map(str, skipped)) + '. All-play and actual records below use the same verified weeks.')
    if not weeks:
        st.info('No complete, valid regular-season weeks found for this season.')
        return
    displayed = standings[standings.Manager.isin(visible_names)].copy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Completed weeks', weeks)
    c2.metric('Teams in league', len(standings))
    c3.metric('All-play opponents / week', max(0, len(standings) - 1))
    c4.metric('All-play matchups / team', weeks * max(0, len(standings) - 1))
    st.subheader('All-play standings & schedule luck')
    st.caption('Equivalent wins = sum of each week’s all-play winning percentage (ties count as half). Luck = actual wins + half actual ties − equivalent wins. Positive luck indicates favorable matchup outcomes.')
    columns = ['All-play rank', 'Manager', 'Actual record', 'All-play record', 'All-play %', 'Equivalent wins', 'Luck', 'PF', 'PA']
    show_table(displayed[columns], f'{season}_all_play_standings.csv', 500)
    st.subheader("What if everyone played everyone else's schedule?")
    st.caption('Rows keep their own weekly scores; columns borrow that manager’s actual opponents. '
               'The diagonal is each manager’s actual record. If the borrowed schedule owner '
               'played the row manager that week, the row manager faces the schedule owner '
               'instead. Every record uses all completed regular-season weeks.')
    swap_matrix, _ = calculate_schedule_swaps(games, mapping, season)
    swap_names = [n for n in swap_matrix.index if n in visible_names]
    shown_swaps = swap_matrix.loc[swap_names, swap_names]
    st.dataframe(shown_swaps, use_container_width=True,
                 height=min(650, 100 + 38 * len(swap_names)))
    st.download_button('Download schedule-swap matrix',
                       shown_swaps.rename_axis('Manager').reset_index().to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'{season}_schedule_swap_matrix.csv', mime='text/csv')
    st.subheader('What if everyone played everyone?')
    st.caption('Each cell is the row manager’s hypothetical W–L–T record against the column manager, comparing their scores in every completed week, regardless of their actual scheduled opponent.')
    shown = [n for n in matrix.index if n in visible_names]
    shown_matrix = matrix.loc[shown, shown]
    st.dataframe(shown_matrix, use_container_width=True, height=min(650, 100 + 38 * len(shown)))
    st.download_button('Download all-play matchup matrix', shown_matrix.rename_axis('Manager').reset_index().to_csv(index=False).encode('utf-8-sig'), file_name=f'{season}_all_play_matrix.csv', mime='text/csv')
    st.subheader('Weekly all-play results')
    st.caption('All-play wins and losses show how each manager scored against the rest of the league in that particular week.')
    show_table(weekly[weekly.Manager.isin(visible_names)].sort_values(['Week', 'Score'], ascending=[False, False]), f'{season}_weekly_all_play.csv', 480)
