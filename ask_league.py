from ask_league_composable import answer_compound
"""Free, deterministic natural-language stats questions for the WWU Streamlit dashboard.
No network calls, model calls, or arbitrary code execution.
"""
import re
import pandas as pd

EXAMPLES = [
    'Who has the most wins all time?',
    'Who has the best winning percentage?',
    'Who has the most losses?',
    'Who has scored the most total points?',
    'Who averages the most points per game?',
    'Who has allowed the fewest points per game?',
    'What is the highest weekly score ever?',
    'What is the lowest weekly score ever?',
    'What is the highest score in a loss?',
    'What is the lowest winning score?',
    'What was the closest game ever?',
    'What was the biggest blowout ever?',
    'Who has the most 150-point games?',
    'Who has the most 100-point games?',
    'Who has the most wins in a season?',
    'Who has scored the most points in a season?',
    'Who has the longest winning streak?',
    'Who has the longest losing streak?',
    'Who has the most championships?',
    'Who has the most runner-up finishes?',
    'Who has the best record against Kruse?',
    'What is Kruse’s record against Blake?',
    'Who has beaten Blake the most?',
    'Who has the most ties?',
    'Who has the best point differential?',
]


def _clean(q):
    return re.sub(r'\s+', ' ', q.casefold().replace('’', "'").strip())


def _contains(q, *words):
    return any(x in q for x in words)


def _answer(title, detail, frame, note=''):
    return {'title': title, 'detail': detail, 'table': frame.reset_index(drop=True), 'note': note}


def _fmt(row, col, digits=2):
    return f'{float(row[col]):,.{digits}f}'


def _find_managers(q, names):
    # Whole-token matches prevent short names matching inside other words.
    hits = []
    for name in sorted(names, key=len, reverse=True):
        pat = r'(?<!\w)' + re.escape(name.casefold()) + r'(?!\w)'
        match = re.search(pat, q)
        if match:
            hits.append((match.start(), name))
    if "RJ/Kruse" in names and not any(n == "RJ/Kruse" for _, n in hits):
        alias = re.search(r"(?<!\w)(kruse|rj)(?!\w)", q)
        if alias:
            hits.append((alias.start(), "RJ/Kruse"))
    return [n for _, n in sorted(hits)]


def _career(rows):
    a = rows.groupby('manager', as_index=False).agg(
        Games=('W', 'size'), Wins=('W', 'sum'), Losses=('L', 'sum'),
        Ties=('T', 'sum'), PF=('points_for', 'sum'), PA=('points_against', 'sum'))
    a['Win %'] = (a.Wins + .5*a.Ties) / a.Games * 100
    a['PPG'] = a.PF / a.Games
    a['PAPG'] = a.PA / a.Games
    a['Diff'] = a.PF - a.PA
    return a


def _game_table(rows):
    return rows[['season', 'week', 'phase', 'manager', 'opponent', 'points_for', 'points_against', 'margin']].rename(columns={
        'season':'Season', 'week':'Week', 'phase':'Phase', 'manager':'Manager',
        'opponent':'Opponent', 'points_for':'Points', 'points_against':'Opponent points', 'margin':'Margin'})


def _streaks(rows, kind):
    out = []
    for name, group in rows.sort_values(['season', 'week']).groupby('manager'):
        count, first = 0, None
        for r in group.itertuples():
            good = (r.W == 1) if kind == 'W' else (r.L == 1)
            if good:
                if count == 0:
                    first = f'{r.season} W{r.week}'
                count += 1
                out.append({'Manager': name, 'Games': count, 'From': first, 'Through': f'{r.season} W{r.week}'})
            else:
                count = 0
    if not out:
        return pd.DataFrame(columns=['Manager', 'Games', 'From', 'Through'])
    return pd.DataFrame(out).sort_values('Games', ascending=False).drop_duplicates('Manager')


def answer_question(question, rows, champs, manager_names, min_games=5):
    """Returns result dict or None. rows = directional filtered game data."""
    q = _clean(question)
    if not q or rows.empty:
        return None
    names = _find_managers(q, manager_names)
    a = _career(rows)
    if a.empty:
        return None
    common_note = 'Uses the season range and game-type filters in the sidebar. Only recorded games are included.'
    # Head-to-head questions: all opponents of a named manager or a named pair.
    if _contains(q, 'against', 'versus', ' vs ', 'vs.', 'head to head', 'head-to-head', 'beaten', 'beat '):
        if len(names) >= 2:
            first, second = names[:2]
            h = rows[(rows.manager == first) & (rows.opponent == second)].sort_values(['season','week'])
            if h.empty:
                return _answer('No recorded meetings', f'No games between {first} and {second} match your filters.', pd.DataFrame(), common_note)
            w, l, t = int(h.W.sum()), int(h.L.sum()), int(h["T"].sum())
            return _answer(f'{first} vs. {second}: {w}–{l}–{t}', f'{len(h)} meetings; {first} has {w} wins, {second} has {l} wins, and {t} ties.', _game_table(h.sort_values(['season','week'], ascending=False)), common_note)
        if len(names) == 1:
            name = names[0]
            h = rows[rows.opponent == name]
            if h.empty:
                return _answer('No matching games', f'No games against {name} match your filters.', pd.DataFrame(), common_note)
            tab = h.groupby('manager', as_index=False).agg(Wins=('W','sum'), Losses=('L','sum'), Ties=('T','sum'), Games=('W','size'))
            tab['Win %'] = ((tab.Wins + .5*tab.Ties) / tab.Games * 100).round(1)
            if _contains(q, 'most', 'beaten', 'beat '):
                tab = tab.sort_values(['Wins','Win %'], ascending=False)
                stat = f'{int(tab.iloc[0].Wins)} wins'
                title = f'{tab.iloc[0].manager} has beaten {name} the most: {stat}'
            else:
                eligible = tab[tab.Games >= min_games]
                if eligible.empty:
                    return _answer('Not enough meetings', f'No manager has at least {min_games} games against {name} in this selection.', tab.rename(columns={'manager':'Manager'}), common_note)
                tab = eligible.sort_values(['Win %','Wins'], ascending=False)
                title = f'{tab.iloc[0].manager} has the best record against {name}: {tab.iloc[0]["Win %"]:.1f}%'
            return _answer(title, f'Head-to-head comparison against {name}.', tab.rename(columns={'manager':'Manager'}).round(2), common_note + f' Winning percentage rankings require at least {min_games} meetings.')
        return None

    if _contains(q, 'championship', 'champion', 'titles', 'title wins', 'runner-up', 'runner up', 'second place finishes'):
        if champs is None or champs.empty:
            return _answer('No verified championships', 'No verified championship records are available.', pd.DataFrame(), 'Championships are taken from the verified championship file.')
        c = champs[champs.season.between(int(rows.season.min()), int(rows.season.max()))]
        if c.empty:
            return _answer('No verified championships', 'No verified championships in the selected years.', pd.DataFrame(), 'Only verified championships count.')
        runner = _contains(q, 'runner-up', 'runner up', 'second place')
        col = 'runner_up_manager' if runner else 'champion_manager'
        t = c[col].dropna().value_counts().rename_axis('Manager').reset_index(name='Finishes' if runner else 'Titles')
        if t.empty:
            return None
        label = 'runner-up finishes' if runner else 'championships'
        return _answer(f'{t.iloc[0].Manager} leads with {int(t.iloc[0,1])} {label}', 'Verified championship games only.', t, 'Season range applies; game-type selection does not affect championships.')

    if _contains(q, 'winning streak', 'win streak', 'consecutive wins', 'losing streak', 'loss streak', 'consecutive losses'):
        kind = 'L' if _contains(q, 'losing', 'loss', 'losses') else 'W'
        tab = _streaks(rows, kind)
        if tab.empty:
            return None
        label = 'losing' if kind == 'L' else 'winning'
        return _answer(f'{tab.iloc[0].Manager}: longest {label} streak ({int(tab.iloc[0].Games)} games)', f'From {tab.iloc[0]["From"]} through {tab.iloc[0]["Through"]}.', tab.head(15), common_note + ' Streaks span seasons, using only games in the selected filters.')

    if _contains(q, 'season', 'single season') and _contains(q, 'most wins', 'best record', 'most points', 'highest points', 'scored the most'):
        tab = rows.groupby(['season','manager'], as_index=False).agg(Games=('W','size'), Wins=('W','sum'), Losses=('L','sum'), PF=('points_for','sum'))
        if _contains(q, 'wins', 'record'):
            tab = tab.sort_values(['Wins','Games'], ascending=False)
            value = f'{int(tab.iloc[0].Wins)} wins'
        else:
            tab = tab.sort_values('PF', ascending=False)
            value = f'{tab.iloc[0].PF:,.2f} points'
        tab = tab.rename(columns={'season':'Season','manager':'Manager','PF':'Points for'})
        return _answer(f'{tab.iloc[0].Manager} in {int(tab.iloc[0].Season)}: {value}', 'Best single-season totals in the selected data.', tab.head(20).round(2), common_note + ' In-progress seasons are included; totals are not projected.')

    # Weekly extremes and special results.
    if _contains(q, 'weekly score', 'single game', 'in a game', 'score ever', 'score in a', 'winning score', 'closest game', 'closest matchup', 'blowout', 'biggest win', 'margin of victory'):
        if _contains(q, 'loss', 'losing score'):
            subset = rows[rows.L == 1].sort_values('points_for', ascending=False)
            label = 'Highest score in a loss'
        elif _contains(q, 'lowest winning', 'lowest score to win'):
            subset = rows[rows.W == 1].sort_values('points_for')
            label = 'Lowest winning score'
        elif _contains(q, 'closest'):
            subset = rows[rows.margin > 0].sort_values('margin')
            label = 'Closest decided game'
        elif _contains(q, 'blowout', 'biggest win', 'margin of victory'):
            subset = rows[rows.W == 1].sort_values('margin', ascending=False)
            label = 'Biggest blowout'
        elif _contains(q, 'lowest', 'fewest', 'worst'):
            subset = rows.sort_values('points_for')
            label = 'Lowest weekly score'
        else:
            subset = rows.sort_values('points_for', ascending=False)
            label = 'Highest weekly score'
        if subset.empty:
            return None
        top = subset.iloc[0]
        return _answer(f'{label}: {top.manager} ({top.season} Week {top.week})', f'{top.points_for:.2f}–{top.points_against:.2f} against {top.opponent}; margin {top.margin:+.2f}.', _game_table(subset.head(15)), common_note)

    threshold = re.search(r'\b(\d{2,3})\s*[- ]?point games?\b', q)
    if threshold or _contains(q, '150+', '100+'):
        points = int(threshold.group(1)) if threshold else (150 if '150' in q else 100)
        tab = rows.assign(qualifies=rows.points_for >= points).groupby('manager', as_index=False).agg(Games=('qualifies','sum'))
        tab = tab[tab.Games > 0].sort_values('Games', ascending=False).rename(columns={'manager':'Manager'})
        if tab.empty:
            return _answer('No qualifying games', f'No recorded scores of {points}+ points match your filters.', tab, common_note)
        return _answer(f'{tab.iloc[0].Manager}: most {points}+ point games ({int(tab.iloc[0].Games)})', f'Count of games scoring at least {points} points.', tab.head(20), common_note)

    # All-time career totals and rate leaders.
    if _contains(q, 'winning percentage', 'win percentage', 'win rate', 'best record', 'highest win %'):
        eligible = a[a.Games >= min_games]
        if eligible.empty:
            return None
        tab = eligible.sort_values(['Win %','Wins'], ascending=False)
        label, val = 'best winning percentage', f'{tab.iloc[0]["Win %"]:.1f}%'
        note = common_note + f' Minimum {min_games} games for rate rankings.'
    elif _contains(q, 'most losses', 'most defeats'):
        tab = a.sort_values('Losses', ascending=False); label, val = 'most losses', str(int(tab.iloc[0].Losses)); note = common_note
    elif _contains(q, 'most ties'):
        tab = a.sort_values('Ties', ascending=False); label, val = 'most ties', str(int(tab.iloc[0].Ties)); note = common_note
    elif _contains(q, 'fewest points allowed', 'lowest points allowed', 'least points against'):
        tab = a[a.Games >= min_games].sort_values('PAPG'); label, val = 'fewest points allowed per game', f'{tab.iloc[0].PAPG:.2f}'; note = common_note
    elif _contains(q, 'average points', 'points per game', 'ppg'):
        tab = a[a.Games >= min_games].sort_values('PPG', ascending=False); label, val = 'highest points per game', f'{tab.iloc[0].PPG:.2f}'; note = common_note
    elif _contains(q, 'point differential', 'scoring differential'):
        tab = a.sort_values('Diff', ascending=False); label, val = 'best point differential', f'{tab.iloc[0].Diff:+,.2f}'; note = common_note
    elif _contains(q, 'most total points', 'most points', 'total points', 'scored the most'):
        tab = a.sort_values('PF', ascending=False); label, val = 'most total points', f'{tab.iloc[0].PF:,.2f}'; note = common_note
    elif _contains(q, 'most wins', 'winningest', 'all time wins'):
        tab = a.sort_values('Wins', ascending=False); label, val = 'most wins', str(int(tab.iloc[0].Wins)); note = common_note
    else:
        return None
    if tab.empty:
        return None
    tab = tab.rename(columns={'manager':'Manager'}).round(2)
    return _answer(f'{tab.iloc[0].Manager}: {label} ({val})', 'Career leaderboard for the selected seasons and game types.', tab.head(20), note)

# Additional composable handlers, while preserving the original question library.
from ask_league_advanced import answer_advanced, EXAMPLES_ADVANCED, _scope
from league_honors import answer_honors

_original_answer_question = answer_question
EXAMPLES = list(dict.fromkeys(EXAMPLES + EXAMPLES_ADVANCED + [
    'Who has the most regular-season championships?',
    'Who has the most playoff appearances?',
    'Who has the highest playoff qualification percentage?',
    'Who has the most playoff runner-up finishes?',
    'Who made the playoffs in 2024?',
    'Who was regular-season champion in 2023?',
    'Has anyone made the playoffs after starting 2-4?',
    'Has anyone won the championship after starting 2-5?',
    'Has anyone made the playoffs after starting 1-5 or worse?',
    'Has anyone won the league after losing 3 in a row during the season?',
    'Who won the championship despite losing four straight?',
    'Has anyone made the playoffs after losing 5 consecutive games?',
    'Has anyone won the regular season after losing 3 straight?',
]))


def answer_question(question, rows, champs, manager_names, min_games=5, honors=None, regular_rows=None):
    if honors is not None and regular_rows is not None:
        import re
        if re.search(r'\b(?:starting|started|start|after going|after an?)\s+\d{1,2}\s*-\s*\d{1,2}\b', question.casefold()):
            honors_answer = answer_honors(question, honors, regular_rows, manager_names)
            if honors_answer is not None:
                return honors_answer
        compound = answer_compound(question, regular_rows, honors, manager_names)
        if compound is not None:
            return compound
        honors_answer = answer_honors(question, honors, regular_rows, manager_names)
        if honors_answer is not None:
            return honors_answer
    advanced = answer_advanced(question, rows, champs, manager_names, min_games)
    if advanced is not None:
        return advanced
    # Legacy handlers now honor natural-language year filters as well as sidebar filters.
    filtered = _scope(_clean(question), rows)
    if filtered.empty:
        return {'title': 'No data in that period', 'detail': 'No games match those filters.',
                'table': pd.DataFrame(), 'note': ''}
    return _original_answer_question(question, filtered, champs, manager_names, min_games)
