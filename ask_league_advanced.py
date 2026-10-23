"""Composable, local-only statistical question handlers for Ask the League.
No API, eval, network, or access to files. All results come from supplied dataframes.
"""
import re
import pandas as pd

EXAMPLES_ADVANCED = [
    'Has anyone won a championship after starting 0-3?',
    'Has anyone won a championship after starting 2-5?',
    'Has anyone won a championship after starting 2-5 or worse?',
    'Has anyone made the championship game after starting 1-4?',
    'Who has the most wins while scoring under 100 points?',
    'Who has the most losses while scoring over 140 points since 2020?',
    'Who has the best record when scoring 120 or more?',
    'Who has the most 150-point games without a championship?',
    'Who has the most losses as the second highest scorer of the week?',
    'Who has the most wins as the lowest scorer of the week?',
    'Who is the luckiest manager by all-play wins?',
    'Who is the unluckiest manager by all-play wins?',
    'Who has the highest all-play winning percentage?',
    'What would the standings look like using all-play records in 2024?',
    'What would the standings look like if ranked by points scored in 2024?',
    'Who has the most 1-point victories?',
    'Who has the most losses by fewer than 5 points?',
    'Who has the most wins by 40 or more points?',
    'Who has the most weekly high scores?',
    'Who has the most weekly low scores?',
    'Who is the most consistent scorer?',
    'Who has the highest scoring volatility?',
    'What is the highest scoring game in 2022?',
    'Who has the best record in Week 1?',
    'Who has the most season opening wins?',
    'Who has the best record in the last 4 weeks of a season?',
    'Who has the most wins since 2020?',
    'Compare Kruse and Blake since 2020',
    'Who has the most playoff wins?',
    'Who has the highest playoff winning percentage?',
    'Who has the most championship losses?',
    'Who has the longest championship drought?',
    'Who won the championship in 2023?',
    'Which season had the highest average weekly score?',
    'Which season had the most blowouts?',
    'Who had the biggest year-over-year scoring improvement?',
    'Who has the most wins against eventual champions?',
    'Who has the best record against Blake since 2020?',
    'Who has the most 150-point games in 2024?',
]


def _res(title, detail, table=None, note=''):
    return {'title': str(title), 'detail': str(detail), 'table': table.reset_index(drop=True) if table is not None else pd.DataFrame(), 'note': note}


def _name_matches(q, names):
    matches = []
    for n in sorted(names, key=len, reverse=True):
        aliases = [n]
        if n.casefold() == 'rj/kruse':
            aliases.extend(['kruse', 'rj'])
        for alias in aliases:
            m = re.search(r'(?<!\w)' + re.escape(alias.casefold()) + r'(?!\w)', q)
            if m:
                matches.append((m.start(), n))
                break
    return [name for _, name in sorted(matches)]


def _scope(q, rows):
    """Apply explicit year and phase filters; sidebar filtering has already happened."""
    result = rows
    yr = re.search(r'\b(?:in|during)\s+(20\d{2})\b', q)
    since = re.search(r'\b(?:since|from)\s+(20\d{2})\b', q)
    before = re.search(r'\bbefore\s+(20\d{2})\b', q)
    between = re.search(r'\bbetween\s+(20\d{2})\s+and\s+(20\d{2})\b', q)
    last = re.search(r'\b(?:last|past)\s+(\d{1,2})\s+(?:completed\s+)?seasons\b', q)
    if between:
        lo, hi = map(int, between.groups())
        result = result[result.season.between(lo, hi)]
    elif since:
        result = result[result.season >= int(since.group(1))]
    elif before:
        result = result[result.season < int(before.group(1))]
    elif yr:
        result = result[result.season == int(yr.group(1))]
    elif last and not result.empty:
        seasons = sorted(result.season.unique())
        # A partial season remains eligible for general stats; no completeness inferred.
        chosen = seasons[-int(last.group(1)):]
        result = result[result.season.isin(chosen)]
    if 'regular season' in q:
        result = result[result.phase.eq('Regular season')]
    elif re.search(r'\b(?:playoff|postseason)\b', q):
        result = result[result.phase.ne('Regular season')]
    return result


def _leader(tab, metric, ascending=False, min_games=0, fmt=None, note=''):
    if min_games and 'Games' in tab:
        tab = tab[tab.Games >= min_games]
    if tab.empty:
        return _res('No qualifying results', 'No recorded games meet these conditions.', tab, note)
    tab = tab.sort_values([metric, 'Manager'] if 'Manager' in tab else [metric], ascending=[ascending, True] if 'Manager' in tab else ascending)
    top = tab.iloc[0]
    val = fmt(top[metric]) if fmt else f'{top[metric]:,.2f}' if isinstance(top[metric], float) else str(top[metric])
    return _res(f'{top.Manager}: {val}', f'Ranked by {metric}.', tab.head(30).round(2), note)


def _record_start(q, rows, champs, names):
    match = re.search(r'\b(?:starting|started|start|after going|after an?)\s+(\d{1,2})\s*[-–]\s*(\d{1,2})\b', q)
    if not match or not re.search(r'\b(champion|championship|title|playoff|finals)\b', q):
        return None
    wins, losses = map(int, match.groups())
    if wins + losses == 0 or wins + losses > 18:
        return _res('Invalid starting record', 'Please use a starting record of at least one and at most 18 games.')
    if champs is None or champs.empty:
        return _res('Championship data unavailable', 'Verified championship results are required for this question.')
    # Historical rows may include consolation; the starting record is regular season only.
    regular = rows[rows.phase.eq('Regular season')].sort_values(['season', 'week'])
    if regular.empty:
        return _res('Regular-season data unavailable', 'Select regular-season or all games in the sidebar.')
    required = wins + losses
    candidates = []
    worse = 'or worse' in q
    for (season, manager), g in regular.groupby(['season', 'manager']):
        first = g.head(required)
        if len(first) < required:
            continue
        w, l, t = int(first.W.sum()), int(first.L.sum()), int(first['T'].sum())
        if t:  # 2-5 means exactly seven decided games, not 2-4-1.
            continue
        matches = (w <= wins and l >= losses) if worse else (w == wins and l == losses)
        if not matches:
            continue
        final = g
        candidates.append({'Season': int(season), 'Manager': manager, 'Start': f'{w}-{l}',
                           'Final regular season': f'{int(final.W.sum())}-{int(final.L.sum())}-{int(final['T'].sum())}',
                           'Games recorded': len(final)})
    candidate_df = pd.DataFrame(candidates)
    if candidate_df.empty:
        return _res('No qualifying starts', f'No manager in the selected data started {wins}-{losses}{" or worse" if worse else ""}.')
    c = champs[champs.season.isin(candidate_df.Season)]
    # This is intentionally a championship winner/runner-up query; playoff qualification
    # cannot be inferred reliably from consolation-labelled matchups.
    if 'made the playoffs' in q or 'make the playoffs' in q or 'reached the playoffs' in q:
        return _res('Playoff qualification unavailable', 'The available data does not reliably identify playoff qualifiers separately from consolation teams. I can verify championship winners or finalists.')
    finalist = bool(re.search(r'(made|reached|appear|got to).{0,20}championship|championship game|finals|runner.up', q))
    allowed = c[['season', 'champion_manager', 'runner_up_manager']].copy()
    if finalist:
        ok = {(int(r.season), n) for r in allowed.itertuples() for n in (r.champion_manager, r.runner_up_manager)}
        label = 'reached the championship game'
    else:
        ok = {(int(r.season), r.champion_manager) for r in allowed.itertuples()}
        label = 'won the championship'
    candidate_df['Qualified'] = candidate_df.apply(lambda r: (r.Season, r.Manager) in ok, axis=1)
    winners = candidate_df[candidate_df.Qualified].drop(columns='Qualified')
    condition = f'{wins}-{losses}{" or worse" if worse else ""}'
    if winners.empty:
        return _res(f'No verified finalists' if finalist else 'No verified champions',
                    f'None of the {len(candidate_df)} qualifying manager-seasons starting {condition} {label}.',
                    candidate_df.drop(columns='Qualified'),
                    'Starting record means the first N regular-season games, not necessarily calendar week N. Only verified championship results count.')
    return _res(f'Yes — {len(winners)} manager-season(s) {label} after starting {condition}',
                f'{len(candidate_df)} manager-seasons matched the starting record. These {len(winners)} {label}.',
                winners, 'Starting record means the first N regular-season games. Only verified championship results count.')


def _threshold(q, rows, champs, min_games):
    # Generic combinatorial conditions: wins/losses, scored thresholds, margin thresholds, date range.
    outcome = None
    if re.search(r'\b(loss|losses|lost|defeats)\b', q):
        outcome = 'L'
    elif re.search(r'\b(win|wins|won|victories)\b', q):
        outcome = 'W'
    score = re.search(r'\b(?:scor(?:ing|ed|e)|putting up|with)\s+(?:at least|over|above|more than|under|below|fewer than|less than|at most)?\s*(\d{2,3})(?:\s*\+)?\s*(?:points|pts)?\b', q)
    # Explicit '150-point games' and '150+ point games'.
    point_games = re.search(r'\b(\d{2,3})\s*\+?\s*[- ]point games?\b', q)
    if not score and point_games:
        score = point_games
    margin = re.search(r'\b(?:by|decided by|margin of)\s+(?:fewer than|less than|under|within|at least|over|more than|greater than)\s+(\d{1,3})(?:\s*points?)?\b', q)
    close = re.search(r'\b(\d{1,3})[- ]point (?:victories|wins|losses|defeats)\b', q)
    if close and not margin:
        margin = close
        outcome = 'L' if re.search(r'loss|defeat', q) else 'W'
    if not score and not margin:
        return None
    # Avoid treating season years as scores and prevent broad matches like 'who scored the most points'.
    if score:
        value = int(score.group(1))
        if not 1 <= value <= 500:
            return None
    if not outcome and not point_games and not score:
        return None
    df = rows.copy()
    if outcome:
        df = df[df[outcome].eq(1)]
    if score:
        text_before = q[max(0, score.start()-32):score.start()]
        low = bool(re.search(r'(under|below|fewer than|less than|at most)', text_before))
        strict = bool(re.search(r'(over|above|more than|under|below|fewer than|less than)', text_before))
        if low:
            df = df[df.points_for.lt(value) if strict else df.points_for.le(value)]
        else:
            df = df[df.points_for.gt(value) if strict else df.points_for.ge(value)]
    if margin:
        value = int(margin.group(1))
        context = q[max(0, margin.start()-10):margin.start()+25]
        if close and margin is close:
            df = df[df.margin.abs() <= value]
        elif re.search(r'(fewer than|less than|under|within)', context):
            df = df[df.margin.abs() < value]
        else:
            df = df[df.margin.abs() >= value]
    if 'without a championship' in q or 'never won a championship' in q:
        if champs is None or champs.empty:
            return _res('Championship data unavailable', 'Cannot verify which managers have won titles.')
        winners = set(champs.champion_manager.dropna())
        df = df[~df.manager.isin(winners)]
    if df.empty:
        return _res('No matching games', 'No games meet the requested conditions.')
    tab = df.groupby('manager', as_index=False).agg(Games=('W','size'), Wins=('W','sum'), Losses=('L','sum'))
    tab['Win %'] = ((tab.Wins / tab.Games) * 100).round(1)
    tab = tab.rename(columns={'manager': 'Manager'})
    rate = bool(re.search(r'(best record|worst record|winning percentage|win rate)', q))
    if rate:
        if outcome:
            # Conditional win rate with a forced win/loss is uninformative.
            return _res('Ambiguous winning percentage', 'For a meaningful conditional winning percentage, omit “wins” or “losses” from the question.')
        tab = tab[tab.Games >= min_games]
        if tab.empty:
            return _res('Not enough qualifying games', f'Minimum {min_games} qualifying games required for rate rankings.')
        metric = 'Win %'
        ascending = 'worst' in q
    else:
        metric = 'Games'
        ascending = 'fewest' in q or 'least' in q
    result = _leader(tab, metric, ascending, note=f'Counts directional manager-games; rate rankings require at least {min_games} qualifying games. Filters: {q}.')
    return result


def _all_play(q, rows, min_games):
    if not any(term in q for term in ('all-play', 'all play', 'luckiest', 'unluckiest', 'schedule luck', 'top-six', 'top six', 'standings look like if ranked by points')):
        return None
    regular = rows[rows.phase.eq('Regular season')].copy()
    if regular.empty:
        return _res('No regular-season games', 'All-play compares scores among managers in each regular-season week.')
    # Directional rows are one row per manager per actual matchup. Use unique manager-week.
    regular = regular.sort_values(['season','week']).drop_duplicates(['season','week','manager'])
    entries = []
    for (_, _), week in regular.groupby(['season','week']):
        for r in week.itertuples():
            other = week[week.manager != r.manager].points_for
            entries.append({'Manager':r.manager, 'Season':int(r.season), 'AW':int((other < r.points_for).sum()),
                            'AL':int((other > r.points_for).sum()), 'AT':int((other == r.points_for).sum()),
                            'Actual W':int(r.W), 'Games':1, 'PF':float(r.points_for)})
    tab = pd.DataFrame(entries).groupby('Manager', as_index=False).agg({'AW':'sum','AL':'sum','AT':'sum','Actual W':'sum','Games':'sum','PF':'sum'})
    tab['All-play %'] = ((tab.AW + 0.5*tab.AT)/(tab.AW+tab.AL+tab.AT)*100).round(1)
    tab['All-play equivalent wins'] = ((tab.AW + .5*tab.AT)/(tab.AW+tab.AL+tab.AT)*tab.Games).round(2)
    tab['Luck (actual minus expected)'] = (tab['Actual W']-tab['All-play equivalent wins']).round(2)
    tab = tab[tab.Games >= min_games]
    if tab.empty:
        return _res('Not enough data', f'Need at least {min_games} games for all-play comparisons.')
    if 'unluckiest' in q or 'hurt most' in q:
        metric, asc = 'Luck (actual minus expected)', True
    elif 'luckiest' in q or 'benefited most' in q:
        metric, asc = 'Luck (actual minus expected)', False
    elif 'winning percentage' in q or 'best' in q or 'highest' in q:
        metric, asc = 'All-play %', False
    elif 'points' in q and 'ranked' in q:
        metric, asc = 'PF', False
    else:
        metric, asc = 'All-play equivalent wins', False
    note = 'All-play compares each recorded manager to every other recorded manager in the same regular-season week. Equivalent wins = all-play percentage × games. Luck = actual wins − equivalent wins; this is a descriptive metric, not proof of luck.'
    return _leader(tab, metric, asc, note=note)


def _weekly_rank(q, rows):
    high = bool(re.search(r'(highest scorer|weekly high|weekly top|top scorer|most weekly high|second.highest scorer)', q))
    low = bool(re.search(r'(lowest scorer|weekly low|bottom scorer|most weekly low)', q))
    if not high and not low:
        return None
    df = rows[rows.phase.eq('Regular season')].copy()
    if df.empty:
        return None
    df['rank_desc'] = df.groupby(['season','week']).points_for.rank(method='min', ascending=False)
    df['rank_asc'] = df.groupby(['season','week']).points_for.rank(method='min')
    if 'second' in q and high:
        df = df[df.rank_desc.eq(2)]
    elif high:
        df = df[df.rank_desc.eq(1)]
    else:
        df = df[df.rank_asc.eq(1)]
    if re.search(r'\b(loss|losses|lost)\b', q):
        df = df[df.L.eq(1)]
    elif re.search(r'\b(win|wins|won)\b', q):
        df = df[df.W.eq(1)]
    if df.empty:
        return _res('No matching weekly rankings', 'No recorded weeks match those conditions.')
    tab = df.groupby('manager', as_index=False).size().rename(columns={'manager':'Manager','size':'Weeks'})
    return _leader(tab, 'Weeks', note='Tied weekly ranks count for each tied manager. Regular-season weeks only.')


def _consistency(q, rows, min_games):
    if not any(x in q for x in ('consistent', 'consistency', 'volatile', 'volatility', 'standard deviation', 'scoring floor', 'scoring ceiling')):
        return None
    tab = rows.groupby('manager').points_for.agg(['size','mean','std','min','max']).reset_index().rename(columns={'manager':'Manager','size':'Games','mean':'Average','std':'Std dev','min':'Floor','max':'Ceiling'})
    tab = tab[tab.Games >= max(min_games, 2)]
    if tab.empty:
        return _res('Not enough games', 'Not enough recorded scores to measure consistency.')
    if 'ceiling' in q:
        metric, asc = 'Ceiling', False
    elif 'floor' in q:
        metric, asc = 'Floor', False
    else:
        metric, asc = 'Std dev', not any(x in q for x in ('least consistent', 'most volatile', 'highest volatility', 'highest standard deviation'))
    return _leader(tab, metric, asc, note='Consistency uses population of recorded games per manager; standard deviation is the sample standard deviation. Smaller means more consistent.')


def _champions(q, rows, champs):
    if champs is None or champs.empty:
        return None
    c = champs[champs.season.between(int(rows.season.min()), int(rows.season.max()))] if not rows.empty else champs.iloc[0:0]
    if c.empty:
        return None
    year = re.search(r'\b(?:in|of|for)\s+(20\d{2})\b', q)
    if year and re.search(r'(who won|champion|runner.up)', q):
        year = int(year.group(1))
        sub = c[c.season.eq(year)]
        if sub.empty:
            return _res('No verified champion', f'No verified championship for {year} within the selected seasons.')
        r = sub.iloc[0]
        return _res(f'{year} champion: {r.champion_manager}', f'Runner-up: {r.runner_up_manager}.', sub[['season','champion_manager','runner_up_manager']].rename(columns={'season':'Season','champion_manager':'Champion','runner_up_manager':'Runner-up'}), 'Only verified championships are used.')
    if 'championship drought' in q:
        last_year = int(c.season.max())
        managers = sorted(set(rows.manager.unique()))
        winners = c.groupby('champion_manager').season.max().to_dict()
        start_year = int(rows.season.min())
        tab = pd.DataFrame([{'Manager':m, 'Seasons since last title':last_year-int(winners[m]) if m in winners else last_year-start_year+1, 'Last title':int(winners[m]) if m in winners else 'Never'} for m in managers])
        return _leader(tab, 'Seasons since last title', note=f'Measured through {last_year}, the latest verified championship year. Never-winners measured since the selected data begins.')
    if re.search(r'(championship losses|lost the most championship|most runner.up|runner.up finishes)', q):
        tab = c.runner_up_manager.value_counts().rename_axis('Manager').reset_index(name='Finals losses')
        return _leader(tab, 'Finals losses', note='Verified championship games only.')
    return None


def _playoffs(q, rows, min_games):
    if not re.search(r'\b(playoff|postseason)\b', q):
        return None
    if not re.search(r'\b(wins|losses|winning percentage|win rate|best record|most points|average)\b', q):
        return None
    return _res('Playoff-only records unavailable', 'Yahoo data currently groups playoff and consolation games together. I will not label those combined results as playoff-only statistics. Championship winners and finalists are verified separately.')
    df = rows[rows.phase.ne('Regular season')]
    if df.empty:
        return _res('No postseason games in selection', 'Choose All games or Postseason / consolation in the sidebar.')
    tab = df.groupby('manager', as_index=False).agg(Games=('W','size'), Wins=('W','sum'), Losses=('L','sum'), PF=('points_for','sum'))
    tab['Win %'] = (tab.Wins / tab.Games * 100).round(1)
    tab['PPG'] = (tab.PF / tab.Games).round(2)
    tab = tab.rename(columns={'manager':'Manager'})
    if 'winning percentage' in q or 'win rate' in q or 'best record' in q:
        metric, asc = 'Win %', False
        tab = tab[tab.Games >= min_games]
    elif 'loss' in q:
        metric, asc = 'Losses', False
    elif 'points' in q or 'average' in q:
        metric, asc = 'PPG' if 'average' in q else 'PF', False
    else:
        metric, asc = 'Wins', False
    return _leader(tab, metric, asc, note='WARNING: Postseason / consolation games are grouped together in the source data. This is NOT a playoff-only record unless consolation games can be distinguished.')


def _compare(q, rows, names):
    if not re.search(r'\b(compare|comparison|who is better|who has more|who averages more)\b', q) or len(names) < 2:
        return None
    pair = names[:2]
    df = rows[rows.manager.isin(pair)]
    if df.empty:
        return None
    tab = df.groupby('manager', as_index=False).agg(Games=('W','size'), Wins=('W','sum'), Losses=('L','sum'), Ties=('T','sum'), PF=('points_for','sum'), PA=('points_against','sum'))
    tab['Win %'] = ((tab.Wins+.5*tab.Ties)/tab.Games*100).round(1)
    tab['PPG'] = (tab.PF/tab.Games).round(2)
    tab['Point diff'] = (tab.PF-tab.PA).round(2)
    tab = tab.rename(columns={'manager':'Manager'})
    return _res(f'{pair[0]} vs. {pair[1]}: career comparison', 'Side-by-side statistics for the selected games; “better” depends on which measure you value.', tab, 'This is a career comparison, not just games played against each other.')


def _weekly_game(q, rows):
    if not re.search(r'(highest.scoring game|lowest.scoring game|highest combined|lowest combined|closest game|biggest blowout)', q):
        return None
    df = rows.copy()
    if df.empty:
        return None
    if 'combined' in q or 'scoring game' in q:
        df['Value'] = df.points_for + df.points_against
        ascending = 'lowest' in q
        label = 'Combined points'
    else:
        df['Value'] = df.margin.abs()
        ascending = 'closest' in q
        label = 'Margin'
    # Each actual matchup occurs twice in directional data.
    df['pair'] = df.apply(lambda r: '|'.join(sorted([r.manager, r.opponent])), axis=1)
    df = df.drop_duplicates(['season','week','pair']).sort_values('Value', ascending=ascending)
    tab = df[['season','week','manager','opponent','points_for','points_against','Value']].head(15).rename(columns={'season':'Season','week':'Week','manager':'Manager','opponent':'Opponent','points_for':'Score','points_against':'Opponent score'})
    top = tab.iloc[0]
    return _res(f'{label}: {top.Value:.2f} ({int(top.Season)} Week {int(top.Week)})', f'{top.Manager} vs. {top.Opponent}: {top.Score:.2f}–{top["Opponent score"]:.2f}.', tab)


def _season_trends(q, rows):
    if not re.search(r'(which season|what season|by year|year.over.year)', q):
        return None
    if not re.search(r'(highest average|lowest average|most blowouts|scoring improvement|scoring decline)', q):
        return None
    df = rows[rows.phase.eq('Regular season')]
    if df.empty:
        return None
    if 'year-over-year' in q:
        agg = df.groupby(['manager','season'], as_index=False).points_for.mean().sort_values(['manager','season'])
        agg['Prev'] = agg.groupby('manager').points_for.shift()
        agg['Prev year'] = agg.groupby('manager').season.shift()
        agg = agg[agg.season.eq(agg['Prev year']+1)].copy()
        agg['Change'] = agg.points_for-agg.Prev
        agg = agg.rename(columns={'manager':'Manager','season':'Season','points_for':'PPG','Prev':'Previous PPG'})
        if agg.empty:
            return None
        asc = 'decline' in q
        return _leader(agg, 'Change', asc, note='Change in average regular-season points per recorded game between consecutive years. Partial seasons can distort comparisons.')
    if 'blowouts' in q:
        # 40+ point margin, one winner per matchup.
        df = df[df.W.eq(1) & df.margin.ge(40)]
        tab = df.groupby('season', as_index=False).size().rename(columns={'season':'Season','size':'Blowouts'})
        if tab.empty:
            return _res('No blowouts', 'No 40+ point regular-season wins in the selected years.')
        tab = tab.sort_values('Blowouts', ascending=False)
        top = tab.iloc[0]
        return _res(f'{int(top.Season)}: {int(top.Blowouts)} blowouts', 'A blowout means a victory by at least 40 points.', tab)
    tab = df.groupby('season', as_index=False).points_for.mean().rename(columns={'season':'Season','points_for':'Average score'})
    tab = tab.sort_values('Average score', ascending='lowest' in q)
    top = tab.iloc[0]
    return _res(f'{int(top.Season)}: {top["Average score"]:.2f} average points', 'Average manager score per recorded regular-season matchup.', tab.round(2), 'Partial seasons are included; scoring environments may differ across seasons.')


def _week_filters(q, rows, min_games):
    m = re.search(r'\bweek\s+(\d{1,2})\b', q)
    opener = 'season opener' in q or 'season opening' in q
    last_four = 'last 4 weeks' in q or 'final four weeks' in q
    if not (m or opener or last_four):
        return None
    if not re.search(r'(most wins|best record|worst record|winning percentage|average score|most points|opening wins)', q):
        return None
    df = rows[rows.phase.eq('Regular season')]
    if m:
        df = df[df.week.eq(int(m.group(1)))]
    elif opener:
        df = df[df.week.eq(1)]
    else:
        maxweek = df.groupby('season').week.transform('max')
        df = df[df.week >= maxweek - 3]
    if df.empty:
        return _res('No games in this week', 'No recorded regular-season games match the requested week.')
    tab = df.groupby('manager', as_index=False).agg(Games=('W','size'), Wins=('W','sum'), Losses=('L','sum'), PF=('points_for','sum'))
    tab['Win %'] = (tab.Wins / tab.Games * 100).round(1)
    tab['PPG'] = (tab.PF / tab.Games).round(2)
    tab = tab.rename(columns={'manager':'Manager'})
    if 'best record' in q or 'winning percentage' in q or 'worst record' in q:
        tab = tab[tab.Games >= min_games]
        metric, asc = 'Win %', 'worst' in q
    elif 'average score' in q:
        metric, asc = 'PPG', False
    elif 'points' in q:
        metric, asc = 'PF', False
    else:
        metric, asc = 'Wins', False
    return _leader(tab, metric, asc, note='Only regular-season games. Last four weeks are based on the latest recorded regular-season week in each year; an incomplete season may be misleading.')


def _against_champions(q, rows, champs):
    if 'against eventual champion' not in q or champs is None or champs.empty:
        return None
    c = champs[['season','champion_manager']].drop_duplicates()
    df = rows.merge(c, on='season', how='inner')
    df = df[df.opponent.eq(df.champion_manager) & df.W.eq(1)]
    if df.empty:
        return _res('No qualifying wins', 'No wins against that season’s verified eventual champion in the selected games.')
    tab = df.groupby('manager', as_index=False).size().rename(columns={'manager':'Manager','size':'Wins'})
    return _leader(tab, 'Wins', note='Wins over the eventual verified champion of the same season; includes the championship game if the sidebar includes postseason.')


def answer_advanced(question, rows, champs, manager_names, min_games=5):
    q = re.sub(r'\s+', ' ', question.casefold().replace('’', "'").strip())
    if not q or rows.empty:
        return None
    names = _name_matches(q, manager_names)
    # Starting record questions require full regular-season history, independent of
    # a question-specific year filter. Sidebar filter still limits available rows.
    start = _record_start(q, rows, champs, names)
    if start is not None:
        return start
    df = _scope(q, rows)
    if df.empty:
        return _res('No data in that period', 'No recorded matchups match the requested season and game-type filters.')
    handlers = [
        lambda: _against_champions(q, df, champs),
        lambda: _compare(q, df, names),
        lambda: _all_play(q, df, min_games),
        lambda: _weekly_rank(q, df),
        lambda: _consistency(q, df, min_games),
        lambda: _champions(q, df, champs),
        lambda: _playoffs(q, df, min_games),
        lambda: _week_filters(q, df, min_games),
        lambda: _weekly_game(q, df),
        lambda: _season_trends(q, df),
        lambda: _threshold(q, df, champs, min_games),
    ]
    for handler in handlers:
        result = handler()
        if result is not None:
            return result
    return None
