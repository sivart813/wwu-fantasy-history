"""Official WWU regular-season honors, derived from complete regular seasons."""
import re
import pandas as pd


def calculate_honors(rows, champions, playoff_slots_by_year=None):
    """Compute honors for complete seasons using observed league sizes.

    Playoff slots default to six (legacy app behavior); override with
    league_config.json playoff_slots_by_year if a season had another format.
    """
    playoff_slots_by_year = playoff_slots_by_year or {}
    reg = rows[rows.phase.eq('Regular season')].copy()
    cols = ['season','manager_id','manager','Games','Wins','Losses','Ties','PF','Win %',
            'Finish','Regular-season champion','Made playoffs','Playoff runner-up','League champion']
    if reg.empty:
        return pd.DataFrame(columns=cols)
    standings = reg.groupby(['season','manager_id','manager'], as_index=False).agg(
        Games=('W','size'), Wins=('W','sum'), Losses=('L','sum'), Ties=('T','sum'), PF=('points_for','sum'))
    standings['Win %'] = (standings.Wins + .5 * standings.Ties) / standings.Games
    good=[]
    for season, g in standings.groupby('season'):
        expected = 14 if season >= 2021 else 13
        expected_teams = 10 if 2007 <= season <= 2012 else 12
        # Do not count an incomplete season as a regular-season title.
        if len(g) != expected_teams or not g.Games.eq(expected).all():
            continue
        g = g.sort_values(['Win %','PF'], ascending=False).copy()
        if g.duplicated(['Win %','PF'], keep=False).any():
            raise ValueError(f'{season}: unresolved standings tie on record and PF')
        g['Finish'] = range(1, len(g)+1)
        g['Regular-season champion'] = g.Finish.eq(1)
        slots = int(playoff_slots_by_year.get(str(season), 6))
        if not 2 <= slots <= expected_teams:
            raise ValueError(f'{season}: invalid playoff slot count {slots}')
        g['Made playoffs'] = g.Finish.le(slots)
        good.append(g)
    if not good:
        return pd.DataFrame(columns=cols)
    out=pd.concat(good,ignore_index=True)
    out['Playoff runner-up']=False
    out['League champion']=False
    if champions is not None and not champions.empty:
        runner=set(zip(champions.season.astype(int),champions.runner_up_id.astype(str)))
        winner=set(zip(champions.season.astype(int),champions.champion_id.astype(str)))
        keys=list(zip(out.season.astype(int),out.manager_id.astype(str)))
        out['Playoff runner-up']=[k in runner for k in keys]
        out['League champion']=[k in winner for k in keys]
    return out[cols].sort_values(['season','Finish']).reset_index(drop=True)


def honor_totals(honors):
    if honors.empty:
        return pd.DataFrame(columns=['manager_id','manager','Eligible seasons','Regular-season titles','Playoff appearances','Playoff rate','Playoff runner-ups','League titles'])
    t = honors.groupby(['manager_id','manager'], as_index=False).agg(
        **{'Eligible seasons':('season','size'),
           'Regular-season titles':('Regular-season champion','sum'),
           'Playoff appearances':('Made playoffs','sum'),
           'Playoff runner-ups':('Playoff runner-up','sum'),
           'League titles':('League champion','sum')})
    t['Playoff rate'] = (100*t['Playoff appearances']/t['Eligible seasons']).round(1)
    return t


def answer_honors(question, honors, regular_rows, visible_names):
    q = re.sub(r'\s+', ' ', question.casefold().replace('–','-').replace('—','-')).strip()
    if not re.search(r'playoff|postseason|regular.season (?:champ|title|winner)|runner.up|starting \d|started \d|start \d', q):
        return None
    if not re.search(r'(?:starting|started|start|after going|after an?)\s+\d', q) and re.search(r'playoff (?:wins?|losses?|streak|score|points|record)', q):
        return None
    def result(title, detail, table, note='Only complete regular seasons are eligible. Standings use win percentage, then points for; top six qualify. Playoff runner-ups are verified championship finalists.'):
        return dict(title=title, detail=detail, table=table.reset_index(drop=True), note=note)
    if honors.empty:
        return result('No complete seasons', 'No eligible completed regular seasons in the selected range.', pd.DataFrame())
    h = honors[honors.manager.isin(visible_names)].copy()
    yr = re.search(r'\b(?:in|during) (20\d{2})\b', q)
    since = re.search(r'\bsince (20\d{2})\b', q)
    if yr: h = h[h.season.eq(int(yr.group(1)))]
    if since: h = h[h.season.ge(int(since.group(1)))]
    if h.empty: return result('No eligible results', 'No completed seasons match the question.', h)
    start = re.search(r'\b(?:starting|started|start|after going|after an?)\s+(\d{1,2})\s*-\s*(\d{1,2})\b', q)
    if start:
        w,l = map(int,start.groups()); n=w+l
        if n<1 or n>14: return result('Invalid start', 'Specify a starting record between 1 and 14 games.', pd.DataFrame())
        reg = regular_rows[regular_rows.phase.eq('Regular season')].sort_values(['season','week'])
        outcomes=[]
        for r in h.itertuples(index=False):
            g=reg[(reg.season==r.season)&(reg.manager_id==r.manager_id)].head(n)
            if len(g)!=n or g['T'].sum(): continue
            gw,gl=int(g.W.sum()),int(g.L.sum())
            match=(gw<=w and gl>=l) if 'or worse' in q else (gw==w and gl==l)
            if match:
                outcomes.append({'Season':r.season,'Manager':r.manager,'Start':f'{gw}-{gl}',
                                 'Final record':f'{r.Wins}-{r.Losses}-{r.Ties}',
                                 'Finish':r.Finish,'Made playoffs':r.Finish<=6,
                                 'Regular-season champion':r.Finish==1,
                                 'Playoff runner-up':bool(r[-2]),'League champion':bool(r[-1])})
        tab=pd.DataFrame(outcomes)
        if tab.empty: return result('No matching starts',f'No completed manager-season started {w}-{l}.',tab)
        missed = bool(re.search(
            r'\b(?:missed|miss|missing|failed to make|did not make|'
            r'didn.t make|never made|failed to reach|did not reach|'
            r'didn.t reach|failed to qualify for)\s+(?:the\s+)?playoffs\b',
            q
        ))
        if missed:
            col = 'Made playoffs'
            matched = tab[~tab[col]]
            label = 'missed the playoffs'
        else:
            if 'runner-up' in q or 'runner up' in q:
                col = 'Playoff runner-up'
            elif re.search(r'playoff|postseason', q):
                col = 'Made playoffs'
            elif 'regular' in q:
                col = 'Regular-season champion'
            else:
                col = 'League champion'
            matched = tab[tab[col]]
            label = col.lower()
        return result(
            f'{len(matched)} qualifying manager-season(s)',
            f'{len(tab)} started {w}-{l}{" or worse" if "or worse" in q else ""}; '
            f'{len(matched)} achieved {label}.',
            matched
        )
    if re.search(r'runner.up|runner up|championship (?:loss|finalist)',q): metric='Playoff runner-ups'
    elif re.search(r'regular.season (?:champ|title|winner)',q): metric='Regular-season titles'
    elif re.search(r'playoff|postseason',q): metric='Playoff appearances'
    else: return None
    if re.search(r'\b(who|which)\b.*\b(?:in|during) 20\d{2}',q) or re.search(r'\b20\d{2}\b.*\b(who|which)\b',q):
        if metric=='Regular-season titles': sub=h[h['Regular-season champion']]
        elif metric=='Playoff appearances': sub=h[h['Made playoffs']]
        else: sub=h[h['Playoff runner-up']]
        return result(f'{len(sub)} qualifying manager(s)',f'{metric} for the selected completed season.',sub[['season','manager','Finish','Wins','Losses','PF']])
    totals=honor_totals(h).sort_values([metric,'manager'],ascending=[False,True])
    names=[name for name in visible_names if re.search(r'(?<!\w)'+re.escape(name.casefold())+r'(?!\w)',q)]
    if names: totals=totals[totals.manager.isin(names)]
    if 'percentage' in q or 'rate' in q: metric='Playoff rate'
    if totals.empty: return result('No matches','No managers match this question.',totals)
    top=totals.iloc[0]
    return result(f'{top.manager}: {top[metric]} {metric.lower()}',f'Career honors across completed seasons in the selected range.',totals.sort_values(metric,ascending=False))
