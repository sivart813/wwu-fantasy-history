"""Rule-based composable season-achievement questions; no API or dynamic execution."""
import re
import pandas as pd

NUMBERS = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12}
NUM = r'(\d{1,2}|' + '|'.join(NUMBERS) + r')'

def _number(s):
    return int(s) if s.isdigit() else NUMBERS[s]

def _result(title, detail, table=None, note='', diagnostic=''):
    return {'title':title,'detail':detail,'table':table if table is not None else pd.DataFrame(), 'note':note,'diagnostic':diagnostic}

def answer_compound(question, rows, honors, manager_names):
    q = re.sub(r'\s+', ' ', question.casefold().replace('–','-').replace('—','-')).strip()
    # Only take over questions with a recognized achievement AND a streak or start condition.
    outcome = None
    if re.search(r'\b(?:made?|make|reach(?:ed)?|qualified? for)\s+(?:the\s+)?playoffs\b|\bplayoff qualification\b',q):
        outcome = ('Made playoffs', 'made the playoffs')
    elif re.search(r'\bregular.season (?:champion|title|winner|championship)\b|\bwon the regular season\b',q):
        outcome = ('Regular-season champion','won the regular season')
    elif re.search(r'\b(?:runner.up|runner up|lost the championship game|lost the final)\b',q):
        outcome = ('Playoff runner-up','finished runner-up')
    elif re.search(r'\b(?:won|win|winning|champion|championship|title|league winner)\b',q) and re.search(r'\b(?:league|champion|championship|title|final|finals)\b',q):
        outcome = ('League champion','won the league championship')
    if outcome is None:
        return None
    streak = re.search(r'\b(?:los(?:t|ing)|defeat(?:ed)?|dropp?ed)\s+(?:at least\s+|exactly\s+)?'+NUM+r'\s+(?:games?\s+)?(?:in a row|straight|consecutive)\b',q)
    if not streak:
        streak = re.search(r'\b'+NUM+r'\s+(?:straight|consecutive)\s+(?:losses|defeats)\b',q)
    if not streak:
        streak = re.search(r'\b(?:losing streak|loss streak)\s+(?:of\s+)?(?:at least\s+|exactly\s+)?'+NUM+r'\b',q)
    win_streak = False
    if not streak:
        streak = re.search(r'\b(?:won|winning)\s+(?:at least\s+|exactly\s+)?'+NUM+r'\s+(?:games?\s+)?(?:in a row|straight|consecutive)\b',q)
        win_streak = bool(streak)
    start = re.search(r'\b(?:start(?:ed|ing)?|after going|after an?)\s+(\d{1,2})\s*-\s*(\d{1,2})\b',q)
    if not streak and not start:
        return None
    if honors is None or honors.empty:
        return _result('No completed seasons', 'No complete regular seasons are available for this question.')
    h = honors[honors.manager.isin(manager_names)].copy()
    year = re.search(r'\b(?:in|during)\s+(20\d{2})\b',q)
    since = re.search(r'\bsince\s+(20\d{2})\b',q)
    before = re.search(r'\bbefore\s+(20\d{2})\b',q)
    if year is not None: h=h[h.season.eq(int(year.group(1)))]
    if since is not None: h=h[h.season.ge(int(since.group(1)))]
    if before is not None: h=h[h.season.lt(int(before.group(1)))]
    for n in manager_names:
        aliases = [n.casefold()] + (['kruse','rj'] if n.casefold()=='rj/kruse' else [])
        if any(re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',q) for a in aliases):
            h=h[h.manager.eq(n)]
            break
    if h.empty:
        return _result('No eligible seasons', 'No completed seasons match these filters.')
    if rows is None or rows.empty:
        return _result('Missing weekly results', 'Weekly regular-season results are needed to calculate streaks or starts.')
    regular = rows[rows.phase.eq('Regular season')].copy()
    if regular.empty:
        return _result('Missing regular-season results', 'Select regular season or all games to analyze weekly results.')
    metric, achievement = outcome
    threshold = _number(streak.group(1)) if streak else None
    if threshold is not None and not 1 <= threshold <= 18:
        return _result('Invalid streak length','Use a streak length between 1 and 18 games.')
    sw, sl = (int(start.group(1)), int(start.group(2))) if start else (None,None)
    if start and (sw+sl<1 or sw+sl>18):
        return _result('Invalid starting record','Use a starting record totaling 1–18 games.')
    exact = bool(re.search(r'\bexactly\b',q))
    candidates=[]
    for r in h.itertuples(index=False):
        g=regular[(regular.season==r.season)&(regular.manager_id==r.manager_id)].sort_values('week')
        # Only contiguous unique weekly results are eligible for a streak claim.
        if g.empty or g.week.duplicated().any() or not all(g.week.diff().dropna().eq(1)):
            continue
        if len(g)!=int(r.Games):
            continue
        longest=0; current=0
        for row in g.itertuples(index=False):
            hit = int(row.W if win_streak else row.L)==1
            current = current+1 if hit else 0
            longest=max(longest,current)
        if streak and (longest!=threshold if exact else longest<threshold):
            continue
        if start:
            first=g.head(sw+sl)
            if len(first)<sw+sl or first['T'].sum():
                continue
            w,l=int(first.W.sum()),int(first.L.sum())
            if not ((w<=sw and l>=sl) if 'or worse' in q else (w==sw and l==sl)):
                continue
        candidates.append({'Season':int(r.season),'Manager':r.manager,'Final record':f'{int(r.Wins)}-{int(r.Losses)}-{int(r.Ties)}',
                           'Longest winning streak' if win_streak else 'Longest losing streak':longest,
                           'Achievement':bool(h.loc[(h.season==r.season)&(h.manager_id==r.manager_id),metric].iloc[0])})
    table=pd.DataFrame(candidates)
    condition = (f'{"winning" if win_streak else "losing"} streak {"exactly" if exact else "at least"} {threshold}' if streak else f'started {sw}-{sl}{" or worse" if "or worse" in q else ""}')
    diagnostic=f'Outcome: {achievement} | Condition: {condition} | Scope: same completed regular season | Filters: sidebar seasons and managers'
    if table.empty:
        return _result('No qualifying manager-seasons',f'No eligible manager-season had a {condition}.',table,diagnostic=diagnostic)
    qualified=table[table.Achievement].drop(columns=['Achievement']).sort_values(['Season','Manager'])
    return _result(f'{"Yes" if len(qualified) else "No"} — {len(qualified)} qualifying manager-season(s)',
                   f'{len(table)} manager-season(s) matched the condition ({condition}); {len(qualified)} {achievement}.',
                   qualified if not qualified.empty else table.drop(columns=['Achievement']),
                   note='Based on complete regular seasons only; postseason championship winners and finalists use verified championship records.', diagnostic=diagnostic)
