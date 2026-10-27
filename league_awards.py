"""Verified historical league awards from completed regular seasons and Yahoo finals."""
import pandas as pd
import streamlit as st


def compute_awards(games, mapping, champs, visible_names):
    """Return award winners and supporting manager statistics.

    Only completed 12-team regular seasons (13 games through 2020, 14 after)
    are included. Championships are sourced from the verified champions table.
    """
    regular = games[games.phase.eq('Regular season')].copy()
    qualified = []
    for year, g in regular.groupby('season'):
        expected = 14 if int(year) >= 2021 else 13
        roster = mapping[mapping.season.eq(year)].manager_id.drop_duplicates()
        participants = pd.concat([g.manager1_id, g.manager2_id], ignore_index=True)
        counts = participants.value_counts()
        if (len(roster) == (10 if 2007 <= int(year) <= 2012 else 12)
        and set(roster) == set(counts.index)
                and counts.eq(expected).all() and g.week.nunique() == expected
                and g[['score1', 'score2']].notna().all().all()
                and g[['score1', 'score2']].gt(0).all().all()):
            qualified.append(int(year))
    regular = regular[regular.season.isin(qualified)].copy()
    names = mapping[['manager_id','manager_name']].drop_duplicates().set_index('manager_id').manager_name.to_dict()
    if regular.empty:
        return pd.DataFrame(), pd.DataFrame(), qualified

    left = regular[['season','week','manager1_id','manager2_id','score1','score2']].rename(columns={
        'manager1_id':'manager_id','manager2_id':'opponent_id','score1':'PF','score2':'PA'})
    right = regular[['season','week','manager2_id','manager1_id','score2','score1']].rename(columns={
        'manager2_id':'manager_id','manager1_id':'opponent_id','score2':'PF','score1':'PA'})
    rows = pd.concat([left,right], ignore_index=True)
    rows['Win'] = rows.PF.gt(rows.PA).astype(int)
    rows['Loss'] = rows.PF.lt(rows.PA).astype(int)
    rows['Margin'] = rows.PF - rows.PA
    rows['Manager'] = rows.manager_id.map(names)
    rows['Opponent'] = rows.opponent_id.map(names)
    rows['All-play expected'] = 0.0
    rows['Bottom half'] = False
    for (_, _), group in rows.groupby(['season','week']):
        points = group.set_index('manager_id').PF.to_dict()
        for idx, r in group.iterrows():
            others = [v for key,v in points.items() if key != r.manager_id]
            rows.at[idx,'All-play expected'] = (sum(r.PF > v for v in others) + .5*sum(r.PF == v for v in others))/len(others)
            # Bottom half: rank 7-12 in a 12-team week; ties at boundary are excluded.
            rows.at[idx, 'Bottom half'] = sum(v > r.PF for v in points.values()) >= len(points) / 2

    champ = champs[champs.season.isin(qualified)].copy() if champs is not None and not champs.empty else pd.DataFrame()
    champions = set(zip(champ.season.astype(int), champ.champion_id.astype(str))) if not champ.empty else set()
    champ_ids = {int(r.season):str(r.champion_id) for r in champ.itertuples()} if not champ.empty else {}
    rows['Against champion'] = [r.opponent_id == champ_ids.get(int(r.season)) for r in rows.itertuples()]
    rows['Heartbreaker'] = rows.Win.eq(1) & rows.Margin.gt(0) & rows.Margin.lt(5)
    rows['Heartbroken'] = rows.Loss.eq(1) & rows.Margin.gt(-5)
    rows['Bully'] = rows.Win.eq(1) & rows.Margin.ge(40)
    rows['Punching Bag'] = rows.Loss.eq(1) & rows.Margin.le(-40)
    rows['Paper Tiger'] = rows.Loss.eq(1) & rows.PF.gt(120)
    rows['Escape Artist'] = rows.Win.eq(1) & rows['Bottom half']
    rows['Giant Killer'] = rows.Win.eq(1) & rows['Against champion']

    stats = rows.groupby(['manager_id','Manager'],as_index=False).agg(
        Games=('Win','size'),Wins=('Win','sum'),Losses=('Loss','sum'),
        PF=('PF','sum'),PPG=('PF','mean'),StdDev=('PF','std'),
        Heartbreaker=('Heartbreaker','sum'),Heartbroken=('Heartbroken','sum'),
        Bully=('Bully','sum'),**{'Punching Bag':('Punching Bag','sum'),
        'Paper Tiger':('Paper Tiger','sum'),'Escape Artist':('Escape Artist','sum'),
        'Giant Killer':('Giant Killer','sum'), 'Expected wins':('All-play expected','sum')})
    stats['Luck'] = stats.Wins - stats['Expected wins']
    if not champ.empty:
        title_counts = champ.champion_id.value_counts()
        runner_counts = champ.runner_up_id.value_counts()
    else:
        title_counts = pd.Series(dtype=int)
        runner_counts = pd.Series(dtype=int)
    stats['Titles'] = stats.manager_id.map(title_counts).fillna(0).astype(int)
    stats['Runner-ups'] = stats.manager_id.map(runner_counts).fillna(0).astype(int)

    eligible = stats[stats.Manager.isin(visible_names)].copy()
    awards=[]
    def award(name, metric, ascending=False, min_games=0, note='', fmt=None, positive_only=False):
        pool=eligible[eligible.Games.ge(min_games)].dropna(subset=[metric]).copy()
        if positive_only:
            pool=pool[pool[metric].gt(0)]
        if pool.empty:
            awards.append({'Award':name,'Winner':'Not enough data','Value':'—','Definition':note})
            return
        best=pool[metric].min() if ascending else pool[metric].max()
        winners=', '.join(sorted(pool.loc[pool[metric].eq(best),'Manager']))
        value=fmt(best) if fmt else (str(int(best)) if float(best).is_integer() else f'{best:.2f}')
        awards.append({'Award':name,'Winner':winners,'Value':value,'Definition':note})

    award('The Dynasty','Titles',positive_only=True,note='Most verified championships in completed seasons')
    award('The Bridesmaid','Runner-ups',positive_only=True,note='Most verified championship-game runner-up finishes')
    award('The Heartbreaker','Heartbreaker',positive_only=True,note='Most regular-season wins by less than 5 points')
    award('The Heartbroken','Heartbroken',positive_only=True,note='Most regular-season losses by less than 5 points')
    award('The Bully','Bully',positive_only=True,note='Most regular-season wins by 40 or more points')
    award('The Punching Bag','Punching Bag',positive_only=True,note='Most regular-season losses by 40 or more points')
    award('The Juggernaut','PPG',min_games=10,note='Highest regular-season points per game (minimum 10 games)',fmt=lambda x:f'{x:.2f}')
    award('The Roller Coaster','StdDev',min_games=10,note='Highest standard deviation of weekly regular-season scores (minimum 10 games)',fmt=lambda x:f'{x:.2f}')
    award('The Metronome','StdDev',ascending=True,min_games=10,note='Lowest standard deviation of weekly regular-season scores (minimum 10 games)',fmt=lambda x:f'{x:.2f}')
    award('The Unluckiest','Luck',ascending=True,min_games=10,note='Fewest actual wins relative to all-play expected wins',fmt=lambda x:f'{x:+.2f}')
    award('The Luckiest','Luck',min_games=10,note='Most actual wins relative to all-play expected wins',fmt=lambda x:f'{x:+.2f}')
    award('The Giant Killer','Giant Killer',positive_only=True,note='Most regular-season wins against that season’s verified eventual champion')
    award('The Paper Tiger','Paper Tiger',positive_only=True,note='Most regular-season losses despite scoring over 120 points')
    award('The Escape Artist','Escape Artist',positive_only=True,note='Most regular-season wins while scoring in the bottom six of the league that week')

    # Nemesis: manager with the most wins against a single opponent, minimum 5 meetings.
    h2h=rows.groupby(['manager_id','opponent_id'],as_index=False).agg(Meetings=('Win','size'),Wins=('Win','sum'),Losses=('Loss','sum'))
    h2h=h2h[h2h.Meetings.ge(5) & h2h.manager_id.map(names).isin(visible_names)].copy()
    if not h2h.empty:
        h2h['Manager']=h2h.manager_id.map(names)
        h2h['Opponent']=h2h.opponent_id.map(names)
        top=h2h.Wins.max()
        winners=h2h[h2h.Wins.eq(top)].sort_values(['Manager','Opponent'])
        detail='; '.join(f'{r.Manager} over {r.Opponent} ({r.Wins}-{r.Losses})' for r in winners.itertuples())
        awards.append({'Award':'The Nemesis','Winner':detail,'Value':str(int(top))+' wins','Definition':'Most regular-season wins over one opponent (minimum 5 meetings)'})
    else:
        awards.append({'Award':'The Nemesis','Winner':'Not enough data','Value':'—','Definition':'Most regular-season wins over one opponent (minimum 5 meetings)'})
    return pd.DataFrame(awards), stats, qualified



def compute_current_awards(games, mapping, visible_names):
    """Calculate current-season awards from complete, positive-score regular-season weeks."""
    season = int(games.season.max())
    roster = mapping[mapping.season.eq(season)][['manager_id','manager_name']].drop_duplicates('manager_id')
    ids = set(roster.manager_id.astype(str))
    names = dict(zip(roster.manager_id.astype(str), roster.manager_name))
    season_games = games[(games.season.eq(season)) & games.phase.eq('Regular season')].copy()
    for col in ['manager1_id','manager2_id']:
        season_games[col] = season_games[col].astype(str)
    valid = []
    skipped = []
    for week, g in season_games.groupby('week'):
        members = pd.concat([g.manager1_id, g.manager2_id]).value_counts()
        if (len(ids) >= 2 and set(members.index) == ids and members.eq(1).all()
                and g[['score1','score2']].notna().all().all()
                and g[['score1','score2']].gt(0).all().all()):
            valid.append(g)
        else:
            skipped.append(week)
    if not valid:
        return season, pd.DataFrame(), pd.DataFrame(), 0, skipped
    g = pd.concat(valid, ignore_index=True)
    left = g[['week','manager1_id','manager2_id','score1','score2']].rename(columns={'manager1_id':'id','manager2_id':'opp','score1':'PF','score2':'PA'})
    right = g[['week','manager2_id','manager1_id','score2','score1']].rename(columns={'manager2_id':'id','manager1_id':'opp','score2':'PF','score1':'PA'})
    rows = pd.concat([left,right],ignore_index=True)
    rows['Win'] = (rows.PF > rows.PA).astype(int)
    rows['Loss'] = (rows.PF < rows.PA).astype(int)
    rows['Tie'] = (rows.PF == rows.PA).astype(int)
    rows['Margin'] = rows.PF - rows.PA
    rows['Expected'] = 0.
    rows['Top3Loss'] = False
    rows['Bottom3Win'] = False
    rows['BottomHalfWin'] = False
    rows['BorrowedWins'] = 0.
    rows['ScheduleCount'] = 0
    # For a borrowed schedule, use that owner's actual opponent. When the
    # owner played the evaluated manager, substitute the owner as opponent.
    for week, group in rows.groupby('week'):
        scores = group.set_index('id').PF.to_dict()
        opponents = group.set_index('id').opp.to_dict()
        for idx, r in group.iterrows():
            others = [score for mid,score in scores.items() if mid != r.id]
            rank = 1 + sum(score > r.PF for score in scores.values())
            rows.at[idx,'Expected'] = (sum(r.PF > x for x in others) + .5*sum(r.PF == x for x in others))/len(others)
            rows.at[idx,'Top3Loss'] = bool(r.Loss and rank <= 3)
            rows.at[idx,'Bottom3Win'] = bool(r.Win and rank >= len(scores)-2)
            rows.at[idx,'BottomHalfWin'] = bool(r.Win and rank > len(scores)/2)
            borrowed = []
            for owner, opponent in opponents.items():
                target = owner if opponent == r.id and owner != r.id else opponent
                borrowed.append(float(r.PF > scores[target]) + .5*float(r.PF == scores[target]))
            rows.at[idx,'BorrowedWins'] = sum(borrowed)
            rows.at[idx,'ScheduleCount'] = len(borrowed)
    rows['CloseWin'] = (rows.Win.eq(1) & rows.Margin.lt(5)).astype(int)
    rows['CloseLoss'] = (rows.Loss.eq(1) & rows.Margin.gt(-5)).astype(int)
    rows['PaperTiger'] = (rows.Loss.eq(1) & rows.PF.gt(120)).astype(int)
    rows['Manager'] = rows.id.map(names)
    stats = rows.groupby(['id','Manager'],as_index=False).agg(
        Games=('Win','size'), Wins=('Win','sum'), Losses=('Loss','sum'), Ties=('Tie','sum'),
        PPG=('PF','mean'), StdDev=('PF','std'), Expected=('Expected','sum'),
        BorrowedWins=('BorrowedWins','sum'), ScheduleCount=('ScheduleCount','sum'),
        Robbed=('Top3Loss','sum'), EscapeArtist=('Bottom3Win','sum'),
        BottomHalfWins=('BottomHalfWin','sum'), PaperTiger=('PaperTiger','sum'),
        Heartbreaker=('CloseWin','sum'), Heartbroken=('CloseLoss','sum'))
    stats['Actual'] = stats.Wins + .5*stats.Ties
    stats['Luck'] = stats.Actual - stats.Expected
    stats['Avg swapped wins'] = stats.BorrowedWins / len(ids)
    stats['Schedule edge'] = stats.Actual - stats['Avg swapped wins']
    streaks = {}
    for mid, group in rows.sort_values('week').groupby('id'):
        count=0
        for r in group.itertuples():
            if r.Win: count+=1
            else: count=0
        streaks[mid]=count
    stats['HotHand'] = stats.id.map(streaks)
    eligible=stats[stats.Manager.isin(visible_names)]
    specs = [
        ('🍀 The Golden Horseshoe','Luck','Most wins above all-play expectation','Lucky Bastards',False,True),
        ('🎭 Schedule Bandit','Schedule edge','Most wins gained from the actual schedule versus average swapped schedules','Lucky Bastards',False,True),
        ('🐀 The Escape Artist','EscapeArtist','Most wins while scoring in the bottom three that week','Lucky Bastards',False,False),
        ('😎 The Heartbreaker','Heartbreaker','Most wins by fewer than 5 points','Lucky Bastards',False,False),
        ('💀 The Cursed','Luck','Most wins below all-play expectation','Getting Screwed',True,True),
        ('😤 Robbed','Robbed','Most losses despite a top-three weekly score','Getting Screwed',False,False),
        ('🔥 The Paper Tiger','PaperTiger','Most losses despite scoring over 120 points','Getting Screwed',False,False),
        ('💔 The Heartbroken','Heartbroken','Most losses by fewer than 5 points','Getting Screwed',False,False),
        ('💪 The Juggernaut','PPG','Highest average weekly score','Performance Awards',False,False),
        ('🎢 The Roller Coaster','StdDev','Highest weekly scoring standard deviation','Performance Awards',False,False),
        ('🚀 The Hot Hand','HotHand','Longest currently active winning streak','Performance Awards',False,False),
        ('🏆 The Road Warrior','Avg swapped wins','Highest average hypothetical wins across borrowed schedules','Performance Awards',False,False),
    ]
    awards=[]
    for name, metric, description, category, ascending, signed in specs:
        pool=eligible.dropna(subset=[metric])
        if pool.empty or (metric in ['EscapeArtist','Heartbreaker','Robbed','PaperTiger','Heartbroken','HotHand'] and pool[metric].max()==0):
            winner, value='Not yet awarded','—'
        else:
            best=pool[metric].min() if ascending else pool[metric].max()
            winner=', '.join(sorted(pool.loc[pool[metric].round(8).eq(round(best,8)),'Manager']))
            value=f'{best:+.2f}' if signed else (f'{best:.2f}' if metric in ['PPG','StdDev','Avg swapped wins'] else str(int(best)))
        awards.append({'Category':category,'Award':name,'Winner':winner,'Value':value,'Definition':description})
    return season, pd.DataFrame(awards), stats, g.week.nunique(), skipped


def _award_cards(awards):
    for i in range(0,len(awards),3):
        cols=st.columns(3)
        for col, (_, a) in zip(cols, awards.iloc[i:i+3].iterrows()):
            with col:
                st.markdown(
                    f"""<div style="border:1px solid rgba(128,128,128,.25);border-radius:15px;
                    padding:17px 16px;min-height:175px;margin-bottom:12px">
                    <div style="font-size:1.04rem;font-weight:700">{a['Award']}</div>
                    <div style="font-size:1.18rem;font-weight:750;margin-top:13px">{a['Winner']}</div>
                    <div style="font-size:1.25rem;font-weight:700;margin-top:5px">{a['Value']}</div>
                    <div style="opacity:.7;font-size:.82rem;margin-top:9px">{a['Definition']}</div>
                    </div>""", unsafe_allow_html=True)


def render_league_awards(games, mapping, champs, visible_names, show_table):
    st.title('🏆 League Awards')
    choice=st.radio('Award collection',['Current Season','All-Time'],horizontal=True,key='league_awards_view')
    if choice=='Current Season':
        season, awards, stats, weeks, skipped=compute_current_awards(games,mapping,visible_names)
        st.caption(f'{season} regular season • {weeks} verified complete weeks • Automatically updates when new results arrive')
        if skipped:
            st.warning('Incomplete weeks excluded: '+', '.join(map(str,sorted(skipped))))
        if awards.empty:
            st.info('No completed, valid regular-season weeks available.')
            return
        shown=stats[stats.Manager.isin(visible_names)].copy()
        st.subheader('🍀 Luck & Misfortune Leaderboard')
        st.caption('Luck = actual wins − all-play expected wins. Schedule edge = actual wins − average wins using every manager’s schedule. Positive favors the manager; negative means a tougher draw.')
        cols=st.columns(3)
        luckiest=shown.sort_values('Luck',ascending=False).iloc[0] if not shown.empty else None
        cursed=shown.sort_values('Luck').iloc[0] if not shown.empty else None
        robbed=shown.sort_values('Robbed',ascending=False).iloc[0] if not shown.empty else None
        if luckiest is not None:
            cols[0].metric('🍀 Luckiest',luckiest.Manager,f'{luckiest.Luck:+.2f} wins',delta_color='off')
            cols[1].metric('💀 Most cursed',cursed.Manager,f'{cursed.Luck:+.2f} wins',delta_color='off')
            cols[2].metric('😤 Most top-three losses',robbed.Manager,f'{int(robbed.Robbed)} games',delta_color='off')
        columns=['Manager','Games','Wins','Losses','Ties','Expected','Luck','Schedule edge','Avg swapped wins','Robbed','EscapeArtist','PaperTiger']
        table=shown[columns].rename(columns={'Expected':'Expected wins','Robbed':'Top-3 losses','EscapeArtist':'Bottom-3 wins','PaperTiger':'120+ losses'})
        for col in ['Expected wins','Luck','Schedule edge','Avg swapped wins']:
            table[col]=table[col].round(2)
        show_table(table.sort_values('Luck',ascending=False),'current_season_luck_leaderboard.csv',560)
        st.caption('The 120+ losses column counts scores strictly greater than 120. Schedule swaps substitute the borrowed schedule owner whenever a matchup would otherwise be against yourself.')
        for category in ['Lucky Bastards','Getting Screwed','Performance Awards']:
            st.subheader({'Lucky Bastards':'🍀 Lucky Bastards','Getting Screwed':'💀 Getting Screwed','Performance Awards':'💪 Performance Awards'}[category])
            _award_cards(awards[awards.Category.eq(category)])
        with st.expander('All current-season awards and supporting statistics'):
            show_table(awards,'current_season_awards.csv',480)
            show_table(shown,'current_season_award_stats.csv',480)
    else:
        awards, stats, seasons=compute_awards(games,mapping,champs,visible_names)
        season_options = ['All-Time'] + [str(y) for y in sorted(seasons, reverse=True)]
        selected_season = st.selectbox(
            'Season',
            season_options,
            key='league_awards_season_filter'
        )
        if selected_season != 'All-Time':
            year = int(selected_season)
            season_games = games[games.season.eq(year)].copy()
            season_mapping = mapping[mapping.season.eq(year)].copy()
            season_champs = (
                champs[champs.season.eq(year)].copy()
                if champs is not None and not champs.empty
                else champs
            )
            awards, stats, seasons = compute_awards(
                season_games, season_mapping, season_champs, visible_names
            )
        st.caption('Verified championships and completed regular seasons only. In-progress seasons are excluded.')
        if awards.empty:
            st.info('No complete regular seasons available to calculate awards.')
            return
        st.caption('Eligible seasons: '+', '.join(map(str,sorted(seasons))))
        for title, award_names in [
            ('🏆 Legacy & Titles',['The Dynasty','The Bridesmaid','The Nemesis']),
            ('💥 Big Wins & Tough Losses',['The Heartbreaker','The Heartbroken','The Bully','The Punching Bag','The Paper Tiger']),
            ('📈 Scoring & Consistency',['The Juggernaut','The Roller Coaster','The Metronome']),
            ('🍀 Luck & Unlikely Victories',['The Unluckiest','The Luckiest','The Giant Killer','The Escape Artist'])
        ]:
            st.subheader(title)
            _award_cards(awards[awards.Award.isin(award_names)])
        with st.expander('All-time awards and supporting career statistics'):
            show_table(awards,'league_awards.csv',570)
            columns=['Manager','Games','Wins','Losses','PPG','StdDev','Expected wins','Luck','Titles','Runner-ups','Heartbreaker','Heartbroken','Bully','Punching Bag','Giant Killer','Paper Tiger','Escape Artist']
            shown=stats[stats.Manager.isin(visible_names)][columns].copy()
            for col in ['PPG','StdDev','Expected wins','Luck']:
                shown[col]=shown[col].round(2)
            show_table(shown.sort_values('Wins',ascending=False),'league_awards_supporting_stats.csv',520)
