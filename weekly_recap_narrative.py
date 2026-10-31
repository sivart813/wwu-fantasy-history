import pandas as pd
import streamlit as st


def _weekly_frame(weekly):
    df = pd.DataFrame(weekly).copy()

    required = {
        "Week", "Manager", "Score", "Actual",
        "All-play W", "All-play L", "All-play T",
    }

    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

    df = df.dropna(subset=["Week", "Score"])

    return df.sort_values(["Week", "Manager"])


def _streak(df, manager, through_week):
    history = df.loc[
        df["Manager"].eq(manager)
        & df["Week"].le(through_week)
    ].sort_values("Week", ascending=False)

    if history.empty:
        return None, 0

    results = history["Actual"].astype(str).tolist()
    current = results[0]

    if current not in ("W", "L"):
        return None, 0

    count = 0

    for result in results:
        if result != current:
            break
        count += 1

    return current, count


def _matchups(games, mapping, season, week):
    games = pd.DataFrame(games)
    mapping = pd.DataFrame(mapping)

    required_games = {
        "season", "week", "phase",
        "manager1_id", "manager2_id",
        "score1", "score2",
    }

    required_mapping = {
        "season", "manager_id", "manager_name",
    }

    if not required_games.issubset(games.columns):
        return pd.DataFrame()

    if not required_mapping.issubset(mapping.columns):
        return pd.DataFrame()

    season_games = games.loc[
        pd.to_numeric(games["season"], errors="coerce").eq(season)
        & pd.to_numeric(games["week"], errors="coerce").eq(week)
        & games["phase"].eq("Regular season")
    ].copy()

    names = mapping.loc[
        pd.to_numeric(mapping["season"], errors="coerce").eq(season),
        ["manager_id", "manager_name"],
    ].drop_duplicates("manager_id")

    name_map = dict(
        zip(names["manager_id"].astype(str),
            names["manager_name"].astype(str))
    )

    rows = []

    for game in season_games.itertuples(index=False):
        try:
            score1 = float(game.score1)
            score2 = float(game.score2)
        except (TypeError, ValueError):
            continue

        if pd.isna(score1) or pd.isna(score2):
            continue

        if score1 <= 0 or score2 <= 0:
            continue

        manager1 = name_map.get(
            str(game.manager1_id), str(game.manager1_id)
        )
        manager2 = name_map.get(
            str(game.manager2_id), str(game.manager2_id)
        )

        if score1 >= score2:
            winner, loser = manager1, manager2
            winning_score, losing_score = score1, score2
        else:
            winner, loser = manager2, manager1
            winning_score, losing_score = score2, score1

        rows.append({
            "Winner": winner,
            "Loser": loser,
            "Winning Score": winning_score,
            "Losing Score": losing_score,
            "Margin": abs(score1 - score2),
            "Combined": score1 + score2,
        })

    return pd.DataFrame(rows)


def render_weekly_narrative(games, mapping, season, weekly, week):
    df = _weekly_frame(weekly)
    if df.empty:
        return

    current = df.loc[df["Week"].eq(week)].copy()
    if current.empty:
        return

    matchups = _matchups(games, mapping, season, week)
    if matchups.empty:
        return

    current = current.sort_values(
        "Score", ascending=False
    ).reset_index(drop=True)

    avg = current["Score"].mean()
    top = current.iloc[0]
    bottom = current.iloc[-1]
    median = current["Score"].median()

    closest = matchups.sort_values(
        ["Margin", "Combined"],
        ascending=[True, False],
    ).iloc[0]

    blowout = matchups.sort_values(
        ["Margin", "Combined"],
        ascending=[False, False],
    ).iloc[0]

    st.header("Around the League")

    st.markdown(
        f"**Week {int(week)} Summary**"
    )
    st.write(
        f"League average: **{avg:.2f}** points | "
        f"Highest: **{top['Manager']} "
        f"({top['Score']:.2f})** | "
        f"Lowest: **{bottom['Manager']} "
        f"({bottom['Score']:.2f})**"
    )

    st.markdown("#### Matchup Highlights")

    st.write(
        f"**Closest game:** {closest['Winner']} defeated "
        f"{closest['Loser']}, "
        f"{closest['Winning Score']:.2f} - "
        f"{closest['Losing Score']:.2f} "
        f"(margin: {closest['Margin']:.2f})."
    )

    st.write(
        f"**Biggest blowout:** {blowout['Winner']} defeated "
        f"{blowout['Loser']}, "
        f"{blowout['Winning Score']:.2f} - "
        f"{blowout['Losing Score']:.2f} "
        f"(margin: {blowout['Margin']:.2f})."
    )

    losers = current.loc[
        current["Actual"].eq("L")
    ].sort_values("Score", ascending=False)

    if not losers.empty:
        bad = losers.iloc[0]
        rank = int(
            current["Score"].rank(
                ascending=False, method="min"
            ).loc[bad.name]
        )
        st.write(
            f"**Highest-scoring loss:** "
            f"{bad['Manager']} - {bad['Score']:.2f} "
            f"(#{rank} weekly score)."
        )

    winners = current.loc[
        current["Actual"].eq("W")
    ].sort_values("Score")

    if not winners.empty:
        lucky = winners.iloc[0]
        if lucky["Score"] < median:
            st.write(
                f"**Lowest-scoring win:** "
                f"{lucky['Manager']} - "
                f"{lucky['Score']:.2f} points."
            )

    winning = []
    losing = []

    for manager in current["Manager"]:
        result, length = _streak(df, manager, week)

        if result == "W" and length >= 2:
            winning.append((manager, length))
        elif result == "L" and length >= 2:
            losing.append((manager, length))

    st.markdown("#### Winning Streaks")

    if winning:
        winning.sort(key=lambda x: (-x[1], x[0]))
        st.write(
            " | ".join(
                f"**{name}:** {length} wins"
                for name, length in winning
            )
        )
    else:
        st.caption("No winning streaks of 2+ games.")

    st.markdown("#### Losing Streaks")

    if losing:
        losing.sort(key=lambda x: (-x[1], x[0]))
        st.write(
            " | ".join(
                f"**{name}:** {length} losses"
                for name, length in losing
            )
        )
    else:
        st.caption("No losing streaks of 2+ games.")

    st.markdown("#### Scoring Leaders")

    leaders = current.head(3)

    st.write(
        " | ".join(
            f"**{i + 1}. {row.Manager}:** "
            f"{row.Score:.2f}"
            for i, row in enumerate(
                leaders.itertuples(index=False)
            )
        )
    )

    through = df.loc[df["Week"].le(week)]
    season_ppg = (
        through.groupby("Manager")["Score"]
        .mean()
        .sort_values(ascending=False)
    )

    if not season_ppg.empty:
        st.write(
            f"**Season scoring leader:** "
            f"{season_ppg.index[0]} "
            f"({season_ppg.iloc[0]:.2f} PPG)"
        )

    st.divider()
