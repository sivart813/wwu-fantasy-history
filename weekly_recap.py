from weekly_recap_narrative import render_weekly_narrative
import pandas as pd
import streamlit as st


def render_weekly_recap(games, mapping, season, weekly, visible_names):
    st.subheader("Weekly Recap")
    st.caption("Completed regular-season weeks only.")

    weekly = pd.DataFrame(weekly).copy()

    if weekly.empty:
        st.info("No completed weeks available.")
        return

    available_weeks = sorted(
        int(w) for w in weekly["Week"].dropna().unique()
    )

    selected_week = st.selectbox(
        "Select week",
        available_weeks,
        index=len(available_weeks) - 1,
        key="league_recap_week",
    )


    results = weekly.loc[
        weekly["Week"].eq(selected_week)
    ].copy()

    roster = mapping.loc[
        mapping["season"].eq(season),
        ["manager_id", "manager_name"]
    ].drop_duplicates()

    names = dict(zip(roster.manager_id, roster.manager_name))

    matches = games.loc[
        games["season"].eq(season)
        & games["week"].eq(selected_week)
        & games["phase"].eq("Regular season")
    ].copy()

    if matches.empty:
        st.info("No matchup details available.")
        return

    matches["score1"] = pd.to_numeric(
        matches["score1"], errors="coerce"
    )
    matches["score2"] = pd.to_numeric(
        matches["score2"], errors="coerce"
    )

    valid_ids = set(roster.manager_id)

    participants = pd.concat(
        [matches.manager1_id, matches.manager2_id],
        ignore_index=True,
    )

    if (
        len(participants) != len(valid_ids)
        or set(participants) != valid_ids
        or participants.duplicated().any()
        or matches[["score1", "score2"]].isna().any().any()
        or (matches[["score1", "score2"]] <= 0).any().any()
    ):
        st.warning("This week's matchup data is incomplete.")
        return

    scoreboard = []

    for r in matches.itertuples():
        a = names.get(r.manager1_id, str(r.manager1_id))
        b = names.get(r.manager2_id, str(r.manager2_id))
        sa, sb = float(r.score1), float(r.score2)

        scoreboard.append({
            "Manager 1": a,
            "Score 1": round(sa, 2),
            "Manager 2": b,
            "Score 2": round(sb, 2),
            "Margin": round(abs(sa - sb), 2),
            "Winner": a if sa > sb else (
                b if sb > sa else "Tie"
            ),
        })

    board = pd.DataFrame(scoreboard)

    highest = results.sort_values(
        ["Score", "Manager"], ascending=[False, True]
    ).iloc[0]

    lowest = results.sort_values(
        ["Score", "Manager"], ascending=[True, True]
    ).iloc[0]

    closest = board.sort_values(
        "Margin", ascending=True
    ).iloc[0]

    biggest = board.sort_values(
        "Margin", ascending=False
    ).iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Highest score",
        f"{highest['Score']:.2f}",
        highest["Manager"],
        delta_color="off",
    )
    c2.metric(
        "Lowest score",
        f"{lowest['Score']:.2f}",
        lowest["Manager"],
        delta_color="off",
    )
    c3.metric(
        "Closest game",
        f"{closest['Margin']:.2f}",
        f"{closest['Manager 1']} vs {closest['Manager 2']}",
        delta_color="off",
    )
    c4.metric(
        "Biggest blowout",
        f"{biggest['Margin']:.2f}",
        f"{biggest['Manager 1']} vs {biggest['Manager 2']}",
        delta_color="off",
    )

    render_weekly_narrative(
        games, mapping, season, weekly, selected_week
    )



