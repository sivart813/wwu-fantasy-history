import pandas as pd
import streamlit as st


WEIGHTS = {
    "All-play": 0.40,
    "Scoring": 0.30,
    "Schedule": 0.20,
    "Consistency": 0.10,
}


def _id(value):
    """Normalize integer/string manager IDs."""
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value


def _prepare(weekly, games, mapping, season):
    df = pd.DataFrame(weekly).copy()
    games = pd.DataFrame(games).copy()
    mapping = pd.DataFrame(mapping).copy()

    required_weekly = {
        "Week", "Manager", "Score", "Actual",
        "All-play W", "All-play L", "All-play T",
    }
    required_games = {
        "season", "week", "phase",
        "manager1_id", "manager2_id",
        "score1", "score2",
    }
    required_mapping = {
        "season", "manager_id", "manager_name",
    }

    for name, frame, required in [
        ("weekly", df, required_weekly),
        ("games", games, required_games),
        ("mapping", mapping, required_mapping),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    if df.empty:
        return df

    df["Week"] = pd.to_numeric(df["Week"], errors="raise")

    for column in [
        "Score", "All-play W", "All-play L", "All-play T"
    ]:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        )

    if df.duplicated(["Week", "Manager"]).any():
        raise ValueError("Duplicate manager/week in all-play data.")

    season = int(season)

    roster = mapping.loc[
        pd.to_numeric(
            mapping["season"], errors="coerce"
        ).eq(season)
    ].copy()

    roster["manager_id"] = roster["manager_id"].map(_id)

    if roster["manager_id"].duplicated().any():
        raise ValueError(
            "Duplicate manager IDs in current-season mapping."
        )

    name_map = dict(zip(
        roster["manager_id"],
        roster["manager_name"].astype(str)
    ))

    if len(set(name_map.values())) != len(name_map):
        raise ValueError(
            "Manager names are not unique for this season."
        )

    regular = games.loc[
        pd.to_numeric(
            games["season"], errors="coerce"
        ).eq(season)
        & games["phase"].eq("Regular season")
    ].copy()

    regular["week"] = pd.to_numeric(
        regular["week"], errors="coerce"
    )

    for column in ["score1", "score2"]:
        regular[column] = pd.to_numeric(
            regular[column], errors="coerce"
        )

    regular["manager1_id"] = regular[
        "manager1_id"
    ].map(_id)

    regular["manager2_id"] = regular[
        "manager2_id"
    ].map(_id)

    records = []

    # Use only weeks already validated by calculate_all_play.
    completed_weeks = sorted(df["Week"].unique())

    for week in completed_weeks:
        week_games = regular.loc[
            regular["week"].eq(week)
        ]

        for game in week_games.itertuples(index=False):
            m1 = name_map.get(game.manager1_id)
            m2 = name_map.get(game.manager2_id)

            if m1 is None or m2 is None:
                continue

            if (
                pd.isna(game.score1)
                or pd.isna(game.score2)
                or game.score1 <= 0
                or game.score2 <= 0
            ):
                continue

            records.extend([
                {
                    "Week": week,
                    "Manager": m1,
                    "Match score": float(game.score1),
                    "Opponent score": float(game.score2),
                },
                {
                    "Week": week,
                    "Manager": m2,
                    "Match score": float(game.score2),
                    "Opponent score": float(game.score1),
                },
            ])

    if not records:
        raise ValueError(
            "No completed current-season matchups found."
        )

    opponents = pd.DataFrame(records)

    if opponents.duplicated(["Week", "Manager"]).any():
        raise ValueError(
            "Duplicate matchups detected in schedule data."
        )

    df = df.merge(
        opponents,
        on=["Week", "Manager"],
        how="left",
        validate="one_to_one",
    )

    if df["Opponent score"].isna().any():
        missing = df.loc[
            df["Opponent score"].isna(),
            ["Week", "Manager"]
        ]
        raise ValueError(
            "Missing opponents for completed weeks: "
            + missing.to_string(index=False)
        )

    mismatch = (
        df["Score"] - df["Match score"]
    ).abs().gt(0.02)

    if mismatch.any():
        raise ValueError(
            "Matchup scores do not agree with all-play "
            "scores. Schedule rankings were not calculated."
        )

    # Top-half finish. Ties at median receive half credit.
    df["Median"] = df["Week"].map(
        df.groupby("Week")["Score"].median()
    )

    df["Top half"] = (
        df["Score"] > df["Median"]
    ).astype(float)

    df.loc[
        df["Score"].eq(df["Median"]),
        "Top half"
    ] = 0.5

    return df.sort_values(["Week", "Manager"])


def calculate_power_rankings(
    weekly, games, mapping, season
):
    df = _prepare(weekly, games, mapping, season)

    if df.empty:
        return pd.DataFrame()

    history = []

    for week in sorted(df["Week"].unique()):
        through = df.loc[df["Week"].le(week)]

        league_ppg = through["Score"].mean()

        rows = []

        for manager, group in through.groupby(
            "Manager", sort=True
        ):
            played = len(group)

            ap_w = group["All-play W"].sum()
            ap_l = group["All-play L"].sum()
            ap_t = group["All-play T"].sum()
            ap_total = ap_w + ap_l + ap_t

            allplay = (
                100 * (ap_w + 0.5 * ap_t) / ap_total
                if ap_total else 50.0
            )

            ppg = group["Score"].mean()

            scoring = (
                100 * ppg / league_ppg
                if league_ppg > 0 else 100.0
            )

            opponent_ppg = group["Opponent score"].mean()

            schedule = (
                100 * opponent_ppg / league_ppg
                if league_ppg > 0 else 100.0
            )

            consistency = (
                100 * group["Top half"].mean()
            )

            wins = int(group["Actual"].eq("W").sum())
            losses = int(group["Actual"].eq("L").sum())
            ties = int(group["Actual"].eq("T").sum())

            actual_wins = wins + 0.5 * ties
            expected_wins = played * allplay / 100
            luck = actual_wins - expected_wins

            allplay_points = 0.40 * allplay
            scoring_points = 0.30 * scoring
            schedule_points = 0.20 * schedule
            consistency_points = 0.10 * consistency

            power = (
                allplay_points
                + scoring_points
                + schedule_points
                + consistency_points
            )

            rows.append({
                "Week": int(week),
                "Manager": manager,
                "Actual record": (
                    f"{wins}-{losses}"
                    + (f"-{ties}" if ties else "")
                ),
                "Power Score": power,
                "All-play %": allplay,
                "Season PPG": ppg,
                "Scoring index": scoring,
                "Opponent PPG": opponent_ppg,
                "Schedule index": schedule,
                "Consistency %": consistency,
                "Expected wins": expected_wins,
                "Schedule luck": luck,
                "All-play contribution": allplay_points,
                "Scoring contribution": scoring_points,
                "Schedule contribution": schedule_points,
                "Consistency contribution":
                    consistency_points,
            })

        frame = pd.DataFrame(rows)

        frame = frame.sort_values(
            [
                "Power Score",
                "All-play %",
                "Season PPG",
                "Manager",
            ],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

        frame["Rank"] = range(1, len(frame) + 1)
        history.append(frame)

    history = pd.concat(history, ignore_index=True)

    history = history.sort_values(["Manager", "Week"])

    history["Previous Rank"] = (
        history.groupby("Manager")["Rank"].shift(1)
    )

    history["Movement"] = (
        history["Previous Rank"] - history["Rank"]
    )

    return history.sort_values(
        ["Week", "Rank"]
    ).reset_index(drop=True)


def render_power_rankings(
    weekly, visible_names, games, mapping, season
):
    st.subheader("Power Rankings")

    st.caption(
        "40% all-play, 30% scoring strength, "
        "20% schedule difficulty, 10% consistency. "
        "Schedule difficulty directly contributes "
        "to the Power Score."
    )

    try:
        history = calculate_power_rankings(
            weekly, games, mapping, season
        )
    except (ValueError, KeyError) as exc:
        st.error(f"Cannot calculate power rankings: {exc}")
        return

    if history.empty:
        st.info("No completed weeks available.")
        return

    weeks = sorted(history["Week"].unique())

    selected_week = st.selectbox(
        "Rankings through week",
        weeks,
        index=len(weeks) - 1,
        key="league_power_week",
    )

    current = history.loc[
        history["Week"].eq(selected_week)
    ].copy()

    if visible_names is not None:
        current = current.loc[
            current["Manager"].isin(visible_names)
        ].copy()

    def movement_label(value):
        if pd.isna(value):
            return "New"
        if value > 0:
            return f"+{int(value)}"
        if value < 0:
            return f"-{abs(int(value))}"
        return "0"

    current["Change"] = current[
        "Movement"
    ].apply(movement_label)

    st.markdown("### Schedule-Adjusted Team Strength")

    main_columns = [
        "Rank", "Manager", "Change",
        "Power Score", "Season PPG",
        "All-play %", "Opponent PPG",
        "Schedule index",
    ]

    display = current[main_columns].copy()

    for column in main_columns:
        if column not in ("Rank", "Manager", "Change"):
            display[column] = display[column].round(2)

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Manager": st.column_config.TextColumn(
                "Manager", pinned=True
            )
        },
    )

    st.caption(
        "Schedule index: 100 = league-average opponent "
        "scoring. Above 100 = tougher schedule. "
        "Below 100 = easier schedule."
    )

    st.markdown("### Schedule and Luck")

    schedule_columns = [
        "Manager", "Actual record",
        "Expected wins", "Schedule luck",
        "Opponent PPG",
    ]

    schedule_display = current[
        schedule_columns
    ].copy()

    for column in [
        "Expected wins", "Schedule luck", "Opponent PPG"
    ]:
        schedule_display[column] = (
            schedule_display[column].round(2)
        )

    st.dataframe(
        schedule_display,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Schedule luck = actual win equivalents minus "
        "all-play expected wins. Negative values indicate "
        "fewer wins than scoring results would suggest."
    )

    st.markdown("### Ranking Breakdown")

    breakdown_columns = [
        "Rank", "Manager",
        "All-play contribution",
        "Scoring contribution",
        "Schedule contribution",
        "Consistency contribution",
        "Power Score",
    ]

    breakdown = current[breakdown_columns].copy()

    for column in breakdown_columns[2:]:
        breakdown[column] = breakdown[column].round(2)

    st.dataframe(
        breakdown,
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Ranking History")

    chart_history = history.loc[
        history["Week"].le(selected_week)
    ].copy()

    if visible_names is not None:
        chart_history = chart_history.loc[
            chart_history["Manager"].isin(visible_names)
        ]

    chart_data = chart_history.pivot(
        index="Week",
        columns="Manager",
        values="Rank",
    )

    if not chart_data.empty:
        st.line_chart(chart_data)

    st.download_button(
        "Download power rankings CSV",
        data=current.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"power_rankings_week_{int(selected_week)}.csv"
        ),
        mime="text/csv",
        key="league_power_download",
    )
