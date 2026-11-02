
"""Monte Carlo fantasy football playoff forecasts."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SIMULATIONS = 50_000
REGULAR_WEEKS = 14


def _column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f"Expected one of {candidates}; "
        f"available columns: {list(df.columns)}"
    )


def _prepare(games, mapping, season, schedule_path):
    season = int(season)

    schedule = pd.read_csv(
        schedule_path,
        dtype={
            "manager1_id": str,
            "manager2_id": str,
            "team1_id": str,
            "team2_id": str,
        },
    )

    schedule = schedule.loc[
        pd.to_numeric(schedule.season, errors="coerce").eq(season)
    ].copy()

    roster = mapping.loc[
        pd.to_numeric(mapping.season, errors="coerce").eq(season),
        ["manager_id", "manager_name"],
    ].drop_duplicates("manager_id")

    names = dict(
        zip(
            roster.manager_id.astype(str),
            roster.manager_name.astype(str),
        )
    )

    ids = sorted(names)
    n = len(ids)

    if n not in (10, 12):
        raise ValueError(
            f"Expected 10 or 12 managers; found {n}"
        )

    expected = REGULAR_WEEKS * n // 2

    if len(schedule) != expected:
        raise ValueError(
            f"Schedule has {len(schedule)} matchups; "
            f"expected {expected}"
        )

    for week in range(1, REGULAR_WEEKS + 1):
        wk = schedule.loc[
            pd.to_numeric(schedule.week).eq(week)
        ]

        participants = (
            list(wk.manager1_id.astype(str))
            + list(wk.manager2_id.astype(str))
        )

        if (
            len(participants) != n
            or set(participants) != set(ids)
            or len(set(participants)) != n
        ):
            raise ValueError(
                f"Week {week}: missing or duplicate managers"
            )

    g = games.loc[
        pd.to_numeric(games.season, errors="coerce").eq(season)
    ].copy()

    phase = _column(g, ["phase"])

    g = g.loc[
        g[phase].astype(str).str.lower().isin(
            ["regular season", "regular"]
        )
    ].copy()

    m1 = _column(g, ["manager1_id", "manager_1_id"])
    m2 = _column(g, ["manager2_id", "manager_2_id"])

    s1 = _column(
        g,
        ["score1", "manager1_score", "team1_score", "score_1"],
    )

    s2 = _column(
        g,
        ["score2", "manager2_score", "team2_score", "score_2"],
    )

    g["week"] = pd.to_numeric(g.week, errors="coerce")
    g["_s1"] = pd.to_numeric(g[s1], errors="coerce")
    g["_s2"] = pd.to_numeric(g[s2], errors="coerce")

    g = g.loc[g.week.between(1, REGULAR_WEEKS)]

    completed = []

    # Only consecutive, fully verified weeks count.
    for week in range(1, REGULAR_WEEKS + 1):
        actual = g.loc[g.week.eq(week)]

        scheduled = schedule.loc[
            pd.to_numeric(schedule.week).eq(week)
        ]

        if (
            len(actual) != len(scheduled)
            or not actual["_s1"].gt(0).all()
            or not actual["_s2"].gt(0).all()
        ):
            break

        expected_pairs = {
            frozenset((
                str(r.manager1_id),
                str(r.manager2_id),
            ))
            for r in scheduled.itertuples()
        }

        actual_pairs = {
            frozenset((
                str(r[m1]),
                str(r[m2]),
            ))
            for _, r in actual.iterrows()
        }

        if actual_pairs != expected_pairs:
            raise ValueError(
                f"Week {week}: results do not match "
                "the collected schedule"
            )

        completed.append(actual)

    actual = (
        pd.concat(completed, ignore_index=True)
        if completed
        else g.iloc[:0].copy()
    )

    return (
        ids,
        names,
        schedule,
        actual,
        len(completed),
        (m1, m2, "_s1", "_s2"),
    )


def _forecast(games, mapping, season, schedule_path):
    (
        ids,
        names,
        schedule,
        actual,
        last_week,
        cols,
    ) = _prepare(games, mapping, season, schedule_path)

    n = len(ids)
    index = {mid: i for i, mid in enumerate(ids)}

    wins = np.zeros(n)
    losses = np.zeros(n)
    points = np.zeros(n)

    samples = [[] for _ in ids]

    m1, m2, s1, s2 = cols

    for _, r in actual.iterrows():
        a = index[str(r[m1])]
        b = index[str(r[m2])]

        x = float(r[s1])
        y = float(r[s2])

        points[a] += x
        points[b] += y

        samples[a].append(x)
        samples[b].append(y)

        if x > y:
            wins[a] += 1
            losses[b] += 1
        elif y > x:
            wins[b] += 1
            losses[a] += 1
        else:
            wins[a] += 0.5
            wins[b] += 0.5
            losses[a] += 0.5
            losses[b] += 0.5

    pooled = [
        score
        for team_scores in samples
        for score in team_scores
    ]

    league_mean = (
        float(np.mean(pooled))
        if pooled else 110.0
    )

    league_std = (
        float(np.std(pooled, ddof=1))
        if len(pooled) > 1 else 25.0
    )

    league_std = max(12.0, league_std)

    # Shrink team estimates toward league average.
    # This reduces overconfidence early in the season.
    means = np.array([
        (sum(scores) + 4 * league_mean)
        / (len(scores) + 4)
        for scores in samples
    ])

    stds = []

    for scores in samples:
        count = len(scores)

        observed_variance = (
            np.var(scores, ddof=1)
            if count > 1 else league_std ** 2
        )

        variance = (
            max(count - 1, 0) * observed_variance
            + 4 * league_std ** 2
        ) / (max(count - 1, 0) + 4)

        stds.append(max(12.0, np.sqrt(variance)))

    stds = np.array(stds)

    rng = np.random.default_rng(
        2026 + int(season) + last_week * 1009 + n
    )

    simulated_wins = np.broadcast_to(
        wins, (SIMULATIONS, n)
    ).copy()

    simulated_points = np.broadcast_to(
        points, (SIMULATIONS, n)
    ).copy()

    remaining = schedule.loc[
        pd.to_numeric(schedule.week).gt(last_week)
    ].sort_values("week")

    for r in remaining.itertuples():
        a = index[str(r.manager1_id)]
        b = index[str(r.manager2_id)]

        x = np.maximum(
            0,
            rng.normal(means[a], stds[a], SIMULATIONS),
        )

        y = np.maximum(
            0,
            rng.normal(means[b], stds[b], SIMULATIONS),
        )

        simulated_wins[:, a] += (
            (x > y).astype(float)
            + 0.5 * (x == y)
        )

        simulated_wins[:, b] += (
            (y > x).astype(float)
            + 0.5 * (x == y)
        )

        simulated_points[:, a] += x
        simulated_points[:, b] += y

    # Sort by wins, then total points.
    # Manager index only breaks an exact residual tie.
    order = np.lexsort(
        (
            np.broadcast_to(np.arange(n), simulated_wins.shape),
            -simulated_points,
            -simulated_wins,
        ),
        axis=1,
    )

    seed_counts = np.zeros((n, n), dtype=np.int64)

    for seed in range(n):
        seed_counts[:, seed] = np.bincount(
            order[:, seed],
            minlength=n,
        )

    playoff_counts = seed_counts[:, :6].sum(axis=1)
    bye_counts = seed_counts[:, :2].sum(axis=1)

    # Playoff bracket:
    # Week 15: 3 vs 6, 4 vs 5
    # Week 16: 2 vs winner(3/6), 1 vs winner(4/5)
    # Week 17: championship

    def duel(a, b):
        x = np.maximum(
            0, rng.normal(means[a], stds[a])
        )
        y = np.maximum(
            0, rng.normal(means[b], stds[b])
        )

        return np.where(x >= y, a, b)

    winner_36 = duel(order[:, 2], order[:, 5])
    winner_45 = duel(order[:, 3], order[:, 4])

    finalist_2 = duel(order[:, 1], winner_36)
    finalist_1 = duel(order[:, 0], winner_45)

    champion = duel(finalist_1, finalist_2)

    championship_counts = np.bincount(
        champion, minlength=n
    )

    return {
        "ids": ids,
        "names": names,
        "week": last_week,
        "wins": wins,
        "losses": losses,
        "mean_wins": simulated_wins.mean(axis=0),
        "seed_counts": seed_counts,
        "playoff_counts": playoff_counts,
        "bye_counts": bye_counts,
        "championship_counts": championship_counts,
    }


def render_playoff_odds(games, mapping, season):
    st.subheader("Playoff Odds")

    schedule_path = Path(__file__).with_name(
        "future_schedule.csv"
    )

    if not schedule_path.exists():
        st.info(
            "No future_schedule.csv found for this league."
        )
        return

    try:
        result = _forecast(
            games,
            mapping,
            season,
            schedule_path,
        )
    except Exception as exc:
        st.error(
            f"Playoff forecast unavailable: {exc}"
        )
        return

    st.caption(
        f"{SIMULATIONS:,} simulated seasons | "
        f"Results through Week {result['week']} | "
        "Regular season ends Week 14 | "
        "Six playoff teams"
    )

    rows = []

    for i, manager_id in enumerate(result["ids"]):
        projected_wins = int(
            np.floor(result["mean_wins"][i] + 0.5)
        )

        projected_wins = min(
            14, max(0, projected_wins)
        )

        rows.append({
            "Manager": result["names"][manager_id],
            "Current Record": (
                f"{result['wins'][i]:g}-"
                f"{result['losses'][i]:g}"
            ),
            "Projected Final Record": (
                f"{projected_wins}-"
                f"{14 - projected_wins}"
            ),
            "Playoff Odds": (
                result["playoff_counts"][i] / SIMULATIONS
            ),
            "Bye Odds": (
                result["bye_counts"][i] / SIMULATIONS
            ),
            "Championship Odds": (
                result["championship_counts"][i]
                / SIMULATIONS
            ),
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["Playoff Odds", "Bye Odds", "Championship Odds"],
        ascending=False,
    )

    percentage_columns = [
        "Playoff Odds",
        "Bye Odds",
        "Championship Odds",
    ]

    st.dataframe(
        df.style.format({
            column: "{:.1%}"
            for column in percentage_columns
        }),
        height=460,
        hide_index=True,
        column_config={
            "Manager": st.column_config.TextColumn(
                "Manager",
                pinned=True,
            ),
        },
        use_container_width=True,
    )

    st.markdown("#### Final Seed Probabilities")

    labels = {
        mid: result["names"][mid]
        for mid in result["ids"]
    }

    selected = st.selectbox(
        "Select manager",
        result["ids"],
        format_func=lambda mid: labels[mid],
        key=f"playoff_seed_{season}",
    )

    idx = result["ids"].index(selected)

    seed_df = pd.DataFrame({
        "Final Seed": list(
            range(1, len(result["ids"]) + 1)
        ),
        "Probability": (
            result["seed_counts"][idx] / SIMULATIONS
        ),
    })

    st.dataframe(
        seed_df.style.format({
            "Probability": "{:.2%}"
        }).set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]}
        ]),
        height=460,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Forecast uses historical scoring averages and "
        "variability, adjusted toward the league average. "
        "Remaining games follow the actual Yahoo schedule. "
        "Injuries, lineup changes, trades, and future Yahoo "
        "projections are not modeled. Championship odds "
        "simulate the playoff bracket and do not yet account "
        "for completed postseason games."
    )
