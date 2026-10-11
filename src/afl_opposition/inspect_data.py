"""Inspect raw match events without cleaning them or constructing ML examples."""

import argparse
from pathlib import Path

import pandas as pd


CHAIN_KEY = ["match_id", "period", "chain_number"]
EVENT_KEY = [*CHAIN_KEY, "display_order"]
REQUIRED_COLUMNS = [
    *EVENT_KEY, "description", "period_seconds", "team_id", "chain_team_id",
    "player_id", "x", "y", "shot_at_goal", "season", "date",
]


def inspect_events(events: pd.DataFrame) -> str:
    """Describe observed rows; raw chains are not assumed to be possessions."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(events.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    if events.empty:
        raise ValueError("The dataset contains no rows.")

    # Raw display order breaks ties when multiple annotations share a timestamp.
    ordered = events.sort_values(["match_id", "period", "display_order"])
    known_teams = events["team_id"].notna() & events["chain_team_id"].notna()
    other_team = known_teams & events["team_id"].ne(events["chain_team_id"])
    # Null shot flags remain null in the data; this mask only counts explicit True.
    shots = events["shot_at_goal"].eq(True).fillna(False)
    gaps = ordered.groupby(["match_id", "period"])["period_seconds"].diff()
    null_counts = events.isna().sum()
    schema = pd.DataFrame({
        "dtype": events.dtypes.astype(str),
        "missing": null_counts,
        "missing_pct": (100 * null_counts / len(events)).round(2),
    })
    first = ordered.iloc[0]
    passage = ordered.loc[
        ordered["match_id"].eq(first["match_id"])
        & ordered["period"].eq(first["period"])
    ].head(25)
    passage_columns = [
        "display_order", "chain_number", "period_seconds", "description",
        "team_id", "chain_team_id", "x", "y", "shot_at_goal",
    ]

    sections = [
        "RAW EVENT DATA INSPECTION",
        f"Seasons: {sorted(events['season'].dropna().unique().tolist())}",
        f"Date range: {events['date'].min()} to {events['date'].max()}",
        f"Rows: {len(events):,}; columns: {len(events.columns)}",
        f"Matches: {events['match_id'].nunique():,}",
        f"Raw chains: {events[CHAIN_KEY].drop_duplicates().shape[0]:,}",
        f"Distinct action descriptions: {events['description'].nunique():,}",
        f"Extra rows with a duplicate raw event key: {events.duplicated(EVENT_KEY).sum():,}",
        f"Rows with a missing raw event key field: {events[EVENT_KEY].isna().any(axis=1).sum():,}",
        f"Rows with known acting and chain teams: {known_teams.sum():,}",
        f"Rows where acting team differs from chain team: {other_team.sum():,}",
        f"Raw chains containing such rows: {events.loc[other_team, CHAIN_KEY].drop_duplicates().shape[0]:,}",
        f"Explicit shot flags: {shots.sum():,}",
        f"Shot flags where acting team differs from chain team: {(shots & other_team).sum():,}",
        f"Adjacent rows at the same time within a match-quarter: {gaps.eq(0).sum():,}",
        f"Time decreases in display order within a match-quarter: {gaps.lt(0).sum():,}",
        f"Raw x range: {events['x'].min()} to {events['x'].max()}",
        f"Raw y range: {events['y'].min()} to {events['y'].max()}",
        "\nSHOT FLAG VALUES (including missing)",
        events["shot_at_goal"].value_counts(dropna=False).to_string(),
        "\nACTION DESCRIPTIONS FOR EXPLICIT SHOT FLAGS",
        events.loc[shots, "description"].value_counts(dropna=False).to_string(),
        "\nSCHEMA AND MISSING VALUES",
        schema.to_string(),
        "\n20 MOST COMMON ACTION DESCRIPTIONS",
        events["description"].value_counts(dropna=False).head(20).to_string(),
        f"\nFIRST 25 ROWS: match {first['match_id']}, quarter {first['period']}",
        passage[passage_columns].to_string(index=False),
        "\nRows include annotations; raw chains may contain actions by both teams.",
        "This is an inspection, not a validated possession table or training dataset.",
    ]
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a downloaded Parquet file")
    parser.add_argument("--output", type=Path, help="Also save the inspection as plain text")
    args = parser.parse_args()
    try:
        # Preserve nullable booleans and integers instead of converting them to floats.
        events = pd.read_parquet(args.path, engine="pyarrow", dtype_backend="numpy_nullable")
        report = f"File: {args.path}\n\n" + inspect_events(events)
        if args.output is not None:
            if args.output.resolve() == args.path.resolve():
                raise ValueError("The report output must differ from the input data path.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        print(report)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Inspection failed: {error}\n")


if __name__ == "__main__":
    main()
