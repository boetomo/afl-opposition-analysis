"""Construct current-state examples for the first subsequent-shot baseline."""

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from afl_opposition.preprocess import FEATURE_COLUMNS

LABEL_VERSION = "0.1.0"
COMPLETED_ENDINGS = {
    "opponent_control", "Out of Bounds", "Ball Up Call", "Goal", "Behind",
    "Out On Full", "Out On Full After Kick", "Rushed", "Centre Bounce",
    "restart_kickIn", "restart_throwIn", "restart_ballUp", "restart_centreBounce",
}
EXAMPLE_COLUMNS = [
    "event_id", "possession_id", "match_id", "period", "event_index", "date",
    "season", "team_id", "display_order", *FEATURE_COLUMNS, "target", "sample_weight",
]


@dataclass
class ExampleSet:
    examples: pd.DataFrame
    possession_audit: pd.DataFrame
    summary: dict


def build_examples(events: pd.DataFrame, possessions: pd.DataFrame) -> ExampleSet:
    """Label prefixes before the first explicitly recorded shot in an episode.

    Reviewed, censored and contradictory episodes are excluded as a whole.
    A negative denotes no recorded explicit shot before a supported endpoint,
    not independent verification that the real match contained no shot.
    """
    required_events = set(EXAMPLE_COLUMNS) - {"target", "sample_weight"}
    required_events.add("is_explicit_shot")
    required_possessions = {"possession_id", "match_id", "period", "team_id", "event_count",
                            "explicit_shot_count", "requires_review", "end_reason", "review_reasons"}
    if required_events - set(events.columns) or required_possessions - set(possessions.columns):
        raise ValueError("Incomplete preprocessing tables for example construction.")
    if events.empty or possessions.empty:
        raise ValueError("No processed events or possessions are available.")
    if events.event_id.duplicated().any() or possessions.possession_id.duplicated().any():
        raise ValueError("Event and possession identifiers must be unique.")
    if events[list(required_events)].isna().any().any():
        # Numeric missingness is allowed only in quarantined episodes, checked below.
        essential = ["event_id", "possession_id", "match_id", "event_index", "date", "team_id", "is_explicit_shot"]
        if events[essential].isna().any().any():
            raise ValueError("Missing event identity, date, team or explicit-shot indicator.")
    if not events.is_explicit_shot.isin([True, False]).all():
        raise ValueError("Explicit-shot indicators must be Boolean.")
    if set(events.possession_id) != set(possessions.possession_id):
        raise ValueError("Events and possessions do not describe the same episodes.")
    grouped = events.groupby("possession_id")
    sizes = grouped.size()
    audit = possessions.copy().set_index("possession_id", drop=False)
    if not sizes.eq(audit.event_count).all():
        raise ValueError("Episode event counts disagree with the event table.")
    shots = grouped.is_explicit_shot.sum()
    if not shots.eq(audit.explicit_shot_count).all():
        raise ValueError("Episode shot counts disagree with the event table.")
    if events.duplicated(["possession_id", "event_index"]).any() or not grouped.event_index.min().eq(0).all() or not grouped.event_index.max().eq(sizes - 1).all():
        raise ValueError("Each episode needs contiguous, zero-based event indices.")
    for col in ["match_id", "period", "team_id"]:
        if not grouped[col].nunique().eq(1).all() or not grouped[col].first().eq(audit[col]).all():
            raise ValueError(f"Inconsistent {col} in a possession.")

    first_shot = events.loc[events.is_explicit_shot].groupby("possession_id").event_index.min()
    audit["first_shot_index"] = audit.index.map(first_shot).astype("Int64")
    audit["target"] = audit.first_shot_index.notna().astype("Int64")
    audit["eligible_prefix_count"] = audit.first_shot_index.fillna(audit.event_count).astype("Int64")
    audit["status"] = "eligible"
    audit.loc[audit.eligible_prefix_count.eq(0), "status"] = "no_pre_shot_action"
    audit.loc[audit.end_reason.isin(["Goal", "Behind"]) & audit.target.eq(0), "status"] = "score_without_shot_flag"
    audit.loc[~audit.end_reason.isin(COMPLETED_ENDINGS), "status"] = "unresolved_endpoint"
    audit.loc[audit.requires_review.fillna(True), "status"] = "preprocessing_review"
    if "label_requires_review" in audit:
        audit.loc[audit.label_requires_review.fillna(True), "status"] = "shot_marker_conflict"
    audit.loc[audit.status.ne("eligible"), "eligible_prefix_count"] = 0
    audit.loc[audit.status.ne("eligible"), "target"] = pd.NA

    eligible = audit.loc[audit.status.eq("eligible")]
    candidates = events.loc[events.possession_id.isin(eligible.index)].copy()
    candidates["prefix_count"] = candidates.possession_id.map(eligible.eligible_prefix_count)
    candidates = candidates.loc[candidates.event_index.lt(candidates.prefix_count)].copy()
    candidates["target"] = candidates.possession_id.map(eligible.target).astype(int)
    candidates["sample_weight"] = 1.0 / candidates.prefix_count.astype(float)
    examples = candidates[EXAMPLE_COLUMNS].sort_values(["date", "match_id", "period", "display_order"]).reset_index(drop=True)
    if examples.empty:
        raise ValueError("No eligible pre-shot examples remain after exclusions.")
    if examples[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Eligible examples contain missing model features.")
    for column in FEATURE_COLUMNS[1:]:
        if not examples[column].map(isfinite).all():
            raise ValueError("Eligible examples contain non-finite numeric features.")
    summary = {
        "label_version": LABEL_VERSION,
        "total_possessions": len(audit),
        "included_possessions": int(audit.status.eq("eligible").sum()),
        "excluded_possessions": int(audit.status.ne("eligible").sum()),
        "examples": len(examples),
        "positive_possessions": int(eligible.target.eq(1).sum()),
        "negative_possessions": int(eligible.target.eq(0).sum()),
        "status_counts": {str(k): int(v) for k, v in audit.status.value_counts().items()},
    }
    return ExampleSet(examples, audit.reset_index(drop=True), summary)


def chronological_split(events: pd.DataFrame, train_fraction: float = 0.7) -> pd.DataFrame:
    """Assign whole match dates to a development train/validation split.

    The cutoff approximates the requested fraction of matches. All matches on
    the cutoff date enter validation so dates never overlap between partitions.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    matches = events[["match_id", "date"]].drop_duplicates().copy()
    if matches.match_id.duplicated().any() or matches.isna().any().any():
        raise ValueError("Each match needs exactly one populated date.")
    matches["date"] = pd.to_datetime(matches.date, errors="raise").dt.normalize()
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    if len(matches) < 2 or matches.date.nunique() < 2:
        raise ValueError("At least two distinct match dates are required.")
    boundary = min(max(int(len(matches) * train_fraction), 1), len(matches) - 1)
    cutoff = matches.loc[boundary, "date"]
    matches["split"] = matches.date.ge(cutoff).map({False: "train", True: "validation"})
    if matches.split.nunique() != 2:
        raise ValueError("The requested cutoff leaves an empty split; choose another fraction.")
    return matches


def season_split(events: pd.DataFrame, *, train_seasons=(2023, 2024),
                 validation_season=2025, test_season=2026) -> pd.DataFrame:
    """Assign whole seasons and reject overlap, missing seasons or date conflicts.

    The test season may be absent when it is being kept unopened. Training and
    validation seasons must all be present; unassigned seasons are errors.
    """
    train_seasons = tuple(train_seasons)
    assigned = [*train_seasons, validation_season, test_season]
    if not train_seasons or len(set(assigned)) != len(assigned):
        raise ValueError("Season assignments must be nonempty and disjoint.")
    if max(train_seasons) >= validation_season or validation_season >= test_season:
        raise ValueError("Training seasons must precede validation and test.")
    matches = events[["match_id", "date", "season"]].drop_duplicates().copy()
    if matches.empty or matches.isna().any().any() or matches.match_id.duplicated().any():
        raise ValueError("Each match needs exactly one populated date and season.")
    present = set(matches.season)
    if not set([*train_seasons, validation_season]) <= present or present - set(assigned):
        raise ValueError("Missing required seasons or unexpected unassigned seasons.")
    matches["date"] = pd.to_datetime(matches.date, errors="raise").dt.normalize()
    if not matches.date.dt.year.eq(matches.season).all():
        raise ValueError("A match date conflicts with its season.")
    mapping = {season: "train" for season in train_seasons}
    mapping.update({validation_season: "validation", test_season: "test"})
    matches["split"] = matches.season.map(mapping)
    return matches.sort_values(["date", "match_id"]).reset_index(drop=True)
