"""Construct observed prefixes and next-action/location labels for AFL possessions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afl_opposition.action_rules import ACTION_TYPES

CONTRACT_VERSION = "next-event-v1"
END = "end_possession"
# A declared domain vocabulary, not categories discovered from the test season.
ACTIONS = tuple(sorted(set(ACTION_TYPES.values()))) + (END,)
NUMERIC = (
    "x_norm", "y_norm", "gap_seconds", "elapsed_seconds",
    "period_seconds", "period", "dx_norm", "dy_norm",
    "distance_to_goal_norm", "angle_to_goal",
)
KNOWN_ENDS = {
    "opponent_control", "Goal", "Behind", "Rushed", "Ball Up Call",
    "Out of Bounds", "Out On Full", "Out On Full After Kick",
    "Centre Bounce", "restart_kickIn", "restart_throwIn",
    "restart_ballUp", "restart_centreBounce",
}


@dataclass
class NextEvents:
    events: pd.DataFrame
    coverage: pd.DataFrame
    summary: dict


def observed_features(events):
    """Only current/past observations; no outcomes, shot flags or final scores."""
    result = events.sort_values(["match_id", "period", "display_order"]).copy()
    groups = result.groupby("possession_id", sort=False)
    for coordinate in ("x_norm", "y_norm"):
        result["d" + coordinate] = groups[coordinate].diff().fillna(0.)
    # Dimensionless geometry in the venue-normalised frame, not metres.
    result["distance_to_goal_norm"] = np.hypot(1 - result.x_norm, result.y_norm)
    result["angle_to_goal"] = np.arctan2(result.y_norm, 1 - result.x_norm)
    return result


def build_next_events(events, possessions):
    """Retain clean episodes, keep internal pairs in censored final episodes.

    Each labelled prefix has one action target. Location exists only for a
    subsequent retained action in the SAME episode. End-of-archive is not END.
    Review-based filtering is retrospective sample selection, not a model input.
    """
    if events.empty or events.event_id.duplicated().any() or possessions.possession_id.duplicated().any():
        raise ValueError("Nonempty events and unique event/possession IDs are required.")
    if events[["match_id", "period", "display_order"]].duplicated().any():
        raise ValueError("Event order must be unique within each match-quarter.")
    audit = possessions.copy()
    reasons = audit.review_reasons.fillna("").astype(str)
    # Censoring alone does not invalidate earlier observed transitions.
    acceptable = reasons.isin(["", "end_of_available_events"])
    acceptable &= ~audit.requires_review.fillna(True) | reasons.eq("end_of_available_events")
    audit["included"] = acceptable
    frame = events.merge(audit[["possession_id", "end_reason", "included"]],
                         on="possession_id", how="left", validate="many_to_one")
    if frame.included.isna().any():
        raise ValueError("An event has no possession record.")
    frame = observed_features(frame.loc[frame.included]).reset_index(drop=True)
    if frame.empty or not np.isfinite(frame[list(NUMERIC)].to_numpy(dtype=float)).all():
        raise ValueError("Eligible observations must have finite numeric features.")
    if not frame.action_type.isin(ACTIONS[:-1]).all():
        raise ValueError("Unsupported action type in the retained stream.")
    groups = frame.groupby("possession_id", sort=False)
    if not frame.event_index.eq(groups.cumcount()).all():
        raise ValueError("Episodes must contain complete, ordered prefixes starting at zero.")
    if (groups[["match_id", "period", "season", "team_id"]].nunique() != 1).any().any():
        raise ValueError("An episode crosses a match, quarter, season or acting team.")
    if (groups.period_seconds.diff().dropna() < 0).any():
        raise ValueError("Time reversals are not valid histories.")
    frame["next_event_id"] = groups.event_id.shift(-1)
    frame["target_action"] = groups.action_type.shift(-1)
    frame["target_x"] = groups.x_norm.shift(-1)
    frame["target_y"] = groups.y_norm.shift(-1)
    frame["has_location"] = frame.next_event_id.notna()
    terminal = ~frame.has_location & frame.end_reason.isin(KNOWN_ENDS)
    frame.loc[terminal, "target_action"] = END
    frame["labelled"] = frame.target_action.notna()
    # Masked placeholders are never scored as a location.
    frame[["target_x", "target_y"]] = frame[["target_x", "target_y"]].fillna(0.)
    frame["history_start"] = np.arange(len(frame)) - frame.event_index.to_numpy(dtype=int)
    season_lookup = events.groupby("possession_id").season.first()
    audit["season"] = audit.possession_id.map(season_lookup)
    coverage = audit.groupby(["season", "team_id", "included", "review_reasons"],
                             dropna=False).size().rename("possessions").reset_index()
    summary = {
        "contract": CONTRACT_VERSION, "actions": list(ACTIONS), "numeric_features": list(NUMERIC),
        "source_events": len(events), "retained_events": len(frame),
        "source_possessions": len(audit), "retained_possessions": int(acceptable.sum()),
        "labelled_prefixes": int(frame.labelled.sum()),
        "location_targets": int(frame.has_location.sum()), "end_targets": int(terminal.sum()),
        "censored_last_prefixes": int((~frame.labelled).sum()),
        "weighting": "Every labelled event prefix has equal weight.",
        "selection": "Exclude reviewed episodes except censoring alone; retrospective archive-quality selection.",
    }
    return NextEvents(frame, coverage, summary)


def summarise_by_season(frame):
    return frame.groupby("season").agg(
        matches=("match_id", "nunique"), possessions=("possession_id", "nunique"),
        observations=("event_id", "size"), labelled_prefixes=("labelled", "sum"),
        locations=("has_location", "sum"),
    ).reset_index()
