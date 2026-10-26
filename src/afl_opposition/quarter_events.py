"""Team-specific Seq2Event streams and chronological quarter streams.

Archived ball-control events are retained in order. Reviewed observations remain
visible as history but do not form scored transitions. Observed stoppages and
restarts have a neutral boundary token. Unresolved raw actions have a gap token.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from afl_opposition.action_rules import ACTION_TYPES
from afl_opposition.baseline import load_preprocessed_seasons

CONTRACT = "quarter-events-v1"
BALL_ACTIONS = tuple(sorted(set(ACTION_TYPES.values())))
BOUNDARY, GAP = "boundary", "gap"
INPUT_ACTIONS = (*BALL_ACTIONS, BOUNDARY, GAP)
TARGET_NAMES = tuple(f"{side}:{a}" for side in ("home", "away") for a in BALL_ACTIONS) + (BOUNDARY,)
NUMERIC = ("x", "y", "period_seconds", "period", "has_coordinates",
           "gap_seconds", "dx", "dy", "distance_to_goal", "angle_to_goal")
OBSERVED_NUMERIC = ("x_home", "y_home", "period_seconds", "period", "has_coordinates",
                    "distance_to_goal", "angle_to_goal")
MODES = ("joint", "team_stream")
K = len(BALL_ACTIONS)


@dataclass
class QuarterData:
    events: pd.DataFrame
    audit: dict
    provenance: dict


def load_quarters(root, seasons=(2023, 2024, 2025)):
    root = Path(root)
    events, possessions, provenance = load_preprocessed_seasons(root, seasons)
    manifest = json.loads((root / "data/sources.json").read_text())
    annotations, metadata = [], []
    for season in seasons:
        annotations.append(pd.read_parquet(root / f"data/processed/{season}/annotations.parquet"))
        metadata.append(pd.read_parquet(root / "data/raw" / manifest["seasons"][str(season)]["filename"],
            columns=["match_id", "home_team_id", "away_team_id"]).drop_duplicates())
    frame, audit = build_quarters(events, possessions, pd.concat(annotations, ignore_index=True),
                                   pd.concat(metadata, ignore_index=True).drop_duplicates())
    return QuarterData(frame, audit, provenance)


def build_quarters(events, possessions, annotations, metadata):
    """Return observed tokens, not labels. Coordinate frame is home attacking +x.

    Periods never join. No fixed context cap or silent history truncation exists.
    Eligibility is retrospective quality selection, never a numerical input.
    """
    if events.empty or events.event_id.duplicated().any() or metadata.match_id.duplicated().any() or possessions.possession_id.duplicated().any():
        raise ValueError("Require nonempty unique events and one metadata row per match.")
    if metadata[["home_team_id", "away_team_id"]].isna().any().any() or metadata.home_team_id.eq(metadata.away_team_id).any():
        raise ValueError("Each match needs two distinct, known teams.")
    audit = possessions.copy()
    reasons = audit.review_reasons.fillna("")
    audit["eligible"] = reasons.isin(["", "end_of_available_events"]) & (
        ~audit.requires_review.fillna(True) | reasons.eq("end_of_available_events"))
    frame = events.merge(audit[["possession_id", "eligible"]], on="possession_id", validate="many_to_one")
    frame = frame.merge(metadata, on="match_id", how="left", validate="many_to_one")
    if frame.eligible.isna().any() or frame.home_team_id.isna().any():
        raise ValueError("Missing possession or match metadata.")
    if not (frame.team_id.eq(frame.home_team_id) | frame.team_id.eq(frame.away_team_id)).all():
        raise ValueError("Acting team is not a participant in its match.")
    if not frame.action_type.isin(BALL_ACTIONS).all():
        raise ValueError("Unsupported ball-control action.")
    frame["side"] = np.where(frame.team_id.eq(frame.home_team_id), 0, 1)
    sign = np.where(frame.side.eq(0), 1., -1.)
    frame["x_home"], frame["y_home"] = frame.x_norm * sign, frame.y_norm * sign
    frame["has_coordinates"] = 1.
    frame["distance_to_goal"] = np.hypot(1 - frame.x_norm, frame.y_norm)
    frame["angle_to_goal"] = np.arctan2(frame.y_norm, 1 - frame.x_norm)
    frame["token_id"] = frame.event_id.astype(str)
    # Only observed timestamps are used for boundary tokens, never episode-end labels.
    raw = annotations.merge(metadata, on="match_id", validate="many_to_one")
    raw = raw.merge(events[["match_id", "period"]].drop_duplicates(), on=["match_id", "period"])
    raw = raw.loc[~raw.get("quarantined", pd.Series(False, index=raw.index)).fillna(False)]
    boundary = raw.handling.isin(["stoppage", "restart"])
    unresolved = raw.handling.isin(["review", "unpaired_kick_annotation"])
    extra = raw.loc[boundary | unresolved].copy()
    extra["action_type"] = np.where(extra.handling.isin(["stoppage", "restart"]), BOUNDARY, GAP)
    extra["eligible"] = extra.action_type.eq(BOUNDARY)
    extra["side"] = 2
    extra["token_id"] = extra.raw_event_id.astype(str) + ":" + extra.action_type
    for name in ("x_home", "y_home", "has_coordinates", "distance_to_goal", "angle_to_goal"):
        extra[name] = 0.
    columns = ["token_id", "match_id", "period", "season", "display_order", "period_seconds",
               "action_type", "side", "home_team_id", "away_team_id", "eligible", "x_home", "y_home",
               "has_coordinates", "distance_to_goal", "angle_to_goal"]
    frame = pd.concat([frame[columns], extra[columns]], ignore_index=True).sort_values(
        ["match_id", "period", "display_order"]).reset_index(drop=True)
    if frame[["match_id", "period", "display_order"]].duplicated().any():
        raise ValueError("Ambiguous token order: do not combine duplicate raw actions.")
    if not np.isfinite(frame[list(OBSERVED_NUMERIC)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite quarter inputs require explicit source review.")
    if (frame.groupby(["match_id", "period"]).period_seconds.diff().dropna() < 0).any():
        raise ValueError("Quarter timestamps reverse.")
    frame["quarter_id"] = frame.match_id.astype(str) + ":q" + frame.period.astype(str)
    frame["position"] = frame.groupby("quarter_id", sort=False).cumcount()
    counts = frame.groupby("quarter_id").size()
    return frame, dict(contract=CONTRACT, quarters=len(counts), tokens=len(frame),
        ball_actions=int(frame.has_coordinates.sum()), boundaries=int(frame.action_type.eq(BOUNDARY).sum()),
        gaps=int(frame.action_type.eq(GAP).sum()), ineligible_tokens=int((~frame.eligible).sum()),
        quarter_length_min=int(counts.min()), quarter_length_median=float(counts.median()),
        quarter_length_max=int(counts.max()), history_cap=None,
        quality_policy="Reviewed observations remain inputs; transitions touching them or gap tokens are not scored.",
        coordinate_frame="Home attacking direction within each quarter; dimensionless venue-normalised coordinates.")


def make_streams(frame, mode):
    """Attach formulation-specific next-token targets.

    Team streams keep each side's actions and neutral tokens. The first observed
    opposing action after own control inserts a boundary. Further opposing actions
    are skipped. Unknown passages remain gap tokens, rather than becoming targets.
    """
    if mode not in MODES:
        raise ValueError("Unknown stream formulation.")
    if mode == "team_stream":
        pieces = []
        for quarter, part in frame.groupby("quarter_id", sort=False):
            for side in (0, 1):
                rows, own_control = [], False
                for row in part.to_dict("records"):
                    if row["side"] == side:
                        rows.append(row.copy())
                        own_control = True
                    elif row["side"] == 2:
                        if own_control or (rows and row["action_type"] == GAP):
                            rows.append(row.copy())
                        own_control = False
                    elif own_control:
                        marker = row.copy()
                        marker.update(token_id=row["token_id"] + f":turnover:{side}", action_type=BOUNDARY, side=2)
                        for name in ("x_home", "y_home", "has_coordinates", "distance_to_goal", "angle_to_goal"):
                            marker[name] = 0.
                        rows.append(marker)
                        own_control = False
                    elif rows and not row["eligible"]:
                        # Do not skip an ambiguous opponent passage and silently join clean targets.
                        marker = row.copy()
                        marker.update(action_type=GAP, side=2, token_id=row["token_id"] + f":gap:{side}")
                        for name in ("x_home", "y_home", "has_coordinates", "distance_to_goal", "angle_to_goal"):
                            marker[name] = 0.
                        rows.append(marker)
                if rows:
                    piece = pd.DataFrame(rows)
                    piece["stream_id"], piece["focal_side"] = f"{quarter}:{side}", side
                    pieces.append(piece)
        result = pd.concat(pieces, ignore_index=True)
    else:
        result = frame.copy()
        result["stream_id"], result["focal_side"] = result.quarter_id, -1
    result = result.reset_index(drop=True)
    groups = result.groupby("stream_id", sort=False)
    result["position"] = groups.cumcount()
    # Team streams use the focal team's attacking frame. Joint streams use home +x.
    sign = np.where(result.focal_side.eq(1), -1., 1.)
    result["x"], result["y"] = result.x_home * sign, result.y_home * sign
    result["gap_seconds"] = groups.period_seconds.diff().fillna(0.)
    result["dx"], result["dy"] = groups.x.diff().fillna(0.), groups.y.diff().fillna(0.)
    # Locations of boundary/gap tokens are unknown, not the centre of the ground.
    prior_location = groups.has_coordinates.shift(1).fillna(0.)
    result.loc[~(result.has_coordinates.eq(1) & prior_location.eq(1)), ["dx", "dy"]] = 0.
    codes = result.action_type.map({a: i for i, a in enumerate(BALL_ACTIONS)})
    result["code"] = np.where(result.side.lt(2), result.side * K + codes.fillna(0), 2 * K)
    result.loc[result.action_type.eq(GAP), "code"] = -1
    if mode == "team_stream":
        result["code"] = codes.fillna(K).astype(int)
        result.loc[result.action_type.eq(GAP), "code"] = -1
    result["next_token_id"] = groups.token_id.shift(-1)
    result["target"] = groups.code.shift(-1).fillna(-1).astype(int)
    next_eligible = groups.eligible.shift(-1).fillna(False).astype(bool)
    result["labelled"] = result.eligible & next_eligible & result.target.ge(0)
    result.loc[~result.labelled, "target"] = -1
    result["target_x"] = groups.x.shift(-1).fillna(0.)
    result["target_y"] = groups.y.shift(-1).fillna(0.)
    result["has_location"] = result.labelled & result.target.lt(K if mode == "team_stream" else 2 * K)
    if mode == "team_stream":
        # Original code ignores coordinate errors for targets immediately following a boundary too.
        result["has_location"] &= result.has_coordinates.eq(1)
    return result
