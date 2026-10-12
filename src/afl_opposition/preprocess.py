"""Build auditable ball-control sequences from raw torpdata annotations.

Usage: python -m afl_opposition.preprocess --season 2023
This stage does not fit scalers, split matches, construct targets or train models.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from math import hypot, isfinite
from pathlib import Path

import pandas as pd

from afl_opposition.action_rules import (
    ACTION_TYPES, KNOWN_DESCRIPTIONS, RULE_VERSION, action_role,
)
from afl_opposition.data import verify_file

ORDER_KEY = ["match_id", "period", "display_order"]
CHAIN_KEY = ["match_id", "period", "chain_number"]
REQUIRED_COLUMNS = list(dict.fromkeys([
    *ORDER_KEY, "chain_number", "chain_team_id", "initial_state",
    "description", "period_seconds", "team_id", "player_id", "x", "y",
    "shot_at_goal", "venue_length", "venue_width", "date", "season",
]))
# Only event-time inputs belong here. All other output columns are identifiers,
# provenance or audit metadata. Shot flags and possession summaries are NOT inputs.
FEATURE_COLUMNS = ["action_type", "x_norm", "y_norm", "gap_seconds", "elapsed_seconds"]
EVENT_COLUMNS = [
    "event_id", "possession_id", "event_index", "match_id", "period",
    "chain_number", "display_order", "date", "season", "period_seconds",
    "description", "action_type", "team_id", "player_id", "raw_x", "raw_y",
    "chain_team_id", "x_attacking", "y_attacking", "x_norm", "y_norm",
    "gap_seconds", "elapsed_seconds", "is_explicit_shot",
]
POSSESSION_COLUMNS = [
    "possession_id", "match_id", "period", "team_id", "start_event_id",
    "last_event_id", "start_seconds", "last_event_seconds", "closed_at_seconds",
    "end_reason", "event_count", "explicit_shot_count", "requires_review", "review_reasons",
]


@dataclass
class PreprocessingResult:
    events: pd.DataFrame
    possessions: pd.DataFrame
    annotations: pd.DataFrame
    action_audit: pd.DataFrame
    summary: dict


def _prepare(raw: pd.DataFrame, *, quarantine_time_reversals=False) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    if raw.empty:
        raise ValueError("The dataset contains no rows.")
    if raw[[*ORDER_KEY, "chain_number", "period_seconds", "description"]].isna().any().any():
        raise ValueError("Event identifiers, times and descriptions must be populated.")
    if raw.duplicated(ORDER_KEY).any():
        raise ValueError("Event order must be unique within each match-quarter.")
    for col in ["period", "chain_number", "display_order", "period_seconds"]:
        values = pd.to_numeric(raw[col], errors="raise")
        if not values.map(isfinite).all() or (values < 0).any():
            raise ValueError(f"Invalid non-negative numeric values in {col}.")
        if col != "period_seconds" and (values % 1 != 0).any():
            raise ValueError(f"Non-integer identifier in {col}.")
    rows = raw[REQUIRED_COLUMNS].sort_values(ORDER_KEY, kind="stable").reset_index(drop=True).copy()
    gaps = rows.groupby(["match_id", "period"])["period_seconds"].diff()
    reversed_quarters = gaps.lt(0).groupby([rows.match_id, rows.period]).transform("any")
    if reversed_quarters.any() and not quarantine_time_reversals:
        raise ValueError("Quarter timestamps decrease in display order.")
    if reversed_quarters.all():
        raise ValueError("No quarters remain after quarantining reversed timestamps.")
    # A contiguous chain has one declared start category and coordinate frame.
    chain_groups = rows.groupby(CHAIN_KEY)
    for col in ["initial_state", "chain_team_id"]:
        if chain_groups[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent {col} within a raw chain.")
    rows["raw_event_id"] = (
        rows.match_id.astype(str) + ":q" + rows.period.astype(str)
        + ":c" + rows.chain_number.astype(str) + ":e" + rows.display_order.astype(str)
    )
    rows["role"] = rows.description.map(action_role)
    rows["handling"] = rows.role.map({
        "action": "event", "context": "context_only", "annotation": "auxiliary_annotation",
        "stoppage": "stoppage", "restart": "restart", "review": "review",
        "kick_annotation": "unpaired_kick_annotation",
    })
    rows["canonical_event_id"] = rows.raw_event_id
    rows["review_reason"] = ""
    rows.loc[rows.role.eq("review"), "review_reason"] = "unsupported_description"

    prev = rows.shift()
    same_quarter = rows.match_id.eq(prev.match_id) & rows.period.eq(prev.period)
    # Both participants must be known. Matching a missing identity is insufficient.
    same_action = rows[["description", "period_seconds", "team_id", "player_id"]].eq(
        prev[["description", "period_seconds", "team_id", "player_id"]]
    ).fillna(False).all(axis=1)
    changed_chain = rows.chain_number.ne(prev.chain_number)
    changed_frame = rows.chain_team_id.ne(prev.chain_team_id).fillna(False)
    mirrored = rows.x.eq(-prev.x) & rows.y.eq(-prev.y)
    same_frame = rows.chain_team_id.eq(prev.chain_team_id).fillna(False)
    same_coords = rows.x.eq(prev.x) & rows.y.eq(prev.y)
    candidates = (same_quarter & changed_chain & same_action).fillna(False)
    duplicates = (candidates & ((changed_frame & mirrored) | (same_frame & same_coords))).fillna(False)
    # Keep the FIRST observation. It was available earlier, before the next chain.
    rows["canonical_event_id"] = rows.raw_event_id.mask(duplicates).ffill()
    rows.loc[duplicates, "handling"] = "boundary_duplicate"
    rows.loc[candidates & ~duplicates, "review_reason"] = "ambiguous_boundary_repeat"

    pair_fields = ["match_id", "period", "chain_number", "period_seconds", "team_id", "player_id", "x", "y"]
    paired_kick = (
        rows.description.eq("Kick Into F50") & prev.description.isin(["Kick", "Ground Kick"])
        & rows[pair_fields].eq(prev[pair_fields]).fillna(False).all(axis=1)
    ).fillna(False)
    rows.loc[paired_kick, "handling"] = "paired_kick_annotation"
    rows.loc[paired_kick, "canonical_event_id"] = rows.canonical_event_id.shift().loc[paired_kick]
    rows.loc[rows.handling.eq("unpaired_kick_annotation"), "review_reason"] = "unpaired_kick_annotation"
    missing_actor = rows.role.eq("action") & rows.team_id.isna()
    rows.loc[missing_actor, "handling"] = "review"
    rows.loc[missing_actor, "review_reason"] = "missing_action_team"
    rows["quarantined"] = reversed_quarters
    rows.loc[reversed_quarters, "handling"] = "quarantined_quarter"
    rows.loc[reversed_quarters, "review_reason"] = "quarter_time_reversal"
    rows.loc[reversed_quarters, "canonical_event_id"] = rows.loc[reversed_quarters, "raw_event_id"]
    return rows


def preprocess_events(raw: pd.DataFrame, *, quarantine_time_reversals=False) -> PreprocessingResult:
    """Return copied, traceable tables; leave the input DataFrame unchanged.

    A possession is an observed same-team ball-control episode. It starts at an
    accepted action and ends on an opponent's accepted action or a stoppage.
    Missing evidence and unsupported actions flag affected episodes for review.
    """
    annotations = _prepare(raw, quarantine_time_reversals=quarantine_time_reversals)
    chain_issues = {}
    for row in annotations.loc[annotations.review_reason.ne("")].itertuples():
        key = (row.match_id, row.period, row.chain_number)
        chain_issues.setdefault(key, set()).add(row.review_reason)
    actions, possessions = [], []

    for (match_id, period), group in annotations.groupby(["match_id", "period"], sort=False):
        if group.quarantined.iloc[0]:
            continue
        current = None
        last_chain = None

        def close(reason, seconds):
            nonlocal current
            if current is None:
                return
            issues = set(current.pop("issues"))
            for chain in current.pop("chains"):
                issues.update(chain_issues.get((match_id, period, chain), set()))
            if reason in {"unresolved_chain_change", "end_of_available_events"}:
                issues.add(reason)
            current.update(end_reason=reason, closed_at_seconds=seconds,
                           requires_review=bool(issues), review_reasons=";".join(sorted(issues)))
            possessions.append(current)
            current = None

        # Replace nullable pandas scalars with None for explicit state transitions.
        records = group.astype(object).where(group.notna(), None).to_dict("records")
        for row in records:
            seconds = row["period_seconds"]
            if row["chain_number"] != last_chain:
                if current is not None:
                    if row["initial_state"] in {"ballUp", "throwIn", "centreBounce", "kickIn"}:
                        close("restart_" + row["initial_state"], seconds)
                    elif row["chain_team_id"] != current["team_id"] and not (
                        row["role"] == "action" and row["team_id"] is not None
                        and row["handling"] != "boundary_duplicate"
                    ):
                        close("unresolved_chain_change", seconds)
                last_chain = row["chain_number"]
            if current is not None:
                current["chains"].add(row["chain_number"])
            if row["handling"] == "boundary_duplicate":
                continue
            if row["role"] in {"stoppage", "restart"}:
                close(row["description"], seconds)
                continue
            if row["role"] != "action":
                continue
            actor = row["team_id"]
            if actor is None:
                key = (match_id, period, row["chain_number"])
                chain_issues.setdefault(key, set()).add("missing_action_team")
                continue
            if current is not None and current["team_id"] != actor:
                close("opponent_control", seconds)
            if current is None:
                pid = row["raw_event_id"] + ":pos"
                current = dict(
                    possession_id=pid, match_id=match_id, period=period, team_id=actor,
                    start_event_id=row["raw_event_id"], last_event_id=row["raw_event_id"],
                    start_seconds=seconds, last_event_seconds=seconds, event_count=0,
                    explicit_shot_count=0, issues=set(), chains={row["chain_number"]},
                )
            sign = 1 if actor == row["chain_team_id"] else -1
            valid = all(row[c] is not None and isfinite(row[c]) for c in ["x", "y", "venue_length", "venue_width"])
            valid = valid and row["chain_team_id"] is not None and row["venue_length"] > 0 and row["venue_width"] > 0
            x = sign * row["x"] if valid else None
            y = sign * row["y"] if valid else None
            if not valid:
                current["issues"].add("missing_or_invalid_coordinates")
            elif abs(2 * x / row["venue_length"]) > 1.1 or abs(2 * y / row["venue_width"]) > 1.1:
                current["issues"].add("coordinate_outside_dimensions")
            if current["event_count"] and valid:
                prev_action = actions[-1]
                if prev_action["x_attacking"] is not None and hypot(x - prev_action["x_attacking"], y - prev_action["y_attacking"]) > 100:
                    current["issues"].add("coordinate_jump_over_100m")
            explicit_shot = row["shot_at_goal"] is True or row["shot_at_goal"] == 1
            event = {key: row[key] for key in ["match_id", "period", "chain_number", "display_order", "date", "season", "period_seconds", "description", "team_id", "player_id", "chain_team_id"]}
            event.update(
                event_id=row["raw_event_id"], possession_id=current["possession_id"],
                event_index=current["event_count"], action_type=ACTION_TYPES[row["description"]],
                raw_x=row["x"], raw_y=row["y"], x_attacking=x, y_attacking=y,
                x_norm=2 * x / row["venue_length"] if valid else None,
                y_norm=2 * y / row["venue_width"] if valid else None,
                gap_seconds=seconds - current["last_event_seconds"],
                elapsed_seconds=seconds - current["start_seconds"], is_explicit_shot=bool(explicit_shot),
            )
            actions.append(event)
            current["event_count"] += 1
            current["explicit_shot_count"] += int(bool(explicit_shot))
            current["last_event_id"] = row["raw_event_id"]
            current["last_event_seconds"] = seconds
        close("end_of_available_events", records[-1]["period_seconds"])

    events = pd.DataFrame(actions, columns=EVENT_COLUMNS).convert_dtypes()
    possession_table = pd.DataFrame(possessions, columns=POSSESSION_COLUMNS).convert_dtypes()
    # Map all source annotations, including boundary copies, back to a kept event.
    event_to_pos = events.set_index("event_id")["possession_id"]
    annotations["possession_id"] = annotations.canonical_event_id.map(event_to_pos)
    action_audit = annotations.groupby(["description", "role", "handling"], dropna=False).size().rename("rows").reset_index()
    action_audit["known_description"] = action_audit.description.isin(KNOWN_DESCRIPTIONS)
    summary = {
        "rule_version": RULE_VERSION, "raw_rows": len(raw), "event_rows": len(events),
        "possessions": len(possession_table),
        "possessions_requiring_review": int(possession_table.requires_review.sum()),
        "boundary_duplicates": int(annotations.handling.eq("boundary_duplicate").sum()),
        "paired_kick_annotations": int(annotations.handling.eq("paired_kick_annotation").sum()),
        "unpaired_kick_annotations": int(annotations.handling.eq("unpaired_kick_annotation").sum()),
        "raw_explicit_shots": int(raw.shot_at_goal.eq(True).fillna(False).sum()),
        "time_reversal_policy": "quarantine_quarter_v1" if quarantine_time_reversals else "reject",
        "quarantined_rows": int(annotations.quarantined.sum()),
        "quarantined_explicit_shots": int(annotations.loc[annotations.quarantined, "shot_at_goal"].eq(True).fillna(False).sum()),
        "quarantined_quarters": annotations.loc[annotations.quarantined, ["match_id", "period"]].drop_duplicates().to_dict("records"),
        "processed_explicit_shots": int(events.is_explicit_shot.sum()),
        "unknown_descriptions": sorted(set(raw.description) - KNOWN_DESCRIPTIONS),
        "feature_columns": FEATURE_COLUMNS,
        "handling_counts": {k: int(v) for k, v in annotations.handling.value_counts().items()},
        "end_reason_counts": {k: int(v) for k, v in possession_table.end_reason.value_counts().items()},
    }
    return PreprocessingResult(events, possession_table, annotations, action_audit, summary)


def write_result(result: PreprocessingResult, output_dir: Path, source_path: Path, *, overwrite=False) -> None:
    """Write derived tables with provenance. Refuse to replace the source file."""
    tables = {"events": result.events, "possessions": result.possessions,
              "annotations": result.annotations, "action_audit": result.action_audit}
    outputs = [output_dir / (name + ".parquet") for name in tables] + [output_dir / "summary.json"]
    if any(path.resolve() == source_path.resolve() for path in outputs):
        raise ValueError("A derived output must not overwrite the source file.")
    if not overwrite and any(path.exists() for path in outputs):
        raise ValueError("Derived files already exist; use --overwrite to regenerate them.")
    with source_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = {**result.summary, "source_filename": source_path.name, "source_sha256": digest}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(output_dir / (name + ".parquet"), index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing derived files")
    parser.add_argument("--quarantine-time-reversals", action="store_true",
                        help="Exclude entire quarters with reversed timestamps, preserving their annotations")
    args = parser.parse_args()
    try:
        manifest = json.loads(Path("data/sources.json").read_text(encoding="utf-8"))
        source = manifest["seasons"].get(str(args.season))
        if source is None:
            raise ValueError("This season has no recorded source in data/sources.json.")
        path = Path("data/raw") / source["filename"]
        verify_file(path, source)
        raw = pd.read_parquet(path, engine="pyarrow", dtype_backend="numpy_nullable")
        result = preprocess_events(raw, quarantine_time_reversals=args.quarantine_time_reversals)
        write_result(result, Path("data/processed") / str(args.season), path, overwrite=args.overwrite)
        print(json.dumps(result.summary, indent=2))
        print(f"Saved derived tables in data/processed/{args.season}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Preprocessing failed: {error}\n")


if __name__ == "__main__":
    main()
