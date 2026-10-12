"""Audit goals and label the existing pre-shot possession prefixes.

Usage: python -m afl_opposition.goal_labels
Only the 2023–2025 development seasons are loaded. No model is fitted.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

from afl_opposition.examples import LABEL_VERSION as SHOT_LABEL_VERSION, build_examples
from afl_opposition.preprocess import FEATURE_COLUMNS

GOAL_LABEL_VERSION = "0.1.0"
DEVELOPMENT_SEASONS = (2023, 2024, 2025)


@dataclass
class GoalLabelSet:
    examples: pd.DataFrame
    possession_audit: pd.DataFrame
    goal_annotation_audit: pd.DataFrame
    summary: dict


def build_goal_examples(events: pd.DataFrame, possessions: pd.DataFrame,
                        annotations: pd.DataFrame) -> GoalLabelSet:
    """Label whether the possessing team scores before its episode ends.

    Inputs are exactly the existing eligible pre-shot prefixes, subject to
    additional goal-attribution exclusions. Goals may follow a recovered miss
    in the same episode. All shot actions and later states remain outside X.
    Goal/team/end metadata are retrospective audit fields, never features.
    """
    shot = build_examples(events, possessions)
    required = {"raw_event_id", "match_id", "period", "display_order", "description",
                "period_seconds", "team_id", "season"}
    if required - set(annotations.columns):
        raise ValueError("Goal attribution needs the complete annotation identity and outcome fields.")
    keys = ["match_id", "period", "display_order"]
    if annotations[["raw_event_id", *keys, "description", "period_seconds", "season"]].isna().any().any():
        raise ValueError("Annotation identity, ordering and timestamps must be populated.")
    if annotations.duplicated(keys).any() or annotations.raw_event_id.duplicated().any():
        raise ValueError("Annotation identifiers and match-quarter order must be unique.")
    # The annotation stream must contain the retained actions used for linking.
    reference = events[["event_id", *keys, "team_id", "period_seconds", "season"]].set_index("event_id")
    source = annotations.set_index("raw_event_id")
    if not reference.index.isin(source.index).all():
        raise ValueError("The annotation stream is missing retained actions.")
    for column in reference.columns:
        if not reference[column].eq(source.loc[reference.index, column]).fillna(False).all():
            raise ValueError(f"Retained actions disagree with annotations: {column}.")
    quarantine = annotations.get("quarantined", pd.Series(False, index=annotations.index)).fillna(False)
    if not quarantine.isin([True, False]).all():
        raise ValueError("Quarter quarantine indicators must be Boolean.")
    if annotations.loc[quarantine, "raw_event_id"].isin(events.event_id).any():
        raise ValueError("Quarantined actions leaked into the event table.")

    # A goal is linked to the preceding retained action by strict display order,
    # within the same match and quarter. Tied timestamps do not reorder events.
    goals = annotations.loc[annotations.description.eq("Goal"),
        ["raw_event_id", *keys, "period_seconds", "team_id", "season"]].copy()
    goals["quarantined"] = quarantine.loc[goals.index]
    goals = goals.rename(columns={"raw_event_id": "goal_event_id", "team_id": "goal_team_id",
                                  "period_seconds": "goal_seconds"})
    goals = goals.astype({"match_id": "str", "period": "int64", "display_order": "int64"})
    previous = events[["match_id", "period", "display_order", "event_id", "possession_id"]].rename(
        columns={"display_order": "action_order", "event_id": "preceding_event_id"})
    previous = previous.astype({"match_id": "str", "period": "int64", "action_order": "int64"})
    linked = pd.merge_asof(goals.sort_values("display_order"), previous.sort_values("action_order"),
        by=["match_id", "period"], left_on="display_order", right_on="action_order",
        direction="backward", allow_exact_matches=False)
    linked = linked.merge(possessions[["possession_id", "team_id", "end_reason", "closed_at_seconds"]]
        .rename(columns={"team_id": "possession_team_id"}), on="possession_id", how="left", validate="many_to_one")
    active = ~linked.quarantined
    counts = linked.loc[active].possession_id.value_counts()
    linked["status"] = "verified"
    # Apply specific diagnostics before missing-link/quarantine precedence.
    linked.loc[~linked.goal_team_id.eq(linked.possession_team_id).fillna(False), "status"] = "goal_team_mismatch"
    linked.loc[linked.goal_team_id.isna(), "status"] = "missing_goal_team"
    linked.loc[~linked.goal_seconds.eq(linked.closed_at_seconds).fillna(False), "status"] = "goal_time_mismatch"
    linked.loc[~linked.end_reason.eq("Goal").fillna(False), "status"] = "endpoint_not_goal"
    linked.loc[linked.possession_id.map(counts).gt(1).fillna(False), "status"] = "multiple_goal_annotations"
    linked.loc[linked.possession_id.isna(), "status"] = "no_preceding_action"
    linked.loc[linked.quarantined, "status"] = "quarantined_quarter"
    linked = linked.sort_values(["season", "match_id", "period", "display_order"]).reset_index(drop=True)

    audit = shot.possession_audit.copy().set_index("possession_id", drop=False)
    audit["shot_target"] = audit.target
    audit["shot_status"] = audit.status
    audit["shot_eligible_prefix_count"] = audit.eligible_prefix_count
    audit["season"] = audit.index.map(events.groupby("possession_id").season.first()).astype("Int64")
    audit["goal_annotation_count"] = audit.index.map(counts).fillna(0).astype(int)
    verified = linked.loc[linked.status.eq("verified")]
    audit["verified_goal_count"] = audit.index.map(verified.possession_id.value_counts()).fillna(0).astype(int)
    invalid = linked.loc[~linked.status.isin(["verified", "quarantined_quarter"]) & linked.possession_id.notna()]
    reasons = invalid.groupby("possession_id").status.agg(lambda values: ";".join(sorted(set(values))))
    audit["goal_review_reasons"] = reasons.reindex(audit.index).astype("string").fillna("")
    missing = audit.end_reason.eq("Goal") & audit.goal_annotation_count.eq(0)
    audit.loc[missing, "goal_review_reasons"] = "missing_goal_annotation"
    # A verified goal must be unique and coincide with a same-team Goal endpoint.
    unexplained = audit.end_reason.eq("Goal") & audit.verified_goal_count.ne(1) & audit.goal_review_reasons.eq("")
    audit.loc[unexplained, "goal_review_reasons"] = "unverified_goal_endpoint"
    audit.loc[audit.shot_status.eq("eligible") & audit.goal_review_reasons.ne(""), "status"] = "goal_attribution_review"
    audit["target"] = audit.verified_goal_count.eq(1).astype("Int64")
    audit.loc[audit.status.ne("eligible"), "target"] = pd.NA
    audit.loc[audit.status.ne("eligible"), "eligible_prefix_count"] = 0
    eligible = audit.loc[audit.status.eq("eligible")]
    examples = shot.examples.loc[shot.examples.possession_id.isin(eligible.index)].copy()
    examples["target"] = examples.possession_id.map(eligible.target).astype(int)
    # Source sample weights remain 1 / pre-shot prefix count, so each possession
    # contributes total weight one, regardless of its length or eventual outcome.
    if not examples.empty and not examples.groupby("possession_id").sample_weight.sum().sub(1).abs().lt(1e-10).all():
        raise ValueError("Goal examples no longer preserve possession weighting.")
    linked["possession_status"] = linked.possession_id.map(audit.status)
    linked["included_positive"] = linked.status.eq("verified") & linked.possession_status.eq("eligible")
    summary = {
        "goal_label_version": GOAL_LABEL_VERSION, "shot_label_version": SHOT_LABEL_VERSION,
        "target_definition": "The same team scores a recorded goal before the inferred possession ends.",
        "input_cutoff": "Strictly before the first explicit shot; all retained actions if there is no shot.",
        "feature_columns": FEATURE_COLUMNS,
        "weighting": "Each eligible possession contributes total weight one across its pre-shot prefixes.",
        "total_possessions": len(audit), "included_possessions": len(eligible),
        "excluded_possessions": int(audit.status.ne("eligible").sum()), "examples": len(examples),
        "positive_possessions": int(eligible.target.eq(1).sum()),
        "negative_possessions": int(eligible.target.eq(0).sum()),
        "shot_eligible_possessions": int(audit.shot_status.eq("eligible").sum()),
        "additional_goal_exclusions": int((audit.shot_status.eq("eligible") & audit.status.ne("eligible")).sum()),
        "raw_goal_annotations": len(linked), "quarantined_goal_annotations": int(linked.quarantined.sum()),
        "verified_goal_annotations": len(verified),
        "unassigned_goal_annotations": int(linked.status.eq("no_preceding_action").sum()),
        "positive_possessions_with_multiple_shots": int((eligible.target.eq(1) & eligible.explicit_shot_count.gt(1)).sum()),
        "status_counts": {str(k): int(v) for k, v in audit.status.value_counts().items()},
        "goal_annotation_status_counts": {str(k): int(v) for k, v in linked.status.value_counts().items()},
    }
    return GoalLabelSet(examples.reset_index(drop=True), audit.reset_index(drop=True), linked, summary)


def label_coverage(dataset: GoalLabelSet) -> pd.DataFrame:
    """Count possessions separately from the correlated prefix examples."""
    rows = []
    for season, group in dataset.possession_audit.groupby("season", sort=True):
        eligible = group.loc[group.status.eq("eligible")]
        goals = dataset.goal_annotation_audit.loc[dataset.goal_annotation_audit.season.eq(season)]
        rows.append({
            "season": int(season), "matches": group.match_id.nunique(),
            "total_possessions": len(group), "eligible_possessions": len(eligible),
            "goal_possessions": int(eligible.target.eq(1).sum()),
            "no_goal_possessions": int(eligible.target.eq(0).sum()),
            "excluded_possessions": len(group) - len(eligible),
            "prefix_examples": int(eligible.eligible_prefix_count.sum()),
            "goal_possession_rate": float(eligible.target.mean()) if len(eligible) else None,
            "raw_goals": len(goals), "quarantined_goals": int(goals.quarantined.sum()),
            "verified_goals": int(goals.status.eq("verified").sum()),
            "multi_shot_goal_possessions": int((eligible.target.eq(1) & eligible.explicit_shot_count.gt(1)).sum()),
        })
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_goal_labels(dataset: GoalLabelSet, directory: Path, provenance: dict, *, overwrite=False) -> None:
    """Save auditable labels with hashes; write the completion record last."""
    tables = {"examples": dataset.examples, "possession_audit": dataset.possession_audit,
              "goal_annotation_audit": dataset.goal_annotation_audit, "coverage": label_coverage(dataset)}
    outputs = [directory / f"{name}.parquet" for name in tables] + [directory / "summary.json"]
    if not overwrite and any(path.exists() for path in outputs):
        raise ValueError("Goal labels already exist; use --overwrite to regenerate them.")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").unlink(missing_ok=True)
    for name, table in tables.items():
        table.to_parquet(directory / f"{name}.parquet", index=False)
    summary = {**dataset.summary, **provenance,
        "artifact_sha256": {name: _sha256(directory / f"{name}.parquet") for name in tables}}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run_development_audit(root: Path, *, overwrite=False) -> GoalLabelSet:
    """Reuse verified development data and shot exclusions; never load 2026."""
    from afl_opposition.baseline import load_preprocessed_seasons

    directory = root / "artifacts/goal-labels-2023-2025"
    if not overwrite and directory.exists() and any(directory.iterdir()):
        raise ValueError("Goal labels already exist; use --overwrite to regenerate them.")
    events, possessions, provenance = load_preprocessed_seasons(root, seasons=DEVELOPMENT_SEASONS)
    annotation_tables = []
    for season in DEVELOPMENT_SEASONS:
        path = root / f"data/processed/{season}/annotations.parquet"
        table = pd.read_parquet(path)
        expected = provenance["sources"][str(season)]["processed_file_sha256"]["annotations.parquet"]
        if _sha256(path) != expected:
            raise ValueError("Annotations changed after provenance verification.")
        if "quarantined" not in table:
            table["quarantined"] = False  # Original 2023 preprocessing had no quarantined quarters.
        annotation_tables.append(table)
    dataset = build_goal_examples(events, possessions, pd.concat(annotation_tables, ignore_index=True))
    provenance.update({
        "seasons": list(DEVELOPMENT_SEASONS), "scope": "Development label audit only; no fitting or test evaluation.",
        "code_sha256": {name: _sha256(root / "src/afl_opposition" / name)
                        for name in ("goal_labels.py", "examples.py", "preprocess.py", "action_rules.py", "baseline.py")},
    })
    save_goal_labels(dataset, directory, provenance, overwrite=overwrite)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate the development labels")
    args = parser.parse_args()
    try:
        dataset = run_development_audit(Path.cwd(), overwrite=args.overwrite)
        print(label_coverage(dataset).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(json.dumps(dataset.summary, indent=2))
        print("Saved labels: artifacts/goal-labels-2023-2025")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Goal labelling failed: {error}\n")


if __name__ == "__main__":
    main()
