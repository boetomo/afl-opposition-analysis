"""Train and evaluate a current-state logistic regression development baseline.

Usage: python -m afl_opposition.baseline
Use --season 2023 to reproduce the initial within-season experiment.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import platform
import warnings

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from afl_opposition.action_rules import RULE_VERSION
from afl_opposition.data import verify_file
from afl_opposition.examples import build_examples, chronological_split, season_split
from afl_opposition.evaluation import confusion_report
from afl_opposition.preprocess import FEATURE_COLUMNS

NUMERIC_FEATURES = FEATURE_COLUMNS[1:]


@dataclass
class BaselineResult:
    pipeline: Pipeline
    examples: pd.DataFrame
    possession_audit: pd.DataFrame
    matches: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    confusion: pd.DataFrame
    calibration: pd.DataFrame
    coverage: pd.DataFrame
    coefficients: pd.DataFrame
    summary: dict


def make_pipeline(*, C=1.0, tol=1e-4, max_iter=1000) -> Pipeline:
    """Current action plus four numeric features; no lagged action columns."""
    transform = ColumnTransformer([
        ("action", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["action_type"]),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True)
    return Pipeline([
        ("features", transform),
        ("classifier", LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=tol, l1_ratio=0.0)),
    ])


def _weighted_mean(values, weights) -> float:
    return float((values * weights).sum() / weights.sum())


def _evaluate(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, calibration_rows = [], []
    for split, group in predictions.groupby("split", sort=True):
        y, weight = group.target, group.sample_weight
        for model, probability in [("constant", "constant_probability"), ("logistic", "shot_probability")]:
            p = group[probability]
            rows.append(dict(
                split=split, model=model, examples=len(group), possessions=group.possession_id.nunique(),
                observed_shot_rate=_weighted_mean(y, weight),
                average_precision=float(average_precision_score(y, p, sample_weight=weight)),
                log_loss=float(log_loss(y, p, sample_weight=weight, labels=[0, 1])),
                brier_score=float(brier_score_loss(y, p, sample_weight=weight)),
            ))
            bins = (p * 10).astype(int).clip(upper=9)
            for index in sorted(bins.unique()):
                mask = bins.eq(index)
                calibration_rows.append(dict(
                    split=split, model=model, bin_lower=index / 10, bin_upper=(index + 1) / 10,
                    examples=int(mask.sum()), weight_sum=float(weight.loc[mask].sum()),
                    mean_prediction=_weighted_mean(p.loc[mask], weight.loc[mask]),
                    observed_shot_rate=_weighted_mean(y.loc[mask], weight.loc[mask]),
                ))
    return pd.DataFrame(rows), pd.DataFrame(calibration_rows)


def train_baseline(events: pd.DataFrame, possessions: pd.DataFrame, *, train_fraction=0.7,
                   train_seasons=None, validation_season=2025, test_season=2026,
                   decision_threshold=0.5) -> BaselineResult:
    """Fit on training examples and evaluate development partitions only.

    With explicit season assignments, test events are excluded before labelling,
    fitting and scoring. The final test is reserved until model selection ends.
    Omit train_seasons to reproduce the original within-season experiment.
    """
    if train_seasons is None:
        matches = chronological_split(events, train_fraction)
    else:
        matches = season_split(events, train_seasons=train_seasons,
                               validation_season=validation_season, test_season=test_season)
        development_ids = matches.loc[matches.split.ne("test"), "match_id"]
        events = events.loc[events.match_id.isin(development_ids)]
        possessions = possessions.loc[possessions.match_id.isin(development_ids)]
    dataset = build_examples(events, possessions)
    split_lookup = matches.set_index("match_id").split
    examples = dataset.examples.copy()
    examples["split"] = examples.match_id.map(split_lookup)
    train = examples.loc[examples.split.eq("train")]
    validation = examples.loc[examples.split.eq("validation")]
    if train.target.nunique() != 2 or validation.target.nunique() != 2:
        raise ValueError("Both development partitions must contain positive and negative examples.")
    pipeline = make_pipeline()
    # Failure to converge is a failed run rather than a silently accepted model.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(train[FEATURE_COLUMNS], train.target, classifier__sample_weight=train.sample_weight.to_numpy(dtype=float))
    constant = _weighted_mean(train.target, train.sample_weight)
    prediction_columns = ["event_id", "possession_id", "match_id", "period", "event_index",
                          "display_order", "date", "team_id", "split", "target", "sample_weight"]
    predictions = examples[prediction_columns].copy()
    predictions["shot_probability"] = pipeline.predict_proba(examples[FEATURE_COLUMNS])[:, 1]
    predictions["constant_probability"] = constant
    metrics, calibration = _evaluate(predictions)
    confusion = confusion_report(predictions, threshold=decision_threshold)
    threshold_metrics = confusion.loc[confusion.weighting.eq("possession_weighted"),
        ["split", "model", "threshold", "accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy"]]
    metrics = metrics.merge(threshold_metrics, on=["split", "model"], validate="one_to_one")
    audit = dataset.possession_audit.copy()
    audit["split"] = audit.match_id.map(split_lookup)
    coverage = audit.groupby(["split", "team_id", "status"]).size().rename("possessions").reset_index()
    coefficients = pd.DataFrame({
        "feature": pipeline.named_steps["features"].get_feature_names_out(),
        "coefficient": pipeline.named_steps["classifier"].coef_[0],
    })
    summary = {
        **dataset.summary, "preprocessing_rule_version": RULE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "model": "LogisticRegression", "C": 1.0, "solver": "lbfgs", "max_iter": 1000,
        "training_loss": "Possession-weighted binary cross-entropy plus L2 regularisation; intercept unpenalised.",
        "training_objective": "sum(s_i * BCE(y_i, p_i))/S + ||beta||^2/(2*C*S), where S=sum(s_i)",
        "decision_threshold": float(decision_threshold),
        "decision_rule": "Predict shot when probability >= threshold; threshold is specified, not optimised.",
        "weighting": "Each eligible possession contributes total sample weight 1, divided across its prefixes.",
        "requested_train_fraction": train_fraction if train_seasons is None else None,
        "season_assignments": None if train_seasons is None else {
            "train": list(train_seasons), "validation": [validation_season], "test_reserved": [test_season]},
        "validation_start_date": matches.loc[matches.split.eq("validation"), "date"].min().date().isoformat(),
        "match_counts": {str(k): int(v) for k, v in matches.split.value_counts().items()},
        "constant_probability": constant,
        "encoded_feature_count": len(coefficients),
        "intercept": float(pipeline.named_steps["classifier"].intercept_[0]),
        "iterations": int(pipeline.named_steps["classifier"].n_iter_[0]),
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "evaluation_scope": (
            "Chronological development validation within one season; no final test evaluation or hyperparameter search."
            if train_seasons is None else
            "Whole-season development validation; test season reserved and not scored. No hyperparameter search."
        ),
    }
    return BaselineResult(pipeline, examples, audit, matches, predictions, metrics, confusion, calibration, coverage, coefficients, summary)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_shot_markers(annotations: pd.DataFrame, possessions: pd.DataFrame, *, quarantine_conflicts=False):
    """Audit auxiliary shot markers without inventing a missing shot label.

    The optional conservative policy excludes all episodes touching a raw chain
    with a contradictory marker. These are retrospective label-quality flags,
    never input features. The returned possession table is a copy.
    """
    keys = ["match_id", "period", "chain_number", "team_id", "player_id"]
    markers = annotations.loc[annotations.description.eq("Shot At Goal"), [*keys, "display_order"]]
    kicks = annotations.loc[annotations.description.isin(["Kick", "Ground Kick"]), [*keys, "display_order", "shot_at_goal"]]
    conflicts = markers.iloc[:0].copy()
    if not markers.empty:
        if markers[keys].isna().any().any():
            raise ValueError("An unidentifiable shot marker needs label review.")
        linked = pd.merge_asof(markers.sort_values("display_order"), kicks.dropna(subset=keys).sort_values("display_order"),
                               by=keys, on="display_order", direction="backward")
        conflicts = linked.loc[~linked.shot_at_goal.eq(True).fillna(False)]
        if not conflicts.empty and not quarantine_conflicts:
            raise ValueError("A retrospective shot marker has no preceding flagged kick; reconcile labels first.")
    result = possessions.copy()
    result["label_requires_review"] = False
    result["label_review_reason"] = ""
    chain_keys = ["match_id", "period", "chain_number"]
    if not conflicts.empty:
        affected = annotations.merge(conflicts[chain_keys].drop_duplicates(), on=chain_keys, how="inner")
        mask = result.possession_id.isin(affected.possession_id.dropna())
        result.loc[mask, "label_requires_review"] = True
        result.loc[mask, "label_review_reason"] = "shot_marker_without_flagged_kick"
    return result, {
        "shot_marker_policy": "exclude_touched_possessions_v1" if quarantine_conflicts else "reject",
        "verified_retrospective_shot_markers": len(markers) - len(conflicts),
        "conflicting_shot_markers": len(conflicts),
        "shot_conflict_possessions": int(result.label_requires_review.sum()),
        "shot_conflict_locations": conflicts[[*chain_keys, "display_order"]].to_dict("records"),
    }


def load_preprocessed(root: Path, season: int, *, quarantine_shot_conflicts=False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Verify snapshot provenance and the shot-marker consistency audit."""
    directory = root / "data/processed" / str(season)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "data/sources.json").read_text(encoding="utf-8"))
    source = manifest["seasons"].get(str(season))
    if source is None:
        raise ValueError("Season missing from the source manifest.")
    verify_file(root / "data/raw" / source["filename"], source)
    if summary["source_sha256"] != source["sha256"] or summary["rule_version"] != RULE_VERSION:
        raise ValueError("Preprocessing provenance differs from the expected snapshot or rule version.")
    events = pd.read_parquet(directory / "events.parquet")
    possessions = pd.read_parquet(directory / "possessions.parquet")
    annotations = pd.read_parquet(directory / "annotations.parquet")
    if len(events) != summary["event_rows"] or len(possessions) != summary["possessions"] or len(annotations) != summary["raw_rows"]:
        raise ValueError("Preprocessed table counts disagree with the summary.")
    quarantined = annotations.get("quarantined", pd.Series(False, index=annotations.index)).fillna(False)
    if int(quarantined.sum()) != summary.get("quarantined_rows", 0):
        raise ValueError("Quarantined annotation counts disagree with the summary.")
    quarantined_shots = int(annotations.loc[quarantined, "shot_at_goal"].eq(True).fillna(False).sum())
    if quarantined_shots != summary.get("quarantined_explicit_shots", 0):
        raise ValueError("Quarantined shot counts disagree with the summary.")
    if int(events.is_explicit_shot.sum()) != summary["raw_explicit_shots"] - quarantined_shots:
        raise ValueError("Explicit shot annotations were lost during preprocessing.")
    excluded_quarters = annotations.loc[quarantined, ["match_id", "period"]].drop_duplicates()
    if not events.merge(excluded_quarters, on=["match_id", "period"], how="inner").empty:
        raise ValueError("A quarantined quarter leaked into processed actions.")
    if not events.season.eq(season).all():
        raise ValueError("Processed events contain a different season.")
    possessions, shot_audit = audit_shot_markers(annotations.loc[~quarantined], possessions,
                                              quarantine_conflicts=quarantine_shot_conflicts)
    provenance = {
        "source_sha256": summary["source_sha256"], "source_filename": summary["source_filename"],
        **shot_audit,
        "time_reversal_policy": summary.get("time_reversal_policy", "reject"),
        "quarantined_rows": int(quarantined.sum()),
        "quarantined_quarters": summary.get("quarantined_quarters", []),
        "processed_file_sha256": {name: _sha256(directory / name) for name in ["events.parquet", "possessions.parquet", "annotations.parquet", "summary.json"]},
    }
    return events, possessions, provenance


def load_preprocessed_seasons(root: Path, seasons=(2023, 2024, 2025)) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Verify each pinned season before combining development tables."""
    if not seasons or len(set(seasons)) != len(seasons):
        raise ValueError("Supply a nonempty list of distinct seasons.")
    loaded = [load_preprocessed(root, season, quarantine_shot_conflicts=True) for season in seasons]
    events = pd.concat([item[0] for item in loaded], ignore_index=True)
    possessions = pd.concat([item[1] for item in loaded], ignore_index=True)
    if events.event_id.duplicated().any() or possessions.possession_id.duplicated().any():
        raise ValueError("Identifiers overlap between season snapshots.")
    provenance = {"sources": {str(season): item[2] for season, item in zip(seasons, loaded)}}
    return events, possessions, provenance


def save_baseline(result: BaselineResult, directory: Path, provenance: dict, *, overwrite=False) -> None:
    tables = {"examples": result.examples, "possession_audit": result.possession_audit,
              "match_split": result.matches, "predictions": result.predictions, "metrics": result.metrics,
              "confusion": result.confusion, "calibration": result.calibration,
              "coverage": result.coverage, "coefficients": result.coefficients}
    outputs = [directory / (name + ".parquet") for name in tables] + [directory / "model.pkl", directory / "summary.json"]
    if not overwrite and any(path.exists() for path in outputs):
        raise ValueError("Baseline files already exist; use --overwrite to regenerate them.")
    directory.mkdir(parents=True, exist_ok=True)
    # A summary is a completion marker; remove an old one before replacing outputs.
    (directory / "summary.json").unlink(missing_ok=True)
    for name, table in tables.items():
        table.to_parquet(directory / (name + ".parquet"), index=False)
    with (directory / "model.pkl").open("wb") as stream:
        pickle.dump(result.pipeline, stream)
    summary = {**result.summary, **provenance}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, help="Run the legacy within-season development split")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for confusion matrices; default 0.5")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if args.season is None:
            events, possessions, provenance = load_preprocessed_seasons(Path.cwd())
            result = train_baseline(events, possessions, train_seasons=(2023, 2024), decision_threshold=args.threshold)
            directory = Path("artifacts/logistic-regression-2023-2025")
        else:
            events, possessions, provenance = load_preprocessed(Path.cwd(), args.season)
            result = train_baseline(events, possessions, decision_threshold=args.threshold)
            directory = Path("artifacts") / f"logistic-regression-{args.season}"
        save_baseline(result, directory, provenance, overwrite=args.overwrite)
        print(result.metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print("Confusion matrices: actual rows / predicted columns, shot is positive")
        print(result.confusion[["split", "model", "weighting", "threshold", "tn", "fp", "fn", "tp"]].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"))
        print(json.dumps(result.summary, indent=2))
        print("Saved baseline:", directory.as_posix())
    except (OSError, ValueError, ConvergenceWarning) as error:
        parser.exit(1, f"Baseline failed: {error}\n")


if __name__ == "__main__":
    main()
