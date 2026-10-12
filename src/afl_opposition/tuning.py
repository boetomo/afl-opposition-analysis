"""Select L2 strength on 2025, refit on 2023-2025, then evaluate 2026.

Usage: python -m afl_opposition.tuning
Lambda multiplies ||beta||^2 in weighted mean BCE + lambda * ||beta||^2.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite, log10
from pathlib import Path
import pickle
import platform
import warnings

import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from afl_opposition.action_rules import RULE_VERSION
from afl_opposition.baseline import (
    BaselineResult, _weighted_mean, load_preprocessed_seasons,
    make_pipeline, save_baseline,
)
from afl_opposition.evaluation import confusion_report
from afl_opposition.examples import build_examples, season_split
from afl_opposition.preprocess import FEATURE_COLUMNS

COARSE_LAMBDAS = (0.0, *[10.0 ** exponent for exponent in range(-8, 2)])
TOL = 1e-8
MAX_ITER = 2000
THRESHOLD = 0.5


@dataclass
class SelectionResult:
    baseline: BaselineResult
    trials: pd.DataFrame
    lambda_l2: float


@dataclass
class RefitResult:
    pipeline: Pipeline
    constant_probability: float
    summary: dict


def lambda_to_C(lambda_l2: float, weight_sum: float) -> float:
    """Preserve the coefficient of ||beta||^2 when the fitted sample changes."""
    if not isfinite(lambda_l2) or lambda_l2 < 0:
        raise ValueError("Lambda must be finite and nonnegative.")
    if not isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("The training weight sum must be finite and positive.")
    return float("inf") if lambda_l2 == 0 else 1.0 / (2 * lambda_l2 * weight_sum)


def _json_C(value):
    return "unregularised" if value == float("inf") else float(value)


def _fit(estimator, X, y, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(X, y, **kwargs)
    return estimator


def _probability_report(predictions, target_name):
    """Compute the same weighted probability metrics for shots or goals."""
    rows, bins = [], []
    rate_name = f"observed_{target_name}_rate"
    for split, group in predictions.groupby("split", sort=True):
        y, weight = group.target, group.sample_weight
        for model, column in [("constant", "constant_probability"), ("logistic", f"{target_name}_probability")]:
            p = group[column]
            rows.append(dict(split=split, model=model, examples=len(group),
                possessions=group.possession_id.nunique(),
                **{rate_name: _weighted_mean(y, weight)},
                average_precision=float(average_precision_score(y, p, sample_weight=weight)),
                log_loss=float(log_loss(y, p, sample_weight=weight, labels=[0, 1])),
                brier_score=float(brier_score_loss(y, p, sample_weight=weight))))
            indices = (p * 10).astype(int).clip(upper=9)
            for index in sorted(indices.unique()):
                mask = indices.eq(index)
                bins.append(dict(split=split, model=model, bin_lower=index / 10, bin_upper=(index + 1) / 10,
                    examples=int(mask.sum()), weight_sum=float(weight.loc[mask].sum()),
                    mean_prediction=_weighted_mean(p.loc[mask], weight.loc[mask]),
                    **{rate_name: _weighted_mean(y.loc[mask], weight.loc[mask])}))
    return pd.DataFrame(rows), pd.DataFrame(bins)


def _report(pipeline, examples, audit, matches, constant, summary, *, target_name="shot") -> BaselineResult:
    if target_name not in {"shot", "goal"}:
        raise ValueError("Unsupported prediction target.")
    columns = ["event_id", "possession_id", "match_id", "period", "event_index",
               "display_order", "date", "team_id", "split", "target", "sample_weight"]
    predictions = examples[columns].copy()
    predictions[f"{target_name}_probability"] = pipeline.predict_proba(examples[FEATURE_COLUMNS])[:, 1]
    predictions["constant_probability"] = constant
    metrics, calibration = _probability_report(predictions, target_name)
    confusion = confusion_report(predictions, threshold=THRESHOLD,
        probability_columns={"constant": "constant_probability", "logistic": f"{target_name}_probability"})
    if target_name == "goal":
        confusion = confusion.rename(columns={"predicted_shot_rate": "predicted_goal_rate"})
    rates = confusion.loc[confusion.weighting.eq("possession_weighted"),
        ["split", "model", "threshold", "accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy"]]
    metrics = metrics.merge(rates, on=["split", "model"], validate="one_to_one")
    coverage = audit.groupby(["split", "team_id", "status"]).size().rename("possessions").reset_index()
    coefficients = pd.DataFrame({
        "feature": pipeline.named_steps["features"].get_feature_names_out(),
        "coefficient": pipeline.named_steps["classifier"].coef_[0],
    })
    summary = {
        **summary, "prediction_target": target_name, "preprocessing_rule_version": RULE_VERSION,
        "feature_columns": FEATURE_COLUMNS, "encoded_feature_count": len(coefficients),
        "model": "LogisticRegression", "solver": "lbfgs", "tol": TOL, "max_iter": MAX_ITER,
        "objective": "weighted mean binary cross-entropy + lambda_l2 * ||beta||^2; intercept unpenalised",
        "decision_threshold": THRESHOLD, "threshold_selected": False,
        "constant_probability": constant,
        "match_counts": {str(k): int(v) for k, v in matches.split.value_counts().items()},
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "iterations": int(pipeline.named_steps["classifier"].n_iter_[0]),
    }
    return BaselineResult(pipeline, examples, audit, matches, predictions, metrics,
                          confusion, calibration, coverage, coefficients, summary)


def select_regularisation(events, possessions, *, lambdas=COARSE_LAMBDAS, refine=True, progress=None) -> SelectionResult:
    """Use only 2023-2025; reject test inputs before constructing examples.

    Fit one encoder/scaler on 2023-2024. Each lambda gets a fresh classifier.
    Select minimum weighted validation BCE, excluding the penalty from the score.
    Exact ties favour larger lambda. Refinement uses validation only.
    """
    if set(events.season) != {2023, 2024, 2025}:
        raise ValueError("Selection requires only 2023, 2024 and 2025; test events are forbidden.")
    if set(possessions.match_id) != set(events.match_id):
        raise ValueError("Development event and possession matches disagree.")
    return select_labelled_regularisation(build_examples(events, possessions), season_split(events),
        target_name="shot", lambdas=lambdas, refine=refine, progress=progress)


def select_labelled_regularisation(dataset, matches, *, target_name,
                                  lambdas=COARSE_LAMBDAS, refine=True, progress=None) -> SelectionResult:
    """Shared validation search after target-specific label construction.

    Both targets use the same objective, grid, refinement and fitting procedure.
    Only audited development examples are allowed; test rows are rejected.
    """
    if target_name not in {"shot", "goal"}:
        raise ValueError("Unsupported prediction target.")
    expected = {2023: "train", 2024: "train", 2025: "validation"}
    if (set(matches.season) != set(expected) or matches.match_id.duplicated().any()
            or not matches.split.eq(matches.season.map(expected)).all()
            or not pd.to_datetime(matches.date).dt.year.eq(matches.season).all()):
        raise ValueError("Selection matches must use only the declared development seasons and splits.")
    if (not set(dataset.examples.season) <= set(expected)
            or not set(dataset.examples.match_id) <= set(matches.match_id)
            or set(dataset.possession_audit.match_id) != set(matches.match_id)):
        raise ValueError("Labelled examples must belong only to development matches.")
    candidates = sorted(set(float(value) for value in lambdas))
    if not candidates:
        raise ValueError("At least one lambda candidate is required.")
    for value in candidates:
        lambda_to_C(value, 1)
    examples = dataset.examples.copy()
    examples["split"] = examples.match_id.map(matches.set_index("match_id").split)
    train, validation = [examples.loc[examples.split.eq(part)] for part in ["train", "validation"]]
    if train.target.nunique() != 2 or validation.target.nunique() != 2:
        raise ValueError("Training and validation must each contain both target classes.")
    weight_sum = float(train.sample_weight.sum())
    transformer = make_pipeline().named_steps["features"]
    X_train = transformer.fit_transform(train[FEATURE_COLUMNS])
    X_validation = transformer.transform(validation[FEATURE_COLUMNS])
    rows, fitted = [], {}

    def evaluate_candidate(value, stage):
        if value in fitted:
            return
        C = lambda_to_C(value, weight_sum)
        classifier = make_pipeline(C=C, tol=TOL, max_iter=MAX_ITER).named_steps["classifier"]
        _fit(classifier, X_train, train.target, sample_weight=train.sample_weight.to_numpy(dtype=float))
        p = classifier.predict_proba(X_validation)[:, 1]
        p_train = classifier.predict_proba(X_train)[:, 1]
        row = dict(
            lambda_l2=value, C=C, stage=stage, training_weight_sum=weight_sum,
            training_log_loss=float(log_loss(train.target, p_train, sample_weight=train.sample_weight)),
            validation_log_loss=float(log_loss(validation.target, p, sample_weight=validation.sample_weight)),
            validation_average_precision=float(average_precision_score(validation.target, p, sample_weight=validation.sample_weight)),
            validation_brier_score=float(brier_score_loss(validation.target, p, sample_weight=validation.sample_weight)),
            iterations=int(classifier.n_iter_[0]),
        )
        rows.append(row)
        fitted[value] = classifier
        if progress:
            progress(row)

    def winner():
        return min(rows, key=lambda row: (row["validation_log_loss"], -row["lambda_l2"]))["lambda_l2"]

    for value in candidates:
        evaluate_candidate(value, "coarse")
    if refine:
        positive = [v for v in candidates if v > 0]
        # One boundary expansion, specified before test inspection.
        if positive and winner() == max(positive):
            evaluate_candidate(max(positive) * 10, "boundary")
        if winner() == 0 and positive:
            for value in [min(positive) / 100, min(positive) / 10]:
                evaluate_candidate(value, "near_zero")
        elif positive and winner() == min(positive):
            evaluate_candidate(min(positive) / 10, "boundary")
        best = winner()
        if best > 0:
            for offset in [-.75, -.5, -.25, .25, .5, .75]:
                evaluate_candidate(10 ** (log10(best) + offset), "refined")
    best = winner()
    trials = pd.DataFrame(rows)
    trials["selected"] = trials.lambda_l2.eq(best)
    pipeline = Pipeline([("features", transformer), ("classifier", fitted[best])])
    audit = dataset.possession_audit.copy()
    audit["split"] = audit.match_id.map(matches.set_index("match_id").split)
    summary = {
        **dataset.summary, "lambda_l2": best, "C": _json_C(lambda_to_C(best, weight_sum)),
        "training_weight_sum": weight_sum,
        "selection_metric": "minimum possession-weighted 2025 binary cross-entropy, without L2 penalty",
        "selection_tie_break": "larger lambda on exact log-loss ties",
        "coarse_lambdas": candidates, "refinement_enabled": refine,
        "training_seasons": [2023, 2024], "validation_season": 2025,
        "test_season": 2026, "test_used_for_selection": False,
        "evaluation_scope": "2025 validation used for lambda selection; not an unbiased final test estimate",
    }
    baseline = _report(pipeline, examples, audit, matches,
                       _weighted_mean(train.target, train.sample_weight), summary, target_name=target_name)
    return SelectionResult(baseline, trials, best)


def refit_selected(selection: SelectionResult) -> RefitResult:
    """Fit a fresh full pipeline on 2023-2025 with the selected lambda fixed."""
    examples = selection.baseline.examples
    if set(examples.season) != {2023, 2024, 2025} or not examples.split.isin(["train", "validation"]).all():
        raise ValueError("Refitting requires exactly the training and validation seasons.")
    weight_sum = float(examples.sample_weight.sum())
    C = lambda_to_C(selection.lambda_l2, weight_sum)
    pipeline = make_pipeline(C=C, tol=TOL, max_iter=MAX_ITER)
    _fit(pipeline, examples[FEATURE_COLUMNS], examples.target,
         classifier__sample_weight=examples.sample_weight.to_numpy(dtype=float))
    summary = {
        "prediction_target": selection.baseline.summary.get("prediction_target", "shot"),
        "lambda_l2": selection.lambda_l2, "C": _json_C(C), "training_weight_sum": weight_sum,
        "refit_seasons": [2023, 2024, 2025], "refit_examples": len(examples),
        "refit_possessions": examples.possession_id.nunique(), "refit_matches": examples.match_id.nunique(),
        "selection_C": selection.baseline.summary["C"],
        "selection_validation_log_loss": float(selection.trials.loc[selection.trials.selected, "validation_log_loss"].iloc[0]),
        "selection_metric": selection.baseline.summary["selection_metric"],
        "test_season": 2026, "decision_threshold": THRESHOLD,
        "tol": TOL, "max_iter": MAX_ITER, "solver": "lbfgs",
        "objective": "weighted mean binary cross-entropy + lambda_l2 * ||beta||^2; intercept unpenalised",
    }
    return RefitResult(pipeline, _weighted_mean(examples.target, examples.sample_weight), summary)


def evaluate_test(refit: RefitResult, events, possessions) -> BaselineResult:
    """Score 2026 with the frozen pipeline; no fitting or model selection."""
    if refit.summary.get("prediction_target", "shot") != "shot":
        raise ValueError("Use goal evaluation for a goal-target model.")
    if set(events.season) != {2026}:
        raise ValueError("Final testing requires only the 2026 season.")
    matches = events[["match_id", "date", "season"]].drop_duplicates().copy()
    if matches.match_id.duplicated().any() or matches.isna().any().any():
        raise ValueError("Test matches must each have one populated date and season.")
    matches["date"] = pd.to_datetime(matches.date, errors="raise").dt.normalize()
    if not matches.date.dt.year.eq(2026).all():
        raise ValueError("Test dates must belong to 2026.")
    matches["split"] = "test"
    dataset = build_examples(events, possessions)
    examples, audit = dataset.examples.copy(), dataset.possession_audit.copy()
    examples["split"], audit["split"] = "test", "test"
    if examples.target.nunique() != 2:
        raise ValueError("Test examples must contain both target classes.")
    summary = {**refit.summary, "test_label_summary": dataset.summary,
               "evaluation_scope": "2026 held-out test after validation selection and full development-data refit"}
    return _report(refit.pipeline, examples, audit, matches, refit.constant_probability, summary)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(root: Path, *, overwrite=False):
    directory = root / "artifacts/logistic-tuned-2026"
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise ValueError("The experiment already exists; inspect its saved results or explicitly use --overwrite.")
    directory.mkdir(parents=True, exist_ok=True)
    # A prior completed report must not appear current if an explicit rerun fails.
    (directory / "test/summary.json").unlink(missing_ok=True)
    events, possessions, development_provenance = load_preprocessed_seasons(root)
    selection = select_regularisation(events, possessions, progress=lambda row: print(
        f"{row['stage']}: lambda={row['lambda_l2']:.8g}, validation BCE={row['validation_log_loss']:.8f}", flush=True))
    selection.trials.to_parquet(directory / "lambda_search.parquet", index=False)
    selection.trials.to_csv(directory / "lambda_search.csv", index=False)
    save_baseline(selection.baseline, directory / "selection", development_provenance, overwrite=overwrite)
    print("Selected lambda:", selection.lambda_l2, flush=True)
    refit = refit_selected(selection)
    with (directory / "refit_model.pkl").open("wb") as stream:
        pickle.dump(refit.pipeline, stream)
    source_names = ["action_rules.py", "preprocess.py", "examples.py", "evaluation.py", "baseline.py", "tuning.py"]
    frozen = {**refit.summary,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_sha256": _digest(directory / "refit_model.pkl"),
              "lambda_search_sha256": _digest(directory / "lambda_search.parquet"),
              "source_sha256": {name: _digest(root / "src/afl_opposition" / name) for name in source_names},
              "development_provenance": development_provenance,
              "test_loaded_after_refit": True}
    frozen_path = directory / "refit.json"
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    frozen_digest = _digest(frozen_path)
    print("Full pipeline refitted and frozen. Loading 2026 for final evaluation.", flush=True)
    test_events, test_possessions, test_provenance = load_preprocessed_seasons(root, (2026,))
    result = evaluate_test(refit, test_events, test_possessions)
    if _digest(frozen_path) != frozen_digest or _digest(directory / "refit_model.pkl") != frozen["model_sha256"]:
        raise ValueError("Frozen refit artifacts changed during evaluation.")
    save_baseline(result, directory / "test", {**test_provenance, "refit_sha256": frozen_digest,
        "test_evaluated_at_utc": datetime.now(timezone.utc).isoformat()}, overwrite=overwrite)
    print(result.metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"), flush=True)
    print(result.confusion.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print("Saved experiment:", directory.relative_to(root).as_posix(), flush=True)
    return selection, refit, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rerun an existing experiment")
    args = parser.parse_args()
    try:
        run_experiment(Path.cwd(), overwrite=args.overwrite)
    except (OSError, ValueError, ConvergenceWarning) as error:
        parser.exit(1, f"Tuning failed: {error}\n")


if __name__ == "__main__":
    main()
