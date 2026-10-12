"""Compare current-state XGBoost shot and goal classifiers with logistic baselines.

Run from the repository root: python -m afl_opposition.xgboost_baseline
Both validation searches and full refits finish before 2026 inputs are loaded.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import pickle
import platform

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
import xgboost as xgb

from afl_opposition.baseline import make_pipeline
from afl_opposition.evaluation import confusion_report
from afl_opposition.examples import build_examples, season_split
from afl_opposition.goal_labels import build_goal_examples
from afl_opposition.goal_tuning import load_goal_inputs
from afl_opposition.preprocess import FEATURE_COLUMNS
from afl_opposition.tuning import _digest, _probability_report

GRID = tuple(dict(max_depth=d, eta=e, reg_lambda=r)
             for d, e, r in product((2, 4, 6), (0.05, 0.1), (0.0, 1.0, 10.0)))
MAX_ROUNDS = 1200
PATIENCE = 60
FIXED_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                    device="cpu", nthread=4, seed=2026, max_bin=256, min_child_weight=5.0,
                    subsample=1.0, colsample_bytree=1.0, reg_alpha=0.0, gamma=0.0,
                    scale_pos_weight=1.0, verbosity=0)
SOURCE_NAMES = ("action_rules.py", "preprocess.py", "examples.py", "goal_labels.py",
                "evaluation.py", "baseline.py", "tuning.py", "goal_tuning.py", "xgboost_baseline.py")


def _mean(y, w):
    return float(np.average(y.to_numpy(dtype=float), weights=w.to_numpy(dtype=float)))


def _matrix(transformer, examples, *, labelled=True):
    values = np.asarray(transformer.transform(examples[FEATURE_COLUMNS]), dtype=np.float32)
    return xgb.DMatrix(values, feature_names=list(transformer.get_feature_names_out()), nthread=4,
                       label=examples.target.to_numpy(dtype=float) if labelled else None,
                       weight=examples.sample_weight.to_numpy(dtype=float) if labelled else None)


@dataclass
class ThreatModel:
    transformer: object
    booster: xgb.Booster
    target_name: str
    constant_probability: float
    parameters: dict

    def predict_proba(self, examples):
        p = self.booster.predict(_matrix(self.transformer, examples, labelled=False)).astype(float)
        return np.column_stack((1.0 - p, p))

    def fingerprint(self):
        return hashlib.sha256(pickle.dumps(self.transformer) + bytes(self.booster.save_raw())).hexdigest()

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / "model.ubj")
        (directory / "transformer.pkl").write_bytes(pickle.dumps(self.transformer))
        _json(directory / "model.json", dict(target_name=self.target_name,
            constant_probability=self.constant_probability, parameters=self.parameters,
            rounds=self.booster.num_boosted_rounds(),
            file_sha256={name: _digest(directory / name) for name in ("model.ubj", "transformer.pkl")}))


def load_model(directory: Path):
    """Load a local model produced by this project; pickle files must be trusted."""
    info = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    for name, expected in info["file_sha256"].items():
        if _digest(directory / name) != expected:
            raise ValueError("Saved model files do not match their recorded checksums.")
    booster = xgb.Booster(params={"nthread": 4})
    booster.load_model(directory / "model.ubj")
    transformer = pickle.loads((directory / "transformer.pkl").read_bytes())
    return ThreatModel(transformer, booster, info["target_name"], info["constant_probability"], info["parameters"])


@dataclass
class Selection:
    model: ThreatModel
    examples: pd.DataFrame
    trials: pd.DataFrame
    history: pd.DataFrame
    winner: dict
    training_weight_sum: float


def build_datasets(events, possessions, annotations, *, development):
    expected = {2023, 2024, 2025} if development else {2026}
    if set(events.season) != expected or set(annotations.season) != expected:
        raise ValueError("Unexpected seasons; test inputs are forbidden during selection.")
    if set(possessions.match_id) != set(events.match_id):
        raise ValueError("Event and possession matches disagree.")
    # Also check chronological match identity/date consistency before labelling.
    if development:
        season_split(events)
    else:
        matches = events[["match_id", "date", "season"]].drop_duplicates()
        if matches.isna().any().any() or matches.match_id.duplicated().any():
            raise ValueError("Each test match needs one populated date and season.")
        if not pd.to_datetime(matches.date, errors="raise").dt.year.eq(2026).all():
            raise ValueError("Test match dates must belong to 2026.")
    return {"shot": build_examples(events, possessions),
            "goal": build_goal_examples(events, possessions, annotations)}


def select_model(dataset, target_name, *, grid=GRID, max_rounds=MAX_ROUNDS,
                 patience=PATIENCE, progress=None):
    """Fit preprocessing on 2023–2024; use weighted 2025 BCE for every choice."""
    if target_name not in {"shot", "goal"}:
        raise ValueError("Target must be shot or goal.")
    examples = dataset.examples.copy()
    if set(examples.season) != {2023, 2024, 2025}:
        raise ValueError("Selection requires only 2023–2025; test inputs are forbidden.")
    if not grid or max_rounds < 1 or patience < 1:
        raise ValueError("Search needs candidates and positive round limits.")
    examples["split"] = np.where(examples.season.eq(2025), "validation", "train")
    train, validation = (examples.loc[examples.split.eq(s)] for s in ("train", "validation"))
    if train.target.nunique() != 2 or validation.target.nunique() != 2:
        raise ValueError("Training and validation each need both target classes.")
    transformer = make_pipeline().named_steps["features"]
    transformer.fit(train[FEATURE_COLUMNS])
    matrices = {"train": _matrix(transformer, train), "validation": _matrix(transformer, validation)}
    constant = _mean(train.target, train.sample_weight)
    rows, curves, best_key, best_model = [], [], None, None
    for trial, candidate in enumerate(grid):
        if set(candidate) != {"max_depth", "eta", "reg_lambda"}:
            raise ValueError("Candidates must specify depth, learning rate and L2 regularisation.")
        params = {**FIXED_PARAMS, **candidate, "base_score": constant}
        history = {}
        booster = xgb.train(params, matrices["train"], num_boost_round=max_rounds,
            evals=[(matrices["train"], "train"), (matrices["validation"], "validation")],
            evals_result=history, verbose_eval=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=patience, data_name="validation",
                       metric_name="logloss", maximize=False, save_best=True)])
        # save_best trims the booster: prediction uses the selected iteration, not the last attempted one.
        rounds = booster.num_boosted_rounds()
        p = booster.predict(matrices["validation"]).astype(float)
        loss = float(log_loss(validation.target, p, sample_weight=validation.sample_weight, labels=[0, 1]))
        row = dict(trial=trial, **candidate, rounds=rounds,
            attempted_rounds=len(history["validation"]["logloss"]),
            best_at_round_cap=rounds == max_rounds, validation_log_loss=loss,
            validation_average_precision=float(average_precision_score(validation.target, p, sample_weight=validation.sample_weight)),
            validation_brier_score=float(brier_score_loss(validation.target, p, sample_weight=validation.sample_weight)),
            training_log_loss=float(history["train"]["logloss"][rounds - 1]))
        rows.append(row)
        for split, values in history.items():
            curves.extend(dict(trial=trial, split=split, round=i + 1, log_loss=float(v))
                          for i, v in enumerate(values["logloss"]))
        # Exact ties prefer fewer trees, shallower trees, stronger L2, then earlier trial.
        key = (loss, rounds, candidate["max_depth"], -candidate["reg_lambda"], trial)
        if best_key is None or key < best_key:
            best_key = key
            best_model = ThreatModel(transformer, booster, target_name, constant, params)
            winner = row.copy()
        if progress:
            progress(row)
    trials = pd.DataFrame(rows)
    trials["selected"] = trials.trial.eq(winner["trial"])
    return Selection(best_model, examples, trials, pd.DataFrame(curves), winner, float(train.sample_weight.sum()))


def refit_model(selection):
    """Fit a fresh transformer and booster on all development data, with fixed rounds."""
    examples = selection.examples
    transformer = make_pipeline().named_steps["features"]
    transformer.fit(examples[FEATURE_COLUMNS])
    constant = _mean(examples.target, examples.sample_weight)
    ratio = float(examples.sample_weight.sum()) / selection.training_weight_sum
    # Native XGBoost sums weighted loss. Scale absolute penalties/child Hessian
    # limits with total weight to preserve their strength relative to mean loss.
    params = {**selection.model.parameters, "base_score": constant,
              "reg_lambda": selection.model.parameters["reg_lambda"] * ratio,
              "min_child_weight": selection.model.parameters["min_child_weight"] * ratio}
    booster = xgb.train(params, _matrix(transformer, examples),
                        num_boost_round=int(selection.winner["rounds"]), verbose_eval=False)
    return ThreatModel(transformer, booster, selection.model.target_name, constant, params)


def report(model, examples, split):
    """Score frozen probabilities; no fitting, early stopping or threshold search."""
    if split == "test" and set(examples.season) != {2026}:
        raise ValueError("Test evaluation requires only 2026.")
    predictions = examples[["event_id", "possession_id", "match_id", "season", "target", "sample_weight"]].copy()
    predictions["split"] = split
    predictions[f"{model.target_name}_probability"] = model.predict_proba(examples)[:, 1]
    predictions["constant_probability"] = model.constant_probability
    # Reuse the exact baseline metric implementation; only its model label changes.
    metrics, calibration = _probability_report(predictions, model.target_name)
    for table in (metrics, calibration):
        table["model"] = table.model.replace({"logistic": "xgboost"})
    confusion = confusion_report(predictions, threshold=0.5, probability_columns={
        "constant": "constant_probability", "xgboost": f"{model.target_name}_probability"})
    if model.target_name == "goal":
        confusion = confusion.rename(columns={"predicted_shot_rate": "predicted_goal_rate"})
    rates = confusion.loc[confusion.weighting.eq("possession_weighted"),
        ["split", "model", "accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy"]]
    metrics = metrics.merge(rates, on=["split", "model"], validate="one_to_one")
    return dict(predictions=predictions, metrics=metrics, calibration=calibration, confusion=confusion)


def compare_logistic(root, target_name, tables):
    """Require identical test event IDs, labels, weights and constant predictions."""
    name = "logistic-tuned-2026" if target_name == "shot" else "logistic-goal-tuned-2026"
    directory = root / "artifacts" / name / "test"
    old = pd.read_parquet(directory / "predictions.parquet")
    columns = ["event_id", "possession_id", "target", "sample_weight", "constant_probability"]
    try:
        assert_frame_equal(old[columns].sort_values("event_id").reset_index(drop=True),
            tables["predictions"][columns].sort_values("event_id").reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError as error:
        raise ValueError("Logistic and XGBoost test examples differ; comparison is invalid.") from error
    comparison = {}
    for key in ("metrics", "confusion", "calibration"):
        old_table = pd.read_parquet(directory / f"{key}.parquet")
        comparison[key] = pd.concat([old_table, tables[key].loc[tables[key].model.eq("xgboost")]], ignore_index=True)
    return comparison


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_tables(directory, tables):
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(directory / f"{name}.parquet", index=False)
        if name != "predictions":
            table.to_csv(directory / f"{name}.csv", index=False)


def run_experiment(root: Path, *, overwrite=False):
    directory = root / "artifacts/xgboost-2026"
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise ValueError("The XGBoost experiment already exists; inspect it or explicitly use --overwrite.")
    # Require comparison artifacts before spending time on a search.
    for name in ("logistic-tuned-2026", "logistic-goal-tuned-2026"):
        for filename in ("predictions.parquet", "metrics.parquet", "confusion.parquet", "calibration.parquet"):
            if not (root / "artifacts" / name / "test" / filename).is_file():
                raise ValueError("Run both tuned logistic experiments before the XGBoost comparison.")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").unlink(missing_ok=True)
    print("Verifying development inputs and building both targets.", flush=True)
    events, possessions, annotations, provenance = load_goal_inputs(root, (2023, 2024, 2025))
    datasets = build_datasets(events, possessions, annotations, development=True)
    del events, possessions, annotations
    models, frozen_hashes, label_summaries = {}, {}, {}
    for target_name, dataset in datasets.items():
        target_dir = directory / target_name
        target_dir.mkdir(parents=True, exist_ok=True)
        selection = select_model(dataset, target_name, progress=lambda r, t=target_name: print(
            f"{t} {r['trial']+1}/{len(GRID)}: depth={r['max_depth']}, eta={r['eta']}, "
            f"L2={r['reg_lambda']}, trees={r['rounds']}, validation BCE={r['validation_log_loss']:.8f}", flush=True))
        _save_tables(target_dir, {"search": selection.trials, "learning_curves": selection.history})
        validation = selection.examples.loc[selection.examples.split.eq("validation")]
        _save_tables(target_dir / "validation", report(selection.model, validation, "validation"))
        dataset.possession_audit.to_parquet(target_dir / "development_possession_audit.parquet", index=False)
        if target_name == "goal":
            dataset.goal_annotation_audit.to_parquet(target_dir / "development_goal_annotation_audit.parquet", index=False)
        model = refit_model(selection)
        model.save(target_dir / "refit")
        refit_weight = float(selection.examples.sample_weight.sum())
        frozen = dict(target_name=target_name, selected=selection.winner,
            selection_parameters=selection.model.parameters, refit_parameters=model.parameters,
            training_weight_sum=selection.training_weight_sum, refit_weight_sum=refit_weight,
            mean_loss_l2=selection.model.parameters["reg_lambda"] / (2 * selection.training_weight_sum),
            constant_probability=model.constant_probability, feature_columns=FEATURE_COLUMNS,
            encoded_features=list(model.transformer.get_feature_names_out()),
            rounds=model.booster.num_boosted_rounds(), frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            model_fingerprint=model.fingerprint(), development_label_summary=dataset.summary,
            development_provenance=provenance,
            file_sha256={str(p.relative_to(target_dir)).replace("\\", "/"): _digest(p)
                         for p in [target_dir / "search.parquet", *sorted((target_dir / "refit").iterdir())]},
            source_sha256={name: _digest(root / "src/afl_opposition" / name) for name in SOURCE_NAMES})
        _json(target_dir / "refit.json", frozen)
        frozen_hashes[target_name] = _digest(target_dir / "refit.json")
        models[target_name] = model
        label_summaries[target_name] = {"development": dataset.summary}
        print(f"{target_name}: refitted {model.booster.num_boosted_rounds()} trees and frozen.", flush=True)
        del selection
    del datasets, dataset, validation
    print("Both models frozen. Loading 2026 once for evaluation.", flush=True)
    events, possessions, annotations, test_provenance = load_goal_inputs(root, (2026,))
    tests = build_datasets(events, possessions, annotations, development=False)
    del events, possessions, annotations
    for target_name, dataset in tests.items():
        target_dir = directory / target_name
        tables = report(models[target_name], dataset.examples, "test")
        _save_tables(target_dir / "test", tables)
        _save_tables(target_dir / "comparison", compare_logistic(root, target_name, tables))
        dataset.possession_audit.to_parquet(target_dir / "test/possession_audit.parquet", index=False)
        if target_name == "goal":
            dataset.goal_annotation_audit.to_parquet(target_dir / "test/goal_annotation_audit.parquet", index=False)
        frozen = json.loads((target_dir / "refit.json").read_text(encoding="utf-8"))
        if frozen_hashes[target_name] != _digest(target_dir / "refit.json") or models[target_name].fingerprint() != frozen["model_fingerprint"]:
            raise ValueError("Frozen model changed during test evaluation.")
        for name, expected in frozen["file_sha256"].items():
            if _digest(target_dir / name) != expected:
                raise ValueError("Frozen model artifact changed during test evaluation.")
        for name, expected in frozen["source_sha256"].items():
            if _digest(root / "src/afl_opposition" / name) != expected:
                raise ValueError("Experiment source changed after freezing.")
        label_summaries[target_name]["test"] = dataset.summary
        print(target_name, tables["metrics"].to_string(index=False), sep="\n", flush=True)
    _json(directory / "summary.json", dict(
        completed_at_utc=datetime.now(timezone.utc).isoformat(), model="XGBoost", targets=["shot", "goal"],
        train_seasons=[2023, 2024], validation_season=2025, refit_seasons=[2023, 2024, 2025], test_season=2026,
        selection_metric="possession-weighted validation binary cross-entropy", decision_threshold=0.5,
        grid=GRID, max_rounds=MAX_ROUNDS, patience=PATIENCE, fixed_parameters=FIXED_PARAMS,
        both_models_frozen_before_test_loading=True, frozen_models_unchanged=True,
        test_previously_observed_for_logistic_models=True, label_summaries=label_summaries,
        test_provenance=test_provenance, refit_sha256=frozen_hashes,
        python_version=platform.python_version(), xgboost_version=xgb.__version__,
        artifact_sha256={p.relative_to(directory).as_posix(): _digest(p)
                         for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "summary.json"}))
    print("Saved experiment: artifacts/xgboost-2026", flush=True)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rerun both XGBoost experiments")
    args = parser.parse_args()
    try:
        run_experiment(Path.cwd(), overwrite=args.overwrite)
    except (OSError, ValueError, xgb.core.XGBoostError) as error:
        parser.exit(1, f"XGBoost experiment failed: {error}\n")


if __name__ == "__main__":
    main()
