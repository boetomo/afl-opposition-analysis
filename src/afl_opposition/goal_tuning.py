"""Tune the goal logistic baseline on 2025, refit, then evaluate 2026.

Usage: python -m afl_opposition.goal_tuning
The goal target uses the existing audited pre-shot possession prefixes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from afl_opposition.baseline import load_preprocessed_seasons, save_baseline
from afl_opposition.examples import season_split
from afl_opposition.goal_labels import build_goal_examples, save_goal_labels
from afl_opposition.tuning import (
    COARSE_LAMBDAS, _digest, _report, refit_selected, select_labelled_regularisation,
)


def load_goal_inputs(root: Path, seasons):
    """Verify source data and shot-marker exclusions before the goal audit."""
    events, possessions, provenance = load_preprocessed_seasons(root, seasons)
    tables = []
    for season in seasons:
        path = root / f"data/processed/{season}/annotations.parquet"
        table = pd.read_parquet(path)
        expected = provenance["sources"][str(season)]["processed_file_sha256"]["annotations.parquet"]
        if _digest(path) != expected:
            raise ValueError("Annotations changed after source verification.")
        if "quarantined" not in table:
            table["quarantined"] = False
        tables.append(table)
    return events, possessions, pd.concat(tables, ignore_index=True), provenance


def select_goal_regularisation(events, possessions, annotations, *, lambdas=COARSE_LAMBDAS,
                               refine=True, progress=None):
    """Reject test data before labels are built; select by weighted 2025 BCE."""
    if set(events.season) != {2023, 2024, 2025} or set(annotations.season) != {2023, 2024, 2025}:
        raise ValueError("Goal selection requires only 2023, 2024 and 2025; test inputs are forbidden.")
    if set(possessions.match_id) != set(events.match_id):
        raise ValueError("Development event and possession matches disagree.")
    dataset = build_goal_examples(events, possessions, annotations)
    selection = select_labelled_regularisation(dataset, season_split(events), target_name="goal",
        lambdas=lambdas, refine=refine, progress=progress)
    return selection, dataset


def evaluate_goal_test(refit, events, possessions, annotations):
    """Apply the frozen goal model to audited 2026 prefixes, without fitting."""
    if refit.summary.get("prediction_target") != "goal":
        raise ValueError("Goal evaluation requires a model fitted to goal labels.")
    if set(events.season) != {2026} or set(annotations.season) != {2026}:
        raise ValueError("Goal testing requires only the 2026 season.")
    if set(possessions.match_id) != set(events.match_id):
        raise ValueError("Test event and possession matches disagree.")
    matches = events[["match_id", "date", "season"]].drop_duplicates().copy()
    if matches.match_id.duplicated().any() or matches.isna().any().any():
        raise ValueError("Test matches must each have one populated date and season.")
    matches["date"] = pd.to_datetime(matches.date, errors="raise").dt.normalize()
    if not matches.date.dt.year.eq(2026).all():
        raise ValueError("Test dates must belong to 2026.")
    matches["split"] = "test"
    dataset = build_goal_examples(events, possessions, annotations)
    examples, audit = dataset.examples.copy(), dataset.possession_audit.copy()
    examples["split"], audit["split"] = "test", "test"
    if examples.target.nunique() != 2:
        raise ValueError("Test examples must contain both goal target classes.")
    summary = {**refit.summary, "test_label_summary": dataset.summary,
        "evaluation_scope": "2026 goal benchmark after validation selection and development-data refit; season previously used for shot evaluation"}
    return _report(refit.pipeline, examples, audit, matches, refit.constant_probability, summary,
                   target_name="goal"), dataset


def run_experiment(root: Path, *, overwrite=False):
    directory = root / "artifacts/logistic-goal-tuned-2026"
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise ValueError("The goal experiment already exists; inspect it or explicitly use --overwrite.")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test/summary.json").unlink(missing_ok=True)
    print("Verifying 2023–2025 and constructing audited goal labels.", flush=True)
    events, possessions, annotations, provenance = load_goal_inputs(root, (2023, 2024, 2025))
    selection, development_labels = select_goal_regularisation(events, possessions, annotations,
        progress=lambda row: print(
            f"{row['stage']}: lambda={row['lambda_l2']:.8g}, validation BCE={row['validation_log_loss']:.8f}", flush=True))
    selection.trials.to_parquet(directory / "lambda_search.parquet", index=False)
    selection.trials.to_csv(directory / "lambda_search.csv", index=False)
    save_goal_labels(development_labels, directory / "development_labels", provenance, overwrite=overwrite)
    save_baseline(selection.baseline, directory / "selection", provenance, overwrite=overwrite)
    print("Selected goal lambda:", selection.lambda_l2, flush=True)
    refit = refit_selected(selection)
    model_payload = pickle.dumps(refit.pipeline)
    (directory / "refit_model.pkl").write_bytes(model_payload)
    source_names = ["action_rules.py", "preprocess.py", "examples.py", "goal_labels.py",
                    "evaluation.py", "baseline.py", "tuning.py", "goal_tuning.py"]
    frozen = {**refit.summary,
        "goal_label_summary": development_labels.summary,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": _digest(directory / "refit_model.pkl"),
        "lambda_search_sha256": _digest(directory / "lambda_search.parquet"),
        "source_sha256": {name: _digest(root / "src/afl_opposition" / name) for name in source_names},
        "development_provenance": provenance, "test_loaded_after_refit": True,
        "test_season_previously_used_for_shot_evaluation": True}
    frozen_path = directory / "refit.json"
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    frozen_digest = _digest(frozen_path)
    # Release the full development annotation stream before loading test inputs.
    del events, possessions, annotations
    print("Goal model refitted and frozen. Loading 2026 for evaluation.", flush=True)
    events, possessions, annotations, test_provenance = load_goal_inputs(root, (2026,))
    result, test_labels = evaluate_goal_test(refit, events, possessions, annotations)
    if (_digest(frozen_path) != frozen_digest or _digest(directory / "refit_model.pkl") != frozen["model_sha256"]
            or hashlib.sha256(pickle.dumps(refit.pipeline)).hexdigest() != frozen["model_sha256"]):
        raise ValueError("The frozen goal model changed during test evaluation.")
    if any(_digest(root / "src/afl_opposition" / name) != digest for name, digest in frozen["source_sha256"].items()):
        raise ValueError("Experiment code changed after the refit was frozen.")
    save_goal_labels(test_labels, directory / "test_labels", test_provenance, overwrite=overwrite)
    save_baseline(result, directory / "test", {**test_provenance, "refit_sha256": frozen_digest,
        "test_evaluated_at_utc": datetime.now(timezone.utc).isoformat()}, overwrite=overwrite)
    print(result.metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)
    print(result.confusion.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print("Saved goal experiment:", directory.relative_to(root).as_posix(), flush=True)
    return selection, refit, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rerun the goal experiment")
    args = parser.parse_args()
    try:
        run_experiment(Path.cwd(), overwrite=args.overwrite)
    except (OSError, ValueError, ConvergenceWarning) as error:
        parser.exit(1, f"Goal tuning failed: {error}\n")


if __name__ == "__main__":
    main()
