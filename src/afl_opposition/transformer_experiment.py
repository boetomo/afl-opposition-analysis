"""Tune Seq2Event-inspired causal Transformers for shots and goals.

Usage: python -m afl_opposition.transformer_experiment [--device auto|cpu|cuda] [--resume]
Both models are selected and refitted before loading any test inputs.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import torch

from afl_opposition.evaluation import confusion_report
from afl_opposition.goal_tuning import load_goal_inputs
from afl_opposition.preprocess import FEATURE_COLUMNS
from afl_opposition.sequence import SEED, SequenceEncoder, make_sequences, state_fingerprint
from afl_opposition.transformer import (
    BATCH_SIZE, configure_runtime, train_candidate, evaluate_bce, predict, save_model, load_model,
)
from afl_opposition.tuning import _digest, _probability_report
from afl_opposition.xgboost_baseline import build_datasets
from afl_opposition.lstm_experiment import SOURCE_NAMES as EARLIER_SOURCES

SPACE = dict(num_layers=[1, 2, 3], d_model=[32, 64, 128], num_heads=[2, 4, 8],
    ff_multiplier=[2, 4], embedding_dim=[4, 8, 16], continuous_dim=[8, 16],
    dropout=[0., .1, .25], learning_rate=[.0003, .001, .003], lambda_l2=[0., 1e-6, 1e-5, 1e-4])
STAGES = ((6, 6), (18, 2), (48, 1))
SOURCE_NAMES = (*EARLIER_SOURCES, "transformer.py", "transformer_experiment.py")
VARIANTS = ("history",)


def candidate_configs(count=24):
    """A seeded, bounded search covering every value in all nine dimensions."""
    rng = np.random.default_rng(20261013)
    columns = {name: rng.permutation(np.resize(np.asarray(values), count)).tolist() for name, values in SPACE.items()}
    candidates = [{name: values[i] for name, values in columns.items()} for i in range(count)]
    candidates[0] = dict(num_layers=1, d_model=64, num_heads=4, ff_multiplier=2,
        embedding_dim=8, continuous_dim=8, dropout=.1, learning_rate=.001, lambda_l2=1e-5)
    if len({json.dumps(c, sort_keys=True) for c in candidates}) != len(candidates):
        raise ValueError("Candidate configurations must be unique.")
    return candidates


def _json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _save_tables(directory, tables):
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(directory / f"{name}.parquet", index=False)
        if name != "predictions":
            table.to_csv(directory / f"{name}.csv", index=False)


def probability_report(examples, p, constant, target, variant, split):
    if split == "test" and set(examples.season) != {2026}:
        raise ValueError("Test evaluation requires only 2026.")
    if variant != "history":
        raise ValueError("This experiment uses the possession history variant only.")
    model_name = "seq2event_transformer"
    predictions = examples[["event_id", "possession_id", "match_id", "season", "event_index", "target", "sample_weight"]].copy()
    predictions["split"] = split
    predictions[f"{target}_probability"] = p
    predictions["constant_probability"] = constant
    metrics, calibration = _probability_report(predictions, target)
    for table in (metrics, calibration):
        table["model"] = table.model.replace({"logistic": model_name})
    confusion = confusion_report(predictions, threshold=.5, probability_columns={
        "constant": "constant_probability", model_name: f"{target}_probability"})
    if target == "goal":
        confusion = confusion.rename(columns={"predicted_shot_rate": "predicted_goal_rate"})
    rates = confusion.loc[confusion.weighting.eq("possession_weighted"),
        ["split", "model", "threshold", "accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy"]]
    metrics = metrics.merge(rates, on=["split", "model"], validate="one_to_one")
    return dict(predictions=predictions, metrics=metrics, calibration=calibration, confusion=confusion)


def select_model(examples, target, variant, directory, *, device="cpu", candidates=None,
                 stages=STAGES, progress=None):
    if set(examples.season) != {2023, 2024, 2025}:
        raise ValueError("Selection requires 2023â€“2025 only; test inputs are forbidden.")
    candidates = candidate_configs() if candidates is None else candidates
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    train_examples = examples.loc[examples.season.isin([2023, 2024])]
    validation_examples = examples.loc[examples.season.eq(2025)]
    for table in [train_examples, validation_examples]:
        if table.target.nunique() != 2:
            raise ValueError("Both classes are required in training and validation.")
    encoder = SequenceEncoder.fit(train_examples)
    train = make_sequences(train_examples, encoder).to(device)
    validation = make_sequences(validation_examples, encoder).to(device)
    active, records, curves = list(range(len(candidates))), {}, {}
    for stage, (budget, survivors) in enumerate(stages, start=1):
        ranking = []
        for trial in active:
            config = candidates[trial]
            prefix = f"{target}/{variant} stage {stage}, trial {trial+1}/{len(candidates)}"
            if progress:
                progress(prefix + f": layers={config['num_layers']}, width={config['d_model']}, heads={config['num_heads']}, "
                         f"embedding={config['embedding_dim']}, dropout={config['dropout']}")
            def epoch_progress(row):
                if progress and (row["epoch"] % 4 == 0 or row["epoch"] == budget):
                    progress(prefix + f", epoch {row['epoch']}: validation BCE={row['validation_bce']:.6f}")
            model, history, best_epoch = train_candidate(train, validation, config,
                len(encoder.categories) + 2, epochs=budget, checkpoint=checkpoints / f"trial-{trial:02d}.pt",
                progress=epoch_progress)
            loss = evaluate_bce(model, validation, BATCH_SIZE)
            parameters = sum(p.numel() for p in model.parameters())
            records[trial] = dict(trial=trial, **config, stage=stage, epoch_budget=budget,
                attempted_epochs=len(history), best_epoch=best_epoch, best_at_budget=best_epoch == budget,
                validation_bce=loss, parameters=parameters)
            curves[trial] = pd.DataFrame(history).assign(trial=trial)
            ranking.append((loss, parameters, trial))
            _save_tables(directory, {"search": pd.DataFrame(records.values()).sort_values("trial"),
                "learning_curves": pd.concat(curves.values(), ignore_index=True)})
            if progress:
                progress(prefix + f" finished: best epoch={best_epoch}, BCE={loss:.6f}")
            del model
        active = [trial for _, _, trial in sorted(ranking)[:survivors]]
    # Includes all candidates' observed checkpoints, with no use of test results.
    winner = min(records.values(), key=lambda r: (r["validation_bce"], r["parameters"], r["trial"]))
    selected_trial = winner["trial"]
    model, _, _ = train_candidate(train, validation, candidates[selected_trial],
        len(encoder.categories) + 2, epochs=winner["attempted_epochs"],
        checkpoint=checkpoints / f"trial-{selected_trial:02d}.pt")
    trials = pd.DataFrame(records.values()).sort_values("trial")
    trials["selected"] = trials.trial.eq(selected_trial)
    _save_tables(directory, {"search": trials, "learning_curves": pd.concat(curves.values(), ignore_index=True)})
    constant = float(train.target.double().mean().item())
    tables = probability_report(validation_examples, predict(model, validation, BATCH_SIZE), constant, target, variant, "validation")
    _save_tables(directory / "validation", tables)
    save_model(directory / "selection", model, encoder, candidates[selected_trial], target, constant, winner["best_epoch"])
    _json(directory / "selection.json", dict(target_name=target, variant=variant, selected=winner,
        encoder=encoder.as_dict(), training_possessions=len(train.lengths), validation_possessions=len(validation.lengths),
        sequence_length_quantiles={str(q):float(np.quantile(train.lengths, q)) for q in [0, .5, .9, .99, 1]},
        training_constant=constant, criterion="possession-weighted 2025 BCE", seed=SEED))
    return winner


def refit_model(examples, target, variant, winner, directory, *, device="cpu", progress=None):
    if set(examples.season) != {2023, 2024, 2025}:
        raise ValueError("Refitting requires only the complete development seasons.")
    encoder = SequenceEncoder.fit(examples)
    data = make_sequences(examples, encoder).to(device)
    config = {name:winner[name] for name in SPACE}
    model, history, epochs = train_candidate(data, None, config, len(encoder.categories) + 2,
        epochs=winner["best_epoch"], checkpoint=directory / "refit-training.pt",
        progress=(lambda r: progress(f"{target}/{variant} refit epoch {r['epoch']}/{winner['best_epoch']}: BCE={r['training_bce']:.6f}") if progress else None))
    constant = float(data.target.double().mean().item())
    save_model(directory / "refit", model, encoder, config, target, constant, epochs)
    _save_tables(directory, {"refit_learning_curve": pd.DataFrame(history)})
    return dict(target_name=target, variant=variant, selected=winner, epochs=epochs, config=config,
                refit_possessions=len(data.lengths), constant_probability=constant,
                model_fingerprint=state_fingerprint(model), frozen_at_utc=datetime.now(timezone.utc).isoformat())


def compare_previous(root, target, tables):
    previous = root / "artifacts/lstm-2026" / target
    old_predictions = pd.read_parquet(previous / "history/test/predictions.parquet")
    columns = ["event_id", "possession_id", "target", "sample_weight", "constant_probability"]
    for variant, table in tables.items():
        try:
            assert_frame_equal(old_predictions[columns].sort_values("event_id").reset_index(drop=True),
                table["predictions"][columns].sort_values("event_id").reset_index(drop=True),
                check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        except AssertionError as error:
            raise ValueError(f"{target}/{variant} test population differs from earlier baselines.") from error
    results = {}
    for name in ["metrics", "confusion", "calibration"]:
        old = pd.read_parquet(previous / "comparison" / f"{name}.parquet")
        results[name] = pd.concat([old, *[t[name].loc[t[name].model.ne("constant")] for t in tables.values()]], ignore_index=True)
    return results


def run_experiment(root, *, device="auto", resume=False, candidates=None, stages=STAGES):
    directory = root / "artifacts/transformer-2026"
    device = configure_runtime(device)
    candidates = candidate_configs() if candidates is None else candidates
    source_hashes = {name:_digest(root / "src/afl_opposition" / name) for name in SOURCE_NAMES}
    plan = dict(candidates=candidates, stages=[list(s) for s in stages], batch_size=BATCH_SIZE,
        seed=SEED, patience=6, source_sha256=source_hashes, torch_version=str(torch.__version__),
        device=device, train_seasons=[2023, 2024], validation_season=2025, refit_seasons=[2023, 2024, 2025],
        test_season=2026, variants=list(VARIANTS), targets=["shot", "goal"], threshold=.5,
        objective="mean of per-possession mean BCE + lambda_l2 * sum of squared matrix weights",
        optimiser="Adam with zero weight_decay; explicit L2", clipping_norm=1.,
        deterministic_algorithms=True, cudnn_enabled=False, tf32=False, python_version=platform.python_version(),
        attention_backend="math", mha_fastpath=False, precision="float32", positional_encoding="sinusoidal",
        layer_norm="pre", activation="gelu", candidate_seed=20261013,
        reference="https://doi.org/10.1145/3534678.3539138", paper_reproduction=False)
    if directory.exists() and any(directory.iterdir()):
        if not resume:
            raise ValueError("The Transformer experiment already exists; use --resume to continue without replacing it.")
        existing = json.loads((directory / "protocol.json").read_text())
        if existing != plan:
            raise ValueError("Source, environment or search protocol changed; cannot resume this experiment.")
        if (directory / "summary.json").exists():
            completed = json.loads((directory / "summary.json").read_text())
            for name, expected in completed["artifact_sha256"].items():
                if _digest(directory / name) != expected:
                    raise ValueError("Completed experiment artifacts changed.")
            print("Transformer experiment already complete and verified; no training or test scoring repeated.")
            return completed
    for target in plan["targets"]:
        for sub, files in [("history/test", ["predictions"]), ("comparison", ["metrics", "confusion", "calibration"])]:
            if any(not (root / "artifacts/lstm-2026" / target / sub / f"{name}.parquet").is_file() for name in files):
                raise ValueError("Complete the LSTM comparison before the Transformer experiment.")
    prior = root / "artifacts/lstm-2026"
    prior_summary = json.loads((prior / "summary.json").read_text())
    for name, expected in prior_summary["artifact_sha256"].items():
        if _digest(prior / name) != expected:
            raise ValueError("Earlier LSTM artifacts changed; review the baseline before comparing.")
    directory.mkdir(parents=True, exist_ok=True)
    _json(directory / "protocol.json", plan)
    print(f"Transformer runtime: {device}, PyTorch {torch.__version__}", flush=True)
    print("Verifying 2023â€“2025 and building both targets.", flush=True)
    events, possessions, annotations, provenance = load_goal_inputs(root, (2023, 2024, 2025))
    if (directory / "development_provenance.json").exists():
        if json.loads((directory / "development_provenance.json").read_text()) != provenance:
            raise ValueError("Development provenance changed since the search began.")
    _json(directory / "development_provenance.json", provenance)
    datasets = build_datasets(events, possessions, annotations, development=True)
    del events, possessions, annotations
    frozen_hashes, label_summaries = {}, {}
    for target, dataset in datasets.items():
        label_summaries[target] = {"development":dataset.summary}
        for variant in VARIANTS:
            current = directory / target / variant
            current.mkdir(parents=True, exist_ok=True)
            if not (current / "refit.json").exists():
                if (current / "selection.json").exists():
                    winner = json.loads((current / "selection.json").read_text())["selected"]
                else:
                    winner = select_model(dataset.examples, target, variant, current, device=device,
                        candidates=candidates, stages=stages, progress=lambda s:print(s, flush=True))
                frozen = refit_model(dataset.examples, target, variant, winner, current, device=device,
                                     progress=lambda s:print(s, flush=True))
                frozen["source_sha256"] = source_hashes
                frozen["file_sha256"] = {p.relative_to(current).as_posix():_digest(p)
                    for p in [current / "search.parquet", current / "selection.json", *sorted((current / "refit").iterdir())]}
                _json(current / "refit.json", frozen)
            frozen = json.loads((current / "refit.json").read_text())
            for name, expected in frozen["file_sha256"].items():
                if _digest(current / name) != expected:
                    raise ValueError("Frozen refit files changed.")
            frozen_hashes[f"{target}/{variant}"] = _digest(current / "refit.json")
            print(f"{target}/{variant} refit frozen: {frozen['epochs']} epochs.", flush=True)
        dataset.possession_audit.to_parquet(directory / target / "development_possession_audit.parquet", index=False)
        if target == "goal":
            dataset.goal_annotation_audit.to_parquet(directory / target / "development_goal_annotation_audit.parquet", index=False)
    del datasets, dataset
    for name, expected in source_hashes.items():
        if _digest(root / "src/afl_opposition" / name) != expected:
            raise ValueError("Source changed during development fitting.")
    print("Both refits frozen. Loading 2026 for the shared benchmark.", flush=True)
    events, possessions, annotations, test_provenance = load_goal_inputs(root, (2026,))
    tests = build_datasets(events, possessions, annotations, development=False)
    del events, possessions, annotations
    for target, dataset in tests.items():
        reports = {}
        for variant in VARIANTS:
            current = directory / target / variant
            model, encoder, info = load_model(current / "refit", device=device)
            before = state_fingerprint(model)
            data = make_sequences(dataset.examples, encoder).to(device)
            reports[variant] = probability_report(dataset.examples, predict(model, data, BATCH_SIZE), info["constant_probability"], target, variant, "test")
            if state_fingerprint(model) != before or _digest(current / "refit.json") != frozen_hashes[f"{target}/{variant}"]:
                raise ValueError("Model changed during test evaluation.")
            _save_tables(current / "test", reports[variant])
            del model, data
        comparison = compare_previous(root, target, reports)
        _save_tables(directory / target / "comparison", comparison)
        dataset.possession_audit.to_parquet(directory / target / "test_possession_audit.parquet", index=False)
        if target == "goal":
            dataset.goal_annotation_audit.to_parquet(directory / target / "test_goal_annotation_audit.parquet", index=False)
        label_summaries[target]["test"] = dataset.summary
        print(target, comparison["metrics"].to_string(index=False), sep="\n", flush=True)
    result = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=_digest(directory / "protocol.json"), refit_sha256=frozen_hashes,
        both_models_frozen_before_test_loading=True, models_unchanged_during_evaluation=True,
        test_previously_observed_for_earlier_models=True, label_summaries=label_summaries,
        test_provenance=test_provenance, device=device,
        gpu_name=torch.cuda.get_device_name() if device == "cuda" else None,
        artifact_sha256={p.relative_to(directory).as_posix():_digest(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix != ".tmp" and p.name != "summary.json"})
    _json(directory / "summary.json", result)
    print("Saved experiment: artifacts/transformer-2026", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true", help="Continue identical saved search/checkpoints")
    args = parser.parse_args()
    try:
        run_experiment(Path.cwd(), device=args.device, resume=args.resume)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Transformer experiment failed: {error}\n")


if __name__ == "__main__":
    main()
