"""Audit, tune, refit and evaluate next-action/location prediction.

Run from the repository root:
    python -m afl_opposition.next_event_experiment --audit-only
    python -m afl_opposition.next_event_experiment [--resume] [--device auto]
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
import torch

from afl_opposition.baseline import load_preprocessed_seasons
from afl_opposition.next_events import ACTIONS, END, NUMERIC, build_next_events, summarise_by_season
from afl_opposition.next_event_model import (
    FeatureScaler, PrefixData, NextEventTransformer, joint_loss, configure,
    predict, save_model, load_model,
)
from afl_opposition.sequence import _cpu_state, _atomic_save, state_fingerprint

SEED = 2026
BATCH_SIZE = 1024
INITIAL_EPOCHS = 4
FINAL_EPOCHS = 12
ACTION_WEIGHT = 5.


def configs():
    """Six prespecified configurations; a bounded first search, not an exhaustive one."""
    common = dict(heads=4, ff_multiplier=4, dense_dim=64, learning_rate=.001, weight_decay=.0001)
    values = [
        (16, 8, 24, 1, .1), (32, 8, 24, 1, .1),
        (64, 8, 24, 1, .1), (32, 16, 48, 1, .1),
        (32, 16, 48, 2, .1), (32, 16, 48, 2, .25),
    ]
    return [{**common, "max_history": cap, "embedding_dim": emb,
             "continuous_dim": cont, "layers": layers, "dropout": dropout}
            for cap, emb, cont, layers, dropout in values]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def metrics(frame, probabilities, xy):
    labelled = frame.loc[frame.labelled].reset_index(drop=True)
    y = labelled.target_action.map({a: i for i, a in enumerate(ACTIONS)}).to_numpy(dtype=int)
    if probabilities.shape != (len(y), len(ACTIONS)) or xy.shape != (len(y), 2):
        raise ValueError("Prediction shapes do not match labelled prefixes.")
    if not np.isfinite(probabilities).all() or not np.isfinite(xy).all():
        raise ValueError("Non-finite predictions.")
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y, predicted, labels=np.arange(len(ACTIONS)))
    loc = labelled.has_location.to_numpy(dtype=bool)
    truth = labelled[["target_x", "target_y"]].to_numpy(dtype=float)
    error = xy[loc] - truth[loc]
    location_rmse = float(np.sqrt(np.mean(error ** 2))) if loc.any() else None
    # These metrics explicitly remain in the venue-normalised coordinate frame.
    ce = float(log_loss(y, probabilities, labels=np.arange(len(ACTIONS))))
    report = {
        "examples": len(y), "location_examples": int(loc.sum()), "action_log_loss": ce,
        "action_accuracy": float(np.mean(y == predicted)),
        "top3_accuracy": float(np.mean((np.argsort(probabilities, axis=1)[:, -3:] == y[:, None]).any(axis=1))),
        "macro_f1": float(f1_score(y, predicted, labels=np.arange(len(ACTIONS)), average="macro", zero_division=0)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - np.eye(len(ACTIONS))[y]) ** 2, axis=1))),
        "location_rmse_norm": location_rmse,
        "location_mean_distance_norm": float(np.linalg.norm(error, axis=1).mean()) if loc.any() else None,
        "selection_loss": ACTION_WEIGHT * ce + (location_rmse or 0.),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=np.arange(len(ACTIONS)), zero_division=0)
    per_action = pd.DataFrame(dict(action=ACTIONS, precision=precision, recall=recall, f1=f1, support=support))
    return report, pd.DataFrame(matrix, index=ACTIONS, columns=ACTIONS), per_action


class TransitionBaseline:
    """Smoothed P(next action | current action) and mean displacement on continuation."""

    def fit(self, frame):
        if not set(frame.season).issubset({2023, 2024, 2025}):
            raise ValueError("Baseline fitting must exclude the test season.")
        eligible = frame.loc[frame.labelled]
        counts = pd.crosstab(eligible.action_type, eligible.target_action).reindex(
            index=ACTIONS[:-1], columns=ACTIONS, fill_value=0).to_numpy(dtype=float)
        self.probabilities = (counts + .5) / (counts + .5).sum(axis=1, keepdims=True)
        global_counts = counts.sum(axis=0) + .5
        self.prior = global_counts / global_counts.sum()
        continuing = eligible.loc[eligible.has_location].copy()
        continuing["dx"] = continuing.target_x - continuing.x_norm
        continuing["dy"] = continuing.target_y - continuing.y_norm
        self.displacement = continuing.groupby("action_type")[["dx", "dy"]].mean().reindex(
            ACTIONS[:-1]).fillna(0.).to_numpy(dtype=float)
        return self

    def predict(self, frame, *, constant=False):
        eligible = frame.loc[frame.labelled]
        current = eligible.action_type.map({a: i for i, a in enumerate(ACTIONS)}).to_numpy(dtype=int)
        xy = eligible[["x_norm", "y_norm"]].to_numpy(dtype=float)
        if constant:
            return np.repeat(self.prior[None], len(eligible), axis=0), xy
        return self.probabilities[current], xy + self.displacement[current]

    def as_dict(self):
        return {"actions": list(ACTIONS), "smoothing": .5, "probabilities": self.probabilities.tolist(),
                "prior": self.prior.tolist(), "displacement": self.displacement.tolist()}


def train(train_data, validation_data, config, epochs, checkpoint, *, seed=SEED, batch_size=BATCH_SIZE):
    """Resume epoch checkpoints; retain validation-best state or final refit epoch."""
    configure(train_data.device, seed)
    model = NextEventTransformer(config).to(train_data.device)
    # Adam L2 weight decay, distinct from AdamW.
    optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    first, best, best_epoch, bad_epochs = 1, float("inf"), 0, 0
    history, best_state = [], None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=train_data.device, weights_only=True)
        if state["config"] != config or state["seed"] != seed or state["validation"] != (validation_data is not None):
            raise ValueError("Checkpoint belongs to a different experiment.")
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        first, best, best_epoch, bad_epochs = state["epoch"] + 1, state["best"], state["best_epoch"], state["bad_epochs"]
        history, best_state = state["history"], state["best_state"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        if train_data.device == "cuda":
            torch.cuda.set_rng_state(state["rng_cuda"].cpu())
    for epoch in range(first, epochs + 1):
        if validation_data is not None and bad_epochs >= 3:
            break
        started = time.perf_counter()
        model.train()
        total = 0.
        for ends in train_data.batches(batch_size, seed + epoch):
            actions, numeric, target, target_xy, location = train_data.batch(ends, config["max_history"])
            optimiser.zero_grad(set_to_none=True)
            logits, xy = model(actions, numeric)
            loss = joint_loss(logits, xy, target, target_xy, location, action_weight=ACTION_WEIGHT)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimiser.step()
            total += float(loss.detach()) * len(ends)
        validation_metrics = None
        if validation_data is not None:
            probabilities, xy = predict(model, validation_data, batch_size)
            validation_metrics = metrics(validation_data.frame, probabilities, xy)[0]
            score = validation_metrics["selection_loss"]
            if score < best:
                best, best_epoch, best_state, bad_epochs = score, epoch, _cpu_state(model), 0
            else:
                bad_epochs += 1
        else:
            best_epoch, best_state = epoch, _cpu_state(model)
        row = {"epoch": epoch, "training_batch_objective": total / len(train_data.ends),
               "validation": validation_metrics, "seconds": time.perf_counter() - started}
        history.append(row)
        _atomic_save(dict(config=config, seed=seed, validation=validation_data is not None,
            epoch=epoch, model=model.state_dict(), optimiser=optimiser.state_dict(),
            best=best, best_epoch=best_epoch, best_state=best_state, bad_epochs=bad_epochs,
            history=history, rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state() if train_data.device == "cuda" else None), checkpoint)
        label = f"val={validation_metrics['selection_loss']:.5f}" if validation_metrics else "refit"
        print(f"{checkpoint.stem}: epoch {epoch}/{epochs}, {label}, {row['seconds']:.1f}s", flush=True)
    if best_state is None:
        raise ValueError("No training epochs completed.")
    model.load_state_dict(best_state)
    return model.eval(), history, best_epoch


def source_hashes(root):
    names = ["next_events.py", "next_event_model.py", "next_event_experiment.py",
             "sequence.py", "baseline.py", "preprocess.py", "action_rules.py", "data.py"]
    return {name: hashlib.sha256((root / "src/afl_opposition" / name).read_bytes()).hexdigest() for name in names}


def run(root, directory, *, device="auto", resume=False, audit_only=False):
    root, directory = Path(root), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "summary.json").exists():
        raise ValueError("This completed experiment is immutable. Choose a new --output directory.")
    if (directory / "plan.json").exists() and not resume and not audit_only:
        raise ValueError("An experiment exists. Use --resume with unchanged sources and settings.")
    events, possessions, provenance = load_preprocessed_seasons(root, seasons=(2023, 2024, 2025))
    development = build_next_events(events, possessions)
    del events, possessions
    development.coverage.to_csv(directory / "development_coverage.csv", index=False)
    summarise_by_season(development.events).to_csv(directory / "development_counts.csv", index=False)
    write_json(directory / "dataset_audit.json", {**development.summary, **provenance})
    # A small, readable input->target table for inspection without training.
    sample_id = development.events.loc[
        development.events.groupby("possession_id").possession_id.transform("size").between(5, 10),
        "possession_id"].iloc[0]
    columns = ["event_id", "possession_id", "event_index", "action_type", "x_norm", "y_norm",
               "target_action", "target_x", "target_y", "has_location", "labelled", "next_event_id"]
    development.events.loc[development.events.possession_id.eq(sample_id), columns].to_csv(
        directory / "example_possession.csv", index=False)
    print(json.dumps(development.summary, indent=2), flush=True)
    if audit_only:
        return
    device = configure(device, SEED)
    plan = {"configs": configs(), "initial_epochs": INITIAL_EPOCHS, "final_epochs": FINAL_EPOCHS,
            "finalists": 2, "patience": 3, "seed": SEED, "batch_size": BATCH_SIZE,
            "selection_metric": "5 * unweighted multiclass log loss + masked location RMSE",
            "source_hashes": source_hashes(root), "provenance": provenance,
            "device": device, "torch_version": str(torch.__version__),
            "train_seasons": [2023, 2024], "validation_seasons": [2025],
            "refit_seasons": [2023, 2024, 2025], "test_seasons": [2026]}
    plan_path = directory / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Resume requires the same source hashes, data, environment and search plan.")
    write_json(plan_path, plan)
    frame = development.events
    training = frame.loc[frame.season.isin([2023, 2024])].reset_index(drop=True)
    validation = frame.loc[frame.season.eq(2025)].reset_index(drop=True)
    scaler = FeatureScaler.fit(training)
    train_data, validation_data = PrefixData(training, scaler, device), PrefixData(validation, scaler, device)
    candidate_rows = []
    for number, config in enumerate(configs()):
        model, history, epoch = train(train_data, validation_data, config, INITIAL_EPOCHS, directory / f"candidate-{number}.pt")
        # A resumed finalist may already have >4 epochs. First-stage ranking
        # must still use only its original first-stage history.
        initial_best = min(history[:INITIAL_EPOCHS],
                           key=lambda value: (value["validation"]["selection_loss"], value["epoch"]))
        score = initial_best["validation"]
        candidate_rows.append(dict(candidate=number, best_epoch=initial_best["epoch"], **score))
        write_json(directory / f"candidate-{number}-history.json", history)
        del model
    ranking = sorted(candidate_rows, key=lambda row: (row["selection_loss"], row["candidate"]))
    final_rows = []
    for row in ranking[:2]:
        number = row["candidate"]
        model, history, epoch = train(train_data, validation_data, configs()[number], FINAL_EPOCHS,
                                      directory / f"candidate-{number}.pt")
        final_rows.append(dict(candidate=number, best_epoch=epoch, **history[epoch - 1]["validation"]))
        write_json(directory / f"candidate-{number}-history.json", history)
        del model
    winner = min(final_rows, key=lambda row: (row["selection_loss"], row["candidate"]))
    config = configs()[winner["candidate"]]
    write_json(directory / "selection.json", {"initial": candidate_rows, "finalists": final_rows, "selected": winner})
    pd.DataFrame(candidate_rows).to_csv(directory / "initial_search.csv", index=False)
    pd.DataFrame(final_rows).to_csv(directory / "finalists.csv", index=False)
    # Restore validation-best weights for comparable validation diagnostics.
    selected, _, _ = train(train_data, validation_data, config, FINAL_EPOCHS,
                            directory / f"candidate-{winner['candidate']}.pt")
    baseline = TransitionBaseline().fit(training)
    results, action_tables = [], []

    def record(split, name, subset, probability, xy):
        score, matrix, per_action = metrics(subset, probability, xy)
        results.append({"split": split, "model": name, **score})
        matrix.to_csv(directory / f"{split}-{name}-confusion.csv", index_label="actual")
        per_action.insert(0, "model", name)
        per_action.insert(0, "split", split)
        action_tables.append(per_action)
        return score

    record("validation", "transformer", validation, *predict(selected, validation_data))
    for name, constant in [("frequency_persistence", True), ("transition", False)]:
        record("validation", name, validation, *baseline.predict(validation, constant=constant))
    # Freeze all choices before loading 2026. Refit scaler as part of refitting.
    del selected, train_data, validation_data
    refit_scaler = FeatureScaler.fit(frame)
    refit_data = PrefixData(frame, refit_scaler, device)
    final_model, history, _ = train(refit_data, None, config, winner["best_epoch"], directory / "refit.pt")
    write_json(directory / "refit-history.json", history)
    baseline = TransitionBaseline().fit(frame)
    write_json(directory / "baseline.json", baseline.as_dict())
    metadata = {"selected_validation": winner, "refit_epochs": winner["best_epoch"],
                "training_seasons": [2023, 2024, 2025], "seed": SEED}
    save_model(directory / "model", final_model, refit_scaler, metadata)
    frozen = state_fingerprint(final_model)
    # Reload the exported model and check its inference result before test access.
    loaded, loaded_scaler, _ = load_model(directory / "model", device)
    probe = refit_data.ends[:8]
    a, n, *_ = refit_data.batch(probe, config["max_history"])
    with torch.inference_mode():
        for expected, actual in zip(final_model(a, n), loaded(a, n)):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    del final_model, refit_data, frame, development, training, validation
    test_events, test_possessions, test_provenance = load_preprocessed_seasons(root, seasons=(2026,))
    test = build_next_events(test_events, test_possessions)
    test.coverage.to_csv(directory / "test_coverage.csv", index=False)
    test_data = PrefixData(test.events, loaded_scaler, device)
    probability, xy = predict(loaded, test_data)
    record("test", "transformer", test.events, probability, xy)
    for name, constant in [("frequency_persistence", True), ("transition", False)]:
        record("test", name, test.events, *baseline.predict(test.events, constant=constant))
    prediction_frame = test.events.loc[test.events.labelled,
        ["event_id", "possession_id", "team_id", "season", "action_type", "target_action",
         "target_x", "target_y", "has_location"]].reset_index(drop=True)
    prediction_frame["predicted_action"] = np.asarray(ACTIONS)[probability.argmax(axis=1)]
    prediction_frame[["predicted_x", "predicted_y"]] = xy
    for i, action in enumerate(ACTIONS):
        prediction_frame["p_" + action] = probability[:, i]
    prediction_frame.to_parquet(directory / "test_predictions.parquet", index=False)
    team_rows = []
    for team, indices in prediction_frame.groupby("team_id").groups.items():
        # This is a diagnostic only: no per-team tuning or identity features.
        subset = test.events.loc[test.events.labelled].iloc[np.asarray(indices)]
        score = metrics(subset, probability[np.asarray(indices)], xy[np.asarray(indices)])[0]
        team_rows.append({"team_id": team, **score})
    pd.DataFrame(team_rows).to_csv(directory / "test_by_team.csv", index=False)
    result_table = pd.DataFrame(results)
    result_table.to_csv(directory / "metrics.csv", index=False)
    pd.concat(action_tables, ignore_index=True).to_csv(directory / "per_action.csv", index=False)
    if state_fingerprint(loaded) != frozen:
        raise ValueError("Parameters changed during test evaluation.")
    write_json(directory / "summary.json", {
        "plan": plan, "selected": winner, "test_dataset": test.summary,
        "test_provenance": test_provenance, "model_fingerprint": frozen,
        "results": results,
        "limitations": [
            "2026 is an already-observed project benchmark, not a pristine holdout.",
            "Retrospective clean-episode selection excludes some difficult passages.",
            "One shared model; no team identity or opponent-specific adaptation.",
            "Prediction covers retained within-possession actions plus END, not every raw annotation.",
            "Single-seed bounded search; no claim of state-of-the-art performance.",
            "Location is a point prediction conditional on continuation, not a spatial distribution.",
        ],
    })
    print(result_table.to_string(index=False), flush=True)
    print("Saved:", directory.relative_to(root).as_posix() if directory.is_relative_to(root) else directory.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="artifacts/next-event-v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    run(Path.cwd(), Path(args.output).resolve(), device=args.device, resume=args.resume, audit_only=args.audit_only)


if __name__ == "__main__":
    main()
