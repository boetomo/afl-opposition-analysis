"""Tune, refit and report two distinct next-event formulations.

Run: python -m afl_opposition.event_comparison [--audit-only | --smoke] [--resume]
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from afl_opposition.quarter_events import load_quarters, make_streams, MODES, BALL_ACTIONS, TARGET_NAMES, K, CONTRACT
from afl_opposition.event_models import (
    EventEncoder, EventData, EventTransformer, class_weights, objective, configure,
    predict, save_model, load_model,
)
from afl_opposition.next_event_experiment import write_json
from afl_opposition.sequence import _cpu_state, _atomic_save, state_fingerprint

SEED = 2026


def configurations(mode):
    common = dict(mode=mode, heads=2, dense_dim=64, weight_decay=1e-4)
    # A common fixed window keeps Seq2Event candidate target rows identical.
    grid = [(40, 64, 8, 1, .1, .001), (40, 64, 16, 2, .1, .001),
            (40, 96, 16, 2, .25, .0003), (40, 128, 32, 2, .1, .0003)]
    return [dict(common, window=w if mode == "team_stream" else 0, width=d,
                 embedding_dim=e, layers=l, dropout=p, learning_rate=lr)
            for w, d, e, l, p, lr in grid]


def make_data(frame, encoder, config, device):
    return EventData(frame, encoder, config["mode"], max(config["window"], 1), device)


def batch_size(mode):
    return 512 if mode == "team_stream" else 8


def score(data, probabilities, xy, weights):
    frame = data.frame.iloc[data.scored_rows]
    y = frame.target.to_numpy(dtype=int)
    classes = K + 1 if data.mode == "team_stream" else 2 * K + 1
    if probabilities.shape != (len(y), classes) or xy.shape != (len(y), 2):
        raise ValueError("Prediction shapes do not match targets.")
    if not np.isfinite(probabilities).all() or not np.isfinite(xy).all():
        raise ValueError("Non-finite predictions.")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1., atol=1e-5):
        raise ValueError("Invalid probability distributions.")
    logp = -np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-30))
    location = frame.has_location.to_numpy(dtype=bool)
    error = xy[location] - frame[["target_x", "target_y"]].to_numpy(dtype=float)[location]
    rmse = float(np.sqrt((error ** 2).mean())) if location.any() else 0.
    w = np.asarray(weights)
    result = dict(examples=len(y), location_examples=int(location.sum()),
        selection_loss=float(5 * (logp * w[y]).mean() + rmse),
        location_rmse_norm=rmse if location.any() else None,
        location_mean_distance_norm=float(np.linalg.norm(error, axis=1).mean()) if location.any() else None)
    matrices, reports = {}, []

    def classification(name, truth, prob, names):
        if not len(truth):
            return
        prediction = prob.argmax(1)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, prediction, labels=np.arange(len(names)), zero_division=0)
        result[name + "_examples"] = len(truth)
        result[name + "_accuracy"] = float((truth == prediction).mean())
        result[name + "_log_loss"] = float(-np.log(np.maximum(prob[np.arange(len(truth)), truth], 1e-30)).mean())
        result[name + "_macro_f1"] = float(f1.mean())
        result[name + "_supported_macro_f1"] = float(f1[support > 0].mean())
        result[name + "_brier"] = float(((prob - np.eye(len(names))[truth]) ** 2).sum(1).mean())
        matrices[name] = pd.DataFrame(confusion_matrix(truth, prediction, labels=np.arange(len(names))), index=names, columns=names)
        reports.append(pd.DataFrame(dict(task=name, label=names, precision=precision, recall=recall, f1=f1, support=support)))

    if data.mode == "team_stream":
        active = y < K
        p = probabilities[active, :K]
        p = p / p.sum(1, keepdims=True)
        classification("action", y[active], p, BALL_ACTIONS)
    else:
        classification("joint", y, probabilities, TARGET_NAMES)
        active = y < 2 * K
        # These are explicitly conditional diagnostics on true ordinary actions.
        p = probabilities[active, :2 * K].reshape(-1, 2, K)
        p = p / p.sum((1, 2), keepdims=True)
        classification("team_given_action", y[active] // K, p.sum(2), ("home", "away"))
        classification("action_given_action", y[active] % K, p.sum(1), BALL_ACTIONS)
        result["boundary_rate"] = float((~active).mean())
    return result, matrices, pd.concat(reports, ignore_index=True)


class TransitionBaseline:
    """Task-specific smoothed transition frequencies and mean location changes."""
    def fit(self, data):
        if not set(data.frame.season).issubset({2023, 2024, 2025}):
            raise ValueError("Baseline fitting cannot include test data.")
        f = data.frame.iloc[data.scored_rows]
        self.mode = data.mode
        self.classes = K + 1 if data.mode == "team_stream" else 2 * K + 1
        counts = np.full((self.classes, self.classes), .5)
        np.add.at(counts, (f.code.to_numpy(dtype=int), f.target.to_numpy(dtype=int)), 1.)
        self.probabilities = counts / counts.sum(1, keepdims=True)
        self.displacement = np.zeros((self.classes, 2))
        local = f.loc[f.has_location].copy()
        local["delta_x"], local["delta_y"] = local.target_x - local.x, local.target_y - local.y
        for code, part in local.groupby("code"):
            self.displacement[int(code)] = part[["delta_x", "delta_y"]].mean().to_numpy()
        return self

    def predict(self, data):
        f = data.frame.iloc[data.scored_rows]
        current = f.code.to_numpy(dtype=int)
        return self.probabilities[current], f[["x", "y"]].to_numpy(dtype=float) + self.displacement[current]

    def as_dict(self):
        return dict(mode=self.mode, smoothing=.5, probabilities=self.probabilities.tolist(), displacement=self.displacement.tolist())


def train(data, validation, config, epochs, checkpoint, *, seed=SEED):
    """Epoch-resumable fitting; best validation epoch determines refit duration."""
    device = configure(data.device, seed)
    clubs = int(max(data.values[3].max(), data.values[4].max()).item())
    model = EventTransformer(config, clubs).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    weights = class_weights(data)
    history, best_state, best, best_epoch, stale, first = [], None, float("inf"), 0, 0, 1
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved["config"] != config or saved["validation"] != (validation is not None) or saved["seed"] != seed:
            raise ValueError("Checkpoint configuration differs.")
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        history, best_state = saved["history"], saved["best_state"]
        best, best_epoch, stale, first = saved["best"], saved["best_epoch"], saved["stale"], saved["epoch"] + 1
        torch.set_rng_state(saved["rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
    for epoch in range(first, epochs + 1):
        if validation is not None and stale >= 3:
            break
        started = time.perf_counter()
        model.train()
        total, count = 0., 0
        for inputs, target, xy, location, _ in data.batches(batch_size(data.mode), seed + epoch):
            eligible = int(target.ge(0).sum())
            if not eligible:
                continue
            optimiser.zero_grad(set_to_none=True)
            output, predicted_xy = model(*inputs)
            loss = objective(output, predicted_xy, target, xy, location, weights)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training objective.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimiser.step()
            total += float(loss.detach()) * eligible
            count += eligible
        if count != len(data.scored_rows):
            raise ValueError("An epoch did not visit each eligible target exactly once.")
        validation_score = None
        if validation is not None:
            validation_score = score(validation, *predict(model, validation, batch_size(data.mode)), weights.cpu().numpy())[0]
            loss = validation_score["selection_loss"]
            if loss < best:
                best, best_epoch, stale, best_state = loss, epoch, 0, _cpu_state(model)
            else:
                stale += 1
        else:
            best_epoch, best_state = epoch, _cpu_state(model)
        history.append(dict(epoch=epoch, training_batch_objective=total / count, validation=validation_score,
                            seconds=time.perf_counter() - started))
        _atomic_save(dict(config=config, seed=seed, validation=validation is not None, epoch=epoch,
            model=model.state_dict(), optimiser=optimiser.state_dict(), history=history, best_state=best_state,
            best=best, best_epoch=best_epoch, stale=stale, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state() if device == "cuda" else None), checkpoint)
        print(f"{data.mode}/{checkpoint.stem}: epoch {epoch}, "
              f"{('validation=' + format(validation_score['selection_loss'], '.5f')) if validation_score else 'refit'}, "
              f"{history[-1]['seconds']:.1f}s", flush=True)
    if best_state is None:
        raise ValueError("No epoch completed.")
    model.load_state_dict(best_state)
    return model.eval(), history, best_epoch, weights.cpu().numpy()


def source_hashes(root):
    names = ["quarter_events.py", "event_models.py", "event_comparison.py", "next_event_model.py",
             "next_event_experiment.py", "sequence.py", "baseline.py", "preprocess.py", "action_rules.py", "data.py"]
    return {n: hashlib.sha256((root / "src/afl_opposition" / n).read_bytes()).hexdigest() for n in names}


def run(root, directory, *, device="auto", resume=False, audit_only=False, smoke=False,
        initial_epochs=2, max_epochs=12):
    root, directory = Path(root), Path(directory)
    if (directory / "summary.json").exists():
        raise ValueError("Completed experiments are immutable. Choose a new output directory.")
    if (directory / "plan.json").exists() and not resume and not audit_only:
        raise ValueError("Use --resume to continue an unchanged experiment.")
    device = configure(device, SEED)
    development = load_quarters(root)
    observed = development.events
    write_json(directory / "development_audit.json", development.audit)
    if smoke:
        # Whole matches; small engineering check, never presented as model evidence.
        chosen = observed.groupby("season", sort=True).match_id.unique().map(lambda x: list(x[:2]))
        observed = observed.loc[observed.match_id.isin(sum(chosen.tolist(), []))].copy()
        initial_epochs = max_epochs = 1
    plan = dict(contract=CONTRACT, modes=list(MODES), seed=SEED, device=device, torch=str(torch.__version__),
        sources=source_hashes(root), provenance=development.provenance, smoke=smoke,
        split=dict(train=[2023, 2024], validation=[2025], refit=[2023, 2024, 2025], test=[2026]),
        initial_epochs=initial_epochs, max_epochs=max_epochs,
        configurations={m: configurations(m)[:1] if smoke else configurations(m) for m in MODES},
        note="Different targets, losses and data streams: no cross-task winner or causal ablation claim.")
    if (directory / "plan.json").exists() and json.loads((directory / "plan.json").read_text()) != plan:
        raise ValueError("Source, data, runtime or settings changed since the frozen plan.")
    if not audit_only:
        write_json(directory / "plan.json", plan)
    results, winners, baselines, frozen_weights = [], {}, {}, {}

    def record(mode, split, model_name, data, p, xy, weights):
        metrics, matrices, report = score(data, p, xy, weights)
        results.append(dict(mode=mode, split=split, model=model_name, **metrics))
        dest = directory / mode
        dest.mkdir(parents=True, exist_ok=True)
        for name, matrix in matrices.items():
            matrix.to_csv(dest / f"{split}-{model_name}-{name}-confusion.csv", index_label="actual")
        report.to_csv(dest / f"{split}-{model_name}-per-class.csv", index=False)
        return metrics

    for mode in MODES:
        frame = make_streams(observed, mode)
        dest = directory / mode
        write_json(dest / "coverage.json", dict(tokens=len(frame), streams=frame.stream_id.nunique(),
            labelled_pairs=int(frame.labelled.sum()), location_pairs=int(frame.has_location.sum()),
            max_stream_length=int(frame.groupby("stream_id").size().max())))
        frame.iloc[:min(600, len(frame))].to_csv(dest / "example_stream.csv", index=False)
        if audit_only:
            print(mode, json.loads((dest / "coverage.json").read_text()), flush=True)
            continue
        fitting = frame.loc[frame.season.isin([2023, 2024])].reset_index(drop=True)
        validation = frame.loc[frame.season.eq(2025)].reset_index(drop=True)
        encoder = EventEncoder.fit(fitting)
        candidates = []
        for index, config in enumerate(plan["configurations"][mode]):
            data, val = make_data(fitting, encoder, config, device), make_data(validation, encoder, config, device)
            model, history, best_epoch, weights = train(data, val, config, initial_epochs, dest / f"candidate-{index}.pt")
            initial = min(r["validation"]["selection_loss"] for r in history[:initial_epochs])
            candidates.append(dict(index=index, config=config, initial_loss=initial))
            write_json(dest / f"candidate-{index}-history.json", history)
        finalists = sorted(candidates, key=lambda c: (c["initial_loss"], c["index"]))[:2]
        for candidate in finalists:
            config, index = candidate["config"], candidate["index"]
            data, val = make_data(fitting, encoder, config, device), make_data(validation, encoder, config, device)
            model, history, best_epoch, weights = train(data, val, config, max_epochs, dest / f"candidate-{index}.pt")
            candidate.update(best_epoch=best_epoch, validation_loss=min(r["validation"]["selection_loss"] for r in history))
            write_json(dest / f"candidate-{index}-history.json", history)
        winner = min(finalists, key=lambda c: (c["validation_loss"], c["index"]))
        winners[mode] = winner
        write_json(dest / "selection.json", dict(candidates=candidates, winner=winner))
        config = winner["config"]
        data, val = make_data(fitting, encoder, config, device), make_data(validation, encoder, config, device)
        model, _, _, weights = train(data, val, config, max_epochs, dest / f"candidate-{winner['index']}.pt")
        record(mode, "validation", "transformer", val, *predict(model, val, batch_size(mode)), weights)
        baseline = TransitionBaseline().fit(data)
        record(mode, "validation", "transition", val, *baseline.predict(val), weights)
        if smoke:
            save_model(dest / "smoke-model", model, encoder, dict(scope="Engineering smoke check only; not a full fit."))
            continue
        encoder = EventEncoder.fit(frame)
        refit_data = make_data(frame, encoder, config, device)
        model, history, _, weights = train(refit_data, None, config, winner["best_epoch"], dest / "refit.pt")
        write_json(dest / "refit-history.json", history)
        save_model(dest / "model", model, encoder, dict(refit_seasons=[2023, 2024, 2025],
            selected=winner, class_weights=weights.tolist()))
        reloaded, _, info = load_model(dest / "model", device)
        probe = next(refit_data.batches(1))[0]
        with torch.inference_mode():
            for expected, actual in zip(model(*probe), reloaded(*probe)):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        frozen_weights[mode] = info["fingerprint"]
        baselines[mode] = TransitionBaseline().fit(refit_data)
        write_json(dest / "baseline.json", baselines[mode].as_dict())
        del refit_data, data, val, model, reloaded
    if audit_only:
        return
    if not smoke:
        # All model selection, final fits and export checks precede test loading.
        test = load_quarters(root, seasons=(2026,))
        write_json(directory / "test_audit.json", dict(test.audit, provenance=test.provenance))
        for mode in MODES:
            dest = directory / mode
            model, encoder, info = load_model(dest / "model", device)
            data = make_data(make_streams(test.events, mode), encoder, model.config, device)
            p, xy = predict(model, data, batch_size(mode))
            record(mode, "test", "transformer", data, p, xy, info["class_weights"])
            record(mode, "test", "transition", data, *baselines[mode].predict(data), info["class_weights"])
            predictions = data.frame.iloc[data.scored_rows][["token_id", "next_token_id", "stream_id", "target", "has_location", "target_x", "target_y"]].copy()
            predictions[["predicted_x", "predicted_y"]] = xy
            predictions["predicted_class"] = p.argmax(1)
            for index in range(p.shape[1]):
                predictions[f"p_{index}"] = p[:, index]
            predictions.to_parquet(dest / "test-predictions.parquet", index=False)
            if state_fingerprint(model) != frozen_weights[mode]:
                raise ValueError("Model parameters changed during evaluation.")
    pd.DataFrame(results).to_csv(directory / "metrics.csv", index=False)
    write_json(directory / "summary.json", dict(plan=plan, selected=winners, results=results,
        limitations=["Different tasks; raw scores and selection objectives are not a direct model ranking.",
                     "2026 is an already-observed benchmark, not an untouched holdout.",
                     "Retrospective quality masks can favour cleaner passages.",
                     "Single-seed bounded hyperparameter search.",
                     "Coordinate outputs are points, not spatial distributions.",
                     "Team streams are an AFL adaptation, not an exact reproduction on soccer data."],
        status="smoke_only" if smoke else "complete"))
    print(pd.DataFrame(results).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="artifacts/event-comparison-v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.output == "artifacts/event-comparison-v1":
        args.output = "artifacts/event-comparison-smoke"
    run(Path.cwd(), Path(args.output), device=args.device, resume=args.resume, audit_only=args.audit_only, smoke=args.smoke)


if __name__ == "__main__":
    main()
