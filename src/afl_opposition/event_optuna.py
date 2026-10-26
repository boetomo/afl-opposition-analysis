"""Resumable, validation-only Optuna searches for two next-event formulations.

Run from the repository root: python -m afl_opposition.event_optuna
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import optuna
import pandas as pd
import torch
from torch.nn import functional as F
from sqlalchemy.pool import NullPool

from afl_opposition.event_comparison import TransitionBaseline, score, source_hashes
from afl_opposition.event_models import (
    EventData, EventEncoder, EventTransformer, class_weights, configure,
    objective, predict, save_model, load_model,
)
from afl_opposition.quarter_events import load_quarters, make_streams, MODES
from afl_opposition.sequence import _cpu_state, _atomic_save, state_fingerprint

SEEDS = (2026, 2027, 2028)
SEARCH_SPACE = dict(width=[64, 96, 128, 192, 256], layers=[1, 4], heads=[2, 4, 8],
    embedding_dim=[8, 16, 32], ff_multiplier=[2, 4], dense_dim=[32, 64, 128],
    dropout=[0., .4], learning_rate=[1e-5, 3e-3], weight_decay=[0., 1e-6, 1e-2],
    club_embedding_dim=[4, 8, 16])
PROTOCOL = dict(trials=60, startup_trials=15, max_epochs=60, patience=10,
    minimum_epochs=15, pruning_startup_trials=10, pruning_min_trials=5,
    finalist_count=3, seeds=list(SEEDS), batches=dict(team_stream=512, joint=8),
    action_loss_coefficient=5., seq2event_window=40,
    objective="Fixed class-weighted CE * 5 + global coordinate RMSE within each formulation",
    refit_epochs="Median best validation epoch across the selected configuration's seeds",
    context="Fixed 40-token team streams; all observed tokens within each chronological quarter")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def suggest_config(trial, mode):
    width = trial.suggest_categorical('width', SEARCH_SPACE['width'])
    club = trial.suggest_categorical('club_embedding_dim', SEARCH_SPACE['club_embedding_dim']) if mode == 'joint' else 0
    # A distinct conditional parameter avoids changing a categorical distribution
    # within one study. Every sampled configuration leaves numerical feature space.
    compact = mode == 'joint' and width == 64 and club == 16
    emb = trial.suggest_categorical('embedding_dim_compact' if compact else 'embedding_dim',
                                    [8, 16] if compact else SEARCH_SPACE['embedding_dim'])
    regularise = trial.suggest_categorical('use_l2', [False, True])
    decay = trial.suggest_float('l2_strength', 1e-6, 1e-2, log=True) if regularise else 0.
    return dict(mode=mode, width=width, embedding_dim=emb, club_embedding_dim=club,
        heads=trial.suggest_categorical('heads', SEARCH_SPACE['heads']),
        layers=trial.suggest_int('layers', 1, 4),
        ff_multiplier=trial.suggest_categorical('ff_multiplier', SEARCH_SPACE['ff_multiplier']),
        dense_dim=trial.suggest_categorical('dense_dim', SEARCH_SPACE['dense_dim']),
        dropout=trial.suggest_float('dropout', 0., .4),
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 3e-3, log=True),
        weight_decay=decay, window=40 if mode == 'team_stream' else 0)


@torch.inference_mode()
def validation_objective(model, data, weights, batch_size):
    """Global reductions match score(), without constructing confusion matrices."""
    model.eval()
    ce_total, squared_error, count, coordinates = 0., 0., 0, 0
    for inputs, target, xy, location, _ in data.batches(batch_size):
        logits, predicted = model(*inputs)
        valid = target.ge(0)
        ce_total += float(F.cross_entropy(logits[valid], target[valid], weight=weights, reduction='sum'))
        count += int(valid.sum())
        squared_error += float(((predicted[location] - xy[location]) ** 2).sum())
        coordinates += int(location.sum()) * 2
    if count != len(data.scored_rows):
        raise ValueError('Validation did not cover exactly the declared target population.')
    rmse = (squared_error / coordinates) ** .5 if coordinates else 0.
    return dict(selection_loss=5 * ce_total / count + rmse,
                weighted_cross_entropy=ce_total / count, location_rmse_norm=rmse)


def fit(data, validation, config, directory, protocol, *, seed, epochs=None, trial=None):
    """Resume an epoch exactly, with independent pruning and early-stopping rules."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = configure(data.device, seed)
    model = EventTransformer(config, int(max(data.values[3].max(), data.values[4].max()))).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    weights = class_weights(data)
    maximum = epochs if epochs is not None else protocol['max_epochs']
    signature = dict(config=config, seed=seed, validation=validation is not None,
                     epochs=maximum, protocol=protocol)
    checkpoint = directory / 'checkpoint.pt'
    history, best_state, best, best_epoch, stale = [], None, float('inf'), 0, 0
    terminal = None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Training configuration changed since the checkpoint.')
        model.load_state_dict(saved['model'])
        optimiser.load_state_dict(saved['optimiser'])
        history, best_state, best = saved['history'], saved['best_state'], saved['best']
        best_epoch, stale, terminal = saved['best_epoch'], saved['stale'], saved['terminal']
        torch.set_rng_state(saved['rng'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    if terminal == 'pruned':
        raise optuna.TrialPruned('Resuming a persisted pruning decision.')
    for epoch in range(len(history) + 1, maximum + 1):
        if terminal:
            break
        started = time.perf_counter()
        model.train()
        total, count = 0., 0
        for inputs, target, xy, location, _ in data.batches(protocol['batches'][data.mode], seed + epoch):
            eligible = int(target.ge(0).sum())
            if not eligible:
                continue
            optimiser.zero_grad(set_to_none=True)
            loss = objective(*model(*inputs), target, xy, location, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training objective.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimiser.step()
            total += float(loss.detach()) * eligible
            count += eligible
        if count != len(data.scored_rows):
            raise ValueError('An epoch did not visit every eligible target exactly once.')
        metrics = validation_objective(model, validation, weights, protocol['batches'][data.mode]) if validation is not None else None
        if metrics:
            value = metrics['selection_loss']
            if not np.isfinite(value):
                raise FloatingPointError('Non-finite validation objective.')
            if value < best:
                best, best_epoch, best_state, stale = value, epoch, _cpu_state(model), 0
            else:
                stale += 1
        else:
            best_epoch, best_state = epoch, _cpu_state(model)
        history.append(dict(epoch=epoch, training_batch_objective=total / count,
                            validation=metrics, seconds=time.perf_counter() - started))
        if trial is not None:
            trial.report(best, step=epoch)
            if epoch >= protocol['minimum_epochs'] and trial.should_prune():
                terminal = 'pruned'
        if not terminal and validation is not None and epoch >= protocol['minimum_epochs'] and stale >= protocol['patience']:
            terminal = 'early_stopping'
        if not terminal and epoch == maximum:
            terminal = 'epoch_limit' if validation is not None else 'refit_complete'
        _atomic_save(dict(signature=signature, model=model.state_dict(), optimiser=optimiser.state_dict(),
            history=history, best_state=best_state, best=best, best_epoch=best_epoch,
            stale=stale, terminal=terminal, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state().cpu() if device == 'cuda' else None), checkpoint)
        write_json(directory / 'history.json', history)
        print(f'{data.mode}/{directory.name}: epoch {epoch}/{maximum}, '
              f'{("validation=" + format(metrics["selection_loss"], ".6f")) if metrics else "refit"}, '
              f'{history[-1]["seconds"]:.1f}s' + (f', {terminal}' if terminal else ''), flush=True)
        if terminal == 'pruned':
            raise optuna.TrialPruned(f'Pruned after {epoch} epochs.')
    if best_state is None:
        raise ValueError('No training epoch completed.')
    model.load_state_dict(best_state)
    result = dict(best_epoch=best_epoch, validation_loss=best if validation is not None else None,
        epochs_completed=len(history), stopping_reason=terminal, seed=seed, config=config)
    write_json(directory / 'result.json', result)
    return model.eval(), result


def atomic_pickle(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as handle:
        pickle.dump(value, handle)
    os.replace(tmp, path)


def sampler_snapshot(study, path):
    atomic_pickle(path, dict(sampler=study.sampler, trial_count=len(study.trials)))


def search(mode, train_data, val_data, directory, protocol):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url='sqlite:///' + (directory / 'study.sqlite3').resolve().as_posix(),
        engine_kwargs={'poolclass': NullPool})
    try:
        return _search(mode, train_data, val_data, directory, protocol, storage)
    finally:
        storage.remove_session()
        storage.engine.dispose()


def _search(mode, train_data, val_data, directory, protocol, storage):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'sampler.pkl'
    sampler = optuna.samplers.TPESampler(seed=SEEDS[0], n_startup_trials=protocol['startup_trials'],
                                        multivariate=True, group=True)
    snapshot = None
    if state_path.exists():
        # This pickle is generated locally, never accepted as a downloaded model.
        with state_path.open('rb') as handle:
            snapshot = pickle.load(handle)
        sampler = snapshot['sampler']
    pruner = optuna.pruners.MedianPruner(n_startup_trials=protocol['pruning_startup_trials'],
        n_warmup_steps=protocol['minimum_epochs'], n_min_trials=protocol['pruning_min_trials'])
    study = optuna.create_study(study_name=mode, storage=storage,
        load_if_exists=True, direction='minimize', sampler=sampler, pruner=pruner)
    if snapshot and snapshot['trial_count'] != len(study.trials):
        raise ValueError('Study and sampler snapshots are inconsistent. Restore their matching backups before resuming.')
    if not snapshot and study.trials:
        raise ValueError('Existing study has no sampler snapshot; cannot promise reproducible continuation.')
    sampler_snapshot(study, state_path)
    while True:
        pending = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        if len(pending) > 1:
            raise ValueError('Only one sequential worker may run a study.')
        if pending:
            frozen = pending[0]
            # Optuna exposes Trial(study, trial_id), while FrozenTrial stores the
            # database ID internally. This bridge is covered by pinned-version tests.
            trial = optuna.Trial(study, frozen._trial_id)
            config = trial.user_attrs['config']
        elif len(study.trials) < protocol['trials']:
            trial = study.ask()
            config = suggest_config(trial, mode)
            trial.set_user_attr('config', config)
            sampler_snapshot(study, state_path)
        else:
            break
        dest = directory / f'trial-{trial.number:03d}'
        print(f'{mode}: trial {trial.number + 1}/{protocol["trials"]} {json.dumps(config)}', flush=True)
        try:
            model, result = fit(train_data, val_data, config, dest, protocol, seed=SEEDS[0], trial=trial)
            del model
            trial.set_user_attr('result', result)
            study.tell(trial.number, result['validation_loss'])
        except optuna.TrialPruned:
            study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
        # Unexpected errors stop the run and leave the trial resumable. They are
        # not silently counted as poor hyperparameters or converted to pruning.
        sampler_snapshot(study, state_path)
        study.trials_dataframe().to_csv(directory / 'trials.csv', index=False)
        gc.collect()
        if train_data.device == 'cuda':
            torch.cuda.empty_cache()
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < protocol['finalist_count']:
        raise ValueError('Insufficient completed trials for finalist selection.')
    finalists = sorted(complete, key=lambda t: (t.value, t.number))[:protocol['finalist_count']]
    summary = dict(trials=len(study.trials), completed=len(complete),
        pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        finalists=[dict(trial=t.number, config=t.user_attrs['config'], validation_loss=t.value) for t in finalists])
    write_json(directory / 'search.json', summary)
    return summary


def select_finalists(train_data, val_data, summary, directory, protocol):
    candidates = []
    for item in summary['finalists']:
        results = []
        for seed in protocol['seeds']:
            dest = directory / f'finalist-{item["trial"]:03d}-seed-{seed}'
            model, result = fit(train_data, val_data, item['config'], dest, protocol, seed=seed)
            del model
            results.append(result)
        values = [r['validation_loss'] for r in results]
        candidates.append(dict(item, seeds=results, mean_validation_loss=float(np.mean(values)),
            std_validation_loss=float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
            refit_epochs=int(np.median([r['best_epoch'] for r in results]))))
    winner = min(candidates, key=lambda c: (c['mean_validation_loss'], c['trial']))
    selected = dict(candidates=candidates, winner=winner)
    write_json(directory / 'selection.json', selected)
    return selected


def sources(root):
    result = source_hashes(root)
    result['event_optuna.py'] = hashlib.sha256((Path(root) / 'src/afl_opposition/event_optuna.py').read_bytes()).hexdigest()
    return result


def record(directory, mode, split, model_name, seed, data, probabilities, xy, weights):
    metrics, matrices, classes = score(data, probabilities, xy, weights)
    stem = f'{split}-{model_name}' + (f'-seed-{seed}' if seed is not None else '')
    for name, matrix in matrices.items():
        matrix.to_csv(directory / f'{stem}-{name}-confusion.csv', index_label='actual')
    classes.to_csv(directory / f'{stem}-per-class.csv', index=False)
    row = dict(mode=mode, split=split, model=model_name, seed=seed, **metrics)
    write_json(directory / f'{stem}-metrics.json', row)
    return row


def aggregate(results):
    rows = []
    frame = pd.DataFrame(results)
    for (mode, split, model), part in frame.groupby(['mode', 'split', 'model'], sort=False):
        for metric in part.select_dtypes(include='number').columns:
            values = part[metric].dropna()
            if metric == 'seed' or values.empty:
                continue
            rows.append(dict(mode=mode, split=split, model=model, metric=metric,
                mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values) > 1 else 0., seeds=len(values)))
    return pd.DataFrame(rows)


@contextmanager
def experiment_lock(output):
    """Prevent two local workers from modifying the same study/checkpoints."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / 'run.lock').open('a+b')
    if os.fstat(handle.fileno()).st_size == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError('This experiment is already running in another process.') from error
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == 'nt':
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def run(root, output, **kwargs):
    with experiment_lock(output):
        return _run(root, output, **kwargs)


def _run(root, output, *, device='auto', resume=False, smoke=False, protocol=None):
    root, output = Path(root), Path(output)
    protocol = dict(PROTOCOL if protocol is None else protocol)
    if smoke:
        protocol.update(trials=2, startup_trials=2, max_epochs=1, minimum_epochs=1,
                        finalist_count=1, seeds=[SEEDS[0]])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise ValueError('Completed experiments are immutable. Choose a new output directory.')
    if (output / 'plan.json').exists() and not resume:
        raise ValueError('Use --resume for an interrupted experiment.')
    device = configure(device, SEEDS[0])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    development = load_quarters(root)
    observed = development.events
    if smoke:
        matches = observed.groupby('season', sort=True).match_id.unique().map(lambda v: list(v[:2]))
        observed = observed.loc[observed.match_id.isin(sum(matches.tolist(), []))].copy()
    plan = dict(contract='event-optuna-v1', protocol=protocol, search_space=SEARCH_SPACE,
        sources=sources(root), provenance=development.provenance,
        runtime=dict(torch=str(torch.__version__), optuna=optuna.__version__, device=device),
        smoke=smoke, split=dict(train=[2023, 2024], validation=[2025], refit=[2023, 2024, 2025], test=[2026]))
    if (output / 'plan.json').exists() and read_json(output / 'plan.json') != plan:
        raise ValueError('Frozen experiment sources, data, runtime or protocol changed.')
    write_json(output / 'plan.json', plan)
    write_json(output / 'development_audit.json', development.audit)
    results, selections, frozen = [], {}, {}
    for mode in MODES:
        dest = output / mode
        dest.mkdir(exist_ok=True)
        frame = make_streams(observed, mode)
        fitting = frame.loc[frame.season.isin([2023, 2024])].reset_index(drop=True)
        validation = frame.loc[frame.season.eq(2025)].reset_index(drop=True)
        encoder = EventEncoder.fit(fitting)
        window = protocol['seq2event_window'] if mode == 'team_stream' else 1
        data, val = [EventData(f, encoder, mode, window, device) for f in (fitting, validation)]
        write_json(dest / 'coverage.json', dict(training_examples=len(data.scored_rows),
            validation_examples=len(val.scored_rows), max_stream_length=int(frame.groupby('stream_id').size().max())))
        summary = search(mode, data, val, dest, protocol)
        selected = select_finalists(data, val, summary, dest, protocol)
        selections[mode] = selected
        winner = selected['winner']
        weights = class_weights(data).cpu().numpy()
        for seed in protocol['seeds']:
            model, _ = fit(data, val, winner['config'], dest / f'finalist-{winner["trial"]:03d}-seed-{seed}', protocol, seed=seed)
            results.append(record(dest, mode, 'validation', 'transformer', seed, val,
                *predict(model, val, protocol['batches'][mode]), weights))
            del model
        baseline = TransitionBaseline().fit(data)
        results.append(record(dest, mode, 'validation', 'transition', None, val, *baseline.predict(val), weights))
        del data, val, fitting, validation
        if smoke:
            continue
        encoder = EventEncoder.fit(frame)
        refit = EventData(frame, encoder, mode, window, device)
        weights = class_weights(refit).cpu().numpy()
        frozen[mode] = {}
        for seed in protocol['seeds']:
            model, _ = fit(refit, None, winner['config'], dest / f'refit-seed-{seed}', protocol,
                seed=seed, epochs=winner['refit_epochs'])
            export = dest / f'model-seed-{seed}'
            save_model(export, model, encoder, dict(refit_seasons=[2023, 2024, 2025],
                selected=winner, seed=seed, class_weights=weights.tolist(), refit_epochs=winner['refit_epochs']))
            reloaded, _, info = load_model(export, device)
            probe = next(refit.batches(1))[0]
            with torch.inference_mode():
                for expected, actual in zip(model(*probe), reloaded(*probe)):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            frozen[mode][str(seed)] = info['fingerprint']
            del model, reloaded, probe
        baseline = TransitionBaseline().fit(refit)
        write_json(dest / 'baseline.json', baseline.as_dict())
        del frame, refit
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    if not smoke:
        # Every study, three-seed selection and all six refits must finish first.
        manifest = dict(models=frozen, selections={m: s['winner'] for m, s in selections.items()})
        write_json(output / 'frozen_models.json', manifest)
        for mode in MODES:
            for seed in protocol['seeds']:
                _, _, info = load_model(output / mode / f'model-seed-{seed}')
                if info['fingerprint'] != frozen[mode][str(seed)]:
                    raise ValueError('Frozen model manifest does not match saved weights.')
        test = load_quarters(root, seasons=(2026,))
        write_json(output / 'test_audit.json', dict(test.audit, provenance=test.provenance))
        for mode in MODES:
            dest = output / mode
            frame = make_streams(test.events, mode)
            for seed in protocol['seeds']:
                model, encoder, info = load_model(dest / f'model-seed-{seed}', device)
                data = EventData(frame, encoder, mode, max(model.config['window'], 1), device)
                probabilities, xy = predict(model, data, protocol['batches'][mode])
                results.append(record(dest, mode, 'test', 'transformer', seed, data, probabilities, xy, info['class_weights']))
                if state_fingerprint(model) != frozen[mode][str(seed)]:
                    raise ValueError('Model changed during test evaluation.')
                if seed == protocol['seeds'][0]:
                    baseline_info = read_json(dest / 'baseline.json')
                    baseline = TransitionBaseline()
                    baseline.probabilities = np.array(baseline_info['probabilities'])
                    baseline.displacement = np.array(baseline_info['displacement'])
                    results.append(record(dest, mode, 'test', 'transition', None, data,
                        *baseline.predict(data), info['class_weights']))
                del model, data, probabilities, xy
    pd.DataFrame(results).to_csv(output / 'metrics.csv', index=False)
    aggregate(results).to_csv(output / 'metrics-summary.csv', index=False)
    write_json(output / 'summary.json', dict(plan=plan, selections=selections, results=results,
        status='smoke_only' if smoke else 'complete', limitations=[
            'The two formulations solve different tasks; their losses and accuracies are not a cross-task ranking.',
            '2026 is an already-observed benchmark, not an untouched holdout.',
            'Sixty trials are a bounded search, not proof of a global optimum.',
            'Three seeds measure optimisation variability, not match-sampling confidence intervals.',
            'Class weights and action/location loss balance are fixed within each study.',
            'Retrospective quality masks can favour cleaner passages; coordinates are point predictions.']))
    print('Completed:', output.name, flush=True)


def preflight(root, device='auto'):
    """Exercise the largest architecture on the longest development histories."""
    device = configure(device, SEEDS[0])
    observed = load_quarters(root).events
    records = []
    for mode in MODES:
        frame = make_streams(observed, mode)
        encoder = EventEncoder.fit(frame.loc[frame.season.isin([2023, 2024])])
        length = 40 if mode == 'team_stream' else int(frame.groupby('stream_id').size().max())
        batch = PROTOCOL['batches'][mode]
        config = dict(mode=mode, width=256, embedding_dim=32, club_embedding_dim=16,
            heads=8, layers=4, ff_multiplier=4, dense_dim=128, dropout=.4,
            learning_rate=.001, weight_decay=.0001, window=40 if mode == 'team_stream' else 0)
        model = EventTransformer(config, len(encoder.clubs)).to(device)
        shape = (batch, length)
        inputs = [torch.ones(shape, dtype=torch.long, device=device),
                  torch.zeros((*shape, len(encoder.mean)), device=device)]
        inputs += [torch.zeros(shape, dtype=torch.long, device=device) for _ in range(3)]
        target_shape = (batch,) if mode == 'team_stream' else shape
        target = torch.zeros(target_shape, dtype=torch.long, device=device)
        xy = torch.zeros((*target_shape, 2), device=device)
        location = torch.ones(target_shape, dtype=torch.bool, device=device)
        weights = torch.ones(model.classes, device=device)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        optimiser = torch.optim.Adam(model.parameters())
        start = time.perf_counter()
        loss = objective(*model(*inputs), target, xy, location, weights)
        loss.backward()
        optimiser.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        records.append(dict(mode=mode, longest_history=length, batch_size=batch,
            seconds=time.perf_counter() - start,
            peak_gpu_GiB=torch.cuda.max_memory_allocated() / 2**30 if device == 'cuda' else None))
        del model, optimiser, inputs, target, xy, location, weights, loss
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    print(json.dumps(records, indent=2), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/event-optuna-v1')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight(Path.cwd(), args.device)
    else:
        output = 'artifacts/event-optuna-smoke' if args.smoke and args.output == 'artifacts/event-optuna-v1' else args.output
        run(Path.cwd(), output, device=args.device, resume=args.resume, smoke=args.smoke)


if __name__ == '__main__':
    main()
