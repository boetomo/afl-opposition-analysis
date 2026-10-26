"""Evaluate an already-selected Optuna decoder without finishing the team-stream study.

This is a separate evaluation-order amendment. It never fits models, changes the
source study, or marks the combined experiment complete.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pandas as pd
import torch
import optuna

from afl_opposition.event_comparison import TransitionBaseline
from afl_opposition.event_models import EventData, configure, load_model, predict
from afl_opposition.event_optuna import (
    aggregate, experiment_lock, read_json, record, sources, write_json,
)
from afl_opposition.quarter_events import K, load_quarters, make_streams
from afl_opposition.sequence import state_fingerprint

CONTRACT = 'event-decoder-evaluation-v1'
AMENDMENT = (
    'Evaluate the completed decoder study before the Seq2Event study finishes. '
    'Decoder configuration, validation selection, refit duration, three seeds and '
    'weights remain fixed. The source combined experiment remains unfinished.'
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(experiment):
    """Fingerprint the frozen decoder and the resumable team-stream state."""
    paths = [experiment / 'plan.json']
    for pattern in ('*.json', '*.csv', '*.sqlite3', 'sampler.pkl',
                    'model-seed-*/*', 'refit-seed-*/result.json'):
        paths.extend((experiment / 'joint').glob(pattern))
    for name in ('study.sqlite3', 'sampler.pkl'):
        path = experiment / 'team_stream' / name
        if path.exists():
            paths.append(path)
    checkpoints = sorted((experiment / 'team_stream').glob('trial-*/checkpoint.pt'))
    if checkpoints:
        paths.extend([checkpoints[-1], checkpoints[-1].with_name('history.json')])
    return {p.relative_to(experiment).as_posix(): digest(p) for p in paths if p.is_file()}


def freeze_decoder(root, experiment):
    """Check every seed and refit before the evaluator can load the test season."""
    plan = read_json(experiment / 'plan.json')
    if sources(root) != plan['sources']:
        raise ValueError('Frozen training source files changed.')
    if str(torch.__version__) != plan['runtime']['torch'] or optuna.__version__ != plan['runtime']['optuna']:
        raise ValueError('Training runtime versions changed.')
    if plan['split'] != dict(train=[2023, 2024], validation=[2025], refit=[2023, 2024, 2025], test=[2026]):
        raise ValueError('Unexpected season split.')
    folder = experiment / 'joint'
    search = read_json(folder / 'search.json')
    budget = plan['protocol']['trials']
    with closing(sqlite3.connect((folder / 'study.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Decoder study database is not intact.')
        trials = db.execute('SELECT number, state FROM trials ORDER BY number').fetchall()
    if ([number for number, _ in trials] != list(range(budget))
            or any(state not in ('COMPLETE', 'PRUNED') for _, state in trials)
            or search['trials'] != budget
            or search['completed'] != sum(state == 'COMPLETE' for _, state in trials)
            or search['pruned'] != sum(state == 'PRUNED' for _, state in trials)):
        raise ValueError('Decoder search is incomplete or its summary is inconsistent.')
    selection = read_json(folder / 'selection.json')
    winner = selection['winner']
    expected_winner = min(selection['candidates'], key=lambda c: (c['mean_validation_loss'], c['trial']))
    seeds = plan['protocol']['seeds']
    if len(seeds) != 3 or len(set(seeds)) != 3 or winner != expected_winner:
        raise ValueError('Require the validation-selected configuration and all three seeds.')
    if sorted(r['seed'] for r in winner['seeds']) != sorted(seeds):
        raise ValueError('Finalist seed selection is incomplete.')
    if winner['refit_epochs'] != int(np.median([r['best_epoch'] for r in winner['seeds']])):
        raise ValueError('Refit duration differs from validation selection.')
    if winner['config']['mode'] != 'joint' or winner['config']['window'] != 0:
        raise ValueError('Expected the full-quarter chronological decoder.')
    models, common_encoder = {}, None
    for seed in seeds:
        model, _, info = load_model(folder / f'model-seed-{seed}', 'cpu')
        refit = read_json(folder / f'refit-seed-{seed}/result.json')
        if (info['seed'] != seed or info['selected'] != winner
                or info['config'] != winner['config'] or info['refit_seasons'] != plan['split']['refit']
                or info['refit_epochs'] != winner['refit_epochs']
                or refit['seed'] != seed or refit['config'] != winner['config']
                or refit['stopping_reason'] != 'refit_complete'
                or refit['epochs_completed'] != winner['refit_epochs']
                or info['class_weights'] != [1.] * (2 * K + 1)):
            raise ValueError(f'Seed {seed} is not the completed selected refit.')
        if common_encoder is not None and info['encoder'] != common_encoder:
            raise ValueError('Refit encoders differ between seeds.')
        common_encoder = info['encoder']
        models[str(seed)] = dict(fingerprint=info['fingerprint'], weights_sha256=info['weights_sha256'])
        del model
    baseline = read_json(folder / 'baseline.json')
    p, xy = np.asarray(baseline['probabilities']), np.asarray(baseline['displacement'])
    if (baseline['mode'] != 'joint' or baseline['smoothing'] != .5
            or p.shape != (2 * K + 1, 2 * K + 1) or xy.shape != (2 * K + 1, 2)
            or not np.isfinite(p).all() or not np.isfinite(xy).all()
            or (p < 0).any() or not np.allclose(p.sum(1), 1.)):
        raise ValueError('Invalid frozen transition baseline.')
    return plan, selection, models


def evaluate(root, experiment, output, device='auto'):
    root, experiment, output = map(lambda p: Path(p).resolve(), (root, experiment, output))
    if output == experiment or output.is_relative_to(experiment) or experiment.is_relative_to(output):
        raise ValueError('Evaluation output must be separate from the source experiment.')
    if not experiment.is_dir():
        raise ValueError('Source experiment does not exist.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Evaluation output exists. Preserve it and choose a new directory.')
    with experiment_lock(experiment), experiment_lock(output):
        protected = snapshot(experiment)
        plan, selection, models = freeze_decoder(root, experiment)
        chosen_device = configure(device, plan['protocol']['seeds'][0])
        manifest = dict(contract=CONTRACT, created_utc=datetime.now(timezone.utc).isoformat(),
            source_experiment=experiment.relative_to(root).as_posix(), amendment=AMENDMENT,
            source_plan_sha256=digest(experiment / 'plan.json'), source_files=protected,
            evaluator_sha256=digest(Path(__file__)), selected=selection['winner'], models=models,
            seeds=plan['protocol']['seeds'], runtime=dict(torch=str(torch.__version__),
            optuna=optuna.__version__, device=chosen_device))
        write_json(output / 'frozen_decoder.json', manifest)
        write_json(output / 'source-plan.json', plan)
        for name in ('search.json', 'selection.json', 'trials.csv', 'coverage.json', 'baseline.json'):
            shutil.copyfile(experiment / 'joint' / name, output / name)
        for path in (experiment / 'joint').glob('validation-*'):
            if path.is_file():
                shutil.copyfile(path, output / path.name)
        results = []
        for seed in [*plan['protocol']['seeds'], None]:
            stem = 'validation-transition' if seed is None else f'validation-transformer-seed-{seed}'
            row = read_json(output / f'{stem}-metrics.json')
            if row['mode'] != 'joint' or row['split'] != 'validation' or row['seed'] != seed:
                raise ValueError('Unexpected saved validation record.')
            results.append(row)
        print('All three decoder refits and baseline frozen. Loading 2026 for evaluation.', flush=True)
        test = load_quarters(root, seasons=(2026,))
        write_json(output / 'test_audit.json', dict(test.audit, provenance=test.provenance))
        frame = make_streams(test.events, 'joint')
        if set(frame.season) != {2026}:
            raise ValueError('Evaluation loader did not return only the test season.')
        for seed in plan['protocol']['seeds']:
            model, encoder, info = load_model(experiment / 'joint' / f'model-seed-{seed}', chosen_device)
            if info['fingerprint'] != models[str(seed)]['fingerprint']:
                raise ValueError('Model differs from the pre-test manifest.')
            data = EventData(frame, encoder, 'joint', 1, chosen_device)
            probabilities, xy = predict(model, data, plan['protocol']['batches']['joint'])
            result = record(output, 'joint', 'test', 'transformer', seed, data, probabilities, xy, info['class_weights'])
            results.append(result)
            write_json(output / f'model-seed-{seed}.json', info)
            if state_fingerprint(model) != models[str(seed)]['fingerprint']:
                raise ValueError('Model weights changed during evaluation.')
            print(f"Seed {seed}: accuracy={result['joint_accuracy']:.6f}, log_loss={result['joint_log_loss']:.6f}, location_RMSE={result['location_rmse_norm']:.6f}", flush=True)
            if seed == plan['protocol']['seeds'][0]:
                baseline_info = read_json(output / 'baseline.json')
                baseline = TransitionBaseline()
                baseline.probabilities = np.asarray(baseline_info['probabilities'])
                baseline.displacement = np.asarray(baseline_info['displacement'])
                results.append(record(output, 'joint', 'test', 'transition', None, data,
                    *baseline.predict(data), info['class_weights']))
            del model, data, probabilities, xy
        if snapshot(experiment) != protected or sources(root) != plan['sources']:
            raise ValueError('Source experiment changed during the evaluation.')
        pd.DataFrame(results).to_csv(output / 'metrics.csv', index=False)
        aggregate(results).to_csv(output / 'metrics-summary.csv', index=False)
        summary = dict(contract=CONTRACT, status='complete', scope='decoder_only',
            source_experiment_status='incomplete' if not (experiment / 'summary.json').exists() else 'complete',
            amendment=AMENDMENT, selected=selection['winner'], seeds=plan['protocol']['seeds'],
            results=results, limitations=[
                '2026 is an already-observed project benchmark, not an untouched holdout.',
                'This decoder-only evaluation changes the original order: Seq2Event is unfinished.',
                'Decoder test results are available before Seq2Event selection ends. Keep its existing search frozen.',
                'All three seeds are reported; none is selected by test performance.',
                'Seed standard deviation measures optimisation variability, not match-sampling uncertainty.',
                'Retrospective quality filters favour cleaner passages; locations are point estimates.',
            ])
        write_json(output / 'summary.json', summary)
        print('Decoder-only evaluation complete. Seq2Event remains paused.', flush=True)
        return summary


def export_report(experiment, output):
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    summary = read_json(experiment / 'summary.json')
    if summary.get('contract') != CONTRACT or summary.get('status') != 'complete' or summary.get('scope') != 'decoder_only':
        raise ValueError('Require a completed decoder-only evaluation.')
    if output == experiment or output.is_relative_to(experiment) or experiment.is_relative_to(output):
        raise ValueError('Report output must be separate from the evaluation.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Report output exists. Preserve it and choose a new directory.')
    output.mkdir(parents=True, exist_ok=True)
    for path in experiment.iterdir():
        if path.is_file() and path.suffix in {'.json', '.csv'}:
            shutil.copyfile(path, output / path.name)
    (output / 'README.md').write_text(
        '# Optuna decoder evaluation\n\n'
        'Completed evaluation of all three validation-selected decoder refits and the frozen transition baseline. '
        'Seq2Event remains a separate unfinished study. The evaluation-order amendment and pre-test weight '
        'fingerprints are recorded in `frozen_decoder.json`. No model weights or raw match archives are included.\n\n'
        'See [the evaluation note](../../../docs/decoder-evaluation.md) and '
        '[Notebook 15](../../../notebooks/15_decoder_evaluation.ipynb).\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='artifacts/event-optuna-v1')
    parser.add_argument('--output', default='artifacts/event-decoder-optuna-v1')
    parser.add_argument('--report', default='reports/examples/event-decoder-optuna-v1')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    evaluate(Path.cwd(), args.experiment, args.output, args.device)
    export_report(args.output, args.report)


if __name__ == '__main__':
    main()
