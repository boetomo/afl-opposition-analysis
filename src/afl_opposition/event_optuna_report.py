"""Export completed Optuna studies as compact, inspectable repository reports."""
import argparse
import json
from pathlib import Path
import shutil

from afl_opposition.event_optuna import read_json, sources, write_json
from afl_opposition.quarter_events import MODES


def export(root, experiment, output):
    root, experiment, output = Path(root), Path(experiment), Path(output)
    summary = read_json(experiment / 'summary.json')
    if summary['status'] != 'complete':
        raise ValueError('Only completed full studies can be exported as evaluation results.')
    if summary['plan']['sources'] != sources(root):
        raise ValueError('Training sources differ from the frozen experiment.')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Report destination is not empty. Preserve it and choose a new directory.')
    for name in ['summary.json', 'plan.json', 'metrics.csv', 'metrics-summary.csv',
                 'development_audit.json', 'test_audit.json', 'frozen_models.json']:
        shutil.copyfile(experiment / name, output / name)
    for mode in MODES:
        src, dest = experiment / mode, output / mode
        dest.mkdir(exist_ok=True)
        for pattern in ['search.json', 'selection.json', 'coverage.json', 'trials.csv',
                        '*-metrics.json', '*-confusion.csv', '*-per-class.csv']:
            for path in src.glob(pattern):
                shutil.copyfile(path, dest / path.name)
        for pattern in ['trial-*/history.json', 'trial-*/result.json', 'finalist-*/history.json',
                        'finalist-*/result.json', 'refit-*/history.json', 'refit-*/result.json',
                        'model-seed-*/model.json']:
            for path in src.glob(pattern):
                target = dest / path.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    (output / 'README.md').write_text(
        '# Optuna next-event studies\n\n'
        'Separate validation searches for Seq2Event team streams and full-quarter causal next-team/action prediction. '
        'Each search has 60 trials, followed by three-seed finalist selection and fresh training-plus-validation refits. '
        'All six final models were frozen before evaluating the already-observed 2026 benchmark.\n\n'
        'The two formulations have different targets, class weights and location masks. Compare each with its own '
        'transition baseline. Seed standard deviations describe training variability, not sampling confidence intervals. '
        'Notebook 14 reads these files without model weights or raw data.\n', encoding='utf-8')
    print('Exported:', output.relative_to(root).as_posix(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='artifacts/event-optuna-v1')
    parser.add_argument('--output', default='reports/examples/event-optuna-v1')
    args = parser.parse_args()
    export(Path.cwd(), args.experiment, args.output)


if __name__ == '__main__':
    main()
