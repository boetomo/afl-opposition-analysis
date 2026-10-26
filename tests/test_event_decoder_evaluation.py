"""Decoder evaluation gates, unchanged source state and complete seed reporting."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import torch

from test_event_comparison import observations, config
from afl_opposition import event_decoder_evaluation as evaluation
from afl_opposition.event_comparison import TransitionBaseline
from afl_opposition.event_models import EventData, EventEncoder, EventTransformer, configure, save_model
from afl_opposition.event_optuna import read_json, write_json
from afl_opposition.quarter_events import QuarterData, make_streams, K


class DecoderEvaluationTests(unittest.TestCase):
    def setUp(self):
        configure('cpu', 2026)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.experiment = self.root / 'experiment'
        self.joint = self.experiment / 'joint'
        self.joint.mkdir(parents=True)
        self.output = self.root / 'evaluation'
        self.seeds = [2026, 2027, 2028]
        self.settings = config('joint')
        observed = observations()[0]
        frames = []
        for season in (2023, 2024, 2025):
            part = observed.copy()
            part['season'] = season
            part['match_id'] = f'M{season}'
            part['quarter_id'] = f'M{season}:q1'
            part['token_id'] = part.token_id.astype(str) + f':{season}'
            frames.append(part)
        frame = make_streams(pd.concat(frames, ignore_index=True), 'joint')
        encoder = EventEncoder.fit(frame)
        data = EventData(frame, encoder, 'joint', 1)
        self.test = observed.copy()
        self.test['season'] = 2026
        self.test['match_id'] = 'M2026'
        self.test['quarter_id'] = 'M2026:q1'
        seed_records = [dict(seed=seed, best_epoch=1, config=self.settings, validation_loss=1.) for seed in self.seeds]
        winner = dict(trial=0, config=self.settings, seeds=seed_records, refit_epochs=1,
                      mean_validation_loss=1., std_validation_loss=0.)
        plan = dict(sources={}, runtime=dict(torch=str(torch.__version__), optuna=optuna.__version__, device='cpu'),
                    protocol=dict(trials=1, seeds=self.seeds, batches=dict(joint=8)),
                    split=dict(train=[2023, 2024], validation=[2025], refit=[2023, 2024, 2025], test=[2026]))
        write_json(self.experiment / 'plan.json', plan)
        write_json(self.joint / 'selection.json', dict(winner=winner, candidates=[winner]))
        write_json(self.joint / 'search.json', dict(trials=1, completed=1, pruned=0))
        write_json(self.joint / 'coverage.json', {})
        (self.joint / 'trials.csv').write_text('number,state\n0,COMPLETE\n')
        with closing(sqlite3.connect(self.joint / 'study.sqlite3')) as db, db:
            db.execute('CREATE TABLE trials (number INTEGER, state TEXT)')
            db.execute("INSERT INTO trials VALUES (0, 'COMPLETE')")
        write_json(self.joint / 'baseline.json', TransitionBaseline().fit(data).as_dict())
        for seed in self.seeds:
            configure('cpu', seed)
            model = EventTransformer(self.settings, len(encoder.clubs))
            save_model(self.joint / f'model-seed-{seed}', model, encoder,
                       dict(seed=seed, selected=winner, refit_seasons=[2023, 2024, 2025],
                            refit_epochs=1, class_weights=[1.] * (2 * K + 1)))
            write_json(self.joint / f'refit-seed-{seed}/result.json',
                       dict(seed=seed, config=self.settings, stopping_reason='refit_complete', epochs_completed=1))
            write_json(self.joint / f'validation-transformer-seed-{seed}-metrics.json',
                       dict(mode='joint', split='validation', model='transformer', seed=seed, selection_loss=1.))
        write_json(self.joint / 'validation-transition-metrics.json',
                   dict(mode='joint', split='validation', model='transition', seed=None, selection_loss=2.))
        paused = self.experiment / 'team_stream/trial-000'
        paused.mkdir(parents=True)
        (paused / 'checkpoint.pt').write_bytes(b'preserve this paused checkpoint')
        write_json(paused / 'history.json', [dict(epoch=1)])
        (paused.parent / 'sampler.pkl').write_bytes(b'preserve this sampler')

    def loader(self, root, seasons):
        self.assertEqual(seasons, (2026,))
        manifest = read_json(self.output / 'frozen_decoder.json')
        self.assertEqual(set(manifest['models']), set(map(str, self.seeds)))
        return QuarterData(self.test.copy(), {}, {})

    def test_three_refits_frozen_before_test_and_original_study_unchanged(self):
        before = evaluation.snapshot(self.experiment)
        with patch.object(evaluation, 'sources', return_value={}), \
             patch.object(evaluation, 'load_quarters', side_effect=self.loader) as load:
            summary = evaluation.evaluate(self.root, self.experiment, self.output, 'cpu')
        self.assertEqual(load.call_count, 1)
        self.assertEqual(summary['scope'], 'decoder_only')
        self.assertEqual(summary['source_experiment_status'], 'incomplete')
        test = [r for r in summary['results'] if r['split'] == 'test']
        self.assertEqual([r['seed'] for r in test if r['model'] == 'transformer'], self.seeds)
        self.assertEqual(sum(r['model'] == 'transition' for r in test), 1)
        aggregates = pd.read_csv(self.output / 'metrics-summary.csv')
        row = aggregates.loc[(aggregates.split == 'test') & (aggregates.model == 'transformer') & (aggregates.metric == 'joint_accuracy')].iloc[0]
        values = [r['joint_accuracy'] for r in test if r['model'] == 'transformer']
        self.assertAlmostEqual(row['mean'], np.mean(values))
        self.assertAlmostEqual(row['std'], np.std(values, ddof=1))
        self.assertEqual(before, evaluation.snapshot(self.experiment))
        self.assertFalse((self.experiment / 'summary.json').exists())
        report = self.root / 'report'
        evaluation.export_report(self.output, report)
        self.assertTrue((report / 'test-transformer-seed-2026-joint-confusion.csv').is_file())
        self.assertFalse(list(report.rglob('*.pt')))
        self.assertFalse(list(report.rglob('*.sqlite3')))
        with self.assertRaises(ValueError):
            evaluation.export_report(self.output, report)
        with self.assertRaises(ValueError):
            evaluation.evaluate(self.root, self.experiment, self.output, 'cpu')

    def assert_rejected_before_test(self, message):
        with patch.object(evaluation, 'sources', return_value={}), \
             patch.object(evaluation, 'load_quarters') as load:
            with self.assertRaisesRegex(ValueError, message):
                evaluation.evaluate(self.root, self.experiment, self.output, 'cpu')
            load.assert_not_called()

    def test_incomplete_search_cannot_load_test(self):
        with closing(sqlite3.connect(self.joint / 'study.sqlite3')) as db, db:
            db.execute("UPDATE trials SET state='RUNNING'")
        self.assert_rejected_before_test('incomplete')

    def test_unfinished_refit_cannot_load_test(self):
        path = self.joint / 'refit-seed-2028/result.json'
        result = read_json(path)
        result['epochs_completed'] = 0
        write_json(path, result)
        self.assert_rejected_before_test('completed selected refit')

    def test_source_changes_cannot_load_test(self):
        plan = read_json(self.experiment / 'plan.json')
        plan['sources'] = {'changed.py': 'mismatch'}
        write_json(self.experiment / 'plan.json', plan)
        self.assert_rejected_before_test('source files changed')

    def test_changed_weights_are_rejected_after_manifest(self):
        def loader(root, seasons):
            test = self.loader(root, seasons)
            path = self.joint / 'model-seed-2026/weights.pt'
            path.write_bytes(b'changed after the manifest was written')
            return test
        with patch.object(evaluation, 'sources', return_value={}), \
             patch.object(evaluation, 'load_quarters', side_effect=loader):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                evaluation.evaluate(self.root, self.experiment, self.output, 'cpu')
        self.assertFalse((self.output / 'summary.json').exists())

    def test_output_cannot_modify_source_experiment(self):
        with self.assertRaisesRegex(ValueError, 'separate'):
            evaluation.evaluate(self.root, self.experiment, self.experiment / 'evaluation', 'cpu')
        with self.assertRaisesRegex(ValueError, 'separate'):
            evaluation.evaluate(self.root, self.experiment, self.root, 'cpu')


if __name__ == '__main__':
    unittest.main()
