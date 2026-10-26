"""Search validity, leakage gates, model compatibility and interruption tests."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import torch

from test_event_comparison import observations, config
from afl_opposition.event_models import EventData, EventEncoder, EventTransformer, configure, class_weights, predict, load_model
from afl_opposition.event_comparison import score
from afl_opposition.quarter_events import make_streams, QuarterData, MODES
from afl_opposition.sequence import state_fingerprint
from afl_opposition import event_optuna as tuning
from afl_opposition import event_optuna_report as reporting


class OptunaTests(unittest.TestCase):
    def setUp(self):
        configure('cpu', 2026)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.observed = observations()[0]
        self.protocol = copy.deepcopy(tuning.PROTOCOL)
        self.protocol.update(trials=2, startup_trials=2, max_epochs=2, minimum_epochs=1,
                             finalist_count=1, seeds=[2026, 2027, 2028], seq2event_window=2)

    def data(self, mode='joint'):
        stream = make_streams(self.observed, mode)
        encoder = EventEncoder.fit(stream)
        return EventData(stream, encoder, mode, 2), encoder

    def test_search_space_always_builds_valid_models(self):
        for mode in MODES:
            study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
            for _ in range(35):
                trial = study.ask()
                settings = tuning.suggest_config(trial, mode)
                model = EventTransformer(settings, 18)
                self.assertGreater(model.continuous.out_features, 0)
                self.assertEqual(model.transformer.layers[0].linear1.out_features,
                                 settings['width'] * settings['ff_multiplier'])
                if mode == 'joint':
                    self.assertEqual(model.club.embedding_dim, settings['club_embedding_dim'])
                study.tell(trial, 1.)

    def test_new_configurations_preserve_causal_attention(self):
        data, _ = self.data()
        settings = dict(config('joint'), width=64, layers=2, club_embedding_dim=16, ff_multiplier=2)
        model = EventTransformer(settings, 2).eval()
        inputs = next(data.batches(1))[0]
        with torch.inference_mode():
            full = model(*inputs)
            prefix = model(*[x[:, :3] for x in inputs])
        for a, b in zip(full, prefix):
            torch.testing.assert_close(a[:, :3], b, atol=1e-6, rtol=1e-5)

    def test_fast_validation_matches_full_metrics(self):
        for mode in MODES:
            data, _ = self.data(mode)
            model = EventTransformer(config(mode), 2).eval()
            weights = class_weights(data)
            actual = tuning.validation_objective(model, data, weights, 2)
            expected = score(data, *predict(model, data, 2), weights.numpy())[0]
            self.assertAlmostEqual(actual['selection_loss'], expected['selection_loss'], places=5)

    def test_checkpoint_resume_is_exact_and_rejects_changes(self):
        data, _ = self.data()
        settings = dict(config('joint'), dropout=.2)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            expected, _ = tuning.fit(data, data, settings, folder / 'a', self.protocol, seed=42)
            real = tuning.validation_objective
            calls = [0]
            def interrupt(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 2:
                    raise InterruptedError('Simulate a process interruption after the first checkpoint.')
                return real(*args, **kwargs)
            with patch.object(tuning, 'validation_objective', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    tuning.fit(data, data, settings, folder / 'b', self.protocol, seed=42)
            actual, _ = tuning.fit(data, data, settings, folder / 'b', self.protocol, seed=42)
            self.assertEqual(state_fingerprint(expected), state_fingerprint(actual))
            with self.assertRaises(ValueError):
                tuning.fit(data, data, dict(settings, learning_rate=.9), folder / 'b', self.protocol, seed=42)

    def test_search_resume_preserves_sampler_and_running_trial(self):
        data, _ = self.data()
        protocol = dict(self.protocol, max_epochs=1, trials=5)
        def fake_fit(data, val, settings, dest, protocol, **kwargs):
            return None, dict(best_epoch=1, validation_loss=float(settings['width']),
                              seed=kwargs['seed'], config=settings)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with patch.object(tuning, 'fit', side_effect=fake_fit):
                tuning.search('joint', data, data, folder / 'a', protocol)
            calls = [0]
            def interrupted(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 4:
                    raise InterruptedError('Simulate an interrupted running trial.')
                return fake_fit(*args, **kwargs)
            with patch.object(tuning, 'fit', side_effect=interrupted):
                with self.assertRaises(InterruptedError):
                    tuning.search('joint', data, data, folder / 'b', protocol)
            with patch.object(tuning, 'fit', side_effect=fake_fit):
                tuning.search('joint', data, data, folder / 'b', protocol)
            a, b = [pd.read_csv(folder / name / 'trials.csv') for name in ['a', 'b']]
            columns = [c for c in a if c.startswith('params_')]
            pd.testing.assert_frame_equal(a[columns], b[columns])
            self.assertEqual(b.state.tolist(), ['COMPLETE'] * 5)

    def test_concurrent_worker_is_rejected_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as folder:
            with tuning.experiment_lock(folder):
                with self.assertRaises(RuntimeError):
                    with tuning.experiment_lock(folder):
                        pass
            with tuning.experiment_lock(folder):
                pass

    def test_pruning_waits_for_minimum_epochs(self):
        data, _ = self.data()
        calls = []
        class Trial:
            def report(self, value, step):
                calls.append(('report', step))
            def should_prune(self):
                calls.append(('prune', None))
                return True
        protocol = dict(self.protocol, minimum_epochs=2)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(optuna.TrialPruned):
                tuning.fit(data, data, config('joint'), Path(folder), protocol, seed=42, trial=Trial())
            self.assertEqual(calls, [('report', 1), ('report', 2), ('prune', None)])
            with self.assertRaises(optuna.TrialPruned):
                tuning.fit(data, data, config('joint'), Path(folder), protocol, seed=42, trial=Trial())

    def test_all_six_models_frozen_before_test_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'experiment'
            accessed = []
            def loader(root, seasons=(2023, 2024, 2025)):
                if seasons == (2026,):
                    self.assertTrue((output / 'frozen_models.json').is_file())
                    for mode in MODES:
                        for seed in self.protocol['seeds']:
                            _, _, metadata = load_model(output / mode / f'model-seed-{seed}')
                            self.assertEqual(metadata['refit_seasons'], [2023, 2024, 2025])
                    accessed.append('test')
                pieces = []
                for season in seasons:
                    part = self.observed.copy()
                    part['season'] = season
                    part['match_id'] = f'M{season}'
                    part['quarter_id'] = f'M{season}:q1'
                    part['token_id'] = part.token_id.astype(str) + f':{season}'
                    pieces.append(part)
                return QuarterData(pd.concat(pieces, ignore_index=True), {}, {})
            protocol = dict(self.protocol, max_epochs=1)
            with patch.object(tuning, 'load_quarters', side_effect=loader), \
                 patch.object(tuning, 'sources', return_value={}), \
                 patch.object(tuning, 'suggest_config', side_effect=lambda trial, mode: config(mode)):
                tuning.run(Path(folder), output, device='cpu', protocol=protocol)
            self.assertEqual(accessed, ['test'])
            summary = tuning.read_json(output / 'summary.json')
            self.assertEqual(summary['status'], 'complete')
            test = [r for r in summary['results'] if r['split'] == 'test' and r['model'] == 'transformer']
            self.assertEqual(len(test), 6)
            report = Path(folder) / 'reports'
            with patch.object(reporting, 'sources', return_value={}):
                reporting.export(Path(folder), output, report)
                self.assertTrue((report / 'summary.json').is_file())
                self.assertTrue((report / 'joint/test-transformer-seed-2026-joint-confusion.csv').is_file())
                self.assertFalse(list(report.rglob('*.pt')))
                with self.assertRaises(ValueError):
                    reporting.export(Path(folder), output, report)
            with self.assertRaises(ValueError):
                tuning.run(Path(folder), output, device='cpu', resume=True, protocol=protocol)

    def test_legacy_weights_still_load(self):
        root = Path(__file__).resolve().parents[1]
        for mode in MODES:
            path = root / 'artifacts/event-comparison-v1' / mode / 'model'
            if not path.exists():
                continue
            model, _, metadata = load_model(path)
            self.assertEqual(state_fingerprint(model), metadata['fingerprint'])


if __name__ == '__main__':
    unittest.main()
