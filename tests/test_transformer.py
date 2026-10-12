"""Causal attention, prefix equivalence, resume and frozen temporal evaluation."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from afl_opposition import sequence as sq
from afl_opposition import transformer as tr
from afl_opposition import transformer_experiment as ex
from test_goal_tuning import fixture
from test_sequence import examples

CONFIG = dict(num_layers=2, d_model=16, num_heads=2, ff_multiplier=2,
              embedding_dim=4, continuous_dim=8, dropout=.1, learning_rate=.003, lambda_l2=1e-5)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        tr.configure_runtime('cpu')
        self.examples = examples()
        self.encoder = sq.SequenceEncoder.fit(self.examples.loc[self.examples.season.lt(2025)])

    def model(self, dropout=0.):
        return tr.PossessionTransformer(len(self.encoder.categories)+2, {**CONFIG, 'dropout':dropout}).eval()

    def test_future_changes_cannot_change_past_predictions(self):
        model = self.model()
        actions = torch.tensor([[2,3,2,3]])
        numeric = torch.randn(1,4,4)
        before = model(actions,numeric)
        actions[:,2:] = 4
        numeric[:,2:] += 100
        torch.testing.assert_close(before[:,:2],model(actions,numeric)[:,:2],rtol=0,atol=0)

    def test_every_full_sequence_prediction_matches_its_observed_prefix(self):
        model = self.model()
        actions = torch.tensor([[2,3,2,3,4]])
        numeric = torch.randn(1,5,4)
        for training in [False, True]:
            model.train(training)
            full = model(actions,numeric)
            for length in range(1,6):
                prefix = model(actions[:,:length],numeric[:,:length])
                torch.testing.assert_close(full[:,length-1],prefix[:,-1],rtol=1e-5,atol=1e-6)

    def test_padding_and_batch_companions_do_not_change_predictions(self):
        model = self.model()
        actions = torch.tensor([[2,3,0,0],[3,4,2,3]])
        numeric = torch.randn(2,4,4)
        expected = model(actions[:1,:2],numeric[:1,:2])
        actual = model(actions,numeric)[:1,:2]
        torch.testing.assert_close(expected,actual,rtol=1e-5,atol=1e-6)
        numeric[0,2:] = 1000
        numeric[1] = -100
        torch.testing.assert_close(actual,model(actions,numeric)[:1,:2],rtol=0,atol=0)

    def test_padded_keys_are_excluded_even_when_inside_an_input(self):
        model = self.model()
        actions = torch.tensor([[2,0,3]])
        numeric = torch.randn(1,3,4)
        before = model(actions,numeric)[:,-1]
        numeric[:,1] += 1000
        torch.testing.assert_close(before,model(actions,numeric)[:,-1],rtol=0,atol=0)

    def test_gradient_is_causal_and_history_is_used(self):
        model = self.model()
        actions = torch.tensor([[2,3,2,3]])
        numeric = torch.randn(1,4,4,requires_grad=True)
        model(actions,numeric)[0,1].backward()
        self.assertEqual(numeric.grad[:,2:].abs().sum().item(),0.)
        self.assertGreater(numeric.grad[:,:1].abs().sum().item(),0.)
        self.assertGreater(model.continuous[0].weight.grad.abs().sum().item(),0.)
        self.assertGreater(model.embedding.weight.grad[2:].abs().sum().item(),0.)

    def test_position_encoding_is_length_independent_and_layers_start_differently(self):
        small = tr.positional_encoding(3,16,'cpu',torch.float32)
        large = tr.positional_encoding(100,16,'cpu',torch.float32)
        torch.testing.assert_close(small,large[:3],rtol=0,atol=0)
        model = self.model()
        self.assertFalse(torch.equal(model.layers[0].linear1.weight,model.layers[1].linear1.weight))
        with self.assertRaisesRegex(ValueError,'divisible'):
            tr.PossessionTransformer(5,{**CONFIG,'num_heads':3})

    def test_rows_restored_and_test_labels_do_not_affect_prediction(self):
        data = examples(seasons=(2026,)).sample(frac=1,random_state=8).reset_index(drop=True)
        model = self.model()
        sequences = sq.make_sequences(data,self.encoder)
        actual = tr.predict(model,sequences)
        sequences.target = 1-sequences.target
        np.testing.assert_array_equal(actual,tr.predict(model,sequences))
        for possession in data.possession_id.unique():
            subset = data.loc[data.possession_id.eq(possession)]
            expected = tr.predict(model,sq.make_sequences(subset,self.encoder,labelled=False))
            np.testing.assert_allclose(actual[subset.index],expected,atol=1e-6,rtol=1e-5)

    def check_resume(self,device):
        tr.configure_runtime(device)
        train = sq.make_sequences(self.examples.loc[self.examples.season.lt(2025)],self.encoder).to(device)
        val = sq.make_sequences(self.examples.loc[self.examples.season.eq(2025)],self.encoder).to(device)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'checkpoint.pt'
            complete,_,_ = tr.train_candidate(train,val,CONFIG,len(self.encoder.categories)+2,epochs=4,batch_size=3)
            tr.train_candidate(train,val,CONFIG,len(self.encoder.categories)+2,epochs=2,batch_size=3,checkpoint=checkpoint)
            resumed,_,_ = tr.train_candidate(train,val,CONFIG,len(self.encoder.categories)+2,epochs=4,batch_size=3,checkpoint=checkpoint)
            self.assertEqual(sq.state_fingerprint(complete),sq.state_fingerprint(resumed))
            with self.assertRaisesRegex(ValueError,'settings'):
                tr.train_candidate(train,val,CONFIG,len(self.encoder.categories)+2,epochs=4,batch_size=4,checkpoint=checkpoint)

    def test_cpu_resume_preserves_dropout_and_optimiser_state(self):
        self.check_resume('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is optional')
    def test_cuda_resume_and_causality(self):
        self.check_resume('cuda')
        model = self.model().to('cuda')
        actions = torch.tensor([[2,3,2]],device='cuda')
        numeric = torch.randn(1,3,4,device='cuda')
        before = model(actions,numeric)
        numeric[:,2] += 100
        torch.testing.assert_close(before[:,:2],model(actions,numeric)[:,:2],rtol=0,atol=0)
        self.assertFalse(torch.backends.cuda.flash_sdp_enabled())

    def test_saved_model_roundtrip_and_tamper_detection(self):
        model = self.model(dropout=.1)
        data = sq.make_sequences(self.examples,self.encoder)
        expected = tr.predict(model,data)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            tr.save_model(directory,model,self.encoder,CONFIG,'goal',.5,1)
            restored,encoder,info = tr.load_model(directory)
            np.testing.assert_array_equal(expected,tr.predict(restored,data))
            self.assertEqual(encoder.as_dict(),self.encoder.as_dict())
            with (directory/'weights.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                tr.load_model(directory)

    def test_search_covers_values_and_valid_attention_dimensions(self):
        candidates = ex.candidate_configs()
        self.assertEqual(len(candidates),24)
        self.assertEqual(candidates,ex.candidate_configs())
        for name,values in ex.SPACE.items():
            self.assertEqual({c[name] for c in candidates},set(values))
        self.assertTrue(all(c['d_model'] % c['num_heads'] == 0 for c in candidates))

    def test_selection_and_fresh_refit_use_the_correct_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaisesRegex(ValueError,'test inputs'):
                ex.select_model(examples(seasons=(2026,)),'goal','history',directory)
            data = self.examples.copy()
            data.loc[data.season.eq(2025),'action_type'] = 'validation_only'
            winner = ex.select_model(data,'goal','history',directory,candidates=[CONFIG],stages=((2,1),))
            search = pd.read_parquet(directory/'search.parquet')
            self.assertEqual(search.validation_bce.min(),winner['validation_bce'])
            self.assertNotIn('validation_only',tr.load_model(directory/'selection')[1].categories)
            with patch.object(ex,'train_candidate',wraps=tr.train_candidate) as fit:
                record = ex.refit_model(data,'goal','history',winner,directory)
            self.assertIsNone(fit.call_args.args[1])
            self.assertEqual(record['epochs'],winner['best_epoch'])
            self.assertEqual(record['config']['lambda_l2'],winner['lambda_l2'])
            self.assertIn('validation_only',tr.load_model(directory/'refit')[1].categories)

    def test_orchestration_freezes_both_models_before_test_and_resume_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'src/afl_opposition'; source.mkdir(parents=True)
            for name in ex.SOURCE_NAMES:
                (source/name).write_text('test source')
            previous = root/'artifacts/lstm-2026'
            for target in ['shot','goal']:
                for sub,names in [('history/test',['predictions']),('comparison',['metrics','confusion','calibration'])]:
                    directory = previous/target/sub; directory.mkdir(parents=True)
                    for name in names:
                        (directory/f'{name}.parquet').touch()
            (previous/'summary.json').write_text(json.dumps({'artifact_sha256':{}}))
            calls=[]
            def load(directory,seasons):
                calls.append(seasons)
                if seasons == (2026,):
                    for target in ['shot','goal']:
                        self.assertTrue((root/f'artifacts/transformer-2026/{target}/history/refit.json').is_file())
                return (*fixture(seasons),{'sources':{}})
            with patch.object(ex,'load_goal_inputs',side_effect=load), \
                 patch.object(ex,'compare_previous',side_effect=lambda r,t,reports: {n:reports['history'][n] for n in ['metrics','confusion','calibration']}), \
                 redirect_stdout(io.StringIO()):
                result = ex.run_experiment(root,device='cpu',candidates=[CONFIG],stages=((1,1),))
                resumed = ex.run_experiment(root,device='cpu',resume=True,candidates=[CONFIG],stages=((1,1),))
            self.assertEqual(result,resumed)
            self.assertEqual(calls,[(2023,2024,2025),(2026,)])
            self.assertTrue(result['both_models_frozen_before_test_loading'])
            (source/'transformer.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                ex.run_experiment(root,device='cpu',resume=True,candidates=[CONFIG],stages=((1,1),))


if __name__ == '__main__':
    unittest.main()
