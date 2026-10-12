"""Mixed-batch gradients, update schedules, seed selection and test isolation."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import torch

from afl_opposition import sequence as sq
from afl_opposition import transformer_training as tr
from afl_opposition import transformer_retraining as ex
from test_sequence import examples
from test_goal_tuning import fixture

CONFIG = dict(num_layers=1,d_model=16,num_heads=2,ff_multiplier=2,embedding_dim=4,
              continuous_dim=8,dropout=.1,learning_rate=.003,lambda_l2=1e-5,warmup_fraction=.1)


class RetrainingTests(unittest.TestCase):
    def setUp(self):
        tr.configure_runtime('cpu')
        self.examples = examples()
        self.encoder = sq.SequenceEncoder.fit(self.examples.loc[self.examples.season.lt(2025)])
        self.train = sq.make_sequences(self.examples.loc[self.examples.season.lt(2025)],self.encoder)
        self.validation = sq.make_sequences(self.examples.loc[self.examples.season.eq(2025)],self.encoder)

    def test_random_batches_cover_every_possession_and_reproduce(self):
        lengths = np.arange(1,1001)%47+1
        batches = tr.batch_indices(lengths,64,37)
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)),np.arange(1000))
        np.testing.assert_array_equal(np.concatenate(batches),np.concatenate(tr.batch_indices(lengths,64,37)))
        self.assertFalse(np.array_equal(np.concatenate(batches),np.concatenate(tr.batch_indices(lengths,64,38))))
        self.assertGreater(min(np.ptp(lengths[b]) for b in batches),30)
        self.assertEqual(len(batches[-1]),40)

    def test_microbatches_only_reorder_the_same_effective_batch(self):
        lengths = np.array([3,20,1,8,2,5,30])
        indices = np.array([5,2,6,0,3])
        micro = tr.microbatch_indices(indices,lengths,2)
        np.testing.assert_array_equal(np.sort(np.concatenate(micro)),np.sort(indices))
        self.assertEqual(list(map(len,micro)),[2,2,1])

    def test_accumulation_matches_full_batch_including_l2_and_partial_microbatch(self):
        config = {**CONFIG,'dropout':0.}
        full = tr.PossessionTransformer(len(self.encoder.categories)+2,config)
        accumulated = tr.PossessionTransformer(len(self.encoder.categories)+2,config)
        accumulated.load_state_dict(full.state_dict())
        indices = np.arange(min(5,len(self.train.lengths)))
        # SGD makes the gradient-equivalence assertion directly observable.
        first = torch.optim.SGD(full.parameters(),lr=.01)
        second = torch.optim.SGD(accumulated.parameters(),lr=.01)
        tr.training_step(full,first,self.train,indices,config,microbatch_size=100)
        tr.training_step(accumulated,second,self.train,indices,config,microbatch_size=2)
        for a,b in zip(full.parameters(),accumulated.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)

    def test_schedule_reaches_floor_and_refit_replays_identical_prefix(self):
        rates=[tr.learning_rate_at(i,100,CONFIG) for i in range(1,101)]
        self.assertAlmostEqual(rates[0],CONFIG['learning_rate']/10)
        self.assertAlmostEqual(rates[9],CONFIG['learning_rate'])
        self.assertAlmostEqual(rates[-1],.05*CONFIG['learning_rate'])
        self.assertTrue(all(a>=b for a,b in zip(rates[10:],rates[11:])))
        self.assertEqual(rates[:63],[tr.learning_rate_at(i,100,CONFIG) for i in range(1,64)])
        no_warmup={**CONFIG,'warmup_fraction':0.}
        self.assertAlmostEqual(tr.learning_rate_at(1,100,no_warmup),CONFIG['learning_rate'])
        with self.assertRaises(ValueError):
            tr.learning_rate_at(101,100,CONFIG)

    def test_eval_summary_matches_event_weights_and_disables_dropout(self):
        model=tr.PossessionTransformer(len(self.encoder.categories)+2,CONFIG)
        model.train()
        result=tr.evaluate_summary(model,self.train,batch_size=2)
        self.assertFalse(model.training)
        p=tr.predict(model,self.train,batch_size=2)
        table=self.examples.loc[self.examples.season.lt(2025)]
        self.assertAlmostEqual(result['bce'],log_loss(table.target,p,sample_weight=table.sample_weight),places=6)
        self.assertAlmostEqual(result['mean_probability'],np.average(p,weights=table.sample_weight),places=6)
        self.assertAlmostEqual(result['observed_rate'],np.average(table.target,weights=table.sample_weight),places=6)

    def resume_check(self,device):
        train=self.train.to(device); validation=self.validation.to(device)
        kwargs=dict(horizon=8,seed=8,evaluate_every=2,batch_size=3,microbatch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'training.pt'
            complete,_=tr.fit_updates(train,validation,CONFIG,len(self.encoder.categories)+2,steps=8,**kwargs)
            tr.fit_updates(train,validation,CONFIG,len(self.encoder.categories)+2,steps=4,checkpoint=checkpoint,**kwargs)
            resumed,_=tr.fit_updates(train,validation,CONFIG,len(self.encoder.categories)+2,steps=8,checkpoint=checkpoint,**kwargs)
            self.assertEqual(sq.state_fingerprint(complete),sq.state_fingerprint(resumed))
            with self.assertRaisesRegex(ValueError,'contract'):
                tr.fit_updates(train,validation,CONFIG,len(self.encoder.categories)+2,steps=8,
                    checkpoint=checkpoint,**{**kwargs,'microbatch_size':1})

    def test_cpu_resume_preserves_schedule_batch_offset_dropout_and_optimiser(self):
        self.resume_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is optional')
    def test_cuda_resume_preserves_schedule_batch_offset_dropout_and_optimiser(self):
        self.resume_check('cuda')

    def test_saved_retrained_model_roundtrip_and_checksum(self):
        model=tr.PossessionTransformer(len(self.encoder.categories)+2,CONFIG).eval()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            tr.save_model(directory,model,self.encoder,CONFIG,'goal',.5,300,500,2026)
            loaded,encoder,info=tr.load_model(directory)
            np.testing.assert_array_equal(tr.predict(model,self.train),tr.predict(loaded,self.train))
            self.assertEqual(info['steps'],300)
            self.assertEqual(info['schedule_horizon'],500)
            self.assertEqual(encoder.as_dict(),self.encoder.as_dict())
            with (directory/'weights.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                tr.load_model(directory)

    def test_broad_search_covers_all_values_and_valid_head_dimensions(self):
        configs=ex.candidate_configs()
        self.assertEqual(configs,ex.candidate_configs())
        self.assertEqual(len(configs),64)
        self.assertEqual(len({json.dumps(c,sort_keys=True) for c in configs}),64)
        for name,values in ex.SPACE.items():
            self.assertEqual({c[name] for c in configs},set(values))
        self.assertEqual({c['num_heads'] for c in configs},{1,2,4,8})
        self.assertTrue(all(8<=c['d_model']//c['num_heads']<=64 and c['d_model']%c['num_heads']==0 for c in configs))

    def test_promotion_preserves_slow_rate_and_capacity_options(self):
        configs=ex.candidate_configs()
        rows=[dict(trial=i,**c,validation_bce=float(i),parameters=i+1) for i,c in enumerate(configs)]
        screen=pd.DataFrame(rows)
        selected=ex.promote_candidates(screen)
        self.assertEqual(selected[:4],[0,1,2,3])
        self.assertEqual(len(set(selected)),8)
        self.assertTrue(screen.loc[screen.trial.isin(selected)].learning_rate.le(.0003).any())

    def test_selection_uses_common_update_and_mean_across_seeds(self):
        data=self.examples.copy()
        data.loc[data.season.eq(2025),'action_type']='validation_only'
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            winner=ex.select_model(data,'goal',directory,candidates=[CONFIG,{**CONFIG,'learning_rate':.001}],
                screen_steps=4,full_steps=6,evaluate_every=2,seeds=(2026,2027),
                promote_count=2,finalist_count=2,diagnostics=False,progress=None)
            table=pd.read_parquet(directory/'seed_selection.parquet')
            eligible=table.loc[table.step.ge(4)]
            self.assertAlmostEqual(winner['mean_validation_bce'],eligible.validation_bce.min())
            self.assertEqual(winner['seeds'],[2026,2027])
            for seed in winner['seeds']:
                _,encoder,info=tr.load_model(directory/'selection'/f'seed-{seed}')
                self.assertNotIn('validation_only',encoder.categories)
                self.assertEqual(info['steps'],winner['steps'])
            with self.assertRaisesRegex(ValueError,'test inputs'):
                ex.select_model(examples(seasons=(2026,)),'goal',directory)

    def test_refit_uses_exact_selected_updates_and_original_horizon(self):
        winner=dict(config=CONFIG,steps=4,schedule_horizon=8,seeds=[2026])
        safe=dict(bce=.1,mean_probability=.5,observed_rate=.5,calibration_gap=0.,possessions=1)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(ex,'fit_updates',wraps=tr.fit_updates) as fit, \
             patch.object(ex,'evaluate_summary',return_value=safe):
            record=ex.refit_models(self.examples,'goal',winner,Path(tmp),progress=None)
            self.assertIsNone(fit.call_args.args[1])
            self.assertEqual(fit.call_args.kwargs['steps'],4)
            self.assertEqual(fit.call_args.kwargs['horizon'],8)
            self.assertEqual(record['steps'],4)

    def test_bad_refit_calibration_stops_before_freeze(self):
        winner=dict(config=CONFIG,steps=2,schedule_horizon=8,seeds=[2026])
        bad=dict(bce=.1,mean_probability=.7,observed_rate=.5,calibration_gap=.2,possessions=1)
        with tempfile.TemporaryDirectory() as tmp,patch.object(ex,'evaluate_summary',return_value=bad):
            with self.assertRaisesRegex(ValueError,'sanity gate'):
                ex.refit_models(self.examples,'goal',winner,Path(tmp),progress=None)
            self.assertFalse((Path(tmp)/'refit.json').exists())

    def test_orchestration_freezes_all_seeds_for_both_targets_before_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'src/afl_opposition'; source.mkdir(parents=True)
            for name in ex.SOURCE_NAMES:
                (source/name).write_text('fixture source')
            previous=root/'artifacts/transformer-2026'; previous.mkdir(parents=True)
            (previous/'summary.json').write_text(json.dumps({'artifact_sha256':{}}))
            calls=[]
            def load(directory,seasons):
                calls.append(seasons)
                if seasons==(2026,):
                    for target in ['shot','goal']:
                        current=root/'artifacts/transformer-retrained-2026'/target
                        self.assertTrue((current/'refit.json').is_file())
                        for seed in [2026,2027]:
                            self.assertTrue((current/'refit'/f'seed-{seed}'/'model.json').is_file())
                return (*fixture(seasons),{'sources':{}})
            safe=dict(bce=.1,mean_probability=.5,observed_rate=.5,calibration_gap=0.,possessions=1)
            kwargs=dict(device='cpu',candidates=[CONFIG],screen_steps=2,full_steps=4,
                evaluate_every=2,seeds=(2026,2027),promote_count=1,finalist_count=1,diagnostics=False)
            with patch.object(ex,'load_goal_inputs',side_effect=load), \
                 patch.object(ex,'evaluate_summary',return_value=safe), \
                 patch.object(ex,'compare_previous',side_effect=lambda r,t,reports:
                    {n:reports['primary'][n] for n in ['metrics','confusion','calibration']}),redirect_stdout(io.StringIO()):
                result=ex.run_experiment(root,**kwargs)
                resumed=ex.run_experiment(root,resume=True,**kwargs)
            self.assertEqual(result,resumed)
            self.assertEqual(calls,[(2023,2024,2025),(2026,)])
            self.assertTrue(result['all_refits_frozen_before_test_loading'])
            (source/'transformer_training.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                ex.run_experiment(root,resume=True,**kwargs)


if __name__=='__main__':
    unittest.main()
