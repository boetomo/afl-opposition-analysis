"""Causality, the history ablation, weighting, checkpoints and test isolation."""
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
from afl_opposition import lstm_experiment as ex
from test_goal_tuning import fixture

CONFIG = dict(num_layers=1, hidden_size=8, dropout=.1, embedding_dim=4, learning_rate=.003, lambda_l2=1e-5)


def examples(target="goal", seasons=(2023, 2024, 2025)):
    return ex.build_datasets(*fixture(seasons), development=2026 not in seasons)[target].examples


class SequenceTests(unittest.TestCase):
    def setUp(self):
        sq.configure_runtime("cpu")
        self.examples = examples()
        self.encoder = sq.SequenceEncoder.fit(self.examples.loc[self.examples.season.lt(2025)])

    def model(self, variant="history", config=None):
        return sq.PossessionLSTM(len(self.encoder.categories)+2, config or CONFIG, variant).eval()

    def test_encoder_fits_development_only_and_handles_unknown_actions(self):
        self.assertNotIn("unknown", self.encoder.categories)
        changed = self.examples.copy()
        changed["action_type"] = "unknown"
        actions, numeric = self.encoder.transform(changed)
        self.assertTrue((actions == 1).all())
        self.assertEqual(numeric.shape[1], 4)
        with self.assertRaisesRegex(ValueError, "test seasons"):
            sq.SequenceEncoder.fit(examples(seasons=(2026,)))

    def test_sequence_rows_restore_original_order_and_weights(self):
        shuffled = self.examples.sample(frac=1, random_state=3).reset_index(drop=True)
        data = sq.make_sequences(shuffled, self.encoder)
        np.testing.assert_array_equal(np.sort(data.row_map[data.row_map>=0]), np.arange(len(shuffled)))
        model = self.model("no_history")
        p = sq.predict(model, data)
        for i in range(len(shuffled)):
            row = shuffled.iloc[[i]].copy()
            row["event_index"] = 0
            alone = sq.make_sequences(row, self.encoder, labelled=False)
            self.assertAlmostEqual(p[i], sq.predict(model, alone)[0], places=6)
        self.assertEqual(len(p), len(shuffled))
        self.assertEqual(sum(len(b) for b in data.batches(2, seed=1)), len(data.lengths))

    def test_noncontiguous_prefixes_and_wrong_weights_are_rejected(self):
        broken = self.examples.copy()
        broken.loc[0,"event_index"] = 99
        with self.assertRaisesRegex(ValueError,"contiguous"):
            sq.make_sequences(broken,self.encoder)
        broken = self.examples.copy()
        broken["sample_weight"] = 1.
        with self.assertRaisesRegex(ValueError,"weight one"):
            sq.make_sequences(broken,self.encoder)

    def test_loss_ignores_padding_and_gives_each_possession_weight_one(self):
        logits=torch.tensor([[0.,1.,999.],[-1.,-2.,-3.]],requires_grad=True)
        y=torch.tensor([1.,0.]); lengths=torch.tensor([2,3])
        losses=sq.per_possession_bce(logits,y,lengths)
        expected=[log_loss([1,1],torch.sigmoid(logits[0,:2]).detach().numpy().astype(float),labels=[0,1]),
                  log_loss([0,0,0],torch.sigmoid(logits[1]).detach().numpy().astype(float),labels=[0,1])]
        np.testing.assert_allclose(losses.detach().numpy(),expected,rtol=1e-6)
        losses.mean().backward()
        self.assertEqual(logits.grad[0,2].item(),0.)

    def test_future_events_cannot_change_earlier_predictions(self):
        model=self.model(config={**CONFIG,"num_layers":2})
        actions=torch.tensor([[2,3,2,3]])
        numeric=torch.randn(1,4,4)
        before=model(actions,numeric)
        numeric[:,2:]+=100
        actions[:,2:]=4
        after=model(actions,numeric)
        torch.testing.assert_close(before[:,:2],after[:,:2],rtol=0,atol=0)

    def test_no_history_is_independent_of_preceding_events(self):
        actions=torch.tensor([[2,3,2]])
        numeric=torch.randn(1,3,4)
        for variant in ["no_history","history"]:
            model=self.model(variant)
            before=model(actions,numeric)[:,-1]
            changed=numeric.clone(); changed[:,0]+=10
            after=model(actions,changed)[:,-1]
            if variant=="no_history":
                torch.testing.assert_close(before,after,rtol=0,atol=0)
            else:
                self.assertGreater(float((before-after).abs().max().detach()),1e-7)

    def test_padding_and_other_possessions_do_not_change_outputs(self):
        model=self.model()
        a=torch.tensor([[2,3]]); x=torch.randn(1,2,4)
        before=model(a,x)
        padded_a=torch.cat([a,torch.zeros(1,3,dtype=torch.long)],dim=1)
        padded_x=torch.cat([x,torch.zeros(1,3,4)],dim=1)
        extended=model(padded_a,padded_x)[:,:2]
        torch.testing.assert_close(before,extended,rtol=1e-6,atol=1e-7)
        batch=model(torch.cat([a,a]),torch.cat([x,x+10]))[:1]
        torch.testing.assert_close(before,batch,rtol=1e-6,atol=1e-7)

    def test_dropout_is_active_for_one_and_multiple_layers(self):
        for layers in [1,2,3]:
            model=self.model(config={**CONFIG,"num_layers":layers,"dropout":.4})
            self.assertEqual(model.dropout.p,.4)
            self.assertEqual(model.lstm.dropout,.4 if layers>1 else 0.)

    def test_l2_regularises_weights_but_not_biases(self):
        model=self.model()
        penalty=float(sq.l2_penalty(model).detach())
        with torch.no_grad():
            model.head.bias.add_(100.)
        self.assertEqual(float(sq.l2_penalty(model).detach()),penalty)

    def test_resumed_training_matches_uninterrupted_training(self):
        train=sq.make_sequences(self.examples.loc[self.examples.season.lt(2025)],self.encoder)
        val=sq.make_sequences(self.examples.loc[self.examples.season.eq(2025)],self.encoder)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"candidate.pt"
            full,_,_=sq.train_candidate(train,val,CONFIG,"history",len(self.encoder.categories)+2,epochs=4,batch_size=3)
            sq.train_candidate(train,val,CONFIG,"history",len(self.encoder.categories)+2,epochs=2,batch_size=3,checkpoint=path)
            resumed,_,_=sq.train_candidate(train,val,CONFIG,"history",len(self.encoder.categories)+2,epochs=4,batch_size=3,checkpoint=path)
            self.assertEqual(sq.state_fingerprint(full),sq.state_fingerprint(resumed))

    def test_save_load_and_test_labels_do_not_change_predictions(self):
        model=self.model()
        data=sq.make_sequences(examples(seasons=(2026,)),self.encoder)
        p=sq.predict(model,data)
        fingerprint=sq.state_fingerprint(model)
        data.target=1-data.target
        np.testing.assert_array_equal(p,sq.predict(model,data))
        self.assertEqual(fingerprint,sq.state_fingerprint(model))
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            sq.save_model(directory,model,self.encoder,CONFIG,"goal",.5,1)
            restored,encoder,info=sq.load_model(directory)
            np.testing.assert_array_equal(p,sq.predict(restored,data))
            self.assertEqual(encoder.as_dict(),self.encoder.as_dict())
            with (directory/"weights.pt").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError,"checksum"):
                sq.load_model(directory)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is optional")
    def test_cuda_resume_and_causality(self):
        sq.configure_runtime("cuda")
        train=sq.make_sequences(self.examples.loc[self.examples.season.lt(2025)],self.encoder).to("cuda")
        val=sq.make_sequences(self.examples.loc[self.examples.season.eq(2025)],self.encoder).to("cuda")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"candidate.pt"
            full,_,_=sq.train_candidate(train,val,CONFIG,"history",len(self.encoder.categories)+2,epochs=3)
            sq.train_candidate(train,val,CONFIG,"history",len(self.encoder.categories)+2,epochs=1,checkpoint=path)
            resumed,_,_=sq.train_candidate(train,val,CONFIG,"history",len(self.encoder.categories)+2,epochs=3,checkpoint=path)
            self.assertEqual(sq.state_fingerprint(full),sq.state_fingerprint(resumed))
            a=torch.tensor([[2,3,2]],device="cuda"); x=torch.randn(1,3,4,device="cuda")
            before=resumed(a,x)
            x[:,2]+=100
            torch.testing.assert_close(before[:,:2],resumed(a,x)[:,:2],rtol=0,atol=0)
        self.assertFalse(torch.backends.cudnn.enabled)

    def test_search_covers_each_requested_hyperparameter(self):
        candidates=ex.candidate_configs()
        self.assertEqual(len(candidates),24)
        self.assertEqual(candidates,ex.candidate_configs())
        for name, values in ex.SPACE.items():
            self.assertEqual({c[name] for c in candidates},set(values))

    def test_search_rejects_test_and_refits_encoder_without_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with self.assertRaisesRegex(ValueError,"test inputs"):
                ex.select_model(examples(seasons=(2026,)),"goal","history",directory)
            data=self.examples.copy()
            data.loc[data.season.eq(2025),"action_type"]="validation_only"
            winner=ex.select_model(data,"goal","history",directory,candidates=[CONFIG,{**CONFIG,"hidden_size":4}],stages=((1,1),(2,1)))
            trials=pd.read_parquet(directory/"search.parquet")
            self.assertAlmostEqual(winner["validation_bce"],trials.validation_bce.min())
            selected_encoder=json.loads((directory/"selection.json").read_text())["encoder"]
            self.assertNotIn("validation_only",selected_encoder["categories"])
            with patch.object(ex,"train_candidate",wraps=sq.train_candidate) as fit:
                refit=ex.refit_model(data,"goal","history",winner,directory)
            self.assertIsNone(fit.call_args.args[1])
            self.assertEqual(refit["epochs"],winner["best_epoch"])
            self.assertEqual(refit["config"]["lambda_l2"],winner["lambda_l2"])
            self.assertIn("validation_only",sq.load_model(directory/"refit")[1].categories)

    def test_orchestration_freezes_all_four_models_before_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"src/afl_opposition"; source.mkdir(parents=True)
            for name in ex.SOURCE_NAMES:
                (source/name).write_text("test source provenance")
            for target in ["shot","goal"]:
                for sub,files in [("test",["predictions"]),("comparison",["metrics","confusion","calibration"])]:
                    d=root/"artifacts/xgboost-2026"/target/sub; d.mkdir(parents=True)
                    for name in files:
                        (d/f"{name}.parquet").touch()
            calls=[]
            def load_inputs(directory,seasons):
                calls.append(seasons)
                if seasons==(2026,):
                    for target in ["shot","goal"]:
                        for variant in ex.VARIANTS:
                            self.assertTrue((root/f"artifacts/lstm-2026/{target}/{variant}/refit.json").is_file())
                return (*fixture(seasons),{"sources":{}})
            with patch.object(ex,"load_goal_inputs",side_effect=load_inputs), \
                 patch.object(ex,"compare_previous",side_effect=lambda r,t,reports: {n:pd.concat([v[n] for v in reports.values()]) for n in ["metrics","confusion","calibration"]}), \
                 redirect_stdout(io.StringIO()):
                result=ex.run_experiment(root,device="cpu",candidates=[CONFIG],stages=((1,1),))
                resumed=ex.run_experiment(root,device="cpu",resume=True,candidates=[CONFIG],stages=((1,1),))
            self.assertEqual(calls,[(2023,2024,2025),(2026,)])
            self.assertEqual(result,resumed)
            self.assertTrue(result["all_four_models_frozen_before_test_loading"])
            with self.assertRaisesRegex(ValueError,"already exists"):
                ex.run_experiment(root,device="cpu",candidates=[CONFIG],stages=((1,1),))


if __name__=="__main__":
    unittest.main()
