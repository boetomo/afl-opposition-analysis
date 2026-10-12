"""Temporal isolation, weighted selection, refitting and model persistence."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import log_loss
import xgboost as xgb

from afl_opposition import xgboost_baseline as xb
from test_goal_tuning import fixture

SMALL_GRID = (dict(max_depth=1, eta=.1, reg_lambda=0.), dict(max_depth=2, eta=.2, reg_lambda=1.))


def small_select(dataset, target_name, **kwargs):
    with patch.dict(xb.FIXED_PARAMS, min_child_weight=.01):
        return xb.select_model(dataset, target_name, grid=SMALL_GRID, max_rounds=12, patience=3)


class XGBoostTests(unittest.TestCase):
    def datasets(self, seasons=(2023, 2024, 2025)):
        return xb.build_datasets(*fixture(seasons), development=2026 not in seasons)

    def select(self, target="goal"):
        return small_select(self.datasets()[target], target)

    def test_test_inputs_rejected_before_label_construction(self):
        for inputs in (fixture((2023, 2024, 2025, 2026)), (*fixture()[:2], fixture((2026,))[2])):
            with patch.object(xb, "build_examples", side_effect=AssertionError("Too late")):
                with self.assertRaisesRegex(ValueError, "test inputs are forbidden"):
                    xb.build_datasets(*inputs, development=True)
        dataset = self.datasets()["goal"]
        dataset.examples.loc[dataset.examples.index[0], "season"] = 2026
        with self.assertRaisesRegex(ValueError, "test inputs are forbidden"):
            small_select(dataset, "goal")

    def test_both_targets_keep_same_prefixes_but_different_labels(self):
        data = self.datasets()
        assert_frame_equal(data["shot"].examples.drop(columns="target"), data["goal"].examples.drop(columns="target"))
        self.assertTrue(data["shot"].examples.target.ge(data["goal"].examples.target).all())
        self.assertFalse(data["shot"].examples.target.equals(data["goal"].examples.target))

    def test_best_iteration_and_weighted_validation_score(self):
        selection = self.select()
        validation = selection.examples.query("split == 'validation'")
        p = selection.model.predict_proba(validation)[:, 1]
        expected = log_loss(validation.target, p, sample_weight=validation.sample_weight)
        self.assertAlmostEqual(expected, selection.winner["validation_log_loss"])
        best_trial = selection.trials.sort_values(["validation_log_loss", "rounds", "max_depth", "reg_lambda", "trial"],
                                                   ascending=[True, True, True, False, True]).iloc[0]
        self.assertEqual(selection.winner["trial"], best_trial.trial)
        self.assertEqual(selection.model.booster.num_boosted_rounds(), selection.model.booster.best_iteration + 1)
        curve = selection.history.query("split == 'validation' and trial == @best_trial.trial")
        self.assertAlmostEqual(expected, curve.loc[curve["round"].eq(selection.winner["rounds"]), "log_loss"].iloc[0], places=6)
        np.testing.assert_allclose(xb._matrix(selection.model.transformer, validation).get_weight(), validation.sample_weight)

    def test_early_stopping_discards_later_trees(self):
        # A constant tree cannot improve: the retained booster must be shorter than the attempted history.
        with patch.dict(xb.FIXED_PARAMS, min_child_weight=100000.):
            selection = xb.select_model(self.datasets()["goal"], "goal", grid=SMALL_GRID[:1], max_rounds=20, patience=3)
        self.assertLess(selection.winner["rounds"], selection.winner["attempted_rounds"])
        self.assertEqual(selection.model.booster.num_boosted_rounds(), selection.winner["rounds"])

    def test_transformer_fits_training_only_then_refits_on_development(self):
        dataset = self.datasets()["goal"]
        dataset.examples.loc[dataset.examples.season.eq(2025), "action_type"] = "validation_only"
        selection = small_select(dataset, "goal")
        self.assertNotIn("validation_only", selection.model.transformer.named_transformers_["action"].categories_[0])
        final = xb.refit_model(selection)
        self.assertIn("validation_only", final.transformer.named_transformers_["action"].categories_[0])

    def test_refit_preserves_mean_loss_penalty_and_selected_tree_count(self):
        selection = self.select()
        # Force a nonzero candidate for an informative scale check.
        selection.model.parameters["reg_lambda"] = 10.
        with patch.object(xb.xgb, "train", wraps=xgb.train) as fit:
            final = xb.refit_model(selection)
        self.assertNotIn("evals", fit.call_args.kwargs)
        self.assertNotIn("callbacks", fit.call_args.kwargs)
        ratio = selection.examples.sample_weight.sum() / selection.training_weight_sum
        self.assertAlmostEqual(final.parameters["reg_lambda"], 10 * ratio)
        self.assertAlmostEqual(final.parameters["min_child_weight"], selection.model.parameters["min_child_weight"] * ratio)
        self.assertEqual(final.booster.num_boosted_rounds(), selection.winner["rounds"])

    def test_test_labels_do_not_change_predictions_or_frozen_model(self):
        final = xb.refit_model(self.select())
        data = self.datasets((2026,))["goal"].examples
        data["action_type"] = "unseen_test_action"
        fingerprint = final.fingerprint()
        with patch.object(xb.xgb, "train", side_effect=AssertionError("No fitting on test")):
            first = xb.report(final, data, "test")
            data["target"] = 1 - data.target
            second = xb.report(final, data, "test")
        assert_frame_equal(first["predictions"].drop(columns="target"), second["predictions"].drop(columns="target"))
        self.assertEqual(final.fingerprint(), fingerprint)
        self.assertTrue(first["predictions"].constant_probability.eq(final.constant_probability).all())
        self.assertEqual(set(first["confusion"].weighting), {"unweighted", "possession_weighted"})
        self.assertIn("predicted_goal_rate", first["confusion"])

    def test_saved_model_roundtrip_and_tamper_detection(self):
        final = xb.refit_model(self.select())
        data = self.datasets((2026,))["goal"].examples
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            final.save(directory)
            restored = xb.load_model(directory)
            np.testing.assert_array_equal(final.predict_proba(data), restored.predict_proba(data))
            with (directory / "model.ubj").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksums"):
                xb.load_model(directory)

    def test_comparison_checks_exact_test_population(self):
        final = xb.refit_model(self.select())
        tables = xb.report(final, self.datasets((2026,))["goal"].examples, "test")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "artifacts/logistic-goal-tuned-2026/test"
            old = {name: table.replace({"model": {"xgboost": "logistic"}}) if "model" in table else table.copy()
                   for name, table in tables.items()}
            xb._save_tables(directory, old)
            comparison = xb.compare_logistic(root, "goal", tables)
            self.assertEqual(set(comparison["metrics"].model), {"constant", "logistic", "xgboost"})
            old["predictions"].loc[0, "target"] = 1 - old["predictions"].loc[0, "target"]
            old["predictions"].to_parquet(directory / "predictions.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "examples differ"):
                xb.compare_logistic(root, "goal", tables)

    def test_both_models_frozen_before_test_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "src/afl_opposition"
            source.mkdir(parents=True)
            for name in xb.SOURCE_NAMES:
                (source / name).write_text("synthetic code provenance")
            for name in ("logistic-tuned-2026", "logistic-goal-tuned-2026"):
                directory = root / "artifacts" / name / "test"
                directory.mkdir(parents=True)
                for filename in ("predictions", "metrics", "confusion", "calibration"):
                    (directory / f"{filename}.parquet").touch()
            calls = []
            def load_inputs(directory, seasons):
                calls.append(seasons)
                if seasons == (2026,):
                    for target in ("shot", "goal"):
                        self.assertTrue((root / f"artifacts/xgboost-2026/{target}/refit.json").is_file())
                        self.assertTrue((root / f"artifacts/xgboost-2026/{target}/refit/model.ubj").is_file())
                    self.assertFalse((root / "artifacts/xgboost-2026/summary.json").exists())
                return (*fixture(seasons), {"sources": {}})
            original_select = xb.select_model
            def fast_search(dataset, target, **kwargs):
                return original_select(dataset, target, grid=SMALL_GRID[:1], max_rounds=3, patience=2)
            with patch.object(xb, "load_goal_inputs", side_effect=load_inputs), \
                 patch.object(xb, "select_model", side_effect=fast_search), \
                 patch.object(xb, "compare_logistic", side_effect=lambda r, t, tables: {k: v for k, v in tables.items() if k != "predictions"}), \
                 redirect_stdout(io.StringIO()):
                models = xb.run_experiment(root)
            self.assertEqual(calls, [(2023, 2024, 2025), (2026,)])
            self.assertEqual(set(models), {"shot", "goal"})
            summary = json.loads((root / "artifacts/xgboost-2026/summary.json").read_text())
            self.assertTrue(summary["frozen_models_unchanged"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                xb.run_experiment(root)


if __name__ == "__main__":
    unittest.main()
