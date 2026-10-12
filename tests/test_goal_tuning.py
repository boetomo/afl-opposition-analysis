"""Goal-target selection, full refitting and frozen test evaluation."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import log_loss

from afl_opposition.baseline import _evaluate
from afl_opposition.goal_tuning import evaluate_goal_test, run_experiment, select_goal_regularisation
from afl_opposition.preprocess import FEATURE_COLUMNS, preprocess_events
from afl_opposition.tuning import (
    _probability_report, evaluate_test, lambda_to_C, refit_selected,
)
from test_goal_labels import row


def fixture(seasons=(2023, 2024, 2025)):
    records = []
    for season in seasons:
        for index, outcome in enumerate(["Goal", "Behind", "Out of Bounds"]):
            for order, description in enumerate(["Gather", "Handball", "Kick", outcome], start=1):
                records.append(row(order, description, match_id=f"M{season}-{index}",
                    date=f"{season}-03-{index+1:02d}", season=season,
                    x=5 * order + index, shot_at_goal=True if order == 3 and index < 2 else None))
    result = preprocess_events(pd.DataFrame(records).convert_dtypes())
    return result.events, result.possessions, result.annotations


class GoalTuningTests(unittest.TestCase):
    def select(self, *args):
        return select_goal_regularisation(*(args or fixture()), lambdas=(0.001, 0.01), refine=False)

    def test_selection_uses_goal_labels_and_lowest_validation_BCE(self):
        selection, labels = self.select()
        winner = selection.trials.sort_values(["validation_log_loss", "lambda_l2"], ascending=[True, False]).iloc[0]
        self.assertEqual(selection.lambda_l2, winner.lambda_l2)
        self.assertEqual(int(selection.trials.selected.sum()), 1)
        behind_ids = labels.possession_audit.loc[labels.possession_audit.end_reason.eq("Behind"), "possession_id"]
        self.assertTrue(selection.baseline.examples.loc[selection.baseline.examples.possession_id.isin(behind_ids), "target"].eq(0).all())
        predictions = selection.baseline.predictions.query("split == 'validation'")
        expected = log_loss(predictions.target, predictions.goal_probability, sample_weight=predictions.sample_weight)
        self.assertAlmostEqual(expected, winner.validation_log_loss)
        self.assertEqual(selection.baseline.summary["prediction_target"], "goal")
        self.assertNotIn("shot_probability", predictions)
        self.assertIn("observed_goal_rate", selection.baseline.metrics)
        self.assertIn("predicted_goal_rate", selection.baseline.confusion)

    def test_test_inputs_rejected_before_labelling(self):
        for contaminated in [fixture((2023, 2024, 2025, 2026)), (*fixture()[:2], fixture((2023, 2024, 2025, 2026))[2])]:
            with patch("afl_opposition.goal_tuning.build_goal_examples", side_effect=AssertionError("Too late")):
                with self.assertRaisesRegex(ValueError, "test inputs are forbidden"):
                    self.select(*contaminated)

    def test_training_transform_isolated_then_refitted_with_validation(self):
        events, possessions, annotations = fixture()
        events.loc[events.season.eq(2025), "action_type"] = "validation_only"
        events.loc[events.season.eq(2025), "x_norm"] = 100
        selection, _ = self.select(events, possessions, annotations)
        initial = selection.baseline.pipeline.named_steps["features"]
        self.assertNotIn("validation_only", initial.named_transformers_["action"].categories_[0])
        training = selection.baseline.examples.query("split == 'train'")
        self.assertAlmostEqual(initial.named_transformers_["numeric"].mean_[0], float(training.x_norm.mean()))
        refit = refit_selected(selection)
        final = refit.pipeline.named_steps["features"]
        self.assertIn("validation_only", final.named_transformers_["action"].categories_[0])
        self.assertAlmostEqual(final.named_transformers_["numeric"].mean_[0], float(selection.baseline.examples.x_norm.mean()))
        self.assertAlmostEqual(refit.pipeline.named_steps["classifier"].C,
                               lambda_to_C(selection.lambda_l2, selection.baseline.examples.sample_weight.sum()))
        self.assertEqual(refit.summary["lambda_l2"], selection.lambda_l2)
        self.assertLess(refit.summary["C"], selection.baseline.summary["C"])
        self.assertEqual(refit.summary["prediction_target"], "goal")

    def test_evaluation_does_not_fit_or_update_the_development_constant(self):
        selection, _ = self.select()
        refit = refit_selected(selection)
        events, possessions, annotations = fixture((2026,))
        events["action_type"] = "test_only"
        before = pickle.dumps(refit.pipeline)
        with patch.object(refit.pipeline, "fit", side_effect=AssertionError("No test fitting")):
            report, _ = evaluate_goal_test(refit, events, possessions, annotations)
        self.assertEqual(before, pickle.dumps(refit.pipeline))
        self.assertTrue(report.predictions.constant_probability.eq(refit.constant_probability).all())
        self.assertNotIn("test_only", refit.pipeline.named_steps["features"].named_transformers_["action"].categories_[0])
        self.assertEqual(set(report.predictions.split), {"test"})

    def test_changing_test_outcomes_does_not_change_predictions_or_lambda(self):
        selection, _ = self.select()
        refit = refit_selected(selection)
        events, possessions, annotations = fixture((2026,))
        first, _ = evaluate_goal_test(refit, events, possessions, annotations)
        possessions["end_reason"] = possessions.end_reason.replace({"Goal": "Behind", "Behind": "Goal"})
        annotations["description"] = annotations.description.replace({"Goal": "Behind", "Behind": "Goal"})
        second, _ = evaluate_goal_test(refit, events, possessions, annotations)
        assert_frame_equal(first.predictions.drop(columns="target"), second.predictions.drop(columns="target"))
        assert_frame_equal(first.coefficients, second.coefficients)
        self.assertFalse(first.predictions.target.equals(second.predictions.target))
        self.assertEqual(first.summary["lambda_l2"], second.summary["lambda_l2"])

    def test_target_specific_evaluators_reject_the_other_model(self):
        selection, _ = self.select()
        refit = refit_selected(selection)
        events, possessions, annotations = fixture((2026,))
        with self.assertRaisesRegex(ValueError, "Use goal evaluation"):
            evaluate_test(refit, events, possessions)
        refit.summary["prediction_target"] = "shot"
        with self.assertRaisesRegex(ValueError, "goal labels"):
            evaluate_goal_test(refit, events, possessions, annotations)

    def test_shared_probability_metrics_preserve_shot_scoring(self):
        selection, _ = self.select()
        predictions = selection.baseline.predictions.rename(columns={"goal_probability": "shot_probability"})
        for expected, actual in zip(_evaluate(predictions), _probability_report(predictions, "shot")):
            assert_frame_equal(expected, actual)

    def test_pipeline_is_frozen_before_test_loading_and_artifacts_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/afl_opposition"
            source.mkdir(parents=True)
            for name in ["action_rules.py", "preprocess.py", "examples.py", "goal_labels.py",
                         "evaluation.py", "baseline.py", "tuning.py", "goal_tuning.py"]:
                (source / name).write_text("synthetic code provenance")
            shot_marker = root / "artifacts/logistic-tuned-2026/keep.txt"
            shot_marker.parent.mkdir(parents=True)
            shot_marker.write_text("previous shot results")
            calls = []

            def load_inputs(directory, seasons):
                calls.append(seasons)
                if seasons == (2026,):
                    experiment = root / "artifacts/logistic-goal-tuned-2026"
                    self.assertTrue((experiment / "refit_model.pkl").is_file())
                    frozen = json.loads((experiment / "refit.json").read_text())
                    self.assertTrue(frozen["test_loaded_after_refit"])
                    self.assertEqual(frozen["prediction_target"], "goal")
                    self.assertFalse((experiment / "test/summary.json").exists())
                return (*fixture(seasons), {"sources": {}})

            def small_search(events, possessions, annotations, **kwargs):
                return select_goal_regularisation(events, possessions, annotations,
                    lambdas=(0.001, 0.01), refine=False)

            with patch("afl_opposition.goal_tuning.load_goal_inputs", side_effect=load_inputs), \
                 patch("afl_opposition.goal_tuning.select_goal_regularisation", side_effect=small_search), \
                 redirect_stdout(io.StringIO()):
                _, refit, report = run_experiment(root)
            self.assertEqual(calls, [(2023, 2024, 2025), (2026,)])
            self.assertEqual(shot_marker.read_text(), "previous shot results")
            directory = root / "artifacts/logistic-goal-tuned-2026"
            with (directory / "refit_model.pkl").open('rb') as stream:
                restored = pickle.load(stream)
            self.assertTrue((restored.predict_proba(report.examples[FEATURE_COLUMNS])[:, 1]
                             == report.predictions.goal_probability.to_numpy()).all())
            summary = json.loads((directory / "test/summary.json").read_text())
            self.assertEqual(summary["lambda_l2"], refit.summary["lambda_l2"])
            self.assertTrue((directory / "test_labels/goal_annotation_audit.parquet").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_experiment(root)


if __name__ == "__main__":
    unittest.main()
