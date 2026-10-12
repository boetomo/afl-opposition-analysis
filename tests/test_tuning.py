"""Validation selection, fixed-lambda refitting and held-out test isolation."""

from math import isinf
from unittest.mock import patch
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from afl_opposition.preprocess import FEATURE_COLUMNS
from afl_opposition.tuning import evaluate_test, lambda_to_C, refit_selected, select_regularisation
from test_seasons import seasons_fixture


def development_fixture():
    events, possessions = seasons_fixture()
    events = events.loc[events.season.ne(2026)].copy()
    return events, possessions.loc[possessions.match_id.isin(events.match_id)].copy()


class RegularisationTests(unittest.TestCase):
    def test_C_preserves_lambda_when_weight_sum_changes(self):
        for S in [10, 20, 10000]:
            C = lambda_to_C(.01, S)
            self.assertAlmostEqual(1 / (2 * C * S), .01)
        self.assertEqual(lambda_to_C(.01, 20), lambda_to_C(.01, 10) / 2)
        self.assertTrue(isinf(lambda_to_C(0, 10)))

    def test_invalid_lambda_and_weight_sum_are_rejected(self):
        for value, S in [(-1, 1), (float("inf"), 1), (float("nan"), 1), (.1, 0), (.1, float("inf"))]:
            with self.subTest(value=value, S=S), self.assertRaises(ValueError):
                lambda_to_C(value, S)

    def test_test_season_cannot_enter_selection(self):
        with self.assertRaises(ValueError):
            select_regularisation(*seasons_fixture(), lambdas=(0.01,), refine=False)

    def test_selection_uses_minimum_validation_BCE(self):
        result = select_regularisation(*development_fixture(), lambdas=(0, .001, .1), refine=False)
        winner = result.trials.sort_values(["validation_log_loss", "lambda_l2"], ascending=[True, False]).iloc[0]
        self.assertEqual(result.lambda_l2, winner.lambda_l2)
        self.assertEqual(int(result.trials.selected.sum()), 1)
        self.assertEqual(set(result.baseline.predictions.split), {"train", "validation"})
        self.assertAlmostEqual(result.baseline.pipeline.named_steps["classifier"].C,
                               lambda_to_C(result.lambda_l2, result.baseline.summary["training_weight_sum"]))

    def test_selection_transform_fits_training_only_and_refit_includes_validation(self):
        events, possessions = development_fixture()
        events.loc[events.season.eq(2025), "action_type"] = "validation_only"
        events.loc[events.season.eq(2025), "x_norm"] = 100
        selection = select_regularisation(events, possessions, lambdas=(.01,), refine=False)
        before = selection.baseline.pipeline.named_steps["features"]
        self.assertNotIn("validation_only", before.named_transformers_["action"].categories_[0])
        training = selection.baseline.examples.query("split == 'train'")
        self.assertAlmostEqual(before.named_transformers_["numeric"].mean_[0], float(training.x_norm.mean()))
        refit = refit_selected(selection)
        after = refit.pipeline.named_steps["features"]
        self.assertIn("validation_only", after.named_transformers_["action"].categories_[0])
        self.assertAlmostEqual(after.named_transformers_["numeric"].mean_[0], float(selection.baseline.examples.x_norm.mean()))
        self.assertNotIn("validation_only", before.named_transformers_["action"].categories_[0])
        self.assertEqual(refit.summary["lambda_l2"], selection.lambda_l2)
        self.assertAlmostEqual(refit.pipeline.named_steps["classifier"].C,
                               lambda_to_C(selection.lambda_l2, refit.summary["training_weight_sum"]))
        self.assertLess(refit.pipeline.named_steps["classifier"].C,
                        selection.baseline.pipeline.named_steps["classifier"].C)

    def test_final_evaluation_never_fits_or_uses_test_rate_as_constant(self):
        selection = select_regularisation(*development_fixture(), lambdas=(.01,), refine=False)
        refit = refit_selected(selection)
        events, possessions = seasons_fixture()
        test = events.loc[events.season.eq(2026)].copy()
        test["action_type"] = "test_only"
        test["x_norm"] = 1000
        test_possessions = possessions.loc[possessions.match_id.isin(test.match_id)].copy()
        classifier = refit.pipeline.named_steps["classifier"]
        before = classifier.coef_.copy()
        with patch.object(refit.pipeline, "fit", side_effect=AssertionError("Test must not fit")):
            result = evaluate_test(refit, test, test_possessions)
        self.assertTrue((classifier.coef_ == before).all())
        self.assertEqual(set(result.predictions.split), {"test"})
        self.assertNotIn("test_only", refit.pipeline.named_steps["features"].named_transformers_["action"].categories_[0])
        self.assertTrue(result.predictions.constant_probability.eq(refit.constant_probability).all())
        expected = refit.pipeline.predict_proba(result.examples[FEATURE_COLUMNS])[:, 1]
        self.assertTrue((result.predictions.shot_probability.to_numpy() == expected).all())

    def test_test_outcomes_do_not_change_frozen_coefficients_or_lambda(self):
        selection = select_regularisation(*development_fixture(), lambdas=(.01,), refine=False)
        refit = refit_selected(selection)
        events, possessions = seasons_fixture()
        test = events.loc[events.season.eq(2026)].copy()
        test_possessions = possessions.loc[possessions.match_id.isin(test.match_id)].copy()
        first = evaluate_test(refit, test, test_possessions)
        # Swap which synthetic possession has a shot, keeping consistent metadata.
        ids = test_possessions.possession_id.tolist()
        test["is_explicit_shot"] = test.possession_id.eq(ids[0]) & test.event_index.eq(3)
        test_possessions["explicit_shot_count"] = test_possessions.possession_id.eq(ids[0]).astype(int)
        second = evaluate_test(refit, test, test_possessions)
        assert_frame_equal(first.coefficients, second.coefficients)
        self.assertEqual(first.summary["lambda_l2"], second.summary["lambda_l2"])
        self.assertEqual(first.summary["C"], second.summary["C"])

    def test_input_development_tables_are_preserved(self):
        events, possessions = development_fixture()
        original_events, original_possessions = events.copy(deep=True), possessions.copy(deep=True)
        select_regularisation(events, possessions, lambdas=(.01,), refine=False)
        assert_frame_equal(events, original_events)
        assert_frame_equal(possessions, original_possessions)


if __name__ == "__main__":
    unittest.main()
