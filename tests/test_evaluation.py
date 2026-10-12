"""Confusion orientation, threshold semantics, weighting and model isolation."""

from math import isnan
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from afl_opposition.baseline import save_baseline, train_baseline
from afl_opposition.evaluation import confusion_report
from test_baseline import fixture


def predictions():
    return pd.DataFrame({
        "split": ["validation"] * 5, "target": [0, 0, 1, 1, 1],
        "sample_weight": [1, .25, .5, .25, 1],
        "shot_probability": [.1, .8, .5, .3, .9], "constant_probability": [.2] * 5,
    })


class ConfusionTests(unittest.TestCase):
    def test_counts_use_actual_rows_predicted_columns_and_include_threshold_ties(self):
        report = confusion_report(predictions()).set_index(["model", "weighting"])
        raw = report.loc[("logistic", "unweighted")]
        self.assertEqual(raw[["tn", "fp", "fn", "tp"]].tolist(), [1, 1, 1, 2])
        self.assertEqual(raw.total, 5)
        self.assertAlmostEqual(raw.accuracy, .6)

    def test_weighted_counts_and_rates_have_consistent_denominators(self):
        report = confusion_report(predictions()).set_index(["model", "weighting"])
        weighted = report.loc[("logistic", "possession_weighted")]
        self.assertEqual(weighted[["tn", "fp", "fn", "tp"]].tolist(), [1, .25, .25, 1.5])
        self.assertEqual(weighted.total, 3)
        self.assertAlmostEqual(weighted.accuracy, 2.5 / 3)
        self.assertAlmostEqual(weighted.precision, 1.5 / 1.75)
        self.assertAlmostEqual(weighted.recall, 1.5 / 1.75)
        self.assertAlmostEqual(weighted.specificity, 1 / 1.25)
        self.assertAlmostEqual(weighted.f1, 3 / 3.5)
        self.assertAlmostEqual(weighted.balanced_accuracy, (1.5 / 1.75 + .8) / 2)

    def test_no_positive_predictions_has_undefined_precision(self):
        row = confusion_report(predictions()).query("model == 'constant' and weighting == 'unweighted'").iloc[0]
        self.assertTrue(isnan(row.precision))
        self.assertEqual(row.recall, 0)
        self.assertEqual(row.f1, 0)
        self.assertEqual(row.specificity, 1)

    def test_lower_threshold_changes_confusion_but_not_input_scores(self):
        p = predictions()
        before = p.copy(deep=True)
        report = confusion_report(p, threshold=.3)
        row = report.query("model == 'logistic' and weighting == 'unweighted'").iloc[0]
        self.assertEqual(row[["tn", "fp", "fn", "tp"]].tolist(), [1, 1, 0, 3])
        assert_frame_equal(p, before)

    def test_invalid_thresholds_are_rejected(self):
        for threshold in [-.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                confusion_report(predictions(), threshold=threshold)

    def test_invalid_probabilities_and_weights_are_rejected(self):
        for column, value in [("shot_probability", 1.1), ("shot_probability", float("nan")),
                              ("sample_weight", 0), ("sample_weight", float("inf")), ("target", 2)]:
            p = predictions()
            p.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                confusion_report(p)

    def test_each_split_and_model_is_evaluated_separately(self):
        p = pd.concat([predictions(), predictions().assign(split="train")], ignore_index=True)
        report = confusion_report(p, probability_columns={"example": "shot_probability"})
        self.assertEqual(len(report), 4)
        self.assertEqual(set(report.model), {"example"})
        self.assertTrue(report.loc[report.weighting.eq("unweighted"), "total"].eq(5).all())

    def test_missing_actual_class_retains_two_by_two_matrix(self):
        report = confusion_report(predictions().assign(target=0))
        self.assertTrue(report.tp.eq(0).all())
        self.assertTrue(report.fn.eq(0).all())
        self.assertTrue(report.recall.isna().all())

    def test_report_threshold_does_not_change_fitted_model_or_probability_metrics(self):
        a = train_baseline(*fixture(), decision_threshold=.5)
        b = train_baseline(*fixture(), decision_threshold=.2)
        assert_frame_equal(a.coefficients, b.coefficients)
        assert_frame_equal(a.predictions, b.predictions)
        assert_frame_equal(a.metrics[["average_precision", "log_loss", "brier_score"]],
                           b.metrics[["average_precision", "log_loss", "brier_score"]])
        self.assertTrue(b.confusion.threshold.eq(.2).all())

    def test_saved_confusion_report_matches_evaluation(self):
        result = train_baseline(*fixture())
        with tempfile.TemporaryDirectory() as directory:
            save_baseline(result, Path(directory), {})
            assert_frame_equal(result.confusion, pd.read_parquet(Path(directory) / "confusion.parquet"))


if __name__ == "__main__":
    unittest.main()
