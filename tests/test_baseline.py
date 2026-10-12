"""Tests for labels, temporal isolation and the current-state model contract."""

import pickle
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from afl_opposition.baseline import make_pipeline, save_baseline, train_baseline
from afl_opposition.examples import build_examples, chronological_split
from afl_opposition.preprocess import FEATURE_COLUMNS


def episode(name, shot_index=None, length=4, date="2023-01-01", match="M1", **changes):
    events = []
    for i in range(length):
        events.append(dict(
            event_id=f"{name}:e{i}", possession_id=name, match_id=match, period=1,
            event_index=i, date=date, season=2023, team_id="A", display_order=i,
            action_type="kick" if i % 2 else "gather", x_norm=i / 10, y_norm=0.1,
            gap_seconds=0 if i == 0 else 2, elapsed_seconds=2 * i,
            is_explicit_shot=i == shot_index,
        ))
    possession = dict(
        possession_id=name, match_id=match, period=1, team_id="A", event_count=length,
        explicit_shot_count=int(shot_index is not None), requires_review=False,
        end_reason="opponent_control", review_reasons="",
    )
    possession.update(changes)
    return pd.DataFrame(events).convert_dtypes(), pd.DataFrame([possession]).convert_dtypes()


def combine(*pairs):
    return tuple(pd.concat([pair[index] for pair in pairs], ignore_index=True) for index in [0, 1])


def fixture():
    pairs = []
    for i in range(6):
        for positive in [False, True]:
            pairs.append(episode(f"{i}-{positive}", shot_index=3 if positive else None,
                                 match=f"M{i}", date=f"2023-01-{i+1:02d}"))
    return combine(*pairs)


class LabelTests(unittest.TestCase):
    def test_only_prefixes_before_first_shot_are_positive(self):
        events, possessions = episode("A", shot_index=3, length=5)
        dataset = build_examples(events, possessions)
        self.assertEqual(dataset.examples.event_index.tolist(), [0, 1, 2])
        self.assertEqual(dataset.examples.target.tolist(), [1, 1, 1])

    def test_all_known_negative_states_are_zero(self):
        dataset = build_examples(*episode("A", length=3))
        self.assertEqual(dataset.examples.target.tolist(), [0, 0, 0])

    def test_first_action_shot_has_no_pre_shot_example(self):
        dataset = build_examples(*combine(episode("A", shot_index=0), episode("B")))
        self.assertEqual(set(dataset.examples.possession_id), {"B"})
        self.assertEqual(dataset.possession_audit.set_index("possession_id").loc["A", "status"], "no_pre_shot_action")

    def test_goal_without_explicit_shot_is_not_negative(self):
        dataset = build_examples(*combine(episode("A", end_reason="Goal"), episode("B")))
        self.assertEqual(set(dataset.examples.possession_id), {"B"})
        audit = dataset.possession_audit.set_index("possession_id")
        self.assertEqual(audit.loc["A", "status"], "score_without_shot_flag")
        self.assertTrue(pd.isna(audit.loc["A", "target"]))

    def test_censored_episode_is_excluded_even_without_review_flag(self):
        dataset = build_examples(*combine(episode("A", end_reason="end_of_available_events"), episode("B")))
        self.assertEqual(set(dataset.examples.possession_id), {"B"})

    def test_reviewed_positive_episode_is_excluded(self):
        dataset = build_examples(*combine(episode("A", shot_index=3, requires_review=True), episode("B")))
        self.assertEqual(set(dataset.examples.possession_id), {"B"})

    def test_multiple_shots_stop_at_first(self):
        events, possessions = episode("A", shot_index=2, length=5)
        events.loc[4, "is_explicit_shot"] = True
        possessions.loc[0, "explicit_shot_count"] = 2
        dataset = build_examples(events, possessions)
        self.assertEqual(dataset.examples.event_index.tolist(), [0, 1])

    def test_each_possession_has_total_weight_one(self):
        dataset = build_examples(*combine(episode("A", length=2), episode("B", length=20)))
        for weight in dataset.examples.groupby("possession_id").sample_weight.sum():
            self.assertAlmostEqual(weight, 1)

    def test_shot_and_end_metadata_are_not_example_features(self):
        dataset = build_examples(*episode("A", shot_index=3))
        for name in ["is_explicit_shot", "explicit_shot_count", "end_reason", "requires_review", "first_shot_index"]:
            self.assertNotIn(name, dataset.examples.columns)

    def test_inconsistent_metadata_is_rejected(self):
        events, possessions = episode("A", shot_index=3)
        possessions.loc[0, "explicit_shot_count"] = 0
        with self.assertRaises(ValueError):
            build_examples(events, possessions)

    def test_mixed_team_episode_is_rejected(self):
        events, possessions = episode("A")
        events.loc[2, "team_id"] = "B"
        with self.assertRaises(ValueError):
            build_examples(events, possessions)

    def test_noncontiguous_event_indices_are_rejected(self):
        events, possessions = episode("A")
        events.loc[2, "event_index"] = 9
        with self.assertRaises(ValueError):
            build_examples(events, possessions)

    def test_input_tables_are_unchanged(self):
        events, possessions = fixture()
        event_copy, possession_copy = events.copy(deep=True), possessions.copy(deep=True)
        build_examples(events, possessions)
        assert_frame_equal(events, event_copy)
        assert_frame_equal(possessions, possession_copy)


class SplitAndModelTests(unittest.TestCase):
    def test_all_matches_on_same_date_stay_together(self):
        events = pd.DataFrame({"match_id": ["A", "B", "C", "D"], "date": ["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"]})
        splits = chronological_split(events, train_fraction=0.5)
        self.assertEqual(splits.groupby("date").split.nunique().max(), 1)
        self.assertLess(splits.loc[splits.split.eq("train"), "date"].max(), splits.loc[splits.split.eq("validation"), "date"].min())

    def test_inconsistent_match_dates_are_rejected(self):
        with self.assertRaises(ValueError):
            chronological_split(pd.DataFrame({"match_id": ["A", "A"], "date": ["2023-01-01", "2023-01-02"]}))

    def test_scaler_and_vocabulary_only_learn_training_data(self):
        events, possessions = fixture()
        validation_mask = events.match_id.eq("M5")
        events.loc[validation_mask, "action_type"] = "validation_only_action"
        events.loc[validation_mask, "x_norm"] = 1000
        result = train_baseline(events, possessions)
        fitted = result.pipeline.named_steps["features"]
        self.assertNotIn("validation_only_action", fitted.named_transformers_["action"].categories_[0])
        train = result.examples.loc[result.examples.split.eq("train")]
        self.assertAlmostEqual(fitted.named_transformers_["numeric"].mean_[0], float(train.x_norm.mean()))
        self.assertTrue(result.predictions.shot_probability.between(0, 1).all())

    def test_future_metadata_cannot_change_prediction(self):
        events, possessions = fixture()
        result = train_baseline(events, possessions)
        values = result.examples.copy()
        before = result.pipeline.predict_proba(values)
        values["target"] = 1 - values.target
        values["final_score"] = 999
        values["future_shot"] = True
        after = result.pipeline.predict_proba(values)
        self.assertTrue((before == after).all())

    def test_no_lagged_features_are_created(self):
        result = train_baseline(*fixture())
        self.assertEqual(result.summary["feature_columns"], FEATURE_COLUMNS)
        self.assertEqual(result.summary["encoded_feature_count"], 6)  # Two actions + four numeric inputs.

    def test_train_and_validation_share_no_matches_or_possessions(self):
        result = train_baseline(*fixture())
        train = result.examples.loc[result.examples.split.eq("train")]
        validation = result.examples.loc[result.examples.split.eq("validation")]
        self.assertFalse(set(train.match_id) & set(validation.match_id))
        self.assertFalse(set(train.possession_id) & set(validation.possession_id))

    def test_constant_is_weighted_training_prevalence(self):
        result = train_baseline(*fixture())
        train = result.examples.loc[result.examples.split.eq("train")]
        expected = (train.target * train.sample_weight).sum() / train.sample_weight.sum()
        self.assertAlmostEqual(result.summary["constant_probability"], expected)
        self.assertTrue(result.predictions.constant_probability.eq(expected).all())

    def test_saved_pipeline_preserves_predictions(self):
        result = train_baseline(*fixture())
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            save_baseline(result, directory, {})
            with (directory / "model.pkl").open("rb") as stream:
                loaded = pickle.load(stream)
            self.assertTrue((loaded.predict_proba(result.examples[FEATURE_COLUMNS]) == result.pipeline.predict_proba(result.examples[FEATURE_COLUMNS])).all())
            with self.assertRaises(ValueError):
                save_baseline(result, directory, {})


if __name__ == "__main__":
    unittest.main()
