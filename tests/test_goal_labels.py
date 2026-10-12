"""Goal attribution, pre-shot causality and label provenance tests."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from afl_opposition.examples import build_examples
from afl_opposition.goal_labels import build_goal_examples, label_coverage, run_development_audit, save_goal_labels
from afl_opposition.preprocess import FEATURE_COLUMNS, preprocess_events


def row(order, description="Kick", **changes):
    record = dict(match_id="M1", period=1, display_order=order, chain_number=1,
        chain_team_id="A", initial_state="possGain", description=description,
        period_seconds=order, team_id="A", player_id="P1", x=10, y=5,
        shot_at_goal=None, venue_length=160, venue_width=120, date="2023-01-01", season=2023)
    record.update(changes)
    return record


def prepare(*records, **kwargs):
    return preprocess_events(pd.DataFrame(records).convert_dtypes(), **kwargs)


def label(result):
    return build_goal_examples(result.events, result.possessions, result.annotations)


def simple_goal():
    return prepare(row(1, "Gather"), row(2, "Handball"), row(3, shot_at_goal=True), row(4, "Goal"))


class GoalLabelTests(unittest.TestCase):
    def test_goal_changes_target_without_including_shot_or_outcome(self):
        result = label(simple_goal())
        self.assertEqual(result.examples.display_order.tolist(), [1, 2])
        self.assertEqual(result.examples.target.tolist(), [1, 1])
        self.assertEqual(result.examples.sample_weight.tolist(), [0.5, 0.5])
        self.assertEqual(result.goal_annotation_audit.status.tolist(), ["verified"])
        self.assertEqual(result.summary["positive_possessions"], 1)
        for col in ["goal_team_id", "goal_annotation_count", "end_reason", "shot_target", "is_explicit_shot"]:
            self.assertNotIn(col, result.examples.columns)

    def test_behind_changes_label_but_not_features_or_prefix_cutoff(self):
        goal = label(simple_goal())
        miss = label(prepare(row(1, "Gather"), row(2, "Handball"), row(3, shot_at_goal=True), row(4, "Behind")))
        self.assertEqual(miss.examples.target.tolist(), [0, 0])
        assert_frame_equal(goal.examples.drop(columns="target"), miss.examples.drop(columns="target"))

    def test_recovered_miss_then_goal_labels_only_before_first_shot(self):
        raw = prepare(row(1, "Gather"), row(2, shot_at_goal=True), row(3, "Loose Ball Get"),
                      row(4, shot_at_goal=True), row(5, "Goal"))
        result = label(raw)
        self.assertEqual(result.examples.display_order.tolist(), [1])
        self.assertEqual(result.examples.target.tolist(), [1])
        self.assertEqual(result.summary["positive_possessions_with_multiple_shots"], 1)

    def test_opponent_goal_does_not_label_previous_team_positive(self):
        result = label(prepare(row(1), row(2, "Gather", team_id="B"),
            row(3, team_id="B", shot_at_goal=True), row(4, "Goal", team_id="B")))
        self.assertEqual(result.examples.team_id.tolist(), ["A", "B"])
        self.assertEqual(result.examples.target.tolist(), [0, 1])

    def test_tied_times_use_display_order_and_do_not_cross_quarters(self):
        result = label(prepare(row(1, "Gather", period_seconds=0), row(2, shot_at_goal=True, period_seconds=0),
            row(3, "Goal", period_seconds=0), row(1, "Goal", period=2, period_seconds=0),
            row(2, "Gather", period=2, period_seconds=0), row(3, "Out of Bounds", period=2, period_seconds=0)))
        self.assertEqual(result.examples.target.tolist(), [1, 0])
        self.assertEqual(result.goal_annotation_audit.status.tolist(), ["verified", "no_preceding_action"])

    def test_goals_cannot_cross_match_boundaries(self):
        result = label(prepare(row(1), row(2, "Out of Bounds"), row(3, "Goal", match_id="M2")))
        self.assertEqual(result.examples.target.tolist(), [0])
        self.assertEqual(result.goal_annotation_audit.status.tolist(), ["no_preceding_action"])

    def test_missing_or_wrong_goal_team_is_excluded_not_negative(self):
        for team in [None, "B"]:
            with self.subTest(team=team):
                raw = simple_goal()
                raw.annotations.loc[raw.annotations.description.eq("Goal"), "team_id"] = team
                result = label(raw)
                self.assertTrue(result.examples.empty)
                self.assertEqual(result.possession_audit.status.tolist(), ["goal_attribution_review"])
                self.assertTrue(result.possession_audit.target.isna().all())

    def test_missing_goal_annotation_is_excluded_not_negative(self):
        raw = simple_goal()
        raw.annotations = raw.annotations.loc[~raw.annotations.description.eq("Goal")]
        result = label(raw)
        self.assertTrue(result.examples.empty)
        self.assertEqual(result.possession_audit.goal_review_reasons.tolist(), ["missing_goal_annotation"])

    def test_duplicate_goal_records_are_excluded(self):
        raw = prepare(row(1, "Gather"), row(2, shot_at_goal=True), row(3, "Goal"), row(4, "Goal"))
        result = label(raw)
        self.assertTrue(result.examples.empty)
        self.assertEqual(result.goal_annotation_audit.status.tolist(), ["multiple_goal_annotations"] * 2)

    def test_goal_after_non_goal_endpoint_is_flagged(self):
        raw = prepare(row(1, "Gather"), row(2, shot_at_goal=True), row(3, "Out of Bounds"), row(4, "Goal"))
        result = label(raw)
        self.assertTrue(result.examples.empty)
        self.assertEqual(result.goal_annotation_audit.status.tolist(), ["endpoint_not_goal"])

    def test_goal_timestamp_must_match_closure(self):
        raw = simple_goal()
        raw.annotations.loc[raw.annotations.description.eq("Goal"), "period_seconds"] += 1
        result = label(raw)
        self.assertTrue(result.examples.empty)
        self.assertEqual(result.goal_annotation_audit.status.tolist(), ["goal_time_mismatch"])

    def test_censoring_unknown_shot_and_first_action_shot_remain_excluded(self):
        raw = prepare(row(1), row(2, "Out of Bounds"),
            row(3, shot_at_goal=True), row(4, "Goal"),
            row(5, "Gather"), row(6, "Goal"), row(7, "Gather"))
        result = label(raw)
        self.assertEqual(result.examples.display_order.tolist(), [1])
        self.assertEqual(result.possession_audit.status.tolist(),
                         ["eligible", "no_pre_shot_action", "score_without_shot_flag", "preprocessing_review"])
        self.assertTrue(result.possession_audit.loc[result.possession_audit.status.ne("eligible"), "target"].isna().all())

    def test_shot_marker_review_carries_into_goal_labels(self):
        raw = prepare(row(1, "Gather"), row(2, shot_at_goal=True), row(3, "Goal"), row(4), row(5, "Out of Bounds"))
        raw.possessions["label_requires_review"] = [True, False]
        result = label(raw)
        self.assertEqual(result.examples.display_order.tolist(), [4])
        self.assertEqual(result.possession_audit.status.tolist(), ["shot_marker_conflict", "eligible"])

    def test_quarantined_goals_are_counted_but_never_linked_into_examples(self):
        raw = prepare(row(1, "Gather"), row(2, shot_at_goal=True), row(3, "Goal"),
            row(1, "Gather", period=2, period_seconds=3), row(2, shot_at_goal=True, period=2, period_seconds=2),
            row(3, "Goal", period=2, period_seconds=4), quarantine_time_reversals=True)
        result = label(raw)
        self.assertEqual(result.summary["raw_goal_annotations"], 2)
        self.assertEqual(result.summary["quarantined_goal_annotations"], 1)
        self.assertEqual(result.summary["positive_possessions"], 1)
        self.assertEqual(result.goal_annotation_audit.status.tolist(), ["verified", "quarantined_quarter"])

    def test_original_schema_without_quarantine_column_is_supported(self):
        raw = simple_goal()
        raw.annotations = raw.annotations.drop(columns="quarantined")
        self.assertEqual(label(raw).examples.target.tolist(), [1, 1])

    def test_no_input_tables_or_original_shot_labels_are_mutated(self):
        raw = simple_goal()
        before = [table.copy(deep=True) for table in [raw.events, raw.possessions, raw.annotations]]
        shot_before = build_examples(raw.events, raw.possessions)
        label(raw)
        for original, table in zip(before, [raw.events, raw.possessions, raw.annotations]):
            assert_frame_equal(original, table)
        assert_frame_equal(shot_before.examples, build_examples(raw.events, raw.possessions).examples)

    def test_annotation_stream_must_contain_the_retained_inputs(self):
        raw = simple_goal()
        raw.annotations = raw.annotations.iloc[1:]
        with self.assertRaisesRegex(ValueError, "missing retained actions"):
            label(raw)

    def test_coverage_and_saved_hashes_reconcile(self):
        result = label(simple_goal())
        coverage = label_coverage(result)
        self.assertEqual(coverage.iloc[0].prefix_examples, 2)
        self.assertEqual(coverage.iloc[0].goal_possessions, 1)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_goal_labels(result, directory, {"seasons": [2023]})
            summary = json.loads((directory / "summary.json").read_text())
            for name, digest in summary["artifact_sha256"].items():
                self.assertEqual(hashlib.sha256((directory / f"{name}.parquet").read_bytes()).hexdigest(), digest)
            assert_frame_equal(pd.read_parquet(directory / "examples.parquet"), result.examples)
            with self.assertRaisesRegex(ValueError, "already exist"):
                save_goal_labels(result, directory, {})

    def test_development_entrypoint_requests_no_test_season(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("afl_opposition.baseline.load_preprocessed_seasons", side_effect=RuntimeError("stop after call")) as loader:
                with self.assertRaisesRegex(RuntimeError, "stop after call"):
                    run_development_audit(Path(tmp))
                self.assertEqual(loader.call_args.kwargs["seasons"], (2023, 2024, 2025))


if __name__ == "__main__":
    unittest.main()
