"""Boundary, provenance and causal-feature checks using small synthetic passages."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from afl_opposition.preprocess import FEATURE_COLUMNS, preprocess_events, write_result


def row(order, description="Kick", **changes):
    record = dict(
        match_id="M1", period=1, display_order=order, chain_number=1,
        chain_team_id="A", initial_state="possGain", description=description,
        period_seconds=order, team_id="A", player_id="P1", x=10, y=5,
        shot_at_goal=None, venue_length=160, venue_width=120,
        date="2023-01-01", season=2023,
    )
    record.update(changes)
    return record


def run(*records):
    return preprocess_events(pd.DataFrame(records).convert_dtypes())


class PreprocessingTests(unittest.TestCase):
    def test_cross_chain_copy_keeps_first_and_changes_control(self):
        result = run(
            row(1), row(2, "Uncontested Mark", team_id="B", player_id="P2", x=30, y=-7),
            row(3, "Uncontested Mark", team_id="B", player_id="P2", x=-30, y=7,
                chain_number=2, chain_team_id="B", period_seconds=2),
            row(4, team_id="B", player_id="P2", chain_number=2, chain_team_id="B"),
        )
        self.assertEqual(result.events.display_order.tolist(), [1, 2, 4])
        self.assertEqual(result.events.x_attacking.tolist(), [10, -30, 10])
        self.assertEqual(result.possessions.event_count.tolist(), [1, 2])
        self.assertEqual(result.possessions.iloc[0].end_reason, "opponent_control")
        self.assertEqual(result.annotations.iloc[2].canonical_event_id, result.events.iloc[1].event_id)

    def test_same_second_different_actions_survive(self):
        result = run(row(1, "Hard Ball Get"), row(2, "Handball", period_seconds=1))
        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.events.gap_seconds.tolist(), [0, 0])

    def test_same_chain_repeats_are_not_blindly_deduplicated(self):
        result = run(row(1), row(2, period_seconds=1))
        self.assertEqual(len(result.events), 2)

    def test_equal_time_and_identity_but_different_coordinates_not_merged(self):
        result = run(row(1), row(2, chain_number=2, period_seconds=1, x=50))
        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.summary["boundary_duplicates"], 0)
        self.assertTrue(result.possessions.requires_review.all())

    def test_opponent_fumble_and_spoil_do_not_take_control(self):
        result = run(row(1), row(2, "Mark Fumbled", team_id="B"),
                     row(3, "Spoil", team_id="B"), row(4, "Loose Ball Get"))
        self.assertEqual(len(result.possessions), 1)
        self.assertEqual(result.events.display_order.tolist(), [1, 4])

    def test_opponent_control_within_one_raw_chain_splits(self):
        result = run(row(1), row(2, "Gather from Opposition", team_id="B"))
        self.assertEqual(result.possessions.team_id.tolist(), ["A", "B"])

    def test_opponent_control_on_new_chain_is_supported(self):
        result = run(row(1), row(2, "Gather", chain_number=2, chain_team_id="B", team_id="B"))
        self.assertEqual(result.possessions.iloc[0].end_reason, "opponent_control")
        self.assertFalse(result.possessions.iloc[0].requires_review)

    def test_stoppages_end_even_if_same_team_resumes(self):
        for stop in ["Goal", "Behind", "Ball Up Call", "Out of Bounds"]:
            with self.subTest(stop=stop):
                result = run(row(1), row(2, stop), row(3))
                self.assertEqual(len(result.possessions), 2)
                self.assertEqual(result.possessions.iloc[0].end_reason, stop)

    def test_restart_metadata_closes_unrecorded_stoppage(self):
        result = run(row(1), row(2, "Gather", chain_number=2, initial_state="throwIn"))
        self.assertEqual(len(result.possessions), 2)
        self.assertEqual(result.possessions.iloc[0].end_reason, "restart_throwIn")

    def test_unresolved_chain_switch_is_quarantined(self):
        result = run(row(1), row(2, "Contested Knock On", chain_number=2, chain_team_id="B", team_id="B"),
                     row(3, "Gather", chain_number=2, chain_team_id="B", team_id="B"))
        self.assertEqual(result.possessions.iloc[0].end_reason, "unresolved_chain_change")
        self.assertTrue(result.possessions.iloc[0].requires_review)

    def test_exact_kick_annotation_links_without_future_feature(self):
        result = run(row(1), row(2, "Kick Into F50", period_seconds=1), row(3, "Goal"))
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.annotations.iloc[1].handling, "paired_kick_annotation")
        self.assertEqual(result.annotations.iloc[1].canonical_event_id, result.events.iloc[0].event_id)
        self.assertNotIn("inside_50", result.events.columns)

    def test_unpaired_annotation_flags_affected_episode(self):
        result = run(row(1), row(2, "Kick Into F50", period_seconds=1, x=11), row(3, "Goal"))
        self.assertEqual(result.summary["unpaired_kick_annotations"], 1)
        self.assertTrue(result.possessions.iloc[0].requires_review)

    def test_unknown_actions_remain_visible(self):
        result = run(row(1), row(2, "Future Provider Annotation"), row(3, "Goal"))
        self.assertEqual(result.summary["unknown_descriptions"], ["Future Provider Annotation"])
        self.assertEqual(len(result.annotations), 3)
        self.assertTrue(result.possessions.iloc[0].requires_review)

    def test_missing_team_is_not_filled_from_chain_owner(self):
        result = run(row(1), row(2, "Handball Received", team_id=None), row(3, "Goal"))
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.annotations.iloc[1].handling, "review")
        self.assertTrue(result.possessions.iloc[0].requires_review)

    def test_different_quarters_never_share_possession_or_gap(self):
        result = run(row(1, period_seconds=100), row(2, period=2, period_seconds=1))
        self.assertEqual(len(result.possessions), 2)
        self.assertEqual(result.events.gap_seconds.tolist(), [0, 0])

    def test_same_order_in_different_matches_is_valid(self):
        result = run(row(1), row(1, match_id="M2"))
        self.assertEqual(result.events.event_id.nunique(), 2)

    def test_end_of_available_data_is_censored(self):
        result = run(row(1))
        self.assertEqual(result.possessions.iloc[0].end_reason, "end_of_available_events")
        self.assertTrue(result.possessions.iloc[0].requires_review)

    def test_missing_coordinate_frame_is_flagged(self):
        result = run(row(1, chain_team_id=None), row(2, "Goal", chain_team_id=None))
        self.assertTrue(pd.isna(result.events.iloc[0].x_norm))
        self.assertIn("missing_or_invalid_coordinates", result.possessions.iloc[0].review_reasons)

    def test_shot_flag_does_not_end_a_missed_shot_possession(self):
        result = run(row(1, shot_at_goal=True), row(2, "Uncontested Mark"), row(3, "Goal"))
        self.assertEqual(len(result.possessions), 1)
        self.assertEqual(result.summary["processed_explicit_shots"], 1)
        self.assertNotIn("is_explicit_shot", FEATURE_COLUMNS)

    def test_unknown_shot_flags_are_not_training_negatives(self):
        result = run(row(1, shot_at_goal=False), row(2))
        self.assertEqual(result.summary["processed_explicit_shots"], 0)
        self.assertNotIn("target", result.events.columns)

    def test_future_rows_do_not_change_existing_features(self):
        records = [row(1, "Gather"), row(2, "Handball"), row(3, "Handball Received"),
                   row(4, shot_at_goal=True), row(5, "Goal")]
        full = run(*records)
        for size in range(1, len(records)):
            prefix = run(*records[:size])
            columns = ["event_id", "possession_id", "event_index", *FEATURE_COLUMNS]
            expected = full.events.set_index("event_id").loc[prefix.events.event_id].reset_index()
            assert_frame_equal(prefix.events[columns], expected[columns], check_dtype=False)

    def test_future_outcomes_are_not_consulted(self):
        raw = pd.DataFrame([row(1), row(2, "Goal")]).convert_dtypes()
        raw["final_state"] = "goal"
        raw["home_team_score_total_score"] = 100
        a = preprocess_events(raw)
        raw["final_state"] = "turnover"
        raw["home_team_score_total_score"] = 0
        b = preprocess_events(raw)
        assert_frame_equal(a.events, b.events)

    def test_input_is_unchanged_and_sorting_is_deterministic(self):
        raw = pd.DataFrame([row(2), row(1)]).convert_dtypes()
        original = raw.copy(deep=True)
        a = preprocess_events(raw)
        assert_frame_equal(raw, original)
        b = preprocess_events(raw.iloc[::-1])
        assert_frame_equal(a.events, b.events)

    def test_bad_order_or_time_is_rejected(self):
        for records in [[row(1), row(1)], [row(1, period_seconds=5), row(2, period_seconds=3)]]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                run(*records)

    def test_write_guards_source_and_requires_explicit_regeneration(self):
        result = run(row(1))
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            source = directory / "source.parquet"
            source.write_bytes(b"unchanged raw fixture")
            out = directory / "derived"
            write_result(result, out, source)
            with self.assertRaises(ValueError):
                write_result(result, out, source)
            with self.assertRaises(ValueError):
                write_result(result, out, out / "events.parquet", overwrite=True)
            self.assertEqual(source.read_bytes(), b"unchanged raw fixture")


if __name__ == "__main__":
    unittest.main()
