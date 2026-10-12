"""Season isolation and conservative handling of archive inconsistencies."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from afl_opposition.baseline import audit_shot_markers, train_baseline
from afl_opposition.examples import build_examples, season_split
from afl_opposition.preprocess import preprocess_events
from test_baseline import combine, episode
from test_preprocess import row


def seasons_fixture():
    pairs = []
    for season in [2023, 2024, 2025, 2026]:
        for positive in [False, True]:
            e, p = episode(f"{season}-{positive}", shot_index=3 if positive else None,
                           match=f"M{season}", date=f"{season}-04-01")
            e["season"] = season
            pairs.append((e, p))
    return combine(*pairs)


class SeasonSplitTests(unittest.TestCase):
    def test_requested_season_assignment(self):
        events, _ = seasons_fixture()
        result = season_split(events)
        self.assertEqual(result.split.tolist(), ["train", "train", "validation", "test"])
        self.assertEqual(result.match_id.nunique(), len(result))

    def test_reserved_test_may_be_absent(self):
        events, _ = seasons_fixture()
        self.assertEqual(set(season_split(events.loc[events.season.ne(2026)]).split), {"train", "validation"})

    def test_missing_validation_is_rejected(self):
        events, _ = seasons_fixture()
        with self.assertRaises(ValueError):
            season_split(events.loc[events.season.ne(2025)])

    def test_overlapping_and_unordered_assignments_are_rejected(self):
        events, _ = seasons_fixture()
        for years in [(2023, 2025), (2023, 2026), (2023, 2023)]:
            with self.subTest(years=years), self.assertRaises(ValueError):
                season_split(events, train_seasons=years)

    def test_date_and_season_conflict_is_rejected(self):
        events, _ = seasons_fixture()
        events.loc[events.season.eq(2025), "date"] = "2024-04-01"
        with self.assertRaises(ValueError):
            season_split(events)

    def test_unassigned_season_is_rejected(self):
        events, _ = seasons_fixture()
        extra = events.loc[events.season.eq(2026)].copy()
        extra["season"], extra["match_id"], extra["date"] = 2027, "M2027", "2027-04-01"
        with self.assertRaises(ValueError):
            season_split(pd.concat([events, extra]))

    def test_test_events_are_not_labelled_fitted_or_scored(self):
        events, possessions = seasons_fixture()
        before = train_baseline(events, possessions, train_seasons=(2023, 2024))
        test = events.season.eq(2026)
        # Invalid test features and outcomes must not even reach build_examples.
        events.loc[test, "x_norm"] = float("nan")
        events.loc[test, "is_explicit_shot"] = pd.NA
        after = train_baseline(events, possessions, train_seasons=(2023, 2024))
        assert_frame_equal(before.coefficients, after.coefficients)
        assert_frame_equal(before.predictions, after.predictions)
        self.assertNotIn("test", set(after.predictions.split))
        self.assertNotIn(2026, set(after.examples.season))

    def test_validation_changes_cannot_change_training_transform_or_weights(self):
        events, possessions = seasons_fixture()
        before = train_baseline(events, possessions, train_seasons=(2023, 2024))
        validation = events.season.eq(2025)
        events.loc[validation, "action_type"] = "validation_only"
        events.loc[validation, "x_norm"] = 9999
        after = train_baseline(events, possessions, train_seasons=(2023, 2024))
        assert_frame_equal(before.coefficients, after.coefficients)
        transform = after.pipeline.named_steps["features"]
        self.assertNotIn("validation_only", transform.named_transformers_["action"].categories_[0])
        self.assertEqual(before.pipeline.named_steps["features"].named_transformers_["numeric"].mean_.tolist(),
                         transform.named_transformers_["numeric"].mean_.tolist())


class QuarterQuarantineTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame([
            row(1, period_seconds=10), row(2, period_seconds=5, shot_at_goal=True),
            row(1, period=2), row(2, "Out of Bounds", period=2),
        ]).convert_dtypes()

    def test_reversed_time_remains_an_error_by_default(self):
        with self.assertRaises(ValueError):
            preprocess_events(self.sample())

    def test_whole_quarter_is_excluded_but_annotations_are_preserved(self):
        raw = self.sample()
        original = raw.copy(deep=True)
        result = preprocess_events(raw, quarantine_time_reversals=True)
        self.assertEqual(set(result.events.period), {2})
        self.assertEqual(len(result.annotations), len(raw))
        self.assertEqual(result.summary["quarantined_rows"], 2)
        self.assertEqual(result.summary["quarantined_explicit_shots"], 1)
        self.assertEqual(result.summary["raw_explicit_shots"], 1)
        self.assertEqual(result.summary["processed_explicit_shots"], 0)
        excluded = result.annotations.loc[result.annotations.quarantined]
        self.assertTrue(excluded.possession_id.isna().all())
        self.assertTrue(excluded.handling.eq("quarantined_quarter").all())
        assert_frame_equal(raw, original)

    def test_all_quarters_invalid_is_rejected(self):
        raw = self.sample()
        with self.assertRaises(ValueError):
            preprocess_events(raw.loc[raw.period.eq(1)], quarantine_time_reversals=True)

    def test_clean_quarters_produce_identical_actions(self):
        raw = self.sample().loc[lambda f: f.period.eq(2)]
        strict = preprocess_events(raw)
        tolerant = preprocess_events(raw, quarantine_time_reversals=True)
        assert_frame_equal(strict.events, tolerant.events)


class ShotAuditTests(unittest.TestCase):
    def sample(self):
        annotations = pd.DataFrame([
            dict(match_id="M1", period=1, chain_number=1, team_id="A", player_id="P",
                 display_order=1, description="Kick", shot_at_goal=False, possession_id="A"),
            dict(match_id="M1", period=1, chain_number=1, team_id="A", player_id="P",
                 display_order=2, description="Shot At Goal", shot_at_goal=None, possession_id=None),
            dict(match_id="M1", period=1, chain_number=1, team_id="B", player_id="Q",
                 display_order=3, description="Gather", shot_at_goal=None, possession_id="B"),
        ]).convert_dtypes()
        events, possessions = combine(episode("A"), episode("B"), episode("C"))
        return annotations, events, possessions

    def test_default_rejects_contradictory_marker(self):
        annotations, _, possessions = self.sample()
        with self.assertRaises(ValueError):
            audit_shot_markers(annotations, possessions)

    def test_conflict_excludes_all_episodes_touching_chain_without_imputing_shot(self):
        annotations, events, possessions = self.sample()
        original = possessions.copy(deep=True)
        reviewed, audit = audit_shot_markers(annotations, possessions, quarantine_conflicts=True)
        self.assertEqual(audit["conflicting_shot_markers"], 1)
        self.assertEqual(audit["shot_conflict_possessions"], 2)
        dataset = build_examples(events, reviewed)
        self.assertEqual(set(dataset.examples.possession_id), {"C"})
        self.assertEqual(set(dataset.possession_audit.loc[lambda f: f.possession_id.isin(["A", "B"]), "status"]),
                         {"shot_marker_conflict"})
        assert_frame_equal(possessions, original)

    def test_consistent_marker_keeps_episode(self):
        annotations, _, possessions = self.sample()
        annotations.loc[0, "shot_at_goal"] = True
        reviewed, audit = audit_shot_markers(annotations, possessions, quarantine_conflicts=True)
        self.assertFalse(reviewed.label_requires_review.any())
        self.assertEqual(audit["verified_retrospective_shot_markers"], 1)


if __name__ == "__main__":
    unittest.main()
