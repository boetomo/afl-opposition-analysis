"""Next-event boundaries, censoring, prefix safety and model/training regression checks."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import torch

from afl_opposition.preprocess import preprocess_events
from afl_opposition.next_events import ACTIONS, END, NUMERIC, build_next_events, observed_features
from afl_opposition.next_event_model import (
    FeatureScaler, PrefixData, NextEventTransformer, configure, joint_loss,
    load_model, save_model, predict, forecast_next,
)
from afl_opposition.next_event_experiment import TransitionBaseline, configs, metrics, train
from afl_opposition.sequence import state_fingerprint
import afl_opposition.next_event_experiment as experiment


def raw(order, description="Kick", **changes):
    value = dict(match_id="M1", period=1, display_order=order, chain_number=1,
                 chain_team_id="A", initial_state="possGain", description=description,
                 period_seconds=order, team_id="A", player_id="P1", x=order, y=5,
                 shot_at_goal=None, venue_length=160, venue_width=120, date="2023-01-01", season=2023)
    value.update(changes)
    return value


def dataset(records=None):
    records = records if records is not None else [
        raw(1, "Gather"), raw(2, "Handball"), raw(3, "Handball Received"),
        raw(4, shot_at_goal=True), raw(5, "Goal")]
    processed = preprocess_events(pd.DataFrame(records).convert_dtypes())
    return build_next_events(processed.events, processed.possessions)


class NextEventDataTests(unittest.TestCase):
    def test_targets_are_next_actions_and_last_is_end(self):
        frame = dataset().events
        self.assertEqual(frame.target_action.tolist(), ["handball", "receive", "kick", END])
        self.assertEqual(frame.has_location.tolist(), [True, True, True, False])
        self.assertAlmostEqual(frame.iloc[0].target_x, 2 / 80)
        self.assertEqual(frame.iloc[0].next_event_id, frame.iloc[1].event_id)
        self.assertNotIn("is_explicit_shot", NUMERIC)

    def test_archive_end_is_unknown_not_a_negative_or_end(self):
        frame = dataset([raw(1), raw(2, "Handball")]).events
        self.assertEqual(frame.labelled.tolist(), [True, False])
        self.assertFalse(frame.has_location.iloc[-1])
        self.assertTrue(pd.isna(frame.target_action.iloc[-1]))

    def test_turnover_does_not_predict_opponents_location_in_wrong_frame(self):
        frame = dataset([raw(1), raw(2, "Gather", team_id="B"), raw(3, "Goal", team_id="B")]).events
        self.assertEqual(frame.target_action.tolist(), [END, END])
        self.assertEqual(frame.event_index.tolist(), [0, 0])
        self.assertFalse(frame.has_location.any())

    def test_new_quarter_never_becomes_next_event(self):
        frame = dataset([raw(1), raw(2, period=2), raw(3, "Goal", period=2)]).events
        self.assertEqual(frame.labelled.tolist(), [False, True])
        self.assertEqual(frame.iloc[-1].target_action, END)

    def test_appending_future_preserves_existing_feature_values(self):
        records = [raw(1, "Gather"), raw(2, "Handball"), raw(3), raw(4, "Goal")]
        full = dataset(records).events
        for count in range(1, 4):
            prefix = dataset(records[:count]).events
            assert_frame_equal(prefix[["event_id", *NUMERIC]],
                               full.iloc[:len(prefix)][["event_id", *NUMERIC]], check_dtype=False)

    def test_reviewed_episode_excluded_without_bridging_gap(self):
        records = [raw(1), raw(2, "Baulked"), raw(3, "Goal"),
                   raw(4, "Gather", chain_number=2), raw(5, "Goal", chain_number=2)]
        result = dataset(records)
        self.assertEqual(result.events.display_order.tolist(), [4])
        self.assertEqual(result.events.target_action.tolist(), [END])
        self.assertTrue((~result.coverage.included).any())

    def test_target_location_not_current_shot_or_chain_final_state(self):
        records = [raw(1, shot_at_goal=True), raw(2, "Uncontested Mark"), raw(3, "Goal")]
        frame = dataset(records).events
        self.assertEqual(frame.iloc[0].target_action, "mark")
        self.assertEqual(frame.iloc[0].action_type, "kick")

    def test_shuffled_source_and_input_preservation(self):
        data = preprocess_events(pd.DataFrame([raw(1), raw(2), raw(3, "Goal")]).convert_dtypes())
        original = data.events.copy(deep=True)
        a = build_next_events(data.events, data.possessions).events
        b = build_next_events(data.events.iloc[::-1], data.possessions).events
        assert_frame_equal(a, b)
        assert_frame_equal(data.events, original)

    def test_duplicate_and_missing_observation_rejected(self):
        data = preprocess_events(pd.DataFrame([raw(1), raw(2), raw(3, "Goal")]).convert_dtypes())
        with self.assertRaises(ValueError):
            build_next_events(pd.concat([data.events, data.events.iloc[:1]]), data.possessions)
        with self.assertRaises(ValueError):
            build_next_events(data.events.iloc[1:], data.possessions)


class NextEventModelTests(unittest.TestCase):
    def setUp(self):
        configure("cpu", 11)
        self.frame = dataset().events
        self.scaler = FeatureScaler.fit(self.frame)
        self.data = PrefixData(self.frame, self.scaler)
        self.config = {**configs()[0], "dropout": .2}

    def test_growing_capped_history_and_target_exclusion(self):
        a, n, target, _, _ = self.data.batch(np.array([0, 1, 3]), cap=2)
        self.assertEqual(a.ne(0).sum(dim=1).tolist(), [1, 2, 2])
        ids = {name: i + 1 for i, name in enumerate(ACTIONS)}
        self.assertEqual(a[0].tolist(), [ids["gather"], 0])
        self.assertEqual(a[2].tolist(), [ids["receive"], ids["kick"]])
        self.assertEqual(target.tolist(), [ACTIONS.index("handball"), ACTIONS.index("receive"), ACTIONS.index(END)])

    def test_padding_does_not_change_prediction(self):
        model = NextEventTransformer(self.config).eval()
        a, n, *_ = self.data.batch(np.array([1]), 16)
        with torch.inference_mode():
            expected = model(a, n)
            padded = model(torch.nn.functional.pad(a, (0, 3)),
                           torch.nn.functional.pad(n, (0, 0, 0, 3)))
        for x, y in zip(expected, padded):
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)

    def test_changing_future_and_targets_does_not_change_input(self):
        changed = self.frame.copy()
        changed.loc[3, "x_norm"] = .95
        changed["target_x"] = 100.
        altered = PrefixData(changed, self.scaler)
        for a, b in zip(self.data.batch(np.array([1]), 16)[:2], altered.batch(np.array([1]), 16)[:2]):
            torch.testing.assert_close(a, b)

    def test_terminal_locations_have_no_gradient(self):
        logits = torch.randn(2, len(ACTIONS), requires_grad=True)
        xy = torch.randn(2, 2, requires_grad=True)
        loss = joint_loss(logits, xy, torch.tensor([0, len(ACTIONS) - 1]),
                          torch.zeros(2, 2), torch.tensor([True, False]))
        loss.backward()
        self.assertEqual(xy.grad[1].abs().sum(), 0)
        self.assertGreater(logits.grad[1].abs().sum(), 0)

    def test_all_terminal_batch_is_finite(self):
        logits = torch.randn(2, len(ACTIONS), requires_grad=True)
        xy = torch.randn(2, 2, requires_grad=True)
        loss = joint_loss(logits, xy, torch.tensor([0, 0]), torch.zeros(2, 2), torch.tensor([False, False]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(xy.grad.abs().sum(), 0)

    def test_scaler_and_baseline_reject_test_fit(self):
        changed = self.frame.assign(season=2026)
        with self.assertRaises(ValueError):
            FeatureScaler.fit(changed)
        with self.assertRaises(ValueError):
            TransitionBaseline().fit(changed)

    def test_baseline_probabilities_confusion_and_metrics(self):
        p, xy = TransitionBaseline().fit(self.frame).predict(self.frame)
        np.testing.assert_allclose(p.sum(axis=1), 1)
        report, confusion, per_action = metrics(self.frame, p, xy)
        self.assertEqual(confusion.to_numpy().sum(), 4)
        self.assertEqual(per_action.support.sum(), 4)
        self.assertEqual(report["location_examples"], 3)

    def test_export_reload_and_checksum(self):
        model = NextEventTransformer(self.config).eval()
        with tempfile.TemporaryDirectory() as name:
            save_model(name, model, self.scaler, {})
            loaded, scaler, info = load_model(name)
            self.assertEqual(state_fingerprint(model), state_fingerprint(loaded))
            p, xy = predict(model, self.data)
            q, z = predict(loaded, PrefixData(self.frame, scaler))
            np.testing.assert_array_equal(p, q)
            np.testing.assert_array_equal(xy, z)
            with (Path(name) / "weights.pt").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(ValueError):
                load_model(name)

    def test_forecast_needs_no_target_columns_and_matches_batch_prediction(self):
        model = NextEventTransformer(self.config).eval()
        history = self.frame.iloc[:2].drop(columns=[
            "target_action", "target_x", "target_y", "has_location", "labelled", "next_event_id"])
        result = forecast_next(model, self.scaler, history)
        a, n, *_ = self.data.batch(np.array([1]), 16)
        with torch.inference_mode():
            logits, xy = model(a, n)
        np.testing.assert_allclose(list(result["probabilities"].values()), logits.softmax(-1)[0].numpy())
        self.assertEqual(result["used_events"], 2)
        self.assertAlmostEqual(result["location_if_continuation"]["x_norm"], float(xy[0, 0]))

    def test_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            full, _, _ = train(self.data, None, self.config, 3, root / "full.pt", seed=44, batch_size=2)
            train(self.data, None, self.config, 1, root / "resume.pt", seed=44, batch_size=2)
            resumed, _, _ = train(self.data, None, self.config, 3, root / "resume.pt", seed=44, batch_size=2)
            self.assertEqual(state_fingerprint(full), state_fingerprint(resumed))

    def test_small_training_can_reduce_joint_loss(self):
        config = {**self.config, "dropout": 0., "learning_rate": .01}
        model = NextEventTransformer(config)
        optimiser = torch.optim.Adam(model.parameters(), lr=.01)
        a, n, y, xy, mask = self.data.batch(self.data.ends, 16)
        first = float(joint_loss(*model(a, n), y, xy, mask).detach())
        for _ in range(60):
            optimiser.zero_grad()
            loss = joint_loss(*model(a, n), y, xy, mask)
            loss.backward()
            optimiser.step()
        last = float(joint_loss(*model(a, n), y, xy, mask).detach())
        self.assertLess(last, first * .35)

    def test_full_workflow_freezes_before_test_and_keeps_splits_separate(self):
        def source(seasons):
            all_events, all_possessions = [], []
            for season in seasons:
                rows = [raw(i, action, match_id=f"M{season}", season=season,
                            date=f"{season}-04-01") for i, action in enumerate(
                                ["Gather", "Handball", "Handball Received", "Kick",
                                 "Uncontested Mark", "Kick", "Goal"], start=1)]
                processed = preprocess_events(pd.DataFrame(rows).convert_dtypes())
                all_events.append(processed.events)
                all_possessions.append(processed.possessions)
            return pd.concat(all_events, ignore_index=True), pd.concat(all_possessions, ignore_index=True), {}

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output = root / "artifacts/experiment"
            calls = []

            def loader(_, seasons):
                calls.append(tuple(seasons))
                if tuple(seasons) == (2026,):
                    self.assertTrue((output / "model/model.json").exists())
                    self.assertTrue((output / "selection.json").exists())
                    info = json.loads((output / "model/model.json").read_text())
                    self.assertEqual(info["training_seasons"], [2023, 2024, 2025])
                return source(seasons)

            small_configs = [{**configs()[0], "dropout": 0.}, {**configs()[0], "max_history": 2, "dropout": 0.}]
            with patch.object(experiment, "load_preprocessed_seasons", side_effect=loader), \
                 patch.object(experiment, "source_hashes", return_value={}), \
                 patch.object(experiment, "configs", return_value=small_configs), \
                 patch.object(experiment, "INITIAL_EPOCHS", 1), \
                 patch.object(experiment, "FINAL_EPOCHS", 2):
                experiment.run(root, output, device="cpu")
                with self.assertRaises(ValueError):
                    experiment.run(root, output, device="cpu", resume=True)
            self.assertEqual(calls, [(2023, 2024, 2025), (2026,)])
            result = pd.read_csv(output / "metrics.csv")
            self.assertEqual(len(result), 6)
            self.assertEqual(set(result.split), {"validation", "test"})


if __name__ == "__main__":
    unittest.main()
