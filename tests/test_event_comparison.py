"""Boundary, information-flow and training checks for the two event formulations."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from afl_opposition.preprocess import preprocess_events
from afl_opposition.quarter_events import build_quarters, make_streams, BALL_ACTIONS, NUMERIC, K, QuarterData
from afl_opposition.event_models import (
    EventEncoder, EventData, EventTransformer, configure, objective, class_weights,
    save_model, load_model, forecast_next,
)
from afl_opposition.event_comparison import configurations, train, run
from afl_opposition.sequence import state_fingerprint


def observations(extra_quarter=False):
    specs = [("Gather", "A"), ("Kick", "A"), ("Gather", "B"),
             ("Handball", "B"), ("Gather", "A"), ("Kick", "A"),
             ("Goal", "A"), ("Centre Bounce", "A"), ("Gather", "B"), ("Kick", "B")]
    rows = []
    for period in ([1, 2] if extra_quarter else [1]):
        for order, (action, team) in enumerate(specs, 1):
            rows.append(dict(match_id="M1", period=period, display_order=order, chain_number=1,
                chain_team_id="A", initial_state="possGain", description=action,
                period_seconds=order * 5, team_id=team, player_id=team + "1", x=order, y=4,
                shot_at_goal=None, venue_length=160, venue_width=120, date="2023-01-01", season=2023))
    processed = preprocess_events(pd.DataFrame(rows).convert_dtypes())
    metadata = pd.DataFrame([dict(match_id="M1", home_team_id="A", away_team_id="B")])
    frame, audit = build_quarters(processed.events, processed.possessions, processed.annotations, metadata)
    return frame, processed, metadata, audit


def config(mode):
    return dict(configurations(mode)[0], width=32, embedding_dim=8, layers=1,
                dropout=0., window=2 if mode == "team_stream" else 0)


class EventStreamTests(unittest.TestCase):
    def test_team_stream_removes_opponent_actions_and_inserts_observed_boundary(self):
        frame, *_ = observations()
        stream = make_streams(frame, "team_stream")
        home = stream.loc[stream.focal_side.eq(0)]
        self.assertEqual(home.action_type.tolist(), ["gather", "kick", "boundary", "gather", "kick", "boundary"])
        self.assertEqual(home.iloc[2].period_seconds, 15)
        self.assertEqual(home.side.tolist(), [0, 0, 2, 0, 0, 2])
        self.assertFalse(home.iloc[-1].labelled)
        self.assertFalse(home.iloc[2].has_location)

    def test_joint_targets_opponent_and_flips_coordinates_to_shared_frame(self):
        frame, processed, *_ = observations()
        joint = make_streams(frame, "joint")
        row = joint.iloc[1]
        self.assertEqual(row.target, K + BALL_ACTIONS.index("gather"))
        opponent = processed.events.loc[processed.events.display_order.eq(3)].iloc[0]
        self.assertAlmostEqual(row.target_x, -opponent.x_norm)
        self.assertAlmostEqual(row.target_y, -opponent.y_norm)

    def test_quarters_never_join_and_last_target_is_censored(self):
        frame, *_ = observations(True)
        for mode in ("joint", "team_stream"):
            stream = make_streams(frame, mode)
            self.assertTrue(stream.groupby("stream_id").tail(1).target.eq(-1).all())
            self.assertTrue(stream.groupby("stream_id").period.nunique().eq(1).all())

    def test_future_append_preserves_inputs_and_team_stream_prefix(self):
        frame, *_ = observations()
        for mode in ("joint", "team_stream"):
            full = make_streams(frame, mode)
            for count in (2, 4, 6):
                prefix = make_streams(frame.iloc[:count], mode)
                expected = full.set_index(["stream_id", "token_id"])
                for _, row in prefix.iterrows():
                    np.testing.assert_allclose(row[list(NUMERIC)].to_numpy(dtype=float),
                        expected.loc[(row.stream_id, row.token_id), list(NUMERIC)].to_numpy(dtype=float))

    def test_reviewed_actions_remain_history_but_do_not_create_scored_bridges(self):
        frame, processed, metadata, _ = observations()
        positions = processed.possessions.copy()
        bad = processed.events.loc[processed.events.display_order.eq(3), "possession_id"].iloc[0]
        positions.loc[positions.possession_id.eq(bad), ["requires_review", "review_reasons"]] = [True, "unsupported_description"]
        revised, _ = build_quarters(processed.events, positions, processed.annotations, metadata)
        self.assertEqual(len(revised), len(frame))
        stream = make_streams(revised, "joint")
        self.assertFalse(stream.loc[stream.display_order.isin([2, 3, 4])].labelled.any())
        self.assertTrue(stream.display_order.eq(3).any())

    def test_scaler_rejects_test_season_and_unknown_clubs_have_own_id(self):
        frame, *_ = observations()
        stream = make_streams(frame, "joint")
        encoder = EventEncoder.fit(stream)
        with self.assertRaises(ValueError):
            EventEncoder.fit(stream.assign(season=2026))
        self.assertTrue((encoder.transform(stream.assign(home_team_id="NEW"))[3] == 0).all())


class EventModelTests(unittest.TestCase):
    def setUp(self):
        configure("cpu", 3)
        self.observed = observations()[0]

    def setup_data(self, mode):
        stream = make_streams(self.observed, mode)
        encoder = EventEncoder.fit(stream)
        settings = config(mode)
        data = EventData(stream, encoder, mode, max(settings["window"], 1))
        model = EventTransformer(settings, len(encoder.clubs)).eval()
        return model, data, encoder

    def test_causal_outputs_equal_prefix_outputs_and_ignore_future_changes(self):
        model, data, _ = self.setup_data("joint")
        model = EventTransformer(dict(model.config, layers=2), model.num_clubs).eval()
        inputs = next(data.batches(1))[0]
        with torch.inference_mode():
            full = model(*inputs)
            prefix = model(*[x[:, :4] for x in inputs])
            altered = [x.clone() for x in inputs]
            altered[0][:, 4:] = 1
            altered[1][:, 4:] = 100.
            for x in altered[2:]:
                x[:, 4:] = 0
            changed = model(*altered)
        for a, b, c in zip(full, prefix, changed):
            torch.testing.assert_close(a[:, :4], b, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(a[:, :4], c[:, :4], atol=1e-6, rtol=1e-5)

    def test_seq2event_does_not_use_team_inputs(self):
        model, data, _ = self.setup_data("team_stream")
        inputs = next(data.batches(2))[0]
        with torch.inference_mode():
            expected = model(*inputs)
            other = [v.clone() for v in inputs]
            for x in other[2:]:
                x.fill_(99)
            actual = model(*other)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_padding_does_not_change_predictions(self):
        for mode in ("joint", "team_stream"):
            model, data, _ = self.setup_data(mode)
            inputs = next(data.batches(1))[0]
            padded = [torch.nn.functional.pad(x, (0, 0, 0, 3) if x.ndim == 3 else (0, 3)) for x in inputs]
            with torch.inference_mode():
                expected, actual = model(*inputs), model(*padded)
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b[:, :a.shape[1]] if mode == "joint" else b, atol=1e-6, rtol=1e-5)

    def test_full_quarter_has_no_context_cap_and_each_target_visited_once(self):
        source = self.observed.iloc[[0]].copy()
        frame = pd.concat([source] * 550, ignore_index=True)
        frame["token_id"] = [f"t{i}" for i in range(550)]
        frame["display_order"] = frame["position"] = np.arange(550)
        frame["period_seconds"] = np.arange(550)
        stream = make_streams(frame, "joint")
        encoder = EventEncoder.fit(stream)
        data = EventData(stream, encoder, "joint")
        inputs, target, *_, indices = next(data.batches(1))
        self.assertEqual(inputs[0].shape, (1, 550))
        self.assertEqual(int(target.ge(0).sum()), 549)
        self.assertEqual(len(np.unique(indices)), 550)

    def test_seq2event_skips_short_prefixes_and_uses_exact_fixed_window(self):
        _, data, _ = self.setup_data("team_stream")
        self.assertTrue(data.frame.iloc[data.scored_rows].position.ge(1).all())
        inputs, *_ = next(data.batches(20))
        self.assertEqual(inputs[0].shape[1], 2)

    def test_boundary_and_location_masks_have_zero_gradient(self):
        model, data, _ = self.setup_data("team_stream")
        logits = torch.randn(2, K + 1, requires_grad=True)
        xy = torch.randn(2, 2, requires_grad=True)
        loss = objective(logits, xy, torch.full((2,), K), torch.zeros(2, 2), torch.zeros(2, dtype=torch.bool), class_weights(data))
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0.)
        self.assertEqual(float(xy.grad.abs().sum()), 0.)

    def test_export_reload_and_label_free_inference(self):
        for mode in ("joint", "team_stream"):
            model, _, encoder = self.setup_data(mode)
            with tempfile.TemporaryDirectory() as folder:
                save_model(folder, model, encoder, {})
                reloaded, enc, _ = load_model(folder)
                actual = forecast_next(reloaded, enc, self.observed.iloc[:6], focal_side=0)
                expected = forecast_next(model, encoder, self.observed.iloc[:6], focal_side=0)
                self.assertEqual(actual, expected)
                path = Path(folder) / "weights.pt"
                path.write_bytes(path.read_bytes() + b"broken")
                with self.assertRaises(ValueError):
                    load_model(folder)

    def test_training_decreases_loss_and_resume_is_exact(self):
        _, data, _ = self.setup_data("joint")
        settings = config("joint")
        with tempfile.TemporaryDirectory() as folder:
            a, _, _, _ = train(data, None, settings, 2, Path(folder) / "a.pt", seed=42)
            train(data, None, settings, 1, Path(folder) / "b.pt", seed=42)
            b, _, _, _ = train(data, None, settings, 2, Path(folder) / "b.pt", seed=42)
            self.assertEqual(state_fingerprint(a), state_fingerprint(b))
        inputs, target, xy, location, _ = next(data.batches(1))
        model = EventTransformer(settings, 2)
        optimiser = torch.optim.Adam(model.parameters(), lr=.01)
        weights = class_weights(data)
        first = float(objective(*model(*inputs), target, xy, location, weights).detach())
        for _ in range(60):
            optimiser.zero_grad()
            loss = objective(*model(*inputs), target, xy, location, weights)
            loss.backward()
            optimiser.step()
        self.assertLess(float(loss.detach()), first * .2)

    def test_both_models_are_frozen_before_test_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "experiment"
            accessed = []
            def loader(root, seasons=(2023, 2024, 2025)):
                if seasons == (2026,):
                    for mode in ("joint", "team_stream"):
                        self.assertTrue((directory / mode / "model/model.json").is_file())
                    accessed.append("test")
                pieces = []
                for season in seasons:
                    part = self.observed.copy()
                    part["season"] = season
                    part["match_id"] = f"M{season}"
                    part["quarter_id"] = f"M{season}:q1"
                    part["token_id"] = part.token_id.astype(str) + f":{season}"
                    pieces.append(part)
                return QuarterData(pd.concat(pieces, ignore_index=True), {}, {})
            with patch("afl_opposition.event_comparison.load_quarters", loader), \
                 patch("afl_opposition.event_comparison.source_hashes", return_value={}), \
                 patch("afl_opposition.event_comparison.configurations", side_effect=lambda mode: [config(mode)]):
                run(Path(folder), directory, device="cpu", initial_epochs=1, max_epochs=1)
            self.assertEqual(accessed, ["test"])
            self.assertTrue((directory / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
