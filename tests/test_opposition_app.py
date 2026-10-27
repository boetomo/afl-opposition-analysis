"""Information boundaries, report selection and browser-app interactions."""
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from afl_opposition.opposition import (
    Bundle, PROBS, clubs, load_bundle, select_matches, review_rows, tendencies,
    pitch, report_html, predictions, sha256,
)
from afl_opposition.prepare_opposition import predict_quarter
from afl_opposition.event_models import EventEncoder, EventTransformer, forecast_next, configure
from afl_opposition.quarter_events import make_streams
from test_event_comparison import observations, config

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'reports/examples/opposition-demo'


class OppositionModelTests(unittest.TestCase):
    def test_batch_forecasts_match_true_prefixes_and_ignore_future(self):
        configure('cpu', 2026)
        quarter = observations()[0]
        encoder = EventEncoder.fit(make_streams(quarter, 'joint'))
        model = EventTransformer(config('joint'), len(encoder.clubs)).eval()
        batch = predict_quarter(model, encoder, quarter)
        for count in (1, 4, len(quarter)):
            result = forecast_next(model, encoder, quarter.iloc[:count])
            p = result['next_event_probabilities']
            expected = [*p['home'].values(), *p['away'].values(), p['boundary']]
            np.testing.assert_allclose(batch.iloc[count-1][PROBS].to_numpy(float), expected, atol=2e-6)
        changed = quarter.copy()
        changed.loc[changed.index[4:], 'x_home'] += .5
        altered = predict_quarter(model, encoder, changed)
        np.testing.assert_allclose(batch.iloc[:4][PROBS], altered.iloc[:4][PROBS], atol=2e-6)


@unittest.skipUnless((DEMO / 'manifest.json').exists(), 'Build the sample app bundle first')
class OppositionReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(DEMO)
        cls.club = next(iter(clubs(cls.bundle)))
        cls.cutoff = max(cls.bundle.matches.date) + timedelta(days=1)
        cls.matches = select_matches(cls.bundle, cls.club, cls.cutoff)

    def test_every_club_has_real_matches_and_probabilities(self):
        self.assertEqual(len(clubs(self.bundle)), 18)
        for club in clubs(self.bundle):
            selected = select_matches(self.bundle, club, self.cutoff)
            self.assertFalse(selected.empty)
        np.testing.assert_allclose(self.bundle.events[PROBS].sum(axis=1), 1, atol=1e-5)

    def test_strict_cutoff_and_team_membership(self):
        first_date = min(self.matches.date)
        self.assertTrue(select_matches(self.bundle, self.club, first_date).empty)
        selected = select_matches(self.bundle, self.club, self.cutoff, 1)
        self.assertEqual(len(selected), 1)
        self.assertTrue((selected.home_team_id.eq(self.club) | selected.away_team_id.eq(self.club)).all())
        with self.assertRaises(ValueError):
            report_html(self.bundle, selected, self.club, first_date)

    def test_away_probabilities_and_quality_masks(self):
        match = self.matches.iloc[0]
        club = match.away_team_id
        selected = select_matches(self.bundle, club, self.cutoff)
        rows = review_rows(self.bundle, selected, club)
        away = rows.loc[rows.focal_side.eq(1)]
        self.assertFalse(away.empty)
        np.testing.assert_allclose(away.p_own_kick, away.p18)
        np.testing.assert_allclose(away.p_other, away[PROBS[:12]].sum(axis=1))
        self.assertTrue(rows.labelled.all())
        self.assertEqual(int(tendencies(rows).Examples.sum()), len(rows))

    def test_replay_hides_actual_until_requested_and_rotates_away(self):
        match = self.matches.iloc[0]
        quarter_id = self.bundle.events.loc[self.bundle.events.match_id.eq(match.match_id), 'quarter_id'].iloc[0]
        quarter = self.bundle.events.loc[self.bundle.events.quarter_id.eq(quarter_id)].reset_index(drop=True)
        pos = next(i for i in range(len(quarter)-1) if quarter.iloc[i].has_coordinates and quarter.iloc[i+1].has_coordinates)
        hidden = pitch(quarter, pos, match, match.away_team_id, reveal=False)
        self.assertFalse(any(t.name == 'Actual next location' for t in hidden.data))
        current = next(t for t in hidden.data if t.name == 'Current event')
        self.assertAlmostEqual(current.x[0], -quarter.iloc[pos].x_home)
        revealed = pitch(quarter, pos, match, match.away_team_id, reveal=True)
        actual = next(t for t in revealed.data if t.name == 'Actual next location')
        self.assertAlmostEqual(actual.y[0], -quarter.iloc[pos+1].y_home)
        pred = predictions(quarter.iloc[pos], match)
        self.assertEqual(len(pred), 25)
        self.assertAlmostEqual(pred.Probability.sum(), 1, places=5)

    def test_report_is_standalone_and_preserves_scope(self):
        html = report_html(self.bundle, self.matches, self.club, self.cutoff)
        self.assertIn(clubs(self.bundle)[self.club], html)
        self.assertIn(str(self.cutoff), html)
        self.assertIn('Demonstration sample', html)
        self.assertIn('Observed kick %', html)
        self.assertIn('Model kick %', html)
        self.assertIn('Plotly.newPlot', html)
        self.assertNotIn('src="https://cdn.plot.ly', html)
        self.assertNotIn('C:\\Users', html)
        with self.assertRaises(ValueError):
            report_html(self.bundle, self.matches.iloc[:0], self.club, self.cutoff)
        unsafe = self.bundle.matches.copy()
        unsafe.loc[unsafe.home_team_id.eq(self.club),'home_name'] = '<script>alert(1)</script>'
        unsafe.loc[unsafe.away_team_id.eq(self.club),'away_name'] = '<script>alert(1)</script>'
        modified = Bundle(self.bundle.events, unsafe, self.bundle.manifest)
        report = report_html(modified, select_matches(modified,self.club,self.cutoff),self.club,self.cutoff)
        self.assertNotIn('<h1><script>', report)
        self.assertIn('&lt;script&gt;', report)

    def test_checksum_rejects_modified_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'manifest.json').write_bytes((DEMO/'manifest.json').read_bytes())
            (folder/'events.parquet').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_bundle(folder)


@unittest.skipUnless((DEMO / 'manifest.json').exists(), 'Build the sample app bundle first')
class OppositionAppTests(unittest.TestCase):
    def test_selection_replay_report_and_empty_dates(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict('os.environ', {'AFL_APP_BUNDLE':str(DEMO)}):
            app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.selectbox(key='club').options), 18)
            self.assertTrue(app.checkbox(key='reveal').value)
            self.assertTrue(any('Actual next' in i.value for i in app.info))
            app.checkbox(key='reveal').uncheck().run()
            self.assertFalse(any('Actual next' in i.value for i in app.info))
            app.button(key='next').click().run()
            self.assertEqual(app.slider(key='event_number').value, 2)
            self.assertFalse(app.checkbox(key='reveal').value)
            app.checkbox(key='reveal').check().run()
            self.assertTrue(any('Actual next' in i.value for i in app.info))
            app.button(key='generate').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.download_button), 1)
            self.assertIn('Opposition briefing', app.session_state['report_html'])
            bundle = load_bundle(DEMO)
            last = list(clubs(bundle))[-1]
            app.checkbox(key='reveal').uncheck().run()
            app.selectbox(key='club').set_value(last).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.slider(key='event_number').value, 1)
            self.assertTrue(app.checkbox(key='reveal').value)
            self.assertTrue(any('Actual next' in i.value for i in app.info))
            self.assertEqual(len(app.download_button), 0)  # no stale report for the previous team
            app.date_input(key='cutoff').set_value(date(2026,1,1)).run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('No recorded matches' in i.value for i in app.info))


if __name__ == '__main__':
    unittest.main()
