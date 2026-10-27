"""Portable opposition views and reports over a frozen, precomputed decoder bundle.

This module has no PyTorch dependency and cannot train or resume a model.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

CONTRACT = 'opposition-app-v1'
ACTIONS = ('bounce', 'free_kick', 'gather', 'ground_kick', 'handball', 'hard_ball_get',
           'kick', 'kick_in', 'kick_in_play_on', 'loose_ball_get', 'mark', 'receive')
PROBS = [f'p{i}' for i in range(25)]
COLOURS = ('#0d9488', '#e87935')


@dataclass
class Bundle:
    events: pd.DataFrame
    matches: pd.DataFrame
    manifest: dict


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(folder):
    folder = Path(folder)
    meta = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if meta['contract'] != CONTRACT or meta['actions'] != list(ACTIONS):
        raise ValueError('Unsupported app data format. Rebuild the bundle.')
    for name in ('events.parquet', 'matches.parquet'):
        if sha256(folder / name) != meta['files'][name]:
            raise ValueError('App data checksum differs. Rebuild the bundle.')
    events, matches = (pd.read_parquet(folder / name) for name in ('events.parquet', 'matches.parquet'))
    if matches.match_id.duplicated().any() or events.token_id.duplicated().any():
        raise ValueError('Duplicate match or event IDs.')
    if not set(events.match_id).issubset(set(matches.match_id)):
        raise ValueError('Missing match metadata.')
    p = events[PROBS].to_numpy()
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid prediction probabilities.')
    if not np.array_equal(events.groupby('quarter_id', sort=False).cumcount(), events.position):
        raise ValueError('Quarter histories must be complete and ordered.')
    if not np.isfinite(events[['pred_x', 'pred_y']].to_numpy()).all():
        raise ValueError('Invalid predicted coordinates.')
    matches['date'] = pd.to_datetime(matches.date).dt.date
    if min(matches.date).isoformat() <= meta['trained_through']:
        raise ValueError('Replay data overlaps the model training period.')
    return Bundle(events, matches, meta)


def clubs(bundle):
    result = {}
    for side in ('home', 'away'):
        for row in bundle.matches.itertuples():
            result[getattr(row, f'{side}_team_id')] = getattr(row, f'{side}_name')
    return dict(sorted(result.items(), key=lambda pair: pair[1]))


def select_matches(bundle, club, cutoff, count=5):
    """Dates strictly before cutoff; count=None includes every matching game."""
    if club not in clubs(bundle) or (count is not None and not 1 <= count <= 30):
        raise ValueError('Choose a known team and between one and 30 matches, or all matches.')
    selected = bundle.matches.loc[(bundle.matches.home_team_id.eq(club) | bundle.matches.away_team_id.eq(club)) & bundle.matches.date.lt(cutoff)]
    selected = selected.sort_values(['date', 'match_id'], ascending=False)
    return selected if count is None else selected.head(count)


def side_for(match, club):
    if match.home_team_id == club:
        return 0
    if match.away_team_id == club:
        return 1
    raise ValueError('Team is not part of the selected match.')


def label(code, home, away):
    code = int(code)
    if code == 24:
        return 'Boundary / stoppage'
    if code < 0:
        return 'Unknown / no next observation'
    return f'{home if code < 12 else away} · {ACTIONS[code % 12].replace("_", " ")}'


def clock(seconds):
    return f'{int(seconds) // 60:02d}:{int(seconds) % 60:02d}'


def predictions(row, match):
    return pd.DataFrame({'Next event': [label(i, match.home_name, match.away_name) for i in range(25)],
                         'Probability': row[PROBS].to_numpy(dtype=float)}).sort_values('Probability', ascending=False).reset_index(drop=True)


def review_rows(bundle, matches, club):
    """Comparable current-team observations with a scored, known next event."""
    selected = bundle.events.loc[bundle.events.match_id.isin(matches.match_id)].copy()
    sides = {r.match_id: side_for(r, club) for r in matches.itertuples()}
    selected['focal_side'] = selected.match_id.map(sides)
    selected = selected.loc[selected.side.eq(selected.focal_side) & selected.labelled].copy()
    if selected.empty:
        return selected
    selected['zone'] = np.select([selected.x_home * np.where(selected.focal_side.eq(0), 1, -1) < -1 / 3,
                                  selected.x_home * np.where(selected.focal_side.eq(0), 1, -1) > 1 / 3],
                                 ['Defensive third', 'Forward third'], default='Middle third')
    selected['p_own_kick'] = np.where(selected.focal_side.eq(0), selected.p6, selected.p18)
    selected['p_own_handball'] = np.where(selected.focal_side.eq(0), selected.p4, selected.p16)
    selected['p_other'] = np.where(selected.focal_side.eq(0), selected[PROBS[12:24]].sum(axis=1), selected[PROBS[:12]].sum(axis=1))
    selected['own_kick'] = selected.target.eq(selected.focal_side * 12 + 6)
    selected['own_handball'] = selected.target.eq(selected.focal_side * 12 + 4)
    selected['other'] = selected.target.lt(24) & selected.target.floordiv(12).ne(selected.focal_side)
    return selected


def tendencies(rows):
    cols = ['Zone', 'Current action', 'Examples', 'Observed kick %', 'Model kick %',
            'Observed handball %', 'Model handball %', 'Observed other-team %', 'Model other-team %']
    if rows.empty:
        return pd.DataFrame(columns=cols)
    grouped = rows.groupby(['zone', 'action_type'], sort=True).agg(
        n=('token_id', 'size'), kick=('own_kick', 'mean'), pkick=('p_own_kick', 'mean'),
        handball=('own_handball', 'mean'), phandball=('p_own_handball', 'mean'),
        other=('other', 'mean'), pother=('p_other', 'mean')).reset_index()
    grouped['action_type'] = grouped.action_type.str.replace('_', ' ')
    grouped.iloc[:, 3:] = grouped.iloc[:, 3:] * 100
    grouped.columns = cols
    return grouped.sort_values('Examples', ascending=False).reset_index(drop=True)


def example_rows(bundle, matches, club, count=6):
    rows = review_rows(bundle, matches, club)
    if rows.empty:
        return rows
    # One passage per match before cycling to another; chronological, not chosen for accuracy.
    rows = rows.sort_values(['match_id', 'period', 'position'])
    rows['within_match'] = rows.groupby('match_id').cumcount()
    return rows.sort_values(['within_match', 'match_id']).head(count)


def pitch(quarter, position, match, club, reveal=False, trail=12):
    """Draw only observed points; the future observation is an explicit opt-in."""
    row = quarter.iloc[position]
    focal = side_for(match, club)
    sign = 1 if focal == 0 else -1
    history = quarter.iloc[max(0, position - trail + 1):position + 1]
    theta = np.linspace(0, 2 * np.pi, 150)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
        line=dict(color='#a6bdb4', width=2), fill='toself', fillcolor='#f1f7f3', hoverinfo='skip', showlegend=False))
    fig.add_shape(type='circle', x0=-.11, x1=.11, y0=-.16, y1=.16, line=dict(color='#b6c9bf'))
    fig.add_shape(type='line', x0=0, x1=0, y0=-1, y1=1, line=dict(color='#d7e3db', dash='dot'))
    # Never draw a continuous route through an unknown-location token.
    x = [float(r.x_home * sign) if r.has_coordinates else None for r in history.itertuples()]
    y = [float(r.y_home * sign) if r.has_coordinates else None for r in history.itertuples()]
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#a4b4b0', width=2), connectgaps=False, hoverinfo='skip', showlegend=False))
    for side in (focal, 1 - focal):
        part = history.loc[history.side.eq(side) & history.has_coordinates.eq(1)]
        name = match.home_name if side == 0 else match.away_name
        fig.add_trace(go.Scatter(x=(part.x_home * sign).tolist(), y=(part.y_home * sign).tolist(), mode='markers',
            name=name, marker=dict(color=COLOURS[int(side != focal)], size=10),
            text=[f'{clock(r.period_seconds)} · {r.action_type.replace("_", " ")}' for r in part.itertuples()],
            hovertemplate='%{text}<extra>%{fullData.name}</extra>'))
    if row.has_coordinates:
        fig.add_trace(go.Scatter(x=[row.x_home * sign], y=[row.y_home * sign], mode='markers', name='Current event',
            marker=dict(symbol='circle-open', size=22, line=dict(width=3, color='#172c38')), hoverinfo='name'))
    if bool(row.eligible) and row.action_type != 'gap':
        fig.add_trace(go.Scatter(x=[float(row.pred_x * sign)], y=[float(row.pred_y * sign)], mode='markers',
            name='Predicted location', marker=dict(symbol='x', size=14, color='#7c3aed'), hoverinfo='name'))
    if reveal and position + 1 < len(quarter):
        actual = quarter.iloc[position + 1]
        if actual.has_coordinates:
            fig.add_trace(go.Scatter(x=[actual.x_home * sign], y=[actual.y_home * sign], mode='markers',
                name='Actual next location', marker=dict(symbol='diamond-open', size=16, color='#101e2c', line=dict(width=2)), hoverinfo='name'))
    fig.update_layout(height=415, margin=dict(l=10, r=10, t=22, b=22), paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Arial', color='#243c49'), legend=dict(orientation='h', y=-.07, x=0),
        xaxis=dict(range=[-1.25, 1.25], visible=False), yaxis=dict(range=[-1.2, 1.2], visible=False, scaleanchor='x', scaleratio=.62),
        annotations=[dict(x=.99, y=1.1, text=f'{escape(club_name(match, club))} attacks →', showarrow=False, xanchor='right')],
        uirevision='pitch')
    return fig


def club_name(match, club):
    return match.home_name if side_for(match, club) == 0 else match.away_name


def report_html(bundle, matches, club, cutoff):
    """A self-contained briefing with evidence tables and predetermined replays."""
    if matches.empty:
        raise ValueError('Select at least one match before exporting a report.')
    # The caller cannot bypass the date/team filter or supply unrelated match rows.
    allowed = select_matches(bundle, club, cutoff, count=None)
    if not set(matches.match_id).issubset(set(allowed.match_id)):
        raise ValueError('Report includes a match outside the team/date selection.')
    name = clubs(bundle)[club]
    rows = review_rows(bundle, matches, club)
    table = tendencies(rows)
    meta = bundle.manifest
    match_table = matches[['date', 'home_name', 'away_name', 'round_number', 'quarters_available']].rename(columns={
        'date':'Date', 'home_name':'Home', 'away_name':'Away', 'round_number':'Round', 'quarters_available':'Recorded quarters'})
    blocks = []
    for i, event in enumerate(example_rows(bundle, matches, club).itertuples()):
        match = matches.loc[matches.match_id.eq(event.match_id)].iloc[0]
        quarter = bundle.events.loc[bundle.events.quarter_id.eq(event.quarter_id)].reset_index(drop=True)
        title = f'{match.date} · {match.home_name} v {match.away_name} · Q{event.period} {clock(event.period_seconds)}'
        forecast = predictions(quarter.iloc[event.position], match).head(5)
        forecast['Probability'] = forecast.Probability.map(lambda v:f'{v:.1%}')
        actual = label(event.target, match.home_name, match.away_name)
        blocks.append(f'<section><h3>{escape(title)}</h3><p>After {escape(event.action_type.replace("_", " "))}. Actual next: {escape(actual)}.</p>'
            + pitch(quarter, event.position, match, club, reveal=True).to_html(full_html=False, include_plotlyjs=True if i == 0 else False,
                  config={'displayModeBar':False}, div_id=f'passage-{i}')
            + forecast.to_html(index=False, escape=True) + '</section>')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(name)} opposition briefing</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#18313d;background:#f5f7f7;margin:0}}main{{max-width:1100px;margin:36px auto;padding:36px;background:white;border-radius:16px}}
h1,h2,h3{{line-height:1.25}}h1{{font-size:38px}}h2{{margin-top:40px}}.eyebrow{{color:#087e75;text-transform:uppercase;letter-spacing:.12em;font-size:12px}}.note{{background:#f0f5f5;padding:16px;border-left:4px solid #0d9488}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #dce5e5;text-align:left}}th{{background:#f2f6f5}}section{{break-inside:avoid;border-top:1px solid #dce5e5;margin-top:24px;padding-top:12px}}.scroll{{overflow-x:auto}}@media print{{body{{background:white}}main{{margin:0;padding:0}}}}
</style></head><body><main><p class="eyebrow">AFL Opposition Analysis · Historical match review</p>
<h1>{escape(name)}<br>Opposition briefing</h1><p>Matches before {escape(str(cutoff))} · {len(matches)} matches · {matches.quarters_available.sum()} recorded quarters · {len(rows):,} eligible transitions after this team's actions</p>
<p class="note">{escape(meta['coverage_note'])} This report describes recorded passages and model estimates. It does not prescribe a defensive tactic or represent a live feed.</p>
<h2>Match coverage</h2><div class="scroll">{match_table.to_html(index=False,escape=True)}</div>
<h2>What happens next?</h2><p>Each row groups the current action and field third, in {escape(name)}'s attacking direction. Observed percentages are historical frequencies. Model percentages average the decoder's probabilities on the same passages. Kick and handball mean this team's next recorded action. Other-team means the next recorded action belongs to the opposition, not a complete turnover-risk estimate. Unlisted actions and boundaries make up the remaining probability.</p>
<div class="scroll">{table.to_html(index=False, escape=True, float_format=lambda value: format(value, '.1f'))}</div>
<p>Small groups are unstable. This is an event-weighted description of these matches, not a comparison that controls for opposition, score or field position.</p>
<h2>Passages to inspect</h2><p>Up to six examples are selected in chronological order, taking one per match before taking additional passages. They are not selected for prediction accuracy. The map shows recent observations; predictions use all observed history in the quarter. The purple cross is one point estimate for an ordinary next action, shared across all possible actions.</p>
{''.join(blocks) if blocks else '<p>No eligible passages in the selected coverage.</p>'}
<h2>Model and data</h2><p>Frozen tuned decoder-only Transformer, seed {meta['seed']}; trained through {escape(meta['trained_through'])}. Model fingerprint: <code>{escape(meta['model_fingerprint'])}</code>. Predictions computed once with causal attention and no training in the app. Data quality warnings are identified during review of historical records. Affected transitions are excluded from this report. The 2026 benchmark has already been observed. Rare-action recall remains weak.</p>
<p>Source: <a href="https://github.com/peteowen1/torpdata/releases/tag/chains-data">torpdata match-event archives</a>. Bundle: {escape(meta['bundle_id'])}. Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.</p></main></body></html>'''
