"""Build app data from the existing decoder; never trains or resumes studies."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from afl_opposition.event_models import load_model, forecast_next, configure
from afl_opposition.quarter_events import load_quarters, make_streams, BALL_ACTIONS
from afl_opposition.opposition import CONTRACT, ACTIONS, PROBS, sha256, load_bundle


def write_bundle(folder, events, matches, meta):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('App bundle already exists. Use a new output directory.')
    folder.mkdir(parents=True, exist_ok=True)
    events.to_parquet(folder / 'events.parquet', index=False)
    matches.to_parquet(folder / 'matches.parquet', index=False)
    meta = dict(meta, files={name:sha256(folder / name) for name in ('events.parquet', 'matches.parquet')})
    (folder / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    load_bundle(folder)


@torch.inference_mode()
def predict_quarter(model, encoder, quarter):
    """All causal prefix forecasts in one pass; labels are not model inputs."""
    frame = make_streams(quarter, 'joint')
    device = next(model.parameters()).device
    inputs = [torch.as_tensor(v, device=device)[None] for v in encoder.transform(frame)]
    logits, xy = model(*inputs)
    p = logits.softmax(-1)[0].cpu().numpy()
    xy = xy[0].cpu().numpy()
    result = frame.copy()
    result[PROBS] = p
    result[['pred_x', 'pred_y']] = xy
    return result


def prepare(root, output, demo_output=None, season=2026):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('App bundle already exists. Use a new output directory.')
    if demo_output and Path(demo_output).exists() and any(Path(demo_output).iterdir()):
        raise ValueError('Demo bundle already exists. Use a new output directory.')
    model_dir = root / 'artifacts/event-optuna-v1/joint/model-seed-2026'
    meta = json.loads((model_dir / 'model.json').read_text())
    trained = max(meta['refit_seasons'])
    if season <= trained:
        raise ValueError('Choose replay matches after the refit seasons.')
    if meta['config']['mode'] != 'joint' or meta['seed'] != 2026:
        raise ValueError('Expected the predeclared first decoder refit, seed 2026.')
    configure('cpu', 2026)
    # Keep inference modest on laptops; no changes to global training settings/files.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model, encoder, info = load_model(model_dir, 'cpu')
    observed = load_quarters(root, seasons=(season,))
    if tuple(BALL_ACTIONS) != ACTIONS:
        raise ValueError('Action order changed.')
    if not (set(observed.events.home_team_id) | set(observed.events.away_team_id)).issubset(encoder.clubs):
        raise ValueError('A club has no trained embedding.')
    manifest = json.loads((root / 'data/sources.json').read_text())
    archive = root / 'data/raw' / manifest['seasons'][str(season)]['filename']
    cols = ['match_id', 'date', 'home_team_id', 'away_team_id', 'home_team_team_name', 'away_team_team_name', 'round_number', 'venue_name']
    matches = pd.read_parquet(archive, columns=cols).drop_duplicates().rename(columns={'home_team_team_name':'home_name','away_team_team_name':'away_name'})
    if matches.match_id.duplicated().any():
        raise ValueError('Inconsistent match metadata.')
    pieces = []
    for i, (_, quarter) in enumerate(observed.events.groupby('quarter_id', sort=False)):
        quarter = quarter.reset_index(drop=True)
        predicted = predict_quarter(model, encoder, quarter)
        # Verify batch inference equals a true truncated prefix, independently of later observations.
        if i in (0, 100):
            n = min(50, len(quarter))
            direct = forecast_next(model, encoder, quarter.iloc[:n])
            probs = direct['next_event_probabilities']
            expected = [*probs['home'].values(), *probs['away'].values(), probs['boundary']]
            np.testing.assert_allclose(predicted.iloc[n - 1][PROBS].to_numpy(dtype=float), expected, atol=2e-6)
            np.testing.assert_allclose(predicted.iloc[n - 1][['pred_x','pred_y']].to_numpy(dtype=float), direct['location'], atol=2e-6)
        pieces.append(predicted)
        if (i + 1) % 100 == 0:
            print(f'Prepared {i + 1} quarters on CPU.', flush=True)
    events = pd.concat(pieces, ignore_index=True)
    matches['quarters_available'] = matches.match_id.map(events.groupby('match_id').period.nunique()).fillna(0).astype(int)
    matches = matches.loc[matches.quarters_available.gt(0)].sort_values(['date','match_id']).reset_index(drop=True)
    base = dict(contract=CONTRACT, actions=list(ACTIONS), seed=2026, trained_through=f'{trained}-12-31',
        model_fingerprint=info['fingerprint'], weights_sha256=info['weights_sha256'], created_utc=datetime.now(timezone.utc).isoformat(),
        data_provenance=observed.provenance, source_audit=observed.audit,
        prediction_policy='Frozen first seed (2026), selected independently of test performance. Full-quarter causal inference verified against truncated prefixes.',
        selected_seed_reason='Predeclared first seed; not selected by test score.')
    write_bundle(output, events, matches, dict(base, bundle_id=f'{season}-archive-seed2026', coverage='archive',
        coverage_note=f'{season} historical archive. Only recorded, non-quarantined quarters are available; inspect the match coverage table.'))
    if demo_output:
        # Deterministic coverage sample: earliest available quarter in each club's earliest match.
        quarter_ids = set()
        for club in sorted(set(matches.home_team_id) | set(matches.away_team_id)):
            first = matches.loc[matches.home_team_id.eq(club) | matches.away_team_id.eq(club)].iloc[0]
            quarter_ids.add(events.loc[events.match_id.eq(first.match_id), 'quarter_id'].iloc[0])
        demo_events = events.loc[events.quarter_id.isin(quarter_ids)].copy()
        demo_matches = matches.loc[matches.match_id.isin(demo_events.match_id)].copy()
        demo_matches['quarters_available'] = demo_matches.match_id.map(demo_events.groupby('match_id').period.nunique())
        write_bundle(demo_output, demo_events, demo_matches, dict(base, bundle_id=f'{season}-sample-seed2026', coverage='sample',
            coverage_note='Demonstration sample: the earliest available quarter in each club\'s earliest archived match. Every club is represented, but these are partial matches, not a recent-form briefing.'))
    print(f'App bundle ready: {len(matches)} matches, {events.quarter_id.nunique()} quarters, {len(events):,} events. No training performed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/opposition-app-v1')
    parser.add_argument('--demo-output')
    parser.add_argument('--season', type=int, default=2026)
    args = parser.parse_args()
    prepare(Path.cwd(), args.output, args.demo_output, args.season)


if __name__ == '__main__':
    main()
