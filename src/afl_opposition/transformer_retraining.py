"""Broader, multi-seed validation search and fixed-update Transformer refits.

Run from the repository root:
python -m afl_opposition.transformer_retraining [--device auto|cpu|cuda] [--resume]
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import torch

from afl_opposition.goal_tuning import load_goal_inputs
from afl_opposition.xgboost_baseline import build_datasets
from afl_opposition.transformer_experiment import (
    SOURCE_NAMES as ORIGINAL_SOURCES, probability_report as original_report,
    _digest, _json, _save_tables,
)
from afl_opposition.transformer_training import (
    EFFECTIVE_BATCH_SIZE, MICROBATCH_SIZE, EVALUATION_BATCH_SIZE,
    SequenceEncoder, make_sequences, PossessionTransformer, configure_runtime,
    fit_updates, evaluate_summary, predict, save_model, load_model, state_fingerprint,
)

SEEDS = (2026, 2027, 2028)
SCREEN_STEPS = 300
FULL_STEPS = 3200
EVALUATE_EVERY = 100
SOURCE_NAMES = (*ORIGINAL_SOURCES, 'transformer_training.py', 'transformer_retraining.py')
SPACE = dict(num_layers=[1,2,3], d_model=[32,64,128], ff_multiplier=[2,4],
    embedding_dim=[4,8,16], continuous_dim=[8,16,32], dropout=[0.,.1,.2,.3],
    learning_rate=[.0001,.0003,.001,.003], lambda_l2=[0.,1e-7,1e-6,1e-5,1e-4],
    warmup_fraction=[0.,.05,.1])


def candidate_configs(count=64):
    if count < 16:
        raise ValueError('Use at least 16 candidates for balanced coverage.')
    rng = np.random.default_rng(20261014)
    columns = {key:rng.permutation(np.resize(values,count)).tolist() for key,values in SPACE.items()}
    candidates = [{key:values[i] for key,values in columns.items()} for i in range(count)]
    for c in candidates:
        heads = [h for h in [1,2,4,8] if c['d_model'] % h == 0 and 8 <= c['d_model']//h <= 64]
        c['num_heads'] = int(rng.choice(heads))
    candidates[:4] = [
        dict(num_layers=3,d_model=32,num_heads=4,ff_multiplier=4,embedding_dim=4,continuous_dim=16,
             dropout=.1,learning_rate=.003,lambda_l2=1e-6,warmup_fraction=.05),
        dict(num_layers=3,d_model=128,num_heads=4,ff_multiplier=2,embedding_dim=4,continuous_dim=16,
             dropout=0.,learning_rate=.001,lambda_l2=1e-6,warmup_fraction=.05),
        dict(num_layers=1,d_model=32,num_heads=2,ff_multiplier=2,embedding_dim=8,continuous_dim=16,
             dropout=0.,learning_rate=.001,lambda_l2=1e-6,warmup_fraction=.05),
        dict(num_layers=2,d_model=64,num_heads=4,ff_multiplier=2,embedding_dim=8,continuous_dim=16,
             dropout=.1,learning_rate=.0003,lambda_l2=1e-6,warmup_fraction=.05),
    ]
    if len({json.dumps(c,sort_keys=True) for c in candidates}) != count:
        raise ValueError('Search candidates must be unique.')
    for key,values in SPACE.items():
        if {c[key] for c in candidates} != set(values):
            raise ValueError(f'Incomplete candidate coverage: {key}')
    return candidates


def promote_candidates(screen, count=8):
    """Top four plus protected slow-rate, compact, deep and wide candidates."""
    ranked = screen.sort_values(['validation_bce','parameters','trial'])
    chosen = ranked.trial.head(min(4,count)).tolist()
    strata = [ranked.learning_rate.le(.0003), ranked.d_model.eq(32),
              ranked.num_layers.eq(3), ranked.d_model.eq(128)]
    for mask in strata:
        remaining = ranked.loc[mask & ~ranked.trial.isin(chosen)]
        if len(chosen) < count and not remaining.empty:
            chosen.append(int(remaining.iloc[0].trial))
    for trial in ranked.trial:
        if len(chosen) >= count:
            break
        if trial not in chosen:
            chosen.append(int(trial))
    return chosen


def probability_report(examples, probabilities, constant, target, split, model_name):
    tables = original_report(examples, probabilities, constant, target, 'history', split)
    for name in ['metrics','confusion','calibration']:
        tables[name]['model'] = tables[name].model.replace({'seq2event_transformer':model_name})
    return tables


def best_curve_row(curve, evaluate_every):
    # Avoid selecting a checkpoint before a full refit pass on these seasons.
    minimum = min(2*evaluate_every,int(curve.step.max()))
    return curve.loc[curve.step.ge(minimum)].sort_values(['validation_bce','step']).iloc[0]


def run_candidate(train, validation, config, vocabulary_size, directory, *, budget, seed,
                  evaluate_every, progress=None, batching='random', schedule='cosine'):
    directory.mkdir(parents=True, exist_ok=True)
    model, history = fit_updates(train, validation, config, vocabulary_size,
        steps=budget, horizon=budget, seed=seed, checkpoint=directory/'training.pt',
        snapshot_directory=directory/'snapshots', evaluate_every=evaluate_every,
        batching=batching, schedule=schedule, progress=progress)
    curve = pd.DataFrame(history)
    _save_tables(directory, {'learning_curve':curve})
    parameters = sum(p.numel() for p in model.parameters())
    del model
    return curve, parameters


def select_model(examples, target, directory, *, device='cpu', candidates=None,
                 screen_steps=SCREEN_STEPS, full_steps=FULL_STEPS,
                 evaluate_every=EVALUATE_EVERY, seeds=SEEDS, promote_count=8,
                 finalist_count=3, progress=print, diagnostics=True):
    if set(examples.season) != {2023,2024,2025}:
        raise ValueError('Selection requires 2023-2025 only; test inputs are forbidden.')
    candidates = candidate_configs() if candidates is None else candidates
    directory.mkdir(parents=True, exist_ok=True)
    train_examples = examples.loc[examples.season.lt(2025)]
    validation_examples = examples.loc[examples.season.eq(2025)]
    encoder = SequenceEncoder.fit(train_examples)
    train = make_sequences(train_examples,encoder).to(device)
    validation = make_sequences(validation_examples,encoder).to(device)
    vocabulary_size = len(encoder.categories)+2
    constant = float(train.target.double().mean().item())
    def announce(text):
        if progress:
            progress(text)
    def callback(label):
        def show(row):
            if row['step'] % (4*evaluate_every) == 0:
                announce(f"{label}: update {row['step']}, validation BCE={row['validation_bce']:.6f}, "
                         f"training audit gap={row['training_audit_calibration_gap']:+.4f}")
        return show
    if diagnostics:
        config = candidates[0]
        diagnostic_rows = []
        # A matched 2x2 diagnostic, not a second model-selection criterion.
        for batching in ['bucketed','random']:
            for schedule in ['constant','cosine']:
                label = f'{target} diagnostic {batching}/{schedule}'
                announce(label)
                curve, _ = run_candidate(train,validation,config,vocabulary_size,
                    directory/'diagnostics'/f'{batching}-{schedule}',budget=600,seed=seeds[0],
                    evaluate_every=evaluate_every,progress=callback(label),batching=batching,schedule=schedule)
                diagnostic_rows.append(curve.assign(batching=batching,schedule=schedule))
        _save_tables(directory/'diagnostics',{'comparison':pd.concat(diagnostic_rows,ignore_index=True)})
    screen_records = []
    for trial,config in enumerate(candidates):
        label = f'{target} screen {trial+1}/{len(candidates)}'
        announce(label)
        curve, parameters = run_candidate(train,validation,config,vocabulary_size,
            directory/'screen'/f'trial-{trial:02d}',budget=screen_steps,seed=seeds[0],
            evaluate_every=evaluate_every)
        best = best_curve_row(curve,evaluate_every)
        screen_records.append(dict(trial=trial,**config,parameters=parameters,
            validation_bce=float(best.validation_bce),best_step=int(best.step)))
        _save_tables(directory,{'screen':pd.DataFrame(screen_records)})
        announce(f"{label} complete: BCE={best.validation_bce:.6f}")
    screen = pd.DataFrame(screen_records)
    promoted = promote_candidates(screen,min(promote_count,len(candidates)))
    screen['promoted'] = screen.trial.isin(promoted)
    _save_tables(directory,{'screen':screen})
    full_records, curves = [], {}
    for trial in promoted:
        label = f'{target} full trial {trial+1}, seed {seeds[0]}'
        announce(label+' (fresh initialisation)')
        curve, parameters = run_candidate(train,validation,candidates[trial],vocabulary_size,
            directory/'full'/f'trial-{trial:02d}'/f'seed-{seeds[0]}',budget=full_steps,seed=seeds[0],
            evaluate_every=evaluate_every,progress=callback(label))
        curves[(trial,seeds[0])] = curve
        best = best_curve_row(curve,evaluate_every)
        full_records.append(dict(trial=trial,**candidates[trial],parameters=parameters,
            validation_bce=float(best.validation_bce),best_step=int(best.step)))
        _save_tables(directory,{'full_search':pd.DataFrame(full_records)})
    full = pd.DataFrame(full_records)
    finalists = full.sort_values(['validation_bce','parameters','trial']).trial.head(finalist_count).tolist()
    full['finalist'] = full.trial.isin(finalists)
    _save_tables(directory,{'full_search':full})
    for trial in finalists:
        for seed in seeds[1:]:
            label = f'{target} finalist {trial+1}, seed {seed}'
            announce(label)
            curve, _ = run_candidate(train,validation,candidates[trial],vocabulary_size,
                directory/'full'/f'trial-{trial:02d}'/f'seed-{seed}',budget=full_steps,seed=seed,
                evaluate_every=evaluate_every,progress=callback(label))
            curves[(trial,seed)] = curve
    all_curves = pd.concat([curves[(t,s)].assign(trial=t,seed=s) for t in finalists for s in seeds],ignore_index=True)
    aggregate = all_curves.groupby(['trial','step'],as_index=False).agg(
        validation_bce=('validation_bce','mean'),seed_std=('validation_bce','std'),
        seed_count=('seed','nunique'),mean_calibration_gap=('validation_calibration_gap','mean'))
    assert aggregate.seed_count.eq(len(seeds)).all()
    aggregate = aggregate.merge(full[['trial','parameters']],on='trial',validate='many_to_one')
    minimum_step = min(2*evaluate_every,full_steps)
    choice = aggregate.loc[aggregate.step.ge(minimum_step)].sort_values(
        ['validation_bce','parameters','trial','step']).iloc[0]
    trial, steps = int(choice.trial), int(choice.step)
    aggregate['selected'] = aggregate.trial.eq(trial) & aggregate.step.eq(steps)
    _save_tables(directory,{'seed_learning_curves':all_curves,'seed_selection':aggregate})
    winner = dict(trial=trial,config=candidates[trial],steps=steps,schedule_horizon=full_steps,
        seeds=list(seeds),mean_validation_bce=float(choice.validation_bce),
        validation_seed_std=float(choice.seed_std) if len(seeds)>1 else 0.,
        parameters=int(choice.parameters),training_possessions=len(train.lengths),
        validation_possessions=len(validation.lengths),criterion='mean 2025 BCE across seeds at a common update',
        training_encoder=encoder.as_dict())
    diagnostics_rows = []
    for seed in seeds:
        configure_runtime(device,seed)
        model = PossessionTransformer(vocabulary_size,candidates[trial],constant).to(device)
        snapshot = directory/'full'/f'trial-{trial:02d}'/f'seed-{seed}'/'snapshots'/f'step-{steps:05d}.pt'
        model.load_state_dict(torch.load(snapshot,map_location=device,weights_only=True))
        model.eval()
        save_model(directory/'selection'/f'seed-{seed}',model,encoder,candidates[trial],target,constant,steps,full_steps,seed)
        diagnostics_rows.append(dict(seed=seed,**evaluate_summary(model,train)))
        tables = probability_report(validation_examples,predict(model,validation,EVALUATION_BATCH_SIZE),
                                    constant,target,'validation',f'transformer_seed_{seed}')
        _save_tables(directory/'validation'/f'seed-{seed}',tables)
        del model
    _save_tables(directory,{'selection_training_diagnostics':pd.DataFrame(diagnostics_rows)})
    _json(directory/'selection.json',winner)
    announce(f"{target} selected trial {trial+1}, {steps} updates, mean validation BCE={choice.validation_bce:.6f}")
    return winner


def refit_models(examples,target,winner,directory,*,device='cpu',progress=print):
    if set(examples.season) != {2023,2024,2025}:
        raise ValueError('Refitting requires the development seasons only.')
    encoder = SequenceEncoder.fit(examples)
    data = make_sequences(examples,encoder).to(device)
    constant = float(data.target.double().mean().item())
    constant_bce = -constant*np.log(constant)-(1-constant)*np.log(1-constant)
    diagnostics, fingerprints = [], {}
    for seed in winner['seeds']:
        current = directory/'refit'/f'seed-{seed}'
        current.mkdir(parents=True,exist_ok=True)
        if progress:
            progress(f"{target} refit seed {seed}: exactly {winner['steps']} updates")
        def show(row):
            if progress and (row['step'] % 400 == 0 or row['step'] == winner['steps']):
                progress(f"{target} refit seed {seed}, update {row['step']}: "
                         f"training audit BCE={row['training_audit_bce']:.6f}, gap={row['training_audit_calibration_gap']:+.4f}")
        model,history = fit_updates(data,None,winner['config'],len(encoder.categories)+2,
            steps=winner['steps'],horizon=winner['schedule_horizon'],seed=seed,
            checkpoint=current/'training.pt',progress=show)
        check = evaluate_summary(model,data)
        # A predeclared development-only sanity gate catches a repeat of the
        # original 7.8-point training overprediction before any test loading.
        if abs(check['calibration_gap']) > .03 or check['bce'] >= constant_bce:
            raise ValueError(f'{target} refit failed the development-only calibration/loss sanity gate.')
        save_model(current,model,encoder,winner['config'],target,constant,
                   winner['steps'],winner['schedule_horizon'],seed)
        _save_tables(current,{'learning_curve':pd.DataFrame(history)})
        diagnostics.append(dict(seed=seed,**check))
        fingerprints[str(seed)] = state_fingerprint(model)
        del model
    _save_tables(directory,{'refit_training_diagnostics':pd.DataFrame(diagnostics)})
    return dict(target=target,selected=winner,steps=winner['steps'],schedule_horizon=winner['schedule_horizon'],
        refit_possessions=len(data.lengths),updates_per_epoch=int(np.ceil(len(data.lengths)/EFFECTIVE_BATCH_SIZE)),
        constant_probability=constant,fingerprints=fingerprints,frozen_at_utc=datetime.now(timezone.utc).isoformat())


def compare_previous(root,target,reports):
    previous = root/'artifacts/transformer-2026'/target
    old = pd.read_parquet(previous/'history/test/predictions.parquet')
    columns = ['event_id','possession_id','target','sample_weight','constant_probability']
    for tables in reports.values():
        assert_frame_equal(old[columns].sort_values('event_id').reset_index(drop=True),
            tables['predictions'][columns].sort_values('event_id').reset_index(drop=True),
            check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    return {name:pd.concat([pd.read_parquet(previous/'comparison'/f'{name}.parquet'),
        *[table[name].loc[table[name].model.ne('constant')] for table in reports.values()]],ignore_index=True)
        for name in ['metrics','confusion','calibration']}


def run_experiment(root,*,device='auto',resume=False,candidates=None,screen_steps=SCREEN_STEPS,
                   full_steps=FULL_STEPS,evaluate_every=EVALUATE_EVERY,seeds=SEEDS,
                   promote_count=8,finalist_count=3,diagnostics=True):
    device = configure_runtime(device)
    directory = root/'artifacts/transformer-retrained-2026'
    candidates = candidate_configs() if candidates is None else candidates
    source_hashes = {name:_digest(root/'src/afl_opposition'/name) for name in SOURCE_NAMES}
    plan = dict(candidates=candidates,screen_steps=screen_steps,full_steps=full_steps,
        evaluate_every=evaluate_every,seeds=list(seeds),primary_seed=seeds[0],
        promote_count=promote_count,finalist_count=finalist_count,diagnostics=diagnostics,
        diagnostic_steps=600,source_sha256=source_hashes,device=device,torch_version=str(torch.__version__),
        python_version=platform.python_version(),candidate_seed=20261014,train_seasons=[2023,2024],
        validation_season=2025,refit_seasons=[2023,2024,2025],test_season=2026,targets=['shot','goal'],
        effective_batch_size=EFFECTIVE_BATCH_SIZE,microbatch_size=MICROBATCH_SIZE,
        evaluation_batch_size=EVALUATION_BATCH_SIZE,batching='random effective batches; sort only within each batch',
        schedule='linear warmup then cosine decay to 5% of peak; refit replays selected update prefix',
        optimiser='Adam, zero weight_decay, explicit L2 on matrix weights',threshold=.5,
        pruning='fresh complete cosine screen; protected strata; fresh full runs; no early stopping',
        refit_calibration_gap_limit=.03,refit_must_beat_training_constant=True,
        architecture='same causal pre-LN architecture as first Transformer',precision='float32',
        attention_backend='math',deterministic_algorithms=True)
    if directory.exists() and any(directory.iterdir()):
        if not resume:
            raise ValueError('Retraining experiment exists; use --resume.')
        if json.loads((directory/'protocol.json').read_text(encoding='utf-8')) != plan:
            raise ValueError('Source, environment or protocol changed; cannot resume.')
        if (directory/'summary.json').exists():
            completed = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            for name,expected in completed['artifact_sha256'].items():
                if _digest(directory/name) != expected:
                    raise ValueError('Completed artifacts changed.')
            print('Retraining experiment complete and verified; no fitting or scoring repeated.')
            return completed
    previous = root/'artifacts/transformer-2026'
    original = json.loads((previous/'summary.json').read_text(encoding='utf-8'))
    for name,expected in original['artifact_sha256'].items():
        if _digest(previous/name) != expected:
            raise ValueError('Earlier Transformer experiment changed.')
    directory.mkdir(parents=True,exist_ok=True)
    _json(directory/'protocol.json',plan)
    print(f'Retraining runtime: {device}; verifying development seasons.',flush=True)
    events,possessions,annotations,provenance = load_goal_inputs(root,(2023,2024,2025))
    provenance_path = directory/'development_provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Development provenance changed.')
    _json(provenance_path,provenance)
    datasets = build_datasets(events,possessions,annotations,development=True)
    del events,possessions,annotations
    frozen_hashes,label_summaries = {},{}
    for target,dataset in datasets.items():
        current = directory/target
        current.mkdir(exist_ok=True)
        label_summaries[target] = {'development':dataset.summary}
        if not (current/'refit.json').exists():
            if (current/'selection.json').exists():
                winner = json.loads((current/'selection.json').read_text())
            else:
                winner = select_model(dataset.examples,target,current,device=device,candidates=candidates,
                    screen_steps=screen_steps,full_steps=full_steps,evaluate_every=evaluate_every,
                    seeds=seeds,promote_count=promote_count,finalist_count=finalist_count,diagnostics=diagnostics,
                    progress=lambda s:print(s,flush=True))
            frozen = refit_models(dataset.examples,target,winner,current,device=device,progress=lambda s:print(s,flush=True))
            frozen['source_sha256'] = source_hashes
            frozen['file_sha256'] = {p.relative_to(current).as_posix():_digest(p) for p in
                [current/'selection.json',current/'refit_training_diagnostics.parquet',
                 *sorted((current/'refit').glob('seed-*/model.json')),*sorted((current/'refit').glob('seed-*/weights.pt'))]}
            _json(current/'refit.json',frozen)
        frozen = json.loads((current/'refit.json').read_text())
        for name,expected in frozen['file_sha256'].items():
            if _digest(current/name) != expected:
                raise ValueError('Frozen refit files changed.')
        frozen_hashes[target] = _digest(current/'refit.json')
        dataset.possession_audit.to_parquet(current/'development_possession_audit.parquet',index=False)
        if target == 'goal':
            dataset.goal_annotation_audit.to_parquet(current/'development_goal_annotation_audit.parquet',index=False)
        print(f'{target}: all {len(seeds)} refits frozen.',flush=True)
    del datasets,dataset
    for name,expected in source_hashes.items():
        if _digest(root/'src/afl_opposition'/name) != expected:
            raise ValueError('Source changed during training.')
    print('Both targets and all seeds frozen. Loading 2026 for the shared benchmark.',flush=True)
    events,possessions,annotations,test_provenance = load_goal_inputs(root,(2026,))
    tests = build_datasets(events,possessions,annotations,development=False)
    del events,possessions,annotations
    for target,dataset in tests.items():
        current = directory/target
        seed_probabilities,seed_metrics,reports = [],[],{}
        for seed in seeds:
            model,encoder,info = load_model(current/'refit'/f'seed-{seed}',device=device)
            before = state_fingerprint(model)
            data = make_sequences(dataset.examples,encoder).to(device)
            p = predict(model,data,EVALUATION_BATCH_SIZE)
            if state_fingerprint(model) != before or _digest(current/'refit.json') != frozen_hashes[target]:
                raise ValueError('Model changed during evaluation.')
            tables = probability_report(dataset.examples,p,info['constant_probability'],target,'test',f'transformer_seed_{seed}')
            _save_tables(current/'test'/f'seed-{seed}',tables)
            seed_probabilities.append(p)
            seed_metrics.append(tables['metrics'].loc[tables['metrics'].model.ne('constant')].assign(seed=seed))
            if seed == seeds[0]:
                reports['primary'] = probability_report(dataset.examples,p,info['constant_probability'],target,'test','transformer_retrained')
            del model,data
        reports['ensemble'] = probability_report(dataset.examples,np.mean(seed_probabilities,axis=0),
            info['constant_probability'],target,'test','transformer_retrained_ensemble')
        _save_tables(current/'test/ensemble',reports['ensemble'])
        comparison = compare_previous(root,target,reports)
        _save_tables(current/'comparison',comparison)
        _save_tables(current,{'test_seed_metrics':pd.concat(seed_metrics,ignore_index=True)})
        dataset.possession_audit.to_parquet(current/'test_possession_audit.parquet',index=False)
        if target == 'goal':
            dataset.goal_annotation_audit.to_parquet(current/'test_goal_annotation_audit.parquet',index=False)
        label_summaries[target]['test'] = dataset.summary
        print(target,comparison['metrics'].to_string(index=False),sep='\n',flush=True)
    result = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=_digest(directory/'protocol.json'),refit_sha256=frozen_hashes,
        all_refits_frozen_before_test_loading=True,models_unchanged_during_evaluation=True,
        test_previously_observed=True,primary_seed=seeds[0],label_summaries=label_summaries,
        test_provenance=test_provenance,device=device,gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
        artifact_sha256={p.relative_to(directory).as_posix():_digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name != 'summary.json' and p.suffix != '.tmp'})
    _json(directory/'summary.json',result)
    print('Saved experiment: artifacts/transformer-retrained-2026',flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    try:
        run_experiment(Path.cwd(),device=args.device,resume=args.resume)
    except (OSError,ValueError,RuntimeError) as error:
        parser.exit(1,f'Transformer retraining failed: {error}\n')


if __name__ == '__main__':
    main()
