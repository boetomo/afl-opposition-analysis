"""Stable, update-budgeted training for the possession Transformer.

The first experiment's architecture and saved artifacts remain unchanged.
Random effective batches, microbatch accumulation and a stateless learning-rate
schedule make batch composition and refit update counts explicit.
"""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch import nn

from afl_opposition.sequence import (
    SequenceEncoder, make_sequences, per_possession_bce, l2_penalty,
    _atomic_save, _cpu_state, state_fingerprint, predict,
)
from afl_opposition.transformer import PossessionTransformer, configure_runtime

EFFECTIVE_BATCH_SIZE = 1024
MICROBATCH_SIZE = 512
EVALUATION_BATCH_SIZE = 512
AUDIT_POSSESSIONS = 4096
MODEL_FAMILY = 'seq2event_transformer_retrained'


def batch_indices(lengths, batch_size, seed, mode='random'):
    """Every possession exactly once; label-free random effective batches."""
    if batch_size < 1 or mode not in {'random', 'bucketed'}:
        raise ValueError('Invalid batch size or batching mode.')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    if mode == 'random':
        return [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
    batches = []
    for start in range(0, len(order), batch_size*16):
        pool = order[start:start+batch_size*16]
        pool = pool[np.argsort(lengths[pool], kind='stable')]
        batches.extend(pool[i:i+batch_size] for i in range(0, len(pool), batch_size))
    rng.shuffle(batches)
    return batches


def microbatch_indices(indices, lengths, microbatch_size):
    """Sort only inside an already selected random effective batch.

No optimiser update occurs between these slices. Accumulated gradients are
normalised by the actual effective-batch count, including its final partial batch.
"""
    order = indices[np.argsort(lengths[indices], kind='stable')]
    return [order[i:i+microbatch_size] for i in range(0, len(order), microbatch_size)]


def learning_rate_at(step, horizon, config, schedule='cosine'):
    """One-based update index. Refit replays the selected schedule prefix."""
    if not 1 <= step <= horizon or schedule not in {'constant', 'cosine'}:
        raise ValueError('Invalid schedule step or horizon.')
    peak = float(config['learning_rate'])
    if schedule == 'constant':
        return peak
    warmup = int(round(horizon*config['warmup_fraction']))
    if warmup and step <= warmup:
        return peak*step/warmup
    # Reach exactly 5% of peak on the final update, including when warmup is zero.
    progress = (step-warmup-1)/max(1, horizon-warmup-1)
    return peak*(.05+.95*.5*(1+math.cos(math.pi*progress)))


@torch.inference_mode()
def evaluate_summary(model, data, *, indices=None, batch_size=EVALUATION_BATCH_SIZE):
    """BCE and calibration in eval mode, each possession total weight one."""
    model.eval()
    selected = np.arange(len(data.lengths)) if indices is None else np.asarray(indices)
    if not len(selected):
        raise ValueError('Evaluation population is empty.')
    loss = torch.zeros((), device=data.actions.device, dtype=torch.float64)
    predicted = torch.zeros_like(loss)
    observed = torch.zeros_like(loss)
    # Sorting evaluation batches is safe: there are no parameter updates.
    order = selected[np.argsort(data.lengths[selected], kind='stable')]
    for start in range(0, len(order), batch_size):
        a, x, y, lengths = data.batch(order[start:start+batch_size])
        logits = model(a, x)
        mask = torch.arange(logits.shape[1], device=logits.device)[None, :] < lengths[:, None]
        loss += per_possession_bce(logits, y, lengths).double().sum()
        predicted += ((logits.sigmoid()*mask).sum(1)/lengths).double().sum()
        observed += y.double().sum()
    result = dict(bce=loss.item()/len(order), mean_probability=predicted.item()/len(order),
                  observed_rate=observed.item()/len(order), possessions=len(order))
    result['calibration_gap'] = result['mean_probability']-result['observed_rate']
    if not all(math.isfinite(result[key]) for key in ['bce', 'mean_probability', 'observed_rate']):
        raise ValueError('Non-finite evaluation.')
    return result


def training_step(model, optimiser, data, indices, config, *, microbatch_size=MICROBATCH_SIZE):
    model.train()
    optimiser.zero_grad(set_to_none=True)
    bce_sum = torch.zeros((), device=data.actions.device)
    for micro in microbatch_indices(indices, data.lengths, microbatch_size):
        actions, numeric, target, lengths = data.batch(micro)
        losses = per_possession_bce(model(actions, numeric), target, lengths)
        objective = losses.sum()/len(indices)
        if not torch.isfinite(objective):
            raise ValueError('Non-finite training BCE.')
        objective.backward()
        bce_sum += losses.detach().sum()
    if config['lambda_l2']:
        penalty = config['lambda_l2']*l2_penalty(model)
        if not torch.isfinite(penalty):
            raise ValueError('Non-finite regularisation penalty.')
        penalty.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimiser.step()
    return bce_sum, len(indices)


def fit_updates(train, validation, config, vocabulary_size, *, steps, horizon=None,
                seed=2026, checkpoint=None, snapshot_directory=None, evaluate_every=100,
                batch_size=EFFECTIVE_BATCH_SIZE, microbatch_size=MICROBATCH_SIZE,
                batching='random', schedule='cosine', progress=None):
    """Run a fixed update budget; all checkpoints remain available for selection.

There is no early stopping or pruning inside a run. Screening and full runs each
start fresh and complete their own warmup/decay horizon. A refit uses the full
run's horizon but stops at the validation-selected update count.
"""
    horizon = steps if horizon is None else horizon
    if steps < 1 or steps > horizon or evaluate_every < 1 or microbatch_size < 1:
        raise ValueError('Invalid training budget.')
    device = train.actions.device.type
    configure_runtime(device, seed)
    constant = float(train.target.double().mean().item())
    model = PossessionTransformer(vocabulary_size, config, constant).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.)
    audit = np.sort(np.random.default_rng(6701).choice(len(train.lengths),
                    min(AUDIT_POSSESSIONS, len(train.lengths)), replace=False))
    contract = dict(config=config, horizon=horizon, seed=seed, batch_size=batch_size,
        microbatch_size=microbatch_size, batching=batching, schedule=schedule,
        evaluate_every=evaluate_every, has_validation=validation is not None,
        vocabulary_size=vocabulary_size, training_possessions=len(train.lengths))
    first_step, history = 0, []
    interval_loss, interval_count = 0., 0
    if checkpoint is not None and Path(checkpoint).exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved['contract'] != contract:
            raise ValueError('Checkpoint training contract changed.')
        first_step, history = saved['step'], saved['history']
        if first_step > steps:
            raise ValueError('Checkpoint exceeds the requested update count.')
        model.load_state_dict(saved['model'])
        optimiser.load_state_dict(saved['optimiser'])
        torch.set_rng_state(saved['rng_cpu'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state(saved['rng_cuda'].cpu())
        interval_loss, interval_count = saved['interval_loss'], saved['interval_count']
    updates_per_epoch = math.ceil(len(train.lengths)/batch_size)
    current_epoch, batches = -1, None
    started = time.perf_counter()
    for step in range(first_step+1, steps+1):
        epoch, offset = divmod(step-1, updates_per_epoch)
        if epoch != current_epoch:
            batches = batch_indices(train.lengths, batch_size, seed+epoch+1, batching)
            current_epoch = epoch
        rate = learning_rate_at(step, horizon, config, schedule)
        for group in optimiser.param_groups:
            group['lr'] = rate
        loss, count = training_step(model, optimiser, train, batches[offset], config,
                                    microbatch_size=microbatch_size)
        interval_loss += float(loss.item())
        interval_count += count
        if step % evaluate_every == 0 or step == steps:
            diagnostic = evaluate_summary(model, train, indices=audit)
            row = dict(step=step, epochs=step/updates_per_epoch, learning_rate=rate,
                       online_training_bce=interval_loss/interval_count,
                       seconds=time.perf_counter()-started,
                       **{'training_audit_'+key:value for key,value in diagnostic.items()})
            if validation is not None:
                row.update({'validation_'+key:value for key,value in evaluate_summary(model, validation).items()})
            history.append(row)
            interval_loss, interval_count = 0., 0
            if snapshot_directory is not None:
                snapshot_directory.mkdir(parents=True, exist_ok=True)
                _atomic_save(_cpu_state(model), snapshot_directory/f'step-{step:05d}.pt')
            if checkpoint is not None:
                _atomic_save(dict(contract=contract, step=step, model=model.state_dict(),
                    optimiser=optimiser.state_dict(), history=history,
                    interval_loss=interval_loss, interval_count=interval_count,
                    rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state() if device == 'cuda' else None), checkpoint)
            if progress:
                progress(row)
            started = time.perf_counter()
    model.eval()
    return model, history


def save_model(directory, model, encoder, config, target, constant, steps, horizon, seed):
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_save(_cpu_state(model), directory/'weights.pt')
    with (directory/'weights.pt').open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    info = dict(model_family=MODEL_FAMILY, encoder=encoder.as_dict(), config=config,
                target_name=target, constant_probability=constant, steps=steps,
                schedule_horizon=horizon, seed=seed, model_fingerprint=state_fingerprint(model),
                weights_sha256=checksum)
    (directory/'model.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')


def load_model(directory, device='cpu'):
    device = configure_runtime(device)
    info = json.loads((directory/'model.json').read_text(encoding='utf-8'))
    if info['model_family'] != MODEL_FAMILY:
        raise ValueError('Unexpected model family.')
    with (directory/'weights.pt').open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != info['weights_sha256']:
            raise ValueError('Checkpoint checksum differs.')
    encoder = SequenceEncoder.from_dict(info['encoder'])
    model = PossessionTransformer(len(encoder.categories)+2, info['config'], info['constant_probability'])
    model.load_state_dict(torch.load(directory/'weights.pt', map_location='cpu', weights_only=True))
    if state_fingerprint(model) != info['model_fingerprint']:
        raise ValueError('Frozen parameters differ.')
    return model.to(device).eval(), encoder, info
