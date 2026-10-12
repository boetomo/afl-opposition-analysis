"""Seq2Event-inspired causal Transformer for possession outcome probabilities.

Action embeddings and projected continuous features follow the mixed-input idea
of Seq2Event (Simpson et al., KDD 2022). The causal per-event BCE objective,
pre-normalised layers and binary head are adaptations, not a paper reproduction.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from afl_opposition.sequence import (
    SEED, SequenceEncoder, configure_runtime as configure_sequence_runtime,
    per_possession_bce, l2_penalty, state_fingerprint, _cpu_state, _atomic_save,
    evaluate_bce, predict,
)

BATCH_SIZE = 512
MODEL_FAMILY = 'seq2event_inspired_transformer'


def configure_runtime(device='auto', seed=SEED):
    device = configure_sequence_runtime(device, seed)
    # Use the same explicit-mask path in fitting and inference, with deterministic
    # float32 mathematical attention rather than device-dependent fused kernels.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return device


def positional_encoding(length, width, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype)
                          * (-math.log(10000.) / width))
    encoding = torch.zeros(length, width, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency)
    return encoding


class PossessionTransformer(nn.Module):
    """One logit per event; attention only sees that event and its observed past."""

    def __init__(self, vocabulary_size, config, constant_probability=.5):
        super().__init__()
        width = config['d_model']
        if width % 2 or width % config['num_heads']:
            raise ValueError('Model width must be even and divisible by attention heads.')
        if not 0 < constant_probability < 1:
            raise ValueError('The fitting population must contain both classes.')
        self.embedding = nn.Embedding(vocabulary_size, config['embedding_dim'], padding_idx=0)
        self.continuous = nn.Sequential(nn.Linear(4, config['continuous_dim']), nn.ReLU())
        self.projection = nn.Linear(config['embedding_dim'] + config['continuous_dim'], width)
        self.input_dropout = nn.Dropout(config['dropout'])
        # Construct each layer independently: cloned layers would start identically.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, config['num_heads'],
                dim_feedforward=width * config['ff_multiplier'], dropout=config['dropout'],
                activation='gelu', batch_first=True, norm_first=True)
            for _ in range(config['num_layers'])
        ])
        self.norm = nn.LayerNorm(width)
        self.output_dropout = nn.Dropout(config['dropout'])
        self.head = nn.Linear(width, 1)
        with torch.no_grad():
            self.embedding.weight.normal_(0, .1)
            self.embedding.weight[:2].zero_()  # PAD=0, training-unseen UNK=1.
            self.head.bias.fill_(float(np.log(constant_probability / (1 - constant_probability))))

    def forward(self, actions, numeric):
        padding = actions.eq(0)
        features = torch.cat((self.embedding(actions), self.continuous(numeric)), dim=-1)
        hidden = self.projection(features)
        hidden = self.input_dropout(hidden + positional_encoding(
            hidden.shape[1], hidden.shape[2], hidden.device, hidden.dtype))
        # For TransformerEncoderLayer boolean masks, True means blocked.
        causal = torch.ones(actions.shape[1], actions.shape[1], device=actions.device,
                            dtype=torch.bool).triu(1)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal, src_key_padding_mask=padding)
        return self.head(self.output_dropout(self.norm(hidden))).squeeze(-1)


def train_candidate(train, validation, config, vocabulary_size, *, epochs,
                    checkpoint=None, seed=SEED, batch_size=BATCH_SIZE, patience=6, progress=None):
    """Select minimum validation BCE; a refit runs a fixed selected epoch count."""
    if epochs < 1:
        raise ValueError('Epoch budget must be positive.')
    device = train.actions.device.type
    configure_runtime(device, seed)
    constant = float(train.target.double().mean().item())
    model = PossessionTransformer(vocabulary_size, config, constant).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.)
    history, best, best_epoch, first_epoch, bad_epochs, stopped = [], float('inf'), 0, 0, 0, False
    best_state = None
    contract = dict(model_family=MODEL_FAMILY, config=config, seed=seed, batch_size=batch_size,
                    patience=patience, has_validation=validation is not None, vocabulary_size=vocabulary_size)
    if checkpoint is not None and Path(checkpoint).exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved['contract'] != contract:
            raise ValueError('Checkpoint settings do not match this candidate.')
        model.load_state_dict(saved['model'])
        optimiser.load_state_dict(saved['optimiser'])
        history, best, best_epoch = saved['history'], saved['best_loss'], saved['best_epoch']
        best_state, first_epoch = saved['best_model'], saved['epoch']
        bad_epochs, stopped = saved['bad_epochs'], saved['stopped']
        torch.set_rng_state(saved['rng_cpu'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state(saved['rng_cuda'].cpu())
        if first_epoch > epochs:
            raise ValueError('Checkpoint exceeds the requested epoch budget.')
    for epoch in range(first_epoch + 1, epochs + 1):
        if stopped:
            break
        started = time.perf_counter()
        model.train()
        train_loss = torch.zeros((), device=device)
        for indices in train.batches(batch_size, seed=seed + epoch):
            actions, numeric, target, lengths = train.batch(indices)
            optimiser.zero_grad(set_to_none=True)
            losses = per_possession_bce(model(actions, numeric), target, lengths)
            loss = losses.mean() + config['lambda_l2'] * l2_penalty(model)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1., error_if_nonfinite=True)
            optimiser.step()
            train_loss += losses.detach().sum()
        training_bce = float(train_loss.item() / len(train.lengths))
        validation_bce = evaluate_bce(model, validation, batch_size) if validation is not None else None
        row = dict(epoch=epoch, training_bce=training_bce, validation_bce=validation_bce,
                   seconds=time.perf_counter() - started)
        history.append(row)
        if validation is not None:
            if validation_bce < best:
                best, best_epoch, best_state, bad_epochs = validation_bce, epoch, _cpu_state(model), 0
            else:
                bad_epochs += 1
            stopped = bad_epochs >= patience
        else:
            best_epoch, best_state = epoch, _cpu_state(model)
        if checkpoint is not None:
            _atomic_save(dict(contract=contract, epoch=epoch, model=model.state_dict(),
                optimiser=optimiser.state_dict(), best_model=best_state, best_loss=best,
                best_epoch=best_epoch, bad_epochs=bad_epochs, stopped=stopped, history=history,
                rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state() if device == 'cuda' else None), checkpoint)
        if progress:
            progress(row)
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def save_model(directory, model, encoder, config, target_name, constant, epochs):
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_save(_cpu_state(model), directory/'weights.pt')
    with (directory/'weights.pt').open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = dict(model_family=MODEL_FAMILY, encoder=encoder.as_dict(), config=config,
                    target_name=target_name, constant_probability=constant, epochs=epochs,
                    model_fingerprint=state_fingerprint(model), weights_sha256=checksum)
    (directory/'model.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def load_model(directory, device='cpu'):
    configure_runtime(device)
    info = json.loads((directory/'model.json').read_text(encoding='utf-8'))
    if info['model_family'] != MODEL_FAMILY:
        raise ValueError('Unexpected model family.')
    with (directory/'weights.pt').open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != info['weights_sha256']:
            raise ValueError('Model checkpoint checksum does not match.')
    encoder = SequenceEncoder.from_dict(info['encoder'])
    model = PossessionTransformer(len(encoder.categories)+2, info['config'], info['constant_probability'])
    model.load_state_dict(torch.load(directory/'weights.pt', map_location='cpu', weights_only=True))
    if state_fingerprint(model) != info['model_fingerprint']:
        raise ValueError('Model parameters differ from the frozen record.')
    return model.to(device).eval(), encoder, info
