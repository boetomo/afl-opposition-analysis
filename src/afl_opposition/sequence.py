"""Causal possession sequences, learned action embeddings and LSTM fitting."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from afl_opposition.preprocess import FEATURE_COLUMNS

NUMERIC_COLUMNS = FEATURE_COLUMNS[1:]
SEED = 2026
BATCH_SIZE = 1024


def configure_runtime(device="auto", seed=SEED):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Native CUDA kernels avoid a cuDNN RNN shutdown crash on the tested Windows build.
    # They also benchmark faster for these short sequences. The LSTM equations are unchanged.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu or cuda.")
    return device


@dataclass
class SequenceEncoder:
    categories: list
    mean: list
    scale: list

    @classmethod
    def fit(cls, examples):
        if set(examples.season) - {2023, 2024, 2025}:
            raise ValueError("Preprocessing fitting cannot include test seasons.")
        scaler = StandardScaler().fit(examples[NUMERIC_COLUMNS])
        return cls(sorted(examples.action_type.unique().tolist()), scaler.mean_.tolist(), scaler.scale_.tolist())

    def transform(self, examples):
        mapping = {name: i + 2 for i, name in enumerate(self.categories)}
        actions = examples.action_type.map(mapping).fillna(1).to_numpy(dtype=np.int64)
        numeric = (examples[NUMERIC_COLUMNS].to_numpy(dtype=float) - np.asarray(self.mean)) / np.asarray(self.scale)
        if not np.isfinite(numeric).all():
            raise ValueError("Sequence inputs must be finite.")
        return actions, numeric.astype(np.float32)

    def as_dict(self):
        return dict(categories=self.categories, mean=self.mean, scale=self.scale,
                    numeric_columns=NUMERIC_COLUMNS, padding_id=0, unknown_id=1)

    @classmethod
    def from_dict(cls, value):
        if value["numeric_columns"] != NUMERIC_COLUMNS:
            raise ValueError("Saved numeric feature contract differs.")
        return cls(value["categories"], value["mean"], value["scale"])


@dataclass
class SequenceData:
    actions: torch.Tensor
    numeric: torch.Tensor
    target: torch.Tensor
    lengths: np.ndarray
    row_map: np.ndarray
    size: int

    def to(self, device):
        return SequenceData(self.actions.to(device), self.numeric.to(device), self.target.to(device),
                            self.lengths, self.row_map, self.size)

    def batches(self, batch_size=BATCH_SIZE, *, seed=None):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.lengths)) if seed is not None else np.arange(len(self.lengths))
        batches = []
        # Random pools, sorted locally by length, reduce right padding without dropping rows.
        for start in range(0, len(order), batch_size * 16):
            pool = order[start:start + batch_size * 16]
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            batches.extend(pool[i:i + batch_size] for i in range(0, len(pool), batch_size))
        if seed is not None:
            rng.shuffle(batches)
        return batches

    def batch(self, indices):
        width = int(self.lengths[indices].max())
        index = torch.as_tensor(indices, device=self.actions.device)
        lengths = torch.as_tensor(self.lengths[indices], device=self.actions.device)
        return self.actions[index, :width], self.numeric[index, :width], self.target[index], lengths


def make_sequences(examples, encoder, *, labelled=True):
    """Group complete observed prefixes; retain original event order for predictions."""
    required = {"event_id", "possession_id", "event_index", *FEATURE_COLUMNS}
    if labelled:
        required |= {"target", "sample_weight"}
    if required - set(examples) or examples.empty or examples[list(required)].isna().any().any():
        raise ValueError("Incomplete sequence examples.")
    if examples.event_id.duplicated().any():
        raise ValueError("Event IDs must be unique.")
    table = examples.copy()
    table["_row"] = np.arange(len(table))
    table = table.sort_values(["possession_id", "event_index"])
    groups = table.groupby("possession_id", sort=False)
    lengths = groups.size().to_numpy(dtype=np.int64)
    group_index = groups.ngroup().to_numpy()
    steps = groups.cumcount().to_numpy()
    if not np.array_equal(steps, table.event_index.to_numpy(dtype=int)):
        raise ValueError("Sequences require contiguous prefixes starting at event zero.")
    if labelled:
        if not table.target.isin([0, 1]).all() or not groups.target.nunique().eq(1).all():
            raise ValueError("Each possession requires one consistent binary target.")
        if not np.allclose(table.sample_weight.to_numpy(dtype=float), 1 / lengths[group_index], atol=1e-12, rtol=1e-12):
            raise ValueError("Each possession must have total prefix weight one.")
    shape = (len(lengths), int(lengths.max()))
    actions = np.zeros(shape, dtype=np.int64)
    numeric = np.zeros((*shape, len(NUMERIC_COLUMNS)), dtype=np.float32)
    row_map = np.full(shape, -1, dtype=np.int64)
    a, x = encoder.transform(table)
    actions[group_index, steps], numeric[group_index, steps] = a, x
    row_map[group_index, steps] = table._row.to_numpy(dtype=int)
    y = groups.target.first().to_numpy(dtype=np.float32) if labelled else np.zeros(len(lengths), dtype=np.float32)
    return SequenceData(torch.from_numpy(actions), torch.from_numpy(numeric), torch.from_numpy(y), lengths, row_map, len(table))


class PossessionLSTM(nn.Module):
    """Forward-only event predictions; no_history runs every event as a length-one sequence."""

    def __init__(self, vocabulary_size, config, variant, constant_probability=0.5):
        super().__init__()
        if variant not in {"history", "no_history"}:
            raise ValueError("Unknown sequence variant.")
        self.variant = variant
        self.embedding = nn.Embedding(vocabulary_size, config["embedding_dim"], padding_idx=0)
        self.lstm = nn.LSTM(config["embedding_dim"] + 4, config["hidden_size"],
            num_layers=config["num_layers"], batch_first=True, bidirectional=False,
            dropout=config["dropout"] if config["num_layers"] > 1 else 0.)
        self.dropout = nn.Dropout(config["dropout"])
        self.head = nn.Linear(config["hidden_size"], 1)
        with torch.no_grad():
            self.embedding.weight.normal_(0, .1)
            # UNK remains zero when unobserved in fitting; no arbitrary random test-category vector.
            self.embedding.weight[:2].zero_()
            self.head.bias.fill_(float(np.log(constant_probability / (1 - constant_probability))))

    def forward(self, actions, numeric):
        x = torch.cat((self.embedding(actions), numeric), dim=-1)
        batch, width, features = x.shape
        if self.variant == "no_history":
            x = x.reshape(batch * width, 1, features)
        # Omitted initial states are fresh zeros on every forward call/possession.
        h, _ = self.lstm(x)
        if self.variant == "no_history":
            h = h.reshape(batch, width, -1)
        return self.head(self.dropout(h)).squeeze(-1)


def per_possession_bce(logits, target, lengths):
    mask = torch.arange(logits.shape[1], device=logits.device)[None, :] < lengths[:, None]
    losses = F.binary_cross_entropy_with_logits(logits, target[:, None].expand_as(logits), reduction="none")
    return (losses * mask).sum(dim=1) / lengths


def l2_penalty(model):
    # Explicit L2 in the objective, not AdamW's decoupled weight decay; biases excluded.
    return sum(p.square().sum() for p in model.parameters() if p.ndim > 1)


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def train_candidate(train, validation, config, variant, vocabulary_size, *, epochs,
                    checkpoint=None, seed=SEED, batch_size=BATCH_SIZE, patience=6, progress=None):
    """Resume optimiser and random state between search stages; choose minimum validation BCE."""
    device = train.actions.device.type
    configure_runtime(device, seed)
    constant = float(train.target.double().mean().item())
    model = PossessionLSTM(vocabulary_size, config, variant, constant).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0.)
    history, best, best_epoch, first_epoch, bad_epochs, stopped = [], float("inf"), 0, 0, 0, False
    best_state = None
    if checkpoint is not None and Path(checkpoint).exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved["config"] != config or saved["variant"] != variant or saved["seed"] != seed:
            raise ValueError("Checkpoint settings do not match this candidate.")
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        history, best, best_epoch = saved["history"], saved["best_loss"], saved["best_epoch"]
        best_state, first_epoch = saved["best_model"], saved["epoch"]
        bad_epochs, stopped = saved["bad_epochs"], saved["stopped"]
        torch.set_rng_state(saved["rng_cpu"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state(saved["rng_cuda"].cpu())
    for epoch in range(first_epoch + 1, epochs + 1):
        if stopped:
            break
        started = time.perf_counter()
        model.train()
        train_loss = torch.zeros((), device=device)
        for indices in train.batches(batch_size, seed=seed + epoch):
            actions, numeric, target, lengths = train.batch(indices)
            optimiser.zero_grad(set_to_none=True)
            logits = model(actions, numeric)
            losses = per_possession_bce(logits, target, lengths)
            loss = losses.mean() + config["lambda_l2"] * l2_penalty(model)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
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
            _atomic_save(dict(config=config, variant=variant, seed=seed, epoch=epoch,
                model=model.state_dict(), optimiser=optimiser.state_dict(), best_model=best_state,
                best_loss=best, best_epoch=best_epoch, bad_epochs=bad_epochs, stopped=stopped,
                history=history, rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state() if device == "cuda" else None), checkpoint)
        if progress:
            progress(row)
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


@torch.inference_mode()
def evaluate_bce(model, dataset, batch_size=BATCH_SIZE):
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=dataset.actions.device)
    for indices in dataset.batches(batch_size):
        actions, numeric, target, lengths = dataset.batch(indices)
        total += per_possession_bce(model(actions, numeric), target, lengths).double().sum()
    return float(total.item() / len(dataset.lengths))


@torch.inference_mode()
def predict(model, dataset, batch_size=BATCH_SIZE):
    model.eval()
    values = np.empty(dataset.size, dtype=float)
    for indices in dataset.batches(batch_size):
        actions, numeric, _, _ = dataset.batch(indices)
        p = model(actions, numeric).sigmoid().cpu().numpy()
        rows = dataset.row_map[indices, :p.shape[1]]
        valid = rows >= 0
        values[rows[valid]] = p[valid]
    return values


def save_model(directory, model, encoder, config, target_name, constant, epochs):
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_save(_cpu_state(model), directory / "weights.pt")
    metadata = dict(encoder=encoder.as_dict(), config=config, target_name=target_name,
                    variant=model.variant, constant_probability=constant, epochs=epochs,
                    model_fingerprint=state_fingerprint(model))
    with (directory / "weights.pt").open("rb") as stream:
        metadata["weights_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (directory / "model.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_model(directory, device="cpu"):
    info = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    with (directory / "weights.pt").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != info["weights_sha256"]:
            raise ValueError("Model checkpoint checksum does not match.")
    encoder = SequenceEncoder.from_dict(info["encoder"])
    model = PossessionLSTM(len(encoder.categories) + 2, info["config"], info["variant"], info["constant_probability"])
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    if state_fingerprint(model) != info["model_fingerprint"]:
        raise ValueError("Model parameters differ from the frozen record.")
    return model.to(device).eval(), encoder, info
