"""Seq2Event-style joint categorical-action and point-location prediction."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from afl_opposition.next_events import ACTIONS, NUMERIC, CONTRACT_VERSION, observed_features
from afl_opposition.sequence import configure_runtime, state_fingerprint, _cpu_state, _atomic_save


@dataclass
class FeatureScaler:
    mean: list
    scale: list

    @classmethod
    def fit(cls, frame):
        if frame.empty or not set(frame.season).issubset({2023, 2024, 2025}):
            raise ValueError("Fit feature scaling on development seasons only.")
        values = frame[list(NUMERIC)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite training inputs.")
        scale = values.std(axis=0)
        scale[scale == 0] = 1.
        return cls(values.mean(axis=0).tolist(), scale.tolist())

    def transform(self, frame):
        values = (frame[list(NUMERIC)].to_numpy(dtype=float) - self.mean) / self.scale
        if not np.isfinite(values).all():
            raise ValueError("Non-finite model inputs.")
        return values.astype("float32")


class PrefixData:
    """Compact flat storage; construct capped, right-padded histories on demand."""

    def __init__(self, frame, scaler, device="cpu"):
        self.frame = frame.reset_index(drop=True)
        if frame.empty:
            raise ValueError("Empty prefix dataset.")
        action_ids = {a: i for i, a in enumerate(ACTIONS)}
        mapped = frame.action_type.map(action_ids)
        if mapped.isna().any():
            raise ValueError("Unknown input action.")
        # PAD=0; input actions are output class indices + 1.
        self.actions = torch.as_tensor(mapped.to_numpy(dtype="int64") + 1, device=device)
        self.numeric = torch.as_tensor(scaler.transform(frame), device=device)
        self.target = torch.as_tensor(frame.target_action.map(action_ids).fillna(-1).to_numpy(dtype="int64", copy=True), device=device)
        self.xy = torch.as_tensor(frame[["target_x", "target_y"]].to_numpy(dtype="float32", copy=True), device=device)
        self.location = torch.as_tensor(frame.has_location.to_numpy(dtype=bool, copy=True), device=device)
        self.ends = np.flatnonzero(frame.labelled.to_numpy(dtype=bool))
        self.starts = np.arange(len(frame)) - frame.event_index.to_numpy(dtype=int)
        if (self.starts < 0).any() or not (frame.possession_id.to_numpy()[self.starts] == frame.possession_id.to_numpy()).all():
            raise ValueError("History would cross a possession boundary.")
        self.device = device

    def batches(self, batch_size, seed=None):
        order = self.ends.copy()
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
            buckets = []
            # Random pools preserve mixing while reducing padding on the GPU.
            for start in range(0, len(order), batch_size * 32):
                pool = order[start:start + batch_size * 32]
                pool = pool[np.argsort(pool - self.starts[pool], kind="stable")]
                buckets.extend(pool[i:i + batch_size] for i in range(0, len(pool), batch_size))
            rng.shuffle(buckets)
            yield from buckets
            return
        for start in range(0, len(order), batch_size):
            yield order[start:start + batch_size]

    def batch(self, ends, cap):
        if cap < 1:
            raise ValueError("History cap must be positive.")
        starts = np.maximum(self.starts[ends], ends - cap + 1)
        lengths = ends - starts + 1
        positions = np.arange(int(lengths.max()))[None, :]
        valid = positions < lengths[:, None]
        rows = np.minimum(starts[:, None] + positions, ends[:, None])
        indices = torch.as_tensor(rows, device=self.device)
        mask = torch.as_tensor(valid, device=self.device)
        actions = self.actions[indices].masked_fill(~mask, 0)
        numeric = self.numeric[indices].masked_fill(~mask[:, :, None], 0.)
        target_rows = torch.as_tensor(ends, device=self.device)
        return actions, numeric, self.target[target_rows], self.xy[target_rows], self.location[target_rows]


def position_encoding(length, width, *, device, dtype):
    """Standard sinusoidal PE; explicit correction of the author's code variant."""
    position = torch.arange(length, device=device, dtype=dtype)[:, None]
    rate = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-np.log(10000.) / width))
    values = torch.zeros(length, width, device=device, dtype=dtype)
    values[:, 0::2] = torch.sin(position * rate)
    values[:, 1::2] = torch.cos(position * rate)
    return values


class NextEventTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        width = config["embedding_dim"] + config["continuous_dim"]
        if width % 2 or width % config["heads"]:
            raise ValueError("Combined dimension must be even and divisible by attention heads.")
        self.embedding = nn.Embedding(len(ACTIONS) + 1, config["embedding_dim"], padding_idx=0)
        self.continuous = nn.Linear(len(NUMERIC), config["continuous_dim"])
        # Independent initialisation, post-norm and ReLU, as in the paper's encoder.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, config["heads"],
                dim_feedforward=width * config["ff_multiplier"],
                dropout=config["dropout"], activation="relu",
                batch_first=True, norm_first=False)
            for _ in range(config["layers"])
        ])
        self.dense = nn.Linear(width, config["dense_dim"])
        self.output = nn.Linear(config["dense_dim"], len(ACTIONS) + 2)

    def forward(self, actions, numeric):
        padding = actions.eq(0)
        if padding.all(dim=1).any():
            raise ValueError("Each history must have at least one observed event.")
        if ((~padding[:, 1:]) & padding[:, :-1]).any():
            raise ValueError("Histories must be right padded.")
        hidden = torch.cat([self.embedding(actions), self.continuous(numeric)], dim=-1)
        hidden = hidden + position_encoding(hidden.shape[1], hidden.shape[2],
                                             device=hidden.device, dtype=hidden.dtype)
        # All tokens are observed history. No future target token is supplied.
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding)
        last = (~padding).sum(dim=1) - 1
        final = hidden[torch.arange(len(last), device=hidden.device), last]
        output = self.output(F.relu(self.dense(final)))
        return output[:, :len(ACTIONS)], output[:, len(ACTIONS):]


def joint_loss(logits, xy, target, target_xy, location, *, action_weight=5.):
    """Unweighted CE + masked coordinate RMSE; natural class frequencies preserved."""
    ce = F.cross_entropy(logits, target)
    if location.any():
        rmse = torch.sqrt(F.mse_loss(xy[location], target_xy[location]) + 1e-12)
    else:
        rmse = xy.sum() * 0.
    return action_weight * ce + rmse


def configure(device="auto", seed=2026):
    chosen = configure_runtime(device, seed)
    torch.backends.mha.set_fastpath_enabled(False)
    if hasattr(torch.nn, "attention"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return chosen


@torch.inference_mode()
def predict(model, data, batch_size=1024):
    model.eval()
    probabilities, coordinates = [], []
    for ends in data.batches(batch_size):
        actions, numeric, _, _, _ = data.batch(ends, model.config["max_history"])
        logits, xy = model(actions, numeric)
        probabilities.append(logits.softmax(-1).cpu().numpy())
        coordinates.append(xy.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(coordinates)


def save_model(directory, model, scaler, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_save(_cpu_state(model), directory / "weights.pt")
    info = {**metadata, "contract": CONTRACT_VERSION, "actions": list(ACTIONS),
            "numeric_features": list(NUMERIC), "config": model.config,
            "scaler": {"mean": scaler.mean, "scale": scaler.scale},
            "fingerprint": state_fingerprint(model),
            "weights_sha256": hashlib.sha256((directory / "weights.pt").read_bytes()).hexdigest()}
    (directory / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def load_model(directory, device="cpu"):
    directory = Path(directory)
    info = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if info["contract"] != CONTRACT_VERSION or info["actions"] != list(ACTIONS) or info["numeric_features"] != list(NUMERIC):
        raise ValueError("Model feature contract differs.")
    if hashlib.sha256((directory / "weights.pt").read_bytes()).hexdigest() != info["weights_sha256"]:
        raise ValueError("Model checkpoint checksum differs.")
    model = NextEventTransformer(info["config"])
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    if state_fingerprint(model) != info["fingerprint"]:
        raise ValueError("Model fingerprint differs.")
    return model.to(device).eval(), FeatureScaler(**info["scaler"]), info


@torch.inference_mode()
def forecast_next(model, scaler, history):
    """Forecast from an observed processed episode prefix; no labels are required.

    Supply all observed actions in one episode, starting at event_index=0.
    Feature construction precedes capping so observed coordinate deltas survive.
    The location is conditional on continuation and is not a per-action location.
    """
    if history.empty or (history[["possession_id", "match_id", "period", "team_id"]].nunique() != 1).any():
        raise ValueError("Supply a nonempty prefix from one possession.")
    frame = observed_features(history).reset_index(drop=True)
    if not np.array_equal(frame.event_index.to_numpy(), np.arange(len(frame))):
        raise ValueError("Supply the complete observed prefix, starting at event_index=0.")
    if (frame.period_seconds.diff().dropna() < 0).any():
        raise ValueError("History contains a time reversal.")
    frame = frame.tail(model.config["max_history"])
    ids = frame.action_type.map({a: i + 1 for i, a in enumerate(ACTIONS[:-1])})
    if ids.isna().any():
        raise ValueError("History contains an unsupported action.")
    device = next(model.parameters()).device
    actions = torch.tensor(ids.to_numpy(dtype="int64"), device=device)[None, :]
    numeric = torch.as_tensor(scaler.transform(frame), device=device)[None, :, :]
    model.eval()
    logits, xy = model(actions, numeric)
    probabilities = logits.softmax(-1)[0].cpu().tolist()
    return {"probabilities": dict(zip(ACTIONS, probabilities)),
            "location_if_continuation": {"x_norm": float(xy[0, 0]), "y_norm": float(xy[0, 1])},
            "observed_events": len(history), "used_events": len(frame)}
