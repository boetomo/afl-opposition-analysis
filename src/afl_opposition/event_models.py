"""Fixed-window Seq2Event and full-quarter causal Transformer implementations."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from afl_opposition.quarter_events import (
    CONTRACT, BALL_ACTIONS, INPUT_ACTIONS, NUMERIC, K, MODES, make_streams,
)
from afl_opposition.next_event_model import position_encoding, configure
from afl_opposition.sequence import state_fingerprint, _cpu_state, _atomic_save


@dataclass
class EventEncoder:
    mean: list
    scale: list
    clubs: list

    @classmethod
    def fit(cls, frame):
        if frame.empty or not set(frame.season).issubset({2023, 2024, 2025}):
            raise ValueError("Encoder fitting must use development seasons only.")
        x = frame[list(NUMERIC)].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite numerical features.")
        std = x.std(0)
        std[std == 0] = 1.
        return cls(x.mean(0).tolist(), std.tolist(), sorted(set(frame.home_team_id) | set(frame.away_team_id)))

    def transform(self, frame):
        a = frame.action_type.map({a: i + 1 for i, a in enumerate(INPUT_ACTIONS)})
        if a.isna().any():
            raise ValueError("Unsupported action.")
        club_map = {club: i + 1 for i, club in enumerate(self.clubs)}
        x = (frame[list(NUMERIC)].to_numpy(dtype=float) - self.mean) / self.scale
        if not np.isfinite(x).all():
            raise ValueError("Non-finite numerical inputs.")
        return (a.to_numpy(dtype=np.int64, copy=True), x.astype(np.float32),
                frame.side.to_numpy(dtype=np.int64, copy=True),
                frame.home_team_id.map(club_map).fillna(0).to_numpy(dtype=np.int64, copy=True),
                frame.away_team_id.map(club_map).fillna(0).to_numpy(dtype=np.int64, copy=True))


class EventData:
    def __init__(self, frame, encoder, mode, window=40, device="cpu"):
        if mode not in MODES or window < 1 or frame.empty:
            raise ValueError("Invalid dataset configuration.")
        self.mode, self.window, self.device = mode, window, device
        self.frame = frame.reset_index(drop=True)
        self.values = [torch.as_tensor(x, device=device) for x in encoder.transform(self.frame)]
        self.targets = torch.as_tensor(self.frame.target.to_numpy(dtype=np.int64, copy=True), device=device)
        self.xy = torch.as_tensor(self.frame[["target_x", "target_y"]].to_numpy(dtype=np.float32, copy=True), device=device)
        self.location = torch.as_tensor(self.frame.has_location.to_numpy(dtype=bool, copy=True), device=device)
        self.starts = np.arange(len(frame)) - self.frame.position.to_numpy(dtype=int)
        ids = self.frame.stream_id.to_numpy()
        if (self.starts < 0).any() or not np.array_equal(ids[self.starts], ids):
            raise ValueError("Incomplete or non-contiguous stream.")
        groups = self.frame.groupby("stream_id", sort=False)
        if not np.array_equal(groups.cumcount().to_numpy(), self.frame.position.to_numpy()):
            raise ValueError("Streams must begin at position zero and remain contiguous.")
        if (groups[["match_id", "period", "season"]].nunique() != 1).any().any():
            raise ValueError("A sequence crosses a match, quarter or season.")
        valid = self.frame.target.ge(0).to_numpy(copy=True)
        if mode == "team_stream":
            valid &= self.frame.position.ge(window - 1).to_numpy()
            self.items = np.flatnonzero(valid)
        else:
            bounds = groups.size().to_numpy()
            self.items = np.c_[np.cumsum(np.r_[0, bounds[:-1]]), bounds]
        self.scored_rows = np.flatnonzero(valid)
        if not len(self.scored_rows):
            raise ValueError("No eligible predictions in this dataset.")

    def batches(self, batch_size, seed=None):
        order = np.arange(len(self.items))
        if seed is not None:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            items = self.items[order[start:start + batch_size]]
            if self.mode == "team_stream":
                rows = items[:, None] - self.window + 1 + np.arange(self.window)[None, :]
                valid = np.ones(rows.shape, dtype=bool)
                output_rows = items
            else:
                positions = np.arange(int(items[:, 1].max()))
                valid = positions[None, :] < items[:, 1, None]
                rows = np.minimum(items[:, 0, None] + positions, (items[:, 0] + items[:, 1] - 1)[:, None])
                output_rows = np.where(valid, rows, -1)
            mask = torch.as_tensor(valid, device=self.device)
            ix = torch.as_tensor(rows, device=self.device)
            inputs = [v[ix].masked_fill(~mask if v.ndim == 1 else ~mask[:, :, None], 0) for v in self.values]
            out = torch.as_tensor(np.maximum(output_rows, 0), device=self.device)
            out_valid = torch.as_tensor(output_rows >= 0, device=self.device)
            target = self.targets[out].masked_fill(~out_valid, -1)
            location = self.location[out] & out_valid
            yield inputs, target, self.xy[out], location, output_rows


class EventTransformer(nn.Module):
    """Same post-norm block family; attention/output semantics differ by mode."""
    def __init__(self, config, num_clubs):
        super().__init__()
        self.config, self.num_clubs = dict(config), num_clubs
        self.mode = config["mode"]
        if self.mode not in MODES:
            raise ValueError("Unknown model formulation.")
        width, emb = config["width"], config["embedding_dim"]
        self.action = nn.Embedding(len(INPUT_ACTIONS) + 1, emb, padding_idx=0)
        if self.mode == "joint":
            self.side = nn.Embedding(3, 4)
            self.club = nn.Embedding(num_clubs + 1, config.get("club_embedding_dim", 8), padding_idx=0)
        team_width = 4 + 2 * config.get("club_embedding_dim", 8) if self.mode == "joint" else 0
        continuous_width = width - emb - team_width
        if continuous_width < 1 or width % 2 or width % config["heads"]:
            raise ValueError("Invalid embedding dimensions.")
        self.continuous = nn.Linear(len(NUMERIC), continuous_width)
        # Like the original TransformerEncoder construction, cloned layers start
        # from the same initial parameters. They learn independently afterwards.
        layer = nn.TransformerEncoderLayer(width, config["heads"], width * config.get("ff_multiplier", 4),
            config["dropout"], activation="relu", batch_first=True, norm_first=False)
        self.transformer = nn.TransformerEncoder(layer, config["layers"], enable_nested_tensor=False)
        self.dense = nn.Linear(width, config["dense_dim"])
        self.classes = K + 1 if self.mode == "team_stream" else 2 * K + 1
        self.output = nn.Linear(config["dense_dim"], self.classes + 2)

    def forward(self, actions, numeric, sides, home, away):
        pad = actions.eq(0)
        if pad.all(1).any() or (pad[:, :-1] & ~pad[:, 1:]).any():
            raise ValueError("Require nonempty, right-padded histories.")
        parts = [self.action(actions), self.continuous(numeric)]
        if self.mode == "joint":
            parts.extend([self.side(sides), self.club(home), self.club(away)])
        hidden = torch.cat(parts, -1)
        hidden = hidden + position_encoding(hidden.shape[1], hidden.shape[2], device=hidden.device, dtype=hidden.dtype)
        mask = None
        if self.mode == "joint":
            mask = torch.ones(hidden.shape[1], hidden.shape[1], device=hidden.device, dtype=torch.bool).triu(1)
        hidden = self.transformer(hidden, mask=mask, src_key_padding_mask=pad)
        if self.mode == "team_stream":
            hidden = hidden[torch.arange(len(hidden), device=hidden.device), (~pad).sum(1) - 1]
        output = self.output(F.relu(self.dense(hidden)))
        return output[..., :self.classes], output[..., self.classes:]


def class_weights(data):
    """Inverse-frequency AFL weights for Seq2Event, normalised to sum to one.

    The boundary is zero-weighted, following the authors' released objective.
    Joint next-team/action prediction uses ordinary, unweighted cross-entropy.
    """
    if data.mode == "joint":
        return torch.ones(2 * K + 1, device=data.device)
    y = data.frame.iloc[data.scored_rows].target.to_numpy(dtype=int)
    counts = np.bincount(y, minlength=K + 1).astype(float)
    weights = np.divide(1., counts, out=np.zeros_like(counts), where=counts > 0)
    weights[-1] = 0.
    if weights.sum() == 0:
        raise ValueError("Training has no ordinary action targets.")
    return torch.tensor(weights / weights.sum(), device=data.device, dtype=torch.float32)


def objective(logits, xy, target, target_xy, location, weights):
    valid = target.ge(0)
    if not valid.any():
        return (logits.sum() + xy.sum()) * 0.
    # Mean over eligible examples, NOT PyTorch's weighted-denominator mean.
    ce = F.cross_entropy(logits[valid], target[valid], weight=weights, reduction="none").mean()
    rmse = torch.sqrt(F.mse_loss(xy[location], target_xy[location]) + 1e-12) if location.any() else xy.sum() * 0.
    return 5 * ce + rmse


@torch.inference_mode()
def predict(model, data, batch_size):
    model.eval()
    rows, probability, xy = [], [], []
    for inputs, target, _, _, indices in data.batches(batch_size):
        logits, coordinates = model(*inputs)
        valid = target.ge(0)
        rows.append(indices[valid.cpu().numpy()])
        probability.append(logits[valid].softmax(-1).cpu().numpy())
        xy.append(coordinates[valid].cpu().numpy())
    rows = np.concatenate(rows)
    order = np.argsort(rows)
    if not np.array_equal(rows[order], data.scored_rows):
        raise ValueError("Prediction alignment differs from labelled examples.")
    return np.concatenate(probability)[order], np.concatenate(xy)[order]


def save_model(directory, model, encoder, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_save(_cpu_state(model), directory / "weights.pt")
    info = dict(metadata, contract=CONTRACT, config=model.config, encoder=asdict(encoder),
        numeric=list(NUMERIC), actions=list(INPUT_ACTIONS), fingerprint=state_fingerprint(model),
        weights_sha256=hashlib.sha256((directory / "weights.pt").read_bytes()).hexdigest())
    (directory / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def load_model(directory, device="cpu"):
    directory = Path(directory)
    info = json.loads((directory / "model.json").read_text())
    if info["contract"] != CONTRACT or info["numeric"] != list(NUMERIC) or info["actions"] != list(INPUT_ACTIONS):
        raise ValueError("Saved feature contract differs.")
    if hashlib.sha256((directory / "weights.pt").read_bytes()).hexdigest() != info["weights_sha256"]:
        raise ValueError("Checkpoint checksum differs.")
    encoder = EventEncoder(**info["encoder"])
    model = EventTransformer(info["config"], len(encoder.clubs))
    model.load_state_dict(torch.load(directory / "weights.pt", weights_only=True, map_location="cpu"))
    if state_fingerprint(model) != info["fingerprint"]:
        raise ValueError("Model fingerprint differs.")
    return model.to(device).eval(), encoder, info


@torch.inference_mode()
def forecast_next(model, encoder, observed_quarter, focal_side=None):
    """Forecast from observed quarter tokens, without future events or labels.

    Joint forecasts include every observed token. Seq2Event requires a focal side
    and at least the configured window's worth of team-stream tokens.
    """
    if observed_quarter.empty or observed_quarter.quarter_id.nunique() != 1:
        raise ValueError("Supply one nonempty observed quarter prefix.")
    ordered = observed_quarter.sort_values("display_order").reset_index(drop=True)
    if not np.array_equal(ordered.position.to_numpy(), np.arange(len(ordered))):
        raise ValueError("Supply the quarter from its first observed token.")
    if (ordered.period_seconds.diff().dropna() < 0).any():
        raise ValueError("Time reversals require source review.")
    frame = make_streams(ordered, model.mode)
    if model.mode == "team_stream":
        if focal_side not in (0, 1):
            raise ValueError("Seq2Event requires focal_side=0 (home) or 1 (away).")
        frame = frame.loc[frame.focal_side.eq(focal_side)]
        if len(frame) < model.config["window"]:
            raise ValueError("Insufficient history for the fixed Seq2Event window.")
        frame = frame.tail(model.config["window"])
    device = next(model.parameters()).device
    inputs = [torch.as_tensor(x, device=device)[None] for x in encoder.transform(frame)]
    model.eval()
    logits, xy = model(*inputs)
    if model.mode == "joint":
        logits, xy = logits[:, -1], xy[:, -1]
    probabilities = logits.softmax(-1)[0].cpu().numpy()
    result = dict(mode=model.mode, history_events=len(frame), location=xy[0].cpu().tolist())
    if model.mode == "team_stream":
        active = probabilities[:K] / probabilities[:K].sum()
        result.update(focal_side=focal_side, action_probabilities=dict(zip(BALL_ACTIONS, active.tolist())),
                      location_frame="focal team attacking +x; meaningful for ordinary continuation")
    else:
        joint = probabilities[:2 * K].reshape(2, K)
        result.update(next_event_probabilities={
            "home": dict(zip(BALL_ACTIONS, joint[0].tolist())),
            "away": dict(zip(BALL_ACTIONS, joint[1].tolist())),
            "boundary": float(probabilities[-1])},
            location_frame="home attacking +x; conditional on an ordinary action")
    return result
