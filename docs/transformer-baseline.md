# Seq2Event-inspired Transformer for shots and goals

This experiment tests causal self-attention on the same possession outcomes used
by the logistic, XGBoost and LSTM baselines. Two independently tuned classifiers
estimate a subsequent shot or a same-team goal from each observed event prefix.

## Run and review

Prepare the four seasons and complete the LSTM comparison described in the README.
The existing sequence extra supplies PyTorch; no additional dependency is needed.
Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model,sequence,notebook]"
.\.venv\Scripts\python.exe -m afl_opposition.transformer_experiment
.\.venv\Scripts\python.exe -m notebook notebooks/10_transformer_comparison.ipynb
```

CUDA is selected when available, otherwise CPU. Use `--device cpu` to force CPU.
See [LSTM installation notes](lstm-baseline.md#run-and-review) for the verified
PyTorch 2.11.0 CUDA 12.8 installation. This run uses float32 mathematical attention,
deterministic algorithms, no TF32, no cuDNN and no fused attention or MHA fastpath.
Exact reproducibility across different library versions or hardware is not promised.

Checkpoints include weights, optimiser state and random-number-generator states.
Resume an interrupted run with the same device and environment:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.transformer_experiment --resume
```

The runner checks the source, runtime, search protocol and data provenance before
resuming. A completed run is verified without training or scoring again. Outputs
are under `artifacts/transformer-2026/`, excluded from Git. Notebook 10 includes
executed outputs and reads saved artifacts without training or rescoring test data.

## Relationship to Seq2Event

[Seq2Event (Simpson et al., KDD 2022)](https://eprints.soton.ac.uk/458099/)
models football event sequences with discrete actions and continuous attributes.
This implementation borrows its mixed-input sequence-modelling idea: embed action
categories, project continuous features and model their sequence with a Transformer.

It is **not a reproduction of Seq2Event**. The original next-event prediction task
differs from the two binary possession-outcome tasks here. This adaptation uses
causal per-event classification, pre-normalised Transformer layers, GELU feed-forward
blocks and BCE heads. It does not forecast the next action, coordinates or time,
simulate matches, or implement a temporal point process.

## Inputs and architecture

The feature columns, label audits and eligible event states match earlier baselines.
Inputs stop strictly before the first explicit shot. The shot label asks whether a
shot follows within the inferred possession. The goal label asks whether the same
team scores before that possession ends, including a recovered miss followed by a
goal. Future shot flags and outcome annotations are labels, never input features.

| Input or stage | Representation |
| --- | --- |
| `action_type` | Learned embedding of dimension `embedding_dim`; training-fitted vocabulary |
| `x_norm`, `y_norm`, `gap_seconds`, `elapsed_seconds` | Training-scaled continuous values; Linear(4, `continuous_dim`) then ReLU |
| Combined event | Concatenate both representations; linear projection to `d_model` |
| Event order | Add sinusoidal positional encoding; apply input dropout |
| Sequence | `num_layers` causal self-attention blocks with `num_heads` heads |
| Feed-forward block | Hidden width `d_model * ff_multiplier`, GELU, dropout and residual connections |
| Event output | Final LayerNorm, output dropout, Linear(`d_model`, 1); sigmoid at inference |

The coordinate and time features stay continuous. Position encoding represents the
event's index; elapsed seconds separately represents time. IDs 0 and 1 are reserved
for padding and unknown actions; their embedding vectors start at zero. There are
12 observed action types in the fitting snapshots. Each Transformer layer receives
its own initial weights rather than copies of an identically initialised layer.

At event `t`, attention can use events `0 ... t` from that possession. A triangular
mask blocks later events; a separate key mask blocks padding. The loss ignores
padded positions. There is no cross-possession attention, future-length feature,
global sequence pooling, team/player embedding or sequence truncation. Development
sequences have median length 4, 99th percentile 20 and maximum 47. Dynamic positional
encoding supports longer prefixes, though generalisation beyond training lengths
is not established.

Tests verify prefix equivalence, future-gradient exclusion, padding invariance,
row alignment and CPU/CUDA checkpoint resumption. The implementation uses
[`TransformerEncoderLayer`](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TransformerEncoderLayer.html)
with explicit causal and padding masks.

## Loss and search

For possession `i` with `n_i` eligible events, the training objective is:

`mean_i(mean_t BCE(y_i, sigmoid(logit_i,t))) + lambda_l2 * sum(matrix_weights**2)`.

Every possession contributes total weight one. BCE is calculated from logits for
numerical stability. Explicit L2 applies to parameter tensors with more than one
dimension, including embedding, projection, attention and feed-forward matrices;
it excludes biases and LayerNorm scale vectors. Adam uses zero weight decay.
There is no class reweighting or oversampling. Gradient norms are clipped at 1.

Batches contain up to **512 possessions**, shuffled in length buckets. This differs
from the LSTM batch size of 1,024. The output bias starts at the fitting population's
class log-odds. Validation uses possession-weighted BCE without dropout or the L2
penalty. The online training curve includes active dropout and changing parameters,
so it is measured differently from end-of-epoch validation BCE.

Both targets use the same predeclared 24 unique, approximately balanced configurations
(candidate seed 20261013), with all fits using training seed 2026:

| Hyperparameter | Candidate values |
| --- | --- |
| Transformer layers | 1, 2, 3 |
| Model width | 32, 64, 128 |
| Attention heads | 2, 4, 8 |
| Feed-forward multiplier | 2, 4 |
| Action embedding dimension | 4, 8, 16 |
| Continuous projection dimension | 8, 16 |
| Dropout | 0, 0.1, 0.25 |
| Adam learning rate | 0.0003, 0.001, 0.003 |
| Explicit L2 lambda | 0, 0.000001, 0.00001, 0.0001 |

All candidates fit **2023–2024**, with the encoder fitted on that partition only.
Train all 24 to at most 6 epochs, continue the best 6 to 18, and then the best 2
to 48. Continuing restores the optimiser and random state. Stop after six epochs
without a new validation minimum. Keep every candidate's best observed checkpoint;
choose the lowest **2025 BCE**, breaking exact ties by parameter count then trial
index. AP, Brier and confusion matrices are diagnostics, not selection criteria.

This is a bounded search, not a full grid or a guarantee of optimal hyperparameters.
Early pruning can remove slower-learning configurations. Epoch count is also selected
using validation. Test results do not alter the candidate set or stopping rules.

## Refit and evaluation

Fit a fresh encoder and model on **2023–2025**, using the selected hyperparameters
and best epoch count. Refit for that fixed duration without validation stopping.
The selected mean-loss L2 coefficient stays unchanged when the dataset grows.
Both refits are saved and hashed before loading **2026** for evaluation. Parameter
fingerprints must remain unchanged during scoring, and test IDs, labels, sample
weights and constant probabilities must match the previous baselines.

The threshold remains **0.5**. Report possession-weighted BCE, Brier score, average
precision, accuracy, precision, recall, specificity, F1 and balanced accuracy, plus
raw and weighted confusion matrices and calibration tables. The completion manifest
is written only after both comparisons and integrity checks succeed.

## Selected configurations

| Setting | shot | goal |
| --- | --- | --- |
| Layers | 3 | 3 |
| Width | 32 | 128 |
| Heads | 4 | 4 |
| FF multiplier | 4 | 2 |
| Embedding | 4 | 4 |
| Continuous | 16 | 16 |
| Dropout | 0.1 | 0.0 |
| Learning rate | 0.003 | 0.001 |
| L2 | 1e-06 | 1e-06 |
| Epochs | 16 | 9 |
| Parameters | 39017 | 400649 |
| Validation BCE | 0.390703 | 0.278519 |

## 2026 benchmark results

All values below use possession weights. Lower BCE/Brier and higher AP are better.
Precision and recall use threshold 0.5.

| Target | Model | BCE | Brier | AP | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| shot | constant | 0.523087 | 0.169815 | 0.216341 | undefined | 0.00% |
| shot | logistic | 0.423148 | 0.134152 | 0.555929 | 67.28% | 28.75% |
| shot | xgboost | 0.398320 | 0.125400 | 0.612536 | 75.70% | 30.88% |
| shot | lstm_no_history | 0.401857 | 0.126606 | 0.606363 | 72.46% | 32.54% |
| shot | lstm_history | 0.394988 | 0.124298 | 0.618982 | 71.63% | 35.34% |
| shot | seq2event_transformer | 0.417787 | 0.133067 | 0.610934 | 58.67% | 47.78% |
| goal | constant | 0.337951 | 0.094666 | 0.105809 | undefined | 0.00% |
| goal | logistic | 0.297569 | 0.085786 | 0.284618 | 58.98% | 1.26% |
| goal | xgboost | 0.287580 | 0.082488 | 0.342662 | 65.71% | 10.49% |
| goal | lstm_no_history | 0.288036 | 0.082635 | 0.340169 | 65.75% | 9.79% |
| goal | lstm_history | 0.285696 | 0.082102 | 0.347233 | 63.41% | 11.53% |
| goal | seq2event_transformer | 0.289381 | 0.083034 | 0.340560 | 58.48% | 13.45% |

Differences below are **Transformer minus comparison model**. Negative BCE/Brier
and positive AP differences favour the Transformer.

| Target | Compared with | BCE difference | Brier difference | AP difference |
| --- | --- | --- | --- | --- |
| shot | lstm_history | +0.022799 | +0.008769 | -0.008049 |
| shot | xgboost | +0.019467 | +0.007667 | -0.001602 |
| goal | lstm_history | +0.003686 | +0.000932 | -0.006673 |
| goal | xgboost | +0.001801 | +0.000546 | -0.002103 |

For shots, the Transformer has higher test BCE than the history LSTM. The lowest BCE is from `lstm_history` and the highest AP is from `lstm_history`.

For goals, the Transformer has higher test BCE than the history LSTM. The lowest BCE is from `lstm_history` and the highest AP is from `lstm_history`.

Confusion matrices have actual classes as rows and predicted classes as columns:
`[[TN, FP], [FN, TP]]`. Raw counts represent event states; weighted counts sum
prefix weights and need not be integers.

| Target | Weighting | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- |
| shot | unweighted | 185345 | 23450 | 51738 | 30376 |
| shot | possession_weighted | 36418.95 | 3731.05 | 5788.35 | 5295.65 |
| goal | unweighted | 246718 | 2807 | 37683 | 3701 |
| goal | possession_weighted | 45295.17 | 517.83 | 4691.75 | 729.25 |

The shot Transformer's weighted mean test probability is **28.4%**, compared with an observed shot rate of **21.6%**. In the 0.5–0.6 probability band, the mean prediction is 54.7% while the observed shot rate is 39.0%. The calibration table shows overprediction, alongside the poorer shot BCE and Brier score. Higher recall at threshold 0.5 therefore does not imply better probability estimates. These test observations are reported without recalibrating or retuning either model.

## Interpretation and inference

This compares independently tuned model families. The Transformer adds continuous
feature projections and positional encoding as well as attention, and uses a
different batch size. It is not an isolated ablation of attention versus recurrence.
Search budgets are bounded, architectures differ and there is one training seed;
the results do not establish statistical significance or general Transformer
superiority. Prefixes within a match are correlated.

2026 was already observed for earlier models. It is a **shared benchmark**, not a
new untouched holdout. Compare models within each target because shot and goal
prevalence differ. Missing fixture coverage, inferred possession boundaries and
the earlier label exclusions still apply. Separately trained classifiers do not
enforce `P(goal) <= P(shot)`. Real deployment needs further temporal validation.

Saved weights are tensor state dictionaries loaded with `weights_only=True`;
preprocessing and architecture metadata are JSON. For inference:

```python
from pathlib import Path
from afl_opposition.transformer import load_model
from afl_opposition.sequence import make_sequences, predict

model, encoder, metadata = load_model(
    Path("artifacts/transformer-2026/goal/history/refit"), device="cpu"
)
data = make_sequences(observed_prefixes, encoder, labelled=False)
probabilities = predict(model, data, batch_size=512)
```

`observed_prefixes` is a DataFrame containing `event_id`, `possession_id`, contiguous
`event_index` starting at zero and the five feature columns above. Supply the complete
observed prefix for every included possession. Predictions align with the original
input row order. Labels and sample weights are unnecessary at inference.
