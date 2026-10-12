# LSTM embeddings: history versus no history

This experiment asks whether the observed action sequence improves shot and goal
probabilities beyond current-state models. Four independently fitted LSTMs cover
two targets and two input modes. Earlier logistic and XGBoost artifacts are kept.

## Run and review

Prepare the four seasons and complete the XGBoost comparison as described in the
README. Install the sequence extra and start from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model,sequence]"
.\.venv\Scripts\python.exe -m afl_opposition.lstm_experiment
.\.venv\Scripts\python.exe -m notebook notebooks/09_lstm_comparison.ipynb
```

The runner chooses CUDA when available, otherwise CPU. Use `--device cpu` to force
CPU operation. For a supported NVIDIA installation, this experiment was verified
with the official CUDA 12.8 wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

A CPU-only wheel is available from the official `https://download.pytorch.org/whl/cpu`
index using the same version. GPU wheel and driver compatibility depends on the
machine; see the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
The tested GPU used native CUDA LSTM kernels: cuDNN is disabled because its RNN
backend crashed during process shutdown on the verified Windows environment.
Native CUDA completed cleanly and was faster in the development benchmark.

Checkpoints include optimiser and random-number-generator states. If interrupted,
resume the same search with:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.lstm_experiment --resume
```

Resumption verifies source, development provenance and the search protocol. An
already completed run is verified without training or scoring it again. Changes
to the source or protocol require a separately planned experiment; `--resume`
does not silently mix incompatible runs. Generated files live in
`artifacts/lstm-2026/`, excluded from Git. Notebook 09 reads saved tables and includes
executed outputs. It does not retrain, tune or rescore test data.

## Inputs and architecture

Each possession contains its eligible events, ordered by `event_index` and starting
at zero. Inputs stop before the first explicit shot, as in all earlier baselines.
The five feature columns, exclusions, shot/goal labels and sample weights are
unchanged. Future outcome annotations are never features.

- `action_type`: training-fitted vocabulary, learned embedding; IDs 0 and 1 are
  reserved for padding and unknown actions. The unobserved unknown vector starts
  at zero. There are 12 observed action types in this snapshot.
- `x_norm`, `y_norm`, `gap_seconds`, `elapsed_seconds`: continuous values, scaled
  using the fitting partition. Coordinates and times are not discretised.
- Concatenate the embedding with the four numeric values and feed a unidirectional
  LSTM. An output dropout layer and linear projection produce one logit at every
  real event. Apply sigmoid for reported probabilities.
- With multiple recurrent layers, the chosen dropout also applies between layers.
  Output dropout is present with every layer count, including one.

`history` carries hidden and cell states through each possession and resets them
at its boundary. `no_history` reshapes every event into a length-one sequence,
resetting both states at every event. It uses the same LSTM implementation and
embedding/head components. The recurrent hidden-to-hidden weights in this control
cannot convey history; nominal parameter counts therefore overstate its effective
recurrent capacity. Predictions are restored to the original event IDs and order.

There is no cross-possession memory, bidirectionality, global sequence pooling,
team/player embedding or future-length input. Right padding is masked out of the
loss. Forward-only computation prevents later actions from changing an earlier
prediction. Development sequences have median length 4, 99th percentile 20 and
maximum 47; no fixed truncation limit is imposed.

See the official [embedding](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Embedding.html)
and [LSTM](https://docs.pytorch.org/docs/2.11/generated/torch.nn.LSTM.html) references.

## Loss and fitting

For `N` possessions and `n_i` eligible events in possession `i`, the objective is:

`mean_i(mean_t BCE(y_i, sigmoid(logit_i,t))) + lambda_l2 * sum(non_bias_weights**2)`.

Each possession contributes total weight one, exactly matching the baseline prefix
weights. Padding contributes zero. The loss is calculated from logits for numerical
stability using unreduced BCE, then averaged within possessions and across the
batch. No class reweighting or oversampling is used. L2 is an explicit term in the
objective, including embedding/recurrent/output matrices and excluding biases.
Adam has zero weight decay; this is not an AdamW approximation to that L2 term.

The batch size is 1,024 possessions, with shuffled length buckets to reduce padding.
Gradient norms are clipped at 1. The output bias starts at the fitting partition's
weighted class log-odds. Numeric features and vocabulary fit on 2023–2024 for the
search, and fit afresh on 2023–2025 for the refit. Explicit mean-loss lambda stays
fixed when adding validation data; no sample-size conversion is needed.

The recorded training curve is online minibatch BCE during training, with dropout
active and parameters changing. Validation BCE is measured at the end of an epoch
in evaluation mode, without dropout or the regularisation penalty.

## Hyperparameter search

All four target/variant combinations use the same predeclared 24 unique candidate
configurations, generated with seed 20261012 and approximately balanced coverage:

| Setting | Candidate values |
| --- | --- |
| Recurrent layers | 1, 2, 3 |
| Hidden units per layer | 32, 64, 128 |
| Embedding dimension | 4, 8, 16 |
| Dropout | 0, 0.1, 0.25, 0.4 |
| Adam learning rate | 0.0003, 0.001, 0.003 |
| Explicit L2 lambda | 0, 0.000001, 0.00001, 0.0001 |

Each candidate fits 2023–2024 and is ranked using possession-weighted **2025 BCE**.
Train all 24 for up to 6 epochs, continue the best 6 to 18 epochs, then continue
the best 2 to 48 epochs. Later stages resume the same weights, optimiser and random
state. Early stopping uses six epochs without a new minimum. Retain each candidate's
best validation checkpoint; select the lowest observed validation BCE, breaking
exact ties by smaller parameter count and then trial index. No test values enter
selection. AP, Brier and confusion matrices are reporting diagnostics.

This is a bounded, staged search rather than an exhaustive grid. Early pruning can
discard a slower-learning configuration. All neural fits use seed 2026; results
are not averages over independently trained seeds. Deterministic algorithms and
backend settings are recorded; exact reproducibility across different hardware
or library versions is not guaranteed.

## Refit and frozen evaluation

For each target and variant, initialise a fresh model and preprocessing on
**2023–2025**, keeping the selected architecture, learning rate, L2 strength and
best epoch count. Refit without a validation set or early stopping. All four refits
are saved and hashed before **2026** is loaded. Test evaluation checks parameter
fingerprints and exact event/label/weight agreement with previous baselines.
The reporting threshold stays at **0.5**. The completion manifest is written only
after the comparisons and integrity checks succeed.

## Selected configurations

| Target | Variant | Layers | Hidden | Embedding | Dropout | Learning rate | L2 | Epochs | Validation BCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shot | no_history | 2 | 32 | 16 | 0.0 | 0.003 | 0.0 | 27 | 0.390954 |
| shot | history | 2 | 32 | 16 | 0.0 | 0.003 | 0.0 | 27 | 0.384916 |
| goal | no_history | 1 | 32 | 16 | 0.0 | 0.003 | 0.0 | 43 | 0.278884 |
| goal | history | 2 | 32 | 16 | 0.0 | 0.003 | 0.0 | 27 | 0.276273 |

## 2026 probability and classification results

All values below use possession weights. Lower BCE/Brier and higher AP are better.
Precision and recall use threshold 0.5.

| Target | Model | BCE | Brier | AP | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| shot | constant | 0.523087 | 0.169815 | 0.216341 | undefined | 0.00% |
| shot | logistic | 0.423148 | 0.134152 | 0.555929 | 67.28% | 28.75% |
| shot | xgboost | 0.398320 | 0.125400 | 0.612536 | 75.70% | 30.88% |
| shot | lstm_no_history | 0.401857 | 0.126606 | 0.606363 | 72.46% | 32.54% |
| shot | lstm_history | 0.394988 | 0.124298 | 0.618982 | 71.63% | 35.34% |
| goal | constant | 0.337951 | 0.094666 | 0.105809 | undefined | 0.00% |
| goal | logistic | 0.297569 | 0.085786 | 0.284618 | 58.98% | 1.26% |
| goal | xgboost | 0.287580 | 0.082488 | 0.342662 | 65.71% | 10.49% |
| goal | lstm_no_history | 0.288036 | 0.082635 | 0.340169 | 65.75% | 9.79% |
| goal | lstm_history | 0.285696 | 0.082102 | 0.347233 | 63.41% | 11.53% |

In this run, history improves BCE, Brier and AP for both targets relative to the
no-history control. The history LSTMs also improve those probability metrics over
XGBoost, while the no-history models remain slightly behind XGBoost. The changes
are modest point estimates from one training seed. At threshold 0.5, history gives
higher recall but lower precision than XGBoost for both targets; probability
improvement does not imply improvement on every classification metric.

The confusion entries are `[[TN, FP], [FN, TP]]`, with actual classes as rows and
predicted classes as columns. Raw counts are event states; weighted entries sum
prefix weights and need not be integers.

| Target | Model | Weighting | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- |
| shot | lstm_no_history | unweighted | 201058 | 7737 | 64609 | 17505 |
| shot | lstm_no_history | possession_weighted | 38779.64 | 1370.36 | 7477.68 | 3606.32 |
| shot | lstm_history | unweighted | 199535 | 9260 | 61953 | 20161 |
| shot | lstm_history | possession_weighted | 38598.61 | 1551.39 | 7167.00 | 3917.00 |
| goal | lstm_no_history | unweighted | 248280 | 1245 | 38878 | 2506 |
| goal | lstm_no_history | possession_weighted | 45536.53 | 276.47 | 4890.19 | 530.81 |
| goal | lstm_history | unweighted | 247653 | 1872 | 38225 | 3159 |
| goal | lstm_history | possession_weighted | 45452.47 | 360.53 | 4796.23 | 624.77 |

## Interpretation

The LSTM variants receive equal search budgets but are tuned independently. Their
winning architectures may differ; the final comparison measures the best observed
version of each within this search. Matched-candidate validation results are also
saved for comparisons holding the candidate configuration fixed. This experiment
does not establish that embeddings outperform one-hot encoding.

2026 was already observed for logistic regression and XGBoost. It remains a shared
benchmark, not a new untouched holdout. Compare models within each target: shot and
goal prevalence differ. Prefixes within a match are correlated; no significance
claim or confidence interval accompanies these point estimates. Data exclusions,
missing coverage and inferred possession boundaries carry over from the earlier
experiments. Separate shot and goal classifiers do not enforce probability ordering.

For inference, `load_model(Path(...))` from `afl_opposition.sequence` returns the
model, fitted encoder and metadata. `make_sequences(frame, encoder, labelled=False)`
accepts complete observed possession prefixes, and `predict(model, data)` returns
probabilities aligned with the input row order. Required metadata are `event_id`,
`possession_id` and contiguous `event_index` starting at zero, alongside the five
feature columns. Model weights use tensor state dictionaries loaded with
`weights_only=True`; preprocessing is stored as JSON.
