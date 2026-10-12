# Transformer retraining: broader search and stable optimisation

Two causal classifiers estimate a subsequent shot or a same-team goal from an
observed possession prefix. This second experiment keeps the first Transformer's
architecture and five input features, and changes the training and selection
procedure. The original experiment, models and executed notebook remain available.

## Reproduce and review

Complete the earlier modelling experiments described in the README first. No new
dependency is required beyond the existing model, sequence and notebook extras.
From the repository root:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.transformer_retraining
.\.venv\Scripts\python.exe -m notebook notebooks/11_transformer_retraining.ipynb
```

The runner selects CUDA when available; `--device cpu` selects CPU explicitly.
Use `--resume` with the same source, protocol, data and runtime after interruption.
Completed runs are verified without fitting or rescoring. Artifacts live under
`artifacts/transformer-retrained-2026/` and are excluded from Git. Notebook 11 reads
these saved results and includes executed outputs. Its HTML export is saved under
`reports/generated/11_transformer_retraining.html`.

The verified run uses float32, deterministic algorithms, mathematical attention,
no TF32, no fused attention/MHA fastpath and no cuDNN. Exact reproducibility across
different library versions and hardware is not guaranteed.

The recorded run used **NVIDIA GeForce RTX 3050 Ti Laptop GPU**, PyTorch 2.11.0+cu128. Saved training intervals total approximately **2.56 hours**, including periodic evaluation but excluding data preparation, final reporting and other overhead. Runtime varies with hardware.

## What changed and why

The earlier shot model overpredicted even on its refit training data: weighted mean
probability was 27.79% against a 19.97% observed rate, with evaluation-mode BCE
0.404744. This pointed to a training issue worth investigating rather than attributing
the whole gap to the 2026 season. It did not demonstrate a causal-mask or label bug.

| Component | First experiment | Revised experiment |
| --- | --- | --- |
| Effective optimiser batch | 512 possessions grouped by length | 1,024 randomly mixed possessions |
| Padding efficiency | Length groups each trigger an update | Sort only inside each effective batch; accumulate microbatches of at most 512 |
| Learning rate | Constant | Optional linear warmup, then cosine decay to 5% of peak |
| Full-run stopping | Validation patience | Complete all 3,200 updates; select a saved checkpoint afterwards |
| Hyperparameter coverage | 24 configurations | 64 approximately balanced configurations with protected promotion |
| Final validation comparison | One seed | Three seeds at a common update count |
| Fresh refit duration | Same epochs, more updates after adding 2025 | Exact selected update count and learning-rate prefix |
| Training diagnostics | Online training loss | Online loss plus fixed-weight evaluation-mode loss and calibration |

Possession length correlates with outcomes. Grouping complete optimiser batches
by length can therefore create consecutive batches with quite different label
prevalence. Random effective batches address that source of optimisation variability.
Sorting their internal microbatches saves padding without changing which examples
contribute to an update. The L2 penalty and gradient clipping are applied once per
effective batch, after accumulating the data gradients. Partial batches use their
actual possession count for normalisation.

The learning-rate schedule is indexed by optimiser update, including during resume
and refit. Fixing updates prevents an unplanned increase when the refit population
grows. It also means fewer passes over the larger refit dataset; fixed updates are
a declared choice, not a claim that epoch-based refitting is always incorrect.

## Unchanged inputs, architecture and objective

Inputs are `action_type`, `x_norm`, `y_norm`, `gap_seconds` and `elapsed_seconds`.
The action embedding and projected continuous features are concatenated, projected
to model width, and combined with sinusoidal positions. Causal pre-normalised
Transformer layers use GELU feed-forward blocks; a final LayerNorm and binary head
produce a logit for each real event. Attention and loss exclude padding. Future
events cannot influence an earlier prediction. There is no sequence truncation.

Inputs stop before the first explicit shot. The goal target includes a recovered
miss followed by a same-team goal within the inferred possession. Both models use
the same eligible prefixes, with different outcome labels. Each possession has
total weight one:

`mean_i(mean_t BCE(y_i, sigmoid(logit_i,t))) + lambda_l2 * sum(matrix_weights**2)`.

Adam has zero weight decay; explicit L2 applies to matrices, excluding biases and
LayerNorm scale vectors. Gradient norms are clipped at 1. Validation minimises
possession-weighted BCE without the L2 penalty and with dropout disabled. No class
oversampling or positive-class weighting is used.

This remains [Seq2Event-inspired](https://eprints.soton.ac.uk/458099/), not a
reproduction of its next-event task. See the [first specification](transformer-baseline.md)
for architecture, causal-mask and label details.

## Search space and selection

Both targets use these candidate values:

| Hyperparameter | Values |
| --- | --- |
| Transformer layers | 1, 2, 3 |
| Model width | 32, 64, 128 |
| Attention heads | 1, 2, 4, 8, conditional on divisibility and head width 8–64 |
| Feed-forward multiplier | 2, 4 |
| Action embedding dimension | 4, 8, 16 |
| Continuous projection dimension | 8, 16, 32 |
| Dropout | 0, 0.1, 0.2, 0.3 |
| Peak learning rate | 0.0001, 0.0003, 0.001, 0.003 |
| Explicit L2 lambda | 0, 1e-7, 1e-6, 1e-5, 1e-4 |
| Warmup fraction | 0, 0.05, 0.1 |

The 64 unique configurations cover every listed value, using candidate seed
20261014. Four anchors include the earlier shot/goal winners and compact alternatives.
This is a bounded conditional search, not the Cartesian product or an exhaustive
optimisation. Every width/head combination is valid and avoids heads narrower than
eight dimensions.

1. Fit each configuration on 2023–2024 for 300 updates, completing a compressed
   warmup/decay schedule. Rank by 2025 BCE at update 200 or 300.
2. Promote the best four and up to four additional configurations protecting low
   learning rates (at most 0.0003), compact width 32, three layers and width 128.
   Fill remaining places by validation rank.
3. Restart these eight configurations from scratch for a complete 3,200-update
   schedule. A screened checkpoint is not continued under a different schedule.
4. Take the best three configurations and repeat full training with two additional
   seeds. Seeds are 2026, 2027 and 2028.
5. Select the configuration and common checkpoint update with the lowest mean
   2025 BCE across all three seeds. Exact ties favour fewer parameters, earlier
   trial, then earlier checkpoint. Checkpoints are evaluated every 100 updates;
   selection begins at update 200. No favourable seed is cherry-picked.

The screen can still discard slower learners despite protected promotion. The full
budget corresponds to about 31.7 passes over the fitting data. Choosing a checkpoint
at the budget boundary is evidence that this finite budget may limit the search.

The vocabulary/scaler are fitted only on 2023–2024 during selection. Fresh refits
fit a new encoder on 2023–2025 and use the selected configuration, exact update
count and original 3,200-update schedule horizon. All three seeds for both targets
are frozen before loading 2026 for evaluation. Seed 2026 is the predeclared primary
single model. Averaging all three probabilities is a separately labelled secondary
ensemble, with additional inference cost.

## Matched development diagnostics

A separate 2×2 diagnostic compares length-bucketed versus random effective batches,
and constant versus cosine learning rates. All four fits use the first anchored
architecture, seed 2026, 1,024-possession effective batches and 600 updates. This
helps assess training behaviour; it does not drive model selection, reproduce the
entire original experiment or isolate every reason for the final result. Training
audit gaps below use a fixed random sample of 4,096 fitting possessions, while
validation uses all 2025 possessions. Gaps are mean prediction minus observed rate.

| target | batching | schedule | validation_BCE | training_audit_gap | validation_gap |
| --- | --- | --- | --- | --- | --- |
| shot | bucketed | constant | 0.407985 | -0.0082 | -0.0141 |
| shot | bucketed | cosine | 0.398639 | -0.0077 | -0.0134 |
| shot | random | constant | 0.387674 | +0.0060 | +0.0005 |
| shot | random | cosine | 0.387830 | +0.0044 | -0.0010 |
| goal | bucketed | constant | 0.285070 | +0.0089 | -0.0045 |
| goal | bucketed | cosine | 0.281083 | +0.0087 | -0.0050 |
| goal | random | constant | 0.277798 | +0.0205 | +0.0074 |
| goal | random | cosine | 0.277702 | +0.0142 | +0.0012 |

## Selected configurations

| setting | shot | goal |
| --- | --- | --- |
| num_layers | 1 | 1 |
| d_model | 128 | 128 |
| ff_multiplier | 2 | 2 |
| embedding_dim | 16 | 16 |
| continuous_dim | 32 | 32 |
| dropout | 0.0 | 0.0 |
| learning_rate | 0.003 | 0.003 |
| lambda_l2 | 1e-07 | 1e-07 |
| warmup_fraction | 0.1 | 0.1 |
| num_heads | 2 | 2 |
| updates | 3200 | 3200 |
| parameters | 139521 | 139521 |
| validation_BCE | 0.382854 | 0.275915 |
| seed_SD | 0.000187 | 0.000140 |

Seed standard deviations describe optimisation variability for three fits. They
are not confidence intervals for population performance or independent match samples.

## Refit calibration checks

Before any new test load, each refit must beat the fitting population's constant
predictor and have an absolute mean-probability gap at most three percentage points
on all refit possessions. This predeclared sanity check catches large average
miscalibration; passing it does not establish calibration in every probability band.

| target | seed | BCE | mean_prediction | observed_rate | gap |
| --- | --- | --- | --- | --- | --- |
| shot | 2026 | 0.373560 | 19.7557% | 19.9692% | -0.2135 pp |
| shot | 2027 | 0.373712 | 20.1010% | 19.9692% | +0.1318 pp |
| shot | 2028 | 0.373917 | 19.2944% | 19.9692% | -0.6748 pp |
| goal | 2026 | 0.269708 | 9.5102% | 9.8555% | -0.3453 pp |
| goal | 2027 | 0.269754 | 9.8077% | 9.8555% | -0.0479 pp |
| goal | 2028 | 0.270004 | 9.4189% | 9.8555% | -0.4367 pp |

## Shared 2026 benchmark

For **shots**, the revised primary model's BCE changes by **-0.024782** relative to the first Transformer and **-0.001983** relative to the history LSTM. The lowest single-model test BCE is from `transformer_retrained`. Negative changes are improvements. At threshold 0.5, the revised model has higher precision, lower recall and lower F1 than the history LSTM.

For **goals**, the revised primary model's BCE changes by **-0.004174** relative to the first Transformer and **-0.000488** relative to the history LSTM. The lowest single-model test BCE is from `transformer_retrained`. Negative changes are improvements. At threshold 0.5, the revised model has higher precision, lower recall and lower F1 than the history LSTM.

Lower BCE and Brier are better; higher average precision is better. All values
below use possession weights. Classification uses the unchanged threshold 0.5.
`transformer_retrained` is seed 2026; `transformer_retrained_ensemble` averages the
three refit probabilities. The latter is not a like-for-like single-model comparison.

| target | model | BCE | Brier | AP | precision | recall |
| --- | --- | --- | --- | --- | --- | --- |
| shot | constant | 0.523087 | 0.169815 | 0.216341 | undefined | 0.0000 |
| shot | logistic | 0.423148 | 0.134152 | 0.555929 | 0.6728 | 0.2875 |
| shot | xgboost | 0.398320 | 0.125400 | 0.612536 | 0.7570 | 0.3088 |
| shot | lstm_no_history | 0.401857 | 0.126606 | 0.606363 | 0.7246 | 0.3254 |
| shot | lstm_history | 0.394988 | 0.124298 | 0.618982 | 0.7163 | 0.3534 |
| shot | seq2event_transformer | 0.417787 | 0.133067 | 0.610934 | 0.5867 | 0.4778 |
| shot | transformer_retrained | 0.393005 | 0.123556 | 0.623397 | 0.7582 | 0.3207 |
| shot | transformer_retrained_ensemble | 0.392694 | 0.123430 | 0.624178 | 0.7590 | 0.3215 |
| goal | constant | 0.337951 | 0.094666 | 0.105809 | undefined | 0.0000 |
| goal | logistic | 0.297569 | 0.085786 | 0.284618 | 0.5898 | 0.0126 |
| goal | xgboost | 0.287580 | 0.082488 | 0.342662 | 0.6571 | 0.1049 |
| goal | lstm_no_history | 0.288036 | 0.082635 | 0.340169 | 0.6575 | 0.0979 |
| goal | lstm_history | 0.285696 | 0.082102 | 0.347233 | 0.6341 | 0.1153 |
| goal | seq2event_transformer | 0.289381 | 0.083034 | 0.340560 | 0.5848 | 0.1345 |
| goal | transformer_retrained | 0.285207 | 0.081911 | 0.351444 | 0.6765 | 0.0993 |
| goal | transformer_retrained_ensemble | 0.284910 | 0.081818 | 0.353180 | 0.6767 | 0.1006 |

### Confusion matrices

Each row encodes `[[TN, FP], [FN, TP]]`. Unweighted entries count event
prefixes; possession-weighted entries sum their inverse-length weights. The latter
sum to the number of possessions, although classification occurs at every event.

| target | model | weighting | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- |
| shot | transformer_retrained | unweighted | 201918 | 6877 | 64200 | 17914 |
| shot | transformer_retrained | possession_weighted | 39016.71 | 1133.29 | 7529.63 | 3554.37 |
| shot | transformer_retrained_ensemble | unweighted | 201875 | 6920 | 64092 | 18022 |
| shot | transformer_retrained_ensemble | possession_weighted | 39018.59 | 1131.41 | 7520.83 | 3563.17 |
| goal | transformer_retrained | unweighted | 248260 | 1265 | 38789 | 2595 |
| goal | transformer_retrained | possession_weighted | 45555.49 | 257.51 | 4882.52 | 538.48 |
| goal | transformer_retrained_ensemble | unweighted | 248250 | 1275 | 38735 | 2649 |
| goal | transformer_retrained_ensemble | possession_weighted | 45552.44 | 260.56 | 4875.58 | 545.42 |

## Load a frozen model for inference

The revised loader verifies the saved weight checksum and model fingerprint. Use
the same preprocessed event-prefix schema as the sequence baselines; `prefixes`
below is a table of observed events with their IDs and order, without outcome labels.

```python
from pathlib import Path
from afl_opposition.transformer_training import load_model
from afl_opposition.sequence import make_sequences, predict

model, encoder, metadata = load_model(
    Path('artifacts/transformer-retrained-2026/goal/refit/seed-2026'),
    device='cpu',
)
sequences = make_sequences(prefixes, encoder, labelled=False)
probabilities = predict(model, sequences, batch_size=512)
```

Predictions are restored to input-row order. Inference keeps the fitted vocabulary,
scaler and weights frozen. For the secondary ensemble, load all three refits and
average their probabilities on the same rows; do not average logits. Raw event
files must first pass through the existing preprocessing and eligibility rules.

## Interpretation and limits

The earlier 2026 results motivated this development investigation, so 2026 is an
already-observed shared benchmark, not a new untouched holdout. All revised tuning
uses 2025, and no calibration or threshold adjustment is fitted on 2026. Better
point estimates do not establish significance or guaranteed generalisation.

The architecture is unchanged, but batching, schedule, search budget and selection
procedure change together. Final differences cannot be assigned to a single change.
The matched development diagnostic is narrower than the complete experiment.
Earlier LSTM/XGBoost searches were not rerun with equivalent budgets or ensembles.

Most event sequences are short: the development median is four events and the
99th percentile is twenty. Greater attention capacity need not improve this task.
The data lack player tracking, defender positions and other match context; model
capacity cannot recover information absent from the features. Possession boundaries
and outcomes remain inferred from public event annotations. The 2024 snapshot is
missing one fixture. Independent shot/goal heads do not enforce probability ordering.

Checkpoints include optimiser and random states. Tests cover gradient accumulation,
dropout-off evaluation, schedule replay, CPU/CUDA resume, validation-only selection,
all-model freezing before test loading and the pre-test calibration gate. Existing
tests continue to cover causal masks, future-event exclusion, padding and label audits.
