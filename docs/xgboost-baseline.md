# Current-state XGBoost benchmark

Two independent boosted-tree classifiers estimate (1) a subsequent recorded shot
and (2) a same-team goal before the inferred possession ends. They use the same
eligible pre-shot states as the tuned logistic models. A recovered miss followed
by a same-team goal remains goal-positive; inputs still stop before the first shot.

This experiment asks whether nonlinear relationships between the **existing five
features** improve probability estimates. It adds no sequence history. The full
interactive opposition analysis remains a later project stage.

## Reproduce and review

Follow the repository README to prepare 2023–2026 and run both tuned logistic
experiments (`afl_opposition.tuning` and `afl_opposition.goal_tuning`), then run
from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model]"
.\.venv\Scripts\python.exe -m afl_opposition.xgboost_baseline
.\.venv\Scripts\python.exe -m notebook notebooks/08_xgboost_comparison.ipynb
```

The command checks that logistic comparison artifacts exist before training.
Repeating a completed or partial XGBoost run requires explicit `--overwrite`.
It saves models, all search trials, per-round learning curves, validation results,
refit records, test predictions, audits, calibration and comparison tables under
`artifacts/xgboost-2026/{shot,goal}/`. These generated files are excluded from Git.
The final `summary.json` is written only after both evaluations and integrity checks
succeed. Notebook 08 reads saved artifacts, verifies checksums and contains executed
outputs; rerunning it does not retrain or rescore the models.

## Features, labels and weights

Inputs are `action_type`, `x_norm`, `y_norm`, `gap_seconds`, and `elapsed_seconds`.
Action type is one-hot encoded. Numeric features retain the logistic pipeline's
standardisation, although trees do not need it. The encoder and scaler fit on the
candidate training partition only, then fit afresh for the final refit. There are
16 encoded features in this snapshot. No outcome annotation or explicit shot flag
is a predictor. Timing summarises the observed possession; past action identities
are not features.

For a possession with `n` eligible event states, each has weight `1/n`. Thus each
possession contributes total weight one in training, validation and primary test
metrics. No class weighting or oversampling is applied. Existing reviewed,
censored and contradictory episodes remain excluded. The comparison explicitly
checks identical event IDs, labels, weights and constant predictions against each
logistic test artifact. See [preprocessing](preprocessing.md) and
[goal labels](goal-labels.md) for the underlying rules.

## Objective and model selection

XGBoost uses additive decision trees with a logistic link and binary cross-entropy
(`binary:logistic`, evaluation metric `logloss`). L2 regularises tree leaf values;
it is a different parameterisation from logistic-regression coefficient shrinkage.
Each target has an independent selection:

1. Fit preprocessing and candidate models on **2023–2024**.
2. Search depth `{2, 4, 6}`, learning rate `{0.05, 0.1}`, and native L2 `{0, 1, 10}`:
   **18 configurations per target**.
3. Use possession-weighted **2025 BCE** for early stopping (60 rounds without
   improvement), with a cap of 1,200 rounds. Save the best iteration, not the last.
4. Select minimum weighted validation BCE. Exact ties favour fewer trees, shallower
   trees, stronger L2, then earlier trial order. AP and Brier are diagnostics only.
5. Refit fresh preprocessing and a fresh booster on **2023–2025**, using the selected
   tree count without early stopping. Freeze both models before loading test inputs.
6. Evaluate **2026** once with unchanged models and a fixed threshold of **0.5**.

Fixed settings are CPU histogram trees, four threads, seed 2026, 256 histogram bins,
full row/column sampling, `min_child_weight=5` in selection, zero L1 and zero split
penalty. The initial score is the weighted prevalence of the relevant fitting data.
The fixed grid is a bounded experiment, not an exhaustive claim of optimality.

Native XGBoost sums weighted BCE and adds `reg_lambda / 2` times the sum of squared
leaf values. To preserve the penalty relative to **mean** weighted BCE when the
refit adds validation data, the native L2 parameter is multiplied by
`S_refit / S_train`, where `S` is total sample weight. The minimum child Hessian is
scaled by the same ratio. Thus `reg_lambda / (2*S)` remains fixed. Depth, learning
rate and selected number of trees stay fixed; no test values enter this adjustment.
The constant comparison is the weighted prevalence of the combined development data.

The official [parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html)
describes these parameters. The [Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
documents weighted `DMatrix` inputs and the `EarlyStopping` callback with `save_best=True`.

## Selected configurations

| Target | Depth | Learning rate | Selection L2 | Trees | Validation BCE | Refit native L2 |
| --- | --- | --- | --- | --- | --- | --- |
| shot | 6 | 0.1 | 1.0 | 147 | 0.387493 | 1.492287 |
| goal | 6 | 0.1 | 10.0 | 120 | 0.277795 | 14.922874 |

Training contributes 103,078 possessions; the refit contributes 153,822. Each test
target has 290,909 prefixes from 51,234 possessions across 218 matches.

Neither selected model reaches the declared tree cap.

## 2026 comparison

All probability metrics and classification rates below use possession weights.
Lower BCE/Brier and higher AP are better. Precision and recall use threshold 0.5.

| Target | Model | BCE | Brier | AP | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| shot | constant | 0.523087 | 0.169815 | 0.216341 | undefined | 0.00% |
| shot | logistic | 0.423148 | 0.134152 | 0.555929 | 67.28% | 28.75% |
| shot | xgboost | 0.398320 | 0.125400 | 0.612536 | 75.70% | 30.88% |
| goal | constant | 0.337951 | 0.094666 | 0.105809 | undefined | 0.00% |
| goal | logistic | 0.297569 | 0.085786 | 0.284618 | 58.98% | 1.26% |
| goal | xgboost | 0.287580 | 0.082488 | 0.342662 | 65.71% | 10.49% |

Confusion matrices for XGBoost: rows are actual classes, columns predicted classes;
the four entries are `[[TN, FP], [FN, TP]]`. Raw values count event states; weighted
values sum prefix weights and need not be integers.

| Target | Weighting | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- |
| shot | unweighted | 202430 | 6365 | 65547 | 16567 |
| shot | possession_weighted | 39051.08 | 1098.92 | 7660.84 | 3423.16 |
| goal | unweighted | 248222 | 1303 | 38728 | 2656 |
| goal | possession_weighted | 45516.36 | 296.64 | 4852.51 | 568.49 |

Notebook 08 also includes accuracy, specificity, F1, balanced accuracy, learning
curves and calibration bands. Each saved comparison directory contains full raw
and weighted confusion reports for constant, logistic and XGBoost models.

## Interpretation and limitations

- Compare AP within a target. Goal and shot prevalence differ, so their AP values
  should not be used to rank the tasks directly. A constant predictor's AP is the
  weighted positive prevalence, rather than 0.5.
- A probability below 0.5 can still identify a relatively threatening state. The
  fixed threshold is for comparison, not a deployment decision or a tuned alert.
- These are separate classifiers, without a constraint that goal probability be
  at most shot probability. Notebook 08 measures any such violations.
- The 2026 results were already observed for both logistic models. This is a shared
  benchmark, not a newly untouched holdout. Hyperparameters and early stopping use
  2025 only. Future claims need further temporal validation or a new season.
- Prefixes within possessions and matches are correlated. No statistical
  significance claim or confidence interval is attached to the point estimates.
- Data coverage is incomplete: the 2024 archive lacks one fixture, and inconsistent
  quarters and possessions are excluded. See [season coverage](season-coverage.md).
- The models estimate associations from public events. They do not measure causal
  action value or validate a live match intervention. Tracking, pressure and
  off-ball information are absent.

Native boosters are saved as UBJ alongside the fitted preprocessing transformer
and JSON metadata. `load_model(Path(...))` from `afl_opposition.xgboost_baseline`
loads a saved refit and offers `predict_proba(frame)`, with the positive probability
in column 1. The frame must obey the existing five-feature contract and possession
construction rules. Load only trusted transformer pickle files.
