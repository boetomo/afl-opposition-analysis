# Current state logistic regression baseline

This baseline estimates whether an observed possession will produce a subsequent recorded shot before its endpoint. A shot need not result in a goal. The model uses the current action, position and timing; it has no lagged action variables or sequence encoder.

This page records the fixed-`C` development experiments retained in notebooks 03 and 04. The subsequent [regularisation selection and final test experiment](tuned-logistic.md) selects lambda on 2025, refits on 2023–2025 and evaluates on 2026. Its results are in notebook 05.

## Run the baseline

Install the modelling extra, then run from the repository root after preprocessing:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model]"
.\.venv\Scripts\python.exe -m afl_opposition.baseline
```

Prepare 2023–2025 using the README commands, including the timestamp-quarantine option. Add `--overwrite` to regenerate the existing model experiment. Results are saved under `artifacts/logistic-regression-2023-2025/`, which is excluded from Git. For an explained, executable example with saved outputs:

```powershell
.\.venv\Scripts\python.exe -m notebook notebooks/04_season_validation.ipynb
```

## Prediction points and labels

Each example refers to an observed action within a processed possession. If the first recorded shot occurs at event index 3, indices 0, 1 and 2 receive target 1. The shot itself and subsequent events are excluded. For an eligible episode with a supported endpoint and no recorded shot, all observed action states receive target 0.

Flagged or censored episodes are excluded initially, including positive episodes with review flags. A score without an explicit shot flag is contradictory and excluded. An episode starting with a shot provides no pre-shot example. The baseline loader checks that each retrospective `Shot At Goal` marker has a preceding flagged kick by the same player and team in the same raw chain. The multi-season loader excludes all episodes touching a contradictory chain and records `shot_marker_conflict` in the label audit. The single-season loader retains its strict failure behavior. This check is for label quality only; it adds no model feature and does not invent a missing shot flag.

These are labels derived from archived annotations. An absent shot flag on an individual raw row is not a negative training label. Dataset annotation completeness and the inferred possession rules remain limitations.

## Features and model

| Input | Transformation |
| --- | --- |
| `action_type` | One-hot encoding of the current action |
| `x_norm`, `y_norm` | Standardisation of attacking-frame position |
| `gap_seconds` | Standardisation of time since the preceding retained action |
| `elapsed_seconds` | Standardisation of elapsed possession time |

The vocabulary and numerical scaling are fitted on training rows only. The [one-hot encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) retains all training categories; an unseen action is represented by zeros in the action block. This is a defined fallback, not evidence of reliability for new action types. In the 2023 run, 12 action categories plus four numerical inputs produce 16 encoded features.

The [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) uses L2 regularisation with `C=1`, the `lbfgs` solver and a maximum of 1,000 iterations. A convergence warning fails the run. There is no hyperparameter search, class balancing or additional probability-calibration fit.

One episode can contribute many overlapping prefixes. Each eligible episode receives total sample weight 1, divided equally among its prefixes. The same weights are used for training, evaluation and calibration summaries. The statistical scaler uses the training rows without episode weights. The weighted prediction objective corresponds to selecting an eligible possession uniformly, then selecting one of its eligible prefixes uniformly.

The feature allowlist prevents shot flags, target labels, final scores, endpoint reasons, episode totals and review decisions from entering the classifier. `elapsed_seconds` and `gap_seconds` summarise timing history; preceding action identities are not represented.

## Training loss

The training loss is binary cross-entropy (log loss), with the possession sample weights and an L2 penalty. For model coefficients $\beta$, intercept $b$, probabilities $p_i=\sigma(\beta^T x_i+b)$, sample weights $s_i$ and $S=\sum_i s_i$, the fitted objective is

$$
J(\beta,b)=-\frac{1}{S}\sum_i s_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
+\frac{\lVert\beta\rVert_2^2}{2CS}.
$$

Here $s_i=1/n_j$ for a prefix belonging to possession $j$ with $n_j$ eligible prefixes. Each possession therefore contributes total weight 1, and $S$ equals the number of eligible training possessions. This weighting addresses unequal possession lengths; it is not class balancing. `C=1` controls inverse regularisation strength, and `lbfgs` optimises the objective. The intercept is not penalised. This is the [scikit-learn binary logistic objective](https://scikit-learn.org/stable/modules/linear_model.html#binary-case), using L2 regularisation.

The reported training and validation `log_loss` values contain the weighted cross-entropy term only, excluding the regularisation penalty. AP, confusion counts, accuracy and F1 are evaluation quantities, not the function being minimised. The decision threshold is applied after probability prediction and does not enter the training loss.

## Whole-season development evaluation

Training uses 2023 and 2024; validation uses 2025. The 2026 archive is reserved for final testing after model selection and is not loaded or scored by the default command. The [coverage audit](season-coverage.md) records the missing 2024 match, excluded quarters and label conflicts. All matches and their possessions remain in one partition. Only training examples fit the encoder, scaler, classifier and constant prediction.

| Partition | Matches | Eligible possessions | Action states |
| --- | ---: | ---: | ---: |
| Training, 2023–2024 | 431 | 103,078 | 563,299 |
| Validation, 2025 | 216 | 50,744 | 281,476 |
| Test, 2026 | 218 in archive | Not evaluated in this development run | Not evaluated in this development run |

The development run excludes 10,422 inferred episodes: 10,312 preprocessing review cases, 102 with no pre-shot action, five scores without a shot flag, and three touching conflicting shot-marker chains. Quarantined quarters are excluded before episode construction and counted separately. The saved coverage table reports episode exclusions by team and split.

| 2025 validation metric | Constant prediction | Logistic regression |
| --- | ---: | ---: |
| Average precision, higher is better | 0.2042 | 0.5387 |
| Log loss, lower is better | 0.5063 | 0.4112 |
| Brier score, lower is better | 0.1625 | 0.1295 |

The constant probability is the weighted training shot rate, 0.1975. Its validation average precision is the validation prevalence, 0.2042, because every prediction has the same rank. The logistic model still uses 16 encoded features, fixed `C=1` and no hyperparameter search. Calibration tables compare mean predictions with observed shot rates in ten fixed probability bins.

## Confusion matrices and decision threshold

The default rule predicts a subsequent shot when $p\geq0.5$. This is an explicitly reported reference threshold, not an optimised operating point. Use `--threshold 0.3`, for example, to inspect a different threshold; select an operational threshold using validation and the relative cost of missed threats and false alerts. Test data must not guide that choice.

Each prediction is made at an eligible action state, not once per match or possession. The raw 2025 validation matrix for logistic regression at 0.5 is:

| Actual outcome | Predict no shot | Predict shot |
| --- | ---: | ---: |
| No subsequent shot | 195,412 TN | 9,935 FP |
| Subsequent shot | 59,808 FN | 16,321 TP |

These are 281,476 action states. Long possessions contribute more raw states. The corresponding possession-weighted matrix divides each state's contribution by the number of eligible states in its possession:

| Actual outcome | Predict no shot | Predict shot |
| --- | ---: | ---: |
| No subsequent shot | 38,860.65 TN | 1,521.35 FP |
| Subsequent shot | 7,431.15 FN | 2,930.85 TP |

Weighted cells are sums of fractional weights, not counts of unique possessions correctly or incorrectly classified. Their total is 50,744, the number of eligible validation possessions. A possession can contribute to multiple prediction columns as the probability changes along its prefixes.

| Metric at threshold 0.5 | Raw action states | Possession weighted |
| --- | ---: | ---: |
| Accuracy | 75.22% | 82.36% |
| Precision | 62.16% | 65.83% |
| Recall / sensitivity | 21.44% | 28.28% |
| Specificity | 95.16% | 96.23% |
| F1 | 31.88% | 39.57% |
| Balanced accuracy | 58.30% | 62.26% |

The matrix shows why accuracy or AP alone is insufficient: at 0.5 the model misses most positive action states. Lowering the threshold will increase recall and the number of positive predictions, normally also increasing false alerts. It does not improve the probability ranking or change AP.

`confusion.parquet` records TN, FP, FN and TP, the threshold and derived rates for each split and model under both weighting schemes. Matrix rows are actual outcomes and columns are predictions; shot is class 1. Precision is undefined when no positive prediction is made and remains missing rather than being presented as a meaningful zero. `metrics.parquet` retains the weighted probability metrics and adds the weighted threshold metrics. The fourth notebook also compares the random-score baseline at the same threshold.

## Coin flips, precision and average precision

A fair coin assigns the positive class with probability 0.5 independently of the outcome. Its expected **accuracy is 0.5**, regardless of class prevalence. Its expected **precision is the positive-class prevalence** (conditional on making positive predictions), because randomly selected examples retain the population's class mix.

For 100 examples containing 20 positives, a fair coin gives these expected counts:

| | Actual shot | Actual no shot |
| --- | ---: | ---: |
| Predict shot | 10 | 40 |
| Predict no shot | 10 | 40 |

Accuracy is `(10 + 40) / 100 = 0.50`; precision is `10 / (10 + 40) = 0.20`; recall is `10 / 20 = 0.50`. Always predicting no shot achieves 80% accuracy here while finding no shots, illustrating why accuracy alone is insufficient.

[Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) summarises precision across score thresholds, weighted by the corresponding increases in recall. It uses prediction scores and is not accuracy, precision at a single threshold, or trapezoidal PR-AUC. Independent random scores produce AP near prevalence, with finite-sample fluctuations. A constant score gives AP exactly equal to prevalence. A chance ROC-AUC of 0.5 is a different metric.

The project uses possession weights, so the relevant prevalence is the weighted rate: approximately 20.42% in the 2025 validation set. The fourth notebook includes an independent random-score simulation; thresholding those scores at 0.5 produces fair-coin predictions.

## Original 2023 experiment

The third notebook and its saved outputs retain the first within-season development experiment. Reproduce it with `python -m afl_opposition.baseline --season 2023`; it saves under `artifacts/logistic-regression-2023/`.

Approximately the earliest 70% of matches form training data, with later matches used for development validation. All matches on the cutoff date enter validation. A match and all of its possessions and prefixes belong to exactly one partition.

The verified 2023 run uses 151 training matches and 65 validation matches, with validation starting on 16 July 2023. It includes 52,429 possessions and 285,771 action states. Exclusions are 3,698 preprocessing review cases, 32 additional episodes with no pre-shot state and one goal without a shot flag. The saved possession audit and coverage table expose exclusions by partition and team.

| Validation metric | Constant prediction | Logistic regression |
| --- | ---: | ---: |
| Average precision, higher is better | 0.1949 | 0.5282 |
| Log loss, lower is better | 0.4932 | 0.4002 |
| Brier score, lower is better | 0.1569 | 0.1253 |

The constant prediction is the weighted training shot rate, approximately 0.1962. Average precision summarises the precision-recall curve; it is not trapezoidal PR-AUC. Calibration tables compare average predicted probability with the observed shot rate within fixed probability bins using the same episode weights.

The original results are a within-season comparison and should not be confused with the newer 2025 validation results. The different validation populations prevent treating the change in AP as a controlled estimate of the benefit of more training data. Neither experiment establishes live-match performance or provides confidence intervals for correlated prefixes. Future uncertainty estimates should resample whole matches. Excluding reviewed cases changes the sample and may bias reported performance. The preprocessed possession rules still require domain review.

## Saved outputs

The artifact directory contains the fitted pipeline (`model.pkl`), example table, possession audit, match assignments, predictions, metrics, confusion matrices, calibration bins, team coverage and coefficients. A JSON summary records the loss, decision rule, model settings, versions, source checksum and hashes of the processed inputs. Load only trusted pickle artifacts in a compatible environment.

The third notebook explains label examples, dummy variables and held-out predictions in the original 2023 experiment. The fourth notebook covers the current season split and metrics. Predicted probabilities can rise or fall between actions. This classifier only sees the current five-feature representation at each prediction point.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover pre-shot exclusion, contradictory and censored outcomes, equal episode weights, match/date/season separation, training-only category and scaler fitting, test-data isolation, explicit timestamp and shot-marker exclusions, rejection of leaked metadata as inputs, and prediction consistency after saving and loading the model.
