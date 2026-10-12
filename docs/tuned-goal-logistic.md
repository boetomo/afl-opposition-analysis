# Goal logistic-regression baseline

The goal baseline uses the five current-state features and the [audited pre-shot goal labels](goal-labels.md). It selects L2 regularisation on 2025, refits the full pipeline on 2023–2025, freezes it, then evaluates on 2026. The selected coefficient is **lambda = 1.77827941e-05**. Test probability metrics improve on a constant goal-probability benchmark; the confusion matrices below show the classification trade-off at the unchanged 0.5 threshold.

## Reproduce and review

Prepare the pinned 2023–2026 snapshots and preprocessed tables using the README, with the `model` and `notebook` extras installed. Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.goal_tuning
.\.venv\Scripts\python.exe -m notebook notebooks/07_goal_tune_and_test.ipynb
```

Results are saved under `artifacts/logistic-goal-tuned-2026/`, separately from the shot experiments. Repeating the training experiment requires explicit `--overwrite`. Notebook 07 reads the saved search, refit record and test reports without fitting or rescoring. Its eight code cells include explanations and executed outputs.

## Target and procedure

The target is a recorded goal by the same team before its inferred possession ends. Inputs stop before the first explicit shot; a recovered miss followed by a goal in the same episode remains positive. A completed episode without a goal is negative. The existing possession, quarantine, shot-marker and goal-attribution rules are applied to every season.

1. Fit the one-hot encoder and numeric scaler on 2023–2024 only. Use current action type, `x_norm`, `y_norm`, `gap_seconds` and `elapsed_seconds`.
2. Fit a fresh logistic classifier for each candidate lambda. Select the minimum possession-weighted 2025 BCE, excluding the L2 penalty from validation scoring.
3. Refit a fresh encoder, scaler and classifier on 2023–2025 with the selected lambda fixed. Fit the constant benchmark on the same combined data.
4. Write the model, selected settings, source/code hashes and freeze record before loading 2026.
5. Construct 2026 goal labels using the existing audit and evaluate the frozen model. Verify the saved and in-memory model are unchanged.

The search is shared with the shot baseline: zero plus powers of ten from `1e-8` through `10`, a predeclared boundary check, and quarter-decade refinement around the best positive candidate. Exact BCE ties favour larger lambda. This run tested **17** candidates. AP and Brier score are diagnostics; neither selects the model. All fitting uses L-BFGS with `tol=1e-8` and `max_iter=2000`; a convergence warning fails the run.

There is no class balancing, threshold search or calibration fitting. The decision threshold remains 0.5.

## Regularisation and refit

Each possession contributes total weight one, split across its eligible prefixes. For weight sum $S$, coefficients $\beta$ and an unpenalised intercept, the objective is

$$J=\frac{1}{S}\sum_i s_i\operatorname{BCE}(y_i,p_i)+\lambda\|\beta\|_2^2.$$

Scikit-learn's equivalent parameter is $C=1/(2\lambda S)$; lambda zero corresponds to infinite `C`. `C` changes on refitting because the fitting weight sum changes. This preserves the selected coefficient in the mean-loss objective. The numerical transformations are refitted on combined development data.

| Stage | Seasons | Possessions / weight sum | C |
|---|---|---:|---:|
| Candidate fitting | 2023–2024 | 103,078 | 0.2727746586 |
| Validation | 2025 | 50,744 | No fitting |
| Final refit | 2023–2025 | 153,822 | 0.1827896287 |
| Test | 2026 | 51,234 | No fitting |

The refit uses **844,775 prefixes** from **647 matches**. The selected validation BCE is **0.2876940151**, compared with **0.2876959539** for the unregularised candidate. The gain is 0.0000019388; the exact selected penalty should not be interpreted as a statistically unique optimum. Selection BCE has already been used for tuning and is not an unbiased final evaluation.

## 2026 probability results

The test contains **290,909 prefixes** from **51,234 eligible possessions** in **218 matches**. There are **5,421 goal possessions** and **45,813 no-goal possessions**.

All probability metrics use possession weights, and evaluation BCE excludes the penalty:

| Metric | Constant probability | Logistic regression |
|---|---:|---:|
| Binary cross-entropy | 0.337951 | 0.297569 |
| Brier score | 0.094666 | 0.085786 |
| Average precision | 0.105809 | 0.284618 |

The constant probability learned on 2023–2025 is **0.098555**. Weighted test goal prevalence is **10.5809%**. The constant's test AP equals that prevalence because every score is tied; its probability was not fitted to test labels.

Goal and shot predictions have different prevalences and targets. Their raw AP values are not directly comparable as measures of model quality.

## Confusion matrices at 0.5

The raw matrix counts predictions at action states:

| Actual outcome | Predict no goal | Predict goal |
|---|---:|---:|
| No goal | 248,910 TN | 615 FP |
| Goal | 40,610 FN | 774 TP |

The possession-weighted matrix is:

| Actual outcome | Predict no goal | Predict goal |
|---|---:|---:|
| No goal | 45,765.58 TN | 47.42 FP |
| Goal | 5,352.83 FN | 68.17 TP |

Weighted cells sum to 51,234, but are not integer counts of unique possessions classified correctly. A possession's different prefixes can contribute fractions to different prediction columns.

| Metric | Raw action states | Possession weighted |
|---|---:|---:|
| Accuracy | 85.83% | 89.46% |
| Precision | 55.72% | 58.98% |
| Recall | 1.87% | 1.26% |
| Specificity | 99.75% | 99.90% |
| F1 | 3.62% | 2.46% |
| Balanced accuracy | 50.81% | 50.58% |

At 0.5, weighted recall is 1.26% and specificity is 99.90%. Read these together: accuracy can be dominated by the no-goal class. Probability-loss improvements do not by themselves establish useful goal detection at this threshold. No threshold was changed after reading these results.

## Coverage and limits

The timestamp policy excludes 3 quarters containing 1,616 raw rows. The goal audit records 5,644 raw goals, including 16 in quarantined quarters. Of the remaining goal annotations, 5,628 pass team, order and endpoint verification. Goal attribution adds 0 exclusions beyond shot eligibility.

Of 54,892 inferred test possessions, 3,658 are excluded: 3,629 `preprocessing_review`; 29 `no_pre_shot_action`. These are retrospective exclusions and can affect representativeness. The data and inferred possession boundaries are not independent video ground truth. Prefixes within a possession and match are correlated; no independent-observation confidence intervals are claimed.

2026 was previously used for the shot benchmark. Goal-specific model selection uses 2025, but this is not a new untouched holdout. Further choices informed by these test results would require fresh held-out data for a new independent generalisation claim.

## Saved artifacts and checks

`lambda_search.csv` and `.parquet` record every candidate. `selection/` contains the training-only winner and validation reports. `refit_model.pkl` and `refit.json` record the combined fit before test loading. `test/` contains probabilities, raw/weighted confusion matrices, probability metrics, calibration bins, coefficients and coverage. `development_labels/` and `test_labels/` preserve complete goal-attribution audits and checksums.

The test summary references the refit record's checksum. The run verifies that the frozen model, its in-memory state and source code remain unchanged through testing. Load pickle files only from trusted sources in a compatible environment.

The full suite passes **106 tests**, including selection isolation, goal/shot target separation, refit transformations, preservation of lambda, frozen test scoring, artifact roundtrips and unchanged shot probability metrics. Earlier notebooks and saved shot experiments are preserved.
