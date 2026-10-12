# Regularisation selection and final test evaluation

The current-state logistic model selects its L2 coefficient using 2025 validation, refits on 2023–2025 and evaluates the frozen pipeline on 2026. The selected coefficient is **lambda = 5.6234132519 × 10⁻⁶**, defined as the multiplier of the squared coefficient norm in weighted mean binary cross-entropy. Its validation advantage over nearby weak penalties is negligible. The final model improves on a constant probability benchmark, but at a 0.5 decision threshold it still misses most positive action states.

## Reproduce and review

Prepare the pinned 2023–2026 snapshots with the commands in the README, then run:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.tuning
```

An existing experiment requires explicit `--overwrite`. Results are saved under `artifacts/logistic-tuned-2026/`, excluded from Git. The executable [fifth notebook](../notebooks/05_tune_and_test.ipynb) includes saved outputs and reads these artifacts without repeating model selection or test scoring. Earlier notebooks retain their original development experiments.

## Selection protocol

1. Construct eligible examples from 2023–2024 training and 2025 validation using the existing possession, feature and label rules. Test events are rejected by the selection function.
2. Fit the one-hot encoder and numeric scaler on 2023–2024 only. Fit a fresh logistic classifier for each penalty using these same transformed inputs.
3. Select the penalty with the lowest possession-weighted 2025 binary cross-entropy. The validation score excludes the L2 penalty. AP and Brier score are diagnostic columns, not selection criteria.
4. Fit a fresh encoder, scaler and classifier on all eligible 2023–2025 examples with the selected lambda fixed. Fit the constant probability benchmark on this combined data too.
5. Save the refitted model, its checksum, selected lambda, input provenance and code hashes before loading the 2026 examples. Apply the existing audit rules to 2026, then evaluate the frozen model. The classification threshold remains 0.5.

The coarse grid is zero plus powers of ten from `1e-8` to `10`. Zero checks the unregularised boundary. The predeclared procedure expands one boundary decade when needed and tests quarter-decade values around the best positive candidate; if zero is best, it first checks two weaker positive penalties. Exact loss ties favour larger lambda. This run evaluated 17 candidates. The result is the best tested value, not proof of a unique global optimum.

All candidates and the final refit use L-BFGS, `tol=1e-8` and `max_iter=2000`. A convergence warning fails the run. There is no class balancing, threshold search, probability-calibration fitting or change in action rules.

## Loss and fixed lambda during refitting

For prefix weight $s_i$, $S=\sum_i s_i$, coefficients $\beta$ and unpenalised intercept $b$:

$$
p_i=\sigma(\beta^T x_i+b),\qquad
J=\frac{1}{S}\sum_i s_i\operatorname{BCE}(y_i,p_i)+\lambda\|\beta\|_2^2.
$$

Each possession receives total weight one, divided across its eligible prefixes. The equivalent scikit-learn parameter is

$$C=\frac{1}{2\lambda S}.$$

This follows the [binary logistic objective](https://scikit-learn.org/stable/modules/linear_model.html#binary-case). When lambda is zero, `C` is infinite. For positive lambda, adding fitting data changes `S`, so `C` must change to preserve the selected coefficient in the mean-loss objective. Lambda acts on the coefficients of the transformed features; the final feature transformations are refitted on the combined data.

| Stage | Seasons | Eligible possessions / weight sum | Action states | C |
| --- | --- | ---: | ---: | ---: |
| Candidate fitting | 2023–2024 | 103,078 | 563,299 | 0.8625892092 for selected lambda |
| Validation | 2025 | 50,744 | 281,476 | No fitting |
| Final refit | 2023–2025 | 153,822 | 844,775 | 0.5780315592 |
| Final test | 2026 | 51,234 | 290,909 | No fitting |

The final refit covers 647 available matches; testing covers 218 matches with eligible examples. The best validation BCE is **0.4111784363**. The unregularised candidate is approximately **0.4111787**. This very small difference does not establish a practically meaningful benefit from the precise selected penalty, although stronger penalties materially worsen validation loss. The selected validation minimum has been used for tuning and is not an unbiased final test estimate.

## Held out 2026 results

The constant probability is learned from 2023–2025. All probability metrics below use possession weights, and reported BCE excludes the L2 term.

| Metric | Constant probability | Refitted logistic regression |
| --- | ---: | ---: |
| Binary cross-entropy / log loss | 0.523087 | **0.423148** |
| Brier score | 0.169815 | **0.134152** |
| Average precision | 0.216341 | **0.555929** |

The weighted test shot prevalence is 21.6341%. The constant predictor's tied scores produce AP equal to this prevalence.

At the fixed threshold $p\geq0.5$, the raw confusion matrix is:

| Actual outcome | Predict no shot | Predict shot |
| --- | ---: | ---: |
| No subsequent shot | 198,638 TN | 10,157 FP |
| Subsequent shot | 64,642 FN | 17,472 TP |

Each count is a prediction at an eligible action state. The possession-weighted matrix is:

| Actual outcome | Predict no shot | Predict shot |
| --- | ---: | ---: |
| No subsequent shot | 38,600.32 TN | 1,549.68 FP |
| Subsequent shot | 7,896.96 FN | 3,187.04 TP |

Weighted cells sum to 51,234 but are not counts of unique possessions classified correctly or incorrectly. One possession can contribute fractions to different prediction columns as its states change.

| Metric at 0.5 | Raw action states | Possession weighted |
| --- | ---: | ---: |
| Accuracy | 74.29% | 81.56% |
| Precision | 63.24% | 67.28% |
| Recall | 21.28% | 28.75% |
| Specificity | 95.14% | 96.14% |
| F1 | 31.84% | 40.29% |
| Balanced accuracy | 58.21% | 62.45% |

The model predicts shots relatively infrequently at 0.5. Its high specificity accompanies low recall; the matrix does not support calling it a reliable detector of all threatening states. Lambda selection optimises probability loss and does not itself select a useful operational threshold. The threshold was not changed after reading test results.

## Test exclusions and interpretation

The existing timestamp policy excludes three quarters containing 1,616 raw rows. Among 54,892 inferred possessions in the remaining quarters, 3,629 require preprocessing review and 29 start with a shot, leaving no pre-shot state. Those 3,658 episodes are excluded, leaving 51,234 eligible possessions. The label audit verified 1,969 auxiliary shot markers and found no additional shot-marker conflicts in the non-quarantined test quarters.

These exclusions follow the policies established before testing, including review of unsupported descriptions. They are retrospective sample selection and can bias performance. Source event completeness, inferred possession boundaries and live availability of the inputs remain limitations. Prefixes within possessions and matches are correlated; no independent-observation confidence intervals are claimed.

The 2026 result is now observed. Later architecture, feature or threshold changes motivated by it would use the test set as development evidence. A new claim of independent generalisation would need fresh held-out data or a separately designed evaluation protocol.

## Saved artifacts and verification

`lambda_search.csv` and `lambda_search.parquet` contain every trial. `selection/` contains the selected development-fit pipeline, validation predictions, metrics and audits. `refit_model.pkl` and `refit.json` record the combined-data fit before test loading. `test/` contains test predictions, raw and weighted confusion matrices, metrics, calibration bins, coverage, coefficients, label exclusions and the source provenance.

The refit record stores code and model hashes. The test summary references that record's hash. Notebook 05 verifies these links and displays the frozen and evaluation timestamps. Pickle files should only be loaded from trusted sources in a compatible environment.

The tests check that test data cannot enter selection, validation categories do not enter the selection encoder, the final refit includes validation features, `C` changes while lambda remains fixed, and test scoring cannot fit or alter the model. The full suite passes 79 tests.
