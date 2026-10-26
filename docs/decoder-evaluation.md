# Decoder evaluation after Optuna tuning

The completed chronological decoder study predicts the **next recorded team and action, or a neutral boundary, plus the next action's location**. It uses all observed events within the current quarter, causal attention and team/club context. The shared model applies across clubs. This is next-event prediction, not a shot/goal classifier or a validated tactical recommendation.

## Evaluation order

On 26 October 2026, the Seq2Event search was paused after epoch 32 of trial 36 (zero-based trial 35). Its model, optimiser, random-number and sampler checkpoints remain available for resumption. The decoder had already completed all 60 trials, nine finalist fits and three fresh refits.

The original [combined protocol](event-optuna.md) waited for both studies before scoring 2026. A separate evaluation-order amendment scores the finished decoder first. No decoder configuration, seed, stopping epoch or weight was changed. All three refits were checked and fingerprinted in [the pre-test manifest](../reports/examples/event-decoder-optuna-v1/frozen_decoder.json) before loading 2026. The source combined experiment remains unfinished and its files are unchanged. Seq2Event's existing search must remain fixed when it resumes.

## Selection and refitting

Candidates trained on **2023–2024** and were selected on **2025**. Of the 60 trials, 36 completed and 24 were pruned. The three strongest completed configurations were each fitted from scratch with seeds 2026, 2027 and 2028. Mean best validation objective selected trial 38 (zero-based). Its best epochs were 50, 57 and 56, giving a fixed **56-epoch refit** on **2023–2025** for each of the three seeds.

| Selected parameter | Value |
| --- | ---: |
| Model width / layers / attention heads | 64 / 4 / 4 |
| Action / club embedding dimensions | 8 / 4 |
| Feed-forward / output hidden dimensions | 128 / 128 |
| Dropout | 0.2613 |
| Adam learning rate | 0.001900 |
| Weight decay | 0 |
| Context | Complete observed quarter |

Selection minimised five times joint cross-entropy plus normalised location RMSE, with unit class weights. No test result selected a seed or an ensemble. The full [search space](event-optuna.md#search-space), [trial table](../reports/examples/event-decoder-optuna-v1/trials.csv) and [selection record](../reports/examples/event-decoder-optuna-v1/selection.json) remain available.

The winner uses the smallest offered width and embedding dimensions, the largest offered layer count and output hidden dimension, and zero weight decay. Two finalist seeds reached the 60-epoch limit, with their best epochs at 57 and 56. These are limits of this bounded search, not evidence that the global optimum was found.

## 2026 benchmark results

All rows below use the same **333,883 eligible next-event targets**, including 29,585 neutral boundaries, from **869 quarters**. Locations are scored on **304,298 valid coordinate targets** in dimensionless venue-normalised coordinates. The data audit and source provenance match the earlier chronological decoder experiment exactly.

| Model | Joint accuracy ↑ | Joint log loss ↓ | Macro F1 ↑ | Location RMSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Transition baseline | 57.856% | 1.23457 | 0.23660 | 0.19921 |
| Earlier decoder, one seed | 63.083% | 1.07954 | 0.31586 | 0.13514 |
| Optuna decoder, mean of three seeds | **63.614%** | **1.05767** | **0.33378** | **0.12488** |
| Optuna decoder, sample SD | 0.074 percentage points | 0.00179 | 0.00616 | 0.00161 |

Relative to the earlier decoder, mean accuracy rises by **0.53 percentage points**, log loss falls by **2.0%**, and location RMSE falls by **7.6%**. These are descriptive improvements. The earlier experiment used a smaller search and one seed, so the comparison does not isolate one architectural or training change.

On ordinary-action targets, team marginal accuracy averages **88.20%** and action marginal accuracy averages **68.46%**. These diagnostics condition on an actual ordinary action and renormalise away boundary probability. They are not the full 25-class prediction accuracy.

[Notebook 15](../notebooks/15_decoder_evaluation.ipynb) shows each seed, full confusion matrices for the predeclared first seed, baseline comparisons and per-class results across all seeds. All three seeds' complete matrices are included in the [compact report](../reports/examples/event-decoder-optuna-v1/). Rare actions remain difficult. For example, seed 2026 has zero recall for hard-ball gets and kick-ins for both teams despite much stronger recall for receives, marks, kicks and handballs. The model's overall accuracy does not imply reliable forecasts for every action.

## Reproduce the evaluation

The notebook reads the compact JSON/CSV reports and requires only the notebook environment. It does not load raw data, load weights or train models:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.\.venv\Scripts\python.exe -m notebook notebooks/15_decoder_evaluation.ipynb
```

Scoring the actual models requires the completed local decoder study, its three refitted models, the documented processed data and the matching training environment. With the combined study stopped, run:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.event_decoder_evaluation
```

The command verifies the frozen training sources, runtime, completed search, validation-selected configuration and all three refits. It locks the source experiment, records the frozen models, then scores 2026 without fitting anything. Results are saved separately to `artifacts/event-decoder-optuna-v1/` and compact reports to `reports/examples/event-decoder-optuna-v1/`. Existing results are preserved. Use new `--output` and `--report` directories for an explicitly requested repeat evaluation. Checkpoints, study databases and weights remain excluded from Git.

## Interpretation

The 2026 season is an already-observed project benchmark, not an untouched holdout. Evaluating the decoder before Seq2Event finishes changes the original evaluation order and makes decoder test results visible during the remaining study. Do not adjust the pending Seq2Event search in response to these results. The seed SD measures optimisation variability, not uncertainty across matches. No statistical significance or live-match reliability is established.

Quality filters favour cleaner recorded passages. Location predictions are point estimates rather than distributions. Opponent-specific aggregation and evaluation are still needed to turn these forecasts into a useful opposition report.
