# AFL Opposition Analysis

A research project modelling AFL ball movement from public match-event data. It asks: **given the play observed so far, which team performs the next recorded action, what is that action, and where does it occur?** Separate models estimate whether a possession will lead to a shot or goal.

The chronological next-event model is shared across clubs and uses team identity and match history as context. Its forecasts provide a starting point for studying opponents' ball-movement choices. Automated opposition reports and an interactive application remain planned.

**Status:** the Optuna-tuned chronological decoder has completed evaluation across three seeds. The Seq2Event Optuna study is paused and remains unfinished. Initial next-event experiments and shot/goal benchmarks are also available below.

## Start here

- **[Tuned decoder results (Notebook 15)](notebooks/15_decoder_evaluation.ipynb):** review the completed 60-trial search, three-seed evaluation, confusion matrices and comparison with the earlier decoder.
- **[Team-aware next-event prediction (Notebook 13)](notebooks/13_event_formulations.ipynb):** inspect the two model formulations, baseline comparisons, confusion matrices and a concrete forecast from an observed quarter.
- **[Within-possession next-action prediction (Notebook 12)](notebooks/12_next_event_prediction.ipynb):** inspect how observed events become inputs and targets, then review the earlier model and its example forecast.
- **[Shot and goal modelling (Notebook 11)](notebooks/11_transformer_retraining.ipynb):** review outcome prediction, validation tuning, calibration and comparisons with logistic regression, XGBoost and LSTMs.

These notebooks retain saved outputs for reading on GitHub. Notebooks 12, 13 and 15 also run from the [committed example reports](reports/examples/) without raw match archives or model weights. To run those report walkthroughs locally after creating the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.\.venv\Scripts\python.exe -m notebook notebooks/13_event_formulations.ipynb
```

The [LSTM history comparison](notebooks/09_lstm_comparison.ipynb) and [goal-label audit](notebooks/06_goal_labels.ipynb) provide further methodological detail. Earlier modelling notebooks require their documented local experiment artifacts to rerun.

## Team streams and chronological next-event prediction

Two completed experiments evaluate an AFL adaptation of the central Seq2Event architecture and a decoder-only Transformer that distinguishes the acting team. Seq2Event reads fixed 40-token team streams with unmasked attention. The decoder reads both teams' events in order, uses causal attention and retains all observed history within each quarter.

The initial search screens four configurations per formulation and continues two finalists, using one random seed. Each model is tuned on 2025, refitted on 2023–2025 and evaluated on the already-observed 2026 benchmark. Results below compare each formulation with its own transition baseline. **The tasks, populations, class weights and coordinate masks differ, so these rows are not a cross-task model ranking.**

| Prediction task | Model | Accuracy ↑ | Log loss ↓ | Location RMSE ↓ |
| --- | --- | ---: | ---: | ---: |
| Team-stream next action | Transition baseline | 67.7% | 0.8540 | 0.1422 |
| Team-stream next action | Seq2Event adaptation | 56.9% | 1.1069 | 0.1057 |
| Chronological next team–action/boundary | Transition baseline | 57.9% | 1.2346 | 0.1992 |
| Chronological next team–action/boundary | Causal decoder | 63.1% | 1.0795 | 0.1351 |

Seq2Event action metrics exclude boundary targets. Decoder joint metrics include a neutral boundary class; team and action marginal diagnostics are reported separately. Location errors use dimensionless venue-normalised coordinates. These experiments do not yet produce validated tactical recommendations or an opposition report.

Notebook 13 also includes a predetermined forecast from the first recorded 2026 quarter: the observed history, next team/action probabilities, predicted location and actual next event. This example was selected independently of prediction quality.

[Formulations, fidelity and reproduction](docs/event-formulations.md) · [Examples, model selection and confusion matrices](notebooks/13_event_formulations.ipynb).

## Next-action and location prediction

Given the observed possession history, predict the next retained ball-control action and its location, or `end_possession`. One shared Transformer uses a growing history capped at a validation-selected length. It fits on 2023–2024, tunes on 2025, refits on 2023–2025 and evaluates the frozen model on 2026.

The 2026 evaluation contains **304,870 labelled prefixes**. Each prefix has equal weight. Location errors are measured only for actual continuations and use venue-normalised coordinates.

| Model | Action accuracy ↑ | Action log loss ↓ | Location RMSE ↓ |
| --- | ---: | ---: | ---: |
| Frequency + persistence | 27.8% | 1.8468 | 0.1625 |
| Action transition | 63.3% | 0.8869 | 0.1413 |
| Seq2Event-style Transformer | 70.8% | 0.7833 | 0.1114 |

The Transformer's macro F1 is **0.296** across thirteen declared classes, including two with no test targets. Recall remains weak for several rare actions; read overall accuracy alongside the per-action results.

This earlier within-possession experiment uses a bounded, single-seed search and the already-observed 2026 benchmark. It has no team-identity inputs and does not generate tactical recommendations. [Dataset, architecture and full evaluation](docs/next-event-prediction.md) · [Walkthrough with saved outputs](notebooks/12_next_event_prediction.ipynb).

## Completed shot and goal benchmarks

Models estimate whether the same team will take a subsequent shot or score a goal before possession ends, using states observed before the first explicit shot. Candidates fit on **2023–2024**, tune on **2025**, then refit on **2023–2025** before scoring **2026**.

The 2026 benchmark contains **290,909 event states from 51,234 possessions across 218 matches**. Metrics give each possession total weight one. Lower binary cross-entropy (BCE) and higher average precision (AP) are better.

| Model | Shot BCE ↓ | Shot AP ↑ | Goal BCE ↓ | Goal AP ↑ |
| --- | ---: | ---: | ---: | ---: |
| Constant prevalence | 0.52309 | 0.21634 | 0.33795 | 0.10581 |
| Logistic regression | 0.42315 | 0.55593 | 0.29757 | 0.28462 |
| XGBoost | 0.39832 | 0.61254 | 0.28758 | 0.34266 |
| LSTM with history | 0.39499 | 0.61898 | 0.28570 | 0.34723 |
| Revised Transformer | **0.39301** | **0.62340** | **0.28521** | **0.35144** |

The revised Transformer leads the single-model comparison on BCE, Brier score and AP for both targets. At threshold 0.5, it has higher precision but lower recall and F1 than the history LSTM. The full comparison includes the LSTM without history, the first Transformer and a separately reported three-seed ensemble.

**Interpretation:** 2026 is an already-observed shared benchmark. Search budgets differ between model families, and these point estimates do not establish statistical superiority or live-match reliability. [Full results, confusion matrices and limitations](docs/transformer-retraining.md).

## Optuna-tuned decoder results

The decoder completed 60 trials, three-seed finalist selection and three fresh 56-epoch refits. Selection used 2025 only. All three refits were frozen before this evaluation on the shared 2026 benchmark.

| Model | Next team/action accuracy ↑ | Log loss ↓ | Macro F1 ↑ | Location RMSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Transition baseline | 57.86% | 1.2346 | 0.2366 | 0.1992 |
| Earlier decoder, one seed | 63.08% | 1.0795 | 0.3159 | 0.1351 |
| Tuned decoder, three-seed mean | **63.61%** | **1.0577** | **0.3338** | **0.1249** |

These rows use the same 333,883 next-event targets. Accuracy improves by 0.53 percentage points and location RMSE falls by 7.6% relative to the earlier decoder. Across seeds, accuracy SD is 0.074 percentage points and location RMSE SD is 0.00161. Rare-action recall remains weak, and 2026 is an already-observed benchmark.

**Evaluation-order amendment:** the Seq2Event study is paused after epoch 32 of trial 36. The decoder was evaluated separately before that study finished, with its configuration, seeds and weights unchanged. This changes the original combined evaluation order. Seq2Event's search remains frozen for resumption, and the combined experiment is not marked complete.

[Results and reproduction](docs/decoder-evaluation.md) · [Executed decoder notebook](notebooks/15_decoder_evaluation.ipynb) · [Search protocol](docs/event-optuna.md) · [Combined-study progress](notebooks/14_optuna_event_models.ipynb). Saved progress outputs are dated snapshots. Model weights, checkpoints and study databases remain outside Git.

## AI-assisted development

This project was developed by Bastian Oetomo with assistance from OpenAI Codex for implementation, debugging, automated tests, experiment execution and documentation. Bastian directed the research questions, model comparisons and evaluation protocol, and requested revisions when methods or results needed further scrutiny.

The workflow combines human direction with agent-assisted development:

- **Define the experiments.** Establish shot and goal targets, compare simple baselines with sequence models, and specify validation-based hyperparameter selection followed by refitting on training plus validation data and evaluation with frozen models.
- **Investigate model behaviour.** Compare LSTMs with and without possession history, then examine the initial Transformer's training diagnostics and revise its optimisation and hyperparameter search. The [LSTM comparison](notebooks/09_lstm_comparison.ipynb) and [Transformer retraining report](docs/transformer-retraining.md) record these experiments.
- **Make the work reviewable.** Retain source checksums, label audits, automated tests and executed notebook outputs; report calibration, confusion matrices and limitations, including repeated use of the 2026 benchmark.

Codex's assistance is acknowledged as part of the development process. The maintainer remains responsible for reviewing the implementation, interpreting the results and assessing whether the models are suitable for a particular use.

## Project aim

> How do opponents move the ball, which actions are likely next, and when does a possession become threatening?

The next-event models describe likely ball movement from observed history. The shot and goal models estimate possession outcomes. A planned reporting layer will combine these predictions with summaries of each team's observed patterns and supporting match examples. Forecasts alone do not establish which defensive response would improve an outcome.

## Planned output

An interactive possession viewer and a downloadable opposition report containing movement patterns, shot and goal rates, model estimates and supporting match examples.

## Data

The source is the public [torpdata match-event dataset](https://github.com/peteowen1/torpdata/releases/tag/chains-data). Snapshots for 2023–2026 are pinned by URL and checksum in [data/sources.json](data/sources.json).

Model selection uses 2023–2024 for fitting and 2025 for validation. The selected model is refitted on 2023–2025 and evaluated on 2026. The 2024 archive contains 215 of 216 fixture matches. Timestamp and label inconsistencies are excluded with an audit trail; see [season coverage](docs/season-coverage.md).

## Development setup

Requires Python 3.14 or later. The setup below is verified on Windows with Python 3.14.7. Run these commands from the repository root in PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

The virtual environment isolates the project's installed packages. The editable installation imports `afl_opposition` directly from `src/`, so source edits are available without reinstalling the package. Pandas handles the tables and PyArrow reads the Parquet files.

Verify the installation:

```powershell
.\.venv\Scripts\python.exe -c "import afl_opposition; print('afl_opposition import: OK')"
```

A successful import prints `afl_opposition import: OK`. The `.venv` directory is excluded from Git.

## Load and inspect data

Run from the repository root after installation:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.data --season 2023
.\.venv\Scripts\python.exe -m afl_opposition.inspect_data data/raw/chains_data_2023_all.parquet --output reports/generated/data-inspection-2023.txt
```

The first command downloads about 5.6 MB into `data/raw/`, verifies its size and SHA-256 checksum, and reuses a matching local copy on later runs. A mismatch stops the command; review the source before changing the recorded checksum or replacing a cached file.

The second command prints the schema, missing values, event counts and an ordered passage of play, and saves the same output as text. It does not clean the events or build model inputs. Raw data and generated reports are excluded from Git. See [the initial data inspection](docs/data-inspection.md) for findings and field meanings.

## Explore in a notebook

Install the optional Jupyter tools, register the project kernel and open the exploration notebook:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.\.venv\Scripts\python.exe -m ipykernel install --user --name afl-opposition-analysis --display-name "Python (afl-opposition-analysis)"
.\.venv\Scripts\python.exe -m notebook notebooks/01_inspect_raw_events.ipynb
```

Choose **Python (afl-opposition-analysis)** in the notebook's kernel selector. The registration command makes this virtual environment available by name in Jupyter; refresh an already-open browser tab to update its kernel list.

The notebook covers the schema, event ordering, team ownership and shot flags. Each code cell is accompanied by an explanation of its calculations and data interpretation. Run cells in order with **Shift+Enter**. The notebook supports launch from the repository root or `notebooks/`.

Set `MATCH_ID`, `QUARTER` and `CHAIN_NUMBER` to select a passage, then rerun subsequent cells. Use **Restart Kernel and Run All Cells** for a complete run from a clean kernel. The notebook includes saved outputs for reference; rerunning the cells refreshes the tables. Stop the local server with **Ctrl+C** in its terminal.

## Preprocess event sequences

After loading the raw snapshot, build the action, possession and audit tables:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.preprocess --season 2023
```

Outputs are saved under `data/processed/2023/` and excluded from Git. Add `--overwrite` to regenerate them. Possessions follow explicit ball-control rules; ambiguous cases carry review flags. This stage does not create training labels or fit a model. See [the preprocessing rules](docs/preprocessing.md) for the feature contract and limitations.

Open the second notebook for before-and-after examples and saved results:

```powershell
.\.venv\Scripts\python.exe -m notebook notebooks/02_preprocess_events.ipynb
```

Run the preprocessing tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Train the logistic regression baseline

Install the modelling extra, prepare the development seasons and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model]"
foreach ($season in 2023, 2024, 2025) {
    .\.venv\Scripts\python.exe -m afl_opposition.data --season $season
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $season" }
    .\.venv\Scripts\python.exe -m afl_opposition.preprocess --season $season --quarantine-time-reversals --overwrite
    if ($LASTEXITCODE -ne 0) { throw "Preprocessing failed for $season" }
}
.\.venv\Scripts\python.exe -m afl_opposition.baseline
```

The baseline predicts a subsequent recorded shot from current action type, position and timing. It fits the encoder, scaler and classifier on 2023–2024 only, then compares logistic regression with a constant shot-rate prediction on 2025. Training minimises possession-weighted binary cross-entropy with L2 regularisation. Evaluation includes raw and weighted confusion matrices, accuracy, precision, recall, specificity, F1, balanced accuracy, AP, log loss and Brier score. The default decision threshold is 0.5; `--threshold` changes the reporting threshold without changing the training objective. Add `--overwrite` to regenerate an existing model experiment. The preprocessing commands above explicitly regenerate derived tables; they preserve the raw snapshots.

Model and evaluation files are saved under `artifacts/logistic-regression-2023-2025/` and excluded from Git. This development command does not load or score 2026. See [the baseline specification and results](docs/logistic-baseline.md) for exclusions, weighting and metrics.

Open the fourth notebook for the season split, loss function, confusion matrices, current results and an explanation of coin-flip accuracy versus average precision:

```powershell
.\.venv\Scripts\python.exe -m notebook notebooks/04_season_validation.ipynb
```

The third notebook retains the original 2023 development experiment and its saved outputs. Reproduce that experiment with `python -m afl_opposition.baseline --season 2023`.

## Select regularisation and evaluate the final model

After preparing 2023–2025 above, prepare the test archive and run the complete experiment:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.data --season 2026
if ($LASTEXITCODE -ne 0) { throw "Test download failed" }
.\.venv\Scripts\python.exe -m afl_opposition.preprocess --season 2026 --quarantine-time-reversals --overwrite
if ($LASTEXITCODE -ne 0) { throw "Test preprocessing failed" }
.\.venv\Scripts\python.exe -m afl_opposition.tuning
```

The search selects lambda by minimum weighted 2025 binary cross-entropy. It then fits a fresh encoder, scaler and classifier on 2023–2025, keeping lambda fixed, and scores 2026 with the frozen pipeline. The threshold remains 0.5. The [experiment specification and results](docs/tuned-logistic.md) explain the search, lambda-to-`C` conversion, exclusions and test metrics.

Outputs are saved under `artifacts/logistic-tuned-2026/`. Repeating the experiment requires an explicit `--overwrite`. The fifth notebook reads the saved search, refit record and test results without repeating model selection or test scoring:

```powershell
.\.venv\Scripts\python.exe -m notebook notebooks/05_tune_and_test.ipynb
```

The test result is now observed. Further choices informed by that result should not be described as independently validated on the same holdout.


## Construct and inspect goal labels

After preparing the development seasons above, generate the goal-label audit:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.goal_labels
.\.venv\Scripts\python.exe -m notebook notebooks/06_goal_labels.ipynb
```

The target is a same-team goal before the inferred possession ends. Inputs retain the existing cutoff before the first explicit shot, including when a later recovery leads to a goal. The command verifies goal attribution, preserves existing exclusions and saves labels with source checksums under `artifacts/goal-labels-2023-2025/`. Repeating label generation requires `--overwrite`.

The sixth notebook includes executed examples of a goal, a behind, a turnover and a recovered miss followed by a goal. It shows which rows become inputs, along with season counts, exclusions and weight checks. This stage does not fit a goal model or load the test season. See [the goal-label specification and results](docs/goal-labels.md).


## Train and evaluate the goal baseline

After preparing all four seasons, run the goal experiment:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.goal_tuning
.\.venv\Scripts\python.exe -m notebook notebooks/07_goal_tune_and_test.ipynb
```

The procedure fits candidates on 2023–2024, selects L2 regularisation by weighted 2025 BCE, refits the full pipeline on 2023–2025 and freezes it before loading 2026. It uses the existing pre-shot goal labels and five current-state features. Test reports include probability metrics, raw and possession-weighted confusion matrices at 0.5, calibration bins and a goal-attribution audit.

Results are saved separately under `artifacts/logistic-goal-tuned-2026/`; repeating training requires explicit `--overwrite`. Notebook 07 reads the saved results and includes executed outputs. See [the goal experiment specification and results](docs/tuned-goal-logistic.md). The earlier shot results remain available for that separate target.


## Compare XGBoost for shots and goals

After running both tuned logistic experiments above, install the updated modelling
extra and train the two XGBoost classifiers:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model]"
.\.venv\Scripts\python.exe -m afl_opposition.xgboost_baseline
.\.venv\Scripts\python.exe -m notebook notebooks/08_xgboost_comparison.ipynb
```

The models use the same current-state inputs, labels and possession weights as
logistic regression. Each searches tree depth, learning rate and L2 regularisation
on 2025, with validation early stopping choosing the number of trees. Fresh models
refit on 2023–2025 and both are frozen before evaluation on 2026.

Results under `artifacts/xgboost-2026/` include validation searches, learning curves,
saved models, test predictions, calibration and raw/weighted confusion matrices at
0.5. Repeating training requires `--overwrite`. Notebook 08 includes executed outputs
and reads the saved results without retraining. See [the specification and comparison](docs/xgboost-baseline.md).


## Compare LSTMs with and without history

After completing the XGBoost experiment, install the sequence extra and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model,sequence]"
.\.venv\Scripts\python.exe -m afl_opposition.lstm_experiment
.\.venv\Scripts\python.exe -m notebook notebooks/09_lstm_comparison.ipynb
```

The history model reads the observed possession sequence; the control resets its
state at every event. Each target and variant independently tunes layer count,
hidden size, embedding dimension, dropout, learning rate and L2 using 2025 BCE.
Fresh models refit on 2023–2025 for the selected epoch counts. All four models are
frozen before evaluation on 2026, with the same labels, weights and threshold as
the existing baselines.

Results and resumable checkpoints are saved under `artifacts/lstm-2026/`. Add
`--resume` after an interruption; a completed run is verified without repeating it.
The runner uses CUDA when available or CPU otherwise. See [the LSTM specification
and results](docs/lstm-baseline.md) for installation options, search details,
limitations and confusion matrices. Notebook 09 includes executed explanations
and reads saved results without training.


## Compare a Seq2Event-inspired Transformer

After completing the LSTM experiment, run with the existing sequence extra:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.transformer_experiment
.\.venv\Scripts\python.exe -m notebook notebooks/10_transformer_comparison.ipynb
```

The Transformer combines action embeddings, projected continuous features and
positional encoding. Causal attention estimates shot or goal probability at each
observed event. Two separate models tune architecture, dropout, learning rate and
L2 on 2025, refit from scratch on 2023–2025, and evaluate on the shared 2026 benchmark.

Results and resumable checkpoints are under `artifacts/transformer-2026/`. Add
`--resume` after an interruption; completed runs are verified without retraining
or rescoring. CUDA is used when available; `--device cpu` selects CPU. See the
[Transformer specification and results](docs/transformer-baseline.md) for the
search, architecture, confusion matrices and limitations. This adapts Seq2Event's
mixed-input modelling idea to possession outcomes rather than reproducing its
next-event task. Notebook 10 explains each cell and includes executed outputs.


## Review the revised Transformer training

After the first Transformer experiment, run the broader validation search and fresh
refits with the existing environment:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.transformer_retraining
.\.venv\Scripts\python.exe -m notebook notebooks/11_transformer_retraining.ipynb
```

The revised procedure mixes possessions within optimiser batches, adds learning-rate
warmup/decay, screens 64 configurations and compares finalists across three seeds.
Fresh refits replay the validation-selected update count and schedule. All six final
models are frozen before evaluation on the shared 2026 benchmark. The earlier
experiment and notebook outputs remain available.

Results are under `artifacts/transformer-retrained-2026/`; use `--resume` after an
interruption. Notebook 11 explains each step and includes executed results. See the
[retraining specification and comparison](docs/transformer-retraining.md) for the
search space, calibration checks, metrics, confusion matrices and limitations.


## Train next-action and location prediction

Using the existing `[model,sequence]` extras and preprocessed seasons:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.next_event_experiment --audit-only
.\.venv\Scripts\python.exe -m afl_opposition.next_event_experiment
.\.venv\Scripts\python.exe -m afl_opposition.next_event_report --overwrite
.\.venv\Scripts\python.exe -m notebook notebooks/12_next_event_prediction.ipynb
```

The first command audits the development examples without training. The experiment
compares six configurations on 2025, continues two finalists, refits the selected
model and evaluates 2026. `--resume` continues an interrupted run with unchanged
sources and settings. Completed runs require a different `--output` directory.
The report command explicitly replaces the committed example report with the
new local run. Model weights and full predictions remain under the ignored
`artifacts/next-event-v1/` directory. See [the specification](docs/next-event-prediction.md)
for data conventions, reproducibility and the `forecast_next` inference API.
