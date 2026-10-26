# Optuna tuning for next-event prediction

Two separate studies tune the [team-stream and chronological formulations](event-formulations.md). The data contract and loss definitions are unchanged. Seq2Event uses exactly 40 observed team-stream tokens, unmasked attention and a last-position readout. The decoder uses every observed token within the quarter, causal attention and explicit acting-team and club inputs. Neither study sees 2026 during selection.

## Search space

| Parameter | Values or distribution |
| --- | --- |
| Model width | 64, 96, 128, 192, 256 |
| Layers | Integers 1–4 |
| Attention heads | 2, 4, 8 |
| Action embedding | 8, 16, 32 |
| Feed-forward multiplier | 2 or 4 times model width |
| Output hidden layer | 32, 64, 128 |
| Dropout | Uniform 0–0.4 |
| Learning rate | Log-uniform 0.00001–0.003 |
| L2 | Zero, or log-uniform 0.000001–0.01 |
| Club embedding, decoder only | 4, 8, 16 |

Width is divisible by every offered head count. For a 64-wide decoder with 16-dimensional club embeddings, the action embedding is restricted to 8 or 16 so the numerical projection retains positive width. Optuna uses a separate conditional parameter name for that branch. Seq2Event has no club embedding. All configurations use Adam, gradient clipping at 1, full precision and the existing deterministic runtime settings. Optimiser batch sizes are 512 fixed windows or eight complete quarters. Quarter histories are never shortened to fit GPU memory.

## Search and selection

Each study runs 60 trials using multivariate, grouped TPE with 15 initial random trials. A trial may run for 60 epochs. Early stopping has patience ten and is disabled before epoch 15. Median pruning also starts no earlier than epoch 15, requires ten completed trials and at least five comparison results at the relevant epoch. Pruned trials count towards the 60-trial budget. Unexpected failures stop the runner with an epoch checkpoint rather than silently becoming poor scores.

The selection objective is fixed within each study:

\[
J = 5\,\frac{1}{N}\sum_{i=1}^{N}w_{y_i}[-\log p(y_i)]
  + \sqrt{\frac{1}{2M}\sum_{i\in\mathcal S}\|\hat{\mathbf x}_i-\mathbf x_i\|^2}.
\]

Here, \(N\) is the number of eligible targets, \(\mathcal S\) contains the \(M\) targets with valid coordinate supervision, and \(w_{y_i}\) is the target class weight. Seq2Event uses inverse-frequency action weights calculated from training targets, normalised to sum to one, with boundary weight zero. The decoder uses unit weights. Coordinate masks follow each formulation's existing contract. Validation aggregates errors over the entire scored dataset. Training averages the batch objectives. Class weights and the coefficient five are not optimisation variables.

Candidates fit on 2023–2024 and select checkpoints using 2025. The three strongest completed trials are fitted again from scratch with seeds 2026, 2027 and 2028, without Optuna pruning. The configuration with the lowest mean best validation objective wins. Its refit duration is the median best epoch across those three seeds. Three fresh models then fit on 2023–2025 for that fixed duration, with newly fitted development-set scaling and class weights. Both studies and all six model exports finish before 2026 is loaded.

Test reporting includes every seed, mean and sample standard deviation, task-specific transition baselines, full confusion matrices and per-class precision/recall/F1. No seed is selected based on test performance. The standard deviation describes training randomness, not uncertainty from sampling different matches. The transition baseline retains the existing fixed additive smoothing of 0.5.

## Running the experiment

From the repository root, with the documented data and sequence environment already prepared:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model,sequence,tuning]"
.\.venv\Scripts\python.exe -m afl_opposition.event_optuna --preflight
.\.venv\Scripts\python.exe -m afl_opposition.event_optuna --smoke
.\.venv\Scripts\python.exe -m afl_opposition.event_optuna
```

The preflight exercises the largest architecture on the longest development histories. The smoke run uses two matches per development season, two one-epoch trials per formulation and one seed. It does not refit on all development data or load 2026. Smoke results are engineering checks, not model-performance evidence.

Full results are stored in `artifacts/event-optuna-v1/`, which is excluded from Git. The search runs sequentially on one GPU. This is substantially more computation than the earlier four-candidate experiment. Keep the computer running and awake while training. After an interruption:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.event_optuna --resume
```

Resume requires unchanged source, data, protocol and runtime records. The SQLite study stores trials, while a separate local sampler snapshot preserves TPE state. Each training checkpoint stores the model, optimiser, random-number states, best checkpoint and completed epochs. A process lock rejects concurrent writers. A rare interruption between a new database trial and its sampler snapshot is detected and requires restoring matching snapshots rather than silently changing the experiment. Do not load sampler pickle files from untrusted sources.

Export a completed full run and inspect the notebook:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.event_optuna_report
.\.venv\Scripts\python.exe -m notebook notebooks/14_optuna_event_models.ipynb
```

The exporter writes compact reports to `reports/examples/event-optuna-v1/` and refuses to overwrite existing reports. Notebook 14 uses these when available and can also inspect local training progress before the studies finish. Earlier reports and notebook outputs remain available.

## Interpretation

The two studies predict different targets and cannot establish an architecture ranking through raw accuracy or objective values. Seq2Event's weighted objective is not ordinary action accuracy or unweighted probability loss. Its fidelity-oriented weighting remains fixed; changes to that weighting would constitute a separate adaptation.

Sixty trials remain a bounded search. Review whether winning values lie near the search boundaries and whether learning curves have plateaued before claiming adequate optimisation. The 2026 data is an already-observed project benchmark, not a fresh holdout. These models predict events and locations, not validated tactical interventions.

Optuna references: [TPE sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html), [median pruning](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html), [persistent studies and sampler state](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html).
