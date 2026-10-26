# Predicting the next AFL action and location

This experiment asks: **given the actions observed so far in a team's possession, what is the next recorded ball-control action, and where will it occur?**

The model is shared across teams. It predicts a distribution over actions and one location conditional on the possession continuing. An `end_possession` class allows the model to predict that there will be no further retained action by that team in this episode. This is a first component for opposition analysis, not yet a team-conditioned tactical report.

The implementation is an AFL adaptation of [Seq2Event (KDD 2022)](https://doi.org/10.1145/3534678.3539138), informed by the [authors' modelling notebook](https://github.com/statsonthecloud/Soccer-SEQ2Event/blob/main/Seq2Event_Notebook02_Modelling.ipynb). It is separate from the repository's earlier Transformer classifiers for eventual shots and goals.

## Reproduce the experiment

Install the existing notebook, model and sequence extras, download the pinned seasons and run preprocessing as described in the main README. No new dependencies are required.

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.next_event_experiment --audit-only
.\.venv\Scripts\python.exe -m afl_opposition.next_event_experiment
```

The first command constructs and audits development examples without fitting a model or loading 2026. The second runs validation selection, refitting and test evaluation. Outputs go to `artifacts/next-event-v1/`. Add `--resume` to continue interrupted training with unchanged data, source code and settings. Epoch checkpoints contain model, optimiser and random states. A completed experiment cannot be overwritten. Use `--output artifacts/next-event-another-run` for a new run.

The [walkthrough notebook](../notebooks/12_next_event_prediction.ipynb) presents example prefixes, the architecture, model selection and evaluation. Saved example reports under `reports/examples/next-event-v1/` allow readers to inspect the results without training.

After training, export the compact report and example forecast:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.next_event_report
```

Add `--overwrite` when deliberately replacing an existing example report. The exporter checks that its source observations and model match the completed experiment.

## What constitutes an event?

The retained stream comes from the existing [preprocessing rules](preprocessing.md). It contains kicks, ground kicks, handballs, receptions, marks, gathers, ball gets, free-kick awards, bounces and kick-in actions. Result annotations, duplicate boundary records, tackles and spoils do not become separate targets.

The twelve retained action types plus `end_possession` form a fixed, documented vocabulary. These classes are declared from the action rules rather than inferred from future seasons. `kick` includes kicks flagged as shots in the archive. This experiment does not use the retrospective shot flag to create a separate shot action.

For an observed episode:

```text
Gather -> Handball -> Receive -> Kick -> recorded Goal
```

the supervised examples are:

| Observed history | Action target | Location target |
| --- | --- | --- |
| Gather | handball | Handball's recorded coordinates |
| Gather, Handball | receive | Reception's recorded coordinates |
| Gather, Handball, Receive | kick | Kick's recorded coordinates |
| Gather, Handball, Receive, Kick | end_possession | None |

The goal closes the episode. It is not treated as a ball-control location target. `end_possession` also covers an opponent gaining control, a stoppage or a recorded restart. It does not distinguish successful scoring from losing the ball. The existing shot/goal models answer those different outcome questions.

At the end of a truncated recording, the final prefix has no known target and is not scored. Earlier observed transitions remain usable. Reviewed episodes are excluded, except when censoring is their only issue. This is retrospective archive-quality selection and can bias the evaluation towards cleaner passages. Coverage is saved by season, team and review reason. The live availability and revision latency of the archived fields have not been verified.

## Inputs and growing history

Each observation supplies its action category and ten numerical values:

| Feature | Meaning |
| --- | --- |
| `x_norm`, `y_norm` | Location in the acting team's attacking frame, divided by the venue half-dimensions |
| `gap_seconds` | Time since the preceding retained action in the episode |
| `elapsed_seconds` | Time since the first observed action in the episode |
| `period_seconds`, `period` | Elapsed quarter time and quarter number |
| `dx_norm`, `dy_norm` | Change in normalised coordinates since the preceding action; zero for the first action |
| `distance_to_goal_norm` | Euclidean distance to `(1, 0)` in the normalised frame |
| `angle_to_goal` | Signed bearing relative to `(1, 0)` in that same normalised frame |

The last two are dimensionless geometric features, not a distance in metres or an exact physical shooting angle. No final score, final chain outcome, shot flag, future action, episode length or endpoint reason is an input. A running score is omitted because the archived scoring annotations require further attribution checks before reconstructing it reliably.

The history grows from one event to `min(observed events, max_history)`. Longer histories use the most recent events. Features are computed before truncation, so the first retained token can retain its observed time and coordinate differences. Position indices restart at zero within each model window. Histories reset at possession boundaries and never cross a match or quarter. All history and target coordinates therefore use the same team's attacking direction.

The input action embedding uses zero only for padding. Numerical features are standardised using the fitting partition. The numerical scaler is refitted on 2023–2025 for the final model. Prefixes are built separately, right padded and supplied with a padding mask. The target event is never included.

## Architecture and loss

For each historical event, concatenate the action embedding and a linear projection of the ten numerical values. Add sinusoidal positional encoding, pass the sequence through a post-normalisation Transformer encoder with ReLU feed-forward layers, and read the last valid position. A linear–ReLU–linear head produces thirteen action logits and two coordinate values.

The encoder attends to all observations in the supplied history. This is valid because the entire input precedes its single target. It would not be valid to run an unmasked full possession and score every earlier position against later targets.

The training objective is:

```text
5 × multiclass cross-entropy + coordinate RMSE
```

Cross-entropy includes `end_possession`. Coordinate RMSE averages squared error over both coordinates of continuation targets only, followed by the square root. An all-terminal batch contributes no coordinate loss. Adam applies L2 weight decay and gradients are clipped to norm 1.

Class weights are uniform, preserving natural-frequency probabilities. The original soccer implementation uses soccer-specific class weights that are not transferable to these AFL actions. The action/location coefficient remains five. Validation selection uses the same coefficient and computes RMSE over the entire validation set; the average of training batch objectives is a separate diagnostic.

The coordinate output is one unconstrained point estimate, conditional on continuation. It is neither a location for each possible action nor a calibrated spatial distribution. Multi-step rollout, next-event timing, player identity, team embeddings and opponent-specific adaptation are outside this version.

### Deliberate differences from the original implementation

- AFL actions and possession boundaries replace the soccer event vocabulary and team streams.
- Short and growing histories replace the original fixed-length windows that skip initial short prefixes.
- `end_possession` receives a training loss rather than being ignored.
- The input features reflect the audited AFL data. No unverified score advantage is supplied.
- Standard paired sine/cosine encoding with base 10,000 is used. The authors' notebook has a different channel-index/base implementation.
- Encoder layers are independently initialised.
- Chronological season splits replace the paper's soccer split.

These adaptations make this a Seq2Event-style next-event model, not an exact reproduction of the paper's results.

## Selection and evaluation

1. Fit on **2023–2024**, using **2025** for selection.
2. Run six prespecified configurations for up to four epochs each.
3. Continue the best two up to twelve epochs, with patience of three epochs.
4. Choose the configuration and epoch count with lowest validation joint loss.
5. Refit preprocessing and model weights from scratch on **2023–2025** for the selected epoch count.
6. Export and reload the frozen model, then evaluate **2026**.

The configurations compare history caps of 16, 32 and 64, combined embedding widths of 32 and 64, one or two encoder layers, and dropout of 0.1 or 0.25. Attention has four heads. The learning rate is 0.001 and Adam L2 weight decay is 0.0001. This is a bounded first search with one random seed, not exhaustive hyperparameter optimisation.

Every labelled event prefix has equal weight. This differs from the earlier possession-weighted shot/goal experiments. Metrics between those tasks are not directly comparable.

Two baselines are fitted on the same allowed fitting partition:

- **Frequency + persistence:** global next-action frequencies, with the next location set to the current location.
- **Action transition:** smoothed next-action frequencies conditional on the current action, plus the mean location displacement following that action.

These baselines test improvement over simple event frequencies and transitions. They do not isolate the benefit of sequence history: the Transformer also receives richer numerical inputs and has greater capacity. A matched current-state neural model would be needed for that ablation.

Macro F1 averages over all thirteen declared output classes, including classes with zero target support. In the 2026 benchmark, `kick_in` and `kick_in_play_on` occur as episode-opening inputs rather than next-action targets. Their F1 contributions are zero. Per-action support is therefore essential when interpreting this aggregate.

The report includes action log loss, accuracy, top-three accuracy, macro F1, multiclass Brier score, per-action precision/recall/F1, the full confusion matrix and location errors in normalised coordinates. Location errors are measured only where a continuation actually exists, regardless of the predicted action. Reporting error only for correctly classified actions would give a misleadingly easy subset.

2026 has already been examined in earlier project experiments. It is an **already-observed benchmark**, not an untouched confirmatory test. It is not used for fitting or selecting this new model. Team-level test metrics are diagnostics only.

## Forecast from an observed prefix

```python
from pathlib import Path
from afl_opposition.next_event_model import load_model, forecast_next

model, scaler, metadata = load_model(Path("artifacts/next-event-v1/model"))
# observed_prefix: processed event rows for ONE ongoing episode, from event_index=0.
forecast = forecast_next(model, scaler, observed_prefix)
print(forecast["probabilities"])
print(forecast["location_if_continuation"])
```

No target labels are needed for this API. Supply only observations available at the prediction time. Team IDs select the episode and establish its coordinate frame but are not embedded as learned identity features.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The new tests cover target alignment, censored endpoints, turnovers, quarter boundaries, exclusion of ambiguous episodes, growing and capped histories, future-data invariance, padding invariance, masked location gradients, train-only fitting, tiny-data learning, deterministic checkpoint resumption, and exported-model reload/checksum checks.
