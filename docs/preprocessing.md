# Preprocessing AFL event annotations

The preprocessing pipeline converts a verified raw snapshot into ordered ball-control actions, inferred possession episodes and an annotation audit. Rule version `0.1.0` supports inspection and subsequent training-data construction. Its episodes are an operational approximation of possession, not provider-certified ground truth.

## Run the pipeline

From the repository root, after downloading the season:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.preprocess --season 2023
```

Add `--overwrite` to regenerate existing derived files. The command verifies the raw snapshot against `data/sources.json` before processing. Outputs are saved under `data/processed/2023/`, which is excluded from Git. The raw snapshot is never rewritten.

| Output | Contents |
| --- | --- |
| `events.parquet` | One row per accepted ball-control or movement observation, with event order, possession ID, attacking coordinates and elapsed times |
| `possessions.parquet` | Episode boundaries, team, action count, explicit shot count and review reasons |
| `annotations.parquet` | Every original row, its source identifier, handling decision and link to a retained observation where applicable |
| `action_audit.parquet` | Counts by description, rule role and handling decision |
| `summary.json` | Counts, rule version, feature allowlist and source filename and SHA-256 checksum |

The second notebook, `notebooks/02_preprocess_events.ipynb`, demonstrates each transformation and includes saved outputs.

## Quarters with reversed timestamps

By default, a timestamp decrease within a match-quarter stops preprocessing. The multi-season workflow uses `--quarantine-time-reversals`: if any timestamp decreases in display order, the entire affected quarter is excluded from the action and possession tables. Every source row remains in `annotations.parquet`, marked `quarantined=True`, with handling `quarantined_quarter` and reason `quarter_time_reversal`. The summary records the affected quarters, excluded rows and excluded explicit shots under policy `quarantine_quarter_v1`.

This conservative archive-quality rule avoids repairing timestamps or reordering potentially repeated passages without evidence. It is retrospective sample selection, not an online feature. Clean quarters retain the original action rules. Other structural errors still stop processing. The [coverage audit](season-coverage.md) lists the affected quarters and season limitations.

## Action selection and duplicates

`src/afl_opposition/action_rules.py` declares the supported descriptions. Kicks, handballs, receptions, gathers, ball gets, recognised marks, free-kick awards, running bounces and kick-in actions enter the event stream. The original description is retained alongside a smaller `action_type` vocabulary.

Spoils, fumbles, tackles and knock-ons remain in the annotation audit. They do not by themselves establish controlled possession. Contest targets, kick-result annotations and statistical credits also stay outside the input stream. A late `Shot At Goal` annotation is not copied backwards onto an earlier kick.

Two specific overlap rules are implemented:

1. An adjacent cross-chain repeat is collapsed only when match, quarter, description, timestamp, player and acting team match, and coordinates agree after accounting for the two chain frames. Both actor identities must be known. The first occurrence is retained, and the later copy points back to it.
2. `Kick Into F50` links to an immediately preceding `Kick` or `Ground Kick` only when match, quarter, chain, timestamp, player, team and coordinates match exactly. The extra annotation does not become a second kick or a retrospective input feature. Unmatched cases remain visible and flag affected episodes for review.

Equal timestamps alone never establish a duplicate. These rules do not claim to reconcile every physical action in the source. In particular, the first version deliberately limits its input stream to supported ball-control observations.

## Possession episodes

An episode starts at an accepted action by a known team. It can start at a disposal if no preceding acquisition was recorded; its start time is the first observed action, not a claim about the exact instant control began.

An accepted action by the opposing team closes the previous episode and starts another. An opponent's fumble or spoil does not. Explicit scores, ball-up calls and boundary exits close the current episode. At a new raw chain, a restart category such as `kickIn` or `throwIn` can establish a stoppage even when a separate stoppage row is absent.

A chain-team change without a supported control action produces `unresolved_chain_change` and a review flag. Match and quarter boundaries are never crossed. The end of the available input is recorded as `end_of_available_events`, which is censored and cannot automatically become a negative target. `last_event_seconds` is the last accepted action; `closed_at_seconds` is when the endpoint was observed, which may be later.

The shot flag alone does not end an episode: a missed shot may be recovered by the same team. Shot flags are retained as audit/target-construction metadata, not input features.

## Coordinates and time

The upstream [coordinate implementation](https://github.com/peteowen1/torp/blob/main/R/clean_pbp.R) identifies raw coordinates as relative to the chain team's attacking direction. This pipeline preserves raw coordinates and reverses both signs when the acting team differs from that coordinate frame. Each accepted event is then expressed relative to its acting team's attacking direction. This is a relative frame, not a fixed compass orientation of the stadium.

For example, Richmond's possession gain at order 34 in the opening 2023 Richmond-Carlton match has raw coordinates `(39, -8)` in Carlton's chain frame. The processed coordinates are `(-39, 8)`, matching the repeated observation in Richmond's next chain. Order 34 is retained and order 35 is linked to it.

Coordinates are divided by the corresponding venue half-dimensions. No statistical scaler is fitted. Values are not clipped, and no future-neighbour interpolation or sign correction is performed. Missing coordinate information, values beyond 110% of the supplied half-dimensions and within-episode jumps above 100 metres flag an episode for review; these thresholds are diagnostics, not proof of an error.

`gap_seconds` measures time since the preceding retained action in the same episode. `elapsed_seconds` measures time since its first retained action. The first action has zero for both. Removing auxiliary annotations therefore changes the gap relative to the raw-row gap shown in the exploration notebook.

## Feature and review contract

The initial feature allowlist is `action_type`, `x_norm`, `y_norm`, `gap_seconds` and `elapsed_seconds`. `event_index` supplies sequence order. Original descriptions remain available for later vocabulary decisions. Vocabularies, imputation and any learned normalisation must be fitted using training data only.

Final chain outcomes, final scores, shot flags, episode shot counts, end reasons and review flags must not enter model inputs. The pipeline does not use final scores or `final_state`. It does not yet generate labels, chronological splits or model tensors. Future target construction must exclude the current shot from pre-shot inputs and handle censoring explicitly.

Review flags include unsupported descriptions, missing action teams, unmatched kick annotations, unresolved chain changes, coordinate anomalies and truncated episodes. A raw-chain issue marks all episodes touching that chain. This is retrospective audit metadata: selecting only unflagged episodes can bias the sample, so exclusion rates must be reported by team and season before evaluation. Unflagged episodes are not a guarantee of annotation completeness.

The source consists of archived annotations. Its real-time publication latency and the availability of each field during a live match have not been established.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover possession switches, repeated boundary actions, tied timestamps, stoppages, coordinate frames, unsupported descriptions, raw-data preservation and invariance of existing features when future events are appended. The notebook adds checks against the recorded 2023 snapshot. General football terminology follows the [AFL statistics glossary](https://www.afl.com.au/news/144837/stats-glossary-every-stat-explained); the processing conventions above are project-specific.
