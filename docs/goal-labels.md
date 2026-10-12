# Goal-prediction labels

## Target and input boundary

Given a pre-shot prefix, the binary target is whether the **same team scores a recorded goal before its inferred possession ends**. It is not a model of shot conversion conditional on observing the shot.

The input rows are the same as the existing subsequent-shot dataset, subject to additional goal-attribution checks:

- For a possession with an explicit shot, include retained actions strictly before the first shot.
- For a completed possession with no shot, include all retained actions.
- A goal at any later point in the same episode makes these prefixes positive, including a goal after a recovered miss. The first shot, recovery and later shot remain outside the inputs.
- A supported episode ending without a same-team goal is negative. A behind does not satisfy the goal target.
- Ambiguous, incomplete or contradictory episodes have missing targets and no example rows; they are not negatives.

Each eligible possession contributes total sample weight one, divided equally across its prefixes. Features remain `action_type`, `x_norm`, `y_norm`, `gap_seconds` and `elapsed_seconds`. End reasons, score/goal annotations, shot flags, shot counts and audit flags are never features. For a future sequence model, each input sequence must end at its prediction row.

## Attribution audit

`build_goal_examples` in `src/afl_opposition/goal_labels.py` first applies the existing shot eligibility policy, then audits the full goal-annotation stream. The input tables and original shot labels are not mutated.

Each goal is linked to the preceding retained action by strict `display_order`, within the same match and quarter. This supplies a candidate possession, not automatic proof of attribution. A verified goal requires:

1. Exactly one goal annotation linked to that episode.
2. A `Goal` endpoint in the possession table.
3. The goal's recorded time equal to the possession's closing time.
4. A known goal team equal to the possession team.

Missing/duplicate goals, mismatched teams or times, and conflicting endpoints trigger `goal_attribution_review` for an otherwise eligible possession. A goal without a preceding action is retained as `no_preceding_action` for review; no possession is invented. Quarantined quarters remain visible in the annotation audit but cannot supply actions or examples.

Goal records need not identify the same player as the preceding action: this is a team outcome target, not individual scoring attribution. Existing score-without-shot, shot-marker, first-action-shot and preprocessing exclusions still apply. `shot_status` and `shot_target` retain the earlier decision alongside the final goal eligibility and `target`.

Goal label version: `0.1.0`. Underlying shot label version: `0.1.0`.

## Generate and inspect

Prepare the 2023–2025 snapshots and preprocessed tables using the README. The command reuses the verified loader from the existing modelling stage, so install the `model` extra as well as `notebook` for the walkthrough:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model,notebook]"
.\.venv\Scripts\python.exe -m afl_opposition.goal_labels
.\.venv\Scripts\python.exe -m notebook notebooks/06_goal_labels.ipynb
```

Run from the repository root. Repeated label generation requires `--overwrite`. The notebook reads saved artifacts and includes executed outputs; rerunning it does not regenerate labels or train a model.

The generated directory is `artifacts/goal-labels-2023-2025/`, excluded from Git:

| File | Contents |
|---|---|
| `examples.parquet` | Pre-shot inputs, goal target and sample weights |
| `possession_audit.parquet` | Shot/goal labels, eligibility, cutoff and review reasons per possession |
| `goal_annotation_audit.parquet` | Every raw goal, candidate possession and attribution status |
| `coverage.parquet` | Counts, exclusions and goal prevalence by season |
| `summary.json` | Target contract, versions, source/code hashes and artifact checksums |

The completion record is written after all tables. The notebook verifies artifact and code hashes before displaying results and checks source-table hashes before showing raw passages.

## Development snapshot results

Only **2023–2025** are loaded by this stage. No goal model is trained, and no 2026 goal labels or model scores are generated.

| Season | Matches | Eligible possessions | Goal | No goal | Prefix examples | Goal prevalence |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | 216 | 52,429 | 5,018 | 47,411 | 285,771 | 9.57% |
| 2024 | 215 | 50,649 | 5,046 | 45,603 | 277,528 | 9.96% |
| 2025 | 216 | 50,744 | 5,096 | 45,648 | 281,476 | 10.04% |
| Total | 647 | 153,822 | 15,160 | 138,662 | 844,775 | 9.86% |

All **15,766** goal annotations outside quarantined quarters have a unique matching team, endpoint and closing time. Another **19** goals belong to quarantined quarters (8 in 2024 and 11 in 2025). The earlier eligibility rules exclude 606 of the verified goal possessions, leaving 15,160 positives. Goal attribution adds **zero** exclusions to the existing development examples.

Across all outcomes, 10,422 inferred possessions are excluded: 10,312 preprocessing reviews, 102 first-action shots with no pre-shot input, five scoring outcomes without an explicit shot flag, and three shot-marker conflicts. Quarantined quarters are removed before possession construction, so they are reported separately rather than added to these possession counts.

There are **249** eligible goal possessions with multiple explicit shots: 87 in 2023, 71 in 2024 and 91 in 2025. The notebook includes a complete example showing why the first shot need not terminate the target horizon.

## Review and limitations

The notebook explains provenance, target/cutoff separation, season counts and exclusions, then displays four raw passages: a goal, a behind, a turnover without a shot, and a recovered miss followed by a goal. It marks input rows explicitly and checks weights and cutoff consistency.

The tests cover team/quarter/match isolation, tied timestamps, missing or duplicate goal annotations, censoring, quarantine, recovered misses, unchanged input features when the outcome changes, and preservation of the earlier shot labels.

Attribution verifies internal consistency in recorded events, not independent video ground truth. Possessions remain project-specific approximations. Scores after a stoppage or opponent control fall outside the episode. The source is missing one 2024 fixture, and inherited exclusions can affect representativeness; see [season coverage](season-coverage.md).

The next modelling stage can establish a goal logistic-regression baseline before trying nonlinear and sequence models. Compare goal metrics against goal prevalence, not the shot target's AP. Model selection remains on development data; the previously observed 2026 shot benchmark is not a newly untouched holdout.
