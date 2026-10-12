# Season coverage and evaluation split

Checked on 11 October 2026 against the public torpdata [event archives](https://github.com/peteowen1/torpdata/releases/tag/chains-data) and [fixture catalogue](https://github.com/peteowen1/torpdata/releases/tag/fixtures-data). Event snapshots are pinned in [sources.json](../data/sources.json). Fixture URLs, SHA-256 hashes, match counts and exclusion details are recorded in [season-coverage.json](../data/season-coverage.json).

| Season | Role | Raw rows | Matches in archive / concluded fixtures | Date range | Quarters excluded for time reversals |
| --- | --- | ---: | ---: | --- | ---: |
| 2023 | Training | 442,482 | 216 / 216 | 16 March–30 September | 0 |
| 2024 | Training | 426,754 | 215 / 216 | 7 March–28 September | 1 |
| 2025 | Validation | 430,827 | 216 / 216 | 7 March–27 September | 2 |
| 2026 | Test | 447,945 | 218 / 218 | 5 March–26 September | 3 |

All fixture records have status `CONCLUDED`, and all archived event match IDs occur in the corresponding fixture catalogue. Matching fixture IDs establishes match presence, not complete or correct event annotation.

The 2024 archive is missing Western Bulldogs versus North Melbourne on 18 August (`CD_M20240142309`). The workflow uses the available 215 matches and records the gap. It does not invent observations for the missing match.

## Evaluation protocol

During model selection, the encoder, numeric scaler, classifier and constant shot-rate benchmark use 2023–2024 only. Validation uses 2025 to select the L2 penalty. Every match, possession and prefix stays within its assigned season. After selection, the final encoder, scaler and classifier are fitted afresh on 2023–2025 with the selected lambda fixed; the constant benchmark also uses this combined data.

The development baseline does not load 2026. The [final experiment](tuned-logistic.md) saves the selected/refitted model before loading 2026 for evaluation, then scores the test season with the existing exclusion policies and a fixed 0.5 threshold. No test outcome selects lambda, changes the threshold or fits a model component. The test results are now observed, so future choices informed by them would use the holdout as development evidence.

## Explicit exclusions

The preprocessing option `--quarantine-time-reversals` excludes an entire match-quarter if timestamps decrease in display order. It leaves every original annotation in the audit table and records policy `quarantine_quarter_v1`. This avoids silently sorting by a different key or treating repeated passages as new actions.

| Season | Match ID | Quarter | Excluded source rows per season |
| --- | --- | ---: | ---: |
| 2024 | `CD_M20240141606` | 3 | 589 |
| 2025 | `CD_M20250140608`, `CD_M20250141305` | 4, 4 | 1,027 |
| 2026 | `CD_M20260140001`, `CD_M20260140206`, `CD_M20260142005` | 1, 2, 3 | 1,616 |

Descriptions outside the original 2023 rule vocabulary remain unsupported and flag affected episodes for review. They are listed by season in the coverage JSON; no new action meanings are inferred from their names.

The multi-season label audit also uses policy `exclude_touched_possessions_v1`. A `Shot At Goal` marker without a preceding explicitly flagged kick by the same player and team in its raw chain excludes every possession touching that chain. The archived flag is not imputed. Conflicts were found at order 459 in `CD_M20240142308`, quarter 1, chain 60, and order 1824 in `CD_M20250140303`, quarter 4, chain 205. The experiment's possession audit identifies the excluded episodes as `shot_marker_conflict`; its source provenance records the marker locations.

These exclusions use retrospective archive-quality information and can bias the sample. Results describe eligible archived possessions, not every possession in a live match. The final experiment applies the same label audit to 2026; its [test report](tuned-logistic.md) records the resulting exclusions.

## Prepare an additional season

The README prepares the development seasons. The same commands can prepare the reserved test archive without scoring it:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.data --season 2026
.\.venv\Scripts\python.exe -m afl_opposition.preprocess --season 2026 --quarantine-time-reversals
```

Add `--overwrite` only when regenerating derived tables. Upstream files can change: a size or checksum mismatch requires reviewing and explicitly updating the pin, rather than accepting a different snapshot silently.
