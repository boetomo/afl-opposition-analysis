# Inspecting the 2023 event data

The 2023 torpdata snapshot contains match-event annotations, coordinates and repeated match metadata. The season falls within the proposed 2021 to 2023 training period. Raw rows and chains need interpretation before they can become possession sequences for modelling.

## Snapshot

The file comes from the [torpdata chains release](https://github.com/peteowen1/torpdata/releases/tag/chains-data). Its URL, release asset ID, size and SHA-256 checksum are recorded in [the source manifest](../data/sources.json). The counts below describe that exact file.

| Measure | Observed value |
| --- | ---: |
| Dates | 16 March to 30 September 2023 |
| Rows | 442,482 |
| Columns | 63 |
| Matches | 216 |
| Raw chains, grouped by match, quarter and chain number | 54,418 |
| Distinct action descriptions | 77 |
| Explicit shot flags | 10,888 |
| Rows with missing acting team and player IDs | 27,844 |
| Extra rows sharing a raw event key | 0 |

The inspection uses `(match_id, period, chain_number, display_order)` as the raw event key. All four fields are populated in this snapshot. Unique keys do not guarantee that rows represent separate physical actions.

## Core fields

| Fields | Meaning and use |
| --- | --- |
| `match_id`, `period`, `display_order` | Identify the match, quarter and event order. Order within each match-quarter by `display_order`; timestamps can tie. |
| `period_seconds` | Time within the quarter, rather than elapsed time across the entire match. |
| `description` | A categorical event annotation, such as `Kick`, `Handball` or `Uncontested Mark`. |
| `team_id`, `player_id` | Acting team and player where supplied. Some stoppage annotations have neither. |
| `chain_number`, `chain_team_id` | Upstream chain grouping and its assigned team. These do not establish an uninterrupted possession by that team. |
| `x`, `y` | Raw event coordinates, stored as integers here. Ranges are -85 to 86 and -71 to 71. Coordinate orientation needs checking before interpreting forward movement. |
| `shot_at_goal` | Explicit shot annotation on the current row: 10,888 `True` values and 431,594 missing values; no explicit `False` values. |
| `initial_state`, `final_state` | Upstream chain start and end categories. `final_state` describes a future outcome for earlier rows and must not be a prefix feature. |
| `disposal` | Disposal outcome classification. Its availability at prediction time needs checking before feature use. |
| Match dates, teams, venue and scores | Metadata repeated across event rows. Final match scores must not enter prefix features. |

The generated inspection lists every column, its loaded dtype and missing-value count. A dtype on an entirely empty column, such as `venue_local_start_time`, does not establish that field's intended meaning.

## A passage of play

The inspection prints the first 25 rows from Richmond versus Carlton on 16 March 2023, quarter one (`CD_M20230140101`). For example:

| Order | Quarter seconds | Description | Acting team | Chain team |
| ---: | ---: | --- | --- | --- |
| 6 | 26 | Kick | CD_T120 | CD_T120 |
| 7 | 26 | Kick Into F50 | CD_T120 | CD_T120 |
| 8 | 27 | Kick Inside 50 Result | missing | CD_T120 |
| 9 | 29 | Mark Fumbled | CD_T30 | CD_T120 |
| 10 | 29 | Mark Dropped | CD_T30 | CD_T120 |
| 11 | 32 | Loose Ball Get Crumb | CD_T120 | CD_T120 |
| 12 | 32 | Kick, shot flag True | CD_T120 | CD_T120 |
| 13 | 35 | Goal | CD_T120 | CD_T120 |

The kick and forward-50 entry annotations share a time and location. They need to be reconciled before counting ball movements. Opposing-team actions can also occur inside the same raw chain.

Across the season, 48,231 rows have different acting and chain teams, affecting 33,329 raw chains. This includes 15 explicitly flagged shots. Assigning every row or shot to `chain_team_id` would therefore be incorrect.

There are 103,989 adjacent rows sharing a timestamp within a match-quarter, and no time decreases when sorted by display order. Equal timestamps alone are not a reason to discard a row.

## Requirements for modelling

Possession construction requires rules for controlled possession, overlapping annotations and coordinate orientation, supported by audits of actual passages. Unresolved cases require separate review. The proposed target is whether the same team takes a subsequent shot before its possession ends, given an observed pre-shot prefix. This differs from the current row's `shot_at_goal` flag.

Reproduce this inspection with the commands in the [README](../README.md#load-and-inspect-data). The downloaded file remains unchanged.
