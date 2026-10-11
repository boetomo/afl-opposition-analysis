# AFL Possession Threat Explorer

An AFL opposition-analysis project for football analysts and researchers, exploring how teams move the ball after gaining possession and which patterns lead to shots.

**Status: early development.** The application, models and report generation are not yet implemented.

## Project aim

> Which recurring ball-movement patterns lead to shots, and how early can a threatening possession be identified?

The project will combine summaries of an opponent's recent possessions with a model that estimates the probability of a subsequent shot from the events observed so far.

## Planned output

An interactive possession viewer and a downloadable opposition report containing movement patterns, shot rates, model estimates and supporting match examples.

## Data

The source is the public [torpdata match-event dataset](https://github.com/peteowen1/torpdata/releases/tag/chains-data). The first supported snapshot is the 2023 season; its source URL and checksum are recorded in [data/sources.json](data/sources.json).

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
