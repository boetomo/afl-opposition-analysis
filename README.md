# AFL Possession Threat Explorer

An AFL opposition-analysis project for football analysts and researchers, exploring how teams move the ball after gaining possession and which patterns lead to shots.

**Status: early development.** The application, models and report generation are not yet implemented.

## Project aim

> Which recurring ball-movement patterns lead to shots, and how early can we recognise a threatening possession?

The project will combine summaries of an opponent's recent possessions with a model that estimates the probability of a subsequent shot from the events observed so far.

## Planned output

An interactive possession viewer and a downloadable opposition report containing movement patterns, shot rates, model estimates and supporting match examples.

## Data

The planned source is the public [torpdata match-event dataset](https://github.com/peteowen1/torpdata/releases/tag/chains-data).

## Development setup

Requires Python 3.14 or later. The setup below is verified on Windows with Python 3.14.7. Run these commands from the repository root in PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

The virtual environment isolates the project's installed packages. The editable installation imports `afl_opposition` directly from `src/`, so source edits are available without reinstalling the package. Runtime dependencies will be added as features are implemented.

Verify the installation:

```powershell
.\.venv\Scripts\python.exe -c "import afl_opposition; print(afl_opposition.__file__)"
```

The printed path should point to `src\afl_opposition\__init__.py` in this checkout. The `.venv` directory is excluded from Git.
