# Opposition review app

A browser application for exploring archived AFL events, inspecting frozen decoder forecasts and exporting an opposition briefing. It supports every club represented in the archive. No training runs when the app starts or while a report is generated.

## Start locally

From the repository root, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[app]"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. Stop the server with Ctrl+C. The app extra contains Streamlit and Plotly; PyTorch is not required to browse prepared data or download a report.

The committed **sample archive covers all 18 clubs**. It contains the earliest available quarter in each club's earliest 2026 match, chosen independently of model performance. These are partial matches, not a representative sample of recent form. The application labels this coverage explicitly.

If `artifacts/opposition-app-v1/` is available, the full local archive is the default. The prepared 2026 archive contains 218 matches and 869 non-quarantined quarters. The match table shows available quarters rather than assuming every match has four. Use the Data selector to switch between the local archive and sample.

## Workflow

1. Select the team to analyse, a briefing date and the number of recent matches. Only games with a recorded date **strictly before** that date are included.
2. Choose a match and quarter. Step through the events or press Play. The field diagram displays the last twelve observations. The decoder forecast uses the complete observed quarter, and the selected team always attacks right.
3. Inspect next team/action probabilities and the predicted location. Reveal the actual next event when needed. Unknown passages and archive quality flags are visible; unreliable current observations have their forecast withheld.
4. Open Opposition briefing to compare observed next-event frequencies with the model's average estimates in the same situations.
5. Generate and download the HTML briefing. It contains coverage, tendency tables and up to six example passages with forecasts and actual outcomes. The HTML includes its plotting library and opens offline. A browser's Print to PDF command provides a static copy.

The replay represents recorded event locations, not video or continuous player/ball tracking. Lines connect observations and do not imply the actual flight path. The predicted point is shared across possible ordinary actions, not an action-specific destination. A neutral boundary has no physical location. Model probabilities have not been calibrated for each club or situation.

## Prepare a full local archive

This step requires the documented model/sequence environment, processed 2026 data and the existing selected decoder export at `artifacts/event-optuna-v1/joint/model-seed-2026/`:

```powershell
.\.venv\Scripts\python.exe -m afl_opposition.prepare_opposition
```

Preparation uses CPU inference only and leaves all training checkpoints untouched. It uses the predeclared first refit seed, 2026, rather than selecting a seed from test results. It verifies model checksums, rejects seasons overlapping the refit period, checks known clubs, and compares full-quarter causal inference with truncated-prefix forecasts. It never trains or resumes Seq2Event.

The output contains observed tokens, cached next-event probabilities/locations, match metadata and a manifest with model/data provenance and file checksums. A single pass yields forecasts for every position because causal attention blocks later tokens. Targets and quality masks support retrospective reporting but are not numerical model inputs. Already prepared outputs are immutable; use `--output` to select a new directory.

To reproduce the sample bundle as well, specify a fresh `--demo-output` directory. The committed bundle is not overwritten automatically. A custom server-side bundle can be selected with the `AFL_APP_BUNDLE` environment variable. The app does not accept arbitrary uploaded model files.

## What the briefing measures

The summaries use eligible transitions **after the selected team's own actions**, grouped by current action and field third. Each eligible event has equal weight. The pitch is normalised to the selected team's attacking direction before assigning thirds. Observed and predicted kick/handball percentages refer to that same team's next recorded action. Other-team probability refers to the next recorded action being performed by the opposition; it is not a complete measure of turnover risk. Boundaries and other action types account for the remaining probability.

The passage list takes one earliest eligible transition per selected match before cycling to additional examples. It is not curated to show correct predictions. Reports state their exact date and match coverage, model fingerprint and sample/full-archive status.

These summaries describe the situations encountered in the selected games. They do not control for opposition strength, score or changing match circumstances, establish a team's causal style, or prescribe defensive tactics. The archive includes retrospective quality decisions and may have been revised after the match. The 2026 benchmark has already been observed. This version is a historical review tool, not a live match feed.

## Deployment and tests

`app.py` is the app entrypoint. `requirements-app.txt` installs only the app environment. A Python 3.14 host can serve the committed sample without raw archives or weights. Copy a prepared full bundle to the host and set `AFL_APP_BUNDLE` to serve that archive. Run the same Streamlit command with a host-appropriate bind address and port. Public hosting, authentication and ingestion of live match data are separate deployment steps.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_opposition_app.py -v
```

Checks cover causal prefix equivalence, team/date filtering, home/away orientation, quality exclusions, optional future reveal, report coverage and HTML escaping, file integrity and the app's team/replay/download workflow.
