"""Export compact, reproducible reports for the two next-event formulations."""
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd
from afl_opposition.quarter_events import load_quarters, make_streams
from afl_opposition.event_models import load_model, forecast_next, configure
from afl_opposition.event_comparison import source_hashes
from afl_opposition.next_event_experiment import write_json


def export(root, experiment, output, overwrite=False):
    root, experiment, output = Path(root), Path(experiment), Path(output)
    summary = json.loads((experiment / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Only a completed full experiment can supply a results report.")
    if summary["plan"]["sources"] != source_hashes(root):
        raise ValueError("Current training sources differ from the completed experiment.")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError("Use --overwrite to replace an existing example report.")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "plan.json", "metrics.csv", "development_audit.json", "test_audit.json"):
        shutil.copyfile(experiment / name, output / name)
    configure("cpu", 2026)
    test = load_quarters(root, (2026,))
    if test.provenance != json.loads((experiment / "test_audit.json").read_text())["provenance"]:
        raise ValueError("Test provenance differs from the frozen evaluation.")
    first_quarter = test.events.quarter_id.iloc[0]
    quarter = test.events.loc[test.events.quarter_id.eq(first_quarter)].reset_index(drop=True)
    prefix = quarter.iloc[:200].copy()
    prefix.to_csv(output / "forecast_prefix.csv", index=False)
    actual = quarter.iloc[200]
    forecasts = dict(selection="First recorded test quarter, after its first 200 tokens; not selected by model performance.",
        actual_next_match_event=dict(action=str(actual.action_type), side=int(actual.side),
            x_home=float(actual.x_home), y_home=float(actual.y_home), has_coordinates=bool(actual.has_coordinates)))
    for mode in ("team_stream", "joint"):
        src, dest = experiment / mode, output / mode
        dest.mkdir(exist_ok=True)
        for pattern in ("coverage.json", "example_stream.csv", "selection.json", "*-history.json", "*-confusion.csv", "*-per-class.csv"):
            for path in src.glob(pattern):
                shutil.copyfile(path, dest / path.name)
        model, encoder, info = load_model(src / "model")
        forecasts[mode] = forecast_next(model, encoder, prefix, focal_side=0)
        if mode == "team_stream":
            stream = make_streams(prefix, mode)
            last = stream.loc[stream.focal_side.eq(0)].iloc[-1]
            forecasts[mode]["location_supported_by_training_mask"] = bool(last.has_coordinates)
        write_json(dest / "model_metadata.json", info)
    write_json(output / "forecast_example.json", forecasts)
    (output / "README.md").write_text(
        "# Next-event formulation comparison\n\n"
        "These files contain the completed validation search, refit metadata, test metrics and a predetermined example forecast. "
        "The team-stream and chronological models solve different tasks. Their raw losses and accuracies do not establish a cross-task winner.\n\n"
        "No raw match archive or model weights are included. Notebook 13 reads these reports without training. "
        "See docs/event-formulations.md for reproduction and interpretation.\n", encoding="utf-8")
    print("Exported:", output.relative_to(root).as_posix() if output.is_relative_to(root) else output.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="artifacts/event-comparison-v1")
    parser.add_argument("--output", default="reports/examples/event-comparison-v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export(Path.cwd(), args.experiment, args.output, args.overwrite)


if __name__ == "__main__":
    main()
