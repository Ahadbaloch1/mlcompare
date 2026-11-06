"""Command-line interface: run and predict."""

import argparse
import os
import sys

import pandas as pd

from . import __version__
from .data import TASKS, load_csv
from .pipeline import predict, run
from .report import load_model, save_model, save_plots, write_metrics_json, write_report


def _fmt(value):
    return "%.3f" % value


def build_parser():
    p = argparse.ArgumentParser(prog="mlcompare", description="Compare and explain ML models on a CSV file")
    p.add_argument("--version", action="version", version="mlcompare " + __version__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Compare models, tune the best one, evaluate and explain")
    r.add_argument("--data", required=True, help="CSV file")
    r.add_argument("--target", required=True, help="Name of the column to predict")
    r.add_argument("--task", choices=TASKS, help="Override automatic task detection")
    r.add_argument("--out", default="outputs", help="Output folder")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--no-tune", action="store_true", help="Skip hyper-parameter tuning")

    q = sub.add_parser("predict", help="Predict with a saved model")
    q.add_argument("--model", default="outputs/model.joblib")
    q.add_argument("--data", required=True)
    q.add_argument("--out", default="predictions.csv")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            X, y, info = load_csv(args.data, args.target)
            if info["rows_dropped_missing_target"]:
                print("Dropped %d rows with a missing target." % info["rows_dropped_missing_target"])
            results = run(X, y, task=args.task, seed=args.seed, tune=not args.no_tune, folds=args.folds)
            os.makedirs(args.out, exist_ok=True)
            print("Task: %s | train rows: %d | test rows: %d\n"
                  % (results["task"], results["n_train"], results["n_test"]))
            print("Cross-validation comparison:")
            shown = results["comparison"].drop(columns=[c for c in results["comparison"].columns
                                                        if c.endswith("_std")])
            print(shown.to_string(index=False, float_format=_fmt))
            print("\nBest model: %s %s" % (results["best_model"], results["best_params"] or ""))
            print("\nTest-set results:")
            for key, value in results["metrics"].items():
                if isinstance(value, float):
                    print("  %-16s %.3f" % (key, value))
            if "report" in results["metrics"]:
                print("\n" + results["metrics"]["report"])
            print("Top features:")
            print(results["importance"].head(5).to_string(index=False, float_format=_fmt))
            results["comparison"].to_csv(os.path.join(args.out, "comparison.csv"), index=False)
            save_model(results["bundle"], os.path.join(args.out, "model.joblib"))
            written = save_plots(results, args.out)
            written += [write_report(results, args.target, args.out), write_metrics_json(results, args.out)]
            print("\nSaved the model and %d report files to %s" % (len(written) + 1, args.out))
        else:
            bundle = load_model(args.model)
            result = predict(bundle, pd.read_csv(args.data))
            parent = os.path.dirname(args.out)
            if parent:
                os.makedirs(parent, exist_ok=True)
            result.to_csv(args.out, index=False)
            print("Predicted %d rows. Saved %s" % (len(result), args.out))
    except (ValueError, OSError, KeyError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    return 0
