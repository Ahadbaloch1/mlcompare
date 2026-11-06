"""Plots, Markdown report and model persistence."""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REQUIRED_KEYS = {"version", "pipeline", "task", "features"}


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_plots(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    task = results["task"]
    comparison = results["comparison"]
    paths = []

    metric = "f1_macro" if task == "classification" else "r2"
    table = comparison.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(table["model"], table[metric], xerr=table[metric + "_std"], color="#3b6fb6", capsize=3)
    ax.set_xlabel("%s (cross-validation mean)" % metric)
    ax.set_title("Model comparison")
    paths.append(_save(fig, outdir, "model_comparison.png"))

    top = results["importance"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#2f8f6b", capsize=3)
    ax.set_xlabel("Score drop when the feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    paths.append(_save(fig, outdir, "permutation_importance.png"))

    metrics = results["metrics"]
    if task == "classification":
        labels, matrix = metrics["classes"], metrics["confusion"]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix (test set)")
        cutoff = matrix.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > cutoff else "black")
        paths.append(_save(fig, outdir, "confusion_matrix.png"))
    else:
        actual, predicted = np.asarray(results["y_test"]), metrics["predicted"]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(actual, predicted, alpha=0.7, color="#3b6fb6")
        low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax.plot([low, high], [low, high], "--", color="gray")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs actual (test set)")
        paths.append(_save(fig, outdir, "predicted_vs_actual.png"))
    return paths


def _table(df, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in df.iterrows():
        cells = [("%.3f" % row[c]) if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(results, target, outdir):
    os.makedirs(outdir, exist_ok=True)
    task, metrics = results["task"], results["metrics"]
    lines = [
        "# MLCompare report", "",
        "- Task: %s" % task,
        "- Target column: `%s`" % target,
        "- Training rows: %d, test rows: %d" % (results["n_train"], results["n_test"]),
        "- Best model: **%s**" % results["best_model"],
        "- Tuned parameters: %s" % (results["best_params"] or "none"), "",
        "## Cross-validation comparison (training split)", "",
    ]
    columns = [c for c in results["comparison"].columns if not c.endswith("_std")]
    lines += [_table(results["comparison"], columns), "", "## Test-set results", ""]
    if task == "classification":
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"):
            if key in metrics:
                lines.append("- %s: %.3f" % (key, metrics[key]))
        lines += ["", "```", metrics["report"].rstrip(), "```"]
    else:
        for key in ("r2", "mae", "rmse"):
            lines.append("- %s: %.3f" % (key, metrics[key]))
    lines += ["", "## Most important features (permutation importance)", "",
              _table(results["importance"].head(10), ["feature", "importance", "std"]), ""]
    path = os.path.join(outdir, "report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def write_metrics_json(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    metrics = {k: v for k, v in results["metrics"].items()
               if isinstance(v, (int, float, str)) and k != "report"}
    payload = {"task": results["task"], "best_model": results["best_model"],
               "best_params": {k: ("None" if v is None else v) for k, v in results["best_params"].items()},
               "test_metrics": metrics}
    path = os.path.join(outdir, "metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def save_model(bundle, path):
    joblib.dump(bundle, path)


def load_model(path):
    """Load a model file. Only load files that you trained yourself or trust."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not REQUIRED_KEYS.issubset(bundle):
        raise ValueError("Not a valid MLCompare model file.")
    return bundle
