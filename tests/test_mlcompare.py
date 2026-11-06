import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlcompare import data, pipeline  # noqa: E402
from mlcompare.cli import main  # noqa: E402
from mlcompare.report import load_model, save_model  # noqa: E402


def make_classification_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    city = rng.choice(["Karachi", "Quetta", "Turbat"], size=n)
    label = np.where(x1 + 0.5 * x2 + (city == "Quetta") * 1.0 + rng.normal(scale=0.3, size=n) > 0.5, "yes", "no")
    df = pd.DataFrame({"x1": x1, "x2": x2, "city": city, "flag": rng.choice([True, False], size=n), "label": label})
    df.loc[rng.choice(n, 10, replace=False), "x2"] = np.nan  # missing numeric values
    df.loc[rng.choice(n, 10, replace=False), "city"] = np.nan  # missing categories
    return df


def make_regression_frame(n=200, seed=1):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": 3 * x1 - 2 * x2 + rng.normal(scale=0.2, size=n)})


class DataTests(unittest.TestCase):
    def test_detect_task(self):
        self.assertEqual(data.detect_task(pd.Series(["a", "b", "a"])), "classification")
        self.assertEqual(data.detect_task(pd.Series([0, 1, 1, 0, 2])), "classification")
        self.assertEqual(data.detect_task(pd.Series(np.linspace(0, 100, 50))), "regression")
        self.assertEqual(data.detect_task(pd.Series([True, False, True])), "classification")

    def test_load_csv_errors_and_cleaning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.csv")
            pd.DataFrame({"a": [1, 2, 3, 4], "empty": [None] * 4, "t": [1, None, 0, 1]}).to_csv(path, index=False)
            X, y, info = data.load_csv(path, "t")
            self.assertEqual(info["rows_dropped_missing_target"], 1)
            self.assertEqual(info["columns_dropped_all_missing"], ["empty"])
            self.assertEqual(list(X.columns), ["a"])
            with self.assertRaises(ValueError):
                data.load_csv(path, "nope")

    def test_clean_features_missing_column(self):
        df = pd.DataFrame({"a": [1.0], "b": ["x"]})
        with self.assertRaises(ValueError):
            data.clean_features(df, ["a", "c"])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = make_classification_frame()
        cls.cls_results = pipeline.run(df.drop(columns=["label"]), df["label"], tune=False, folds=3)
        reg = make_regression_frame()
        cls.reg_results = pipeline.run(reg.drop(columns=["y"]), reg["y"], tune=False, folds=3)

    def test_classification_beats_baseline(self):
        results = self.cls_results
        self.assertEqual(results["task"], "classification")
        best = results["comparison"].iloc[0]
        baseline = results["comparison"][results["comparison"]["model"] == "Baseline (most common / mean)"].iloc[0]
        self.assertGreater(best["f1_macro"], baseline["f1_macro"] + 0.1)
        self.assertGreater(results["metrics"]["accuracy"], 0.8)
        self.assertIn("roc_auc", results["metrics"])

    def test_baseline_never_selected(self):
        self.assertNotEqual(self.cls_results["best_model"], "Baseline (most common / mean)")
        self.assertNotEqual(self.reg_results["best_model"], "Baseline (most common / mean)")

    def test_regression(self):
        results = self.reg_results
        self.assertEqual(results["task"], "regression")
        self.assertGreater(results["metrics"]["r2"], 0.9)
        self.assertEqual(set(results["importance"]["feature"]), {"x1", "x2"})

    def test_importance_ranks_informative_feature(self):
        self.assertEqual(self.reg_results["importance"].iloc[0]["feature"], "x1")

    def test_tuning_returns_params(self):
        df = make_regression_frame(120, seed=3)
        results = pipeline.run(df.drop(columns=["y"]), df["y"], tune=True, folds=3)
        self.assertTrue(results["best_params"])

    def test_predict_handles_unseen_category_and_extra_columns(self):
        bundle = self.cls_results["bundle"]
        new = pd.DataFrame({"x1": [0.5, -2.0], "x2": [0.1, np.nan], "city": ["Gwadar", "Quetta"],
                            "flag": [True, False], "extra": [1, 2]})
        out = pipeline.predict(bundle, new)
        self.assertEqual(len(out), 2)
        self.assertIn("prediction", out.columns)
        self.assertIn("confidence", out.columns)

    def test_predict_missing_column(self):
        with self.assertRaises(ValueError):
            pipeline.predict(self.cls_results["bundle"], pd.DataFrame({"x1": [1.0]}))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.joblib")
            save_model(self.reg_results["bundle"], path)
            loaded = load_model(path)
            new = make_regression_frame(10, seed=9).drop(columns=["y"])
            pd.testing.assert_frame_equal(pipeline.predict(self.reg_results["bundle"], new),
                                          pipeline.predict(loaded, new))

    def test_invalid_model_file(self):
        import joblib
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.joblib")
            joblib.dump({"hello": 1}, path)
            with self.assertRaises(ValueError):
                load_model(path)

    def test_too_few_rows_and_single_class(self):
        with self.assertRaises(ValueError):
            pipeline.run(pd.DataFrame({"a": range(5)}), pd.Series([0, 1, 0, 1, 0]))
        with self.assertRaises(ValueError):
            pipeline.run(pd.DataFrame({"a": range(30)}), pd.Series(["x"] * 30))


class CliTests(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "data.csv")
            df = make_classification_frame(150, seed=5)
            df.to_csv(csv_path, index=False)
            out = os.path.join(tmp, "out")
            self.assertEqual(main(["run", "--data", csv_path, "--target", "label", "--out", out,
                                   "--no-tune", "--folds", "3"]), 0)
            for name in ("model.joblib", "report.md", "metrics.json", "comparison.csv",
                         "model_comparison.png", "confusion_matrix.png", "permutation_importance.png"):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            pred = os.path.join(tmp, "pred.csv")
            self.assertEqual(main(["predict", "--model", os.path.join(out, "model.joblib"),
                                   "--data", csv_path, "--out", pred]), 0)
            self.assertIn("prediction", pd.read_csv(pred).columns)

    def test_missing_target_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "data.csv")
            make_regression_frame(50).to_csv(csv_path, index=False)
            self.assertEqual(main(["run", "--data", csv_path, "--target", "nope"]), 1)


if __name__ == "__main__":
    unittest.main()
