"""
controllers/eval_controller.py
================================
Computes all evaluation metrics for every model.
Pure computation — no plotting happens here (that's in views/).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    average_precision_score,
    confusion_matrix, roc_curve,
    precision_recall_curve,
    classification_report,
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


class EvalController:
    """
    Usage:
        ec = EvalController()
        ec.evaluate_all(models, X_test, y_test)
        df = ec.metrics_df        # full metrics table
        ec.print_report("XGBoost")
    """

    def __init__(self):
        self.results: dict = {}   # {model_name: metric_dict}
        self._X_test = None
        self._y_test = None
        self._models = None

    # ─────────────────────────────────────────────────────────
    #  PUBLIC
    # ─────────────────────────────────────────────────────────

    def evaluate_all(self, models: dict, X_test, y_test) -> "EvalController":
        self._X_test = X_test
        self._y_test = y_test
        self._models = models

        print("\n=== EVALUATION ON HELD-OUT TEST SET ===")
        for name, model in models.items():
            r = self._compute_metrics(model, X_test, y_test)
            self.results[name] = r
            print(f"  {name:24s}  AUC={r['auc_roc']:.4f}  Acc={r['accuracy']:.4f}  "
                  f"F1={r['f1']:.4f}  Sens={r['sensitivity']:.4f}  "
                  f"Spec={r['specificity']:.4f}")
        return self

    @property
    def metrics_df(self) -> pd.DataFrame:
        """Tidy DataFrame — one row per model, columns = metrics."""
        rows = []
        for name, r in self.results.items():
            rows.append({
                "Model"      : name,
                "Accuracy"   : r["accuracy"],
                "AUC-ROC"    : r["auc_roc"],
                "F1-Score"   : r["f1"],
                "Sensitivity": r["sensitivity"],
                "Specificity": r["specificity"],
                "Precision"  : r["precision"],
                "Avg Precision": r["avg_precision"],
            })
        return (
            pd.DataFrame(rows)
            .set_index("Model")
            .sort_values("AUC-ROC", ascending=False)
        )

    def get_curves(self, model_name: str) -> dict:
        """Returns ROC + PR curve arrays for a single model."""
        return self.results[model_name]["curves"]

    def get_confusion_matrix(self, model_name: str) -> np.ndarray:
        return self.results[model_name]["cm"]

    def get_probabilities(self, model_name: str) -> np.ndarray:
        return self.results[model_name]["y_proba"]

    def print_report(self, model_name: str):
        r = self.results[model_name]
        print(f"\n=== Classification Report: {model_name} ===")
        print(classification_report(
            self._y_test, r["y_pred"],
            target_names=["Susceptible", "Resistant"],
        ))

    def best_model_name(self) -> str:
        return self.metrics_df["AUC-ROC"].idxmax()

    # ─────────────────────────────────────────────────────────
    #  INTERNAL
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(model, X_test, y_test) -> dict:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
        prec, rec, pr_thresh = precision_recall_curve(y_test, y_proba)

        return {
            "y_pred"        : y_pred,
            "y_proba"       : y_proba,
            "accuracy"      : accuracy_score(y_test, y_pred),
            "auc_roc"       : roc_auc_score(y_test, y_proba),
            "f1"            : f1_score(y_test, y_pred),
            "sensitivity"   : recall_score(y_test, y_pred),           # = recall for positive class
            "specificity"   : recall_score(y_test, y_pred, pos_label=0),
            "precision"     : precision_score(y_test, y_pred, zero_division=0),
            "avg_precision" : average_precision_score(y_test, y_proba),
            "cm"            : confusion_matrix(y_test, y_pred),
            "curves": {
                "fpr": fpr, "tpr": tpr, "roc_thresh": roc_thresh,
                "precision": prec, "recall": rec, "pr_thresh": pr_thresh,
            },
        }