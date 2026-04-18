"""
controllers/train_controller.py
=================================
Orchestrates model training and cross-validation.
Keeps all training logic OUT of the view layer.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import ModelConfig
from models.amr_models import train_all_models, BaseAMRModel


class TrainController:
    """
    Usage:
        tc = TrainController()
        tc.train(X_train_bal, y_train_bal)
        tc.cross_validate(X_train, y_train)   # uses original (not SMOTE'd)
        models = tc.models
        cv_results = tc.cv_results
    """

    def __init__(self):
        self.models: dict[str, BaseAMRModel] = {}
        self.cv_results: dict[str, np.ndarray] = {}

    def train(self, X_train, y_train, verbose: bool = True) -> "TrainController":
        """Train all 4 models on (optionally SMOTE-balanced) training data."""
        print("\n=== MODEL TRAINING ===")
        self.models = train_all_models(X_train, y_train, verbose=verbose)
        print(f"  All {len(self.models)} models trained.\n")
        return self

    def cross_validate(
        self,
        X_train, y_train,
        n_splits: int = None,
        scoring: str = "roc_auc",
        verbose: bool = True,
    ) -> "TrainController":
        """
        Stratified K-Fold CV on original (not SMOTE'd) training data.
        SMOTE is intentionally NOT applied inside CV to avoid data leakage.
        """
        n_splits = n_splits or ModelConfig.CV_FOLDS
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=ModelConfig.RANDOM_STATE)

        print(f"\n=== {n_splits}-FOLD CROSS-VALIDATION ({scoring}) ===")
        for name, model in self.models.items():
            scores = cross_val_score(
                model.sklearn_model, X_train, y_train,
                cv=skf, scoring=scoring, n_jobs=-1,
            )
            self.cv_results[name] = scores
            if verbose:
                print(f"  {name:24s}  mean={scores.mean():.4f}  std=±{scores.std():.4f}")
        return self

    def cv_summary_df(self) -> pd.DataFrame:
        """Returns a tidy DataFrame of CV results."""
        rows = []
        for name, scores in self.cv_results.items():
            for fold_i, s in enumerate(scores, 1):
                rows.append({"Model": name, "Fold": fold_i, "AUC_ROC": s})
        return pd.DataFrame(rows)

    def best_model(self) -> tuple[str, BaseAMRModel]:
        """Returns (name, model) of best CV performer."""
        if not self.cv_results:
            raise ValueError("Run .cross_validate() first.")
        best_name = max(self.cv_results, key=lambda k: self.cv_results[k].mean())
        return best_name, self.models[best_name]