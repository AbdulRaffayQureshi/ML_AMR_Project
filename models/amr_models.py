"""
models/amr_models.py
======================
All 4 ML models with a unified interface.
Each model wraps sklearn/xgboost with project-specific defaults from config.py.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import ModelConfig


class BaseAMRModel:
    """Abstract base. All models implement this interface."""

    name:  str = "Base"
    color: str = "#ffffff"

    def __init__(self):
        self._model = None
        self._fitted = False

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X):
        assert self._fitted, f"{self.name} not trained. Call .fit() first."
        return self._model.predict(X)

    def predict_proba(self, X):
        assert self._fitted
        return self._model.predict_proba(X)

    def score(self, X, y) -> float:
        return roc_auc_score(y, self.predict_proba(X)[:, 1])

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def sklearn_model(self):
        return self._model


class XGBoostAMRModel(BaseAMRModel):
    """
    Primary model — XGBoost Classifier.
    Chosen because:
      • Native handling of sparse binary gene matrices
      • Built-in class imbalance via scale_pos_weight
      • Compatible with SHAP TreeExplainer (fast exact SHAP)
      • Consistently top performer in AMR literature (Gao et al. 2024: 98.36%)
    """
    name  = "XGBoost"
    color = ModelConfig.MODEL_COLORS["XGBoost"]
    rationale = (
        "XGBoost handles sparse binary genomic matrices natively. "
        "scale_pos_weight corrects class imbalance. "
        "SHAP TreeExplainer provides fast, exact feature attributions — "
        "critical for clinical interpretability and viva explanation."
    )

    def fit(self, X_train, y_train):
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        params = {**ModelConfig.XGBOOST, "scale_pos_weight": n_neg / max(n_pos, 1)}
        self._model = xgb.XGBClassifier(**params)
        self._model.fit(X_train, y_train)
        self._fitted = True
        return self

    @property
    def feature_importances_(self):
        return self._model.feature_importances_


class RandomForestAMRModel(BaseAMRModel):
    """
    Ensemble baseline — Random Forest.
    Chosen because:
      • Robust to overfitting on high-dimensional genomic data
      • Built-in feature importance (Gini impurity)
      • No feature scaling required for binary inputs
    """
    name  = "Random Forest"
    color = ModelConfig.MODEL_COLORS["Random Forest"]
    rationale = (
        "Random Forest is the standard ensemble baseline for genomic AMR tasks. "
        "It naturally handles binary gene matrices without scaling, "
        "and provides reliable Gini feature importance."
    )

    def fit(self, X_train, y_train):
        self._model = RandomForestClassifier(**ModelConfig.RANDOM_FOREST)
        self._model.fit(X_train, y_train)
        self._fitted = True
        return self

    @property
    def feature_importances_(self):
        return self._model.feature_importances_


class GradientBoostingAMRModel(BaseAMRModel):
    """
    Sequential boosting alternative — Gradient Boosting.
    Chosen because:
      • Smooth probability outputs
      • Strong generalisation with deviance loss
      • Published results: Gao et al. (2024) used SHAP-enhanced GBM
    """
    name  = "Gradient Boosting"
    color = ModelConfig.MODEL_COLORS["Gradient Boosting"]
    rationale = (
        "Gradient Boosting produces well-calibrated probabilities. "
        "Gao et al. (2024) [4] used SHAP-enhanced GBM for ICU CRAB prediction, "
        "achieving sensitivity 0.91 — we replicate this for comparison."
    )

    def fit(self, X_train, y_train):
        self._model = GradientBoostingClassifier(**ModelConfig.GRADIENT_BOOSTING)
        self._model.fit(X_train, y_train)
        self._fitted = True
        return self

    @property
    def feature_importances_(self):
        return self._model.feature_importances_


class LogisticRegressionAMRModel(BaseAMRModel):
    """
    Linear baseline — Logistic Regression.
    Chosen because:
      • Wang et al. (2023) used LASSO (L1 regularisation) — this is the L1-free counterpart
      • Interpretable coefficients
      • Fast training; useful as sanity-check baseline
    """
    name  = "Logistic Regression"
    color = ModelConfig.MODEL_COLORS["Logistic Regression"]
    rationale = (
        "Logistic Regression mirrors Wang et al. (2023) who used LASSO for AMR prediction "
        "(AUC 0.97). Our version uses C=1 (L2) to compare linear vs nonlinear approaches."
    )

    def fit(self, X_train, y_train):
        self._model = LogisticRegression(**ModelConfig.LOGISTIC_REGRESSION)
        self._model.fit(X_train, y_train)
        self._fitted = True
        return self

    @property
    def coefficients(self) -> np.ndarray:
        return self._model.coef_[0]


# ─────────────────────────────────────────────
#  MODEL REGISTRY — single import point
# ─────────────────────────────────────────────

ALL_MODEL_CLASSES = [
    XGBoostAMRModel,
    RandomForestAMRModel,
    GradientBoostingAMRModel,
    LogisticRegressionAMRModel,
]


def build_all_models() -> dict[str, BaseAMRModel]:
    """Returns dict of {name: unfitted_model_instance}."""
    return {cls.name: cls() for cls in ALL_MODEL_CLASSES}


def train_all_models(
    X_train, y_train,
    verbose: bool = True,
) -> dict[str, BaseAMRModel]:
    """
    Trains all 4 models. Returns dict of {name: fitted_model}.
    Pass balanced (SMOTE'd) training data for best results.
    """
    models = {}
    for cls in ALL_MODEL_CLASSES:
        m = cls()
        if verbose:
            print(f"  Training {m.name}...", end=" ", flush=True)
        m.fit(X_train, y_train)
        models[m.name] = m
        if verbose:
            print("[OK]")
    return models