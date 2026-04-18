"""
views/shap_views.py
=====================
All SHAP-related visualisations.
Separated from plots.py because SHAP has its own rendering logic.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import ShapConfig
from views.plots import DARK_LAYOUT, MODEL_COLORS


class SHAPAnalyser:
    """
    Wraps SHAP computation and provides view-ready outputs.

    Usage:
        sa = SHAPAnalyser(xgb_model, feature_names)
        sa.compute(X_test)            # compute SHAP values
        fig = sa.beeswarm()           # summary beeswarm
        fig = sa.waterfall(idx=0)     # single sample explanation
        fig = sa.global_importance()  # mean |SHAP| bar chart
    """

    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self._explainer  = None
        self._shap_vals  = None
        self._X_sample   = None
        self._base_value = None

    def compute(self, X_test: np.ndarray, max_samples: int = None) -> "SHAPAnalyser":
        """Compute SHAP values (TreeExplainer — fast, exact)."""
        n = min(max_samples or ShapConfig.MAX_SAMPLES, len(X_test))
        self._X_sample  = X_test[:n]
        self._explainer = shap.TreeExplainer(self.model.sklearn_model)
        self._shap_vals  = self._explainer.shap_values(self._X_sample)
        self._base_value = float(self._explainer.expected_value)
        print(f"  SHAP computed for {n} samples. Base value = {self._base_value:.4f}")
        return self

    # ─────────────────────────────────────────────────────────
    #  GLOBAL PLOTS
    # ─────────────────────────────────────────────────────────

    def global_importance(self, top_n: int = None) -> go.Figure:
        """Mean |SHAP| bar chart — global feature importance."""
        top_n = top_n or ShapConfig.TOP_FEATURES
        mean_sv = np.abs(self._shap_vals).mean(axis=0)
        imp = (
            pd.DataFrame({"Gene": self.feature_names, "Mean_SHAP": mean_sv})
            .sort_values("Mean_SHAP", ascending=False)
            .head(top_n)
        )
        fig = go.Figure(go.Bar(
            x=imp["Mean_SHAP"].values,
            y=imp["Gene"].values,
            orientation="h",
            marker=dict(
                color=imp["Mean_SHAP"].values,
                colorscale="OrRd",
            ),
        ))
        fig.update_layout(
            **DARK_LAYOUT, height=500,
            title=f"Global SHAP Importance — Top {top_n} Features (Mean |SHAP|)",
            xaxis_title="Mean |SHAP Value|",
            yaxis={"categoryorder": "total ascending"},
        )
        return fig

    def beeswarm(self, top_n: int = 15) -> tuple:
        """matplotlib beeswarm / dot summary plot."""
        mean_sv  = np.abs(self._shap_vals).mean(axis=0)
        top_idx  = np.argsort(mean_sv)[-top_n:]
        top_names = [self.feature_names[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1b2a")
        shap.summary_plot(
            self._shap_vals[:, top_idx],
            self._X_sample[:, top_idx],
            feature_names=top_names,
            show=False, plot_type="dot",
        )
        ax.set_title(f"SHAP Beeswarm — Top {top_n} Features",
                      color="#c9d8e8", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#90caf9")
        plt.tight_layout()
        return fig, ax

    # ─────────────────────────────────────────────────────────
    #  PER-SAMPLE PLOTS
    # ─────────────────────────────────────────────────────────

    def waterfall(self, idx: int, top_n: int = 14) -> go.Figure:
        """Plotly waterfall for a single sample — shows cumulative probability shift."""
        sv = self._shap_vals[idx]
        top_idx  = np.argsort(np.abs(sv))[-top_n:]
        top_sv   = sv[top_idx]
        top_names = [self.feature_names[i] for i in top_idx]

        # Build cumulative waterfall
        cum = self._base_value
        bars = []
        for name_, v_ in zip(top_names, top_sv):
            bars.append((name_, v_, cum, cum + v_))
            cum += v_

        fig = go.Figure()
        for name_, v_, base_, top_ in bars:
            fig.add_trace(go.Bar(
                x=[name_], y=[abs(v_)],
                base=[min(base_, top_)],
                marker_color="#ef5350" if v_ > 0 else "#26a69a",
                hovertemplate=f"<b>{name_}</b><br>SHAP={v_:+.4f}<extra></extra>",
                showlegend=False,
            ))
        fig.add_hline(y=self._base_value, line_dash="dot", line_color="#ffa726",
                       annotation_text=f"Base={self._base_value:.3f}")
        fig.update_layout(
            **DARK_LAYOUT, height=400,
            title="SHAP Waterfall — Feature Contributions for This Sample",
            xaxis_title="Gene", yaxis_title="Probability Contribution",
            xaxis={"tickangle": -40},
        )
        return fig

    def resistance_gauge(self, model, X_sample: np.ndarray) -> go.Figure:
        """Probability gauge for a single sample."""
        prob = float(model.predict_proba(X_sample.reshape(1, -1))[0][1])
        col  = "#ef5350" if prob > 0.5 else "#26a69a"
        fig  = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Resistance Probability (%)",
                   "font": {"size": 14, "color": "#c9d8e8"}},
            number={"suffix": "%", "font": {"color": col, "size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#546e7a"},
                "bar" : {"color": col},
                "bgcolor": "#0d1b2a", "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0,  30], "color": "#0d2b1a"},
                    {"range": [30, 70], "color": "#1a1a0d"},
                    {"range": [70, 100],"color": "#2b0d0d"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 50},
            },
        ))
        fig.update_layout(**DARK_LAYOUT, height=320)
        return fig

    def sample_attribution_bar(self, idx: int, top_n: int = 12) -> go.Figure:
        """Horizontal bar of SHAP contributions for one sample."""
        sv     = self._shap_vals[idx]
        top_idx = np.argsort(np.abs(sv))[-top_n:]
        names  = [self.feature_names[i] for i in top_idx]
        values = sv[top_idx]

        fig = go.Figure(go.Bar(
            x=names, y=values,
            marker=dict(
                color=values,
                colorscale=[[0, "#26a69a"], [0.5, "#ffa726"], [1, "#ef5350"]],
                cmin=-max(abs(values)), cmax=max(abs(values)),
            ),
        ))
        fig.add_hline(y=0, line_color="white", line_width=1)
        fig.update_layout(
            **DARK_LAYOUT, height=360,
            title="SHAP Attribution — Top Drivers for This Sample",
            xaxis_title="Gene", yaxis_title="SHAP Contribution",
            xaxis={"tickangle": -35},
        )
        return fig

    # ─────────────────────────────────────────────────────────
    #  ACCESSORS
    # ─────────────────────────────────────────────────────────

    @property
    def shap_values(self) -> np.ndarray:
        return self._shap_vals

    @property
    def base_value(self) -> float:
        return self._base_value

    @property
    def mean_abs_shap(self) -> pd.Series:
        return pd.Series(
            np.abs(self._shap_vals).mean(axis=0),
            index=self.feature_names,
        ).sort_values(ascending=False)