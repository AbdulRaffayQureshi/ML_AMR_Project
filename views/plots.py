"""
views/plots.py
================
ALL visualisation functions.
Every function accepts raw data / controller outputs and returns a figure.
Nothing is trained or computed here — pure rendering.

Convention:
  plotly functions return go.Figure (for Streamlit / notebook widgets)
  matplotlib functions return (fig, axes) (for Jupyter inline display)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
#  SHARED PLOTLY THEME
# ─────────────────────────────────────────────
DARK_LAYOUT = dict(
    template      = "plotly_dark",
    paper_bgcolor = "#0d1117",
    plot_bgcolor  = "#0d1b2a",
    font          = dict(family="DM Sans, sans-serif", color="#c9d8e8"),
    margin        = dict(t=55, b=45, l=45, r=20),
)

MODEL_COLORS = {
    "XGBoost"            : "#ef5350",
    "Random Forest"      : "#26a69a",
    "Gradient Boosting"  : "#42a5f5",
    "Logistic Regression": "#ab47bc",
}


# ═══════════════════════════════════════════════════════════
#  EDA PLOTS
# ═══════════════════════════════════════════════════════════

def plot_class_distribution(y: np.ndarray) -> go.Figure:
    """Donut chart — Resistant vs Susceptible."""
    n_res = int((y == 1).sum())
    n_sus = int((y == 0).sum())
    fig = go.Figure(go.Pie(
        labels=["Resistant", "Susceptible"],
        values=[n_res, n_sus],
        hole=0.58,
        marker=dict(colors=["#ef5350", "#26a69a"],
                    line=dict(color="#0d1117", width=3)),
        textinfo="label+percent",
        textfont=dict(size=13),
    ))
    fig.update_layout(
        **DARK_LAYOUT, height=300, showlegend=False,
        title="Class Distribution",
        annotations=[dict(
            text=f"<b>{len(y):,}</b><br>Strains",
            x=0.5, y=0.5, font_size=14,
            font_color="#e0f7fa", showarrow=False,
        )],
    )
    return fig


def plot_gene_prevalence(
    df: pd.DataFrame, feature_names: list[str],
    top_n: int = 25,
) -> go.Figure:
    """Grouped bars: gene prevalence in Resistant vs Susceptible isolates."""
    res = df[df.resistance == 1][feature_names].mean()
    sus = df[df.resistance == 0][feature_names].mean()
    top_genes = res.sort_values(ascending=False).head(top_n).index.tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Resistant", x=top_genes, y=res[top_genes].values,
        marker_color="#ef5350", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Susceptible", x=top_genes, y=sus[top_genes].values,
        marker_color="#26a69a", opacity=0.85,
    ))
    fig.update_layout(
        **DARK_LAYOUT, height=430, barmode="group",
        title=f"Top {top_n} Genes — Prevalence by Phenotype",
        xaxis_title="Gene", yaxis_title="Proportion of Strains",
        xaxis={"tickangle": -42},
    )
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, feature_names: list[str], top_n: int = 20,
) -> tuple:
    """matplotlib heatmap — top N genes vs resistance label."""
    res_prev = df[feature_names].mean().sort_values(ascending=False)
    top_cols  = res_prev.head(top_n).index.tolist()
    corr = df[top_cols + ["resistance"]].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1b2a")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=mask, linewidths=0.4, linecolor="#0d1117",
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Gene × Resistance Correlation Matrix", color="#c9d8e8",
                  fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="#90caf9", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════
#  PREPROCESSING PLOTS
# ═══════════════════════════════════════════════════════════

def plot_smote_balance(y_before: np.ndarray, y_after: np.ndarray) -> go.Figure:
    """Side-by-side bars showing class balance before/after SMOTE."""
    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=["Before SMOTE", "After SMOTE"])
    for col_i, (y, title) in enumerate([(y_before, "Before"), (y_after, "After")], 1):
        n_s = int((y == 0).sum())
        n_r = int((y == 1).sum())
        opacity = 1.0 if title == "After" else 0.6
        fig.add_trace(go.Bar(
            x=["Susceptible", "Resistant"], y=[n_s, n_r],
            marker_color=["#26a69a", "#ef5350"],
            text=[n_s, n_r], textposition="outside",
            showlegend=False, opacity=opacity,
        ), row=1, col=col_i)
    fig.update_layout(**DARK_LAYOUT, height=330,
                       title="Class Balance Before vs After SMOTE",
                       yaxis_title="Samples")
    return fig


def plot_pca_scatter(X: np.ndarray, y: np.ndarray) -> go.Figure:
    """2D PCA scatter coloured by resistance label."""
    pca = PCA(n_components=2, random_state=42)
    Xp  = pca.fit_transform(X)
    ev  = pca.explained_variance_ratio_
    df  = pd.DataFrame({
        "PC1": Xp[:, 0], "PC2": Xp[:, 1],
        "Status": ["Resistant" if r == 1 else "Susceptible" for r in y],
    })
    fig = px.scatter(
        df, x="PC1", y="PC2", color="Status",
        color_discrete_map={"Resistant": "#ef5350", "Susceptible": "#26a69a"},
        opacity=0.55,
        title=f"PCA 2D Projection  (PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%)",
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(**DARK_LAYOUT, height=430)
    return fig


def plot_cumulative_variance(X: np.ndarray, n_components: int = 20) -> go.Figure:
    """Cumulative explained variance curve."""
    pca  = PCA(n_components=min(n_components, X.shape[1]), random_state=42).fit(X)
    cumv = np.cumsum(pca.explained_variance_ratio_) * 100
    fig  = px.area(x=list(range(1, len(cumv)+1)), y=cumv,
                    labels={"x": "Components", "y": "Variance (%)"},
                    title="Cumulative Explained Variance")
    fig.add_hline(y=80, line_dash="dash", line_color="#ffa726",
                   annotation_text="80% threshold")
    fig.update_layout(**DARK_LAYOUT, height=280)
    return fig


def plot_chi2_ranking(chi2_df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    """Horizontal bar chart — chi-squared feature importance."""
    top = chi2_df.head(top_n)
    fig = px.bar(top, y="Gene", x="Chi2_Score", orientation="h",
                  color="Chi2_Score", color_continuous_scale="Reds",
                  title=f"Top {top_n} Genes — Chi-Squared Discriminative Score")
    fig.update_layout(**DARK_LAYOUT, height=520)
    return fig


# ═══════════════════════════════════════════════════════════
#  EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════

def plot_roc_curves(eval_results: dict, y_test: np.ndarray) -> go.Figure:
    """Multi-model ROC curves."""
    from sklearn.metrics import auc
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dot", color="#546e7a"), name="Random (0.50)",
    ))
    for name, r in eval_results.items():
        fpr = r["curves"]["fpr"]
        tpr = r["curves"]["tpr"]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name}  (AUC={auc(fpr,tpr):.4f})",
            line=dict(color=MODEL_COLORS.get(name, "#ffffff"), width=2.5),
        ))
    fig.update_layout(
        **DARK_LAYOUT, height=420,
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.08),
    )
    return fig


def plot_pr_curves(eval_results: dict) -> go.Figure:
    """Multi-model Precision-Recall curves."""
    fig = go.Figure()
    for name, r in eval_results.items():
        prec = r["curves"]["precision"]
        rec  = r["curves"]["recall"]
        ap   = r["avg_precision"]
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{name}  (AP={ap:.4f})",
            line=dict(color=MODEL_COLORS.get(name, "#ffffff"), width=2.5),
        ))
    fig.update_layout(
        **DARK_LAYOUT, height=420,
        title="Precision-Recall Curves",
        xaxis_title="Recall", yaxis_title="Precision",
        legend=dict(x=0.01, y=0.08),
    )
    return fig


def plot_confusion_matrices(eval_results: dict) -> tuple:
    """2×2 matplotlib grid of confusion matrices."""
    n = len(eval_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.patch.set_facecolor("#0d1117")
    if n == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, eval_results.items()):
        cm = r["cm"]
        ax.set_facecolor("#0d1b2a")
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Susceptible", "Resistant"],
            yticklabels=["Susceptible", "Resistant"],
            linewidths=0.5, linecolor="#0d1117",
            annot_kws={"size": 14, "color": "white"},
        )
        acc = r["accuracy"]
        ax.set_title(f"{name}\nAcc={acc:.4f}", color="#c9d8e8",
                      fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", color="#90caf9")
        ax.set_ylabel("Actual",    color="#90caf9")
        ax.tick_params(colors="#90caf9", labelsize=8)
    plt.tight_layout()
    return fig, axes


def plot_radar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Spider/radar chart — multi-metric model comparison."""
    cats = ["Accuracy", "AUC-ROC", "F1-Score", "Sensitivity", "Specificity", "Precision"]
    fig  = go.Figure()
    for name, row in metrics_df.iterrows():
        vals = [row.get(c, 0) for c in cats] + [row.get(cats[0], 0)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=name,
            line_color=MODEL_COLORS.get(name, "#ffffff"), opacity=0.45,
        ))
    fig.update_layout(
        **DARK_LAYOUT, height=480,
        polar=dict(
            bgcolor="#0d1b2a",
            radialaxis=dict(visible=True, range=[0.75, 1.0], color="#546e7a"),
            angularaxis=dict(color="#90caf9"),
        ),
        title="Multi-Metric Model Comparison",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_feature_importance(model, feature_names: list[str], top_n: int = 20) -> go.Figure:
    """XGBoost / RF feature importance bar chart."""
    fi = pd.DataFrame({
        "Gene": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)

    fig = px.bar(fi, y="Gene", x="Importance", orientation="h",
                  color="Importance", color_continuous_scale="Reds",
                  title=f"Top {top_n} Features — {model.name} Gain")
    fig.update_layout(**DARK_LAYOUT, height=500)
    return fig


def plot_cv_box(cv_summary_df: pd.DataFrame) -> go.Figure:
    """Box plot of cross-validation AUC scores per model."""
    fig = go.Figure()
    for name in cv_summary_df["Model"].unique():
        vals = cv_summary_df[cv_summary_df["Model"] == name]["AUC_ROC"].values
        fig.add_trace(go.Box(
            y=vals, name=name,
            marker_color=MODEL_COLORS.get(name, "#ffffff"),
            boxpoints="all", jitter=0.3, pointpos=-1.8,
        ))
    fig.update_layout(
        **DARK_LAYOUT, height=420,
        title="5-Fold CV AUC-ROC Distribution",
        yaxis_title="AUC-ROC", yaxis_range=[0.80, 1.01],
    )
    return fig