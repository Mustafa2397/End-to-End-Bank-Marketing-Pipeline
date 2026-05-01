# ============================================================
# visualization.py
# Responsible for generating and saving all project plots.
#
# Plots produced:
#   1. Distribution of numerical features
#   2. Count plots for categorical features
#   3. Target variable distribution (bar + pie)
#   4. Correlation heatmap
#   5. Key features vs target (boxplots)
#   6. Confusion matrices for all models
#   7. Model comparison bar chart
#   8. Feature importance (Random Forest)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Apply a consistent visual theme to all plots
sns.set_theme(style="whitegrid", palette="muted")

# Output folder for saved plots
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
#--------------------------------------------------------------


def _save(filename):
    """Helper: save the current figure and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"    → Saved: {path}")


# ------------------------------------------------------------------
# EDA Plots
# ------------------------------------------------------------------

def plot_numerical_distributions(df, num_cols):
    """Bar histograms for every numerical feature."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=40, color="steelblue",
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
    plt.suptitle("Distribution of Numerical Features",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save("01_numerical_distributions.png")

#------------------------------------------------------------------

def plot_categorical_counts(df, cat_cols):
    """Count bar plots for categorical features (dynamic grid)."""
    n_cols = 4
    n_rows = -(-len(cat_cols) // n_cols)   # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, ax=axes[i],
                      order=order, palette="Set2")
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xticklabels(
            axes[i].get_xticklabels(),
            rotation=45, ha="right", fontsize=8
        )
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")

    # Hide any unused subplot cells
    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Count Plots for Categorical Features",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("02_categorical_counts.png")

#----------------------------------------------------------------

def plot_target_distribution(df, target_col):
    """Bar chart + pie chart for target variable balance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    counts = df[target_col].value_counts()

    # Bar chart
    ax1.bar(counts.index, counts.values,
            color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.5)
    ax1.set_title("Target Distribution (Count)", fontweight="bold")
    ax1.set_xlabel("Subscribed (y)")
    ax1.set_ylabel("Count")
    for rect, val in zip(ax1.patches, counts.values):
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 200,
            f"{val:,}", ha="center", fontweight="bold"
        )

    # Pie chart
    ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=["#e74c3c", "#2ecc71"], startangle=90,
            wedgeprops=dict(edgecolor="white"))
    ax2.set_title("Target Distribution (%)", fontweight="bold")

    plt.suptitle("Target Variable: Term Deposit Subscription (y)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("03_target_distribution.png")

#----------------------------------------------------------------

def plot_correlation_heatmap(df, num_cols):
    """Lower-triangle correlation heatmap for numerical columns."""
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap of Numerical Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("04_correlation_heatmap.png")

#---------------------------------------------------------------

def plot_features_vs_target(df, target_col):
    """Boxplots comparing key numerical features by target class."""
    features = [
        "age", "duration", "campaign",
        "emp.var.rate", "euribor3m", "cons.conf.idx"
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.boxplot(data=df, x=target_col, y=col,
                    palette={"no": "#e74c3c", "yes": "#2ecc71"},
                    ax=axes[i])
        axes[i].set_title(f"{col} vs Subscription", fontweight="bold")
        axes[i].set_xlabel("Subscribed (y)")
        axes[i].set_ylabel(col)
    plt.suptitle("Key Features vs Target Variable",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("05_features_vs_target.png")

#-------------------------------------------------------------------

# ------------------------------------------------------------------
# Model Evaluation Plots
# ------------------------------------------------------------------

def plot_confusion_matrices(trained_models, predictions, y_test):
    """
    Side-by-side confusion matrices for all models.

    Args:
        trained_models (dict): Trained model objects.
        predictions    (dict): { model_name: y_pred }
        y_test         (array): True test labels.
    """
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import accuracy_score, f1_score

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["No", "Yes"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        ax.set_title(
            f"{name}\nAcc={acc:.4f}  |  F1={f1:.4f}",
            fontweight="bold", fontsize=10
        )
    plt.suptitle("Confusion Matrices — All Models",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("06_confusion_matrices.png")

#---------------------------------------------------------------

def plot_model_comparison(results):
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1
    across all models.

    Args:
        results (dict): Output from evaluate_models().
    """
    import pandas as pd

    metrics_df = pd.DataFrame(results).T
    model_names = metrics_df.index.tolist()
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors      = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    x     = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (metric, color) in enumerate(zip(metric_cols, colors)):
        ax.bar(x + i * width, metrics_df[metric], width,
               label=metric, color=color, alpha=0.88)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — All Metrics",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, alpha=0.5)
    plt.tight_layout()
    _save("07_model_comparison.png")

#-----------------------------------------------------------------

def plot_feature_importance(trained_models, selected_features):
    """
    Horizontal bar chart of feature importances from Random Forest.

    Args:
        trained_models    (dict): Trained model objects.
        selected_features (list): Names of features used for training.
    """
    import pandas as pd

    # Only Random Forest exposes feature_importances_
    rf = trained_models.get("Random Forest")
    if rf is None:
        print("    → Random Forest model not found. Skipping feature importance plot.")
        return

    importances = rf.feature_importances_
    feat_df = pd.DataFrame({
        "Feature":    selected_features,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"],
                   color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Feature Importances — Random Forest",
                 fontweight="bold", fontsize=13)
    for bar, val in zip(bars, feat_df["Importance"]):
        ax.text(val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    _save("08_feature_importance.png")

#-------------------------------------------------------------------

# ------------------------------------------------------------------
# Run all EDA plots at once
# ------------------------------------------------------------------

def run_eda_plots(df, num_cols, cat_cols, target_col):
    """
    Convenience wrapper: run all EDA plots in one call.

    Args:
        df         (DataFrame): Cleaned dataset.
        num_cols   (list)     : Numerical column names.
        cat_cols   (list)     : Categorical column names.
        target_col (str)      : Name of the target column.
    """
    print("[4] Generating EDA plots...")
    plot_numerical_distributions(df, num_cols)
    plot_categorical_counts(df, cat_cols)
    plot_target_distribution(df, target_col)
    plot_correlation_heatmap(df, num_cols)
    plot_features_vs_target(df, target_col)
    print()
