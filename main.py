# ============================================================
# main.py
# Entry point of the project.
# Runs the complete ML workflow from data loading to evaluation.
#
# Usage:
#   python main.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

# Project modules
from config          import TEST_SIZE, RANDOM_STATE, NUMERICAL_COLS, TARGET_COL
from data_loader     import load_data
from preprocessing   import clean_data, preprocess_data
from feature_selection import select_features
from models          import get_models, train_models
from evaluation      import evaluate_models, print_comparison_table
from visualization   import (
    run_eda_plots,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_feature_importance,
)

#--------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("   BANK MARKETING — ML CLASSIFICATION WORKFLOW")
    print("="*60 + "\n")

    # ── Step 1 & 2: Load raw data ────────────────────────────────
    df_raw = load_data()

    # ── Step 3: Clean data ───────────────────────────────────────
    df_clean = clean_data(df_raw.copy())

    # ── Step 4: EDA plots ────────────────────────────────────────
    cat_cols = [
        c for c in df_clean.select_dtypes(include="object").columns
        if c != TARGET_COL
    ]
    run_eda_plots(df_clean, NUMERICAL_COLS, cat_cols, TARGET_COL)

    # ── Step 5: Preprocess (encode + scale) ──────────────────────
    X_scaled, y, feature_names = preprocess_data(df_clean.copy())

    # ── Step 6: Train / Test Split ───────────────────────────────
    print("[4] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # keep class ratio in both splits
    )
    print(f"    → Train samples : {len(X_train):,}")
    print(f"    → Test  samples : {len(X_test):,}\n")

    # ── Step 7: Feature Selection ────────────────────────────────
    X_train_sel, X_test_sel, selected_features = select_features(
        X_train, X_test, y_train, feature_names
    )

    # ── Step 8: Train Models ─────────────────────────────────────
    models         = get_models()
    trained_models = train_models(models, X_train_sel, y_train)

    # ── Step 9: Evaluate Models ──────────────────────────────────
    results, predictions = evaluate_models(trained_models, X_test_sel, y_test)

    # ── Step 10: Print Comparison Table ──────────────────────────
    print_comparison_table(results)

    # ── Step 11: Save All Model Plots ────────────────────────────
    print("[8] Saving model evaluation plots...")
    plot_confusion_matrices(trained_models, predictions, y_test)
    plot_model_comparison(results)
    plot_feature_importance(trained_models, selected_features)
    print()

    print("="*60)
    print("   ✅  Workflow complete! All plots saved in /plots")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
