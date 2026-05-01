# ============================================================
# evaluation.py
# Responsible for evaluating trained models on the test set.
#
# Metrics calculated per model:
#   - Accuracy  : overall correct predictions
#   - Precision : of all predicted "yes", how many were real
#   - Recall    : of all real "yes", how many were caught
#   - F1 Score  : harmonic mean of Precision and Recall
# ============================================================

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

#---------------------------------------------------------------
def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate each trained model and return their metrics.

    Args:
        trained_models (dict)  : Dictionary of trained model objects.
        X_test         (array) : Test feature matrix.
        y_test         (array) : True labels for the test set.

    Returns:
        results     (dict) : { model_name: { metric: value, ... } }
        predictions (dict) : { model_name: predicted_labels }
    """
    print("[7] Evaluating models...\n")

    results = {}
    predictions = {}

    for name, model in trained_models.items():
        # Generate predictions
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        # Calculate metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)

        results[name] = {
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1 Score":  round(f1,   4),
        }

        # Print detailed report
        print(f"  {'='*54}")
        print(f"  Model : {name}")
        print(f"  {'='*54}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"\n  Full Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

    return results, predictions

#----------------------------------------------------------------------------

def print_comparison_table(results):
    """
    Print a formatted side-by-side comparison table of all models
    and highlight the best model by F1 Score.

    Args:
        results (dict): Output from evaluate_models().
    """
    print("\n" + "="*60)
    print("  MODEL COMPARISON TABLE")
    print("="*60)

    # Build a DataFrame for clean display
    df = pd.DataFrame(results).T
    print(df.to_string())

    # Identify the best model
    best_model = df["F1 Score"].idxmax()
    best_f1    = df.loc[best_model, "F1 Score"]
    best_acc   = df.loc[best_model, "Accuracy"]

    print("\n" + "="*60)
    print(f"  🏆  Best Model : {best_model}")
    print(f"      F1 Score   : {best_f1:.4f}")
    print(f"      Accuracy   : {best_acc:.4f}")
    print("="*60 + "\n")
