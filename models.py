# ============================================================
# models.py
# Responsible for creating and training ML models.
#
# Models used:
#   1. Logistic Regression  — simple linear baseline
#   2. Decision Tree        — interpretable rule-based model
#   3. Random Forest        — ensemble of decision trees
#
# Note: class_weight='balanced' is used because the dataset
#       is imbalanced (~89% "no", ~11% "yes"). This setting
#       tells the model to pay more attention to the minority
#       class (subscribers).
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from config import RANDOM_STATE

#----------------------------------------------------------------


def get_models():
    """
    Create and return a dictionary of untrained ML models.

    Returns:
        models (dict): { model_name: model_object }
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced"   # handles class imbalance
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,                # use all CPU cores
            class_weight="balanced"
        ),
    }
    return models
#--------------------------------------------------------------------------


def train_models(models, X_train, y_train):
    """
    Train all models on the training data.

    Args:
        models  (dict)  : Dictionary of untrained models.
        X_train (array) : Training feature matrix.
        y_train (array) : Training target vector.

    Returns:
        trained_models (dict): Dictionary of trained models.
    """
    print("[6] Training models...")

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"    → {name} trained ✓")

    print()
    return trained_models
