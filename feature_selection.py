# ============================================================
# feature_selection.py
# Responsible for selecting the most important features.
#
# Method: SelectKBest with f_classif (ANOVA F-test)
#   - Scores each feature by its statistical relationship
#     with the target variable.
#   - Keeps only the top-K most informative features.
#   - Reduces noise and speeds up model training.
# ============================================================

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from config import TOP_K_FEATURES


def select_features(X_train, X_test, y_train, feature_names):
    """
    Select the top K features using the ANOVA F-test (SelectKBest).

    Args:
        X_train      (array): Training feature matrix.
        X_test       (array): Testing feature matrix.
        y_train      (array): Training target vector.
        feature_names (list): Names of all input features.

    Returns:
        X_train_sel (array) : Training matrix with selected features only.
        X_test_sel  (array) : Testing matrix with selected features only.
        selected    (list)  : Names of the selected features.
    """
    print(f"[5] Feature Selection — SelectKBest (k={TOP_K_FEATURES})...")

    # Fit the selector on training data only
    selector = SelectKBest(score_func=f_classif, k=TOP_K_FEATURES)
    selector.fit(X_train, y_train)

    # Apply the selector to both train and test sets
    X_train_sel = selector.transform(X_train)
    X_test_sel  = selector.transform(X_test)

    # Get the names of the selected features
    mask = selector.get_support()
    feature_names_array = np.array(feature_names)
    selected = feature_names_array[mask].tolist()

    print(f"    → Features before selection : {X_train.shape[1]}")
    print(f"    → Features after  selection : {len(selected)}")
    print(f"    → Selected features:")
    for f in selected:
        print(f"        • {f}")
    print()

    return X_train_sel, X_test_sel, selected
