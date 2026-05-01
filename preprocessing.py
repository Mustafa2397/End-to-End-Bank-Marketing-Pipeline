# ============================================================
# preprocessing.py
# Responsible for cleaning and preparing the data for modeling.
# Steps:
#   1. Remove duplicate rows
#   2. Replace 'unknown' values with the column mode
#   3. Cap outliers using the IQR method
#   4. Encode categorical columns (One-Hot Encoding)
#   5. Encode the target column (Label Encoding: no=0, yes=1)
#   6. Scale numerical features using StandardScaler
# ============================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import NUMERICAL_COLS, TARGET_COL

#---------------------------------------------------------------

def clean_data(df):
    """
    Clean the raw DataFrame:
      - Remove duplicates
      - Replace 'unknown' strings with the column's most frequent value (mode)
      - Cap outliers using the IQR (Interquartile Range) method

    Args:
        df (DataFrame): Raw dataset.

    Returns:
        df (DataFrame): Cleaned dataset.
    """
    print("[2] Cleaning data...")

    # --- Step 1: Remove duplicate rows ---
    before = len(df)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    print(f"    → Duplicates removed : {before - len(df)}")

    # --- Step 2: Replace 'unknown' with column mode ---
    unknown_count = 0
    for col in df.select_dtypes(include="object").columns:
        if col == TARGET_COL:
            continue
        n = (df[col] == "unknown").sum()
        if n > 0:
            mode_value = df[col][df[col] != "unknown"].mode()[0]
            df[col] = df[col].replace("unknown", mode_value)
            unknown_count += n
    print(f"    → 'unknown' values replaced : {unknown_count}")

    # --- Step 3: Cap outliers using 1.5 × IQR ---
    for col in NUMERICAL_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    print(f"    → Outliers capped on {len(NUMERICAL_COLS)} numerical columns")
    print(f"    → Clean dataset shape: {df.shape}\n")

    return df

#---------------------------------------------------------------------

def preprocess_data(df):
    """
    Encode and scale the cleaned DataFrame so it is ready for ML models.

    Steps:
      - One-Hot Encode all categorical columns (except the target)
      - Label Encode the target column (no → 0, yes → 1)
      - Apply StandardScaler to all feature columns

    Args:
        df (DataFrame): Cleaned dataset.

    Returns:
        X_scaled  (DataFrame) : Scaled feature matrix.
        y         (Series)    : Encoded target vector.
        feat_names(list)      : List of feature column names.
    """
    print("[3] Preprocessing data...")

    # --- Step 4: One-Hot Encode categorical columns ---
    cat_cols = [
        c for c in df.select_dtypes(include="object").columns
        if c != TARGET_COL
    ]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"    → Columns after One-Hot Encoding: {df_enc.shape[1]}")

    # --- Step 5: Label Encode the target column ---
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])   # no → 0, yes → 1
    print(f"    → Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # --- Step 6: Separate features and apply StandardScaler ---
    X = df_enc.drop(TARGET_COL, axis=1)
    feat_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feat_names
    )
    print(f"    → Feature matrix shape: {X_scaled.shape}")
    print(f"    → StandardScaler applied (mean≈0, std≈1)\n")

    return X_scaled, y, feat_names
