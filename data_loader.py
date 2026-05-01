# ============================================================
# data_loader.py
# Responsible for loading the dataset from disk.
# ============================================================

import pandas as pd
from config import DATA_FILE, DATA_SEPARATOR


def load_data():
    """
    Load the CSV dataset into a pandas DataFrame.

    Returns:
        df (DataFrame): The raw dataset.
    """
    print(f"[1] Loading dataset: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, sep=DATA_SEPARATOR)

    print(f"    → Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"    → Columns    : {df.columns.tolist()}")
    print(f"    → Data types :\n{df.dtypes.to_string()}")
    print(f"\n    First 5 rows:\n{df.head().to_string()}\n")

    return df
