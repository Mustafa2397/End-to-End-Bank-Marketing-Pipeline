# ============================================================
# config.py
# Central place for all project settings and constants.
# Change values here to affect the whole project.
# ============================================================

# Path to the dataset file
DATA_FILE = "bank-additional-full.csv"

# Separator used in the CSV file
DATA_SEPARATOR = ";"

# Fraction of data used for testing (0.2 = 20%)
TEST_SIZE = 0.2

# Random seed for reproducibility
RANDOM_STATE = 42

# Number of top features to keep after feature selection
TOP_K_FEATURES = 20

# Numerical columns in the dataset
NUMERICAL_COLS = [
    "age", "duration", "campaign", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]

# Target column name
TARGET_COL = "y"
