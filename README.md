# Bank Marketing Subscription Prediction

This project implements a complete Machine Learning workflow to predict the success of bank telemarketing campaigns. The goal is to determine if a client will subscribe to a term deposit based on demographic and social-economic data.

## 🚀 Project Overview
- **Data Source:** Bank Marketing (UCI Machine Learning Repository).
- **Techniques:** Outlier capping (IQR), Mode imputation, One-Hot Encoding, Feature Selection (ANOVA F-test).
- **Models:** Logistic Regression, Decision Tree, Random Forest.
- **Handling Imbalance:** Used `class_weight='balanced'` to handle the skewed dataset.

## 📁 Project Structure
- `main.py`: The entry point that runs the entire pipeline.
- `preprocessing.py`: Data cleaning and feature engineering.
- `feature_selection.py`: Identifying the top-K most important features.
- `models.py`: Model definition and training logic.
- `visualization.py`: Generates EDA and performance plots (Confusion Matrix, Feature Importance).
- `config.py`: Centralized configuration for hyperparameters and paths.

## 📊 Key Results
The project compares multiple models and selects the best performer based on the **F1-Score**, ensuring a balance between Precision and Recall.

## 🛠️ How to Run
1. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn`
2. Run the pipeline: `python main.py`
