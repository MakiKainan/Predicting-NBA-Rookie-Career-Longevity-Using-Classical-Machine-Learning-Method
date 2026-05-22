# Implementation Plan: NBA Rookie Career Longevity Predictor (Logistic Regression)

This plan details the construction of a complete, production-grade Logistic Regression pipeline using the user's specified 12-step workflow on `nba_data.csv`. The target variable is `5Yrs` (a binary label indicating whether a rookie player will have a career longevity of 5+ years).

## User Review Required

> [!IMPORTANT]
> **Dependencies**: This pipeline requires `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`. If any of these are missing from your environment, the script will attempt to explain how to install them, or we can install them in the workspace environment.
>
> **Feature Selection**: We will drop the `Name` column as a feature for prediction (since player names are unique identifiers and text-based) but will preserve it if we want to trace individual predictions. All other columns (`GP`, `MIN`, `PTS`, etc.) are numerical and will be used as features.

## Open Questions

> [!NOTE]
> 1. **Handling Missing Values**: Standard practice for this dataset is median or mean imputation for any missing values (usually a few missing values occur in percentage columns if they were present, or shooting stats). We plan to use median imputation to maintain robustness. Is this acceptable, or would you prefer dropping rows with missing values?
> 2. **Evaluation Visualizations**: We plan to save performance visualization plots (Confusion Matrix, ROC Curve, and Feature Importance Coefficients) as `.png` files in the workspace so you can inspect them visually. Would you like us to generate these?

## Proposed Changes

### Machine Learning Pipeline Component

We will create a clean, modular Python script containing the entire pipeline. The script will be designed to run end-to-end, producing detailed log outputs, performance metrics, and visualization figures.

#### [NEW] [nba_logistic_regression.py](file:///c:/Users/Kevin%20Sukias/Documents/InshaAllah%20my%20projects/NBA%20Machine%20Learning/nba_logistic_regression.py)

This file will contain the python script implementing the 12-step pipeline:
1. **Load Dataset**: Read `nba_data.csv`.
2. **Clean Missing Values**: Detect and impute missing numerical values using `SimpleImputer` (median strategy).
3. **Remove/Encode Text**: Extract and drop the non-numeric `Name` column.
4. **Feature Scaling**: Scale numeric features using `StandardScaler`.
5. **Train-Test Split**: Stratified 80-20 split to maintain the ratio of the target `5Yrs`.
6. **Train Logistic Regression**: Train a baseline model with L2 regularization.
7. **Predict**: Predict class labels and probabilities on the test set.
8. **Evaluate Metrics**: Calculate and print Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and Classification Report.
9. **Interpret Coefficients**: Map and sort the standardized coefficients $w_i$. Since the features are scaled, these coefficients represent the change in the log-odds of the positive class (`5Yrs = 1`) per standard deviation increase in the corresponding feature. We will also compute and plot the **Odds Ratios** ($\exp(w_i)$) to convert log-odds to highly intuitive multiplicative changes in odds, providing a direct, non-technical translation for sports analytics stakeholders.
10. **Tune Regularization**: Perform a grid search (`GridSearchCV`) over `C` (regularization strength) and penalties (`l1`, `l2`) using the `liblinear` solver.
11. **Final Baseline Benchmark**: Train the optimized model and present a clear comparison table between the baseline and tuned models.
12. **Visualizations**: Generate and save plots for the Confusion Matrix, ROC Curve, and Coefficient Importance.

## Verification Plan

### Automated Verification
- We will execute the Python script `nba_logistic_regression.py` and capture its outputs in the terminal.
- We will verify that all 12 pipeline steps complete successfully without error.
- We will confirm that the three visualization PNG files (`confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`) are correctly created in the workspace.

### Manual Verification
- Review the metrics (Accuracy, F1, ROC-AUC) to ensure the model exhibits good predictive power without severe overfitting.
- Interpret the most important features (e.g., Games Played `GP` and Rebounds `OREB`/`DREB` are traditionally strong predictors of career longevity in this dataset).
