"""
NBA Rookie Career Longevity Predictor - Decision Tree Classifier Pipeline
Author: Antigravity AI
Description: A comprehensive, 12-step machine learning pipeline that loads NBA rookie data,
             preprocesses features, trains a baseline Decision Tree Classifier, interprets
             feature importances, tunes hyperparameters (criterion, max_depth, min_samples_split,
             min_samples_leaf, max_features) via GridSearchCV, and benchmarks the models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

# Set custom styling for premium visualizations
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})

# Custom premium palette
PRIMARY_COLOR = "#2C3E50"  # Midnight Blue
ACCENT_COLOR = "#E74C3C"   # Coral Red
SUCCESS_COLOR = "#2ECC71"  # Emerald Green
NEUTRAL_LIGHT = "#ECF0F1"  # Cool Light Grey

def main():
    print("=" * 80)
    print("        STARTING NBA ROOKIE CAREER LONGEVITY DECISION TREE PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Load Dataset
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: Loading Dataset ---")
    csv_path = "nba_data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing '{csv_path}' in the workspace directory.")
    
    df = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("Columns available:", list(df.columns))
    
    # -------------------------------------------------------------------------
    # STEP 2: Clean Missing Values
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: Cleaning Missing Values ---")
    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0]
    
    if len(columns_with_missing) > 0:
        print("Missing values detected:")
        for col, count in columns_with_missing.items():
            print(f"  - {col}: {count} missing values ({count / len(df) * 100:.2f}%)")
    else:
        print("No missing values detected in the raw CSV!")

    # Separate target '5Yrs' and features
    target_col = "5Yrs"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
        
    X_raw = df.drop(columns=[target_col])
    y = df[target_col]

    # Clean numerical missing values using Median Imputation
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    
    # Impute missing values for numeric columns
    imputer = SimpleImputer(strategy="median")
    X_imputed = X_raw.copy()
    X_imputed[num_cols] = imputer.fit_transform(X_raw[num_cols])
    print(f"Median imputation successfully completed for numerical features.")

    # -------------------------------------------------------------------------
    # STEP 3: Remove/Encode Text Features
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: Removing/Encoding Text Features ---")
    # Identify non-numeric columns (usually player Name)
    text_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Non-numeric columns identified: {text_cols}")
    
    # Save Name column as metadata for tracking individual predictions
    player_names = None
    if "Name" in text_cols:
        player_names = X_imputed["Name"]
        X_features = X_imputed.drop(columns=["Name"])
        print("Dropped 'Name' column from training features and saved it as tracking metadata.")
    else:
        X_features = X_imputed
        print("No 'Name' column found. Features remain as-is.")
        
    feature_names = X_features.columns.tolist()
    print(f"Final feature count for training: {len(feature_names)}")
    print("Features list:", feature_names)

    # -------------------------------------------------------------------------
    # STEP 4: Feature Scaling
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: Feature Scaling ---")
    # Note: Decision Trees are invariant to feature scaling (they split on
    # thresholds, not magnitudes). We apply StandardScaler here to keep the
    # preprocessing pipeline identical to the Logistic Regression model,
    # enabling a fair and direct comparison between the two approaches.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    print("Applied StandardScaler for pipeline consistency with Logistic Regression.")
    print("Note: Scaling does not affect Decision Tree behavior.")

    # -------------------------------------------------------------------------
    # STEP 5: Train-Test Split
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: Train-Test Split ---")
    # Stratified 80-20 split to maintain the class balance of the target '5Yrs'
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X_scaled, y, player_names, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Splitting completed (Stratified 80-20, Random State = 42):")
    print(f"  - Train set shape: {X_train.shape[0]} rows")
    print(f"  - Test set shape: {X_test.shape[0]} rows")
    print(f"  - Class distribution in Train set: {np.bincount(y_train)}")
    print(f"  - Class distribution in Test set: {np.bincount(y_test)}")

    # -------------------------------------------------------------------------
    # STEP 6: Train Baseline Decision Tree Classifier
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: Training Baseline Decision Tree Classifier ---")
    # Baseline: Gini criterion, no depth limit, default splits — likely overfits
    baseline_model = DecisionTreeClassifier(criterion="gini", random_state=42)
    baseline_model.fit(X_train, y_train)
    
    # Report baseline tree complexity
    print("Baseline model trained successfully.")
    print(f"  - Tree depth: {baseline_model.get_depth()}")
    print(f"  - Number of leaves: {baseline_model.get_n_leaves()}")
    print(f"  - Number of features used: {np.sum(baseline_model.feature_importances_ > 0)}/{len(feature_names)}")
    
    # Check for overfitting by comparing train vs test accuracy
    train_accuracy = accuracy_score(y_train, baseline_model.predict(X_train))
    print(f"  - Train accuracy: {train_accuracy:.4f} (watch for overfitting if >> test accuracy)")

    # -------------------------------------------------------------------------
    # STEP 7: Generate Predictions
    # -------------------------------------------------------------------------
    print("\n--- STEP 7: Generating Predictions ---")
    y_pred_base = baseline_model.predict(X_test)
    y_prob_base = baseline_model.predict_proba(X_test)[:, 1]
    print("Class labels and prediction probabilities successfully generated for the test set.")

    # -------------------------------------------------------------------------
    # STEP 8: Evaluate Baseline Metrics
    # -------------------------------------------------------------------------
    print("\n--- STEP 8: Evaluating Baseline Model Metrics ---")
    
    # Calculate key classification metrics
    base_accuracy = accuracy_score(y_test, y_pred_base)
    base_precision = precision_score(y_test, y_pred_base)
    base_recall = recall_score(y_test, y_pred_base)
    base_f1 = f1_score(y_test, y_pred_base)
    base_auc = roc_auc_score(y_test, y_prob_base)
    
    print("-" * 50)
    print(f"Baseline Accuracy:  {base_accuracy:.4f}")
    print(f"Baseline Precision: {base_precision:.4f}")
    print(f"Baseline Recall:    {base_recall:.4f}")
    print(f"Baseline F1-Score:  {base_f1:.4f}")
    print(f"Baseline ROC-AUC:   {base_auc:.4f}")
    print("-" * 50)
    
    print("\nClassification Report (Baseline Decision Tree):")
    print(classification_report(y_test, y_pred_base, target_names=["< 5 Yrs", ">= 5 Yrs"]))

    # -------------------------------------------------------------------------
    # STEP 9: Interpret Feature Importances
    # -------------------------------------------------------------------------
    print("\n--- STEP 9: Interpreting Feature Importances ---")
    importances = baseline_model.feature_importances_
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance (Gini)": importances,
        "Importance (%)": importances * 100
    }).sort_values(by="Importance (Gini)", ascending=False)
    
    print("\nFeature Importances (sorted by Gini importance):")
    print(importance_df[["Feature", "Importance (Gini)", "Importance (%)"]].to_string(index=False))
    
    # Identify which features the tree actually uses for splitting
    used_features = importance_df[importance_df["Importance (Gini)"] > 0]
    unused_features = importance_df[importance_df["Importance (Gini)"] == 0]
    print(f"\nFeatures actively used in splits: {len(used_features)}/{len(feature_names)}")
    if len(unused_features) > 0:
        print(f"Unused features: {list(unused_features['Feature'])}")
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    colors = [SUCCESS_COLOR if imp > np.mean(importances) else PRIMARY_COLOR 
              for imp in importance_df["Importance (Gini)"]]
    
    sns.barplot(
        x="Importance (Gini)", 
        y="Feature", 
        data=importance_df, 
        palette=colors,
        hue="Feature",
        legend=False
    )
    plt.axvline(x=np.mean(importances), color=ACCENT_COLOR, linestyle="--", 
                linewidth=1.5, alpha=0.7, label=f"Mean Importance ({np.mean(importances):.4f})")
    plt.title("Decision Tree Feature Importances (Gini Impurity Reduction)", pad=15)
    plt.xlabel("Importance Score (Gini)")
    plt.ylabel("Rookie Stat Feature")
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    importance_plot_path = "dt_feature_importance.png"
    plt.savefig(importance_plot_path, dpi=300)
    plt.close()
    print(f"\nFeature importance plot successfully saved to '{importance_plot_path}'")

    # -------------------------------------------------------------------------
    # STEP 10: Tune Hyperparameters (GridSearchCV)
    # -------------------------------------------------------------------------
    print("\n--- STEP 10: Tuning Hyperparameters (GridSearchCV) ---")
    
    # Define hyperparameter grid for pre-pruning
    param_grid = {
        "criterion": ["gini", "entropy"],            # Split quality metric
        "max_depth": [3, 5, 7, 10, 15, None],        # Tree depth limit
        "min_samples_split": [2, 5, 10, 20],         # Min samples to split a node
        "min_samples_leaf": [1, 2, 5, 10],           # Min samples at a leaf node
        "max_features": ["sqrt", "log2", None]       # Feature subset at each split
    }
    
    # Use Stratified 5-Fold Cross Validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="f1",  # Target F1-Score as optimization metric
        n_jobs=-1,
        verbose=0
    )
    
    print("Running GridSearchCV... (this may take a moment)")
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    
    print(f"\nOptimal parameters found:")
    print(f"  - Criterion:         {best_params['criterion'].capitalize()}")
    print(f"  - Max Depth:         {best_params['max_depth']}")
    print(f"  - Min Samples Split: {best_params['min_samples_split']}")
    print(f"  - Min Samples Leaf:  {best_params['min_samples_leaf']}")
    print(f"  - Max Features:      {best_params['max_features']}")
    print(f"  - Best CV F1-Score:  {grid_search.best_score_:.4f}")
    
    # Save the tuned model
    tuned_model = grid_search.best_estimator_
    
    # Report tuned tree complexity
    print(f"\nTuned tree complexity:")
    print(f"  - Tree depth: {tuned_model.get_depth()}")
    print(f"  - Number of leaves: {tuned_model.get_n_leaves()}")

    # -------------------------------------------------------------------------
    # STEP 11: Final Benchmark Comparison
    # -------------------------------------------------------------------------
    print("\n--- STEP 11: Final Benchmark Comparison ---")
    
    # Generate predictions from tuned model
    y_pred_tuned = tuned_model.predict(X_test)
    y_prob_tuned = tuned_model.predict_proba(X_test)[:, 1]
    
    # Calculate tuned metrics
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    tuned_precision = precision_score(y_test, y_pred_tuned)
    tuned_recall = recall_score(y_test, y_pred_tuned)
    tuned_f1 = f1_score(y_test, y_pred_tuned)
    tuned_auc = roc_auc_score(y_test, y_prob_tuned)
    
    # Compile comparison table
    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Baseline DT (Gini, No Pruning)": [base_accuracy, base_precision, base_recall, base_f1, base_auc],
        "Tuned DT": [tuned_accuracy, tuned_precision, tuned_recall, tuned_f1, tuned_auc]
    })
    comparison_df["Difference"] = comparison_df["Tuned DT"] - comparison_df["Baseline DT (Gini, No Pruning)"]
    
    print("\n" + "="*70)
    print("              DECISION TREE MODEL PERFORMANCE COMPARISON")
    print("="*70)
    print(comparison_df.to_string(index=False, formatters={
        "Baseline DT (Gini, No Pruning)": "{:.4f}".format,
        "Tuned DT": "{:.4f}".format,
        "Difference": "{:+.4f}".format
    }))
    print("="*70)
    
    # Check generalization improvement
    tuned_train_accuracy = accuracy_score(y_train, tuned_model.predict(X_train))
    print(f"\nGeneralization check:")
    print(f"  - Baseline: Train Acc = {train_accuracy:.4f}, Test Acc = {base_accuracy:.4f}, "
          f"Gap = {train_accuracy - base_accuracy:.4f}")
    print(f"  - Tuned:    Train Acc = {tuned_train_accuracy:.4f}, Test Acc = {tuned_accuracy:.4f}, "
          f"Gap = {tuned_train_accuracy - tuned_accuracy:.4f}")

    # -------------------------------------------------------------------------
    # STEP 12: Generate Evaluation Visualizations
    # -------------------------------------------------------------------------
    print("\n--- STEP 12: Generating Evaluation Visualizations ---")
    
    # 1. Confusion Matrix Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    cm_base = confusion_matrix(y_test, y_pred_base)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    
    # Baseline Confusion Matrix Heatmap
    sns.heatmap(
        cm_base, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
        xticklabels=["< 5 Yrs", ">= 5 Yrs"], yticklabels=["< 5 Yrs", ">= 5 Yrs"]
    )
    axes[0].set_title("Baseline Decision Tree\nConfusion Matrix", pad=10)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # Tuned Confusion Matrix Heatmap
    sns.heatmap(
        cm_tuned, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1],
        xticklabels=["< 5 Yrs", ">= 5 Yrs"], yticklabels=["< 5 Yrs", ">= 5 Yrs"]
    )
    axes[1].set_title(f"Tuned Decision Tree\n({best_params['criterion'].capitalize()}, "
                      f"depth={best_params['max_depth']})", pad=10)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    
    plt.suptitle("Decision Tree Confusion Matrix Comparison", y=1.02)
    plt.tight_layout()
    cm_plot_path = "dt_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix comparison to '{cm_plot_path}'")

    # 2. ROC Curve Plot
    plt.figure(figsize=(8, 6.5))
    
    # Calculate ROC curve values
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_prob_tuned)
    
    plt.plot(fpr_base, tpr_base, color=PRIMARY_COLOR, linewidth=2.5, 
             label=f"Baseline DT (AUC = {base_auc:.4f})")
    plt.plot(fpr_tuned, tpr_tuned, color=SUCCESS_COLOR, linewidth=2.5, linestyle="--", 
             label=f"Tuned DT (AUC = {tuned_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#BDC3C7", linestyle=":", linewidth=1.5, 
             label="Random Guess (AUC = 0.5000)")
    
    plt.title("Decision Tree ROC Curve Comparison", pad=15)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    roc_plot_path = "dt_roc_curve.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve comparison to '{roc_plot_path}'")
    
    print("\n" + "=" * 80)
    print("             PIPELINE COMPLETED SUCCESSFULLY! ALL ARTIFACTS GENERATED.")
    print("=" * 80)

if __name__ == "__main__":
    main()
