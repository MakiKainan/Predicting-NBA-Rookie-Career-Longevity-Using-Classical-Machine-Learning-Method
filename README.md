# NBA Rookie Career Longevity Predictor: Model Comparison & Walkthrough

This document reports and compares the results of two different machine learning models trained on rookie stats to predict whether an NBA rookie's career will last $\ge 5$ years (`5Yrs` target):
1. **Logistic Regression Classifier** 
2. **Decision Tree Classifier** 

Both models utilize automated 12-step pipelines (from raw data loading and preprocessing to hyperparameters tuning via Stratified 5-Fold Grid Search, evaluation, and visualization).

---

## 1. Head-to-Head Model Performance Comparison

Here is the direct comparison of the optimized (tuned) models evaluated on the test dataset:

| Metric | Tuned Logistic Regression | Tuned Decision Tree | Model Performance Winner | Analysis & Takeaways |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | 69.47% | **70.23%** | **Decision Tree (+0.76%)** | The Decision Tree correctly classifies slightly more total rookies than Logistic Regression. |
| **Precision** | 71.51% (Baseline) / **73.45%** | **73.58%** | **Decision Tree (+0.13%)** | Extremely close. The Decision Tree has a marginally lower false-positive rate. |
| **Recall** | **79.75%** | 75.00% | **Logistic Regression (+4.75%)** | Logistic Regression is significantly more sensitive. It successfully identifies 79.75% of players who reach the 5-year career threshold. |
| **F1-Score** | **76.47%** | 74.29% | **Logistic Regression (+2.18%)** | Logistic Regression maintains a better overall harmonic balance between Precision and Recall. |
| **ROC-AUC** | 74.74% | **75.44%** | **Decision Tree (+0.70%)** | The Decision Tree offers slightly better discriminative capabilities across all decision thresholds. |

### Key Performance Insights
- **The Talent Acquisition Trade-off**: 
  - If a front office's primary goal is **talent retention** (avoiding false negatives, i.e., making sure they don't miss out on or cut a rookie who will succeed long-term), **Logistic Regression is the superior choice** due to its **higher Recall (79.75% vs. 75.00%)**.
  - If the goal is **overall classification correctness** and maximizing the ROC-AUC profile, the **Decision Tree Classifier** has a slight edge.
- **Overfitting & Generalization Control**:
  - The baseline Decision Tree memorized the training data perfectly (**100% train accuracy vs. 62.60% test accuracy**). Tuning the tree (limiting depth to 5, leaves to 14) successfully slashed this overfitting gap from **37.40%** to just **3.42%** (73.64% train vs 70.23% test).
  - The baseline Logistic Regression was tuned using L2 regularization with $C \approx 0.1624$ (equivalent to penalty strength $\alpha \approx 6.16$), preventing overfitting and boosting test accuracy by **+2.29%** over the baseline.

---

## 2. Sports Analytics & Feature Interpretation Comparison

Each model extracts feature importance in a different way, providing complementary scout insights:

### A. Games Played (`GP`) is the Primary Predictor
- **Decision Tree**: Ranks **GP** as the #1 most important feature with a **14.18% Gini Importance**.
- **Logistic Regression**: Assigns a strong positive coefficient of **+0.623** to **GP**, representing an **Odds Ratio (OR) of 1.86**.
- **Scouting Insight**: Durability, health, and earning rotation minutes early on are the strongest indicators of long-term career longevity. A standard deviation increase in games played multiplies the odds of reaching 5+ years in the league by **1.86x**.

### B. Contrast: Linear vs. Non-linear Insights
- **Logistic Regression (Linear Coefficient Odds Ratios)**:
  - Highlights **efficiency vs. volume**:
    - **`3PM` (3-Pointers Made)** has a massive positive coefficient of **+1.168** (Odds Ratio = **3.21**).
    - **`3PA` (3-Point Attempts)** has a heavy negative coefficient of **-1.188** (Odds Ratio = **0.30**).
    - *Insight:* High-volume rookie shooters who miss their three-pointers are a severe liability to career longevity. Efficiency is valued over volume.
    - **`FTM` (+0.480, OR=1.62)** vs. **`FTA` (-0.401, OR=0.67)** follows the same trend: drawing fouls is only helpful if they are converted.
    - **`OREB` (Offensive Rebounds)** is a strong positive predictor (**+0.506, OR=1.66**), reflecting hustle and active post efforts, which secure roster spots.
- **Decision Tree (Gini Importances)**:
  - Highlights **volume and usage thresholds**:
    - Ranks **`FGA` (Field Goal Attempts - 9.84%)**, **`MIN` (Minutes - 9.29%)**, and **`PTS` (Points - 8.48%)** right behind games played.
    - *Insight:* The tree makes decisions based on physical workload and scoring thresholds (e.g., separating players who cross specific minute and shooting volume marks).

---


## 3. Pipeline-Specific Breakdowns

### Pipeline A: Logistic Regression
- **Script**: [nba_logistic_regression.py](file:///c:/Users/Kevin Sukias/Documents/InshaAllah my projects/NBA Machine Learning/nba_logistic_regression.py)
- **Tuning Mechanism**: Stratified 5-Fold Grid Search over regularization penalty types (`l1`, `l2`) and inverse regularization strengths (`C`).
- **Optimal Hyperparameters**: L2 regularization penalty, $C \approx 0.1624$.
- **Performance Details**:

| Metric | Baseline Model (L2, C=1.0) | Tuned Model (L2, C=0.1624) | Improvement |
|:---|:---:|:---:|:---:|
| **Accuracy** | 67.18% | **69.47%** | **+2.29%** |
| **Precision** | 71.51% | **73.45%** | **+1.94%** |
| **Recall** | 78.53% | **79.75%** | **+1.23%** |
| **F1-Score** | 74.85% | **76.47%** | **+1.62%** |
| **ROC-AUC** | 74.36% | **74.74%** | **+0.38%** |

### Pipeline B: Decision Tree Classifier
- **Script**: [nba_decision_tree.py](file:///c:/Users/Kevin Sukias/Documents/InshaAllah my projects/NBA Machine Learning/nba_decision_tree.py)
- **Tuning Mechanism**: Grid Search over `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.
- **Optimal Hyperparameters**:
  - **Criterion**: Entropy
  - **Max Depth**: 5 (reduced from 21)
  - **Min Samples Split**: 20
  - **Min Samples Leaf**: 10
  - **Max Features**: None (all features considered)
  - **Number of Leaves**: 14 (reduced from 208)
- **Performance Details**:

| Metric | Baseline DT (unpruned) | Tuned DT | Improvement |
|---|---|---|---|
| **Accuracy** | 62.60% | **70.23%** | **+7.63%** |
| **Precision** | 67.95% | **73.58%** | **+5.62%** |
| **Recall** | 67.95% | **75.00%** | **+7.05%** |
| **F1-Score** | 67.95% | **74.29%** | **+6.33%** |
| **ROC-AUC** | 62.41% | **75.44%** | **+13.03%** |

---

## 4. Verification Checklist

- [x] Preprocessed features are standardized using StandardScaler to ensure alignment for both linear and non-linear calculations.
- [x] Run both model scripts: [nba_logistic_regression.py] and [nba_decision_tree.py]
- [x] Confirm saved visual artifacts in [Logistic Regression output image]and [Decision Tree Output image]
- [x] Verify that inference module [nba_inference.py]
