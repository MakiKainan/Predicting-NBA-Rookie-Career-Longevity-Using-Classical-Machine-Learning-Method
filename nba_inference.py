"""
NBA Rookie Career Longevity Predictor - Inference CLI
Author: Antigravity AI
Description: An interactive command-line interface that loads the NBA rookie dataset,
             trains the tuned Logistic Regression and Decision Tree models on the training set,
             and allows users to query player names to predict career longevity.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Terminal color codes for premium aesthetic
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text, color=CYAN):
    print(f"\n{color}{BOLD}{'=' * 80}{RESET}")
    print(f"{color}{BOLD}{text.center(80)}{RESET}")
    print(f"{color}{BOLD}{'=' * 80}{RESET}\n")

def normalize_name(name):
    """Normalize names for robust lookup: lowercase, strip spaces, and remove trailing asterisk."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    if name.endswith("*"):
        name = name[:-1].strip()
    return name

def load_and_train_pipelines():
    """Loads dataset and trains models with optimized hyperparameters found during grid search."""
    csv_path = "nba_data.csv"
    if not os.path.exists(csv_path):
        print(f"{RED}{BOLD}Error: Missing '{csv_path}' in the workspace directory.{RESET}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    target_col = "5Yrs"
    if target_col not in df.columns:
        print(f"{RED}{BOLD}Error: Target column '{target_col}' not found in the dataset.{RESET}")
        sys.exit(1)
        
    X_raw = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Impute missing values (median strategy)
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy="median")
    X_imputed = X_raw.copy()
    X_imputed[num_cols] = imputer.fit_transform(X_raw[num_cols])
    
    player_names = X_imputed["Name"]
    X_features = X_imputed.drop(columns=["Name"])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # Stratified split to match training pipeline splitting
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X_scaled, y, player_names, test_size=0.2, random_state=42, stratify=y
    )
    
    # Instantiate models with optimal parameters from GridSearchCV
    lr_model = LogisticRegression(
        solver="liblinear", 
        penalty="l2", 
        C=0.162378, 
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    dt_model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    # Return all elements needed for search and inference
    return df, X_scaled, y, lr_model, dt_model

def get_predictions(lr_model, dt_model, feature_row):
    """Generates predictions and probabilities from both models for a scaled feature row."""
    # Reshape feature row for single prediction
    feat = feature_row.reshape(1, -1)
    
    # Logistic Regression
    lr_pred = lr_model.predict(feat)[0]
    lr_prob = lr_model.predict_proba(feat)[0][1] # Probability of class 1 (>= 5 years)
    
    # Decision Tree
    dt_pred = dt_model.predict(feat)[0]
    dt_prob = dt_model.predict_proba(feat)[0][1] # Probability of class 1 (>= 5 years)
    
    return lr_pred, lr_prob, dt_pred, dt_prob

def display_player_stats(player_row):
    """Displays the key statistics of the queried player in a clean layout."""
    print(f"{BOLD}--- Rookie Season Statistics for {player_row['Name']} ---{RESET}")
    stats_config = [
        ("Games Played (GP)", "GP", ".0f"),
        ("Minutes Per Game (MIN)", "MIN", ".1f"),
        ("Points Per Game (PTS)", "PTS", ".1f"),
        ("Field Goals Made (FGS)", "FGS", ".1f"),
        ("Field Goals Attempted (FGA)", "FGA", ".1f"),
        ("3-Points Made (3PM)", "3PM", ".1f"),
        ("3-Points Attempted (3PA)", "3PA", ".1f"),
        ("Free Throws Made (FTM)", "FTM", ".1f"),
        ("Free Throws Attempted (FTA)", "FTA", ".1f"),
        ("Offensive Rebounds (OREB)", "OREB", ".1f"),
        ("Defensive Rebounds (DREB)", "DREB", ".1f"),
        ("Assists (AST)", "AST", ".1f"),
        ("Steals (STL)", "STL", ".1f"),
        ("Blocks (BLK)", "BLK", ".1f"),
        ("Turnovers (TOV)", "TOV", ".1f"),
    ]
    
    # Print in two columns for compact premium layout
    for i in range(0, len(stats_config), 2):
        label1, col1, fmt1 = stats_config[i]
        val1 = player_row[col1]
        str1 = f"  • {label1}: {val1:{fmt1}}"
        
        if i + 1 < len(stats_config):
            label2, col2, fmt2 = stats_config[i+1]
            val2 = player_row[col2]
            str2 = f"  • {label2}: {val2:{fmt2}}"
            print(f"{str1:<45} {str2}")
        else:
            print(str1)
    print()

def main():
    print(f"{CYAN}{BOLD}Initializing NBA Rookie Career Longevity Predictor Inference CLI...{RESET}")
    df, X_scaled, y, lr_model, dt_model = load_and_train_pipelines()
    
    # Create normalized list of names for robust search mapping
    df_normalized_names = df["Name"].apply(normalize_name).tolist()
    
    print(f"{GREEN}{BOLD}Initialization Complete. Models trained and ready.{RESET}")
    
    print_header("NBA ROOKIE CAREER LONGEVITY PREDICTOR (CLI)", CYAN)
    print("Welcome to the NBA Rookie Longevity Tester!")
    print("This utility uses machine learning to predict if a rookie player's career")
    print("will last 5 or more years in the NBA based on their rookie season stats.")
    print("You can search for any rookie from the historical dataset.")
    print("Type 'exit' or 'quit' at any time to leave the interface.")
    print("-" * 80)
    
    while True:
        try:
            query = input(f"\n{BOLD}Enter NBA Player Name: {RESET}").strip()
            if not query:
                continue
                
            if query.lower() in ["exit", "quit"]:
                print(f"\n{CYAN}{BOLD}Thank you for using the NBA Rookie Longevity Predictor! Goodbye.{RESET}\n")
                break
                
            norm_query = normalize_name(query)
            
            # Find indices where normalized name matches exactly
            matching_indices = [i for i, name in enumerate(df_normalized_names) if name == norm_query]
            
            # If no exact match, try substring matching
            if not matching_indices:
                matching_indices = [i for i, name in enumerate(df_normalized_names) if norm_query in name]
                if matching_indices:
                    print(f"{YELLOW}No exact match found. Showing partial matches...{RESET}")
                else:
                    print(f"{RED}No player found matching '{query}'. Please try again.{RESET}")
                    continue
            
            # If multiple matches, let the user resolve duplicates
            selected_idx = None
            if len(matching_indices) > 1:
                print(f"\n{YELLOW}{BOLD}Multiple players found matching your query:{RESET}")
                for idx, df_idx in enumerate(matching_indices):
                    row = df.iloc[df_idx]
                    actual_txt = ">= 5 Years" if row['5Yrs'] == 1 else "< 5 Years"
                    print(f"  [{idx + 1}] {row['Name']} (GP: {int(row['GP'])}, PTS: {row['PTS']:.1f}, Actual Career: {actual_txt})")
                
                while True:
                    choice = input(f"{BOLD}Please select a player [1-{len(matching_indices)}]: {RESET}").strip()
                    if choice.lower() in ["exit", "quit"]:
                        break
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(matching_indices):
                            selected_idx = matching_indices[choice_idx]
                            break
                        else:
                            print(f"{RED}Invalid option. Enter a number between 1 and {len(matching_indices)}.{RESET}")
                    except ValueError:
                        print(f"{RED}Please enter a valid number.{RESET}")
                
                if selected_idx is None:  # User typed exit/quit to break out of inner loop
                    print(f"\n{CYAN}{BOLD}Thank you for using the NBA Rookie Longevity Predictor! Goodbye.{RESET}\n")
                    break
            else:
                selected_idx = matching_indices[0]
            
            # Fetch player stats and scale representations
            player_row = df.iloc[selected_idx]
            feature_row = X_scaled[selected_idx]
            
            # Print player statistics
            print()
            display_player_stats(player_row)
            
            # Run model predictions
            lr_pred, lr_prob, dt_pred, dt_prob = get_predictions(lr_model, dt_model, feature_row)
            
            # Ground truth
            actual_val = int(player_row['5Yrs'])
            actual_label = f"{GREEN}YES (>= 5 Years){RESET}" if actual_val == 1 else f"{RED}NO (< 5 Years){RESET}"
            
            # Display Prediction Results
            print(f"{CYAN}{BOLD}{'=' * 70}{RESET}")
            print(f"{CYAN}{BOLD}                         PREDICTION REPORT{RESET}")
            print(f"{CYAN}{BOLD}{'=' * 70}{RESET}")
            print(f"{BOLD}Player Name:                {RESET}{player_row['Name']}")
            print(f"{BOLD}Actual Career Longevity:    {RESET}{actual_label}")
            print("-" * 70)
            
            # 1. Logistic Regression output
            lr_label = f"{GREEN}YES (Sustainable Rookie){RESET}" if lr_pred == 1 else f"{RED}NO (Not Sustainable){RESET}"
            lr_pct = lr_prob * 100 if lr_pred == 1 else (1.0 - lr_prob) * 100
            lr_outcome = ">= 5 years" if lr_pred == 1 else "< 5 years"
            print(f"{BOLD}1. LOGISTIC REGRESSION:{RESET}")
            print(f"   • Predicted Sustainable: {lr_label}")
            print(f"   • Confidence:            {lr_pct:.2f}% probability of {lr_outcome}")
            
            # 2. Decision Tree output
            dt_label = f"{GREEN}YES (Sustainable Rookie){RESET}" if dt_pred == 1 else f"{RED}NO (Not Sustainable){RESET}"
            dt_pct = dt_prob * 100 if dt_pred == 1 else (1.0 - dt_prob) * 100
            dt_outcome = ">= 5 years" if dt_pred == 1 else "< 5 years"
            print(f"\n{BOLD}2. DECISION TREE:{RESET}")
            print(f"   • Predicted Sustainable: {dt_label}")
            print(f"   • Confidence:            {dt_pct:.2f}% probability of {dt_outcome}")
            
            # Agreement summary
            if lr_pred == dt_pred:
                agreement = f"{GREEN}{BOLD}AGREE{RESET} on prediction {lr_label}"
            else:
                agreement = f"{YELLOW}{BOLD}DISAGREE{RESET} (Logistic Regression: {lr_label}, Decision Tree: {dt_label})"
            
            print("-" * 70)
            print(f"{BOLD}Model Consensus:            {RESET}{agreement}")
            print(f"{CYAN}{BOLD}{'=' * 70}{RESET}\n")
            
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{CYAN}{BOLD}Program interrupted. Goodbye!{RESET}\n")
            break

if __name__ == "__main__":
    main()
