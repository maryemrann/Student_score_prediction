import argparse
import os
import re
import itertools
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils import load_data, select_xy, regression_metrics, save_json, scatter_pred_vs_actual

def find_candidate_features(df: pd.DataFrame, target_col: str) -> list:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # keep numeric except the target
    numeric_cols = [c for c in numeric_cols if c != target_col]
    # prefer columns that look relevant if present
    priority_patterns = [
        r'sleep', r'participation', r'attend', r'previous', r'sample', r'paper', r'extra', r'sports?',
        r'parent', r'tuit', r'health', r'study'
    ]
    # sort by whether they match patterns
    def score(c):
        name = c.lower()
        return sum(1 for p in priority_patterns if re.search(p, name)) * -1  # earlier = higher priority
    numeric_cols.sort(key=score)
    return numeric_cols

def evaluate_combo(df: pd.DataFrame, features: list, target_col: str, test_size: float, random_state: int):
    X = df[features].astype(float)
    y = df[target_col].astype(float)
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].values
    y = y.loc[mask].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    m = regression_metrics(y_test, y_pred)
    return m, model, (X_test, y_test, y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="StudentPerformanceFactors.csv")
    parser.add_argument("--study-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--max-features", type=int, default=5, help="Try up to this many features in a combo")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.csv)
    # figure out baseline study and target
    _, y, s_col, t_col = select_xy(df, args.study_col, args.target_col)
    # merge back to data frame to ensure numeric types
    df = df.copy()
    df[t_col] = pd.to_numeric(df[t_col], errors='coerce')

    # candidates
    candidates = find_candidate_features(df, t_col)
    # ensure study column is included as a candidate
    if s_col not in candidates and s_col in df.columns:
        candidates = [s_col] + candidates

    results = []
    best = None
    best_details = None

    # Try combinations: start with 1 feature (study) then add up to max-features
    topK = min(len(candidates), args.max_features)
    for r in range(1, topK + 1):
        for combo in itertools.combinations(candidates[:args.max_features], r):
            try:
                m, model, details = evaluate_combo(df, list(combo), t_col, args.test_size, args.random_state)
                entry = {"features": list(combo), **m}
                results.append(entry)
                if best is None or m["R2"] > best["R2"]:
                    best = entry
                    best_details = (model, details)
            except Exception as e:
                # skip bad combos (e.g., constant columns)
                continue

    os.makedirs("outputs", exist_ok=True)
    res_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    res_df.to_csv("outputs/feature_experiments.csv", index=False)

    # Save best model + a plot
    if best is not None and best_details is not None:
        model, (X_test, y_test, y_pred) = best_details
        scatter_pred_vs_actual(y_test, y_pred, "Best Feature Combo: Predicted vs Actual", "outputs/feature_best_pred_vs_actual.png")
        dump(model, "outputs/feature_best_model.joblib")
        save_json(best, "outputs/feature_best_metrics.json")

    print("Saved: outputs/feature_experiments.csv (+ optional best model/artifacts)")
    if best is not None:
        print("Best combo:", best)

if __name__ == "__main__":
    main()
