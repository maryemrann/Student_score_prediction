import argparse
import os
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils import load_data, select_xy, regression_metrics, scatter_with_line, scatter_pred_vs_actual, residual_plot, save_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="StudentPerformanceFactors.csv")
    parser.add_argument("--study-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.csv)
    X, y, s_col, t_col = select_xy(df, args.study_col, args.target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=args.test_size, random_state=args.random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    os.makedirs("outputs", exist_ok=True)
    save_json(metrics, "outputs/linear_metrics.json")
    dump(model, "outputs/linear_model.joblib")

    # Plot study vs target with fitted line (using test set range for visualization)
    x_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    scatter_with_line(X_test.reshape(-1, 1), y_test.reshape(-1, 1), y_pred.reshape(-1, 1),
                      xlabel=s_col, ylabel=t_col,
                      title="Study Hours vs Score (Test)",
                      out_path="outputs/linear_study_vs_score.png")

    # Predicted vs Actual
    scatter_pred_vs_actual(y_test, y_pred, "Linear Regression: Predicted vs Actual", "outputs/linear_pred_vs_actual.png")

    # Residuals
    residual_plot(y_test, y_pred, "Linear Regression Residuals", "outputs/linear_residuals.png")

    print("Saved:")
    print(" - outputs/linear_metrics.json")
    print(" - outputs/linear_model.joblib")
    print(" - outputs/linear_study_vs_score.png")
    print(" - outputs/linear_pred_vs_actual.png")
    print(" - outputs/linear_residuals.png")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
