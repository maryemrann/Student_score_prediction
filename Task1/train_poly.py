import argparse
import os
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from utils import load_data, select_xy, regression_metrics, scatter_pred_vs_actual, residual_plot, save_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="StudentPerformanceFactors.csv")
    parser.add_argument("--study-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.csv)
    X, y, s_col, t_col = select_xy(df, args.study_col, args.target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=args.test_size, random_state=args.random_state
    )

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=args.degree, include_bias=False)),
        ("linreg", LinearRegression())
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    os.makedirs("outputs", exist_ok=True)
    save_json(metrics, "outputs/poly_metrics.json")
    dump(model, "outputs/poly_model_degree%d.joblib" % args.degree)

    scatter_pred_vs_actual(y_test, y_pred, f"Polynomial (degree={args.degree}) Predicted vs Actual", "outputs/poly_pred_vs_actual.png")
    residual_plot(y_test, y_pred, f"Polynomial (degree={args.degree}) Residuals", "outputs/poly_residuals.png")

    print("Saved:")
    print(" - outputs/poly_metrics.json")
    print(" - outputs/poly_model_degree%d.joblib" % args.degree)
    print(" - outputs/poly_pred_vs_actual.png")
    print(" - outputs/poly_residuals.png")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
