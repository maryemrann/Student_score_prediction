
import argparse
import os
import pandas as pd
from utils import load_data, select_xy, hist_plot, corr_heatmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="StudentPerformanceFactor.csv")
    parser.add_argument("--study-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--head", type=int, default=10)
    args = parser.parse_args()

    df = load_data("StudentPerformanceFactors.csv")

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(args.head))

    # Try to locate study & target for quick visuals
    try:
        X, y, s_col, t_col = select_xy(df, args.study_col, args.target_col)
        os.makedirs("outputs", exist_ok=True)
        hist_plot(X[s_col], f"Distribution of {s_col}", "outputs/eda_hist_study.png")
        hist_plot(y, f"Distribution of {t_col}", "outputs/eda_hist_target.png")
    except Exception as e:
        print("Skipping study/target histograms:", e)

    # Correlation heatmap for numeric columns
    corr_heatmap(df, "outputs/eda_corr_heatmap.png")
    print("Saved EDA plots in outputs/: eda_hist_study.png, eda_hist_target.png, eda_corr_heatmap.png (where applicable).")

if __name__ == "__main__":
    main()
