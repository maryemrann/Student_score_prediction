import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict

# ---------- Column guessing & cleaning ----------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r'[^0-9a-zA-Z]+', '_', c.strip().lower()).strip('_')
        for c in df.columns
    ]
    return df

def _guess_from_patterns(columns: List[str], patterns: List[str]) -> Optional[str]:
    for col in columns:
        for pat in patterns:
            if re.search(pat, col, flags=re.IGNORECASE):
                return col
    return None

def guess_study_col(columns: List[str]) -> Optional[str]:
    patterns = [
        r'\bhours?_?\s*stud(y|ies|ied)?\b',
        r'\bstudy_?hours?\b',
        r'\bhours?\b.*\bstud(y|ies|ied)\b',
        r'\bstud(y|ies|ied)\b'
    ]
    return _guess_from_patterns(columns, patterns)

def guess_target_col(columns: List[str]) -> Optional[str]:
    patterns = [
        r'\b(performance|exam|final|total)\s*_?(score|index|marks|result)\b',
        r'\bscore\b',
        r'\bmarks?\b',
        r'\bgrade[s]?\b',
        r'\bg3\b'   # classic UCI dataset
    ]
    return _guess_from_patterns(columns, patterns)

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')

# ---------- Data loading & slicing ----------

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = standardize_columns(df)
    return df

def select_xy(df: pd.DataFrame, study_col: Optional[str], target_col: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, str, str]:
    cols = list(df.columns)
    s_col = study_col if study_col else guess_study_col(cols)
    t_col = target_col if target_col else guess_target_col(cols)
    if s_col is None or t_col is None:
        raise ValueError(
            "Could not auto-detect columns. "
            f"Available columns: {cols}. "
            "Please pass --study-col and --target-col explicitly."
        )
    x = to_numeric(df[s_col]).to_frame(name=s_col)
    y = to_numeric(df[t_col])
    # Drop rows with missing in X or y
    mask = x[s_col].notna() & y.notna()
    x = x.loc[mask]
    y = y.loc[mask]
    return x, y, s_col, t_col

# ---------- Metrics & plotting ----------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    # R2
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def scatter_with_line(x: np.ndarray, y: np.ndarray, y_line: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x, y)
    # line: ensure sorted by x
    order = np.argsort(x.reshape(-1))
    plt.plot(x.reshape(-1)[order], y_line.reshape(-1)[order])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def scatter_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(y_true, y_pred)
    # reference diagonal
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def residual_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str):
    import matplotlib.pyplot as plt
    plt.figure()
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def hist_plot(series: pd.Series, title: str, out_path: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(series.dropna())
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def corr_heatmap(df: pd.DataFrame, out_path: str, title: str = "Correlation Heatmap"):
    import matplotlib.pyplot as plt
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return
    corr = numeric_df.corr()
    plt.figure()
    im = plt.imshow(corr.values, aspect='auto')
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
