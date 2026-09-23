"""Data loading and cleaning for the credit-default dataset.

Dataset: "Default of Credit Card Clients" (Yeh & Lien, 2009), UCI ML Repository id 350.
30,000 credit-card clients in Taiwan, 23 features, binary target (default next month).

This module is reused in every week of the project, so the "clean baseline"
is always produced the same way.
"""
from pathlib import Path

import pandas as pd

TARGET = "default"

FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]      # repayment status
    + [f"BILL_AMT{i}" for i in range(1, 7)]                        # bill statement amounts
    + [f"PAY_AMT{i}" for i in range(1, 7)]                         # amounts paid
)

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

_TARGET_NAMES = {"default payment next month", "default.payment.next.month", "y", "default"}


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Give every version of the dataset the same column names and order.

    Versions differ (original XLS, Kaggle CSV, ucimlrepo's X1..X23), but the
    column ORDER is identical in all of them, so features are renamed by position.
    """
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if str(c).strip().upper() == "ID"])
    df = df.rename(columns={c: TARGET for c in df.columns
                            if str(c).strip().lower() in _TARGET_NAMES})
    if TARGET not in df.columns:
        raise ValueError(f"Could not find the target column. Columns: {list(df.columns)}")
    feats = [c for c in df.columns if c != TARGET]
    if len(feats) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} features, found {len(feats)}: {feats}")
    df = df.rename(columns=dict(zip(feats, FEATURES)))
    return df[FEATURES + [TARGET]].apply(pd.to_numeric).astype("int64")


def load_raw(data_dir="data/raw") -> pd.DataFrame:
    """Load the raw dataset from data/raw/ (XLS or CSV), or download it with ucimlrepo."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.xls*")) + sorted(data_dir.glob("*.csv"))
    if files:
        f = files[0]
        # The original XLS has an extra title row above the real header.
        df = pd.read_excel(f, header=1) if f.suffix.startswith(".xls") else pd.read_csv(f)
        print(f"Loaded {f.name}")
        return _standardize(df)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise FileNotFoundError(
            "No data file in data/raw/ and ucimlrepo is not installed.\n"
            "Either `pip install ucimlrepo`, or download the dataset from\n"
            "https://archive.ics.uci.edu/dataset/350 and put the .xls file in data/raw/."
        ) from e

    ds = fetch_ucirepo(id=350)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "credit_default_uci.csv", index=False)   # cache for next time
    print("Downloaded from UCI and cached in data/raw/")
    return _standardize(df)


def clean(df: pd.DataFrame):
    """Fix documented category-code problems. Returns (clean_df, log).

    Every decision is logged, because in this project the cleaning choices
    ARE part of the data-quality story.
    """
    df = df.copy()
    log = []

    # EDUCATION is documented as 1=graduate school, 2=university, 3=high school,
    # 4=others. Codes 0, 5, 6 appear in the data but are undocumented.
    bad = ~df["EDUCATION"].isin([1, 2, 3, 4])
    if bad.any():
        log.append(f"EDUCATION: {bad.sum()} rows with undocumented codes "
                   f"{sorted(int(v) for v in df.loc[bad, 'EDUCATION'].unique())} -> mapped to 4 (others)")
        df.loc[bad, "EDUCATION"] = 4

    # MARRIAGE is documented as 1=married, 2=single, 3=others. Code 0 is undocumented.
    bad = ~df["MARRIAGE"].isin([1, 2, 3])
    if bad.any():
        log.append(f"MARRIAGE: {bad.sum()} rows with undocumented codes "
                   f"{sorted(int(v) for v in df.loc[bad, 'MARRIAGE'].unique())} -> mapped to 3 (others)")
        df.loc[bad, "MARRIAGE"] = 3

    # PAY_* is documented as -1=paid duly, 1..9=months delayed.
    # Values -2 and 0 also appear and are undocumented. They are common and
    # carry signal, so they are KEPT as-is and only reported.
    for col in PAY_STATUS:
        n = int(df[col].isin([-2, 0]).sum())
        log.append(f"{col}: {n} rows use undocumented codes -2/0 (kept as-is)")

    n_dup = int(df.duplicated().sum())
    log.append(f"Exact duplicate rows (ID removed): {n_dup} (kept; reported only)")
    return df, log


def load_clean(processed_path="data/processed/credit_clean.csv") -> pd.DataFrame:
    """Load the clean baseline saved in Week 1. Used from Week 2 onward."""
    return pd.read_csv(processed_path)
