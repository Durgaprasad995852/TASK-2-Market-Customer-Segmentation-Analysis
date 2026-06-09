from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INCOME_COLUMNS = ["Annual Income (₹)", "Annual Income", "Income"]
REQUIRED_COLUMNS = ["CustomerID", "Customer Name", "Gender", "Age", "Spending Score"]


def _strip_currency(value) -> float:
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()
    text = text.replace("₹", "").replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text:
        return float("nan")
    return float(text)


def format_inr(value: float) -> str:
    number = int(round(float(value)))
    sign = "-" if number < 0 else ""
    digits = str(abs(number))

    if len(digits) <= 3:
        formatted = digits
    else:
        tail = digits[-3:]
        head = digits[:-3]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        formatted = ",".join(groups + [tail])

    return f"{sign}₹{formatted}"


def _resolve_income_column(df: pd.DataFrame) -> str:
    for column in INCOME_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"Missing income column. Expected one of: {', '.join(INCOME_COLUMNS)}")


def load_data(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [column.strip() for column in df.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    income_column = _resolve_income_column(df)
    numeric_income = df[income_column].apply(_strip_currency)

    cleaned = df.copy()
    cleaned["Annual Income Numeric"] = numeric_income
    cleaned["Annual Income (₹)"] = cleaned[income_column].apply(lambda value: format_inr(_strip_currency(value)))
    cleaned["Customer Name"] = cleaned["Customer Name"].astype(str).str.strip()
    cleaned["Gender"] = cleaned["Gender"].astype(str).str.strip()
    cleaned["Age"] = pd.to_numeric(cleaned["Age"], errors="coerce")
    cleaned["Spending Score"] = pd.to_numeric(cleaned["Spending Score"], errors="coerce")

    if cleaned[["Customer Name", "Gender", "Age", "Annual Income Numeric", "Spending Score"]].isna().any().any():
        raise ValueError("Input data contains invalid or missing values after preprocessing.")

    return cleaned[["CustomerID", "Customer Name", "Gender", "Age", "Annual Income (₹)", "Annual Income Numeric", "Spending Score"]]


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Age", "Annual Income Numeric", "Spending Score"]].copy()
