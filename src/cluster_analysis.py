from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_preprocessing import format_inr


SEGMENT_NAMES = [
    "Champions",
    "Potential Loyalists",
    "Growth Opportunities",
    "Budget / Nurture",
    "At Risk / Needs Attention",
]


def profile_clusters(df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    grouped = (
        df.groupby("Cluster")
        .agg(
            Customers=("Customer Name", "count"),
            **{"Customer Names": ("Customer Name", lambda values: ", ".join(values))},
            Avg_Age=("Age", "mean"),
            Avg_Income=("Annual Income Numeric", "mean"),
            Avg_Spending_Score=("Spending Score", "mean"),
        )
        .reset_index()
        .sort_values(["Avg_Spending_Score", "Avg_Income"], ascending=False)
        .reset_index(drop=True)
    )

    grouped["Segment Name"] = [
        SEGMENT_NAMES[index] if index < len(SEGMENT_NAMES) else f"Segment {index + 1}"
        for index in range(len(grouped))
    ]
    grouped["Avg_Income"] = grouped["Avg_Income"].apply(format_inr)
    grouped["Avg_Age"] = grouped["Avg_Age"].round(1)
    grouped["Avg_Spending_Score"] = grouped["Avg_Spending_Score"].round(1)

    output_file = output_path / "cluster_report.csv"
    grouped.to_csv(output_file, index=False)
    return grouped
