from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data_preprocessing import feature_matrix


def create_pca_plot(df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    features = feature_matrix(df)
    scaled = StandardScaler().fit_transform(features)

    pca = PCA(n_components=2, random_state=42)
    transformed = pca.fit_transform(scaled)

    pca_df = pd.DataFrame(transformed, columns=["PCA 1", "PCA 2"])
    pca_df["Cluster"] = df["Cluster"].values
    pca_df["Customer Name"] = df["Customer Name"].values

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        pca_df["PCA 1"],
        pca_df["PCA 2"],
        c=pca_df["Cluster"],
        cmap="tab10",
        s=80,
        edgecolor="white",
        linewidth=0.6,
    )
    plt.title("Customer Segments in PCA Space")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(alpha=0.2)
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_path / "pca_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    return pca_df
