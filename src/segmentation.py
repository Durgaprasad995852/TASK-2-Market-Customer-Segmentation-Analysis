from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data_preprocessing import feature_matrix


def find_best_k(features: pd.DataFrame, min_k: int = 2, max_k: int = 8) -> tuple[int, list[float], list[float]]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(scaled, labels)))

    best_index = max(range(len(silhouettes)), key=silhouettes.__getitem__)
    best_k = min_k + best_index
    return best_k, inertias, silhouettes


def plot_elbow_method(features: pd.DataFrame, output_dir: str | Path, min_k: int = 2, max_k: int = 8) -> int:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_k, inertias, _ = find_best_k(features, min_k=min_k, max_k=max_k)
    cluster_range = list(range(min_k, max_k + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertias, marker="o", linewidth=2)
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.xticks(cluster_range)
    plt.grid(alpha=0.25)
    plt.axvline(best_k, color="crimson", linestyle="--", alpha=0.7, label=f"Suggested k = {best_k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "elbow_method.png", dpi=300, bbox_inches="tight")
    plt.close()

    return best_k


def perform_clustering(df: pd.DataFrame, output_dir: str | Path, n_clusters: int | None = None) -> tuple[pd.DataFrame, KMeans, StandardScaler, int]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    features = feature_matrix(df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    if n_clusters is None:
        n_clusters, _, _ = find_best_k(features)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)

    clustered = df.copy()
    clustered["Cluster"] = labels

    model_path = output_path / "kmeans_model.pkl"
    joblib.dump({"model": model, "scaler": scaler, "feature_columns": list(features.columns)}, model_path)

    return clustered, model, scaler, n_clusters
