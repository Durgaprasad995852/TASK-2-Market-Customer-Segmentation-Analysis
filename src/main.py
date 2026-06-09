from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.cluster_analysis import profile_clusters
    from src.data_preprocessing import load_data
    from src.pca_visualization import create_pca_plot
    from src.segmentation import perform_clustering, plot_elbow_method
else:
    from .cluster_analysis import profile_clusters
    from .data_preprocessing import load_data
    from .pca_visualization import create_pca_plot
    from .segmentation import perform_clustering, plot_elbow_method


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mall_customers.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customers = load_data(DATA_PATH)
    suggested_k = plot_elbow_method(customers[["Age", "Annual Income Numeric", "Spending Score"]], OUTPUT_DIR)
    clustered, model, scaler, best_k = perform_clustering(customers, OUTPUT_DIR, n_clusters=4)

    clustered_output = OUTPUT_DIR / "clustered_customers.csv"
    clustered.to_csv(clustered_output, index=False)

    pca_df = create_pca_plot(clustered, OUTPUT_DIR)
    pca_df.to_csv(OUTPUT_DIR / "pca_coordinates.csv", index=False)

    report = profile_clusters(clustered, OUTPUT_DIR)

    print("Market & Customer Segmentation Analysis")
    print(f"Data source: {DATA_PATH}")
    print(f"Elbow suggestion: {suggested_k}")
    print(f"Chosen clusters: {best_k}")
    print()
    print("Preview of cleaned data:")
    try:
        print(clustered[["Customer Name", "Annual Income (₹)", "Spending Score", "Cluster"]].head().to_string(index=False))
    except UnicodeEncodeError:
        temp_df = clustered[["Customer Name", "Annual Income (₹)", "Spending Score", "Cluster"]].head().copy()
        temp_df.columns = ["Customer Name", "Annual Income (INR)", "Spending Score", "Cluster"]
        temp_df["Annual Income (INR)"] = temp_df["Annual Income (INR)"].astype(str).str.replace("₹", "INR")
        print(temp_df.to_string(index=False))
    print()
    print("Cluster summary:")
    try:
        print(report.to_string(index=False))
    except UnicodeEncodeError:
        temp_report = report.copy()
        if "Avg_Income" in temp_report.columns:
            temp_report["Avg_Income"] = temp_report["Avg_Income"].astype(str).str.replace("₹", "INR")
        print(temp_report.to_string(index=False))
    print()
    print(f"Saved: {clustered_output}")
    print(f"Saved: {OUTPUT_DIR / 'cluster_report.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'elbow_method.png'}")
    print(f"Saved: {OUTPUT_DIR / 'pca_clusters.png'}")
    print(f"Saved: {OUTPUT_DIR / 'kmeans_model.pkl'}")


if __name__ == "__main__":
    main()
