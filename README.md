# Market & Customer Segmentation Analysis

This project groups customers by purchasing habits to support targeted marketing campaigns. It uses the Mall Customers dataset style, with Indian customer names and INR-formatted annual income values.

## What it covers

- RFM-style business thinking for customer targeting
- Dimensionality reduction with PCA
- Cluster profiling with KMeans
- Indian currency formatting for annual income

## Project Structure

```text
TASK-2 CLUSTER/
├── data/
│   └── mall_customers.csv
├── models/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── cluster_analysis.py
│   ├── data_preprocessing.py
│   ├── main.py
│   ├── pca_visualization.py
│   └── segmentation.py
├── requirements.txt
└── README.md
```

## How to run

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Run the project:

```bash
python src/main.py
```

## Output files

- `outputs/clustered_customers.csv`
- `outputs/cluster_report.csv`
- `outputs/elbow_method.png`
- `outputs/pca_coordinates.csv`
- `outputs/pca_clusters.png`
- `outputs/kmeans_model.pkl`

## Output images
<img width="1920" height="1011" alt="image" src="https://github.com/user-attachments/assets/b94b8507-634f-44e8-8284-8bf59b358b36" />


## Dataset format

The annual income column is stored in INR format like `₹1,20,000`. The preprocessing step converts it to a numeric value for clustering and keeps the formatted version for reporting.
