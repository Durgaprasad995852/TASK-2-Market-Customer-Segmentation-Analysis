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
<img width="1920" height="1073" alt="image" src="https://github.com/user-attachments/assets/e435dfcc-12f4-4b3d-a087-3ceab948f73e" />
<img width="1920" height="1178" alt="image" src="https://github.com/user-attachments/assets/12daf29f-0c95-4745-a154-f0b488071326" />
<img width="1920" height="1473" alt="image" src="https://github.com/user-attachments/assets/6625bbe5-5f09-4c70-b547-9dd443720da1" />
<img width="1920" height="1253" alt="image" src="https://github.com/user-attachments/assets/f2d00781-cf40-4ffe-a8ce-3f7ba945f5ec" />






## Dataset format

The annual income column is stored in INR format like `₹1,20,000`. The preprocessing step converts it to a numeric value for clustering and keeps the formatted version for reporting.
