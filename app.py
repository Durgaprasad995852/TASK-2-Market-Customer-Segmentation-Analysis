from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path
from werkzeug.utils import secure_filename

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mall_customers.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

app = Flask(__name__)

# Cache for dynamic data reloading
data_cache = {}
last_modified_timestamp = 0.0

def reload_resources_if_needed() -> None:
    """Reloads dataset and reports if files have changed on disk."""
    global last_modified_timestamp, data_cache
    
    customers_csv = OUTPUT_DIR / "clustered_customers.csv"
    report_csv = OUTPUT_DIR / "cluster_report.csv"
    pca_csv = OUTPUT_DIR / "pca_coordinates.csv"
    model_pkl = OUTPUT_DIR / "kmeans_model.pkl"
    
    csv_mtime = customers_csv.stat().st_mtime if customers_csv.exists() else 0.0
    report_mtime = report_csv.stat().st_mtime if report_csv.exists() else 0.0
    model_mtime = model_pkl.stat().st_mtime if model_pkl.exists() else 0.0
    current_stamp = csv_mtime + report_mtime + model_mtime
    
    if current_stamp != last_modified_timestamp or not data_cache:
        if customers_csv.exists() and report_csv.exists():
            try:
                # Load clustered customers
                df_cust = pd.read_csv(customers_csv)
                
                # Load report
                df_rep = pd.read_csv(report_csv)
                
                # Load PCA coordinates
                df_pca = pd.read_csv(pca_csv) if pca_csv.exists() else pd.DataFrame()
                
                # Map segments to clusters
                segment_mapping = dict(zip(df_rep["Cluster"].astype(int), df_rep["Segment Name"]))
                
                # Add segment names to customer dataframe
                df_cust["Segment Name"] = df_cust["Cluster"].astype(int).map(segment_mapping)
                
                # Calculate summary stats
                total_customers = len(df_cust)
                avg_age = float(df_cust["Age"].mean())
                avg_spending = float(df_cust["Spending Score"].mean())
                avg_income = float(df_cust["Annual Income Numeric"].mean())
                
                # Gender counts
                gender_counts = df_cust["Gender"].value_counts().to_dict()
                
                # Prepare clusters for UI
                clusters_list = []
                for _, row in df_rep.iterrows():
                    c_id = int(row["Cluster"])
                    clusters_list.append({
                        "cluster_id": c_id,
                        "customers_count": int(row["Customers"]),
                        "avg_age": float(row["Avg_Age"]),
                        "avg_income": str(row["Avg_Income"]),
                        "avg_spending": float(row["Avg_Spending_Score"]),
                        "segment_name": str(row["Segment Name"]),
                        "names": [n.strip() for n in str(row["Customer Names"]).split(",")]
                    })
                
                # Prepare PCA coordinates
                pca_data = []
                if not df_pca.empty:
                    for _, row in df_pca.iterrows():
                        c_id = int(row["Cluster"])
                        pca_data.append({
                            "x": float(row["PCA 1"]),
                            "y": float(row["PCA 2"]),
                            "cluster_id": c_id,
                            "segment_name": segment_mapping.get(c_id, f"Cluster {c_id}"),
                            "name": str(row["Customer Name"])
                        })
                
                data_cache = {
                    "total_customers": total_customers,
                    "avg_age": round(avg_age, 1),
                    "avg_spending": round(avg_spending, 1),
                    "avg_income": round(avg_income, 2),
                    "gender_counts": gender_counts,
                    "customers": df_cust.to_dict(orient="records"),
                    "clusters": clusters_list,
                    "pca_data": pca_data,
                    "segment_mapping": {str(k): v for k, v in segment_mapping.items()}
                }
                last_modified_timestamp = current_stamp
            except Exception as e:
                print(f"Error loading resources: {e}", file=sys.stderr)
                data_cache = {"error": f"Failed to load data outputs: {str(e)}"}
        else:
            data_cache = {"error": "Outputs not found. Please run the clustering pipeline."}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/stats")
def get_stats():
    reload_resources_if_needed()
    return jsonify(data_cache)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        req_data = request.get_json() or {}
        name = req_data.get("name", "").strip() or "Guest"
        gender = req_data.get("gender", "Male")
        age = float(req_data.get("age", 30))
        income = float(req_data.get("income", 500000))
        spending_score = float(req_data.get("spending_score", 50))
        
        model_path = OUTPUT_DIR / "kmeans_model.pkl"
        if not model_path.exists():
            return jsonify({"error": "Model has not been trained yet."}), 400
            
        # Load K-Means model and scaler
        saved = joblib.load(model_path)
        scaler = saved["scaler"]
        model = saved["model"]
        
        # Scale inputs and predict
        scaled = scaler.transform([[age, income, spending_score]])
        predicted_cluster = int(model.predict(scaled)[0])
        
        # Get explanation (proximity to each cluster center)
        distances = model.transform(scaled)[0]
        # Calculate matching percentage (closer = higher percentage)
        similarities = [1.0 / (1.0 + d) for d in distances]
        sum_sim = sum(similarities)
        similarities_pct = [(s / sum_sim) * 100 for s in similarities]
        
        # Load segment name
        reload_resources_if_needed()
        segment_mapping = data_cache.get("segment_mapping", {})
        predicted_segment = segment_mapping.get(str(predicted_cluster), f"Cluster {predicted_cluster}")
        
        # Prepare match probabilities
        match_scores = []
        for c_idx, pct in enumerate(similarities_pct):
            seg_name = segment_mapping.get(str(c_idx), f"Cluster {c_idx}")
            match_scores.append({
                "cluster_id": c_idx,
                "segment_name": seg_name,
                "match_percentage": round(pct, 1)
            })
            
        # Sort by match percentage descending
        match_scores.sort(key=lambda x: x["match_percentage"], reverse=True)
            
        return jsonify({
            "name": name,
            "predicted_cluster": predicted_cluster,
            "predicted_segment": predicted_segment,
            "match_scores": match_scores,
            "input_data": {
                "age": age,
                "income": income,
                "spending_score": spending_score
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith(".csv"):
        try:
            # Secure filepath and save
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            file.save(DATA_PATH)
            
            # Execute cluster pipeline main()
            # We add root directory to system path to import src.main
            if str(BASE_DIR) not in sys.path:
                sys.path.append(str(BASE_DIR))
                
            from src.main import main as run_pipeline
            run_pipeline()
            
            # Reset cache to force reload
            global last_modified_timestamp
            last_modified_timestamp = 0.0
            reload_resources_if_needed()
            
            return jsonify({"success": True, "message": "Dataset uploaded and clustering completed successfully."})
        except Exception as e:
            return jsonify({"error": f"Error running clustering pipeline: {str(e)}"}), 500
    return jsonify({"error": "Invalid file format. Only CSV files are supported."}), 400

@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    # Ensure reports exist before starting server
    if not (OUTPUT_DIR / "clustered_customers.csv").exists():
        print("Initial clustering files not found. Running pipeline first...")
        if str(BASE_DIR) not in sys.path:
            sys.path.append(str(BASE_DIR))
        from src.main import main as run_pipeline
        run_pipeline()
        
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
