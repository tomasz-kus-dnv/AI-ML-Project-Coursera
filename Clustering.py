# Early Fault Detection Model using SCADA Data

import glob
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Unsupervised learning imports
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Load your SCADA dataset ---
# Define the path to your CSV files
csv_folder_path = '../wind-turbines-data/setA/Wind Farm A/datasets'  # Update to your actual path
csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))

# List to hold individual DataFrames
dataframes = []

# Read each CSV
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

def downsample_every_nth_row(df, n=6):
    """Keep every nth row from the dataframe"""
    return df.iloc[::n].reset_index(drop=True)

combined_df_downsampled = downsample_every_nth_row(combined_df, n=18)
    
print(f"Downsampled data shape: {combined_df_downsampled.shape}")
print(f"Data reduction: {combined_df.shape[0] / combined_df_downsampled.shape[0]:.1f}x fewer rows")

# Use the downsampled data for further processing
combined_df = combined_df_downsampled.copy()

# Combine all DataFrames into one

eventInfo = pd.read_csv('../wind-turbines-data/setA/Wind Farm A/comma_event_info.csv')
fault_data = eventInfo[eventInfo['event_label'] == 'anomaly']

# --- Step 2: Add fault labels based on known fault events ---
df = combined_df.copy()
df['time_stamp'] = pd.to_datetime(df['time_stamp'])  # Convert once outside the loop

# Create arrays for both start and end times
fault_start_times = pd.to_datetime(fault_data['event_start'].values)
fault_end_times = pd.to_datetime(fault_data['event_end'].values)

df['fault_label'] = 0
for start_time, end_time in zip(fault_start_times, fault_end_times):
    # Label pre-fault window (7 days before start)
    # rows_to_keep = ~((df['time_stamp'] > start_time) & (df['time_stamp'] <= end_time))
    # df = df[rows_to_keep]
    pre_fault_window = (df['time_stamp'] >= start_time - pd.Timedelta(days=3)) & (df['time_stamp'] <= end_time)
    df.loc[pre_fault_window, 'fault_label'] = 1

# Drop rows with NaNs
df.dropna(inplace=True)

# --- Step 4: Prepare training and testing sets ---
X_train = df[df['train_test'] == 'train']
X_test = df[df['train_test'] == 'prediction']
y_train = X_train['fault_label']
y_test = X_test['fault_label']
columns_to_drop=['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id', 'fault_label']
X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)

# --- Step 5: Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 6: Train multiple models for comparison ---
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# === UNSUPERVISED LEARNING: PATTERN DETECTION ===
print("\n=== UNSUPERVISED LEARNING: DETECTING FUNCTIONING PATTERNS ===")

# Prepare data for unsupervised learning (using all data, not just training)
# Remove fault labels to make it truly unsupervised
unsupervised_data = df.drop(columns=['fault_label', 'time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id'])
unsupervised_data_scaled = StandardScaler().fit_transform(unsupervised_data)

# --- 1. CLUSTERING ANALYSIS ---
print("\n1. CLUSTERING ANALYSIS")

# K-Means Clustering
print("\nK-Means Clustering:")
optimal_k = 5  # You can use elbow method to find optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels_kmeans = kmeans.fit_predict(unsupervised_data_scaled)

# DBSCAN Clustering (density-based)
print("DBSCAN Clustering:")
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels_dbscan = dbscan.fit_predict(unsupervised_data_scaled)

# Gaussian Mixture Model
print("Gaussian Mixture Model:")
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
cluster_labels_gmm = gmm.fit_predict(unsupervised_data_scaled)

# Hierarchical Clustering
# print("Hierarchical Clustering:")
# hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
# cluster_labels_hierarchical = hierarchical.fit_predict(unsupervised_data_scaled)

# Evaluate clustering quality
clustering_results = {}
for name, labels in [('K-Means', cluster_labels_kmeans), 
                    ('GMM', cluster_labels_gmm)]: 
                    # ('Hierarchical', cluster_labels_hierarchical)]:
    if len(set(labels)) > 1:  # Only calculate if more than one cluster
        silhouette_avg = silhouette_score(unsupervised_data_scaled, labels)
        clustering_results[name] = {
            'n_clusters': len(set(labels)),
            'silhouette_score': silhouette_avg
        }
        print(f"{name}: {len(set(labels))} clusters, Silhouette Score: {silhouette_avg:.4f}")

# DBSCAN results
n_clusters_dbscan = len(set(cluster_labels_dbscan)) - (1 if -1 in cluster_labels_dbscan else 0)
n_noise_dbscan = list(cluster_labels_dbscan).count(-1)
print(f"DBSCAN: {n_clusters_dbscan} clusters, {n_noise_dbscan} noise points")

# --- 2. ANOMALY DETECTION ---
print("\n2. ANOMALY DETECTION")

# Isolation Forest
print("Isolation Forest:")
iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
anomaly_labels_iso = iso_forest.fit_predict(unsupervised_data_scaled)
n_anomalies_iso = (anomaly_labels_iso == -1).sum()
print(f"Detected {n_anomalies_iso} anomalies ({n_anomalies_iso/len(anomaly_labels_iso)*100:.2f}%)")

# Local Outlier Factor
print("Local Outlier Factor:")
lof = LocalOutlierFactor(contamination=0.1, n_jobs=-1)
anomaly_labels_lof = lof.fit_predict(unsupervised_data_scaled)
n_anomalies_lof = (anomaly_labels_lof == -1).sum()
print(f"Detected {n_anomalies_lof} anomalies ({n_anomalies_lof/len(anomaly_labels_lof)*100:.2f}%)")

# --- ANOMALY DETECTION ACCURACY EVALUATION ---
print("\n=== ANOMALY DETECTION ACCURACY EVALUATION ===")

# Convert anomaly predictions to binary format (1 = anomaly, 0 = normal)
iso_forest_predictions = (anomaly_labels_iso == -1).astype(int)
lof_predictions = (anomaly_labels_lof == -1).astype(int)

# Create ensemble anomaly prediction (anomaly if detected by either method)
ensemble_predictions = ((iso_forest_predictions == 1) | (lof_predictions == 1)).astype(int)

# Get true fault labels for comparison
true_labels = df['fault_label'].values

# Ensure arrays have same length
min_length = min(len(true_labels), len(iso_forest_predictions))
true_labels = true_labels[:min_length]
iso_forest_predictions = iso_forest_predictions[:min_length]
lof_predictions = lof_predictions[:min_length]
ensemble_predictions = ensemble_predictions[:min_length]

print(f"Evaluating {min_length} samples for anomaly detection accuracy")
print(f"True fault labels: {true_labels.sum()} faults ({true_labels.mean()*100:.2f}%)")

# Import metrics for evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_anomaly_detection(y_true, y_pred, method_name):
    """Evaluate anomaly detection performance"""
    print(f"\n{method_name} Performance:")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives: {tn}")
    print(f"    False Positives: {fp}")
    print(f"    False Negatives: {fn}")
    print(f"    True Positives: {tp}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fault']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'balanced_accuracy': balanced_accuracy,
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }

# Evaluate each anomaly detection method
iso_results = evaluate_anomaly_detection(true_labels, iso_forest_predictions, "Isolation Forest")
lof_results = evaluate_anomaly_detection(true_labels, lof_predictions, "Local Outlier Factor")
ensemble_results = evaluate_anomaly_detection(true_labels, ensemble_predictions, "Ensemble (ISO + LOF)")

# Summary comparison
print("\n=== ANOMALY DETECTION METHODS COMPARISON ===")
comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']

comparison_df = pd.DataFrame({
    'Isolation_Forest': [iso_results[metric] for metric in comparison_metrics],
    'Local_Outlier_Factor': [lof_results[metric] for metric in comparison_metrics],
    'Ensemble': [ensemble_results[metric] for metric in comparison_metrics]
}, index=comparison_metrics)

print("\nMetrics Comparison:")
print(comparison_df.round(4))

# Find best performing method
best_method = {}
for metric in comparison_metrics:
    values = comparison_df.loc[metric]
    best_method[metric] = values.idxmax()

print("\nBest performing method for each metric:")
for metric, method in best_method.items():
    print(f"  {metric}: {method} ({comparison_df.loc[metric, method]:.4f})")

# Analysis insights
print("\n=== ANOMALY DETECTION INSIGHTS ===")

# Calculate detection rates
iso_detection_rate = iso_forest_predictions.sum() / len(iso_forest_predictions)
lof_detection_rate = lof_predictions.sum() / len(lof_predictions)
ensemble_detection_rate = ensemble_predictions.sum() / len(ensemble_predictions)
true_fault_rate = true_labels.sum() / len(true_labels)

print(f"Detection rates:")
print(f"  True fault rate: {true_fault_rate:.4f}")
print(f"  Isolation Forest: {iso_detection_rate:.4f}")
print(f"  Local Outlier Factor: {lof_detection_rate:.4f}")
print(f"  Ensemble: {ensemble_detection_rate:.4f}")

# Check for over/under detection
print(f"\nDetection Analysis:")
if iso_detection_rate > true_fault_rate * 2:
    print("  ⚠️  Isolation Forest may be over-detecting anomalies")
elif iso_detection_rate < true_fault_rate * 0.5:
    print("  ⚠️  Isolation Forest may be under-detecting anomalies")
else:
    print("  ✅ Isolation Forest detection rate is reasonable")

if lof_detection_rate > true_fault_rate * 2:
    print("  ⚠️  Local Outlier Factor may be over-detecting anomalies")
elif lof_detection_rate < true_fault_rate * 0.5:
    print("  ⚠️  Local Outlier Factor may be under-detecting anomalies")
else:
    print("  ✅ Local Outlier Factor detection rate is reasonable")

# Save evaluation results
evaluation_results = {
    'evaluation_summary': {
        'total_samples': int(min_length),
        'true_faults': int(true_labels.sum()),
        'true_fault_rate': float(true_fault_rate)
    },
    'isolation_forest': iso_results,
    'local_outlier_factor': lof_results,
    'ensemble_method': ensemble_results,
    'best_methods': best_method,
    'detection_rates': {
        'isolation_forest': float(iso_detection_rate),
        'local_outlier_factor': float(lof_detection_rate),
        'ensemble': float(ensemble_detection_rate)
    }
}

# Save to file
import json
with open('anomaly_detection_evaluation.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print(f"\nEvaluation results saved to 'anomaly_detection_evaluation.json'")

# Save detailed predictions for further analysis
predictions_df = pd.DataFrame({
    'true_label': true_labels,
    'isolation_forest_prediction': iso_forest_predictions,
    'lof_prediction': lof_predictions,
    'ensemble_prediction': ensemble_predictions,
    'iso_correct': (true_labels == iso_forest_predictions).astype(int),
    'lof_correct': (true_labels == lof_predictions).astype(int),
    'ensemble_correct': (true_labels == ensemble_predictions).astype(int)
})

predictions_df.to_csv('anomaly_detection_predictions.csv', index=False)
print("Detailed predictions saved to 'anomaly_detection_predictions.csv'")

print("\n" + "="*60)
print("ANOMALY DETECTION ACCURACY EVALUATION COMPLETE")
print("="*60)
