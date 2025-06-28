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

# Combine all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

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
print("Hierarchical Clustering:")
hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
cluster_labels_hierarchical = hierarchical.fit_predict(unsupervised_data_scaled)

# Evaluate clustering quality
clustering_results = {}
for name, labels in [('K-Means', cluster_labels_kmeans), 
                    ('GMM', cluster_labels_gmm), 
                    ('Hierarchical', cluster_labels_hierarchical)]:
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

# --- 3. DIMENSIONALITY REDUCTION FOR VISUALIZATION ---
print("\n3. DIMENSIONALITY REDUCTION & PATTERN VISUALIZATION")

# PCA for understanding data structure
pca = PCA(n_components=2)
data_pca = pca.fit_transform(unsupervised_data_scaled)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# t-SNE for non-linear pattern detection (use subset for speed)
if len(unsupervised_data_scaled) > 5000:
    subset_indices = np.random.choice(len(unsupervised_data_scaled), 5000, replace=False)
    tsne_data = unsupervised_data_scaled[subset_indices]
    tsne_kmeans_labels = cluster_labels_kmeans[subset_indices]
else:
    tsne_data = unsupervised_data_scaled
    tsne_kmeans_labels = cluster_labels_kmeans

print("Computing t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
data_tsne = tsne.fit_transform(tsne_data)

# --- 4. PATTERN ANALYSIS ---
print("\n4. FUNCTIONING PATTERN ANALYSIS")

# Add cluster labels back to original dataframe for analysis
df_with_clusters = df.copy()
df_with_clusters['kmeans_cluster'] = cluster_labels_kmeans
df_with_clusters['gmm_cluster'] = cluster_labels_gmm
df_with_clusters['iso_forest_anomaly'] = (anomaly_labels_iso == -1).astype(int)
df_with_clusters['lof_anomaly'] = (anomaly_labels_lof == -1).astype(int)

# Analyze cluster characteristics
print("\nCluster Characteristics (K-Means):")
feature_cols = [col for col in unsupervised_data.columns]
for cluster_id in range(optimal_k):
    cluster_data = df_with_clusters[df_with_clusters['kmeans_cluster'] == cluster_id]
    if len(cluster_data) > 0:
        print(f"\nCluster {cluster_id} ({len(cluster_data)} samples):")
        # Show mean values of key features for this cluster
        key_features = feature_cols[:5] if len(feature_cols) >= 5 else feature_cols
        for feature in key_features:
            if feature in cluster_data.columns:
                mean_val = cluster_data[feature].mean()
                std_val = cluster_data[feature].std()
                print(f"  {feature}: {mean_val:.2f} ± {std_val:.2f}")

# Correlation between clusters and known fault labels
if 'fault_label' in df.columns:
    print("\nCorrelation between clusters and known fault patterns:")
    for cluster_id in range(optimal_k):
        cluster_mask = df_with_clusters['kmeans_cluster'] == cluster_id
        fault_rate = df_with_clusters[cluster_mask]['fault_label'].mean()
        print(f"Cluster {cluster_id}: {fault_rate:.4f} fault rate")

# --- 5. SAVE RESULTS ---
print("\n5. SAVING UNSUPERVISED LEARNING RESULTS")

# Create a summary of patterns discovered
pattern_summary = {
    'clustering_results': clustering_results,
    'kmeans_clusters': len(set(cluster_labels_kmeans)),
    'dbscan_clusters': n_clusters_dbscan,
    'dbscan_noise_points': n_noise_dbscan,
    'isolation_forest_anomalies': n_anomalies_iso,
    'lof_anomalies': n_anomalies_lof,
    'pca_variance_explained': pca.explained_variance_ratio_.sum()
}

print("Pattern Discovery Summary:")
for key, value in pattern_summary.items():
    print(f"  {key}: {value}")

# Save detailed results to CSV
df_with_clusters.to_csv('wind_turbine_patterns.csv', index=False)
print("\nDetailed results saved to 'wind_turbine_patterns.csv'")

# Save cluster centroids for future reference
cluster_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=unsupervised_data.columns)
cluster_centroids.to_csv('cluster_centroids.csv', index=False)
print("Cluster centroids saved to 'cluster_centroids.csv'")

# --- 6. FEATURE IMPORTANCE FOR CLUSTERING ---
print("\n6. FEATURE IMPORTANCE FOR PATTERN DETECTION")

# Calculate feature importance based on variance across clusters
feature_importance_clustering = {}
for i, feature in enumerate(unsupervised_data.columns):
    cluster_means = []
    for cluster_id in range(optimal_k):
        cluster_mask = cluster_labels_kmeans == cluster_id
        if cluster_mask.sum() > 0:
            cluster_means.append(unsupervised_data_scaled[cluster_mask, i].mean())
    
    if len(cluster_means) > 1:
        feature_variance = np.var(cluster_means)
        feature_importance_clustering[feature] = feature_variance

# Sort features by importance
sorted_features = sorted(feature_importance_clustering.items(), key=lambda x: x[1], reverse=True)
print("\nTop features for distinguishing patterns:")
for feature, importance in sorted_features[:10]:
    print(f"  {feature}: {importance:.4f}")

print("\n=== UNSUPERVISED LEARNING COMPLETE ===")

# --- 7. VISUALIZATION FUNCTIONS ---
def create_pattern_visualizations():
    """Create visualizations for the detected patterns"""
    print("\n7. CREATING PATTERN VISUALIZATIONS")
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Wind Turbine Functioning Patterns Analysis', fontsize=16)
        
        # Plot 1: PCA with K-Means clusters
        scatter1 = axes[0, 0].scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_labels_kmeans, 
                                     cmap='viridis', alpha=0.6, s=20)
        axes[0, 0].set_title('PCA with K-Means Clusters')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # Plot 2: PCA with Anomalies (Isolation Forest)
        colors = ['blue' if x == 1 else 'red' for x in anomaly_labels_iso]
        axes[0, 1].scatter(data_pca[:, 0], data_pca[:, 1], c=colors, alpha=0.6, s=20)
        axes[0, 1].set_title('PCA with Isolation Forest Anomalies')
        axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        
        # Plot 3: t-SNE with K-Means clusters
        if len(data_tsne) > 0:
            scatter3 = axes[0, 2].scatter(data_tsne[:, 0], data_tsne[:, 1], 
                                         c=tsne_kmeans_labels, cmap='viridis', alpha=0.6, s=20)
            axes[0, 2].set_title('t-SNE with K-Means Clusters')
            axes[0, 2].set_xlabel('t-SNE 1')
            axes[0, 2].set_ylabel('t-SNE 2')
            plt.colorbar(scatter3, ax=axes[0, 2])
        
        # Plot 4: Cluster size distribution
        cluster_counts = pd.Series(cluster_labels_kmeans).value_counts().sort_index()
        axes[1, 0].bar(cluster_counts.index, cluster_counts.values)
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Samples')
        
        # Plot 5: Feature importance for clustering
        if len(sorted_features) > 0:
            top_features = sorted_features[:10]
            feature_names = [f[0] for f in top_features]
            feature_scores = [f[1] for f in top_features]
            
            axes[1, 1].barh(range(len(feature_names)), feature_scores)
            axes[1, 1].set_yticks(range(len(feature_names)))
            axes[1, 1].set_yticklabels(feature_names)
            axes[1, 1].set_title('Feature Importance for Pattern Detection')
            axes[1, 1].set_xlabel('Variance Across Clusters')
        
        # Plot 6: Anomaly detection comparison
        anomaly_comparison = {
            'Isolation Forest': n_anomalies_iso,
            'Local Outlier Factor': n_anomalies_lof
        }
        methods = list(anomaly_comparison.keys())
        anomaly_counts = list(anomaly_comparison.values())
        
        axes[1, 2].bar(methods, anomaly_counts)
        axes[1, 2].set_title('Anomaly Detection Results')
        axes[1, 2].set_ylabel('Number of Anomalies')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('wind_turbine_patterns_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved to 'wind_turbine_patterns_analysis.png'")
        plt.show()
        
    except Exception as e:
        print(f"Visualization creation failed: {e}")
        print("This might be due to missing matplotlib. Continuing without visualizations.")

# Create visualizations
create_pattern_visualizations()

# --- 8. ELBOW METHOD FOR OPTIMAL CLUSTERS ---
def find_optimal_clusters(max_clusters=10):
    """Find optimal number of clusters using elbow method"""
    print("\n8. FINDING OPTIMAL NUMBER OF CLUSTERS")
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(unsupervised_data_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(unsupervised_data_scaled, labels))
    
    # Find elbow point (simplified method)
    differences = np.diff(inertias)
    second_differences = np.diff(differences)
    elbow_point = k_range[np.argmax(second_differences) + 1] if len(second_differences) > 0 else optimal_k
    
    print(f"Suggested optimal number of clusters: {elbow_point}")
    print("\nSilhouette scores for different k values:")
    for k, score in zip(k_range, silhouette_scores):
        print(f"  k={k}: {score:.4f}")
    
    return elbow_point, silhouette_scores

# Find optimal clusters
optimal_clusters, silhouette_scores = find_optimal_clusters()

# --- 9. PATTERN INTERPRETATION ---
def interpret_patterns():
    """Interpret the discovered patterns and provide insights"""
    print("\n9. PATTERN INTERPRETATION & INSIGHTS")
    
    # Analyze temporal patterns
    if 'time_stamp' in df_with_clusters.columns:
        print("\nTemporal Pattern Analysis:")
        df_with_clusters['hour'] = pd.to_datetime(df_with_clusters['time_stamp']).dt.hour
        df_with_clusters['day_of_week'] = pd.to_datetime(df_with_clusters['time_stamp']).dt.dayofweek
        
        # Cluster distribution by time of day
        for cluster_id in range(optimal_k):
            cluster_data = df_with_clusters[df_with_clusters['kmeans_cluster'] == cluster_id]
            if len(cluster_data) > 0:
                avg_hour = cluster_data['hour'].mean()
                most_common_dow = cluster_data['day_of_week'].mode().iloc[0] if len(cluster_data['day_of_week'].mode()) > 0 else 0
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                print(f"  Cluster {cluster_id}: Peak at {avg_hour:.1f}h, Most common day: {days[most_common_dow]}")
    
    # Analyze operational patterns
    print("\nOperational Pattern Analysis:")
    operational_features = [col for col in unsupervised_data.columns 
                          if any(keyword in col.lower() for keyword in 
                               ['power', 'wind', 'temp', 'speed', 'pressure'])]
    
    for cluster_id in range(optimal_k):
        cluster_data = df_with_clusters[df_with_clusters['kmeans_cluster'] == cluster_id]
        if len(cluster_data) > 0:
            print(f"\n  Cluster {cluster_id} Operational Profile:")
            for feature in operational_features[:5]:  # Show top 5 operational features
                if feature in cluster_data.columns:
                    mean_val = cluster_data[feature].mean()
                    global_mean = df_with_clusters[feature].mean()
                    deviation = ((mean_val - global_mean) / global_mean) * 100
                    status = "HIGH" if deviation > 10 else "LOW" if deviation < -10 else "NORMAL"
                    print(f"    {feature}: {mean_val:.2f} ({deviation:+.1f}% vs average) - {status}")
    
    # Pattern-based recommendations
    print("\nPattern-Based Operational Insights:")
    
    # Find the most anomalous cluster (highest anomaly rate)
    anomaly_rates = {}
    for cluster_id in range(optimal_k):
        cluster_mask = df_with_clusters['kmeans_cluster'] == cluster_id
        iso_anomaly_rate = df_with_clusters[cluster_mask]['iso_forest_anomaly'].mean()
        lof_anomaly_rate = df_with_clusters[cluster_mask]['lof_anomaly'].mean()
        anomaly_rates[cluster_id] = (iso_anomaly_rate + lof_anomaly_rate) / 2
    
    most_anomalous_cluster = max(anomaly_rates, key=anomaly_rates.get)
    most_normal_cluster = min(anomaly_rates, key=anomaly_rates.get)
    
    print(f"  Most anomalous pattern: Cluster {most_anomalous_cluster} ({anomaly_rates[most_anomalous_cluster]*100:.1f}% anomaly rate)")
    print(f"  Most normal pattern: Cluster {most_normal_cluster} ({anomaly_rates[most_normal_cluster]*100:.1f}% anomaly rate)")
    
    # Correlation with known faults
    if 'fault_label' in df_with_clusters.columns:
        print("\nPattern-Fault Correlation:")
        for cluster_id in range(optimal_k):
            cluster_mask = df_with_clusters['kmeans_cluster'] == cluster_id
            fault_rate = df_with_clusters[cluster_mask]['fault_label'].mean()
            if fault_rate > 0.1:  # If more than 10% fault rate
                print(f"  ⚠️  Cluster {cluster_id}: HIGH RISK PATTERN (fault rate: {fault_rate:.1%})")
            elif fault_rate > 0.05:  # If more than 5% fault rate
                print(f"  ⚡ Cluster {cluster_id}: Moderate risk pattern (fault rate: {fault_rate:.1%})")
            else:
                print(f"  ✅ Cluster {cluster_id}: Low risk pattern (fault rate: {fault_rate:.1%})")
    
    return anomaly_rates

# Run pattern interpretation
pattern_insights = interpret_patterns()

# --- 10. EXPORT DETAILED ANALYSIS ---
def export_analysis_report():
    """Export a comprehensive analysis report"""
    print("\n10. EXPORTING COMPREHENSIVE ANALYSIS REPORT")
    
    report = {
        'analysis_summary': {
            'total_samples': len(df_with_clusters),
            'features_analyzed': len(unsupervised_data.columns),
            'clusters_found': len(set(cluster_labels_kmeans)),
            'anomalies_detected': {
                'isolation_forest': n_anomalies_iso,
                'local_outlier_factor': n_anomalies_lof
            }
        },
        'cluster_profiles': {},
        'feature_importance': dict(sorted_features[:10]),
        'anomaly_rates_by_cluster': pattern_insights,
        'recommendations': []
    }
    
    # Detailed cluster profiles
    for cluster_id in range(optimal_k):
        cluster_data = df_with_clusters[df_with_clusters['kmeans_cluster'] == cluster_id]
        if len(cluster_data) > 0:
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_with_clusters) * 100,
                'anomaly_rate': pattern_insights.get(cluster_id, 0),
            }
            
            # Add feature statistics
            for feature in unsupervised_data.columns[:10]:  # Top 10 features
                if feature in cluster_data.columns:
                    profile[f'{feature}_mean'] = float(cluster_data[feature].mean())
                    profile[f'{feature}_std'] = float(cluster_data[feature].std())
            
            if 'fault_label' in cluster_data.columns:
                profile['fault_rate'] = float(cluster_data['fault_label'].mean())
            
            report['cluster_profiles'][f'cluster_{cluster_id}'] = profile
    
    # Generate recommendations
    if 'fault_label' in df_with_clusters.columns:
        for cluster_id, fault_rate in [(i, df_with_clusters[df_with_clusters['kmeans_cluster'] == i]['fault_label'].mean()) 
                                      for i in range(optimal_k)]:
            if fault_rate > 0.1:
                report['recommendations'].append(f"Monitor Cluster {cluster_id} closely - high fault risk")
            elif pattern_insights.get(cluster_id, 0) > 0.2:
                report['recommendations'].append(f"Investigate Cluster {cluster_id} - high anomaly rate")
    
    # Save report as JSON
    import json
    with open('wind_turbine_pattern_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("Comprehensive analysis report saved to 'wind_turbine_pattern_analysis_report.json'")
    return report

# Export the analysis report
analysis_report = export_analysis_report()

print("\n" + "="*60)
print("UNSUPERVISED LEARNING PATTERN DETECTION COMPLETE")
print("="*60)
print("Files generated:")
print("- wind_turbine_patterns.csv (detailed data with cluster labels)")
print("- cluster_centroids.csv (cluster center coordinates)")
print("- wind_turbine_patterns_analysis.png (visualizations)")
print("- wind_turbine_pattern_analysis_report.json (comprehensive report)")
print("="*60)
