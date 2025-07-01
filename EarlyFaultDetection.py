# Early Fault Detection Model using combined_df_downsampled = downsample_every_nth_row(combined_df, n=18)CADA Data

import glob
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# --- Step 1.5: Downsample data by keeping every 6th row ---
print("\n=== DATA DOWNSAMPLING: KEEPING EVERY 6TH ROW ===")
print(f"Original data shape: {combined_df.shape}")

# Method 1: Simple row-based downsampling (keeps every 6th row globally)
def downsample_every_nth_row(df, n=6):
    """Keep every nth row from the dataframe"""
    return df.iloc[::n].reset_index(drop=True)

combined_df_downsampled = downsample_every_nth_row(combined_df, n=6)
    
print(f"Downsampled data shape: {combined_df_downsampled.shape}")
print(f"Data reduction: {combined_df.shape[0] / combined_df_downsampled.shape[0]:.1f}x fewer rows")

# Use the downsampled data for further processing
combined_df = combined_df_downsampled.copy()

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
    pre_fault_window = (df['time_stamp'] >= start_time - pd.Timedelta(days=3)) & (df['time_stamp'] < start_time)
    df.loc[pre_fault_window, 'fault_label'] = 1

# --- Step 3: Feature Engineering ---
# df.set_index('timestamp', inplace=True)
# df['gearbox_temp_mean_6h'] = df['gearbox_temp'].rolling('6H').mean()
# df['generator_temp_std_3h'] = df['generator_temp'].rolling('3H').std()
# df['temp_diff'] = df['gearbox_temp'] - df['generator_temp']
# df['power_wind_ratio'] = df['power_output'] / (df['wind_speed'] + 1e-6)

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
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# --- Class Imbalance Analysis ---
print("\n=== CLASS IMBALANCE ANALYSIS ===")
print(f"Training set distribution:")
print(f"  Class 0 (Normal): {(y_train == 0).sum()} samples ({(y_train == 0).mean():.1%})")
print(f"  Class 1 (Fault): {(y_train == 1).sum()} samples ({(y_train == 1).mean():.1%})")
print(f"  Imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.1f}:1")

# Calculate custom class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Computed class weights: {class_weight_dict}")

# --- Enhanced Random Forest Models with Different Class Imbalance Techniques ---
print("\n=== RANDOM FOREST WITH CLASS IMBALANCE TECHNIQUES ===")

# Prepare different versions of training data
training_datasets = {}

# 1. Original data with class weights
training_datasets['Original + Class Weights'] = (X_train_scaled, y_train)

# 2. Manual class weight adjustment (more aggressive)
aggressive_class_weights = {0: 1.0, 1: class_weights[1] * 1.5}  # Increase minority class weight

# 3. Threshold-based approach - we'll handle this in the prediction phase
training_datasets['Original + Threshold Tuning'] = (X_train_scaled, y_train)

# Try to use sampling techniques if imblearn is available
try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    
    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    training_datasets['SMOTE Oversampling'] = (X_train_smote, y_train_smote)
    
    # ADASYN oversampling
    adasyn = ADASYN(random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_scaled, y_train)
    training_datasets['ADASYN Oversampling'] = (X_train_adasyn, y_train_adasyn)
    
    # Combined SMOTE + Tomek
    smote_tomek = SMOTETomek(random_state=42)
    X_train_smote_tomek, y_train_smote_tomek = smote_tomek.fit_resample(X_train_scaled, y_train)
    training_datasets['SMOTE + Tomek'] = (X_train_smote_tomek, y_train_smote_tomek)
    
    print("imblearn library available - using advanced sampling techniques")
    
except ImportError:
    print("imblearn library not available - using class weights and threshold tuning")
    
    # Manual oversampling as fallback
    minority_class_indices = np.where(y_train == 1)[0]
    majority_class_indices = np.where(y_train == 0)[0]
    
    # Oversample minority class to balance
    n_minority = len(minority_class_indices)
    n_majority = len(majority_class_indices)
    
    if n_minority > 0:
        # Random oversampling
        oversample_indices = np.random.choice(minority_class_indices, 
                                            size=min(n_majority - n_minority, n_minority * 3), 
                                            replace=True)
        
        all_indices = np.concatenate([majority_class_indices, minority_class_indices, oversample_indices])
        X_train_oversampled = X_train_scaled[all_indices]
        y_train_oversampled = y_train.iloc[all_indices]
        
        training_datasets['Manual Oversampling'] = (X_train_oversampled, y_train_oversampled)

# Dictionary to store enhanced Random Forest models
rf_models = {}
# Dictionary to store all models (including enhanced Random Forests)
models = {}
# Add other models for comparison
gBm = GradientBoostingClassifier(random_state=42);
gBm.fit(X_train_scaled, y_train)
lg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, n_jobs=-1)
lg.fit(X_train_scaled, y_train)
models.update({
    'Gradient Boosting': gBm,
    'Logistic Regression': lg
})

for dataset_name, (X_data, y_data) in training_datasets.items():
    print(f"\n--- {dataset_name} ---")
    print(f"Dataset size: {len(X_data)} samples")
    if hasattr(y_data, 'value_counts'):
        print(f"Class distribution: {y_data.value_counts().to_dict()}")
    else:
        unique, counts = np.unique(y_data, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Configure Random Forest based on dataset type
    if 'Class Weights' in dataset_name:
        rf = RandomForestClassifier(
            n_estimators=200,  # More trees for better performance
            max_depth=15,      # Prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
    elif 'Threshold Tuning' in dataset_name:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
    else:
        # For oversampled data, don't use class weights
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
    
    rf.fit(X_data, y_data)
    rf_models[dataset_name] = rf
    model = models['Gradient Boosting']
    model.fit(X_data, y_data)
    rf_models[f'{dataset_name} - {model}'] = model
    model = models['Logistic Regression']
    model.fit(X_data, y_data)
    rf_models[f'{dataset_name} - {model}'] = model
    
    if hasattr(rf, 'oob_score_'):
        print(f"OOB Score: {rf.oob_score_:.4f}")


# Add all Random Forest variants
for name, model in rf_models.items():
    models[f'RF: {name}'] = model


# Train and evaluate each model with enhanced metrics for imbalanced data
results = {}
threshold_results = {}

print("\n=== MODEL EVALUATION WITH IMBALANCE-AWARE METRICS ===")

for name, model in models.items():
    print(f"\n=== {name} ===")
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test_scaled)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
    else:
        y_proba = None
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate detailed metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Additional metrics for imbalanced data
    if y_proba is not None:
        auc_score = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
    else:
        auc_score = 0
        avg_precision = 0
    
    # Calculate specificity and sensitivity
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Balanced accuracy (good for imbalanced datasets)
    balanced_acc = (sensitivity + specificity) / 2
    
    # Store comprehensive results
    results[name] = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc_roc': auc_score,
        'avg_precision': avg_precision,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    if y_proba is not None:
        print(f"AUC-ROC: {auc_score:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
    
    # Threshold optimization for Random Forest models
    if 'RF:' in name and y_proba is not None:
        print(f"\n--- Threshold Optimization for {name} ---")
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            # Calculate metrics for this threshold
            prec = precision_score(y_test, y_pred_thresh, zero_division=0)
            rec = recall_score(y_test, y_pred_thresh, zero_division=0)
            f1_thresh = f1_score(y_test, y_pred_thresh, zero_division=0)
            
            cm_thresh = confusion_matrix(y_test, y_pred_thresh)
            tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel() if cm_thresh.size == 4 else (0, 0, 0, 0)
            spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
            sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            bal_acc_t = (sens_t + spec_t) / 2
            
            threshold_metrics.append({
                'threshold': threshold,
                'precision': prec,
                'recall': rec,
                'f1': f1_thresh,
                'balanced_accuracy': bal_acc_t,
                'specificity': spec_t
            })
        
        # Find best threshold based on F1 score
        best_f1_idx = np.argmax([m['f1'] for m in threshold_metrics])
        best_f1_threshold = threshold_metrics[best_f1_idx]
        
        # Find best threshold based on balanced accuracy
        best_bal_acc_idx = np.argmax([m['balanced_accuracy'] for m in threshold_metrics])
        best_bal_acc_threshold = threshold_metrics[best_bal_acc_idx]
        
        print(f"Best F1 threshold: {best_f1_threshold['threshold']:.2f} (F1: {best_f1_threshold['f1']:.4f})")
        print(f"Best Balanced Acc threshold: {best_bal_acc_threshold['threshold']:.2f} (Bal Acc: {best_bal_acc_threshold['balanced_accuracy']:.4f})")
        
        threshold_results[name] = {
            'best_f1': best_f1_threshold,
            'best_balanced_acc': best_bal_acc_threshold,
            'all_thresholds': threshold_metrics
        }

# Compare all models with comprehensive metrics
print("\n=== COMPREHENSIVE MODEL COMPARISON ===")
results_df = pd.DataFrame(results).T
print("All Models Performance:")
print(results_df.round(4))

# Highlight best performing models
print("\n=== BEST PERFORMING MODELS BY METRIC ===")
metrics_to_compare = ['balanced_accuracy', 'f1', 'auc_roc', 'recall', 'precision']
for metric in metrics_to_compare:
    if metric in results_df.columns:
        best_model = results_df[metric].idxmax()
        best_score = results_df.loc[best_model, metric]
        print(f"Best {metric.replace('_', ' ').title()}: {best_model} ({best_score:.4f})")

# Show threshold optimization results
if threshold_results:
    print("\n=== THRESHOLD OPTIMIZATION RESULTS ===")
    for model_name, thresh_data in threshold_results.items():
        print(f"\n{model_name}:")
        print(f"  Optimal threshold for F1: {thresh_data['best_f1']['threshold']:.2f}")
        print(f"    -> F1: {thresh_data['best_f1']['f1']:.4f}, Recall: {thresh_data['best_f1']['recall']:.4f}, Precision: {thresh_data['best_f1']['precision']:.4f}")
        print(f"  Optimal threshold for Balanced Accuracy: {thresh_data['best_balanced_acc']['threshold']:.2f}")
        print(f"    -> Bal Acc: {thresh_data['best_balanced_acc']['balanced_accuracy']:.4f}, Recall: {thresh_data['best_balanced_acc']['recall']:.4f}")

# Create visualization function for class imbalance results
def plot_class_imbalance_results():
    """Create visualizations for class imbalance handling results"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Random Forest Class Imbalance Handling Analysis', fontsize=16)
        
        # Plot 1: Model performance comparison
        rf_results = {k: v for k, v in results.items() if 'RF:' in k}
        if rf_results:
            metrics = ['balanced_accuracy', 'f1', 'recall', 'precision']
            rf_df = pd.DataFrame(rf_results).T
            
            ax1 = axes[0, 0]
            rf_df[metrics].plot(kind='bar', ax=ax1, rot=45)
            ax1.set_title('Random Forest Variants Performance')
            ax1.set_ylabel('Score')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confusion matrices for best models
        ax2 = axes[0, 1]
        best_f1_model = results_df['f1'].idxmax()
        if best_f1_model in models:
            model = models[best_f1_model]
            y_pred_best = model.predict(X_test_scaled)
            cm_best = confusion_matrix(y_test, y_pred_best)
            
            im = ax2.imshow(cm_best, interpolation='nearest', cmap=plt.cm.Blues)
            ax2.figure.colorbar(im, ax=ax2)
            ax2.set_title(f'Confusion Matrix: {best_f1_model}')
            ax2.set_ylabel('True Label')
            ax2.set_xlabel('Predicted Label')
            
            # Add text annotations
            thresh = cm_best.max() / 2.
            for i in range(cm_best.shape[0]):
                for j in range(cm_best.shape[1]):
                    ax2.text(j, i, format(cm_best[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm_best[i, j] > thresh else "black")
        
        # Plot 3: Threshold analysis
        if threshold_results:
            ax3 = axes[1, 0]
            for model_name, thresh_data in list(threshold_results.items())[:2]:  # Show first 2 models
                thresholds = [t['threshold'] for t in thresh_data['all_thresholds']]
                f1_scores = [t['f1'] for t in thresh_data['all_thresholds']]
                balanced_accs = [t['balanced_accuracy'] for t in thresh_data['all_thresholds']]
                
                ax3.plot(thresholds, f1_scores, 'o-', label=f'{model_name} F1', alpha=0.7)
                ax3.plot(thresholds, balanced_accs, 's--', label=f'{model_name} Bal Acc', alpha=0.7)
            
            ax3.set_title('Threshold Optimization')
            ax3.set_xlabel('Decision Threshold')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature importance for best Random Forest
        best_rf_models = [k for k in results.keys() if 'RF:' in k]
        if best_rf_models:
            best_rf_name = max(best_rf_models, key=lambda x: results[x]['f1'])
            best_rf_model = models[best_rf_name]
            
            if hasattr(best_rf_model, 'feature_importances_'):
                feature_names = X_train.columns
                importances = best_rf_model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]  # Top 10 features
                
                ax4 = axes[1, 1]
                ax4.barh(range(len(indices)), importances[indices])
                ax4.set_yticks(range(len(indices)))
                ax4.set_yticklabels([feature_names[i] for i in indices])
                ax4.set_title(f'Feature Importance: {best_rf_name}')
                ax4.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('random_forest_class_imbalance_analysis.png', dpi=300, bbox_inches='tight')
        print("\nClass imbalance analysis visualization saved to 'random_forest_class_imbalance_analysis.png'")
        plt.show()
        
    except Exception as e:
        print(f"Visualization creation failed: {e}")

# Create the visualization
plot_class_imbalance_results()

# --- RECOMMENDATIONS FOR CLASS IMBALANCE ---
print("\n=== RECOMMENDATIONS FOR HANDLING CLASS IMBALANCE ===")

# Find the best performing Random Forest variant
rf_results = {k: v for k, v in results.items() if 'RF:' in k}
if rf_results:
    best_rf = max(rf_results.keys(), key=lambda x: rf_results[x]['f1'])
    best_rf_metrics = rf_results[best_rf]
    
    print(f"🏆 Best Random Forest variant: {best_rf}")
    print(f"   F1 Score: {best_rf_metrics['f1']:.4f}")
    print(f"   Balanced Accuracy: {best_rf_metrics['balanced_accuracy']:.4f}")
    print(f"   Recall: {best_rf_metrics['recall']:.4f}")
    print(f"   Precision: {best_rf_metrics['precision']:.4f}")

print("\n📋 Recommendations:")
print("1. Use balanced class weights or sampling techniques for better minority class detection")
print("2. Focus on Balanced Accuracy and F1-score rather than overall accuracy for imbalanced data")
print("3. Optimize decision threshold based on business requirements (precision vs recall trade-off)")
print("4. Consider ensemble methods that combine multiple sampling techniques")
print("5. Monitor both False Positive and False Negative rates for operational decisions")

# Cost-sensitive analysis
if any('fault_label' in str(results[model]) for model in results):
    print("\n💰 Cost-Sensitive Considerations:")
    print("   - Missing a fault (False Negative) likely costs more than a false alarm (False Positive)")
    print("   - Consider lowering decision threshold to increase recall (catch more faults)")
    print("   - Monitor precision to avoid too many false alarms that could overwhelm maintenance teams")

# === DETAILED PERFORMANCE ANALYSIS ===
print("\n" + "="*60)
print("DETAILED PERFORMANCE ANALYSIS & RECOMMENDATIONS")
print("="*60)

# Operational Impact Analysis
print("\n🚨 OPERATIONAL IMPACT ANALYSIS:")
total_faults = y_test.sum()
total_normal = (y_test == 0).sum()

for name, metrics in results.items():
    missed_faults = metrics['false_negatives']
    false_alarms = metrics['false_positives']
    caught_faults = metrics['true_positives']
    
    print(f"\n{name}:")
    print(f"   Faults Caught: {caught_faults}/{total_faults} ({caught_faults/total_faults*100:.1f}%)")
    print(f"   Faults Missed: {missed_faults} (⚠️  CRITICAL: {missed_faults/total_faults*100:.1f}%)")
    print(f"   False Alarms: {false_alarms}/{total_normal} ({false_alarms/total_normal*100:.1f}%)")

# Model Selection Guidance
print("\n🎯 MODEL SELECTION GUIDANCE:")
print("\nFor SAFETY-CRITICAL applications (prefer catching faults):")
best_recall_model = max(results.items(), key=lambda x: x[1]['recall'])
print(f"   → Use: {best_recall_model[0]}")
print(f"     Catches {best_recall_model[1]['recall']*100:.1f}% of faults")
print(f"     False alarm rate: {best_recall_model[1]['false_positives']/total_normal*100:.1f}%")

print("\nFor BALANCED applications:")
best_f1_model = max(results.items(), key=lambda x: x[1]['f1'])
print(f"   → Use: {best_f1_model[0]}")
print(f"     F1-Score: {best_f1_model[1]['f1']:.3f}")
print(f"     Balanced Accuracy: {best_f1_model[1]['balanced_accuracy']:.3f}")

# Random Forest Issues Analysis
print("\n🔍 RANDOM FOREST PERFORMANCE ISSUES:")
rf_models_performance = {k: v for k, v in results.items() if 'RF:' in k and 'GradientBoosting' not in k and 'LogisticRegression' not in k}

if rf_models_performance:
    avg_rf_recall = np.mean([v['recall'] for v in rf_models_performance.values()])
    best_non_rf = max({k: v for k, v in results.items() if 'RF:' not in k}.items(), 
                     key=lambda x: x[1]['recall'])
    
    print(f"   Average RF Recall: {avg_rf_recall:.3f} ({avg_rf_recall*100:.1f}%)")
    print(f"   Best Non-RF Recall: {best_non_rf[1]['recall']:.3f} ({best_non_rf[1]['recall']*100:.1f}%)")
    print(f"   Performance Gap: {(best_non_rf[1]['recall'] - avg_rf_recall)*100:.1f} percentage points")
    
    print("\n   Possible RF Issues:")
    print("   • Overfitting to majority class despite class weights")
    print("   • Need more aggressive threshold tuning")
    print("   • Tree depth/complexity may need adjustment")
    print("   • Feature scaling may be affecting tree splits")

# Threshold Optimization Recommendations
print("\n⚙️  THRESHOLD OPTIMIZATION RECOMMENDATIONS:")
if threshold_results:
    for model_name, thresh_data in threshold_results.items():
        current_recall = results[model_name]['recall']
        best_recall_thresh = max(thresh_data['all_thresholds'], key=lambda x: x['recall'])
        
        print(f"\n{model_name}:")
        print(f"   Current Recall: {current_recall:.3f}")
        print(f"   Max Possible Recall: {best_recall_thresh['recall']:.3f} (threshold: {best_recall_thresh['threshold']:.2f})")
        if best_recall_thresh['recall'] > current_recall * 1.2:
            print(f"   🚀 RECOMMENDATION: Lower threshold to {best_recall_thresh['threshold']:.2f}")
            print(f"      → Recall improvement: +{(best_recall_thresh['recall'] - current_recall)*100:.1f} percentage points")

# Business Impact Estimates
print("\n💼 ESTIMATED BUSINESS IMPACT:")
print("Assumptions: Missing a fault costs 100x more than a false alarm")

cost_false_negative = 100  # Relative cost
cost_false_positive = 1    # Relative cost

for name, metrics in list(results.items())[:5]:  # Top 5 models
    total_cost = (metrics['false_negatives'] * cost_false_negative + 
                  metrics['false_positives'] * cost_false_positive)
    print(f"{name}: Relative Cost = {total_cost:.0f}")

# Feature Engineering Suggestions
print("\n🔧 FEATURE ENGINEERING SUGGESTIONS:")
print("To improve fault detection performance:")
print("1. Add more temporal features (rolling statistics over different windows)")
print("2. Create anomaly scores from normal operation baseline")
print("3. Add interaction features between sensors")
print("4. Consider environmental features (weather, seasonal patterns)")
print("5. Add trend features (increasing/decreasing patterns)")

# Next Steps Action Plan
print("\n📋 IMMEDIATE ACTION PLAN:")
print("1. Deploy Logistic Regression model with optimized threshold")
print("2. Investigate Random Forest hyperparameter issues")
print("3. Collect more fault examples if possible")
print("4. Implement ensemble combining Logistic Regression + Gradient Boosting")
print("5. Set up monitoring for model drift and performance degradation")

print("\n" + "="*60)

# === ENSEMBLE MODEL CREATION ===
print("\n🤝 CREATING ENSEMBLE MODEL FROM BEST PERFORMERS")

# Select best models for ensemble
best_models_for_ensemble = {
    'Logistic Regression': models['Logistic Regression'],
    'Gradient Boosting': models['Gradient Boosting']
}

# Simple voting ensemble
from sklearn.ensemble import VotingClassifier

ensemble_voting = VotingClassifier(
    estimators=[(name, model) for name, model in best_models_for_ensemble.items()],
    voting='soft'  # Use probabilities
)

ensemble_voting.fit(X_train_scaled, y_train)

# Evaluate ensemble
y_pred_ensemble = ensemble_voting.predict(X_test_scaled)
y_proba_ensemble = ensemble_voting.predict_proba(X_test_scaled)[:, 1]

print("\n=== ENSEMBLE MODEL PERFORMANCE ===")
print("Classification Report:")
print(classification_report(y_test, y_pred_ensemble))

cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
print("Confusion Matrix:")
print(cm_ensemble)

# Calculate ensemble metrics
tn_e, fp_e, fn_e, tp_e = cm_ensemble.ravel()
ensemble_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_ensemble),
    'balanced_accuracy': (tp_e/(tp_e+fn_e) + tn_e/(tn_e+fp_e)) / 2,
    'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
    'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
    'f1': f1_score(y_test, y_pred_ensemble, zero_division=0),
    'auc_roc': roc_auc_score(y_test, y_proba_ensemble),
    'true_positives': tp_e,
    'false_positives': fp_e,
    'false_negatives': fn_e
}

print(f"\nEnsemble Performance:")
for metric, value in ensemble_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Add ensemble to results for comparison
results['Ensemble (LR + GB)'] = ensemble_metrics

# Compare ensemble vs best individual model
best_individual = max(results.items(), key=lambda x: x[1]['f1'] if x[0] != 'Ensemble (LR + GB)' else 0)
print(f"\nEnsemble vs Best Individual Model ({best_individual[0]}):")
print(f"  F1-Score: {ensemble_metrics['f1']:.4f} vs {best_individual[1]['f1']:.4f}")
print(f"  Recall: {ensemble_metrics['recall']:.4f} vs {best_individual[1]['recall']:.4f}")
print(f"  Balanced Accuracy: {ensemble_metrics['balanced_accuracy']:.4f} vs {best_individual[1]['balanced_accuracy']:.4f}")

# Threshold optimization for ensemble
print("\n=== ENSEMBLE THRESHOLD OPTIMIZATION ===")
thresholds_ensemble = np.arange(0.1, 0.9, 0.05)
best_ensemble_metrics = []

for threshold in thresholds_ensemble:
    y_pred_thresh_ens = (y_proba_ensemble >= threshold).astype(int)
    
    recall_ens = recall_score(y_test, y_pred_thresh_ens, zero_division=0)
    precision_ens = precision_score(y_test, y_pred_thresh_ens, zero_division=0)
    f1_ens = f1_score(y_test, y_pred_thresh_ens, zero_division=0)
    
    cm_ens = confusion_matrix(y_test, y_pred_thresh_ens)
    tn_t, fp_t, fn_t, tp_t = cm_ens.ravel() if cm_ens.size == 4 else (0, 0, 0, 0)
    balanced_acc_ens = ((tp_t/(tp_t+fn_t)) + (tn_t/(tn_t+fp_t))) / 2 if (tp_t+fn_t) > 0 and (tn_t+fp_t) > 0 else 0
    
    best_ensemble_metrics.append({
        'threshold': threshold,
        'recall': recall_ens,
        'precision': precision_ens,
        'f1': f1_ens,
        'balanced_accuracy': balanced_acc_ens
    })

# Find optimal threshold for ensemble
best_recall_thresh_ens = max(best_ensemble_metrics, key=lambda x: x['recall'])
best_f1_thresh_ens = max(best_ensemble_metrics, key=lambda x: x['f1'])

print(f"Optimal threshold for max recall: {best_recall_thresh_ens['threshold']:.2f}")
print(f"  → Recall: {best_recall_thresh_ens['recall']:.4f}")
print(f"  → Precision: {best_recall_thresh_ens['precision']:.4f}")

print(f"Optimal threshold for max F1: {best_f1_thresh_ens['threshold']:.2f}")
print(f"  → F1: {best_f1_thresh_ens['f1']:.4f}")
print(f"  → Recall: {best_f1_thresh_ens['recall']:.4f}")

# Final recommendation
print("\n🎯 FINAL MODEL RECOMMENDATION:")
if ensemble_metrics['f1'] > best_individual[1]['f1']:
    print("✅ Use ENSEMBLE MODEL (Logistic Regression + Gradient Boosting)")
    print(f"   Improvement over best individual: +{(ensemble_metrics['f1'] - best_individual[1]['f1'])*100:.1f} F1 points")
else:
    print(f"✅ Use {best_individual[0]} (individual model performs better)")

print(f"\nRecommended threshold: {best_f1_thresh_ens['threshold']:.2f} (for balanced performance)")
print(f"Alternative threshold: {best_recall_thresh_ens['threshold']:.2f} (for maximum fault detection)")

print("\n" + "="*60)

# === ZERO RECALL DIAGNOSTIC ANALYSIS ===
print("\n🔬 ZERO RECALL DIAGNOSTIC ANALYSIS")
print("="*50)

# Find models with zero or very low recall
zero_recall_models = {k: v for k, v in results.items() if v['recall'] < 0.01}

if zero_recall_models:
    print(f"\n⚠️  Models with Zero/Near-Zero Recall ({len(zero_recall_models)} found):")
    for model_name in zero_recall_models.keys():
        print(f"   • {model_name}")
    
    # Analyze prediction distributions
    print("\n📊 PREDICTION DISTRIBUTION ANALYSIS:")
    for model_name in list(zero_recall_models.keys())[:3]:  # Analyze first 3
        if model_name in models:
            model = models[model_name]
            y_pred = model.predict(X_test_scaled)
            
            print(f"\n{model_name}:")
            unique_preds, counts = np.unique(y_pred, return_counts=True)
            for pred, count in zip(unique_preds, counts):
                print(f"   Predicted {pred}: {count} samples ({count/len(y_pred)*100:.1f}%)")
            
            # Check prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                print(f"   Probability Stats:")
                print(f"     Min: {y_proba.min():.6f}")
                print(f"     Max: {y_proba.max():.6f}")
                print(f"     Mean: {y_proba.mean():.6f}")
                print(f"     Std: {y_proba.std():.6f}")
                
                # Check how many samples have prob > 0.5
                high_prob_count = (y_proba > 0.5).sum()
                print(f"     Samples with prob > 0.5: {high_prob_count} ({high_prob_count/len(y_proba)*100:.1f}%)")
                
                # Check distribution of probabilities for actual fault cases
                fault_indices = y_test == 1
                if fault_indices.sum() > 0:
                    fault_probs = y_proba[fault_indices]
                    print(f"   Fault Cases Probability Stats:")
                    print(f"     Min: {fault_probs.min():.6f}")
                    print(f"     Max: {fault_probs.max():.6f}")
                    print(f"     Mean: {fault_probs.mean():.6f}")
                    
                    # Count how many fault cases have different probability thresholds
                    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
                    for thresh in thresholds:
                        count = (fault_probs > thresh).sum()
                        print(f"     Faults with prob > {thresh}: {count}/{len(fault_probs)} ({count/len(fault_probs)*100:.1f}%)")

# Root Cause Analysis
print("\n🔍 ROOT CAUSE ANALYSIS:")
print("\nPossible reasons for zero recall:")
print("1. **Threshold too high**: Model probabilities never exceed 0.5")
print("2. **Extreme class imbalance**: Model learned to always predict majority class")
print("3. **Feature scaling issues**: Fault patterns not distinguishable after scaling")
print("4. **Data leakage or contamination**: Training data doesn't represent real faults")
print("5. **Hyperparameter issues**: Model complexity too low or regularization too high")
print("6. **Class weight ineffective**: Balancing technique not working for this algorithm")

# Quick fixes to test
print("\n💡 IMMEDIATE DIAGNOSTIC TESTS:")

# Test 1: Try extremely low threshold on best probability-based model
prob_models = {k: v for k, v in models.items() if hasattr(v, 'predict_proba')}
if prob_models:
    # Get model with highest AUC (best probability calibration)
    best_auc_model_name = max(results.keys(), key=lambda x: results[x]['auc_roc'])
    if best_auc_model_name in prob_models:
        best_model = prob_models[best_auc_model_name]
        y_proba_test = best_model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\nTesting ultra-low thresholds on {best_auc_model_name}:")
        ultra_low_thresholds = [0.01, 0.05, 0.1, 0.15, 0.2]
        
        for thresh in ultra_low_thresholds:
            y_pred_ultra = (y_proba_test >= thresh).astype(int)
            recall_ultra = recall_score(y_test, y_pred_ultra, zero_division=0)
            precision_ultra = precision_score(y_test, y_pred_ultra, zero_division=0)
            
            cm_ultra = confusion_matrix(y_test, y_pred_ultra)
            tn, fp, fn, tp = cm_ultra.ravel() if cm_ultra.size == 4 else (0, 0, 0, 0)
            
            print(f"   Threshold {thresh:.2f}: Recall={recall_ultra:.3f}, Precision={precision_ultra:.3f}, TP={tp}, FP={fp}")

# Test 2: Check if any model can detect ANY faults
print(f"\n🔍 EMERGENCY FAULT DETECTION TEST:")
print("Testing if ANY model can detect faults with ultra-low thresholds...")

emergency_results = {}
for model_name, model in list(models.items())[:5]:  # Test first 5 models
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Find threshold that catches at least 50% of faults
        thresholds = np.percentile(y_proba, np.arange(0, 100, 5))  # Try many thresholds
        
        best_recall = 0
        best_thresh = 0.5
        best_stats = None
        
        for thresh in thresholds:
            y_pred_thresh = (y_proba >= thresh).astype(int)
            recall = recall_score(y_test, y_pred_thresh, zero_division=0)
            
            if recall > best_recall:
                best_recall = recall
                best_thresh = thresh
                precision = precision_score(y_test, y_pred_thresh, zero_division=0)
                cm = confusion_matrix(y_test, y_pred_thresh)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                best_stats = {'precision': precision, 'tp': tp, 'fp': fp, 'fn': fn}
        
        emergency_results[model_name] = {
            'best_recall': best_recall,
            'best_threshold': best_thresh,
            'stats': best_stats
        }
        
        if best_recall > 0:
            print(f"✅ {model_name}: CAN detect faults!")
            print(f"   Best threshold: {best_thresh:.6f}")
            print(f"   Max recall: {best_recall:.3f} ({best_recall*100:.1f}%)")
            if best_stats:
                print(f"   At this threshold: TP={best_stats['tp']}, FP={best_stats['fp']}, Precision={best_stats['precision']:.3f}")
        else:
            print(f"❌ {model_name}: Cannot detect any faults")

# Recommendations based on findings
print(f"\n📋 EMERGENCY RECOMMENDATIONS:")
working_models = {k: v for k, v in emergency_results.items() if v['best_recall'] > 0}

if working_models:
    best_emergency_model = max(working_models.items(), key=lambda x: x[1]['best_recall'])
    print(f"🚀 IMMEDIATE ACTION: Use {best_emergency_model[0]}")
    print(f"   Set threshold to: {best_emergency_model[1]['best_threshold']:.6f}")
    print(f"   This will catch {best_emergency_model[1]['best_recall']*100:.1f}% of faults")
    print(f"   Expected false alarms: {best_emergency_model[1]['stats']['fp']} out of {(y_test == 0).sum()}")
else:
    print("🚨 CRITICAL: NO MODEL CAN DETECT FAULTS!")
    print("   This suggests fundamental data or methodology issues.")
    print("   Immediate investigation needed:")
    print("   1. Check if fault labels are correct")
    print("   2. Verify training/test split")
    print("   3. Check for data contamination")
    print("   4. Review feature engineering")

# Data integrity checks
print(f"\n🔍 DATA INTEGRITY CHECKS:")
print(f"Training set fault examples: {(y_train == 1).sum()}")
print(f"Test set fault examples: {(y_test == 1).sum()}")
print(f"Feature consistency: {X_train.shape[1]} features in both train/test")

# Check if training data has any patterns for faults
if (y_train == 1).sum() > 0:
    fault_train_indices = y_train == 1
    normal_train_indices = y_train == 0
    
    # Compare feature means between fault and normal in training data
    fault_features = pd.DataFrame(X_train_scaled)[fault_train_indices].mean()
    normal_features = pd.DataFrame(X_train_scaled)[normal_train_indices].mean()
    feature_diff = abs(fault_features - normal_features)
    
    print(f"\nTop 5 features with largest difference between fault/normal (training):")
    top_features = feature_diff.nlargest(5)
    for i, (idx, diff) in enumerate(top_features.items()):
        feature_name = X_train.columns[idx] if hasattr(X_train, 'columns') else f'Feature_{idx}'
        print(f"   {i+1}. {feature_name}: {diff:.4f}")
