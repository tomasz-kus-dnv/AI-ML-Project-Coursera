import os
from imblearn.combine import SMOTEENN
from numpy.typing import NDArray
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, f1_score

def downsample_every_nth_row(df, n=12):
    """Keep every nth row from the dataframe"""
    return df.iloc[::n].reset_index(drop=True)

def downsample_ndArray_every_nth_row(arr: NDArray, n=12):
    """Keep every nth row from the NDArray"""
    return arr[::n]

def create_sliding_windows(df, event_start_id=None, event_end_id=None,
                           window_size=12, step_size=6, prefault_horizon=18):
    df = df.sort_values("id").reset_index(drop=True)
    feature_cols = [col for col in df.columns if col.startswith("sensor_") or col.startswith("wind_") or col.startswith("power_") or col.startswith("reactive_")]
    df.dropna(inplace=True)
    df = df[df['status_type_id'] != '2']  # Exclude fault status
    df = df[df['status_type_id'] != '3']  # Exclude fault status
    df = df[df['status_type_id'] != '4']  # Exclude fault status
    # df_prediction = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
    df_prediction = df.copy().reset_index(drop=True)

    # anomaly_ids = set(range(event_start_id - prefault_horizon, event_start_id)) if event_start_id else set()
    anomaly_ids = set(range(event_start_id - prefault_horizon, event_start_id)) if event_start_id else set()
    fault_ids = set(range(event_start_id, event_end_id)) if event_start_id else set()

    X_windows = []
    y_labels = []

    for start in range(0, len(df_prediction) - window_size + 1, step_size):
        end = start + window_size
        window = df_prediction.iloc[start:end]
        window_ids = set(window["id"])

        if int(len(fault_ids.intersection(window_ids))) > 0:
            continue

        label = int(len(anomaly_ids.intersection(window_ids)) > 0)
        X = window[feature_cols].values
        X_windows.append(X)
        y_labels.append(label)

    return X_windows, y_labels

def create_test_sliding_windows(df, event_start_id=None,
                           window_size=12, step_size=6, prefault_horizon=18):
    df = df.sort_values("id").reset_index(drop=True)
    feature_cols = [col for col in df.columns if col.startswith("sensor_") or col.startswith("wind_") or col.startswith("power_") or col.startswith("reactive_")]
    df.dropna(inplace=True)
    # df_prediction = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
    df_prediction = df.copy().reset_index(drop=True)

    # Check if event_start_id is smaller than the first data point
    anomaly_before_data = event_start_id is not None and event_start_id > df_prediction['id'].iloc[0] and event_start_id < df_prediction['id'].iloc[-1]
    label = 0
    if anomaly_before_data:
        label = 1

    X_windows = []
    y_labels = []

    for start in range(0, len(df_prediction) - window_size + 1, step_size):
        end = start + window_size
        window = df_prediction.iloc[start:end]

        X = window[feature_cols].values
        X_windows.append(X)
        y_labels.append(label)

    return X_windows, y_labels

def load_event_info(event_info_path):
    return pd.read_csv(event_info_path).set_index("event_id")

def process_all_events(data_dir, window_size=12, step_size=6, prefault_horizon=18):
    all_X = []
    all_y = []
    all_X_test = []
    all_y_test = []
    
    # for farm in os.listdir(data_dir):
    farm_path = data_dir # os.path.join(data_dir, farm)
    event_info_path = os.path.join(farm_path, "comma_event_info.csv")
    dataset_path = os.path.join(farm_path, "datasets")

    # if not os.path.isfile(event_info_path):
    #    continue

    event_info = load_event_info(event_info_path)

    for csv_file in os.listdir(dataset_path):
        if not csv_file.endswith(".csv"):
            continue

        event_id = os.path.splitext(csv_file)[0].replace("comma_", "")
        # if int(event_id) > 42:
        #    continue
        df = pd.read_csv(os.path.join(dataset_path, csv_file))
        
        try:
            event_start_id = int(event_info.loc[int(event_id), "event_start_id"])
            label_str = event_info.loc[int(event_id), "event_label"]
            event_end_id = int(event_info.loc[int(event_id), "event_end_id"])
        except KeyError:
            continue  # skip unknown event

        is_anomaly = label_str.strip().lower() == "anomaly"
        event_start_id = event_start_id if is_anomaly else None

        # X_windows, y_labels = create_sliding_windows(df[df['train_test'] == 'train'], event_start_id, event_end_id,
        X_windows, y_labels = create_sliding_windows(df, event_start_id, event_end_id,
                                                        window_size, step_size, prefault_horizon)
        all_X.extend(X_windows)
        all_y.extend(y_labels)
        # X_test_windows, y_test_labels = create_test_sliding_windows(df[df['train_test'] == 'prediction'], event_start_id,
        #                                                 window_size, step_size, prefault_horizon)
        # all_X.extend(X_windows)
        # all_y.extend(y_labels)
        # all_X_test.extend(X_test_windows)
        # all_y_test.extend(y_test_labels)

    return np.array(all_X), np.array(all_y), np.array(all_X_test), np.array(all_y_test)

def flatten_windows(X):
    # reshape (num_samples, time, features) -> (num_samples, time * features)
    return X.reshape(X.shape[0], -1)

def prepare_ml_dataset(data_dir, test_size=0.2, random_state=42,
                       window_size=12, step_size=6, prefault_horizon=18):
    print("Loading and processing SCADA data...")
    X, y, X_test, y_test = process_all_events(data_dir, window_size, step_size, prefault_horizon)

    # X = downsample_ndArray_every_nth_row(X, 4).copy()
    # y = downsample_ndArray_every_nth_row(y, 4).copy()
    # X_test = downsample_ndArray_every_nth_row(X_test, 4).copy()
    # y_test = downsample_ndArray_every_nth_row(y_test, 4).copy()

    print(f"Total samples: {len(X)}, Anomalies: {np.sum(y)}")

    X_flat = flatten_windows(X)
    # X_test_flat = flatten_windows(X_test)

    print("Normalizing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    # X_test_scaled = scaler.fit_transform(X_test_flat)

    # print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # return X_scaled, X_test_scaled, y, y_test, scaler
    return X_train, X_test, y_train, y_test, scaler

DATA_DIR = '../wind-turbines-data/setA/Wind Farm A'  # root dir containing WindFarmA/, WindFarmB/, etc.

X_train, X_test, y_train, y_test, scaler = prepare_ml_dataset(
    data_dir=DATA_DIR,
    test_size=0.2,
    window_size=18,
    step_size=3,
    prefault_horizon=480
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

smote_enn = SMOTEENN(random_state=42, n_jobs=-1)
X_train_smote, y_train_smote = smote_enn.fit_resample(X_train, y_train)
# X_train_smote = X_train
# y_train_smote = y_train

# model = RandomForestClassifier(
#     n_estimators=500,           # More trees = better performance (but slower)
#     max_depth=20,               # Deeper trees for complex patterns
#     min_samples_split=5,        # Allow smaller splits
#     min_samples_leaf=2,         # Allow smaller leaf nodes
#     max_features='sqrt',        # Feature selection strategy
#     bootstrap=True,
#     oob_score=True,
#     random_state=42,
#     n_jobs=-1,
#     class_weight=None,          # Already balanced with SMOTE
#     criterion='gini',           # Try 'entropy' as alternative
#     max_samples=0.8,            # Use 80% of samples per tree
# )
# model = RandomForestClassifier(
#         n_estimators=800,  # More trees for better performance
#         max_depth=20,      # Prevent overfitting
#         min_samples_split=10,
#         min_samples_leaf=5,
#         random_state=42,
#         n_jobs=-1,
#         bootstrap=True,
#         oob_score=True
#         )
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train_smote, y_train_smote)
# model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(f"Model accuracy: {model.score(X_test, y_test):.4f}")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print("\n" + "="*60)
print("DETAILED CLASSIFICATION RESULTS EXPLANATION")
print("="*60)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\n=== BASIC METRICS ===")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)

print(f"\nConfusion Matrix Breakdown:")
print(f"  True Negatives (TN):  {tn} - Correctly predicted NORMAL")
print(f"  False Positives (FP): {fp} - Incorrectly predicted as FAULT (False Alarms)")
print(f"  False Negatives (FN): {fn} - Missed FAULTS (Dangerous!)")
print(f"  True Positives (TP):  {tp} - Correctly detected FAULTS")

print(f"\n=== WHAT THESE NUMBERS MEAN ===")
total_normal = tn + fp
total_faults = tp + fn
total_samples = len(y_test)

print(f"Total test samples: {total_samples}")
print(f"  Normal samples: {total_normal} ({total_normal/total_samples*100:.1f}%)")
print(f"  Fault samples:  {total_faults} ({total_faults/total_samples*100:.1f}%)")

if total_faults > 0:
    fault_detection_rate = tp / total_faults
    print(f"\n🎯 FAULT DETECTION PERFORMANCE:")
    print(f"  Caught {tp} out of {total_faults} actual faults")
    print(f"  Detection rate: {fault_detection_rate:.1%}")
    print(f"  Missed faults: {fn} ({fn/total_faults*100:.1f}%)")

if total_normal > 0:
    false_alarm_rate = fp / total_normal
    print(f"\n⚠️  FALSE ALARM ANALYSIS:")
    print(f"  False alarms: {fp} out of {total_normal} normal cases")
    print(f"  False alarm rate: {false_alarm_rate:.1%}")

print("\n=== DETAILED CLASSIFICATION REPORT ===")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fault']))

print("\n=== METRICS EXPLANATION ===")
print("""
PRECISION (How reliable are fault predictions?):
  - When model predicts FAULT, how often is it correct?
  - Formula: TP / (TP + FP)
  - High precision = Few false alarms
  
RECALL/SENSITIVITY (How many faults are caught?):
  - Of all actual faults, how many did we detect?
  - Formula: TP / (TP + FN)  
  - High recall = Catches most faults
  
F1-SCORE (Balance between precision and recall):
  - Harmonic mean of precision and recall
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)
  - Good for imbalanced datasets like fault detection
  
SUPPORT:
  - Number of actual samples in each class
  - Shows class distribution in test set
""")

print("\n=== BUSINESS IMPACT ANALYSIS ===")
if total_faults > 0 and total_normal > 0:
    print(f"💰 COST ANALYSIS (hypothetical):")
    
    # Hypothetical costs (adjust based on your domain)
    cost_missed_fault = 100000  # Cost of missing a fault (equipment damage, downtime)
    cost_false_alarm = 1000     # Cost of false alarm (unnecessary maintenance)
    
    total_cost_missed = fn * cost_missed_fault
    total_cost_false_alarms = fp * cost_false_alarm
    total_cost = total_cost_missed + total_cost_false_alarms
    
    print(f"  Missed faults cost: ${total_cost_missed:,} ({fn} × ${cost_missed_fault:,})")
    print(f"  False alarms cost: ${total_cost_false_alarms:,} ({fp} × ${cost_false_alarm:,})")
    print(f"  Total cost: ${total_cost:,}")

print(f"\n=== MODEL PERFORMANCE ASSESSMENT ===")
if recall >= 0.8:
    print("✅ EXCELLENT: Model catches most faults (≥80%)")
elif recall >= 0.6:
    print("🟡 GOOD: Model catches majority of faults (≥60%)")
elif recall >= 0.4:
    print("⚠️  FAIR: Model catches some faults (≥40%)")
else:
    print("❌ POOR: Model misses too many faults (<40%)")

if precision >= 0.7:
    print("✅ LOW FALSE ALARMS: Good precision (≥70%)")
elif precision >= 0.5:
    print("🟡 MODERATE FALSE ALARMS: Fair precision (≥50%)")
else:
    print("⚠️  HIGH FALSE ALARMS: Poor precision (<50%)")

print(f"\n=== RECOMMENDATIONS ===")
if recall < 0.7:
    print("🔧 TO IMPROVE FAULT DETECTION:")
    print("  - Lower prediction threshold")
    print("  - Use more sensitive models")
    print("  - Engineer better features")
    print("  - Collect more fault examples")

if precision < 0.6:
    print("🔧 TO REDUCE FALSE ALARMS:")
    print("  - Raise prediction threshold")
    print("  - Improve feature engineering")
    print("  - Use ensemble methods")
    print("  - Remove noisy features")

print("="*60)

# === EARLY WARNING TIME ANALYSIS ===
print(f"\n=== EARLY WARNING TIME ANALYSIS ===")

def calculate_early_warning_metrics(y_test, y_pred, y_proba, window_size, step_size):
    """
    Calculate how much advance warning the model provides before actual faults
    """
    # Find all predicted anomalies (both true positives and false positives)
    predicted_anomaly_indices = np.where(y_pred == 1)[0]
    
    # Find all actual faults in test set
    actual_fault_indices = np.where(y_test == 1)[0]
    
    if len(actual_fault_indices) == 0:
        print("⚠️  No actual faults in test set - cannot calculate early warning metrics")
        return None, None, None
    
    if len(predicted_anomaly_indices) == 0:
        print("⚠️  No anomalies predicted - model provides no early warning")
        return None, None, None
    
    early_warning_times = []
    fault_detection_details = []
    
    print(f"Analyzing {len(actual_fault_indices)} actual faults...")
    
    for actual_fault_idx in actual_fault_indices:
        # Find the first prediction that came before this actual fault
        earlier_predictions = predicted_anomaly_indices[predicted_anomaly_indices <= actual_fault_idx]
        
        if len(earlier_predictions) > 0:
            # Find the first (earliest) prediction before this fault
            first_prediction_idx = earlier_predictions[0]
            
            # Calculate time difference in windows
            warning_time_windows = actual_fault_idx - first_prediction_idx
            
            # Convert to time units (assuming each window represents step_size time units)
            warning_time_units = warning_time_windows * step_size
            
            early_warning_times.append(warning_time_units)
            
            # Get prediction confidence for this early warning
            prediction_confidence = y_proba[first_prediction_idx]
            
            fault_detection_details.append({
                'actual_fault_window': actual_fault_idx,
                'first_prediction_window': first_prediction_idx,
                'warning_time_windows': warning_time_windows,
                'warning_time_units': warning_time_units,
                'prediction_confidence': prediction_confidence
            })
            
            # print(f"  Fault at window {actual_fault_idx}: First warning at window {first_prediction_idx}")
            # print(f"    Warning time: {warning_time_windows} windows ({warning_time_units} time units)")
            # print(f"    Prediction confidence: {prediction_confidence:.3f}")
        # else:
            # print(f"  Fault at window {actual_fault_idx}: NO EARLY WARNING (missed)")
    
    return early_warning_times, fault_detection_details, actual_fault_indices

# Calculate early warning metrics
early_warning_times, detection_details, all_faults = calculate_early_warning_metrics(
    y_test, y_pred, y_proba, window_size=50, step_size=3
)

if early_warning_times and len(early_warning_times) > 0:
    # Calculate statistics
    avg_warning_time = np.mean(early_warning_times)
    median_warning_time = np.median(early_warning_times)
    min_warning_time = np.min(early_warning_times)
    max_warning_time = np.max(early_warning_times)
    std_warning_time = np.std(early_warning_times)
    
    print(f"\n📊 EARLY WARNING STATISTICS:")
    print(f"  Faults with early warning: {len(early_warning_times)} out of {len(all_faults)}")
    print(f"  Early detection rate: {len(early_warning_times)/len(all_faults)*100:.1f}%")
    print(f"  ")
    print(f"  Average warning time: {avg_warning_time:.1f} time units")
    print(f"  Median warning time:  {median_warning_time:.1f} time units")
    print(f"  Minimum warning time: {min_warning_time:.1f} time units")
    print(f"  Maximum warning time: {max_warning_time:.1f} time units")
    print(f"  Standard deviation:   {std_warning_time:.1f} time units")
    
    # Convert to more meaningful time units (assuming each time unit = 10 minutes for SCADA data)
    time_unit_minutes = 10  # Adjust based on your data collection frequency
    
    print(f"\n🕐 CONVERTED TO REAL TIME (assuming {time_unit_minutes} min/unit):")
    print(f"  Average warning time: {avg_warning_time * time_unit_minutes:.0f} minutes ({avg_warning_time * time_unit_minutes / 60:.1f} hours)")
    print(f"  Median warning time:  {median_warning_time * time_unit_minutes:.0f} minutes ({median_warning_time * time_unit_minutes / 60:.1f} hours)")
    print(f"  Range: {min_warning_time * time_unit_minutes:.0f} - {max_warning_time * time_unit_minutes:.0f} minutes")
    
    # Warning time distribution analysis
    print(f"\n📈 WARNING TIME DISTRIBUTION:")
    
    # Create bins for warning time analysis
    if len(early_warning_times) >= 5:
        bins = [0, 30, 60, 120, 300, float('inf')]  # time units
        bin_labels = ['0-30', '30-60', '60-120', '120-300', '300+']
        
        for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            count = sum(1 for t in early_warning_times if bin_start <= t < bin_end)
            percentage = count / len(early_warning_times) * 100
            print(f"  {bin_labels[i]} units: {count} faults ({percentage:.1f}%)")
    
    # Confidence analysis for early warnings
    if detection_details:
        confidences = [d['prediction_confidence'] for d in detection_details]
        avg_confidence = np.mean(confidences)
        median_confidence = np.median(confidences)
        
        print(f"\n🎯 EARLY WARNING CONFIDENCE:")
        print(f"  Average prediction confidence: {avg_confidence:.3f}")
        print(f"  Median prediction confidence:  {median_confidence:.3f}")
        
        # Analyze relationship between warning time and confidence
        high_conf_warnings = [d['warning_time_units'] for d in detection_details if d['prediction_confidence'] >= 0.7]
        low_conf_warnings = [d['warning_time_units'] for d in detection_details if d['prediction_confidence'] < 0.7]
        
        if high_conf_warnings:
            print(f"  High confidence warnings (≥0.7): {len(high_conf_warnings)} (avg time: {np.mean(high_conf_warnings):.1f} units)")
        if low_conf_warnings:
            print(f"  Low confidence warnings (<0.7): {len(low_conf_warnings)} (avg time: {np.mean(low_conf_warnings):.1f} units)")

else:
    print("⚠️  No early warnings detected - model does not provide advance fault prediction")

print(f"\n=== BUSINESS VALUE OF EARLY WARNING ===")
if early_warning_times and len(early_warning_times) > 0:
    # Calculate business value
    maintenance_hours_needed = 4  # Hours needed to perform preventive maintenance
    maintenance_cost_per_hour = 500  # Cost of maintenance team per hour
    downtime_cost_per_hour = 5000   # Cost of unplanned downtime per hour
    
    avg_warning_hours = avg_warning_time * time_unit_minutes / 60
    median_warning_hours = median_warning_time * time_unit_minutes / 60
    
    if avg_warning_hours >= maintenance_hours_needed:
        print(f"✅ EXCELLENT: Average warning time ({avg_warning_hours:.1f}h) allows planned maintenance")
        potential_savings_per_fault = downtime_cost_per_hour * 8 - maintenance_cost_per_hour * maintenance_hours_needed
        total_potential_savings = potential_savings_per_fault * len(early_warning_times)
        print(f"   Potential savings per prevented fault: ${potential_savings_per_fault:,.0f}")
        print(f"   Total potential savings: ${total_potential_savings:,.0f}")
    elif median_warning_hours >= maintenance_hours_needed:
        print(f"🟡 GOOD: Median warning time ({median_warning_hours:.1f}h) usually allows planned maintenance")
    else:
        print(f"⚠️  WARNING: Short warning times may not allow sufficient maintenance planning")
        print(f"   Average warning: {avg_warning_hours:.1f}h, Maintenance needed: {maintenance_hours_needed}h")

print(f"\n=== EARLY WARNING RECOMMENDATIONS ===")
if early_warning_times and len(early_warning_times) > 0:
    if len(early_warning_times) / len(all_faults) < 0.8:
        print("🔧 TO IMPROVE EARLY DETECTION RATE:")
        print("  - Lower prediction threshold for earlier warnings")
        print("  - Increase window size to capture longer-term patterns")
        print("  - Add more sensitive anomaly detection features")
    
    if avg_warning_time < 100:  # If average warning is less than 100 time units
        print("🔧 TO INCREASE WARNING TIME:")
        print("  - Extend prefault_horizon parameter")
        print("  - Use longer sliding windows")
        print("  - Add trend analysis features")
        print("  - Consider multiple severity levels of alerts")
    
    if np.std(early_warning_times) > avg_warning_time * 0.5:
        print("🔧 TO IMPROVE WARNING CONSISTENCY:")
        print("  - Standardize feature scaling")
        print("  - Use ensemble methods for more stable predictions")
        print("  - Add temporal smoothing to predictions")
else:
    print("🚨 CRITICAL: No early warnings detected!")
    print("  - Check if fault labeling is correct")
    print("  - Verify prefault_horizon parameter")
    print("  - Consider using more sensitive anomaly detection")
    print("  - Review window size and step size parameters")

# === FALSE ALARM RATE PER DAY ANALYSIS ===
print(f"\n=== FALSE ALARM RATE PER DAY ANALYSIS ===")

def calculate_false_alarm_rate_per_day(y_test, y_pred, step_size, samples_per_day=None):
    """
    Calculate false alarm rate per day for operational planning
    
    Parameters:
    - y_test: True labels
    - y_pred: Predicted labels  
    - step_size: Time units between each prediction window
    - samples_per_day: Number of prediction samples per day (auto-calculate if None)
    """
    
    # Calculate false positives
    false_positives = np.sum((y_test == 0) & (y_pred == 1))
    total_normal_samples = np.sum(y_test == 0)
    total_samples = len(y_test)
    
    # Estimate samples per day if not provided
    if samples_per_day is None:
        # Assuming SCADA data every 10 minutes and step_size represents this interval
        minutes_per_sample = 10 * step_size  # Adjust based on your data frequency
        samples_per_day = (24 * 60) / minutes_per_sample
    
    # Calculate days covered by test set
    days_in_test_set = total_samples / samples_per_day
    
    # Calculate false alarm rate per day
    false_alarms_per_day = false_positives / days_in_test_set
    
    print(f"📊 FALSE ALARM ANALYSIS:")
    print(f"  Total false positives: {false_positives}")
    print(f"  Total normal samples: {total_normal_samples}")
    print(f"  Total test samples: {total_samples}")
    print(f"  Estimated samples per day: {samples_per_day:.1f}")
    print(f"  Days covered by test set: {days_in_test_set:.1f}")
    print(f"  ")
    print(f"🚨 FALSE ALARM RATE PER DAY: {false_alarms_per_day:.2f}")
    
    # Convert to different time periods for context
    false_alarms_per_week = false_alarms_per_day * 7
    false_alarms_per_month = false_alarms_per_day * 30
    false_alarms_per_year = false_alarms_per_day * 365
    
    print(f"  False alarms per week: {false_alarms_per_week:.1f}")
    print(f"  False alarms per month: {false_alarms_per_month:.1f}")
    print(f"  False alarms per year: {false_alarms_per_year:.0f}")
    
    # Operational impact analysis
    print(f"\n⚙️  OPERATIONAL IMPACT:")
    
    # Time impact (assuming each false alarm requires investigation)
    investigation_time_minutes = 30  # Time to investigate each false alarm
    daily_investigation_time = false_alarms_per_day * investigation_time_minutes
    weekly_investigation_time = daily_investigation_time * 7
    
    print(f"  Daily investigation time: {daily_investigation_time:.1f} minutes ({daily_investigation_time/60:.1f} hours)")
    print(f"  Weekly investigation time: {weekly_investigation_time:.1f} minutes ({weekly_investigation_time/60:.1f} hours)")
    
    # Cost impact
    cost_per_false_alarm = 500  # Cost of unnecessary maintenance/investigation
    daily_false_alarm_cost = false_alarms_per_day * cost_per_false_alarm
    monthly_false_alarm_cost = daily_false_alarm_cost * 30
    yearly_false_alarm_cost = daily_false_alarm_cost * 365
    
    print(f"\n💰 COST IMPACT (at ${cost_per_false_alarm}/false alarm):")
    print(f"  Daily cost: ${daily_false_alarm_cost:.0f}")
    print(f"  Monthly cost: ${monthly_false_alarm_cost:.0f}")
    print(f"  Yearly cost: ${yearly_false_alarm_cost:.0f}")
    
    # Workload impact
    maintenance_team_capacity = 3  # Number of maintenance tasks team can handle per day
    workload_percentage = (false_alarms_per_day / maintenance_team_capacity) * 100
    
    print(f"\n👥 MAINTENANCE TEAM IMPACT:")
    print(f"  Team capacity: {maintenance_team_capacity} tasks/day")
    print(f"  False alarm workload: {workload_percentage:.1f}% of daily capacity")
    
    if workload_percentage > 50:
        print(f"  ⚠️  WARNING: False alarms consume >50% of maintenance capacity!")
    elif workload_percentage > 25:
        print(f"  🟡 MODERATE: False alarms consume >25% of maintenance capacity")
    else:
        print(f"  ✅ ACCEPTABLE: False alarms consume <25% of maintenance capacity")
    
    # Industry benchmarks and assessment
    print(f"\n📏 INDUSTRY BENCHMARK COMPARISON:")
    
    if false_alarms_per_day <= 1:
        print(f"  ✅ EXCELLENT: ≤1 false alarm/day (industry best practice)")
        performance_rating = "Excellent"
    elif false_alarms_per_day <= 3:
        print(f"  ✅ GOOD: 1-3 false alarms/day (acceptable for most operations)")
        performance_rating = "Good"
    elif false_alarms_per_day <= 5:
        print(f"  🟡 FAIR: 3-5 false alarms/day (manageable but could be improved)")
        performance_rating = "Fair"
    elif false_alarms_per_day <= 10:
        print(f"  ⚠️  POOR: 5-10 false alarms/day (high operational burden)")
        performance_rating = "Poor"
    else:
        print(f"  ❌ UNACCEPTABLE: >10 false alarms/day (unsustainable)")
        performance_rating = "Unacceptable"
    
    # Return metrics for further analysis
    return {
        'false_alarms_per_day': false_alarms_per_day,
        'false_alarms_per_week': false_alarms_per_week,
        'false_alarms_per_month': false_alarms_per_month,
        'false_alarms_per_year': false_alarms_per_year,
        'daily_investigation_time_minutes': daily_investigation_time,
        'daily_cost': daily_false_alarm_cost,
        'yearly_cost': yearly_false_alarm_cost,
        'workload_percentage': workload_percentage,
        'performance_rating': performance_rating,
        'days_in_test_set': days_in_test_set,
        'total_false_positives': false_positives
    }

# Calculate false alarm rate per day
false_alarm_metrics = calculate_false_alarm_rate_per_day(
    y_test, y_pred, step_size=3, samples_per_day=None  # Auto-calculate based on step_size
)

print(f"\n🔧 FALSE ALARM REDUCTION RECOMMENDATIONS:")

if false_alarm_metrics['false_alarms_per_day'] > 3:
    print("  HIGH PRIORITY ACTIONS:")
    print("  • Increase prediction threshold to reduce sensitivity")
    print("  • Implement prediction confidence filtering (e.g., only alert if confidence >0.8)")
    print("  • Add temporal smoothing (require multiple consecutive predictions)")
    print("  • Review and remove noisy or irrelevant features")
    print("  • Consider ensemble voting (require multiple models to agree)")
    
if false_alarm_metrics['false_alarms_per_day'] > 1:
    print("  MEDIUM PRIORITY ACTIONS:")
    print("  • Implement alert suppression during known maintenance windows")
    print("  • Add context-aware filtering (weather, operational state)")
    print("  • Tune hyperparameters to optimize precision")
    print("  • Consider cost-sensitive learning with higher penalty for false positives")

if false_alarm_metrics['workload_percentage'] > 30:
    print("  OPERATIONAL PROCESS IMPROVEMENTS:")
    print("  • Implement automated first-level alert filtering")
    print("  • Create alert priority levels (high/medium/low)")
    print("  • Develop rapid investigation protocols")
    print("  • Consider increasing maintenance team capacity")

print(f"\n📋 ALERT SYSTEM CONFIGURATION SUGGESTIONS:")
print(f"  Current threshold optimization needed: {false_alarm_metrics['performance_rating']}")

# Suggest optimal threshold based on false alarm rate
current_threshold = 0.5  # Default threshold
suggested_thresholds = []

if false_alarm_metrics['false_alarms_per_day'] > 5:
    suggested_thresholds.extend([0.7, 0.8, 0.9])
elif false_alarm_metrics['false_alarms_per_day'] > 3:
    suggested_thresholds.extend([0.6, 0.7])
elif false_alarm_metrics['false_alarms_per_day'] > 1:
    suggested_thresholds.append(0.6)

if suggested_thresholds:
    print(f"  Suggested threshold values to test: {suggested_thresholds}")
    print(f"  Current threshold: {current_threshold}")
    print(f"  • Higher thresholds = fewer false alarms but might miss some faults")
    print(f"  • Test each threshold and measure impact on both false alarms and fault detection")

# Save false alarm analysis to file
false_alarm_summary = {
    'analysis_date': pd.Timestamp.now().isoformat(),
    'model_type': 'RandomForest',
    'test_samples': len(y_test),
    'metrics': false_alarm_metrics,
    'recommendations': {
        'performance_rating': false_alarm_metrics['performance_rating'],
        'requires_immediate_action': false_alarm_metrics['false_alarms_per_day'] > 5,
        'suggested_thresholds': suggested_thresholds,
        'operational_impact': 'High' if false_alarm_metrics['workload_percentage'] > 50 else 'Medium' if false_alarm_metrics['workload_percentage'] > 25 else 'Low'
    }
}

try:
    import json
    with open('false_alarm_analysis.json', 'w') as f:
        json.dump(false_alarm_summary, f, indent=2)
    print(f"\n💾 False alarm analysis saved to: false_alarm_analysis.json")
except Exception as e:
    print(f"⚠️  Could not save analysis to file: {e}")

print("="*60)

# === CUSTOM THRESHOLD PREDICTION ===
def predict_with_threshold(model, X, threshold=0.3):
    """Make predictions with a custom threshold"""
    y_proba = model.predict_proba(X)[:, 1]  # Get probability of positive class
    return (y_proba >= threshold).astype(int), y_proba

# Use lower threshold for better fault detection
CUSTOM_THRESHOLD = 0.3  # Lower threshold for higher sensitivity
print(f"Using custom threshold: {CUSTOM_THRESHOLD}")

y_pred_custom, y_proba = predict_with_threshold(model, X_test, CUSTOM_THRESHOLD)

# Compare with default threshold
y_pred_default = model.predict(X_test)
y_proba_default = model.predict_proba(X_test)[:, 1]

print(f"\nThreshold Comparison:")
print(f"Default (0.5): {np.sum(y_pred_default)} predictions as fault")
print(f"Custom ({CUSTOM_THRESHOLD}): {np.sum(y_pred_custom)} predictions as fault")

# Use custom predictions for evaluation
y_pred = y_pred_custom
y_proba = y_proba_default  # Keep original probabilities for analysis

# === THRESHOLD OPTIMIZATION ===
def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find optimal threshold based on specified metric"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    scores = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred_thresh, zero_division=0)
        
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    
    return optimal_threshold, optimal_score, thresholds, scores

# Find optimal thresholds for different metrics
optimal_f1_thresh, optimal_f1_score, thresholds, f1_scores = find_optimal_threshold(y_test, y_proba, 'f1')
optimal_recall_thresh, optimal_recall_score, _, recall_scores = find_optimal_threshold(y_test, y_proba, 'recall')

print(f"\n=== THRESHOLD OPTIMIZATION RESULTS ===")
print(f"Optimal F1 threshold: {optimal_f1_thresh:.2f} (F1: {optimal_f1_score:.3f})")
print(f"Optimal Recall threshold: {optimal_recall_thresh:.2f} (Recall: {optimal_recall_score:.3f})")

# Recommend threshold for fault detection (prioritize recall)
recommended_threshold = min(0.4, optimal_recall_thresh)  # Cap at 0.4 for safety
print(f"Recommended threshold for fault detection: {recommended_threshold:.2f}")

# === MULTIPLE THRESHOLD ANALYSIS ===
def evaluate_multiple_thresholds(y_true, y_proba, thresholds=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
    """Evaluate model performance at multiple thresholds"""
    results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_thresh)
        precision = precision_score(y_true, y_pred_thresh, zero_division=0)
        recall = recall_score(y_true, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
        
        # Calculate confusion matrix components
        cm = confusion_matrix(y_true, y_pred_thresh)
        print(f"\nConfusion Matrix for threshold {threshold:.1f}:")
        print(cm)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = cm[0,0], 0, 0, 0
            
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    return results

# Evaluate different thresholds
threshold_results = evaluate_multiple_thresholds(y_test, y_proba)

print(f"\n=== THRESHOLD COMPARISON TABLE ===")
print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<8} {'F1':<8} {'FP':<4} {'FN':<4}")
print("-" * 50)

for result in threshold_results:
    print(f"{result['threshold']:<10.1f} {result['precision']:<10.3f} {result['recall']:<8.3f} "
          f"{result['f1']:<8.3f} {result['fp']:<4} {result['fn']:<4}")

# Highlight best thresholds
best_recall = max(threshold_results, key=lambda x: x['recall'])
best_f1 = max(threshold_results, key=lambda x: x['f1'])

print(f"\nBest for Recall (catching faults): {best_recall['threshold']:.1f} (Recall: {best_recall['recall']:.3f})")
print(f"Best for F1 (balance): {best_f1['threshold']:.1f} (F1: {best_f1['f1']:.3f})")

# === DAILY FAULT PREDICTION ANALYSIS ===
def create_sliding_windows_with_timestamps(df, event_start_id=None, event_end_id=None,
                           window_size=12, step_size=6, prefault_horizon=18):
    """
    Modified version of create_sliding_windows that preserves timestamp information
    Returns windows, labels, and corresponding timestamp information
    """
    df = df.sort_values("id").reset_index(drop=True)
    feature_cols = [col for col in df.columns if col.startswith("sensor_") or col.startswith("wind_") or col.startswith("power_") or col.startswith("reactive_")]
    df.dropna(inplace=True)
    df = df[df['status_type_id'] != '2']  # Exclude fault status
    df = df[df['status_type_id'] != '3']  # Exclude fault status
    df = df[df['status_type_id'] != '4']  # Exclude fault status
    df_prediction = df.copy().reset_index(drop=True)

    # Check if timestamp column exists, if not create one based on id
    timestamp_col = None
    for col in ['timestamp', 'datetime', 'time', 'date']:
        if col in df_prediction.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        # Create synthetic timestamps based on id (assuming 10-minute intervals)
        base_time = pd.Timestamp('2023-01-01 00:00:00')
        df_prediction['synthetic_timestamp'] = base_time + pd.to_timedelta(df_prediction['id'] * 10, unit='minutes')
        timestamp_col = 'synthetic_timestamp'

    anomaly_ids = set(range(event_start_id - prefault_horizon, event_start_id)) if event_start_id else set()
    fault_ids = set(range(event_start_id, event_end_id)) if event_start_id else set()

    X_windows = []
    y_labels = []
    window_timestamps = []
    window_metadata = []

    for start in range(0, len(df_prediction) - window_size + 1, step_size):
        end = start + window_size
        window = df_prediction.iloc[start:end]
        window_ids = set(window["id"])

        if int(len(fault_ids.intersection(window_ids))) > 0:
            continue

        label = int(len(anomaly_ids.intersection(window_ids)) > 0)
        X = window[feature_cols].values
        
        # Get timestamp info for the window (use the end of the window for prediction time)
        window_end_timestamp = window.iloc[-1][timestamp_col]
        window_start_timestamp = window.iloc[0][timestamp_col]
        window_center_timestamp = window_start_timestamp + (window_end_timestamp - window_start_timestamp) / 2
        
        X_windows.append(X)
        y_labels.append(label)
        window_timestamps.append(window_end_timestamp)  # When the prediction is made
        
        window_metadata.append({
            'window_start_id': window.iloc[0]['id'],
            'window_end_id': window.iloc[-1]['id'],
            'window_start_time': window_start_timestamp,
            'window_end_time': window_end_timestamp,
            'window_center_time': window_center_timestamp,
            'prediction_time': window_end_timestamp
        })

    return X_windows, y_labels, window_timestamps, window_metadata

def create_test_sliding_windows_with_timestamps(df, event_start_id=None,
                           window_size=12, step_size=6, prefault_horizon=18):
    """
    Modified version of create_test_sliding_windows that preserves timestamp information
    """
    df = df.sort_values("id").reset_index(drop=True)
    feature_cols = [col for col in df.columns if col.startswith("sensor_") or col.startswith("wind_") or col.startswith("power_") or col.startswith("reactive_")]
    df.dropna(inplace=True)
    df_prediction = df.copy().reset_index(drop=True)

    # Check if timestamp column exists, if not create one based on id
    timestamp_col = None
    for col in ['timestamp', 'datetime', 'time', 'date']:
        if col in df_prediction.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        # Create synthetic timestamps based on id (assuming 10-minute intervals)
        base_time = pd.Timestamp('2023-01-01 00:00:00')
        df_prediction['synthetic_timestamp'] = base_time + pd.to_timedelta(df_prediction['id'] * 10, unit='minutes')
        timestamp_col = 'synthetic_timestamp'

    # Check if event_start_id is smaller than the first data point
    anomaly_before_data = event_start_id is not None and event_start_id > df_prediction['id'].iloc[0] and event_start_id < df_prediction['id'].iloc[-1]
    label = 0
    if anomaly_before_data:
        label = 1

    X_windows = []
    y_labels = []
    window_timestamps = []
    window_metadata = []

    for start in range(0, len(df_prediction) - window_size + 1, step_size):
        end = start + window_size
        window = df_prediction.iloc[start:end]

        X = window[feature_cols].values
        
        # Get timestamp info for the window
        window_end_timestamp = window.iloc[-1][timestamp_col]
        window_start_timestamp = window.iloc[0][timestamp_col]
        window_center_timestamp = window_start_timestamp + (window_end_timestamp - window_start_timestamp) / 2
        
        X_windows.append(X)
        y_labels.append(label)
        window_timestamps.append(window_end_timestamp)
        
        window_metadata.append({
            'window_start_id': window.iloc[0]['id'],
            'window_end_id': window.iloc[-1]['id'],
            'window_start_time': window_start_timestamp,
            'window_end_time': window_end_timestamp,
            'window_center_time': window_center_timestamp,
            'prediction_time': window_end_timestamp
        })

    return X_windows, y_labels, window_timestamps, window_metadata

def analyze_daily_fault_predictions(y_pred, window_timestamps, window_metadata, threshold_results=None):
    """
    Analyze and display the number of fault predictions per day with dates
    
    Parameters:
    - y_pred: Array of predictions (0/1)
    - window_timestamps: Array of timestamps for each prediction window
    - window_metadata: List of dictionaries with window metadata
    - threshold_results: Optional results from multiple threshold analysis
    """
    
    print(f"\n" + "="*80)
    print(f"DAILY FAULT PREDICTION ANALYSIS")
    print(f"="*80)
    
    # Convert timestamps to pandas datetime if they aren't already
    timestamps = pd.to_datetime(window_timestamps)
    
    # Create DataFrame for easier manipulation
    predictions_df = pd.DataFrame({
        'timestamp': timestamps,
        'prediction': y_pred,
        'date': timestamps.dt.date
    })
    
    # Group by date and count fault predictions
    daily_predictions = predictions_df.groupby('date').agg({
        'prediction': ['sum', 'count'],
        'timestamp': ['min', 'max']
    }).round(2)
    
    # Flatten column names
    daily_predictions.columns = ['fault_predictions', 'total_predictions', 'first_prediction_time', 'last_prediction_time']
    daily_predictions = daily_predictions.reset_index()
    
    # Calculate daily statistics
    total_days = len(daily_predictions)
    days_with_faults = len(daily_predictions[daily_predictions['fault_predictions'] > 0])
    total_fault_predictions = daily_predictions['fault_predictions'].sum()
    avg_fault_predictions_per_day = daily_predictions['fault_predictions'].mean()
    max_fault_predictions_per_day = daily_predictions['fault_predictions'].max()
    
    print(f"\n📊 DAILY PREDICTION SUMMARY:")
    print(f"  Analysis period: {daily_predictions['date'].min()} to {daily_predictions['date'].max()}")
    print(f"  Total days analyzed: {total_days}")
    print(f"  Days with fault predictions: {days_with_faults} ({days_with_faults/total_days*100:.1f}%)")
    print(f"  Total fault predictions: {total_fault_predictions}")
    print(f"  Average fault predictions per day: {avg_fault_predictions_per_day:.2f}")
    print(f"  Maximum fault predictions in a day: {max_fault_predictions_per_day}")
    
    print(f"\n📅 DAILY FAULT PREDICTIONS BY DATE:")
    print(f"{'Date':<12} {'Faults':<8} {'Total Pred':<12} {'Fault %':<10} {'First Pred':<12} {'Last Pred':<12}")
    print("-" * 80)
    
    for _, row in daily_predictions.iterrows():
        fault_percentage = (row['fault_predictions'] / row['total_predictions']) * 100 if row['total_predictions'] > 0 else 0
        first_time = row['first_prediction_time'].strftime('%H:%M') if pd.notna(row['first_prediction_time']) else 'N/A'
        last_time = row['last_prediction_time'].strftime('%H:%M') if pd.notna(row['last_prediction_time']) else 'N/A'
        
        print(f"{row['date']!s:<12} {row['fault_predictions']:<8.0f} {row['total_predictions']:<12.0f} "
              f"{fault_percentage:<10.1f} {first_time:<12} {last_time:<12}")
    
    # Identify high-risk days
    high_risk_threshold = max(5, avg_fault_predictions_per_day + 2 * daily_predictions['fault_predictions'].std())
    high_risk_days = daily_predictions[daily_predictions['fault_predictions'] >= high_risk_threshold]
    
    if len(high_risk_days) > 0:
        print(f"\n🚨 HIGH-RISK DAYS (≥{high_risk_threshold:.0f} fault predictions):")
        for _, row in high_risk_days.iterrows():
            print(f"  {row['date']}: {row['fault_predictions']:.0f} fault predictions")
    else:
        print(f"\n✅ No extremely high-risk days detected (threshold: {high_risk_threshold:.0f} predictions/day)")
    
    # Weekly and monthly aggregation
    predictions_df['week'] = timestamps.dt.isocalendar().week
    predictions_df['month'] = timestamps.dt.month
    predictions_df['year'] = timestamps.dt.year
    
    weekly_predictions = predictions_df.groupby(['year', 'week'])['prediction'].sum().reset_index()
    monthly_predictions = predictions_df.groupby(['year', 'month'])['prediction'].sum().reset_index()
    
    if len(weekly_predictions) > 1:
        print(f"\n📈 WEEKLY FAULT PREDICTIONS:")
        for _, row in weekly_predictions.iterrows():
            print(f"  Year {row['year']}, Week {row['week']:2.0f}: {row['prediction']:.0f} fault predictions")
    
    if len(monthly_predictions) > 1:
        print(f"\n📊 MONTHLY FAULT PREDICTIONS:")
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for _, row in monthly_predictions.iterrows():
            month_name = month_names[int(row['month'])] if row['month'] <= 12 else f"M{row['month']}"
            print(f"  {month_name} {row['year']}: {row['prediction']:.0f} fault predictions")
    
    # Analyze patterns
    print(f"\n🔍 TEMPORAL PATTERNS:")
    
    # Day of week analysis
    predictions_df['day_of_week'] = timestamps.dt.day_name()
    dow_predictions = predictions_df.groupby('day_of_week')['prediction'].mean().sort_values(ascending=False)
    
    print(f"  Average fault predictions by day of week:")
    for day, avg_preds in dow_predictions.items():
        print(f"    {day}: {avg_preds:.2f}")
    
    # Hour of day analysis (if we have enough temporal resolution)
    if len(predictions_df) > 24:  # Only if we have enough data points
        predictions_df['hour'] = timestamps.dt.hour
        hourly_predictions = predictions_df.groupby('hour')['prediction'].mean()
        
        peak_hours = hourly_predictions.nlargest(3)
        if peak_hours.max() > 0:
            print(f"  Peak fault prediction hours:")
            for hour, avg_preds in peak_hours.items():
                print(f"    {hour:02d}:00 - {avg_preds:.3f} avg predictions")
    
    # Return summary for further analysis
    summary = {
        'daily_predictions': daily_predictions,
        'total_days': total_days,
        'days_with_faults': days_with_faults,
        'total_fault_predictions': total_fault_predictions,
        'avg_fault_predictions_per_day': avg_fault_predictions_per_day,
        'max_fault_predictions_per_day': max_fault_predictions_per_day,
        'high_risk_days': high_risk_days,
        'weekly_predictions': weekly_predictions,
        'monthly_predictions': monthly_predictions
    }
    
    return summary

def compare_daily_predictions_across_thresholds(y_proba, window_timestamps, thresholds=[0.1, 0.3, 0.5, 0.7]):
    """
    Compare daily fault predictions across different threshold values
    """
    print(f"\n" + "="*80)
    print(f"DAILY PREDICTIONS COMPARISON ACROSS THRESHOLDS")
    print(f"="*80)
    
    timestamps = pd.to_datetime(window_timestamps)
    
    # Create predictions for each threshold
    threshold_results = {}
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        predictions_df = pd.DataFrame({
            'timestamp': timestamps,
            'prediction': y_pred_thresh,
            'date': timestamps.dt.date
        })
        
        daily_predictions = predictions_df.groupby('date')['prediction'].sum().reset_index()
        daily_predictions.columns = ['date', 'fault_predictions']
        
        threshold_results[threshold] = daily_predictions
    
    # Create combined comparison table
    all_dates = sorted(set().union(*[df['date'].tolist() for df in threshold_results.values()]))
    
    print(f"\n📊 DAILY FAULT PREDICTIONS BY THRESHOLD:")
    header = f"{'Date':<12}"
    for threshold in thresholds:
        header += f" {f'T={threshold:.1f}':<8}"
    print(header)
    print("-" * (12 + 8 * len(thresholds)))
    
    for date in all_dates:
        row = f"{date!s:<12}"
        for threshold in thresholds:
            df = threshold_results[threshold]
            predictions = df[df['date'] == date]['fault_predictions'].sum() if len(df[df['date'] == date]) > 0 else 0
            row += f" {predictions:<8.0f}"
        print(row)
    
    # Summary statistics by threshold
    print(f"\n📈 THRESHOLD COMPARISON SUMMARY:")
    print(f"{'Threshold':<10} {'Avg/Day':<10} {'Max/Day':<10} {'Total':<8} {'Days>0':<8}")
    print("-" * 48)
    
    for threshold in thresholds:
        df = threshold_results[threshold]
        avg_per_day = df['fault_predictions'].mean()
        max_per_day = df['fault_predictions'].max()
        total_predictions = df['fault_predictions'].sum()
        days_with_predictions = len(df[df['fault_predictions'] > 0])
        
        print(f"{threshold:<10.1f} {avg_per_day:<10.2f} {max_per_day:<10.0f} {total_predictions:<8.0f} {days_with_predictions:<8}")
    
    return threshold_results

# === END OF DAILY FAULT PREDICTION ANALYSIS ===

# === IMPLEMENT DAILY FAULT PREDICTION TRACKING ===
print(f"\n" + "="*80)
print(f"IMPLEMENTING DAILY FAULT PREDICTION ANALYSIS")
print(f"="*80)

# Note: The current analysis uses pre-processed windowed data without timestamp preservation
# To get daily predictions with dates, we need to re-process a sample dataset with timestamp tracking

# Let's demonstrate with the current predictions and synthetic timestamps
print(f"📝 Current Analysis Limitation:")
print(f"  The current pipeline processes data through multiple stages that lose timestamp information.")
print(f"  To show daily predictions with actual dates, we need to:")
print(f"  1. Re-process source data with timestamp preservation")
print(f"  2. Track timestamps through window creation")
print(f"  3. Map predictions back to actual dates")

# Create synthetic timestamps for demonstration (since we don't have them in current processing)
print(f"\n🔧 Creating Demonstration with Synthetic Timestamps:")

# Estimate timestamps based on test set size and typical SCADA frequency
total_test_samples = len(y_test)
estimated_days = total_test_samples / (24 * 6)  # Assuming 6 samples per hour (10-min intervals)

# Create synthetic timestamps spanning the estimated period
start_date = pd.Timestamp('2023-01-01 00:00:00')
time_intervals = pd.timedelta_range(start='0 minutes', periods=total_test_samples, freq='10min')
synthetic_timestamps = [start_date + interval for interval in time_intervals]

print(f"  Test samples: {total_test_samples}")
print(f"  Estimated analysis period: {estimated_days:.1f} days")
print(f"  Synthetic date range: {start_date.date()} to {(start_date + time_intervals[-1]).date()}")

# Create synthetic window metadata for demonstration
synthetic_metadata = []
for i, timestamp in enumerate(synthetic_timestamps):
    synthetic_metadata.append({
        'window_start_id': i * 3,
        'window_end_id': (i * 3) + 18,
        'window_start_time': timestamp - pd.Timedelta(minutes=90),  # 18 * 10min window
        'window_end_time': timestamp,
        'window_center_time': timestamp - pd.Timedelta(minutes=45),
        'prediction_time': timestamp
    })

# Analyze daily predictions using current model results
print(f"\n📊 Analyzing Daily Fault Predictions (with synthetic timestamps):")
daily_analysis = analyze_daily_fault_predictions(
    y_pred_custom, 
    synthetic_timestamps, 
    synthetic_metadata
)

# Compare across different thresholds
print(f"\n🔍 Comparing Daily Predictions Across Multiple Thresholds:")
threshold_comparison = compare_daily_predictions_across_thresholds(
    y_proba, 
    synthetic_timestamps, 
    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]
)

# === RECOMMENDATIONS FOR PRODUCTION IMPLEMENTATION ===
print(f"\n" + "="*80)
print(f"PRODUCTION IMPLEMENTATION RECOMMENDATIONS")
print(f"="*80)

print(f"""
🚀 TO IMPLEMENT REAL-TIME DAILY PREDICTION TRACKING:

1. MODIFY DATA INGESTION:
   - Preserve timestamp columns during CSV loading
   - Ensure datetime parsing: pd.read_csv(file, parse_dates=['timestamp'])
   - Handle timezone information if needed

2. UPDATE SLIDING WINDOW FUNCTIONS:
   - Use create_sliding_windows_with_timestamps() instead of current functions
   - Pass timestamp information through the entire pipeline
   - Store window metadata alongside features

3. MODIFY MODEL TRAINING:
   - Track timestamps during train/test split
   - Ensure temporal ordering is preserved
   - Consider time-based splitting instead of random splitting

4. IMPLEMENT REAL-TIME MONITORING:
   - Create daily aggregation pipeline
   - Set up automated reporting
   - Add alert thresholds for unusual prediction patterns

5. EXAMPLE PRODUCTION CODE:
   ```python
   # Load data with timestamps
   df = pd.read_csv('turbine_data.csv', parse_dates=['timestamp'])
   
   # Create windows with timestamp tracking
   X, y, timestamps, metadata = create_sliding_windows_with_timestamps(
       df, event_start_id, event_end_id, window_size=18, step_size=3
   )
   
   # Make predictions
   predictions = model.predict(X_scaled)
   
   # Analyze daily patterns
   daily_summary = analyze_daily_fault_predictions(
       predictions, timestamps, metadata
   )
   ```

6. DASHBOARD INTEGRATION:
   - Daily prediction counts by date
   - Weekly/monthly trend analysis
   - Real-time alert system
   - Historical pattern comparison
""")

# === SAVE DAILY ANALYSIS RESULTS ===
try:
    # Save synthetic daily analysis to file
    daily_summary_data = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'model_info': {
            'model_type': 'XGBoost',
            'test_samples': len(y_test),
            'custom_threshold': CUSTOM_THRESHOLD
        },
        'daily_statistics': {
            'total_days': int(daily_analysis['total_days']),
            'days_with_faults': int(daily_analysis['days_with_faults']),
            'total_fault_predictions': int(daily_analysis['total_fault_predictions']),
            'avg_fault_predictions_per_day': float(daily_analysis['avg_fault_predictions_per_day']),
            'max_fault_predictions_per_day': int(daily_analysis['max_fault_predictions_per_day'])
        },
        'note': 'This analysis uses synthetic timestamps for demonstration. Production implementation should use actual timestamp data.'
    }
    
    import json
    with open('daily_fault_predictions_analysis.json', 'w') as f:
        json.dump(daily_summary_data, f, indent=2)
    print(f"\n💾 Daily analysis summary saved to: daily_fault_predictions_analysis.json")
    
except Exception as e:
    print(f"⚠️  Could not save daily analysis to file: {e}")

print(f"\n✅ Daily Fault Prediction Analysis Complete!")
print(f"="*80)