# Early Fault Detection Model using SCADA Data

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
    rows_to_keep = ~((df['time_stamp'] > start_time) & (df['time_stamp'] <= end_time))
    df = df[rows_to_keep]
    pre_fault_window = (df['time_stamp'] >= start_time - pd.Timedelta(days=5)) & (df['time_stamp'] < start_time)
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

# Dictionary to store different models
models = {
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    # 'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    # 'Extra Trees': ExtraTreesClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    # 'SVM': SVC(class_weight='balanced', random_state=42),
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, n_jobs=-1),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Store results for comparison
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }

# Compare all models
print("\n=== MODEL COMPARISON ===")
import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df.round(4))

# --- Debug: Data distribution ---
# print("\n=== DATA ANALYSIS ===")
# print(f"Training set fault ratio: {y_train.mean():.4f}")
# print(f"Test set fault ratio: {y_test.mean():.4f}")
# print(f"Training set size: {len(X_train)}")
# print(f"Test set size: {len(X_test)}")

# print(f"\nActual test set distribution:")
# print(f"  Actual 0 (no fault): {(y_test == 0).sum()}")
# print(f"  Actual 1 (fault): {(y_test == 1).sum()}")

# # Check for potential data leakage
# print(f"\nAvailable features: {list(X_train.columns)}")
# print(f"Any 'fault' related features: {[col for col in X_train.columns if 'fault' in col.lower()]}")

# # Show feature importance for Random Forest
# if 'Random Forest' in results:
#     rf_model = models['Random Forest']
#     feature_names = X_train.columns
#     importances = rf_model.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     print(f"\nRandom Forest Feature Importance (top 5):")
#     for i in range(min(5, len(feature_names))):
#         print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
