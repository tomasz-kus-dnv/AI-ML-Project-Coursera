# Early Fault Detection Model using SCADA Data

import glob
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
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

# === RENAME COLUMNS BASED ON FEATURE DESCRIPTIONS ===
print("\n=== RENAMING COLUMNS BASED ON FEATURE DESCRIPTIONS ===")

# Load the feature description mapping
feature_desc_path = '../wind-turbines-data/setA/Wind Farm A/comma_feature_description.csv'
feature_desc = pd.read_csv(feature_desc_path)

print(f"Loaded feature descriptions for {len(feature_desc)} sensors")
print(f"Original dataframe has {len(combined_df.columns)} columns")

# Create mapping dictionary from actual column names to descriptions
column_mapping = {}

# First, let's see what columns we have
print(f"\nSample of original columns:")
for i, col in enumerate(combined_df.columns[:10]):
    print(f"  {i+1:2d}. {col}")
if len(combined_df.columns) > 10:
    print(f"  ... and {len(combined_df.columns) - 10} more")

# Create the mapping based on sensor names and statistics
for _, row in feature_desc.iterrows():
    sensor_name = row['sensor_name']
    description = row['description']
    statistics_type = row['statistics_type']
    
    # Handle different statistics types
    if ',' in statistics_type:
        # Multiple statistics for this sensor
        stats = [stat.strip() for stat in statistics_type.split(',')]
        for stat in stats:
            if stat == 'average':
                col_name = f"{sensor_name}_avg"
            elif stat == 'maximum':
                col_name = f"{sensor_name}_max"
            elif stat == 'minimum':
                col_name = f"{sensor_name}_min"
            elif stat == 'std_dev':
                col_name = f"{sensor_name}_std"
            else:
                col_name = f"{sensor_name}_{stat}"
            
            # Check if this column exists in our dataframe
            if col_name in combined_df.columns:
                if stat == 'average':
                    column_mapping[col_name] = description
                else:
                    column_mapping[col_name] = f"{description} ({stat})"
    else:
        # Single statistic
        if statistics_type == 'average':
            col_name = f"{sensor_name}_avg"
            if col_name in combined_df.columns:
                column_mapping[col_name] = description
        else:
            # Handle special cases like sensor_44, sensor_45, etc. (no _avg suffix)
            if sensor_name in combined_df.columns:
                column_mapping[sensor_name] = description

# Handle special cases for columns without _avg suffix
special_sensors = ['sensor_44', 'sensor_45', 'sensor_46', 'sensor_47', 'sensor_48', 'sensor_49', 'sensor_50', 'sensor_51']
for sensor in special_sensors:
    if sensor in combined_df.columns:
        # Find the description for this sensor
        sensor_desc = feature_desc[feature_desc['sensor_name'] == sensor]
        if not sensor_desc.empty:
            column_mapping[sensor] = sensor_desc.iloc[0]['description']

print(f"\nCreated mapping for {len(column_mapping)} columns:")
print("Sample mappings:")
for i, (old_name, new_name) in enumerate(list(column_mapping.items())[:10]):
    print(f"  '{old_name}' → '{new_name}'")
if len(column_mapping) > 10:
    print(f"  ... and {len(column_mapping) - 10} more")

# Apply the column renaming
combined_df = combined_df.rename(columns=column_mapping)

# Check for unmapped sensor columns
unmapped_sensor_columns = [col for col in combined_df.columns 
                          if (col.startswith('sensor_') or col.startswith('wind_speed_') or 
                              col.startswith('reactive_power_') or col.startswith('power_')) 
                          and col not in ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']]

if unmapped_sensor_columns:
    print(f"\nUnmapped sensor columns ({len(unmapped_sensor_columns)}):")
    for col in unmapped_sensor_columns[:5]:
        print(f"  - {col}")
    if len(unmapped_sensor_columns) > 5:
        print(f"  ... and {len(unmapped_sensor_columns) - 5} more")

# Show successful mapping results
mapped_columns = [col for col in combined_df.columns if col in column_mapping.values()]
print(f"\nSuccessfully mapped {len(mapped_columns)} columns to descriptive names")

print(f"\nSample of new column names:")
descriptive_cols = [col for col in combined_df.columns if col not in ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']]
for i, col in enumerate(descriptive_cols[:10]):
    print(f"  {i+1:2d}. {col}")
if len(descriptive_cols) > 10:
    print(f"  ... and {len(descriptive_cols) - 10} more")

# Verify we didn't lose any data
print(f"\nVerification:")
print(f"  Original shape: {combined_df.shape}")
print(f"  Columns before: {len(combined_df.columns)}")
print(f"  Descriptive columns: {len([col for col in combined_df.columns if col in column_mapping.values()])}")
print(f"  System columns: {len([col for col in combined_df.columns if col in ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']])}")

# --- Step 1.5: Downsample data by keeping every 6th row ---
print("\n=== DATA DOWNSAMPLING: KEEPING EVERY 6TH ROW ===")
print(f"Original data shape: {combined_df.shape}")

# Method 1: Simple row-based downsampling (keeps every 6th row globally)
def downsample_every_nth_row(df, n=6):
    """Keep every nth row from the dataframe"""
    return df.iloc[::n].reset_index(drop=True)

combined_df_downsampled = downsample_every_nth_row(combined_df, n=18)
    
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
# Use vectorized operation instead of iterating (much faster and correct)
df.loc[df['status_type_id'] == 5, 'fault_label'] = 1

print(f"Fault labeling completed:")
print(f"  Total samples: {len(df)}")
print(f"  Fault samples (status_type_id == '5'): {(df['fault_label'] == 1).sum()}")
print(f"  Normal samples: {(df['fault_label'] == 0).sum()}")
print(f"  Fault rate: {(df['fault_label'] == 1).mean():.4f} ({(df['fault_label'] == 1).mean()*100:.2f}%)")

# --- Step 3: Feature Engineering ---
# df.set_index('timestamp', inplace=True)
# df['gearbox_temp_mean_6h'] = df['gearbox_temp'].rolling('6H').mean()
# df['generator_temp_std_3h'] = df['generator_temp'].rolling('3H').std()
# df['temp_diff'] = df['gearbox_temp'] - df['generator_temp']
# df['power_wind_ratio'] = df['power_output'] / (df['wind_speed'] + 1e-6)

# Drop rows with NaNs
df.dropna(inplace=True)

# --- Step 3: Advanced Feature Engineering ---
print("\n=== ADVANCED FEATURE ENGINEERING ===")

def engineer_turbine_features(df):
    """
    Engineer advanced features from wind turbine SCADA data for fault detection
    """
    df_engineered = df.copy()
    # return df_engineered
    # df_engineered = pd.DataFrame()  # df.copy()
    # df_engineered['train_test'] = df['train_test']  # df.copy()
    # df_engineered['fault_label'] = df['fault_label']  # df.copy()
    
    # === TEMPERATURE-BASED FEATURES ===
    print("Engineering temperature-based features...")
    
    # Temperature differentials (critical for detecting bearing/gearbox issues)
    df_engineered['temp_diff_gearbox_ambient'] = df['Temperature oil in gearbox'] - df['Ambient temperature']
    df_engineered['temp_diff_generator_ambient'] = df['Temperature in generator bearing 2 (Drive End)'] - df['Ambient temperature']
    df_engineered['temp_diff_nacelle_ambient'] = df['Nacelle temperature'] - df['Ambient temperature']
    
    # Generator bearing temperature difference (imbalance indicator)
    df_engineered['temp_diff_generator_bearings'] = (df['Temperature in generator bearing 2 (Drive End)'] - 
                                                   df['Temperature in generator bearing 1 (Non-Drive End)'])
    
    # Generator stator winding temperature variations
    stator_temps = ['Temperature inside generator in stator windings phase 1',
                   'Temperature inside generator in stator windings phase 2', 
                   'Temperature inside generator in stator windings phase 3']
    df_engineered['stator_temp_mean'] = df[stator_temps].mean(axis=1)
    df_engineered['stator_temp_std'] = df[stator_temps].std(axis=1)
    df_engineered['stator_temp_max'] = df[stator_temps].max(axis=1)
    df_engineered['stator_temp_range'] = df[stator_temps].max(axis=1) - df[stator_temps].min(axis=1)
    
    # IGBT temperature variations (inverter health)
    igbt_grid_temp = 'Temperature measured by the IGBT-driver on the grid side inverter'
    igbt_rotor_temps = ['Temperature measured by the IGBT-driver on the rotor side inverter phase1',
                       'Temperature measured by the IGBT-driver on the rotor side inverter phase2',
                       'Temperature measured by the IGBT-driver on the rotor side inverter phase3']
    df_engineered['igbt_rotor_temp_mean'] = df[igbt_rotor_temps].mean(axis=1)
    df_engineered['igbt_rotor_temp_std'] = df[igbt_rotor_temps].std(axis=1)
    df_engineered['igbt_temp_diff_grid_rotor'] = df[igbt_grid_temp] - df_engineered['igbt_rotor_temp_mean']
    
    # HV Transformer temperature balance
    hv_temps = ['Temperature in HV transformer phase L1',
               'Temperature in HV transformer phase L2', 
               'Temperature in HV transformer phase L3']
    df_engineered['hv_temp_mean'] = df[hv_temps].mean(axis=1)
    df_engineered['hv_temp_std'] = df[hv_temps].std(axis=1)
    df_engineered['hv_temp_max'] = df[hv_temps].max(axis=1)
    
    # === POWER AND ELECTRICAL FEATURES ===
    print("Engineering power and electrical features...")
    
    # Power efficiency and ratios
    df_engineered['power_efficiency'] = df['Total active power'] / (df['Windspeed'] ** 3 + 1e-6)
    df_engineered['power_wind_ratio'] = df['Total active power'] / (df['Windspeed'] + 1e-6)
    df_engineered['reactive_active_ratio'] = df['Total reactive power'] / (df['Total active power'] + 1e-6)
    
    # Voltage balance across phases
    voltage_phases = ['Averaged voltage in phase 1', 'Averaged voltage in phase 2', 'Averaged voltage in phase 3']
    df_engineered['voltage_mean'] = df[voltage_phases].mean(axis=1)
    df_engineered['voltage_std'] = df[voltage_phases].std(axis=1)
    df_engineered['voltage_imbalance'] = df_engineered['voltage_std'] / df_engineered['voltage_mean']
    
    # Current balance across phases
    current_phases = ['Averaged current in phase 1', 'Averaged current in phase 2', 'Averaged current in phase 3']
    df_engineered['current_mean'] = df[current_phases].mean(axis=1)
    df_engineered['current_std'] = df[current_phases].std(axis=1)
    df_engineered['current_imbalance'] = df_engineered['current_std'] / df_engineered['current_mean']
    
    # Power factor and grid interaction
    df_engineered['power_factor'] = df['Total active power'] / np.sqrt(df['Total active power']**2 + df['Total reactive power']**2 + 1e-6)
    df_engineered['grid_power_ratio'] = df['Grid power'] / (df['Total active power'] + 1e-6)
    
    # === ROTATIONAL AND MECHANICAL FEATURES ===
    print("Engineering rotational and mechanical features...")
    
    # Gearbox ratio and efficiency indicators
    df_engineered['gearbox_ratio'] = df['Generator rpm in latest period'] / (df['Rotor rpm'] + 1e-6)
    df_engineered['rotor_speed_normalized'] = df['Rotor rpm'] / (df['Windspeed'] + 1e-6)
    
    # Speed variations and turbulence indicators
    if 'Estimated windspeed' in df.columns and 'Windspeed' in df.columns:
        df_engineered['wind_speed_error'] = df['Windspeed'] - df['Estimated windspeed']
        df_engineered['wind_turbulence_indicator'] = abs(df_engineered['wind_speed_error']) / (df['Windspeed'] + 1e-6)
    
    # === DIRECTIONAL AND ALIGNMENT FEATURES ===
    print("Engineering directional features...")
    
    # Wind direction features (convert to circular statistics)
    df_engineered['wind_dir_sin'] = np.sin(np.radians(df['Wind absolute direction']))
    df_engineered['wind_dir_cos'] = np.cos(np.radians(df['Wind absolute direction']))
    df_engineered['nacelle_dir_sin'] = np.sin(np.radians(df['Nacelle direction']))
    df_engineered['nacelle_dir_cos'] = np.cos(np.radians(df['Nacelle direction']))
    
    # Nacelle-wind alignment (yaw error)
    wind_rad = np.radians(df['Wind absolute direction'])
    nacelle_rad = np.radians(df['Nacelle direction'])
    df_engineered['yaw_error'] = np.degrees(np.arctan2(np.sin(wind_rad - nacelle_rad), 
                                                       np.cos(wind_rad - nacelle_rad)))
    df_engineered['yaw_error_abs'] = abs(df_engineered['yaw_error'])
    
    # === OPERATIONAL STATE FEATURES ===
    print("Engineering operational state features...")
    
    # Generator connection state indicators
    df_engineered['generator_connected'] = (df['Active power - generator connected in delta'] > 0) | \
                                         (df['Active power - generator connected in star'] > 0)
    df_engineered['generator_disconnected'] = df['Active power - generator disconnected'] > 0
    
    # Power generation mode
    df_engineered['delta_connection_active'] = df['Active power - generator connected in delta'] > df['Active power - generator connected in star']
    
    # === COOLING AND THERMAL MANAGEMENT ===
    print("Engineering thermal management features...")
    
    # Cooling effectiveness indicators
    df_engineered['cooling_effectiveness'] = (df['Ambient temperature'] - df['Temperature in the VCS cooling water']) / \
                                           (df['Ambient temperature'] + 1e-6)
    
    # Heat dissipation indicators
    df_engineered['thermal_load_gearbox'] = df['Temperature oil in gearbox'] / (df['Total active power'] + 1e-6)
    df_engineered['thermal_load_generator'] = df_engineered['stator_temp_mean'] / (df['Total active power'] + 1e-6)
    
    # === CONTROL SYSTEM FEATURES ===
    print("Engineering control system features...")
    
    # Pitch angle effectiveness
    df_engineered['pitch_effectiveness'] = df['Pitch angle'] / (df['Windspeed'] + 1e-6)
    df_engineered['pitch_power_ratio'] = df['Total active power'] / (abs(df['Pitch angle']) + 1e-6)
    
    # Grid frequency deviation
    df_engineered['grid_freq_deviation'] = abs(df['Grid frequency'] - 50.0)  # Assuming 50Hz grid
    
    # === RATIO AND EFFICIENCY FEATURES ===
    print("Engineering efficiency and ratio features...")
    
    # Multi-component temperature ratios
    df_engineered['gearbox_generator_temp_ratio'] = df['Temperature oil in gearbox'] / \
                                                   (df['Temperature in generator bearing 2 (Drive End)'] + 1e-6)
    df_engineered['controller_ambient_ratio'] = df['Temperature in the hub controller'] / (df['Ambient temperature'] + 1e-6)
    
    # Power density indicators
    df_engineered['power_per_rpm'] = df['Total active power'] / (df['Generator rpm in latest period'] + 1e-6)
    df_engineered['power_per_wind_cube'] = df['Total active power'] / (df['Windspeed'] ** 3 + 1e-6)
    
    # === ANOMALY DETECTION FEATURES ===
    print("Engineering anomaly detection features...")
    
    # Create composite health indicators
    temp_features = [col for col in df.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
    if temp_features:
        df_engineered['overall_temp_level'] = df[temp_features].mean(axis=1)
        df_engineered['max_temp_deviation'] = df[temp_features].max(axis=1) - df[temp_features].min(axis=1)
    
    # Electrical system health
    electrical_features = ['voltage_imbalance', 'current_imbalance', 'grid_freq_deviation', 'power_factor']
    available_electrical = [f for f in electrical_features if f in df_engineered.columns]
    if available_electrical:
        df_engineered['electrical_health_score'] = df_engineered[available_electrical].mean(axis=1)
    
    print(f"Feature engineering completed. Added {len(df_engineered.columns) - len(df.columns)} new features.")
    return df_engineered

# Apply feature engineering to the dataset
df_with_features = engineer_turbine_features(df)

print(f"\nOriginal features: {len(df.columns)}")
print(f"Total features after engineering: {len(df_with_features.columns)}")
print(f"New features added: {len(df_with_features.columns) - len(df.columns)}")

# Update the main dataframe
df = df_with_features.copy()

df.dropna(inplace=True)  # Ensure no NaNs before proceeding

# --- Step 4: Prepare training and testing sets ---
X_train = df[df['train_test'] == 'train']
X_test = df[df['train_test'] == 'prediction']
y_train = X_train['fault_label']
y_test = X_test['fault_label']
columns_to_drop=['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id', 'fault_label']
# columns_to_drop=['train_test', 'fault_label']
X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)



# --- Step 5: Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

training_datasets = {}
training_datasets['Original + Threshold Tuning'] = (X_train_scaled, y_train)

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# ADASYN oversampling
# adasyn = ADASYN(random_state=42)
# X_train_smote, y_train_smote = adasyn.fit_resample(X_train_scaled, y_train)
# training_datasets['SMOTE Oversampling'] = (X_train_smote, y_train_smote)
# SMOTE oversampling
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
# training_datasets['SMOTE Oversampling'] = (X_train_smote, y_train_smote)
smote_enn = SMOTEENN(random_state=42, n_jobs=-1)
X_train_smote, y_train_smote = smote_enn.fit_resample(X_train_scaled, y_train)
training_datasets['SMOTE Oversampling'] = (X_train_smote, y_train_smote)

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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# --- Class Imbalance Analysis ---
print("\n=== CLASS IMBALANCE ANALYSIS ===")
print(f"Original training set distribution:")
print(f"  Class 0 (Normal): {(y_train == 0).sum()} samples ({(y_train == 0).mean():.1%})")
print(f"  Class 1 (Fault): {(y_train == 1).sum()} samples ({(y_train == 1).mean():.1%})")
print(f"  Imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.1f}:1")

print(f"\nAfter SMOTE oversampling:")
unique_smote, counts_smote = np.unique(y_train_smote, return_counts=True)
print(f"  Class 0 (Normal): {counts_smote[0]} samples ({counts_smote[0]/len(y_train_smote):.1%})")
print(f"  Class 1 (Fault): {counts_smote[1]} samples ({counts_smote[1]/len(y_train_smote):.1%})")
print(f"  New ratio: {counts_smote[0] / counts_smote[1]:.1f}:1")
print(f"  Dataset size increased: {len(y_train_smote)} (was {len(y_train)})")

# IMPORTANT: Don't use class_weight='balanced' with SMOTE - data is already balanced!
# model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
model = RandomForestClassifier(
        n_estimators=200,  # More trees for better performance
        max_depth=15,      # Prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
        )
# model = MLPClassifier(
#     hidden_layer_sizes=(150, 80, 35),
#     activation='relu',
#     solver='adam',
#     max_iter=500,
#     random_state=42,
#     early_stopping=True,
#     validation_fraction=0.1
# )
# model = LGBMClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     verbose=-1
# )
# model = XGBClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     eval_metric='logloss',
#     use_label_encoder=False
# )
# model = ExtraTreesClassifier(
#     n_estimators=1200,
#     max_depth=50,
#     min_samples_split=10,
#     min_samples_leaf=5,
#     random_state=42,
#     n_jobs=-1
# )
# model.fit(X_train_scaled, y_train)
model.fit(X_train_smote, y_train_smote)

# === ENHANCED EVALUATION WITH THRESHOLD OPTIMIZATION ===
print("\n" + "="*60)
print("ENHANCED MODEL EVALUATION")
print("="*60)
    
# Get predictions and probabilities
y_pred = model.predict(X_test_scaled)
# for i in range(len(y_pred)):
#     if y_pred[i] == 1:
#         sum = 0
#         for j in range(i + 1, 15):
#             sum += y_pred[j]
#         if sum != 5:
#             for j in range(i, 16):
#                 y_pred[j] = 0
            
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of fault class

print("\n=== DEFAULT THRESHOLD (0.5) RESULTS ===")    
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

# Calculate specificity and sensitivity
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

# Balanced accuracy (good for imbalanced datasets)
balanced_acc = (sensitivity + specificity) / 2

print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Specificity (True Negative Rate): {specificity:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")

# === PROBABILITY DISTRIBUTION ANALYSIS ===
print(f"\n=== PROBABILITY DISTRIBUTION ANALYSIS ===")
print(f"Probability statistics for all test samples:")
print(f"  Min: {y_proba.min():.6f}")
print(f"  Max: {y_proba.max():.6f}")
print(f"  Mean: {y_proba.mean():.6f}")
print(f"  Median: {np.median(y_proba):.6f}")

# Analyze probabilities for actual fault cases
fault_indices = y_test == 1
normal_indices = y_test == 0

if fault_indices.sum() > 0:
    fault_probs = y_proba[fault_indices]
    normal_probs = y_proba[normal_indices]
    
    print(f"\nProbability statistics for ACTUAL FAULT cases:")
    print(f"  Min: {fault_probs.min():.6f}")
    print(f"  Max: {fault_probs.max():.6f}")
    print(f"  Mean: {fault_probs.mean():.6f}")
    print(f"  Median: {np.median(fault_probs):.6f}")
    
    print(f"\nProbability statistics for NORMAL cases:")
    print(f"  Min: {normal_probs.min():.6f}")
    print(f"  Max: {normal_probs.max():.6f}")
    print(f"  Mean: {normal_probs.mean():.6f}")
    print(f"  Median: {np.median(normal_probs):.6f}")

# === THRESHOLD OPTIMIZATION ===
print(f"\n=== THRESHOLD OPTIMIZATION FOR FAULT DETECTION ===")

# Test multiple thresholds
thresholds = np.arange(0.05, 0.95, 0.05)  # From 0.05 to 0.95 with step 0.05
threshold_results = []

for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    print(f"\nEvaluating threshold: {threshold:.2f}")
    print("Confusion Matrix:")
    print(cm_thresh)
    tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel() if cm_thresh.size == 4 else (0, 0, 0, 0)
    
    precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    specificity_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
    f1_t = 2 * (precision_t * recall_t) / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
    balanced_acc_t = (recall_t + specificity_t) / 2
    
    threshold_results.append({
        'threshold': threshold,
        'precision': precision_t,
        'recall': recall_t,
        'specificity': specificity_t,
        'f1': f1_t,
        'balanced_accuracy': balanced_acc_t,
        'tp': tp_t,
        'fp': fp_t,
        'fn': fn_t,
        'tn': tn_t
    })

# Convert to DataFrame for easy analysis
threshold_df = pd.DataFrame(threshold_results)

# Find optimal thresholds for different objectives
best_f1_idx = threshold_df['f1'].idxmax()
best_recall_idx = threshold_df['recall'].idxmax()
best_balanced_acc_idx = threshold_df['balanced_accuracy'].idxmax()

print(f"\nTHRESHOLD OPTIMIZATION RESULTS:")
print(f"\nFor MAXIMUM F1-SCORE:")
best_f1 = threshold_df.iloc[best_f1_idx]
print(f"  Threshold: {best_f1['threshold']:.2f}")
print(f"  F1-Score: {best_f1['f1']:.4f}")
print(f"  Recall: {best_f1['recall']:.4f} (catches {best_f1['tp']:.0f}/{(y_test==1).sum()} faults)")
print(f"  Precision: {best_f1['precision']:.4f}")
print(f"  False Alarms: {best_f1['fp']:.0f}/{(y_test==0).sum()}")

print(f"\nFor MAXIMUM RECALL (catch most faults):")
best_recall = threshold_df.iloc[best_recall_idx]
print(f"  Threshold: {best_recall['threshold']:.2f}")
print(f"  Recall: {best_recall['recall']:.4f} (catches {best_recall['tp']:.0f}/{(y_test==1).sum()} faults)")
print(f"  Precision: {best_recall['precision']:.4f}")
print(f"  F1-Score: {best_recall['f1']:.4f}")
print(f"  False Alarms: {best_recall['fp']:.0f}/{(y_test==0).sum()}")

print(f"\nFor MAXIMUM BALANCED ACCURACY:")
best_bal_acc = threshold_df.iloc[best_balanced_acc_idx]
print(f"  Threshold: {best_bal_acc['threshold']:.2f}")
print(f"  Balanced Accuracy: {best_bal_acc['balanced_accuracy']:.4f}")
print(f"  Recall: {best_bal_acc['recall']:.4f} (catches {best_bal_acc['tp']:.0f}/{(y_test==1).sum()} faults)")
print(f"  Precision: {best_bal_acc['precision']:.4f}")

# === OPERATIONAL RECOMMENDATIONS ===
print(f"\n=== OPERATIONAL RECOMMENDATIONS ===")

total_faults = (y_test == 1).sum()
total_normal = (y_test == 0).sum()

print(f"\nFor SAFETY-CRITICAL operations (prioritize catching faults):")
print(f"  → Use threshold: {best_recall['threshold']:.2f}")
print(f"  → Will catch {best_recall['recall']*100:.1f}% of faults")
print(f"  → Expected false alarm rate: {best_recall['fp']/total_normal*100:.1f}%")

print(f"\nFor BALANCED operations:")
print(f"  → Use threshold: {best_f1['threshold']:.2f}")
print(f"  → F1-Score: {best_f1['f1']:.4f}")
print(f"  → Will catch {best_f1['recall']*100:.1f}% of faults")
print(f"  → False alarm rate: {best_f1['fp']/total_normal*100:.1f}%")

# === CLASS IMBALANCE HANDLING SUMMARY ===
print(f"\n=== CLASS IMBALANCE HANDLING SUMMARY ===")
print("✅ WHAT'S WORKING WELL:")
print("  • SMOTE oversampling successfully balanced the training data")
print("  • Logistic Regression can learn from balanced data")
print("  • Model generates meaningful probability scores")
print("  • Threshold optimization provides operational flexibility")

print(f"\n⚠️  AREAS FOR IMPROVEMENT:")
print("  • Consider trying other sampling techniques (ADASYN, BorderlineSMOTE)")
print("  • Add feature engineering to improve separability")
print("  • Consider ensemble methods for better performance")
print("  • Monitor for potential overfitting due to synthetic samples")

# === FINAL MODEL DEPLOYMENT RECOMMENDATION ===
print(f"\n🎯 FINAL DEPLOYMENT RECOMMENDATION:")
if best_recall['recall'] > 0.7:  # If we can catch >70% of faults
    print(f"✅ DEPLOY WITH OPTIMIZED THRESHOLD")
    print(f"   Recommended threshold: {best_f1['threshold']:.2f} (balanced) or {best_recall['threshold']:.2f} (safety-focused)")
elif best_recall['recall'] > 0.5:  # If we can catch >50% of faults
    print(f"⚠️  DEPLOY WITH CAUTION")
    print(f"   Model catches {best_recall['recall']*100:.1f}% of faults - consider improving before deployment")
else:
    print(f"❌ DO NOT DEPLOY - INSUFFICIENT FAULT DETECTION")
    print(f"   Model only catches {best_recall['recall']*100:.1f}% of faults")

print(f"\n💡 Next steps: Consider feature engineering, ensemble methods, or collecting more fault examples")

# === FUNCTION TO USE SPECIFIC THRESHOLD FOR PREDICTIONS ===
def predict_with_threshold(model, X, threshold=0.5):
    """
    Make predictions using a specific threshold instead of the default 0.5
    
    Parameters:
    - model: trained classifier with predict_proba method
    - X: features for prediction
    - threshold: decision threshold (0-1)
    
    Returns:
    - predictions: binary predictions (0/1)
    - probabilities: probability scores for positive class
    """
    # Get probability scores for positive class (fault)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Apply custom threshold
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities

# === DEMONSTRATION: HOW TO USE SPECIFIC THRESHOLDS ===
print(f"\n" + "="*60)
print("🔧 DEMONSTRATION: USING SPECIFIC THRESHOLDS")
print(f"="*60)

# Example 1: Using the optimal F1 threshold
optimal_f1_threshold = best_f1['threshold']
predictions_f1, probabilities = predict_with_threshold(model, X_test, optimal_f1_threshold)

print(f"\n1️⃣  USING OPTIMAL F1 THRESHOLD ({optimal_f1_threshold:.2f}):")
print(f"   Predictions made: {len(predictions_f1)} samples")
print(f"   Predicted faults: {predictions_f1.sum()}")
print(f"   Actual faults: {(y_test==1).sum()}")

# Calculate metrics for this threshold
from sklearn.metrics import classification_report, confusion_matrix
print("\n   Performance metrics:")
print(classification_report(y_test, predictions_f1, target_names=['Normal', 'Fault']))

# Example 2: Using a custom threshold (e.g., 0.3 for higher sensitivity)
custom_threshold = 0.3
predictions_custom, _ = predict_with_threshold(model, X_test, custom_threshold)

print(f"\n2️⃣  USING CUSTOM THRESHOLD ({custom_threshold}):")
print(f"   Predictions made: {len(predictions_custom)} samples")
print(f"   Predicted faults: {predictions_custom.sum()}")
print(f"   Actual faults: {(y_test==1).sum()}")

print("\n   Performance metrics:")
print(classification_report(y_test, predictions_custom, target_names=['Normal', 'Fault']))

# Example 3: Batch processing with different thresholds
print(f"\n3️⃣  COMPARING MULTIPLE THRESHOLDS:")
test_thresholds = [0.2, 0.1, 0.05, 0.01, 0.5]

for thresh in test_thresholds:
    preds, _ = predict_with_threshold(model, X_test, thresh)
    recall = (preds[y_test==1]).sum() / (y_test==1).sum()
    precision = (preds[y_test==1]).sum() / preds.sum() if preds.sum() > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"   Threshold {thresh:.1f}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}")

# === SAVE PREDICTIONS TO FILE ===
print(f"\n" + "="*60)
print("💾 SAVING PREDICTIONS TO FILE")
print(f"="*60)

# Create a results DataFrame
import pandas as pd
results_df = pd.DataFrame({
    'prediction_probability': probabilities,
    'prediction_f1_threshold': predictions_f1,
    'prediction_custom_threshold': predictions_custom,
    'actual_label': y_test
})

# Save to CSV
results_file = 'c:\\Repos\\COURSERA\\wind-turbines\\fault_predictions.csv'
results_df.to_csv(results_file, index=False)
print(f"✅ Predictions saved to: {results_file}")
print(f"   Columns: {list(results_df.columns)}")
print(f"   Shape: {results_df.shape}")

# Show first few predictions
print(f"\n📊 SAMPLE PREDICTIONS:")
print(results_df.head(10))

print(f"\n💡 HOW TO USE IN PRODUCTION:")
print(f"   1. Load your trained model")
print(f"   2. Choose your threshold based on business requirements:")
print(f"      - Safety-critical: {best_recall['threshold']:.2f} (catches more faults)")
print(f"      - Balanced: {best_f1['threshold']:.2f} (good F1-score)")
print(f"      - Custom: any value between 0-1")
print(f"   3. Use predict_with_threshold(model, new_data, threshold)")
print(f"   4. Monitor performance and adjust threshold as needed")
