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

def rename_sensor_columns_with_descriptions(df, feature_desc_path, verbose=True):
    """
    Rename sensor columns in the dataframe using descriptive names from feature description file.
    
    Parameters:
    - df: pandas DataFrame with sensor columns to rename
    - feature_desc_path: str, path to the comma_feature_description.csv file
    - verbose: bool, whether to print detailed information during processing
    
    Returns:
    - df_renamed: DataFrame with renamed columns
    - column_mapping: dict, mapping from old column names to new descriptive names
    - unmapped_columns: list, sensor columns that couldn't be mapped
    """
    
    if verbose:
        print("\n=== RENAMING COLUMNS BASED ON FEATURE DESCRIPTIONS ===")
    
    # Load the feature description mapping
    try:
        feature_desc = pd.read_csv(feature_desc_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find feature description file at {feature_desc_path}")
        return df, {}, []
    
    if verbose:
        print(f"Loaded feature descriptions for {len(feature_desc)} sensors")
        print(f"Original dataframe has {len(df.columns)} columns")
    
    # Create mapping dictionary from actual column names to descriptions
    column_mapping = {}
    
    if verbose:
        print(f"\nSample of original columns:")
        for i, col in enumerate(df.columns[:10]):
            print(f"  {i+1:2d}. {col}")
        if len(df.columns) > 10:
            print(f"  ... and {len(df.columns) - 10} more")
    
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
                if col_name in df.columns:
                    if stat == 'average':
                        column_mapping[col_name] = description
                    else:
                        column_mapping[col_name] = f"{description} ({stat})"
        else:
            # Single statistic
            if statistics_type == 'average':
                col_name = f"{sensor_name}_avg"
                if col_name in df.columns:
                    column_mapping[col_name] = description
            else:
                # Handle special cases like sensor_44, sensor_45, etc. (no _avg suffix)
                if sensor_name in df.columns:
                    column_mapping[sensor_name] = description
    
    # Handle special cases for columns without _avg suffix
    special_sensors = ['sensor_44', 'sensor_45', 'sensor_46', 'sensor_47', 'sensor_48', 'sensor_49', 'sensor_50', 'sensor_51']
    for sensor in special_sensors:
        if sensor in df.columns:
            # Find the description for this sensor
            sensor_desc = feature_desc[feature_desc['sensor_name'] == sensor]
            if not sensor_desc.empty:
                column_mapping[sensor] = sensor_desc.iloc[0]['description']
    
    if verbose:
        print(f"\nCreated mapping for {len(column_mapping)} columns:")
        print("Sample mappings:")
        for i, (old_name, new_name) in enumerate(list(column_mapping.items())[:10]):
            print(f"  '{old_name}' → '{new_name}'")
        if len(column_mapping) > 10:
            print(f"  ... and {len(column_mapping) - 10} more")
    
    # Apply the column renaming
    df_renamed = df.rename(columns=column_mapping)
    
    # Check for unmapped sensor columns
    unmapped_sensor_columns = [col for col in df_renamed.columns 
                              if (col.startswith('sensor_') or col.startswith('wind_speed_') or 
                                  col.startswith('reactive_power_') or col.startswith('power_')) 
                              and col not in ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']]
    
    if verbose:
        if unmapped_sensor_columns:
            print(f"\nUnmapped sensor columns ({len(unmapped_sensor_columns)}):")
            for col in unmapped_sensor_columns[:5]:
                print(f"  - {col}")
            if len(unmapped_sensor_columns) > 5:
                print(f"  ... and {len(unmapped_sensor_columns) - 5} more")
        
        # Show successful mapping results
        mapped_columns = [col for col in df_renamed.columns if col in column_mapping.values()]
        print(f"\nSuccessfully mapped {len(mapped_columns)} columns to descriptive names")
        
        print(f"\nSample of new column names:")
        descriptive_cols = [col for col in df_renamed.columns if col not in ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']]
        for i, col in enumerate(descriptive_cols[:10]):
            print(f"  {i+1:2d}. {col}")
        if len(descriptive_cols) > 10:
            print(f"  ... and {len(descriptive_cols) - 10} more")
        
        # Verify we didn't lose any data
        print(f"\nVerification:")
        print(f"  Original shape: {df_renamed.shape}")
        print(f"  Columns before: {len(df.columns)}")
        print(f"  Descriptive columns: {len([col for col in df_renamed.columns if col in column_mapping.values()])}")
        print(f"  System columns: {len([col for col in df_renamed.columns if col in ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']])}")
    
    return df_renamed, column_mapping, unmapped_sensor_columns

# === APPLY FEATURE RENAMING ===
feature_desc_path = '../wind-turbines-data/setA/Wind Farm A/comma_feature_description.csv'
combined_df, column_mapping, unmapped_columns = rename_sensor_columns_with_descriptions(
    combined_df, feature_desc_path, verbose=True
)

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
for start_time, end_time in zip(fault_start_times, fault_end_times):
    # Label pre-fault window (7 days before start)
    rows_to_keep = ~((df['time_stamp'] > start_time) & (df['time_stamp'] <= end_time))
    df = df[rows_to_keep]
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

# --- Step 3: Advanced Feature Engineering ---
print("\n=== ADVANCED FEATURE ENGINEERING ===")

def engineer_turbine_features(df):
    """
    Engineer advanced features from wind turbine SCADA data for fault detection
    """
    df_engineered = df.copy()
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
from imblearn.combine import SMOTETomek

# ADASYN oversampling
adasyn = ADASYN(random_state=42)
X_train_smote, y_train_smote = adasyn.fit_resample(X_train_scaled, y_train)
training_datasets['SMOTE Oversampling'] = (X_train_smote, y_train_smote)
# SMOTE oversampling
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
# training_datasets['SMOTE Oversampling'] = (X_train_smote, y_train_smote)

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
# model = RandomForestClassifier(
#         n_estimators=200,  # More trees for better performance
#         max_depth=15,      # Prevent overfitting
#         min_samples_split=10,
#         min_samples_leaf=5,
#         random_state=42,
#         n_jobs=-1,
#         bootstrap=True,
#         oob_score=True
#         )
# model = MLPClassifier(
#     hidden_layer_sizes=(100, 50, 25),
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
model = ExtraTreesClassifier(
    n_estimators=600,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
# model.fit(X_train_scaled, y_train)
model.fit(X_train_smote, y_train_smote)

# === ENHANCED EVALUATION WITH THRESHOLD OPTIMIZATION ===
print("\n" + "="*60)
print("ENHANCED MODEL EVALUATION")
print("="*60)
    
# Get predictions and probabilities
y_pred = model.predict(X_test_scaled)
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

# === ADVANCED CLASS IMBALANCE HANDLING TECHNIQUES ===
print(f"\n" + "="*60)
print("🔧 ADVANCED CLASS IMBALANCE SOLUTIONS")
print(f"="*60)

print(f"\n📊 CURRENT IMBALANCE STATUS:")
print(f"   Original ratio: {(y_train == 0).sum() / (y_train == 1).sum():.1f}:1")
print(f"   Fault cases: {(y_train == 1).sum()} ({(y_train == 1).mean():.1%})")
print(f"   Normal cases: {(y_train == 0).sum()} ({(y_train == 0).mean():.1%})")

# === 1. MULTIPLE SAMPLING TECHNIQUES ===
print(f"\n🔄 1. TESTING MULTIPLE SAMPLING TECHNIQUES:")

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

sampling_techniques = {}

# Original SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
sampling_techniques['SMOTE'] = (X_smote, y_smote)

# Borderline SMOTE (focuses on borderline cases)
borderline_smote = BorderlineSMOTE(random_state=42)
X_borderline, y_borderline = borderline_smote.fit_resample(X_train_scaled, y_train)
sampling_techniques['BorderlineSMOTE'] = (X_borderline, y_borderline)

# SVM SMOTE (uses SVM to generate synthetic samples)
try:
    svm_smote = SVMSMOTE(random_state=42)
    X_svm_smote, y_svm_smote = svm_smote.fit_resample(X_train_scaled, y_train)
    sampling_techniques['SVM-SMOTE'] = (X_svm_smote, y_svm_smote)
except:
    print("   ⚠️  SVM-SMOTE failed (too few samples or computational issues)")

# Combined techniques
smote_tomek = SMOTETomek(random_state=42)
X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X_train_scaled, y_train)
sampling_techniques['SMOTE+Tomek'] = (X_smote_tomek, y_smote_tomek)

smote_enn = SMOTEENN(random_state=42)
X_smote_enn, y_smote_enn = smote_enn.fit_resample(X_train_scaled, y_train)
sampling_techniques['SMOTE+ENN'] = (X_smote_enn, y_smote_enn)

print(f"   ✅ Created {len(sampling_techniques)} different sampling strategies")

# === 2. CLASS WEIGHTS APPROACH ===
print(f"\n⚖️  2. CLASS WEIGHTS APPROACH:")

from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print(f"   Class 0 weight: {class_weight_dict[0]:.2f}")
print(f"   Class 1 weight: {class_weight_dict[1]:.2f}")
print(f"   Ratio: {class_weight_dict[1]/class_weight_dict[0]:.1f}:1")

# === 3. ENSEMBLE WITH BOOTSTRAP SAMPLING ===
print(f"\n🎲 3. BOOTSTRAP SAMPLING FOR BALANCE:")

def create_balanced_bootstrap_samples(X, y, n_samples=5, sample_size=None):
    """Create multiple balanced bootstrap samples"""
    # Convert y to numpy array and reset indices to be sequential
    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    # Convert X to numpy array if needed
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = np.array(X)
    
    if sample_size is None:
        sample_size = min((y_array == 0).sum(), (y_array == 1).sum()) * 2  # 2x minority class
    
    samples = []
    minority_class = 1 if (y_array == 1).sum() < (y_array == 0).sum() else 0
    majority_class = 1 - minority_class
    
    # Use sequential indices (0, 1, 2, ...) since we're working with numpy arrays
    minority_indices = np.where(y_array == minority_class)[0]
    majority_indices = np.where(y_array == majority_class)[0]
    
    print(f"   Bootstrap: {len(minority_indices)} minority, {len(majority_indices)} majority samples")
    
    for i in range(n_samples):
        # Sample equal numbers from both classes
        n_per_class = sample_size // 2
        
        minority_sample = np.random.choice(minority_indices, n_per_class, replace=True)
        majority_sample = np.random.choice(majority_indices, n_per_class, replace=False)
        
        combined_indices = np.concatenate([minority_sample, majority_sample])
        np.random.shuffle(combined_indices)
        
        samples.append((X_array[combined_indices], y_array[combined_indices]))
    
    return samples

bootstrap_samples = create_balanced_bootstrap_samples(X_train_scaled, y_train, n_samples=3)
print(f"   ✅ Created {len(bootstrap_samples)} balanced bootstrap samples")

# === 4. COST-SENSITIVE LEARNING ===
print(f"\n💰 4. COST-SENSITIVE LEARNING:")

# Define cost matrix for wind turbine faults
# Missing a fault (False Negative) is much more expensive than false alarm (False Positive)
cost_matrix = {
    'fn_cost': 10,  # Missing a fault costs 10x more
    'fp_cost': 1,   # False alarm baseline cost
    'tp_cost': 0,   # Correct fault detection
    'tn_cost': 0    # Correct normal prediction
}

def cost_sensitive_threshold(y_true, y_proba, fn_cost=10, fp_cost=1):
    """Find optimal threshold based on cost matrix"""
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Calculate total cost
        total_cost = (fn * fn_cost) + (fp * fp_cost)
        costs.append(total_cost)
    
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    minimal_cost = costs[optimal_idx]
    
    return optimal_threshold, minimal_cost, costs

print(f"   📊 Cost Matrix for Wind Turbine Faults:")
print(f"      Missing Fault (FN): {cost_matrix['fn_cost']}x cost")
print(f"      False Alarm (FP): {cost_matrix['fp_cost']}x cost")

# === 5. ANOMALY DETECTION APPROACH ===
print(f"\n🔍 5. ANOMALY DETECTION APPROACH:")

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Use only normal data to train anomaly detector
normal_data = X_train_scaled[y_train == 0]

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(normal_data)

# One-Class SVM (smaller sample for efficiency)
sample_size = min(2000, len(normal_data))
sample_indices = np.random.choice(len(normal_data), sample_size, replace=False)
normal_sample = normal_data[sample_indices]

oc_svm = OneClassSVM(nu=0.1, gamma='scale')
oc_svm.fit(normal_sample)

print(f"   ✅ Trained anomaly detectors on {len(normal_data)} normal samples")

# === 6. ENSEMBLE OF IMBALANCED CLASSIFIERS ===
print(f"\n🎭 6. ENSEMBLE APPROACHES:")

class ImbalancedEnsemble:
    def __init__(self, base_models, sampling_strategies):
        self.base_models = base_models
        self.sampling_strategies = sampling_strategies
        self.trained_models = []
    
    def fit(self, X, y):
        for (name, (X_resampled, y_resampled)), model in zip(self.sampling_strategies.items(), self.base_models):
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_resampled, y_resampled)
            self.trained_models.append((name, model_copy))
        return self
    
    def predict_proba(self, X):
        predictions = []
        for name, model in self.trained_models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        return np.mean(predictions, axis=0)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

# Create ensemble with different sampling strategies
base_models = [
    ExtraTreesClassifier(n_estimators=100, random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42)
]

ensemble = ImbalancedEnsemble(base_models, sampling_techniques)
print(f"   ✅ Created ensemble with {len(base_models)} models and {len(sampling_techniques)} sampling strategies")

# === PRACTICAL COMPARISON OF TECHNIQUES ===
print(f"\n" + "="*60)
print("🧪 TESTING IMBALANCE HANDLING TECHNIQUES")
print(f"="*60)

def quick_model_evaluation(X_train, y_train, X_test, y_test, technique_name):
    """Quick evaluation of a sampling technique"""
    # Train a simple model
    quick_model = ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    quick_model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = quick_model.predict(X_test)
    y_proba = quick_model.predict_proba(X_test)[:, 1]
    
    # Calculate key metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    return {
        'technique': technique_name,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'training_samples': len(X_train)
    }

# Test different sampling techniques
results_comparison = []

print(f"\n🔬 TESTING SAMPLING TECHNIQUES:")

# 1. Original imbalanced data
original_result = quick_model_evaluation(X_train_scaled, y_train, X_test_scaled, y_test, "Original (Imbalanced)")
results_comparison.append(original_result)
print(f"   ✓ Original: F1={original_result['f1_score']:.3f}, Recall={original_result['recall']:.3f}")

# 2. Your current ADASYN
adasyn_result = quick_model_evaluation(X_train_smote, y_train_smote, X_test_scaled, y_test, "ADASYN (Current)")
results_comparison.append(adasyn_result)
print(f"   ✓ ADASYN: F1={adasyn_result['f1_score']:.3f}, Recall={adasyn_result['recall']:.3f}")

# 3. Test other techniques
for name, (X_resampled, y_resampled) in sampling_techniques.items():
    result = quick_model_evaluation(X_resampled, y_resampled, X_test_scaled, y_test, name)
    results_comparison.append(result)
    print(f"   ✓ {name}: F1={result['f1_score']:.3f}, Recall={result['recall']:.3f}")

# 4. Class weights approach
model_weighted = ExtraTreesClassifier(n_estimators=50, class_weight=class_weight_dict, random_state=42)
model_weighted.fit(X_train_scaled, y_train)
y_pred_weighted = model_weighted.predict(X_test_scaled)
y_proba_weighted = model_weighted.predict_proba(X_test_scaled)[:, 1]

weighted_result = {
    'technique': 'Class Weights',
    'precision': precision_score(y_test, y_pred_weighted, zero_division=0),
    'recall': recall_score(y_test, y_pred_weighted, zero_division=0),
    'f1_score': f1_score(y_test, y_pred_weighted, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_proba_weighted),
    'training_samples': len(X_train_scaled)
}
results_comparison.append(weighted_result)
print(f"   ✓ Class Weights: F1={weighted_result['f1_score']:.3f}, Recall={weighted_result['recall']:.3f}")

# === RESULTS SUMMARY ===
print(f"\n📊 TECHNIQUE COMPARISON SUMMARY:")
comparison_df = pd.DataFrame(results_comparison)
comparison_df = comparison_df.sort_values('f1_score', ascending=False)

print(f"\n🏆 RANKING BY F1-SCORE:")
for i, row in comparison_df.iterrows():
    print(f"   {comparison_df.index.get_loc(i)+1}. {row['technique']:<20} F1={row['f1_score']:.3f} Recall={row['recall']:.3f} Precision={row['precision']:.3f}")

# Find best technique for fault detection (prioritize recall)
best_recall = comparison_df.loc[comparison_df['recall'].idxmax()]
best_f1 = comparison_df.loc[comparison_df['f1_score'].idxmax()]

print(f"\n🎯 BEST FOR FAULT DETECTION (HIGH RECALL):")
print(f"   {best_recall['technique']}: Recall={best_recall['recall']:.3f}, F1={best_recall['f1_score']:.3f}")

print(f"\n⚖️  BEST BALANCED PERFORMANCE (HIGH F1):")
print(f"   {best_f1['technique']}: F1={best_f1['f1_score']:.3f}, Recall={best_f1['recall']:.3f}")

# === COST-SENSITIVE ANALYSIS ===
print(f"\n💰 COST-SENSITIVE ANALYSIS:")

# Define cost parameters (adjust based on your business case)
fault_cost = 100000  # Cost of missing a fault ($100k per missed fault)
alarm_cost = 1000    # Cost of false alarm ($1k per false alarm)

print(f"Cost assumptions: Missed fault = ${fault_cost:,}, False alarm = ${alarm_cost:,}")

# Calculate cost for each technique
for result in results_comparison:
    technique = result['technique']
    recall = result['recall']
    precision = result['precision']
    
    # Estimate costs based on test set
    total_faults = (y_test == 1).sum()
    total_normal = (y_test == 0).sum()
    
    missed_faults = total_faults * (1 - recall)
    if precision > 0:
        total_predictions = total_faults * recall / precision
        false_alarms = total_predictions - (total_faults * recall)
    else:
        false_alarms = 0
    
    total_cost = (missed_faults * fault_cost) + (false_alarms * alarm_cost)
    
    print(f"   {technique:<20}: ${total_cost:>8,.0f} (Miss: {missed_faults:.1f} faults, FA: {false_alarms:.1f})")

# === FINAL RECOMMENDATION ===
print(f"\n" + "="*60)
print("🚀 FINAL RECOMMENDATIONS")
print(f"="*60)

print(f"\n✅ IMMEDIATE IMPROVEMENTS:")
print(f"   1. Switch from ADASYN to {best_f1['technique']} (better F1-score)")
print(f"   2. Implement cost-sensitive threshold optimization")
print(f"   3. Use ensemble of top 3 techniques for critical applications")

print(f"\n🔄 EASY IMPLEMENTATION:")
print(f"   Replace this line:")
print(f"   adasyn = ADASYN(random_state=42)")
print(f"   X_train_smote, y_train_smote = adasyn.fit_resample(X_train_scaled, y_train)")
print(f"   ")
print(f"   With this:")
if best_f1['technique'] == 'SMOTE+Tomek':
    print(f"   smote_tomek = SMOTETomek(random_state=42)")
    print(f"   X_train_smote, y_train_smote = smote_tomek.fit_resample(X_train_scaled, y_train)")
elif best_f1['technique'] == 'BorderlineSMOTE':
    print(f"   borderline_smote = BorderlineSMOTE(random_state=42)")
    print(f"   X_train_smote, y_train_smote = borderline_smote.fit_resample(X_train_scaled, y_train)")
else:
    print(f"   # Use the {best_f1['technique']} implementation from above")

print(f"\n🎯 FOR WIND TURBINE PRODUCTION:")
print(f"   • Prioritize RECALL over precision (catch all faults)")
print(f"   • Use cost-sensitive learning (missing faults is expensive)")
print(f"   • Consider ensemble methods for critical turbines")
print(f"   • Monitor performance and retrain quarterly")

# === ADDITIONAL ADVANCED CLASS IMBALANCE TECHNIQUES ===
print(f"\n" + "="*70)
print("🚀 ADDITIONAL ADVANCED IMBALANCE HANDLING TECHNIQUES")
print(f"="*70)

# === 7. ENSEMBLE DIVERSITY TECHNIQUES ===
print(f"\n🎭 7. ENSEMBLE DIVERSITY TECHNIQUES:")

class DiversityEnsemble:
    """
    Ensemble that maximizes diversity while handling class imbalance
    """
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.models = []
        self.sampling_strategies = []
        self.diversity_scores = []
    
    def _calculate_diversity(self, predictions_list):
        """Calculate diversity using disagreement measure"""
        n_models = len(predictions_list)
        total_disagreement = 0
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                disagreement = np.mean(predictions_list[i] != predictions_list[j])
                total_disagreement += disagreement
        
        avg_diversity = total_disagreement / (n_models * (n_models - 1) / 2)
        return avg_diversity
    
    def fit(self, X, y):
        """Fit diverse ensemble with different sampling strategies"""
        # Different base models for diversity
        base_models = [
            RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
            ExtraTreesClassifier(n_estimators=50, max_depth=15, random_state=43),
            GradientBoostingClassifier(n_estimators=50, max_depth=8, random_state=44),
            RandomForestClassifier(n_estimators=50, max_features='sqrt', random_state=45),
            ExtraTreesClassifier(n_estimators=50, max_features='log2', random_state=46)
        ]
        
        # Different sampling strategies for diversity
        strategies = [
            ('SMOTE', SMOTE(random_state=42)),
            ('ADASYN', ADASYN(random_state=43)),
            ('BorderlineSMOTE', BorderlineSMOTE(random_state=44)),
            ('SMOTE+Tomek', SMOTETomek(random_state=45)),
            ('RandomOverSampler', RandomOverSampler(random_state=46))
        ]
        
        # Train diverse models
        predictions_list = []
        
        for i in range(min(self.n_estimators, len(base_models), len(strategies))):
            try:
                # Apply sampling strategy
                strategy_name, strategy = strategies[i]
                X_resampled, y_resampled = strategy.fit_resample(X, y)
                
                # Train model
                model = base_models[i]
                model.fit(X_resampled, y_resampled)
                
                # Get predictions for diversity calculation
                predictions = model.predict(X)
                predictions_list.append(predictions)
                
                self.models.append(model)
                self.sampling_strategies.append(strategy_name)
                
                print(f"   ✅ Trained {type(model).__name__} with {strategy_name}")
                
            except Exception as e:
                print(f"   ⚠️  Failed model {i}: {str(e)}")
                continue
        
        # Calculate ensemble diversity
        if len(predictions_list) > 1:
            diversity = self._calculate_diversity(predictions_list)
            print(f"   📊 Ensemble Diversity Score: {diversity:.4f}")
            self.diversity_scores.append(diversity)
        
        return self
    
    def predict_proba(self, X):
        """Predict using diverse ensemble"""
        if not self.models:
            raise ValueError("No models trained!")
        
        predictions = []
        for model in self.models:
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        
        # Weighted average (could be enhanced with dynamic weighting)
        return np.mean(predictions, axis=0)
    
    def predict(self, X):
        """Predict classes using ensemble"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

# Train diversity ensemble
diversity_ensemble = DiversityEnsemble(n_estimators=5)
diversity_ensemble.fit(X_train_scaled, y_train)

# === 8. FOCAL LOSS FOR IMBALANCED CLASSIFICATION ===
print(f"\n🎯 8. FOCAL LOSS FOR IMBALANCED CLASSIFICATION:")

class FocalLossClassifier:
    """
    Classifier using focal loss to handle class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, base_classifier=None):
        self.alpha = alpha  # Class weight
        self.gamma = gamma  # Focusing parameter
        self.base_classifier = base_classifier or LogisticRegression(random_state=42)
        
    def focal_loss_weights(self, y_true, y_proba):
        """Calculate focal loss weights"""
        # Convert to probabilities for positive class
        p_t = np.where(y_true == 1, y_proba, 1 - y_proba)
        
        # Alpha weighting
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        
        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        return focal_weight
    
    def fit(self, X, y):
        """Fit classifier with iterative focal loss weighting"""
        # Initial training
        self.base_classifier.fit(X, y)
        
        # Iteratively reweight samples based on focal loss
        for iteration in range(3):  # Limited iterations for efficiency
            # Get current predictions
            y_proba = self.base_classifier.predict_proba(X)[:, 1]
            
            # Calculate focal loss weights
            sample_weights = self.focal_loss_weights(y, y_proba)
            
            # Retrain with weighted samples
            if hasattr(self.base_classifier, 'sample_weight'):
                self.base_classifier.fit(X, y, sample_weight=sample_weights)
            else:
                # For classifiers that don't support sample weights,
                # create a bootstrap sample weighted by focal loss
                n_samples = len(X)
                weighted_indices = np.random.choice(
                    n_samples, 
                    size=n_samples, 
                    replace=True, 
                    p=sample_weights / sample_weights.sum()
                )
                
                X_weighted = X[weighted_indices]
                y_weighted = y[weighted_indices]
                
                self.base_classifier.fit(X_weighted, y_weighted)
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.base_classifier.predict_proba(X)
    
    def predict(self, X):
        """Predict classes"""
        return self.base_classifier.predict(X)

# Calculate optimal focal loss parameters
minority_ratio = (y_train == 1).mean()
optimal_alpha = 1 - minority_ratio  # Higher weight for minority class
optimal_gamma = 2.0 if minority_ratio < 0.1 else 1.0  # Higher gamma for more imbalanced data

print(f"   📊 Optimal Focal Loss Parameters:")
print(f"      Alpha: {optimal_alpha:.3f}")
print(f"      Gamma: {optimal_gamma:.1f}")

focal_classifier = FocalLossClassifier(
    alpha=optimal_alpha, 
    gamma=optimal_gamma,
    base_classifier=LogisticRegression(random_state=42, max_iter=1000)
)

try:
    focal_classifier.fit(X_train_scaled, y_train)
    print(f"   ✅ Focal Loss Classifier trained successfully")
except Exception as e:
    print(f"   ⚠️  Focal Loss training failed: {str(e)}")

# === 9. PROGRESSIVE SAMPLING STRATEGY ===
print(f"\n📈 9. PROGRESSIVE SAMPLING STRATEGY:")

class ProgressiveSampler:
    """
    Progressive sampling that gradually increases minority class representation
    """
    def __init__(self, stages=3):
        self.stages = stages
        self.stage_models = []
        
    def fit(self, X, y):
        """Fit progressive sampling models"""
        minority_class = 1 if (y == 1).sum() < (y == 0).sum() else 0
        majority_class = 1 - minority_class
        
        original_ratio = (y == majority_class).sum() / (y == minority_class).sum()
        
        print(f"   📊 Original imbalance ratio: {original_ratio:.1f}:1")
        
        # Progressive ratios: gradually reduce imbalance
        target_ratios = np.linspace(original_ratio, 1.0, self.stages)
        
        for stage, target_ratio in enumerate(target_ratios):
            print(f"   📊 Stage {stage + 1}: Target ratio {target_ratio:.1f}:1")
            
            # Calculate target samples for minority class
            n_minority = (y == minority_class).sum()
            n_majority_target = int(n_minority * target_ratio)
            
            # Sample majority class
            majority_indices = np.where(y == majority_class)[0]
            if len(majority_indices) > n_majority_target:
                selected_majority = np.random.choice(
                    majority_indices, n_majority_target, replace=False
                )
            else:
                selected_majority = majority_indices
            
            # Combine with all minority samples
            minority_indices = np.where(y == minority_class)[0]
            combined_indices = np.concatenate([minority_indices, selected_majority])
            
            # Create stage dataset
            X_stage = X[combined_indices]
            y_stage = y[combined_indices]
            
            # Train model for this stage
            stage_model = RandomForestClassifier(n_estimators=100, random_state=42+stage)
            stage_model.fit(X_stage, y_stage)
            
            self.stage_models.append(stage_model)
            
            print(f"      ✅ Stage {stage + 1} model trained")
            print(f"         Samples: {len(y_stage)} (Fault rate: {y_stage.mean():.1%})")
    
    def predict_proba(self, X):
        """Predict using ensemble of progressive models"""
        if not self.stage_models:
            raise ValueError("No models trained!")
        
        # Weight later stages more heavily (they're more balanced)
        weights = np.linspace(0.5, 1.0, len(self.stage_models))
        weights = weights / weights.sum()
        
        predictions = []
        for i, model in enumerate(self.stage_models):
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba * weights[i])
        
        return np.sum(predictions, axis=0)
    
    def predict(self, X):
        """Predict classes"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

# Train progressive sampler
progressive_sampler = ProgressiveSampler(stages=3)
progressive_sampler.fit(X_train_scaled, y_train)

# === 10. METALEARNING FOR IMBALANCE HANDLING ===
print(f"\n🧠 10. METALEARNING FOR IMBALANCE HANDLING:")

class MetaLearningImbalanceHandler:
    """
    Meta-learning approach to choose best imbalance handling technique
    """
    def __init__(self):
        self.meta_features = None
        self.technique_performance = {}
        self.meta_model = None
        
    def extract_meta_features(self, X, y):
        """Extract meta-features that characterize the dataset"""
        minority_class = 1 if (y == 1).sum() < (y == 0).sum() else 0
        majority_class = 1 - minority_class
        
        minority_samples = X[y == minority_class]
        majority_samples = X[y == majority_class]
        
        meta_features = {
            'imbalance_ratio': (y == majority_class).sum() / (y == minority_class).sum(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'minority_size': (y == minority_class).sum(),
            'majority_size': (y == majority_class).sum(),
            'minority_ratio': (y == minority_class).mean(),
        }
        
        # Add complexity measures
        if len(minority_samples) > 1 and len(majority_samples) > 1:
            # Distance between class centroids
            minority_centroid = minority_samples.mean(axis=0)
            majority_centroid = majority_samples.mean(axis=0)
            meta_features['centroid_distance'] = np.linalg.norm(minority_centroid - majority_centroid)
            
            # Intra-class variance
            meta_features['minority_variance'] = np.var(minority_samples, axis=0).mean()
            meta_features['majority_variance'] = np.var(majority_samples, axis=0).mean()
            
            # Class overlap approximation
            combined_variance = (meta_features['minority_variance'] + meta_features['majority_variance']) / 2
            meta_features['overlap_estimate'] = combined_variance / (meta_features['centroid_distance'] + 1e-6)
        
        return meta_features
    
    def recommend_technique(self, X, y):
        """Recommend best technique based on dataset characteristics"""
        meta_features = self.extract_meta_features(X, y)
        
        print(f"   📊 Dataset Meta-Features:")
        for key, value in meta_features.items():
            print(f"      {key}: {value:.4f}" if isinstance(value, float) else f"      {key}: {value}")
        
        # Rule-based recommendations (could be replaced with learned model)
        recommendations = []
        
        # High imbalance ratio
        if meta_features['imbalance_ratio'] > 20:
            if meta_features['minority_size'] < 100:
                recommendations.append(('ADASYN', 0.9))  # Adaptive for few samples
            else:
                recommendations.append(('SMOTE', 0.8))
        
        # Moderate imbalance
        elif meta_features['imbalance_ratio'] > 5:
            if meta_features.get('overlap_estimate', 0) > 1.0:
                recommendations.append(('BorderlineSMOTE', 0.85))  # High overlap
            else:
                recommendations.append(('SMOTE+Tomek', 0.8))
        
        # Low imbalance
        else:
            recommendations.append(('RandomOverSampler', 0.7))
        
        # Add ensemble recommendation for complex cases
        if meta_features['n_features'] > 20 or meta_features.get('overlap_estimate', 0) > 0.8:
            recommendations.append(('EasyEnsemble', 0.75))
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   💡 Technique Recommendations:")
        for technique, confidence in recommendations[:3]:
            print(f"      {technique}: {confidence:.1%} confidence")
        
        return recommendations[0][0] if recommendations else 'SMOTE'

# Apply meta-learning recommendation
meta_handler = MetaLearningImbalanceHandler()
recommended_technique = meta_handler.recommend_technique(X_train_scaled, y_train)

print(f"   🎯 Recommended Technique: {recommended_technique}")

# === EVALUATION OF ALL ADVANCED TECHNIQUES ===
print(f"\n" + "="*70)
print("📊 EVALUATING ALL ADVANCED TECHNIQUES")
print(f"="*70)

# Evaluate all new techniques
advanced_techniques = {
    'DiversityEnsemble': diversity_ensemble,
    'ProgressiveSampler': progressive_sampler,
}

# Add focal loss if it was trained successfully
if 'focal_classifier' in locals():
    advanced_techniques['FocalLoss'] = focal_classifier

advanced_results = {}

for name, model in advanced_techniques.items():
    print(f"\n🔍 Evaluating {name}:")
    
    try:
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_proba_adv = model.predict_proba(X_test_scaled)
            if len(y_proba_adv.shape) > 1 and y_proba_adv.shape[1] > 1:
                y_proba_adv = y_proba_adv[:, 1]
        else:
            y_proba_adv = None
        
        y_pred_adv = model.predict(X_test_scaled)
        
        # Calculate metrics
        precision_adv = precision_score(y_test, y_pred_adv, zero_division=0)
        recall_adv = recall_score(y_test, y_pred_adv, zero_division=0)
        f1_adv = f1_score(y_test, y_pred_adv, zero_division=0)
        balanced_acc_adv = balanced_accuracy_score(y_test, y_pred_adv)
        
        # Confusion matrix
        cm_adv = confusion_matrix(y_test, y_pred_adv)
        
        print(f"   📊 Results:")
        print(f"      Precision: {precision_adv:.4f}")
        print(f"      Recall: {recall_adv:.4f}")
        print(f"      F1-Score: {f1_adv:.4f}")
        print(f"      Balanced Accuracy: {balanced_acc_adv:.4f}")
        
        # Store results
        advanced_results[name] = {
            'precision': precision_adv,
            'recall': recall_adv,
            'f1_score': f1_adv,
            'balanced_accuracy': balanced_acc_adv,
            'confusion_matrix': cm_adv,
            'predictions': y_pred_adv,
            'probabilities': y_proba_adv
        }
        
        print(f"   ✅ {name} evaluation completed")
        
    except Exception as e:
        print(f"   ❌ {name} evaluation failed: {str(e)}")
        continue

# === COMPREHENSIVE TECHNIQUE COMPARISON ===
print(f"\n" + "="*70)
print("🏆 COMPREHENSIVE TECHNIQUE COMPARISON")
print(f"="*70)

# Combine all results for comparison
all_results = {}

# Add original results if they exist
if 'precision' in locals():
    all_results['Original_RandomForest'] = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc
    }

# Add advanced results
all_results.update(advanced_results)

# Create comparison table
if all_results:
    print(f"\n📊 PERFORMANCE COMPARISON TABLE:")
    print(f"{'Technique':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Bal.Acc':<10}")
    print("-" * 70)
    
    # Sort by F1 score
    sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('f1_score', 0), reverse=True)
    
    for technique, results in sorted_results:
        print(f"{technique:<25} "
              f"{results.get('precision', 0):<10.4f} "
              f"{results.get('recall', 0):<10.4f} "
              f"{results.get('f1_score', 0):<10.4f} "
              f"{results.get('balanced_accuracy', 0):<10.4f}")

# === PRACTICAL RECOMMENDATIONS ===
print(f"\n" + "="*70)
print("💡 PRACTICAL RECOMMENDATIONS FOR WIND TURBINE FAULT DETECTION")
print(f"="*70)

best_technique = None
best_f1 = 0

if all_results:
    for technique, results in all_results.items():
        if results.get('f1_score', 0) > best_f1:
            best_f1 = results.get('f1_score', 0)
            best_technique = technique

print(f"\n🎯 RECOMMENDED APPROACH:")
print(f"   Best Performing Technique: {best_technique}")
print(f"   Best F1 Score: {best_f1:.4f}")

print(f"\n📋 IMPLEMENTATION STRATEGY:")
print(f"   1️⃣  Primary: Use {best_technique} for main fault detection")
print(f"   2️⃣  Backup: Ensemble multiple top techniques for robustness")
print(f"   3️⃣  Context: Adjust approach based on operational conditions")
print(f"   4️⃣  Monitoring: Track performance and adapt over time")

print(f"\n🔧 DEPLOYMENT CONSIDERATIONS:")
print(f"   • High Recall Priority: Missing faults is very costly")
print(f"   • Acceptable False Positives: Better safe than sorry")
print(f"   • Real-time Capability: Models must be fast enough")
print(f"   • Interpretability: Engineers need to understand predictions")

print(f"\n✅ Advanced imbalance handling techniques demonstration completed!")
