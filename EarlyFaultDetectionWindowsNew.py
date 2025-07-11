import os
from imblearn.combine import SMOTEENN
from numpy.typing import NDArray
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, f1_score

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


def downsample_every_nth_row(df, n=12):
    """Keep every nth row from the dataframe"""
    return df.iloc[::n].reset_index(drop=True)

def downsample_ndArray_every_nth_row(arr: NDArray, n=12):
    """Keep every nth row from the NDArray"""
    return arr[::n]

def create_sliding_windows(df, event_start_id=None, event_end_id=None,
                           window_size=12, step_size=6, prefault_horizon=18):
    df = df.sort_values("id").reset_index(drop=True)

    # === APPLY FEATURE RENAMING ===
    # feature_desc_path = '../wind-turbines-data/setA/Wind Farm A/comma_feature_description.csv'
    # df, column_mapping, unmapped_columns = rename_sensor_columns_with_descriptions(df, feature_desc_path, verbose=True)

    # df = engineer_turbine_features(df)
    
    # Separate feature columns and timestamp columns
    # feature_cols = [col for col in df.columns if col.startswith("sensor_") or col.startswith("wind_") or col.startswith("power_") or col.startswith("reactive_")] 
    feature_cols = [col for col in df.columns if col != 'time_stamp' and col != 'asset_id' and col != 'id' and col != 'train_test' and col != 'status_type_id'] 
    timestamp_cols = [col for col in df.columns if col in ["time_stamp", "timestamp", "datetime"]]
    
    df.dropna(inplace=True)
    # df = df[df['status_type_id'] != '2']  # Exclude fault status
    # df = df[df['status_type_id'] != '3']  # Exclude fault status
    # df = df[df['status_type_id'] != '4']  # Exclude fault status
    # df_prediction = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)

    # anomaly_ids = set(range(event_start_id - prefault_horizon, event_start_id)) if event_start_id else set()
    # anomaly_ids = set(range(event_start_id - prefault_horizon, event_start_id)) if event_start_id else set()
    # fault_ids = set(range(event_start_id, event_end_id)) if event_start_id else set()

    X_windows = []
    X_timestamps = []
    y_labels = []

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end]
        window_ids = set(window["id"])

        # if int(len(fault_ids.intersection(window_ids))) > 0:
        #    continue

        statusId = df.iloc[start]["status_type_id"]
        label = 0
        if statusId in [1, 3, 4, 5] or statusId in ['1', '3', '4', '5']:
            label = 1  # Anomaly detected
        
        # Extract features (exclude timestamps from scaling)
        X_features = window[feature_cols].values
        X_windows.append(X_features)
        
        # Extract timestamps separately
        if timestamp_cols:
            X_timestamp = window[timestamp_cols].values
            X_timestamps.append(X_timestamp)
        
        y_labels.append(label)

    return X_windows, y_labels, X_timestamps if timestamp_cols else []

def create_test_sliding_windows(df, event_start_id=None,
                           window_size=12, step_size=6, prefault_horizon=18):
    df = df.sort_values("id").reset_index(drop=True)
    
    # Separate feature columns and timestamp columns
    feature_cols = [col for col in df.columns if col.startswith("sensor_") or col.startswith("wind_") or col.startswith("power_") or col.startswith("reactive_")]
    timestamp_cols = [col for col in df.columns if col in ["time_stamp", "timestamp", "datetime"]]
    
    df.dropna(inplace=True)
    # df_prediction = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
    df_prediction = df.copy().reset_index(drop=True)

    # Check if event_start_id is smaller than the first data point
    anomaly_before_data = event_start_id is not None and event_start_id > df_prediction['id'].iloc[0] and event_start_id < df_prediction['id'].iloc[-1]
    label = 0
    if anomaly_before_data:
        label = 1

    X_windows = []
    X_timestamps = []
    y_labels = []

    for start in range(0, len(df_prediction) - window_size + 1, step_size):
        end = start + window_size
        window = df_prediction.iloc[start:end]

        # Extract features (exclude timestamps from scaling)
        X_features = window[feature_cols].values
        X_windows.append(X_features)
        
        # Extract timestamps separately
        if timestamp_cols:
            X_timestamp = window[timestamp_cols].values
            X_timestamps.append(X_timestamp)
        
        y_labels.append(label)

    return X_windows, y_labels, X_timestamps if timestamp_cols else []

def load_event_info(event_info_path):
    return pd.read_csv(event_info_path).set_index("event_id")

def process_all_events(data_dir, window_size=12, step_size=6, prefault_horizon=18, data_type="_train"):
    all_X = []
    all_y = []
    all_timestamps = []
    
    # for farm in os.listdir(data_dir):
    farm_path = data_dir # os.path.join(data_dir, farm)
    event_info_path = os.path.join(farm_path, "comma_event_info.csv")
    dataset_path = os.path.join(farm_path, "datasets-actual")

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

        # if data_type == "_test":
        #    df = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
        # X_windows, y_labels = create_sliding_windows(df[df['train_test'] == 'train'], event_start_id, event_end_id,
        if data_type == "_train":
            df = df[df["train_test"] == "train"].copy().reset_index(drop=True)
        else:
            df = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)

        X_windows, y_labels, X_timestamps = create_sliding_windows(df, event_start_id, event_end_id,
                                                        window_size, step_size, prefault_horizon)
        all_X.extend(X_windows)
        all_y.extend(y_labels)
        all_timestamps.extend(X_timestamps)
        # X_test_windows, y_test_labels = create_test_sliding_windows(df[df['train_test'] == 'prediction'], event_start_id,
        #                                                 window_size, step_size, prefault_horizon)
        # all_X.extend(X_windows)
        # all_y.extend(y_labels)
        # all_X_test.extend(X_test_windows)
        # all_y_test.extend(y_test_labels)

    return np.array(all_X), np.array(all_y), np.array(all_timestamps)

def flatten_windows(X):
    # reshape (num_samples, time, features) -> (num_samples, time * features)
    # Note: This function now only processes feature data, timestamps are handled separately
    return X.reshape(X.shape[0], -1)

def prepare_ml_dataset(data_dir, test_size=0.2, random_state=42,
                       window_size=12, step_size=6, prefault_horizon=18):
    print("Loading and processing SCADA data...")
    X, y, _ = process_all_events(data_dir, window_size, step_size, prefault_horizon, "_train")
    X_test, y_test, timestamps = process_all_events(data_dir, window_size, step_size, prefault_horizon, "_test")

    # X = downsample_ndArray_every_nth_row(X, 4).copy()
    # y = downsample_ndArray_every_nth_row(y, 4).copy()
    # X_test = downsample_ndArray_every_nth_row(X_test, 4).copy()
    # y_test = downsample_ndArray_every_nth_row(y_test, 4).copy()

    print(f"Total samples: {len(X)}, Anomalies: {np.sum(y)}")

    X_flat = flatten_windows(X)
    
    X_test_flat = flatten_windows(X_test)
    timestamps_flat = flatten_windows(timestamps)
    
    # Replace zeros in any column with the column mean (excluding zeros)
    # print("Replacing zeros with column averages...")
    # X_flat_no_zeros = X_flat.copy()
    # for col in range(X_flat_no_zeros.shape[1]):
    #     col_vals = X_flat_no_zeros[:, col]
    #     nonzero_vals = col_vals[col_vals != 0]
    #     if nonzero_vals.size > 0:
    #         col_mean = nonzero_vals.mean()
    #         zeros_count = np.sum(col_vals == 0)
    #         if zeros_count > 0:
    #             col_vals[col_vals == 0] = col_mean
    #             X_flat_no_zeros[:, col] = col_vals
    #             print(f"  Column {col}: Replaced {zeros_count} zeros with mean {col_mean:.3f}")
    
    # # Apply same zero replacement to test data
    # X_test_flat_no_zeros = X_test_flat.copy()
    # for col in range(X_test_flat_no_zeros.shape[1]):
    #     col_vals = X_test_flat_no_zeros[:, col]
    #     # Use the same mean from training data for consistency
    #     train_col_vals = X_flat_no_zeros[:, col]
    #     nonzero_vals = train_col_vals[train_col_vals != 0]
    #     if nonzero_vals.size > 0:
    #         col_mean = nonzero_vals.mean()
    #         zeros_count = np.sum(col_vals == 0)
    #         if zeros_count > 0:
    #             col_vals[col_vals == 0] = col_mean
    #             X_test_flat_no_zeros[:, col] = col_vals
    #             print(f"  Test Column {col}: Replaced {zeros_count} zeros with training mean {col_mean:.3f}")
    
    # # Use the cleaned data for scaling
    # X_flat = X_flat_no_zeros
    # X_test_flat = X_test_flat_no_zeros

    print("Normalizing...")
    print("ℹ️  Note: Timestamp columns (time_stamp, timestamp, datetime) are excluded from scaling")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    # X_test_scaled = scaler.fit_transform(X_test_flat)

    # print("Splitting...")
    # X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = train_test_split(
    #    X_scaled, y, timestamps_flat, test_size=test_size, random_state=random_state, stratify=y
    #)

    # Use the first timestamp from each test sample as the test timestamp
    test_timestamps = timestamps_flat[:, 0] if timestamps_flat.ndim > 1 else timestamps_flat

    # return X_scaled, X_test_scaled, y, y_test, scaler
    # return X_train, X_test, y_train, y_test, scaler, test_timestamps
    return X_scaled, X_test_scaled, y, y_test, scaler, test_timestamps

# === DAILY ALARM ANALYSIS FUNCTION ===
def analyze_daily_alarms(y_true, y_pred, test_timestamps, show_details=True):
    """
    Analyze alarms by day and categorize them into:
    - False Alarms (FP): Model predicted fault but no actual fault
    - Uncaught Alarms (FN): Actual fault but model didn't predict it
    - Correctly Predicted Alarms (TP): Model correctly predicted actual fault
    - Correct No-Alarms (TN): Model correctly predicted no fault when there was none
    
    Parameters:
    - y_true: True labels (0/1)
    - y_pred: Predicted labels (0/1)
    - test_timestamps: Array of timestamps for each prediction
    - show_details: Whether to show detailed daily breakdown
    
    Returns:
    - Dictionary with daily analysis results
    """
    
    print(f"\n" + "="*80)
    print(f"DAILY ALARM ANALYSIS")
    print(f"="*80)
    
    # Convert timestamps to pandas datetime if they aren't already
    timestamps = pd.to_datetime(test_timestamps)
    
    # Create DataFrame for analysis
    alarms_df = pd.DataFrame({
        'timestamp': timestamps,
        'date': timestamps.date,  # Remove .dt since timestamps is already DatetimeIndex
        'y_true': y_true,
        'y_pred': y_pred,
        'false_alarm': ((y_true == 0) & (y_pred == 1)).astype(int),  # FP
        'uncaught_alarm': ((y_true == 1) & (y_pred == 0)).astype(int),  # FN
        'correct_alarm': ((y_true == 1) & (y_pred == 1)).astype(int),  # TP
        'correct_no_alarm': ((y_true == 0) & (y_pred == 0)).astype(int)  # TN
    })
    
    # Group by date and aggregate
    daily_alarms = alarms_df.groupby('date').agg({
        'false_alarm': 'sum',
        'uncaught_alarm': 'sum', 
        'correct_alarm': 'sum',
        'correct_no_alarm': 'sum',
        'y_true': 'sum',  # Total actual faults per day
        'y_pred': 'sum',  # Total predicted faults per day
        'timestamp': ['count', 'min', 'max']  # Total predictions, first and last time
    }).round(0)
    
    # Flatten column names
    daily_alarms.columns = [
        'false_alarms', 'uncaught_alarms', 'correct_alarms', 'correct_no_alarms',
        'actual_faults', 'predicted_faults', 'total_predictions', 'first_time', 'last_time'
    ]
    daily_alarms = daily_alarms.reset_index()
    
    # Calculate summary statistics
    total_days = len(daily_alarms)
    days_with_false_alarms = len(daily_alarms[daily_alarms['false_alarms'] > 0])
    days_with_uncaught_alarms = len(daily_alarms[daily_alarms['uncaught_alarms'] > 0])
    days_with_correct_alarms = len(daily_alarms[daily_alarms['correct_alarms'] > 0])
    
    total_false_alarms = daily_alarms['false_alarms'].sum()
    total_uncaught_alarms = daily_alarms['uncaught_alarms'].sum()
    total_correct_alarms = daily_alarms['correct_alarms'].sum()
    
    print(f"\n📊 DAILY ALARM SUMMARY:")
    print(f"  Analysis period: {daily_alarms['date'].min()} to {daily_alarms['date'].max()}")
    print(f"  Total days analyzed: {total_days}")
    print(f"  ")
    print(f"  📈 ALARM STATISTICS:")
    print(f"    Total False Alarms (FP): {total_false_alarms}")
    print(f"    Total Uncaught Alarms (FN): {total_uncaught_alarms}")
    print(f"    Total Correct Alarms (TP): {total_correct_alarms}")
    print(f"  ")
    print(f"  📅 DAYS WITH ALARMS:")
    print(f"    Days with False Alarms: {days_with_false_alarms} ({days_with_false_alarms/total_days*100:.1f}%)")
    print(f"    Days with Uncaught Alarms: {days_with_uncaught_alarms} ({days_with_uncaught_alarms/total_days*100:.1f}%)")
    print(f"    Days with Correct Alarms: {days_with_correct_alarms} ({days_with_correct_alarms/total_days*100:.1f}%)")
    
    if show_details:
        print(f"\n📅 DETAILED DAILY ALARM BREAKDOWN:")
        print(f"{'Date':<12} {'False':<7} {'Uncaught':<9} {'Correct':<8} {'Total Pred':<10} {'Total Act':<10} {'First':<8} {'Last':<8}")
        print("-" * 85)
        
        for _, row in daily_alarms.iterrows():
            first_time = row['first_time'].strftime('%H:%M') if pd.notna(row['first_time']) else 'N/A'
            last_time = row['last_time'].strftime('%H:%M') if pd.notna(row['last_time']) else 'N/A'
            
            # Color coding for problematic days
            status = ""
            if row['uncaught_alarms'] > 0:
                status = "🚨"  # Critical: missed actual faults
            elif row['false_alarms'] > 3:
                status = "⚠️ "  # Warning: many false alarms
            elif row['correct_alarms'] > 0:
                status = "✅"  # Good: caught actual faults
            
            print(f"{row['date']!s:<12} {row['false_alarms']:<7.0f} {row['uncaught_alarms']:<9.0f} "
                  f"{row['correct_alarms']:<8.0f} {row['predicted_faults']:<10.0f} {row['actual_faults']:<10.0f} "
                  f"{first_time:<8} {last_time:<8} {status}")
    
    # Identify problematic days
    print(f"\n🚨 CRITICAL DAYS (with uncaught alarms):")
    critical_days = daily_alarms[daily_alarms['uncaught_alarms'] > 0]
    if len(critical_days) > 0:
        for _, row in critical_days.iterrows():
            print(f"  {row['date']}: {row['uncaught_alarms']:.0f} uncaught alarms "
                  f"(missed {row['uncaught_alarms']:.0f} out of {row['actual_faults']:.0f} actual faults)")
    else:
        print(f"  ✅ No critical days - all actual faults were detected!")
    
    print(f"\n⚠️  HIGH FALSE ALARM DAYS (>3 false alarms):")
    high_fa_days = daily_alarms[daily_alarms['false_alarms'] > 3]
    if len(high_fa_days) > 0:
        for _, row in high_fa_days.iterrows():
            print(f"  {row['date']}: {row['false_alarms']:.0f} false alarms")
    else:
        print(f"  ✅ No days with excessive false alarms!")
    
    print(f"\n✅ SUCCESSFUL DETECTION DAYS (with correct alarms):")
    success_days = daily_alarms[daily_alarms['correct_alarms'] > 0]
    if len(success_days) > 0:
        for _, row in success_days.iterrows():
            detection_rate = (row['correct_alarms'] / row['actual_faults'] * 100) if row['actual_faults'] > 0 else 0
            print(f"  {row['date']}: {row['correct_alarms']:.0f} correct alarms "
                  f"({detection_rate:.1f}% detection rate)")
    else:
        print(f"  ⚠️  No days with successful fault detection!")

    allarm_days = daily_alarms[(daily_alarms['correct_alarms'] > 5) | (daily_alarms['false_alarms'] > 5)]
    isFirst = True
    for _, row in allarm_days.iterrows():
        if isFirst:
            print(f"\n🔔 FIRST DAY WITH HIGH ALARM COUNT: {row['date']}")
            isFirst = False
        elif (row['date'] - last_time).days > 1:
            print(f"\n🔔 FIRST DAY WITH HIGH ALARM COUNT: {row['date']}")
        last_time = row['date']
    
    # Weekly analysis
    alarms_df['week'] = timestamps.isocalendar().week
    alarms_df['year'] = timestamps.year
    
    weekly_alarms = alarms_df.groupby(['year', 'week']).agg({
        'false_alarm': 'sum',
        'uncaught_alarm': 'sum',
        'correct_alarm': 'sum'
    }).reset_index()
    
    if len(weekly_alarms) > 1:
        print(f"\n📈 WEEKLY ALARM SUMMARY:")
        print(f"{'Week':<12} {'False':<7} {'Uncaught':<9} {'Correct':<8}")
        print("-" * 40)
        for _, row in weekly_alarms.iterrows():
            week_label = f"{row['year']}-W{row['week']:02.0f}"
            print(f"{week_label:<12} {row['false_alarm']:<7.0f} {row['uncaught_alarm']:<9.0f} {row['correct_alarm']:<8.0f}")
    
    # Calculate daily averages and trends
    avg_false_alarms = daily_alarms['false_alarms'].mean()
    avg_uncaught_alarms = daily_alarms['uncaught_alarms'].mean()
    avg_correct_alarms = daily_alarms['correct_alarms'].mean()
    
    print(f"\n📊 DAILY AVERAGES:")
    print(f"  Average False Alarms per day: {avg_false_alarms:.2f}")
    print(f"  Average Uncaught Alarms per day: {avg_uncaught_alarms:.2f}")
    print(f"  Average Correct Alarms per day: {avg_correct_alarms:.2f}")
    
    # Performance metrics
    total_actual_faults = daily_alarms['actual_faults'].sum()
    total_predicted_faults = daily_alarms['predicted_faults'].sum()
    
    if total_actual_faults > 0:
        overall_detection_rate = (total_correct_alarms / total_actual_faults) * 100
        overall_miss_rate = (total_uncaught_alarms / total_actual_faults) * 100
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"  Detection Rate: {overall_detection_rate:.1f}% ({total_correct_alarms}/{total_actual_faults})")
        print(f"  Miss Rate: {overall_miss_rate:.1f}% ({total_uncaught_alarms}/{total_actual_faults})")
    
    if total_predicted_faults > 0:
        precision = (total_correct_alarms / total_predicted_faults) * 100
        print(f"  Precision: {precision:.1f}% ({total_correct_alarms}/{total_predicted_faults})")
    
    # Return structured results
    results = {
        'daily_alarms': daily_alarms,
        'summary': {
            'total_days': total_days,
            'days_with_false_alarms': days_with_false_alarms,
            'days_with_uncaught_alarms': days_with_uncaught_alarms,
            'days_with_correct_alarms': days_with_correct_alarms,
            'total_false_alarms': total_false_alarms,
            'total_uncaught_alarms': total_uncaught_alarms,
            'total_correct_alarms': total_correct_alarms,
            'avg_false_alarms_per_day': avg_false_alarms,
            'avg_uncaught_alarms_per_day': avg_uncaught_alarms,
            'avg_correct_alarms_per_day': avg_correct_alarms
        },
        'critical_days': critical_days,
        'high_false_alarm_days': high_fa_days,
        'successful_days': success_days,
        'weekly_alarms': weekly_alarms
    }
    
    return results

DATA_DIR = '../wind-turbines-data/setA/Wind Farm B'  # root dir containing WindFarmA/, WindFarmB/, etc.

X_train, X_test, y_train, y_test, scaler, test_timestamps = prepare_ml_dataset(
    data_dir=DATA_DIR,
    test_size=0.2,
    window_size=1,
    step_size=1,
    prefault_horizon=1000
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# smote_enn = SMOTEENN(random_state=42, n_jobs=-1)
# X_train_smote, y_train_smote = smote_enn.fit_resample(X_train, y_train)
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
model = IsolationForest(
    n_estimators=2000,        # Number of trees in the forest
    max_samples='auto',      # Use all samples for each tree
    contamination=0.1,       # Expected proportion of anomalies (adjust based on your data)
    max_features=1.0,        # Use all features
    bootstrap=False,         # Don't bootstrap samples
    n_jobs=-1,              # Use all available cores
    random_state=42,        # For reproducible results
    verbose=0               # Silent mode
)
# model = XGBClassifier(
#     n_estimators=200,
#     max_depth=5,
#     learning_rate=0.025,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     eval_metric='logloss',
#     use_label_encoder=False
# )

# model = MLPClassifier(
#     hidden_layer_sizes=(100, 50, 25),
#     activation='relu',
#     solver='adam',
#     max_iter=500,
#     random_state=42,
#     early_stopping=True,
#     validation_fraction=0.1
# )

# model.fit(X_train_smote, y_train_smote)
# For IsolationForest, we fit on normal data only (unsupervised learning)
# Use only normal samples for training (label 0)
normal_samples_mask = y_train == 0
X_train_normal = X_train[normal_samples_mask]
print(f"Training IsolationForest on {len(X_train_normal)} normal samples out of {len(X_train)} total samples")

model.fit(X_train_normal)

test_timestamps = pd.to_datetime(test_timestamps)

# IsolationForest returns -1 for anomalies and 1 for normal points
# We need to convert this to 0/1 format (0=normal, 1=anomaly)
y_pred_raw = model.predict(X_test)
y_pred = np.where(y_pred_raw == -1, 1, 0)  # Convert -1 (anomaly) to 1, and 1 (normal) to 0

# For probability scores, use decision_function (lower scores = more anomalous)
y_decision_scores = model.decision_function(X_test)
# Convert decision scores to probabilities (0-1 range)
# More negative scores = higher anomaly probability
y_proba = 1 / (1 + np.exp(y_decision_scores))  # Sigmoid transformation

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# Calculate accuracy manually for IsolationForest
model_accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {model_accuracy:.4f}")

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

# === DAILY ALARM ANALYSIS ===
print(f"\n" + "="*80)
print(f"ANALYZING DAILY ALARM PATTERNS")
print(f"="*80)

# Run daily alarm analysis
daily_alarm_results = analyze_daily_alarms(y_test, y_pred, test_timestamps, show_details=True)

print(f"\n📋 DAILY ALARM ANALYSIS COMPLETE")
print(f"="*80)

# === END OF DAILY ALARM ANALYSIS ===
