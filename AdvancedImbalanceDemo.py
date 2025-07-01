"""
Advanced Class Imbalance Handling - Wind Turbine Fault Detection Demo
====================================================================

This script demonstrates the implementation of advanced class imbalance handling
techniques specifically for wind turbine fault detection using SCADA data.

Author: GitHub Copilot
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our advanced imbalance handling module
from AdvancedImbalanceHandling import (
    AdaptiveThresholdClassifier,
    DynamicSamplingStrategy,
    MultiObjectiveThresholdOptimizer,
    HybridSamplingEnsemble,
    CostSensitiveLearning,
    comprehensive_imbalance_comparison
)

# Standard ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                           precision_recall_curve, roc_curve, roc_auc_score)
from sklearn.linear_model import LogisticRegression

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced

# Install advanced libraries if needed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

def load_turbine_data():
    """
    Load and prepare wind turbine SCADA data for advanced imbalance handling demonstration.
    This function simulates loading the actual data from your project.
    """
    print("📊 Loading Wind Turbine SCADA Data...")
    
    # In your actual implementation, replace this with your data loading logic
    data_path = Path("c:/Repos/COURSERA/wind-turbines-data/setA/Wind Farm A/datasets")
    
    # Check if data exists
    if data_path.exists():
        print(f"   ✅ Data directory found: {data_path}")
        # You would load your actual data here
        # For demo purposes, we'll create synthetic data
        return create_synthetic_turbine_data()
    else:
        print("   ⚠️  Data directory not found. Creating synthetic data for demonstration.")
        return create_synthetic_turbine_data()

def create_synthetic_turbine_data():
    """Create synthetic wind turbine data that mimics real SCADA characteristics"""
    print("🔧 Creating synthetic wind turbine SCADA data...")
    
    np.random.seed(42)
    n_samples = 10000
    
    # Generate realistic turbine features
    data = {
        'WindSpeed': np.random.gamma(2, 2, n_samples),  # Wind speed (m/s)
        'TotalActivePower': np.random.gamma(2, 500, n_samples),  # Power output (kW)
        'GeneratorRPM': np.random.normal(1500, 200, n_samples),  # Generator RPM
        'GearboxTemp': np.random.normal(45, 10, n_samples),  # Gearbox temperature (°C)
        'GeneratorTemp': np.random.normal(55, 15, n_samples),  # Generator temperature (°C)
        'BearingTemp': np.random.normal(40, 8, n_samples),  # Bearing temperature (°C)
        'HubTemp': np.random.normal(25, 12, n_samples),  # Hub temperature (°C)
        'AmbientTemp': np.random.normal(15, 20, n_samples),  # Ambient temperature (°C)
        'VoltagePhase1': np.random.normal(690, 20, n_samples),  # Voltage Phase 1 (V)
        'VoltagePhase2': np.random.normal(690, 20, n_samples),  # Voltage Phase 2 (V)
        'VoltagePhase3': np.random.normal(690, 20, n_samples),  # Voltage Phase 3 (V)
        'CurrentPhase1': np.random.normal(500, 100, n_samples),  # Current Phase 1 (A)
        'CurrentPhase2': np.random.normal(500, 100, n_samples),  # Current Phase 2 (A)
        'CurrentPhase3': np.random.normal(500, 100, n_samples),  # Current Phase 3 (A)
        'GridFrequency': np.random.normal(50, 0.5, n_samples),  # Grid frequency (Hz)
        'NacellePosition': np.random.uniform(0, 360, n_samples),  # Nacelle position (degrees)
        'PitchAngle': np.random.uniform(0, 30, n_samples),  # Pitch angle (degrees)
    }
    
    df = pd.DataFrame(data)
    
    # Create fault conditions (realistic fault patterns)
    fault_probability = 0.05  # 5% fault rate
    
    # Create more sophisticated fault patterns
    fault_conditions = (
        (df['GearboxTemp'] > 70) |  # Overheating
        (df['GeneratorTemp'] > 85) |  # Generator overheating
        (df['BearingTemp'] > 65) |  # Bearing issues
        (np.abs(df['VoltagePhase1'] - df['VoltagePhase2']) > 50) |  # Voltage imbalance
        (np.abs(df['VoltagePhase2'] - df['VoltagePhase3']) > 50) |
        (np.abs(df['VoltagePhase1'] - df['VoltagePhase3']) > 50) |
        (df['GridFrequency'] < 49.5) |  # Grid issues
        (df['GridFrequency'] > 50.5) |
        (df['TotalActivePower'] < 100) & (df['WindSpeed'] > 5)  # Power issues
    )
    
    # Add some random faults
    random_faults = np.random.random(n_samples) < fault_probability
    
    df['Fault'] = (fault_conditions | random_faults).astype(int)
    
    # Add some noise and correlation adjustments
    df.loc[df['Fault'] == 1, 'GearboxTemp'] += np.random.normal(0, 5, (df['Fault'] == 1).sum())
    df.loc[df['Fault'] == 1, 'GeneratorTemp'] += np.random.normal(0, 8, (df['Fault'] == 1).sum())
    
    print(f"   ✅ Created {n_samples} samples")
    print(f"   📊 Fault distribution: {df['Fault'].value_counts().to_dict()}")
    print(f"   ⚖️  Imbalance ratio: {(df['Fault'] == 0).sum()}:{(df['Fault'] == 1).sum()}")
    
    return df

def engineer_advanced_features(df):
    """Engineer advanced features for better fault detection"""
    print("🔧 Engineering advanced features...")
    
    df_eng = df.copy()
    
    # Temperature ratios and differences
    df_eng['GearboxGenTempRatio'] = df['GearboxTemp'] / (df['GeneratorTemp'] + 1e-6)
    df_eng['TempDiffGearboxAmbient'] = df['GearboxTemp'] - df['AmbientTemp']
    df_eng['TempDiffGeneratorAmbient'] = df['GeneratorTemp'] - df['AmbientTemp']
    df_eng['OverallTempLevel'] = (df['GearboxTemp'] + df['GeneratorTemp'] + df['BearingTemp']) / 3
    
    # Electrical features
    df_eng['VoltageImbalance'] = (
        np.abs(df['VoltagePhase1'] - df['VoltagePhase2']) +
        np.abs(df['VoltagePhase2'] - df['VoltagePhase3']) +
        np.abs(df['VoltagePhase1'] - df['VoltagePhase3'])
    ) / 3
    
    df_eng['CurrentImbalance'] = (
        np.abs(df['CurrentPhase1'] - df['CurrentPhase2']) +
        np.abs(df['CurrentPhase2'] - df['CurrentPhase3']) +
        np.abs(df['CurrentPhase1'] - df['CurrentPhase3'])
    ) / 3
    
    # Power efficiency indicators
    df_eng['PowerPerRPM'] = df['TotalActivePower'] / (df['GeneratorRPM'] + 1e-6)
    df_eng['PowerPerWindCube'] = df['TotalActivePower'] / (df['WindSpeed'] ** 3 + 1e-6)
    df_eng['PowerEfficiency'] = df['TotalActivePower'] / (df['WindSpeed'] * 100 + 1e-6)
    
    # Grid connection health
    df_eng['GridFreqDeviation'] = np.abs(df['GridFrequency'] - 50.0)
    
    # Mechanical indicators
    df_eng['RPMPerWindSpeed'] = df['GeneratorRPM'] / (df['WindSpeed'] + 1e-6)
    
    print(f"   ✅ Added {len(df_eng.columns) - len(df.columns)} new features")
    
    return df_eng

def demonstrate_adaptive_threshold_classifier(X_train, y_train, X_test, y_test):
    """Demonstrate the Adaptive Threshold Classifier"""
    print("\n" + "="*60)
    print("🎯 ADAPTIVE THRESHOLD CLASSIFIER DEMONSTRATION")
    print("="*60)
    
    # Standard classifier for comparison
    print("\n1️⃣  Training Standard Random Forest...")
    standard_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    standard_rf.fit(X_train, y_train)
    
    y_pred_standard = standard_rf.predict(X_test)
    y_proba_standard = standard_rf.predict_proba(X_test)[:, 1]
    
    # Adaptive threshold classifier
    print("2️⃣  Training Adaptive Threshold Classifier...")
    adaptive_clf = AdaptiveThresholdClassifier(
        base_classifier=RandomForestClassifier(n_estimators=100, random_state=42),
        adaptation_method='density',
        n_neighbors=5
    )
    adaptive_clf.fit(X_train, y_train)
    
    y_pred_adaptive = adaptive_clf.predict(X_test)
    y_proba_adaptive = adaptive_clf.predict_proba(X_test)[:, 1]
    
    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    
    from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
    
    # Standard classifier metrics
    f1_standard = f1_score(y_test, y_pred_standard)
    precision_standard = precision_score(y_test, y_pred_standard)
    recall_standard = recall_score(y_test, y_pred_standard)
    balanced_acc_standard = balanced_accuracy_score(y_test, y_pred_standard)
    
    # Adaptive classifier metrics
    f1_adaptive = f1_score(y_test, y_pred_adaptive)
    precision_adaptive = precision_score(y_test, y_pred_adaptive)
    recall_adaptive = recall_score(y_test, y_pred_adaptive)
    balanced_acc_adaptive = balanced_accuracy_score(y_test, y_pred_adaptive)
    
    print(f"\nStandard Random Forest:")
    print(f"   F1 Score: {f1_standard:.4f}")
    print(f"   Precision: {precision_standard:.4f}")
    print(f"   Recall: {recall_standard:.4f}")
    print(f"   Balanced Accuracy: {balanced_acc_standard:.4f}")
    
    print(f"\nAdaptive Threshold Classifier:")
    print(f"   F1 Score: {f1_adaptive:.4f}")
    print(f"   Precision: {precision_adaptive:.4f}")
    print(f"   Recall: {recall_adaptive:.4f}")
    print(f"   Balanced Accuracy: {balanced_acc_adaptive:.4f}")
    
    # Improvement analysis
    print(f"\n📈 IMPROVEMENTS:")
    print(f"   F1 Score: {((f1_adaptive - f1_standard) / f1_standard * 100):+.2f}%")
    print(f"   Precision: {((precision_adaptive - precision_standard) / precision_standard * 100):+.2f}%")
    print(f"   Recall: {((recall_adaptive - recall_standard) / recall_standard * 100):+.2f}%")
    print(f"   Balanced Acc: {((balanced_acc_adaptive - balanced_acc_standard) / balanced_acc_standard * 100):+.2f}%")
    
    return {
        'standard': {
            'f1': f1_standard,
            'precision': precision_standard,
            'recall': recall_standard,
            'balanced_accuracy': balanced_acc_standard,
            'predictions': y_pred_standard,
            'probabilities': y_proba_standard
        },
        'adaptive': {
            'f1': f1_adaptive,
            'precision': precision_adaptive,
            'recall': recall_adaptive,
            'balanced_accuracy': balanced_acc_adaptive,
            'predictions': y_pred_adaptive,
            'probabilities': y_proba_adaptive
        }
    }

def demonstrate_dynamic_sampling(X_train, y_train, X_test, y_test):
    """Demonstrate Dynamic Sampling Strategy"""
    print("\n" + "="*60)
    print("🔄 DYNAMIC SAMPLING STRATEGY DEMONSTRATION")
    print("="*60)
    
    dynamic_sampler = DynamicSamplingStrategy()
    
    # Simulate multiple iterations of adaptive sampling
    results = []
    
    for iteration in range(3):
        print(f"\n🔄 Iteration {iteration + 1}:")
        
        # Get adaptive sampling
        X_resampled, y_resampled, strategy = dynamic_sampler.adaptive_sample(
            X_train, y_train, iteration=iteration
        )
        
        print(f"   Selected Strategy: {strategy}")
        print(f"   Original samples: {len(y_train)} (Fault rate: {y_train.mean():.1%})")
        print(f"   Resampled samples: {len(y_resampled)} (Fault rate: {y_resampled.mean():.1%})")
        
        # Train model with resampled data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_resampled, y_resampled)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        
        # Record performance for adaptation
        performance = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'strategy': strategy
        }
        
        dynamic_sampler.record_performance(performance)
        results.append(performance)
    
    # Show adaptation learning
    print(f"\n🧠 ADAPTATION LEARNING:")
    for i, result in enumerate(results):
        print(f"   Iteration {i+1}: {result['strategy']} → F1: {result['f1_score']:.4f}")
    
    return results

def demonstrate_multi_objective_optimization(y_test, y_proba):
    """Demonstrate Multi-Objective Threshold Optimization"""
    print("\n" + "="*60)
    print("🎯 MULTI-OBJECTIVE THRESHOLD OPTIMIZATION")
    print("="*60)
    
    # Define cost matrix for wind turbine context
    cost_matrix = {
        'fn_cost': 15,  # Missing a fault is very expensive
        'fp_cost': 1,   # False alarm has some cost
        'tp_cost': 0,   # Correct detection
        'tn_cost': 0    # Correct normal prediction
    }
    
    print(f"💰 Cost Matrix:")
    print(f"   Missing Fault (FN): ${cost_matrix['fn_cost']}k")
    print(f"   False Alarm (FP): ${cost_matrix['fp_cost']}k")
    print(f"   Correct Detection (TP): ${cost_matrix['tp_cost']}k")
    print(f"   Correct Normal (TN): ${cost_matrix['tn_cost']}k")
    
    # Multi-objective optimization
    optimizer = MultiObjectiveThresholdOptimizer(
        objectives=['f1_score', 'recall', 'total_cost'],
        weights=[1.0, 1.5, 2.0]  # Prioritize cost and recall
    )
    
    optimal_thresholds, results_df = optimizer.optimize_threshold(
        y_test, y_proba, cost_matrix
    )
    
    print(f"\n🎯 OPTIMAL THRESHOLDS:")
    for objective, threshold in optimal_thresholds.items():
        if objective != 'pareto_solutions':
            print(f"   {objective}: {threshold:.3f}")
    
    # Show best solutions
    print(f"\n📊 BEST SOLUTIONS:")
    
    # Best F1
    best_f1_idx = results_df['f1_score'].idxmax()
    best_f1_row = results_df.iloc[best_f1_idx]
    print(f"\nBest F1 Score (threshold: {best_f1_row['threshold']:.3f}):")
    print(f"   F1: {best_f1_row['f1_score']:.4f}")
    print(f"   Recall: {best_f1_row['recall']:.4f}")
    print(f"   Precision: {best_f1_row['precision']:.4f}")
    print(f"   Cost: ${best_f1_row['total_cost']:.0f}k")
    
    # Best Cost
    best_cost_idx = results_df['total_cost'].idxmin()
    best_cost_row = results_df.iloc[best_cost_idx]
    print(f"\nBest Cost (threshold: {best_cost_row['threshold']:.3f}):")
    print(f"   F1: {best_cost_row['f1_score']:.4f}")
    print(f"   Recall: {best_cost_row['recall']:.4f}")
    print(f"   Precision: {best_cost_row['precision']:.4f}")
    print(f"   Cost: ${best_cost_row['total_cost']:.0f}k")
    
    # Multi-objective solution
    multi_obj_threshold = optimal_thresholds['multi_objective']
    multi_obj_row = results_df[results_df['threshold'] == multi_obj_threshold].iloc[0]
    print(f"\nMulti-Objective Solution (threshold: {multi_obj_threshold:.3f}):")
    print(f"   F1: {multi_obj_row['f1_score']:.4f}")
    print(f"   Recall: {multi_obj_row['recall']:.4f}")
    print(f"   Precision: {multi_obj_row['precision']:.4f}")
    print(f"   Cost: ${multi_obj_row['total_cost']:.0f}k")
    
    return optimal_thresholds, results_df

def demonstrate_context_aware_cost_learning(X_train, y_train, X_test, y_test):
    """Demonstrate Context-Aware Cost-Sensitive Learning"""
    print("\n" + "="*60)
    print("💰 CONTEXT-AWARE COST-SENSITIVE LEARNING")
    print("="*60)
    
    cost_learner = CostSensitiveLearning()
    
    # Demonstrate different operational contexts
    contexts = [
        {
            'name': 'Summer Day - Normal Operations',
            'context': {
                'season': 'summer',
                'wind_speed': 8,
                'maintenance_window': False,
                'turbine_age': 5,
                'grid_demand': 'normal'
            }
        },
        {
            'name': 'Winter Storm - High Demand',
            'context': {
                'season': 'winter',
                'wind_speed': 20,
                'maintenance_window': False,
                'turbine_age': 12,
                'grid_demand': 'high'
            }
        },
        {
            'name': 'Maintenance Window - Low Demand',
            'context': {
                'season': 'spring',
                'wind_speed': 5,
                'maintenance_window': True,
                'turbine_age': 8,
                'grid_demand': 'low'
            }
        }
    ]
    
    context_results = []
    
    for context_info in contexts:
        print(f"\n🌍 Scenario: {context_info['name']}")
        
        # Set operational context
        cost_learner.set_operational_context(context_info['context'])
        
        # Calculate dynamic costs
        costs = cost_learner.calculate_dynamic_costs()
        print(f"   Dynamic Costs:")
        print(f"     Missing Fault (FN): ${costs['fn_cost']:.1f}k")
        print(f"     False Alarm (FP): ${costs['fp_cost']:.1f}k")
        
        # Train context-aware model
        model, _ = cost_learner.train_cost_sensitive_model(X_train, y_train, 'random_forest')
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
        
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Calculate actual costs
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        total_cost = fn * costs['fn_cost'] + fp * costs['fp_cost']
        
        print(f"   Performance:")
        print(f"     F1 Score: {f1:.4f}")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall: {recall:.4f}")
        print(f"     Total Cost: ${total_cost:.1f}k")
        print(f"     False Negatives: {fn}")
        print(f"     False Positives: {fp}")
        
        context_results.append({
            'scenario': context_info['name'],
            'costs': costs,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'total_cost': total_cost,
            'fn': fn,
            'fp': fp
        })
    
    # Compare scenarios
    print(f"\n📊 SCENARIO COMPARISON:")
    for result in context_results:
        print(f"\n{result['scenario']}:")
        print(f"   Cost Ratio (FN:FP): {result['costs']['fn_cost']:.1f}:{result['costs']['fp_cost']:.1f}")
        print(f"   Total Cost: ${result['total_cost']:.1f}k")
        print(f"   F1 Score: {result['f1']:.4f}")
        print(f"   Missed Faults: {result['fn']}")
        print(f"   False Alarms: {result['fp']}")
    
    return context_results

def create_comprehensive_comparison_report(X_train, y_train, X_test, y_test):
    """Create a comprehensive comparison report of all techniques"""
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE IMBALANCE HANDLING COMPARISON REPORT")
    print("="*80)
    
    # Run comprehensive comparison
    results_df, techniques = comprehensive_imbalance_comparison(X_train, y_train, X_test, y_test)
    
    # Create summary report
    print(f"\n📊 SUMMARY REPORT:")
    print(f"   Total techniques tested: {len(results_df)}")
    print(f"   Test set size: {len(y_test)}")
    print(f"   Test set imbalance: {(y_test == 0).sum()}:{(y_test == 1).sum()}")
    
    # Top performers
    print(f"\n🏆 TOP PERFORMERS:")
    
    top_f1 = results_df.nlargest(3, 'f1_score')
    print(f"\nTop 3 by F1 Score:")
    for i, (_, row) in enumerate(top_f1.iterrows(), 1):
        print(f"   {i}. {row['technique']}: {row['f1_score']:.4f}")
    
    top_recall = results_df.nlargest(3, 'recall')
    print(f"\nTop 3 by Recall (Critical for fault detection):")
    for i, (_, row) in enumerate(top_recall.iterrows(), 1):
        print(f"   {i}. {row['technique']}: {row['recall']:.4f}")
    
    top_balanced = results_df.nlargest(3, 'balanced_accuracy')
    print(f"\nTop 3 by Balanced Accuracy:")
    for i, (_, row) in enumerate(top_balanced.iterrows(), 1):
        print(f"   {i}. {row['technique']}: {row['balanced_accuracy']:.4f}")
    
    # Create recommendations
    print(f"\n💡 RECOMMENDATIONS FOR WIND TURBINE FAULT DETECTION:")
    
    best_recall = results_df.loc[results_df['recall'].idxmax()]
    best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    best_balanced = results_df.loc[results_df['balanced_accuracy'].idxmax()]
    
    print(f"\n1️⃣  For Maximum Fault Detection (High Recall):")
    print(f"    Technique: {best_recall['technique']}")
    print(f"    Recall: {best_recall['recall']:.4f}")
    print(f"    F1 Score: {best_recall['f1_score']:.4f}")
    print(f"    Trade-off: May have more false alarms")
    
    print(f"\n2️⃣  For Balanced Performance (High F1):")
    print(f"    Technique: {best_f1['technique']}")
    print(f"    F1 Score: {best_f1['f1_score']:.4f}")
    print(f"    Recall: {best_f1['recall']:.4f}")
    print(f"    Precision: {best_f1['precision']:.4f}")
    
    print(f"\n3️⃣  For Overall Balanced Accuracy:")
    print(f"    Technique: {best_balanced['technique']}")
    print(f"    Balanced Accuracy: {best_balanced['balanced_accuracy']:.4f}")
    print(f"    Good for both classes equally")
    
    return results_df

def main():
    """Main demonstration function"""
    print("🚀 ADVANCED CLASS IMBALANCE HANDLING FOR WIND TURBINE FAULT DETECTION")
    print("="*80)
    
    # Load and prepare data
    df = load_turbine_data()
    df_engineered = engineer_advanced_features(df)
    
    # Prepare features and target
    feature_columns = [col for col in df_engineered.columns if col != 'Fault']
    X = df_engineered[feature_columns].values
    y = df_engineered['Fault'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Training samples: {len(X_train)} (Fault rate: {y_train.mean():.1%})")
    print(f"   Test samples: {len(X_test)} (Fault rate: {y_test.mean():.1%})")
    print(f"   Features: {X.shape[1]}")
    print(f"   Imbalance ratio: {(y == 0).sum()}:{(y == 1).sum()}")
    
    # Demonstrate techniques
    
    # 1. Adaptive Threshold Classifier
    adaptive_results = demonstrate_adaptive_threshold_classifier(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # 2. Dynamic Sampling Strategy
    dynamic_results = demonstrate_dynamic_sampling(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # 3. Multi-Objective Threshold Optimization
    # Use probabilities from standard classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_proba = rf.predict_proba(X_test_scaled)[:, 1]
    
    threshold_results = demonstrate_multi_objective_optimization(y_test, y_proba)
    
    # 4. Context-Aware Cost-Sensitive Learning
    context_results = demonstrate_context_aware_cost_learning(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # 5. Comprehensive Comparison
    comparison_results = create_comprehensive_comparison_report(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Final recommendations
    print("\n" + "="*80)
    print("🎯 FINAL RECOMMENDATIONS")
    print("="*80)
    
    print("\n💡 Key Insights for Wind Turbine Fault Detection:")
    print("\n1️⃣  Recall is Critical:")
    print("   - Missing a fault can lead to catastrophic failure")
    print("   - Consider techniques that maximize recall")
    print("   - Accept some false alarms to catch all faults")
    
    print("\n2️⃣  Context Matters:")
    print("   - Different operational conditions require different approaches")
    print("   - Winter/storm conditions need higher sensitivity")
    print("   - Maintenance windows allow more conservative approaches")
    
    print("\n3️⃣  Multi-Objective Optimization:")
    print("   - Balance multiple metrics (F1, recall, cost)")
    print("   - Use business context to weight objectives")
    print("   - Consider Pareto-optimal solutions")
    
    print("\n4️⃣  Ensemble Approaches:")
    print("   - Combine multiple techniques for robustness")
    print("   - Different techniques excel in different scenarios")
    print("   - Meta-learning can optimize combinations")
    
    print("\n5️⃣  Adaptive Systems:")
    print("   - Performance should improve over time")
    print("   - Feedback loops enable continuous optimization")
    print("   - Adjust thresholds based on local conditions")
    
    print("\n✅ Demonstration completed successfully!")
    print("   Use these techniques in your main wind turbine fault detection pipeline.")

if __name__ == "__main__":
    main()
