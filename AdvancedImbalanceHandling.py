"""
Advanced Class Imbalance Handling Techniques for Wind Turbine Fault Detection
=============================================================================

This module provides cutting-edge techniques for handling class imbalance
beyond traditional SMOTE and undersampling approaches.

Author: GitHub Copilot
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                             GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, f1_score, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Imbalanced learning libraries
from imblearn.over_sampling import (SMOTE, ADASYN, RandomOverSampler, 
                                   BorderlineSMOTE, SVMSMOTE, KMeansSMOTE)
from imblearn.under_sampling import (RandomUnderSampler, EditedNearestNeighbours, 
                                    TomekLinks, CondensedNearestNeighbour,
                                    OneSidedSelection, NeighbourhoodCleaningRule)
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import (BalancedRandomForestClassifier, 
                              BalancedBaggingClassifier,
                              EasyEnsembleClassifier,
                              RUSBoostClassifier)

# Advanced libraries (install if needed)
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    ADVANCED_LIBS = True
except ImportError:
    print("⚠️  Advanced libraries (XGBoost, LightGBM, CatBoost) not available")
    print("   Install with: pip install xgboost lightgbm catboost")
    ADVANCED_LIBS = False

# Clustering and manifold learning
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Statistical libraries
from scipy import stats
from scipy.spatial.distance import cdist


class AdaptiveThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive threshold classifier that adjusts decision threshold based on 
    local neighborhood density and data distribution.
    """
    
    def __init__(self, base_classifier=None, adaptation_method='density', 
                 n_neighbors=5, contamination=0.1):
        self.base_classifier = base_classifier or RandomForestClassifier(random_state=42)
        self.adaptation_method = adaptation_method
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
    def fit(self, X, y):
        """Fit the adaptive threshold classifier"""
        self.base_classifier.fit(X, y)
        
        # Calculate local density for each training sample
        self._calculate_local_densities(X, y)
        
        # Determine adaptive thresholds
        self._calculate_adaptive_thresholds(X, y)
        
        return self
    
    def _calculate_local_densities(self, X, y):
        """Calculate local neighborhood density for each sample"""
        from sklearn.neighbors import NearestNeighbors
        
        # Fit nearest neighbors
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        self.nn_.fit(X)
        
        # Calculate distances to k nearest neighbors
        distances, indices = self.nn_.kneighbors(X)
        
        # Local density = 1 / mean distance to k nearest neighbors
        self.local_densities_ = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
        
    def _calculate_adaptive_thresholds(self, X, y):
        """Calculate adaptive thresholds based on local properties"""
        # Get prediction probabilities
        y_proba = self.base_classifier.predict_proba(X)[:, 1]
        
        # Sort samples by density
        density_order = np.argsort(self.local_densities_)
        
        # Calculate percentile-based thresholds
        self.density_percentiles_ = np.percentile(self.local_densities_, 
                                                 [25, 50, 75, 90])
        
        # For each density region, find optimal threshold
        self.region_thresholds_ = {}
        
        for i, percentile in enumerate([25, 50, 75, 90, 100]):
            if i == 0:
                mask = self.local_densities_ <= self.density_percentiles_[0]
            elif i == len(self.density_percentiles_):
                mask = self.local_densities_ > self.density_percentiles_[-1]
            else:
                mask = ((self.local_densities_ > self.density_percentiles_[i-1]) & 
                       (self.local_densities_ <= self.density_percentiles_[i]))
            
            if mask.sum() > 10:  # Enough samples
                region_proba = y_proba[mask]
                region_y = y[mask]
                
                # Find optimal threshold for this region
                optimal_thresh = self._find_optimal_threshold(region_y, region_proba)
                self.region_thresholds_[percentile] = optimal_thresh
            else:
                self.region_thresholds_[percentile] = 0.5
    
    def _find_optimal_threshold(self, y_true, y_proba):
        """Find optimal threshold using F1 score"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            if len(np.unique(y_pred)) > 1:  # Both classes predicted
                f1 = f1_score(y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        return best_threshold
    
    def predict_proba(self, X):
        """Predict class probabilities with adaptive thresholds"""
        return self.base_classifier.predict_proba(X)
    
    def predict(self, X):
        """Predict classes using adaptive thresholds"""
        # Get base probabilities
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate local densities for test samples
        test_distances, _ = self.nn_.kneighbors(X)
        test_densities = 1.0 / (test_distances[:, 1:].mean(axis=1) + 1e-6)
        
        # Assign adaptive thresholds based on density
        predictions = np.zeros(len(X))
        
        for i, density in enumerate(test_densities):
            # Find appropriate threshold based on density
            if density <= self.density_percentiles_[0]:
                threshold = self.region_thresholds_[25]
            elif density <= self.density_percentiles_[1]:
                threshold = self.region_thresholds_[50]
            elif density <= self.density_percentiles_[2]:
                threshold = self.region_thresholds_[75]
            elif density <= self.density_percentiles_[3]:
                threshold = self.region_thresholds_[90]
            else:
                threshold = self.region_thresholds_[100]
            
            predictions[i] = 1 if y_proba[i] >= threshold else 0
        
        return predictions.astype(int)


class DynamicSamplingStrategy:
    """
    Dynamic sampling strategy that adapts based on data characteristics
    and model performance feedback.
    """
    
    def __init__(self):
        self.sampling_history_ = []
        self.performance_history_ = []
    
    def adaptive_sample(self, X, y, iteration=0):
        """
        Adaptively choose sampling strategy based on data characteristics
        and historical performance.
        """
        # Analyze data characteristics
        characteristics = self._analyze_data_characteristics(X, y)
        
        # Choose sampling strategy based on characteristics
        if iteration == 0:
            # First iteration - use data characteristics
            strategy = self._choose_initial_strategy(characteristics)
        else:
            # Subsequent iterations - use performance feedback
            strategy = self._choose_adaptive_strategy(characteristics)
        
        # Apply chosen strategy
        X_resampled, y_resampled = self._apply_strategy(X, y, strategy)
        
        # Store for future adaptation
        self.sampling_history_.append(strategy)
        
        return X_resampled, y_resampled, strategy
    
    def _analyze_data_characteristics(self, X, y):
        """Analyze data characteristics to guide sampling strategy"""
        minority_class = 1 if (y == 1).sum() < (y == 0).sum() else 0
        majority_class = 1 - minority_class
        
        minority_samples = X[y == minority_class]
        majority_samples = X[y == majority_class]
        
        characteristics = {
            'imbalance_ratio': (y == majority_class).sum() / (y == minority_class).sum(),
            'minority_size': (y == minority_class).sum(),
            'dimensionality': X.shape[1],
            'minority_density': self._calculate_class_density(minority_samples),
            'majority_density': self._calculate_class_density(majority_samples),
            'overlap_score': self._calculate_class_overlap(minority_samples, majority_samples),
            'noise_level': self._estimate_noise_level(X, y)
        }
        
        return characteristics
    
    def _calculate_class_density(self, samples):
        """Calculate density of a class using k-nearest neighbors"""
        if len(samples) < 5:
            return 0.0
        
        from sklearn.neighbors import NearestNeighbors
        k = min(5, len(samples) - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(samples)
        distances, _ = nn.kneighbors(samples)
        
        return 1.0 / (distances[:, 1:].mean() + 1e-6)
    
    def _calculate_class_overlap(self, minority_samples, majority_samples):
        """Calculate overlap between classes using centroid distance"""
        if len(minority_samples) == 0 or len(majority_samples) == 0:
            return 0.0
        
        minority_centroid = minority_samples.mean(axis=0)
        majority_centroid = majority_samples.mean(axis=0)
        
        # Distance between centroids
        centroid_distance = np.linalg.norm(minority_centroid - majority_centroid)
        
        # Average intra-class distances
        minority_spread = np.mean([np.linalg.norm(sample - minority_centroid) 
                                  for sample in minority_samples])
        majority_spread = np.mean([np.linalg.norm(sample - majority_centroid) 
                                  for sample in majority_samples])
        
        # Overlap score (lower = more overlap)
        overlap_score = centroid_distance / (minority_spread + majority_spread + 1e-6)
        
        return 1.0 / (1.0 + overlap_score)  # Normalize to [0, 1]
    
    def _estimate_noise_level(self, X, y):
        """Estimate noise level using k-NN consistency"""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        # Use small k-NN to estimate noise
        knn = KNeighborsClassifier(n_neighbors=3)
        scores = cross_val_score(knn, X, y, cv=3, scoring='accuracy')
        
        # Lower accuracy suggests higher noise
        return 1.0 - scores.mean()
    
    def _choose_initial_strategy(self, characteristics):
        """Choose initial sampling strategy based on data characteristics"""
        imbalance_ratio = characteristics['imbalance_ratio']
        minority_size = characteristics['minority_size']
        overlap_score = characteristics['overlap_score']
        noise_level = characteristics['noise_level']
        
        # Decision rules based on characteristics
        if imbalance_ratio > 20:  # Very imbalanced
            if minority_size < 100:  # Very few samples
                return 'ADASYN'  # Adaptive synthetic sampling
            elif overlap_score > 0.7:  # High overlap
                return 'BorderlineSMOTE'  # Focus on borderline cases
            else:
                return 'SMOTE'
        elif imbalance_ratio > 5:  # Moderately imbalanced
            if noise_level > 0.3:  # High noise
                return 'SMOTE+Tomek'  # Remove noisy samples
            else:
                return 'SMOTE'
        else:  # Slightly imbalanced
            if noise_level > 0.3:
                return 'EditedNearestNeighbours'  # Clean majority class
            else:
                return 'RandomOverSampler'
    
    def _choose_adaptive_strategy(self, characteristics):
        """Choose strategy based on historical performance"""
        if len(self.performance_history_) == 0:
            return self._choose_initial_strategy(characteristics)
        
        # Find best performing strategy so far
        best_idx = np.argmax([perf['f1_score'] for perf in self.performance_history_])
        best_strategy = self.sampling_history_[best_idx]
        
        # If recent performance is declining, try a different strategy
        if len(self.performance_history_) >= 3:
            recent_f1 = [perf['f1_score'] for perf in self.performance_history_[-3:]]
            if recent_f1[0] > recent_f1[-1]:  # Performance declining
                # Try alternative strategy
                alternatives = ['SMOTE', 'ADASYN', 'BorderlineSMOTE', 'SMOTE+Tomek']
                alternatives = [s for s in alternatives if s != best_strategy]
                return np.random.choice(alternatives)
        
        return best_strategy
    
    def _apply_strategy(self, X, y, strategy):
        """Apply the chosen sampling strategy"""
        strategies = {
            'SMOTE': SMOTE(random_state=42),
            'ADASYN': ADASYN(random_state=42),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
            'SMOTE+Tomek': SMOTETomek(random_state=42),
            'SMOTE+ENN': SMOTEENN(random_state=42),
            'RandomOverSampler': RandomOverSampler(random_state=42),
            'EditedNearestNeighbours': EditedNearestNeighbours()
        }
        
        if strategy in strategies:
            return strategies[strategy].fit_resample(X, y)
        else:
            return X, y
    
    def record_performance(self, performance_metrics):
        """Record performance for adaptive learning"""
        self.performance_history_.append(performance_metrics)


class MultiObjectiveThresholdOptimizer:
    """
    Multi-objective threshold optimization considering multiple metrics
    and business constraints for wind turbine fault detection.
    """
    
    def __init__(self, objectives=['f1', 'recall', 'cost'], weights=None):
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
        
    def optimize_threshold(self, y_true, y_proba, cost_matrix=None):
        """
        Find Pareto-optimal thresholds considering multiple objectives.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            cost_matrix: Dict with 'fn_cost', 'fp_cost', 'tp_cost', 'tn_cost'
        
        Returns:
            Dict with optimal thresholds for different objectives
        """
        if cost_matrix is None:
            cost_matrix = {'fn_cost': 10, 'fp_cost': 1, 'tp_cost': 0, 'tn_cost': 0}
        
        thresholds = np.arange(0.01, 1.0, 0.01)
        
        # Calculate all metrics for all thresholds
        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            metrics = self._calculate_all_metrics(y_true, y_pred, y_proba, cost_matrix)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # Find optimal thresholds for each objective
        optimal_thresholds = {}
        
        # Single objective optimizations
        optimal_thresholds['max_f1'] = results_df.loc[results_df['f1_score'].idxmax(), 'threshold']
        optimal_thresholds['max_recall'] = results_df.loc[results_df['recall'].idxmax(), 'threshold']
        optimal_thresholds['max_precision'] = results_df.loc[results_df['precision'].idxmax(), 'threshold']
        optimal_thresholds['min_cost'] = results_df.loc[results_df['total_cost'].idxmin(), 'threshold']
        optimal_thresholds['max_balanced_acc'] = results_df.loc[results_df['balanced_accuracy'].idxmax(), 'threshold']
        
        # Multi-objective optimization using weighted sum
        results_df['composite_score'] = 0
        for i, objective in enumerate(self.objectives):
            if objective == 'cost':
                # For cost, we want to minimize (so take negative)
                normalized = (results_df['total_cost'].max() - results_df['total_cost']) / \
                           (results_df['total_cost'].max() - results_df['total_cost'].min() + 1e-6)
            else:
                # For other metrics, we want to maximize
                normalized = (results_df[objective] - results_df[objective].min()) / \
                           (results_df[objective].max() - results_df[objective].min() + 1e-6)
            
            results_df['composite_score'] += self.weights[i] * normalized
        
        optimal_thresholds['multi_objective'] = results_df.loc[results_df['composite_score'].idxmax(), 'threshold']
        
        # Pareto frontier analysis
        pareto_thresholds = self._find_pareto_frontier(results_df)
        optimal_thresholds['pareto_solutions'] = pareto_thresholds
        
        return optimal_thresholds, results_df
    
    def _calculate_all_metrics(self, y_true, y_pred, y_proba, cost_matrix):
        """Calculate comprehensive metrics for a given prediction"""
        from sklearn.metrics import (precision_score, recall_score, f1_score, 
                                    balanced_accuracy_score, roc_auc_score)
        
        # Handle edge cases
        if len(np.unique(y_pred)) == 1:
            if y_pred[0] == 1:  # All predicted as positive
                precision = (y_true == 1).sum() / len(y_true)
                recall = 1.0 if (y_true == 1).sum() > 0 else 0.0
            else:  # All predicted as negative
                precision = 0.0
                recall = 0.0
            f1 = 0.0
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Calculate costs
        total_cost = (fn * cost_matrix['fn_cost'] + 
                     fp * cost_matrix['fp_cost'] + 
                     tp * cost_matrix['tp_cost'] + 
                     tn * cost_matrix['tn_cost'])
        
        # Other metrics
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'specificity': specificity,
            'total_cost': total_cost,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def _find_pareto_frontier(self, results_df):
        """Find Pareto-optimal solutions"""
        # For this example, consider F1 vs Cost trade-off
        pareto_solutions = []
        
        for i, row in results_df.iterrows():
            is_dominated = False
            
            for j, other_row in results_df.iterrows():
                if i != j:
                    # Check if current solution is dominated
                    # (other has higher F1 AND lower cost)
                    if (other_row['f1_score'] >= row['f1_score'] and 
                        other_row['total_cost'] <= row['total_cost'] and
                        (other_row['f1_score'] > row['f1_score'] or 
                         other_row['total_cost'] < row['total_cost'])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_solutions.append(row['threshold'])
        
        return pareto_solutions


class HybridSamplingEnsemble:
    """
    Hybrid ensemble that combines multiple sampling techniques with
    different base models and uses meta-learning to combine predictions.
    """
    
    def __init__(self, base_models=None, sampling_strategies=None, meta_learner=None):
        self.base_models = base_models or self._get_default_models()
        self.sampling_strategies = sampling_strategies or self._get_default_strategies()
        self.meta_learner = meta_learner or LogisticRegression(random_state=42)
        self.trained_models = []
        
    def _get_default_models(self):
        """Get default set of diverse base models"""
        models = [
            RandomForestClassifier(n_estimators=100, random_state=42),
            ExtraTreesClassifier(n_estimators=100, random_state=42),
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            LogisticRegression(random_state=42),
            SVC(probability=True, random_state=42)
        ]
        
        # Add advanced models if available
        if ADVANCED_LIBS:
            models.extend([
                xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                lgb.LGBMClassifier(random_state=42, verbose=-1),
                CatBoostClassifier(random_state=42, verbose=False)
            ])
        
        return models
    
    def _get_default_strategies(self):
        """Get default sampling strategies"""
        return {
            'SMOTE': SMOTE(random_state=42),
            'ADASYN': ADASYN(random_state=42),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
            'SMOTE+Tomek': SMOTETomek(random_state=42),
            'RandomOverSampler': RandomOverSampler(random_state=42)
        }
    
    def fit(self, X, y):
        """Fit the hybrid ensemble"""
        print("🔧 Training Hybrid Sampling Ensemble...")
        
        # Train models with different sampling strategies
        self.trained_models = []
        meta_features = []
        meta_targets = []
        
        # Use cross-validation to generate meta-features
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            fold_predictions = []
            
            for strategy_name, strategy in self.sampling_strategies.items():
                try:
                    # Apply sampling strategy
                    X_resampled, y_resampled = strategy.fit_resample(X_train_fold, y_train_fold)
                    
                    for model_idx, model in enumerate(self.base_models):
                        # Clone and train model
                        model_copy = model.__class__(**model.get_params())
                        model_copy.fit(X_resampled, y_resampled)
                        
                        # Predict on validation set
                        val_proba = model_copy.predict_proba(X_val_fold)[:, 1]
                        fold_predictions.append(val_proba)
                        
                        # Store model for final training
                        if fold == 0:  # Only store on first fold
                            self.trained_models.append({
                                'model': model_copy,
                                'strategy': strategy_name,
                                'model_type': type(model).__name__
                            })
                
                except Exception as e:
                    print(f"   ⚠️  Skipped {strategy_name}: {str(e)}")
                    continue
            
            # Collect meta-features for this fold
            if fold_predictions:
                meta_features.extend(np.column_stack(fold_predictions))
                meta_targets.extend(y_val_fold)
        
        # Train meta-learner
        if meta_features:
            meta_X = np.array(meta_features)
            meta_y = np.array(meta_targets)
            self.meta_learner.fit(meta_X, meta_y)
            print(f"   ✅ Trained meta-learner with {meta_X.shape[1]} base predictions")
        
        # Final training on full dataset
        self._train_final_models(X, y)
        
        return self
    
    def _train_final_models(self, X, y):
        """Train final models on full dataset"""
        self.final_models = []
        
        for strategy_name, strategy in self.sampling_strategies.items():
            try:
                # Apply sampling strategy
                X_resampled, y_resampled = strategy.fit_resample(X, y)
                
                for model in self.base_models:
                    # Clone and train model
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_resampled, y_resampled)
                    
                    self.final_models.append({
                        'model': model_copy,
                        'strategy': strategy_name,
                        'model_type': type(model).__name__
                    })
            
            except Exception as e:
                continue
    
    def predict_proba(self, X):
        """Predict probabilities using meta-learning"""
        if not hasattr(self, 'final_models'):
            raise ValueError("Model not fitted yet!")
        
        # Get base model predictions
        base_predictions = []
        
        for model_info in self.final_models:
            try:
                proba = model_info['model'].predict_proba(X)[:, 1]
                base_predictions.append(proba)
            except:
                continue
        
        if not base_predictions:
            raise ValueError("No valid base predictions!")
        
        # Stack predictions for meta-learner
        meta_X = np.column_stack(base_predictions)
        
        # Get meta-learner predictions
        if hasattr(self.meta_learner, 'predict_proba'):
            return self.meta_learner.predict_proba(meta_X)[:, 1]
        else:
            return self.meta_learner.predict(meta_X)
    
    def predict(self, X):
        """Predict classes"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


class CostSensitiveLearning:
    """
    Advanced cost-sensitive learning with dynamic cost adjustment
    based on operational conditions and maintenance windows.
    """
    
    def __init__(self):
        self.cost_history = []
        self.operational_context = None
    
    def set_operational_context(self, context):
        """
        Set operational context for dynamic cost adjustment.
        
        Args:
            context: Dict with keys like 'season', 'wind_conditions', 
                    'maintenance_window', 'turbine_age', etc.
        """
        self.operational_context = context
    
    def calculate_dynamic_costs(self, base_fn_cost=10, base_fp_cost=1):
        """Calculate dynamic costs based on operational context"""
        if self.operational_context is None:
            return {'fn_cost': base_fn_cost, 'fp_cost': base_fp_cost}
        
        # Adjust costs based on context
        fn_multiplier = 1.0
        fp_multiplier = 1.0
        
        # Seasonal adjustments
        if self.operational_context.get('season') == 'winter':
            fn_multiplier *= 1.5  # Higher cost in winter (harder repairs)
        elif self.operational_context.get('season') == 'summer':
            fn_multiplier *= 0.8  # Lower cost in summer
        
        # Wind condition adjustments
        wind_speed = self.operational_context.get('wind_speed', 0)
        if wind_speed > 15:  # High wind
            fn_multiplier *= 1.3  # Harder to repair in high wind
            fp_multiplier *= 0.7  # False alarms less costly (would shut down anyway)
        
        # Maintenance window adjustments
        if self.operational_context.get('maintenance_window', False):
            fp_multiplier *= 0.5  # False alarms cheaper during maintenance
        
        # Turbine age adjustments
        turbine_age = self.operational_context.get('turbine_age', 0)
        if turbine_age > 10:  # Old turbine
            fn_multiplier *= 1.2  # Missing faults more costly
        
        # Grid demand adjustments
        grid_demand = self.operational_context.get('grid_demand', 'normal')
        if grid_demand == 'high':
            fp_multiplier *= 1.5  # False shutdowns very costly during high demand
        elif grid_demand == 'low':
            fp_multiplier *= 0.8  # False shutdowns less costly
        
        return {
            'fn_cost': base_fn_cost * fn_multiplier,
            'fp_cost': base_fp_cost * fp_multiplier
        }
    
    def train_cost_sensitive_model(self, X, y, model_type='random_forest'):
        """Train cost-sensitive model with current context"""
        costs = self.calculate_dynamic_costs()
        
        # Calculate class weights based on costs
        minority_class = 1 if (y == 1).sum() < (y == 0).sum() else 0
        majority_class = 1 - minority_class
        
        # Weight inversely proportional to cost of misclassification
        class_weights = {
            minority_class: costs['fn_cost'],
            majority_class: costs['fp_cost']
        }
        
        # Normalize weights
        total_weight = sum(class_weights.values())
        class_weights = {k: v/total_weight for k, v in class_weights.items()}
        
        # Train model with calculated weights
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100, 
                class_weight=class_weights,
                random_state=42
            )
        elif model_type == 'xgboost' and ADVANCED_LIBS:
            # For XGBoost, use scale_pos_weight
            scale_pos_weight = class_weights[1] / class_weights[0]
            model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            model = LogisticRegression(
                class_weight=class_weights,
                random_state=42
            )
        
        model.fit(X, y)
        return model, costs


def comprehensive_imbalance_comparison(X_train, y_train, X_test, y_test):
    """
    Comprehensive comparison of all available imbalance handling techniques.
    
    Returns detailed results for analysis and decision making.
    """
    print("🧪 COMPREHENSIVE IMBALANCE HANDLING COMPARISON")
    print("="*60)
    
    techniques = {}
    results = []
    
    # 1. Original (no resampling)
    techniques['Original'] = (X_train, y_train)
    
    # 2. Basic SMOTE variants
    try:
        smote = SMOTE(random_state=42)
        techniques['SMOTE'] = smote.fit_resample(X_train, y_train)
    except:
        print("   ⚠️  SMOTE failed")
    
    try:
        adasyn = ADASYN(random_state=42)
        techniques['ADASYN'] = adasyn.fit_resample(X_train, y_train)
    except:
        print("   ⚠️  ADASYN failed")
    
    try:
        borderline = BorderlineSMOTE(random_state=42)
        techniques['BorderlineSMOTE'] = borderline.fit_resample(X_train, y_train)
    except:
        print("   ⚠️  BorderlineSMOTE failed")
    
    # 3. Combined techniques
    try:
        smote_tomek = SMOTETomek(random_state=42)
        techniques['SMOTE+Tomek'] = smote_tomek.fit_resample(X_train, y_train)
    except:
        print("   ⚠️  SMOTE+Tomek failed")
    
    try:
        smote_enn = SMOTEENN(random_state=42)
        techniques['SMOTE+ENN'] = smote_enn.fit_resample(X_train, y_train)
    except:
        print("   ⚠️  SMOTE+ENN failed")
    
    # 4. Advanced techniques
    try:
        # K-Means SMOTE
        kmeans_smote = KMeansSMOTE(random_state=42)
        techniques['KMeans-SMOTE'] = kmeans_smote.fit_resample(X_train, y_train)
    except:
        print("   ⚠️  KMeans-SMOTE failed")
    
    # 5. Ensemble approaches
    try:
        # Balanced Random Forest
        balanced_rf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
        balanced_rf.fit(X_train, y_train)
        techniques['BalancedRF'] = (X_train, y_train, balanced_rf)  # Special case
    except:
        print("   ⚠️  Balanced Random Forest failed")
    
    try:
        # Easy Ensemble
        easy_ensemble = EasyEnsembleClassifier(n_estimators=10, random_state=42)
        easy_ensemble.fit(X_train, y_train)
        techniques['EasyEnsemble'] = (X_train, y_train, easy_ensemble)  # Special case
    except:
        print("   ⚠️  Easy Ensemble failed")
    
    # 6. Undersampling techniques
    try:
        # Edited Nearest Neighbours
        enn = EditedNearestNeighbours()
        techniques['ENN'] = enn.fit_resample(X_train, y_train)
    except:
        print("   ⚠️  ENN failed")
    
    try:
        # Tomek Links
        tomek = TomekLinks()
        techniques['TomekLinks'] = tomek.fit_resample(X_train, y_train)
    except:
        print("   ⚠️  Tomek Links failed")
    
    # Evaluate each technique
    for name, data in techniques.items():
        print(f"\nEvaluating: {name}")
        
        if len(data) == 3:  # Special ensemble cases
            X_train_tech, y_train_tech, model = data
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        else:
            X_train_tech, y_train_tech = data
            
            # Train a standard model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_tech, y_train_tech)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import (classification_report, confusion_matrix,
                                   precision_score, recall_score, f1_score,
                                   balanced_accuracy_score, roc_auc_score)
        
        # Handle edge cases
        if len(np.unique(y_pred)) == 1:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
        
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        # AUC (if probabilities available)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        result = {
            'technique': name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'training_samples': len(y_train_tech),
            'training_balance': (y_train_tech == 1).mean()
        }
        
        results.append(result)
        
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        print(f"   Balanced Accuracy: {balanced_acc:.3f}")
        if auc:
            print(f"   AUC: {auc:.3f}")
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Rank techniques
    print(f"\n" + "="*60)
    print("🏆 TECHNIQUE RANKINGS")
    print("="*60)
    
    for metric in ['f1_score', 'recall', 'precision', 'balanced_accuracy']:
        if metric in results_df.columns:
            top_3 = results_df.nlargest(3, metric)[['technique', metric]]
            print(f"\nTop 3 by {metric.upper()}:")
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                print(f"   {i}. {row['technique']}: {row[metric]:.3f}")
    
    return results_df, techniques


# Usage example and demonstration
if __name__ == "__main__":
    print("🚀 Advanced Class Imbalance Handling for Wind Turbine Fault Detection")
    print("="*80)
    
    print("\n📋 Available Advanced Techniques:")
    print("   1. Adaptive Threshold Classifier")
    print("   2. Dynamic Sampling Strategy")
    print("   3. Multi-Objective Threshold Optimizer")
    print("   4. Hybrid Sampling Ensemble")
    print("   5. Cost-Sensitive Learning with Context")
    print("   6. Comprehensive Technique Comparison")
    
    print("\n💡 Example Usage:")
    print("""
    # Example 1: Adaptive Threshold Classifier
    adaptive_clf = AdaptiveThresholdClassifier(
        base_classifier=RandomForestClassifier(n_estimators=100),
        adaptation_method='density',
        n_neighbors=5
    )
    adaptive_clf.fit(X_train, y_train)
    y_pred = adaptive_clf.predict(X_test)
    
    # Example 2: Dynamic Sampling
    dynamic_sampler = DynamicSamplingStrategy()
    X_resampled, y_resampled, strategy = dynamic_sampler.adaptive_sample(X_train, y_train)
    
    # Example 3: Multi-Objective Optimization
    optimizer = MultiObjectiveThresholdOptimizer(
        objectives=['f1', 'recall', 'cost'],
        weights=[1.0, 1.5, 2.0]  # Prioritize cost and recall
    )
    optimal_thresholds, results = optimizer.optimize_threshold(y_test, y_proba)
    
    # Example 4: Hybrid Ensemble
    hybrid_ensemble = HybridSamplingEnsemble()
    hybrid_ensemble.fit(X_train, y_train)
    y_pred = hybrid_ensemble.predict(X_test)
    
    # Example 5: Context-Aware Cost-Sensitive Learning
    cost_learner = CostSensitiveLearning()
    cost_learner.set_operational_context({
        'season': 'winter',
        'wind_speed': 18,
        'maintenance_window': False,
        'turbine_age': 12,
        'grid_demand': 'high'
    })
    model, costs = cost_learner.train_cost_sensitive_model(X_train, y_train)
    """)
    
    print("\n✅ Module loaded successfully!")
    print("   Import this module in your main script to use these techniques.")
