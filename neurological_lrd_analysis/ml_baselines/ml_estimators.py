"""
Machine Learning Baseline Estimators for Hurst Exponent Estimation.

This module provides classical machine learning methods for estimating Hurst exponents,
including Random Forest, Support Vector Regression, and Gradient Boosting Trees.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import joblib
from pathlib import Path

# Lazy imports for ML libraries
def _lazy_import_sklearn():
    """Lazy import of sklearn modules"""
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        return (RandomForestRegressor, GradientBoostingRegressor, SVR, 
                cross_val_score, train_test_split, StandardScaler,
                mean_squared_error, mean_absolute_error, r2_score)
    except ImportError:
        warnings.warn("scikit-learn not available, ML estimators will be disabled")
        return None, None, None, None, None, None, None, None, None

def _lazy_import_optuna():
    """Lazy import of optuna"""
    try:
        import optuna
        return optuna
    except ImportError:
        warnings.warn("Optuna not available, hyperparameter optimization will be disabled")
        return None


class MLBaselineType(Enum):
    """Available ML baseline types."""
    RANDOM_FOREST = "random_forest"
    SVR = "svr"
    GRADIENT_BOOSTING = "gradient_boosting"


@dataclass
class MLTrainingResult:
    """Results from ML model training."""
    model: Any
    training_score: float
    validation_score: float
    cross_val_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    training_time: float = 0.0


@dataclass
class MLPredictionResult:
    """Results from ML model prediction."""
    hurst_estimate: float
    confidence_interval: Optional[Tuple[float, float]] = None
    prediction_uncertainty: Optional[float] = None
    feature_contributions: Optional[Dict[str, float]] = None


class BaseMLEstimator:
    """Base class for ML-based Hurst estimators."""
    
    def __init__(self, 
                 model_name: str,
                 feature_extractor: Optional[Any] = None,
                 scaler: Optional[Any] = None,
                 model: Optional[Any] = None):
        """
        Initialize the ML estimator.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        feature_extractor : Any, optional
            Feature extractor instance
        scaler : Any, optional
            Data scaler instance
        model : Any, optional
            Trained model
        """
        self.model_name = model_name
        self.feature_extractor = feature_extractor
        self.scaler = scaler
        self.model = model
        self.is_trained = model is not None
        
        # Lazy import ML libraries
        self._sklearn_modules = _lazy_import_sklearn()
        self._optuna = _lazy_import_optuna()
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              validation_split: float = 0.2,
              random_state: int = 42) -> MLTrainingResult:
        """
        Train the ML model.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values (Hurst exponents)
        validation_split : float
            Fraction of data for validation
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        MLTrainingResult
            Training results
        """
        if self._sklearn_modules[0] is None:
            raise ImportError("scikit-learn not available")
        
        import time
        start_time = time.time()
        
        # Split data
        X_train, X_val, y_train, y_val = self._sklearn_modules[4](
            X, y, test_size=validation_split, random_state=random_state
        )
        
        # Scale features
        if self.scaler is None:
            self.scaler = self._sklearn_modules[5]()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        # Cross-validation (adjust folds based on data size)
        n_folds = min(5, len(y_train) // 2)  # Ensure we have enough samples per fold
        if n_folds < 2:
            cv_scores = np.array([0.0])  # Skip CV for very small datasets
        else:
            cv_scores = self._sklearn_modules[3](
                self.model, X_train_scaled, y_train, cv=n_folds, scoring='neg_mean_squared_error'
            )
            cv_scores = -cv_scores  # Convert to positive MSE
        
        # Feature importance
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                getattr(self.feature_extractor, 'feature_names', []), 
                self.model.feature_importances_
            ))
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        return MLTrainingResult(
            model=self.model,
            training_score=train_score,
            validation_score=val_score,
            cross_val_scores=cv_scores.tolist(),
            feature_importance=feature_importance,
            training_time=training_time
        )
    
    def predict(self, X: np.ndarray) -> MLPredictionResult:
        """
        Predict Hurst exponent using trained model.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        MLPredictionResult
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        hurst_estimate = float(self.model.predict(X_scaled)[0])
        
        # Calculate confidence interval (simplified)
        # In practice, use ensemble methods or bootstrap for better uncertainty
        confidence_interval = self._calculate_confidence_interval(X_scaled)
        
        # Feature contributions (for tree-based models)
        feature_contributions = None
        if hasattr(self.model, 'feature_importances_'):
            feature_contributions = dict(zip(
                getattr(self.feature_extractor, 'feature_names', []),
                self.model.feature_importances_
            ))
        
        return MLPredictionResult(
            hurst_estimate=hurst_estimate,
            confidence_interval=confidence_interval,
            feature_contributions=feature_contributions
        )
    
    def _calculate_confidence_interval(self, X_scaled: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for prediction."""
        # Simplified confidence interval calculation
        # In practice, use ensemble methods or bootstrap
        if hasattr(self.model, 'predict_proba'):
            # For models with probability output
            proba = self.model.predict_proba(X_scaled)
            # Simplified CI calculation
            std = np.std(proba)
            return (self.model.predict(X_scaled)[0] - 1.96 * std,
                    self.model.predict(X_scaled)[0] + 1.96 * std)
        else:
            # For regression models, use a simplified approach
            prediction = self.model.predict(X_scaled)[0]
            # Assume 10% uncertainty (this should be improved with proper uncertainty quantification)
            uncertainty = 0.1
            return (prediction - uncertainty, prediction + uncertainty)
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor,
            'model_name': self.model_name
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_extractor = model_data['feature_extractor']
        self.model_name = model_data['model_name']
        self.is_trained = True


class RandomForestEstimator(BaseMLEstimator):
    """Random Forest estimator for Hurst exponent prediction."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Random Forest estimator.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int
            Minimum samples to split a node
        min_samples_leaf : int
            Minimum samples in a leaf
        random_state : int
            Random state for reproducibility
        **kwargs
            Additional parameters for RandomForestRegressor
        """
        super().__init__("RandomForest")
        
        if self._sklearn_modules[0] is None:
            raise ImportError("scikit-learn not available")
        
        self.model = self._sklearn_modules[0](
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )


class SVREstimator(BaseMLEstimator):
    """Support Vector Regression estimator for Hurst exponent prediction."""
    
    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: Union[str, float] = 'scale',
                 epsilon: float = 0.1,
                 **kwargs):
        """
        Initialize SVR estimator.
        
        Parameters:
        -----------
        kernel : str
            Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
        C : float
            Regularization parameter
        gamma : str or float
            Kernel coefficient
        epsilon : float
            Epsilon-tube parameter
        **kwargs
            Additional parameters for SVR
        """
        super().__init__("SVR")
        
        if self._sklearn_modules[2] is None:
            raise ImportError("scikit-learn not available")
        
        self.model = self._sklearn_modules[2](
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            **kwargs
        )


class GradientBoostingEstimator(BaseMLEstimator):
    """Gradient Boosting estimator for Hurst exponent prediction."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Gradient Boosting estimator.
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages
        learning_rate : float
            Learning rate
        max_depth : int
            Maximum depth of trees
        min_samples_split : int
            Minimum samples to split a node
        min_samples_leaf : int
            Minimum samples in a leaf
        random_state : int
            Random state for reproducibility
        **kwargs
            Additional parameters for GradientBoostingRegressor
        """
        super().__init__("GradientBoosting")
        
        if self._sklearn_modules[1] is None:
            raise ImportError("scikit-learn not available")
        
        self.model = self._sklearn_modules[1](
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )


class MLBaselineFactory:
    """Factory for creating ML baseline estimators."""
    
    @staticmethod
    def create_estimator(estimator_type: MLBaselineType, **kwargs) -> BaseMLEstimator:
        """
        Create an ML estimator of the specified type.
        
        Parameters:
        -----------
        estimator_type : MLBaselineType
            Type of estimator to create
        **kwargs
            Parameters for the estimator
            
        Returns:
        --------
        BaseMLEstimator
            Created estimator
        """
        if estimator_type == MLBaselineType.RANDOM_FOREST:
            return RandomForestEstimator(**kwargs)
        elif estimator_type == MLBaselineType.SVR:
            return SVREstimator(**kwargs)
        elif estimator_type == MLBaselineType.GRADIENT_BOOSTING:
            return GradientBoostingEstimator(**kwargs)
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
    
    @staticmethod
    def create_all_estimators(**kwargs) -> Dict[MLBaselineType, BaseMLEstimator]:
        """
        Create all available ML estimators.
        
        Parameters:
        -----------
        **kwargs
            Common parameters for all estimators
            
        Returns:
        --------
        Dict[MLBaselineType, BaseMLEstimator]
            Dictionary of created estimators
        """
        estimators = {}
        for estimator_type in MLBaselineType:
            try:
                estimators[estimator_type] = MLBaselineFactory.create_estimator(
                    estimator_type, **kwargs
                )
            except ImportError as e:
                warnings.warn(f"Could not create {estimator_type.value}: {e}")
        
        return estimators


def train_ml_ensemble(X: np.ndarray, 
                     y: np.ndarray,
                     estimator_types: Optional[List[MLBaselineType]] = None,
                     validation_split: float = 0.2,
                     random_state: int = 42) -> Dict[MLBaselineType, MLTrainingResult]:
    """
    Train an ensemble of ML estimators.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    estimator_types : List[MLBaselineType], optional
        Types of estimators to train
    validation_split : float
        Fraction of data for validation
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    Dict[MLBaselineType, MLTrainingResult]
        Training results for each estimator
    """
    if estimator_types is None:
        estimator_types = list(MLBaselineType)
    
    results = {}
    for estimator_type in estimator_types:
        try:
            estimator = MLBaselineFactory.create_estimator(estimator_type)
            result = estimator.train(X, y, validation_split, random_state)
            results[estimator_type] = result
        except Exception as e:
            warnings.warn(f"Failed to train {estimator_type.value}: {e}")
    
    return results


def evaluate_ml_ensemble(estimators: Dict[MLBaselineType, BaseMLEstimator],
                        X_test: np.ndarray,
                        y_test: np.ndarray) -> Dict[MLBaselineType, Dict[str, float]]:
    """
    Evaluate an ensemble of trained ML estimators.
    
    Parameters:
    -----------
    estimators : Dict[MLBaselineType, BaseMLEstimator]
        Trained estimators
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets
        
    Returns:
    --------
    Dict[MLBaselineType, Dict[str, float]]
        Evaluation metrics for each estimator
    """
    results = {}
    
    for estimator_type, estimator in estimators.items():
        if not estimator.is_trained:
            continue
        
        try:
            # Make predictions
            predictions = []
            for i in range(len(X_test)):
                pred_result = estimator.predict(X_test[i:i+1])
                predictions.append(pred_result.hurst_estimate)
            
            predictions = np.array(predictions)
            
            # Calculate metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            r2 = 1 - mse / np.var(y_test)
            
            results[estimator_type] = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'rmse': float(np.sqrt(mse))
            }
            
        except Exception as e:
            warnings.warn(f"Failed to evaluate {estimator_type.value}: {e}")
    
    return results
