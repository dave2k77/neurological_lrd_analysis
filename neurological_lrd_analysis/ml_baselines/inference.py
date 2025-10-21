"""
Inference System for Pretrained ML Models.

This module provides a high-level interface for using pretrained models
for Hurst exponent estimation, including batch processing, ensemble predictions,
and uncertainty quantification.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings
from pathlib import Path

from .pretrained_models import PretrainedModelManager, ModelMetadata, ModelStatus
from .feature_extraction import TimeSeriesFeatureExtractor
from .ml_estimators import MLBaselineType, BaseMLEstimator


@dataclass
class PredictionResult:
    """Result from model prediction."""
    hurst_estimate: float
    confidence_interval: Optional[Tuple[float, float]] = None
    uncertainty: Optional[float] = None
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    prediction_time: Optional[float] = None
    feature_contributions: Optional[Dict[str, float]] = None


@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    mean_estimate: float
    std_estimate: float
    individual_predictions: List[float]
    model_weights: Optional[Dict[str, float]] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class PretrainedInference:
    """
    High-level interface for pretrained model inference.
    
    Provides easy-to-use methods for Hurst exponent estimation using
    pretrained ML models with support for single predictions, batch processing,
    and ensemble methods.
    """
    
    def __init__(self, models_dir: Union[str, Path] = "pretrained_models"):
        """
        Initialize the inference system.
        
        Parameters:
        -----------
        models_dir : str or Path
            Directory containing pretrained models
        """
        self.manager = PretrainedModelManager(models_dir)
        self._feature_extractor = None
        self._model_cache = {}
    
    def _get_feature_extractor(self, model_metadata: ModelMetadata) -> TimeSeriesFeatureExtractor:
        """Get or create feature extractor with model-specific configuration."""
        config_key = str(model_metadata.feature_extractor_config)
        
        if config_key not in self._model_cache:
            self._model_cache[config_key] = TimeSeriesFeatureExtractor(
                **model_metadata.feature_extractor_config
            )
        
        return self._model_cache[config_key]
    
    def predict_single(self, 
                      data: np.ndarray,
                      model_id: Optional[str] = None,
                      model_type: Optional[MLBaselineType] = None,
                      use_best: bool = True) -> PredictionResult:
        """
        Predict Hurst exponent for a single time series.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        model_id : str, optional
            Specific model ID to use
        model_type : MLBaselineType, optional
            Type of model to use (will select best if multiple available)
        use_best : bool
            Whether to use the best performing model if model_id not specified
            
        Returns:
        --------
        PredictionResult
            Prediction result with metadata
        """
        import time
        start_time = time.time()
        
        # Select model
        if model_id is not None:
            estimator, metadata = self.manager.load_model(model_id)
        elif model_type is not None:
            if use_best:
                estimator, metadata = self.manager.get_best_model(model_type)
            else:
                # Get first available model of this type
                models = self.manager.list_models(model_type=model_type, status=ModelStatus.TRAINED)
                if not models:
                    raise ValueError(f"No trained models of type {model_type.value} found")
                estimator, metadata = self.manager.load_model(models[0].model_id)
        else:
            # Use best overall model
            all_models = self.manager.list_models(status=ModelStatus.TRAINED)
            if not all_models:
                raise ValueError("No trained models available")
            
            # Find best model by validation score
            best_model = max(all_models, key=lambda x: x.performance_metrics.get('validation_score', -np.inf))
            estimator, metadata = self.manager.load_model(best_model.model_id)
        
        # Extract features
        feature_extractor = self._get_feature_extractor(metadata)
        features = feature_extractor.extract_features(data)
        
        # Make prediction
        prediction = estimator.predict(features.combined.reshape(1, -1))
        
        prediction_time = time.time() - start_time
        
        return PredictionResult(
            hurst_estimate=prediction.hurst_estimate,
            confidence_interval=prediction.confidence_interval,
            uncertainty=prediction.prediction_uncertainty,
            model_id=metadata.model_id,
            model_type=metadata.model_type.value,
            prediction_time=prediction_time,
            feature_contributions=prediction.feature_contributions
        )
    
    def predict_batch(self, 
                     data_list: List[np.ndarray],
                     model_id: Optional[str] = None,
                     model_type: Optional[MLBaselineType] = None,
                     use_best: bool = True,
                     show_progress: bool = True) -> List[PredictionResult]:
        """
        Predict Hurst exponents for multiple time series.
        
        Parameters:
        -----------
        data_list : List[np.ndarray]
            List of time series data
        model_id : str, optional
            Specific model ID to use
        model_type : MLBaselineType, optional
            Type of model to use
        use_best : bool
            Whether to use the best performing model
        show_progress : bool
            Whether to show progress during batch processing
            
        Returns:
        --------
        List[PredictionResult]
            List of prediction results
        """
        results = []
        
        for i, data in enumerate(data_list):
            if show_progress and i % 10 == 0:
                print(f"Processing {i+1}/{len(data_list)} time series...")
            
            try:
                result = self.predict_single(data, model_id, model_type, use_best)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Failed to predict for time series {i}: {e}")
                # Create a failed result
                results.append(PredictionResult(
                    hurst_estimate=np.nan,
                    model_id=model_id,
                    model_type=model_type.value if model_type else None
                ))
        
        if show_progress:
            print(f"Completed batch prediction: {len(results)} results")
        
        return results
    
    def predict_ensemble(self, 
                        data: np.ndarray,
                        model_types: Optional[List[MLBaselineType]] = None,
                        weights: Optional[Dict[str, float]] = None,
                        include_uncertainty: bool = True) -> EnsembleResult:
        """
        Predict using ensemble of models.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        model_types : List[MLBaselineType], optional
            Types of models to include in ensemble
        weights : Dict[str, float], optional
            Weights for each model type
        include_uncertainty : bool
            Whether to include uncertainty quantification
            
        Returns:
        --------
        EnsembleResult
            Ensemble prediction result
        """
        if model_types is None:
            model_types = list(MLBaselineType)
        
        # Get available models
        available_models = []
        for model_type in model_types:
            models = self.manager.list_models(model_type=model_type, status=ModelStatus.TRAINED)
            if models:
                available_models.extend(models)
        
        if not available_models:
            raise ValueError("No trained models available for ensemble")
        
        # Make predictions with all models
        predictions = []
        model_weights = {}
        
        for model in available_models:
            try:
                result = self.predict_single(data, model_id=model.model_id)
                predictions.append(result.hurst_estimate)
                
                # Set weight based on model performance
                if weights and model.model_type.value in weights:
                    model_weights[model.model_id] = weights[model.model_type.value]
                else:
                    # Use validation score as weight
                    model_weights[model.model_id] = model.performance_metrics.get('validation_score', 0.0)
                
            except Exception as e:
                warnings.warn(f"Failed to predict with model {model.model_id}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No successful predictions for ensemble")
        
        predictions = np.array(predictions)
        weights_array = np.array([model_weights.get(model.model_id, 1.0) for model in available_models[:len(predictions)]])
        
        # Normalize weights
        if np.sum(weights_array) > 0:
            weights_array = weights_array / np.sum(weights_array)
        else:
            weights_array = np.ones(len(predictions)) / len(predictions)
        
        # Calculate ensemble statistics
        mean_estimate = np.average(predictions, weights=weights_array)
        std_estimate = np.sqrt(np.average((predictions - mean_estimate)**2, weights=weights_array))
        
        # Calculate confidence interval
        confidence_interval = None
        if include_uncertainty and len(predictions) > 1:
            # Use t-distribution for confidence interval
            from scipy import stats
            alpha = 0.05
            n = len(predictions)
            t_val = stats.t.ppf(1 - alpha/2, n-1)
            margin = t_val * std_estimate / np.sqrt(n)
            confidence_interval = (mean_estimate - margin, mean_estimate + margin)
        
        return EnsembleResult(
            mean_estimate=mean_estimate,
            std_estimate=std_estimate,
            individual_predictions=predictions.tolist(),
            model_weights=model_weights,
            confidence_interval=confidence_interval
        )
    
    def compare_models(self, 
                      data: np.ndarray,
                      model_types: Optional[List[MLBaselineType]] = None) -> Dict[str, PredictionResult]:
        """
        Compare predictions from different model types.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        model_types : List[MLBaselineType], optional
            Types of models to compare
            
        Returns:
        --------
        Dict[str, PredictionResult]
            Predictions from each model type
        """
        if model_types is None:
            model_types = list(MLBaselineType)
        
        results = {}
        
        for model_type in model_types:
            try:
                result = self.predict_single(data, model_type=model_type, use_best=True)
                results[model_type.value] = result
            except Exception as e:
                warnings.warn(f"Failed to predict with {model_type.value}: {e}")
                results[model_type.value] = PredictionResult(
                    hurst_estimate=np.nan,
                    model_type=model_type.value
                )
        
        return results
    
    def get_model_info(self, model_id: Optional[str] = None) -> Union[ModelMetadata, List[ModelMetadata]]:
        """
        Get information about available models.
        
        Parameters:
        -----------
        model_id : str, optional
            Specific model ID, or None for all models
            
        Returns:
        --------
        Union[ModelMetadata, List[ModelMetadata]]
            Model metadata
        """
        if model_id is not None:
            if model_id not in self.manager._metadata_registry:
                raise ValueError(f"Model {model_id} not found")
            return self.manager._metadata_registry[model_id]
        else:
            return self.manager.list_models(status=ModelStatus.TRAINED)
    
    def benchmark_models(self, 
                        test_data: List[np.ndarray],
                        true_hurst: List[float],
                        model_types: Optional[List[MLBaselineType]] = None) -> Dict[str, Dict[str, float]]:
        """
        Benchmark model performance on test data.
        
        Parameters:
        -----------
        test_data : List[np.ndarray]
            Test time series data
        true_hurst : List[float]
            True Hurst exponents
        model_types : List[MLBaselineType], optional
            Types of models to benchmark
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Performance metrics for each model type
        """
        if model_types is None:
            model_types = list(MLBaselineType)
        
        results = {}
        
        for model_type in model_types:
            try:
                # Get predictions
                predictions = []
                for data in test_data:
                    result = self.predict_single(data, model_type=model_type, use_best=True)
                    predictions.append(result.hurst_estimate)
                
                predictions = np.array(predictions)
                true_values = np.array(true_hurst)
                
                # Calculate metrics
                mse = np.mean((predictions - true_values) ** 2)
                mae = np.mean(np.abs(predictions - true_values))
                r2 = 1 - mse / np.var(true_values)
                correlation = np.corrcoef(predictions, true_values)[0, 1]
                
                results[model_type.value] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'correlation': float(correlation),
                    'rmse': float(np.sqrt(mse))
                }
                
            except Exception as e:
                warnings.warn(f"Failed to benchmark {model_type.value}: {e}")
                results[model_type.value] = {
                    'mse': np.nan,
                    'mae': np.nan,
                    'r2': np.nan,
                    'correlation': np.nan,
                    'rmse': np.nan
                }
        
        return results


def quick_predict(data: np.ndarray, 
                 models_dir: Union[str, Path] = "pretrained_models",
                 model_type: Optional[MLBaselineType] = None) -> float:
    """
    Quick prediction function for single time series.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    models_dir : str or Path
        Directory containing pretrained models
    model_type : MLBaselineType, optional
        Type of model to use
        
    Returns:
    --------
    float
        Predicted Hurst exponent
    """
    inference = PretrainedInference(models_dir)
    result = inference.predict_single(data, model_type=model_type)
    return result.hurst_estimate


def quick_ensemble_predict(data: np.ndarray,
                          models_dir: Union[str, Path] = "pretrained_models",
                          model_types: Optional[List[MLBaselineType]] = None) -> Tuple[float, float]:
    """
    Quick ensemble prediction function.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    models_dir : str or Path
        Directory containing pretrained models
    model_types : List[MLBaselineType], optional
        Types of models to include in ensemble
        
    Returns:
    --------
    Tuple[float, float]
        (mean_estimate, std_estimate)
    """
    inference = PretrainedInference(models_dir)
    result = inference.predict_ensemble(data, model_types=model_types)
    return result.mean_estimate, result.std_estimate
