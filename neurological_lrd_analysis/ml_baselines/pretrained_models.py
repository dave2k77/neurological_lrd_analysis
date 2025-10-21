"""
Pretrained Model System for ML Baseline Estimators.

This module provides a comprehensive system for creating, storing, and loading
pretrained ML models for Hurst exponent estimation. It includes model training
pipelines, efficient storage, metadata management, and inference systems.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
import joblib
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
import shutil
from enum import Enum

# Import ML components
from .ml_estimators import (
    MLBaselineType, RandomForestEstimator, SVREstimator, GradientBoostingEstimator,
    MLBaselineFactory, BaseMLEstimator, MLTrainingResult
)
from .feature_extraction import TimeSeriesFeatureExtractor
from .hyperparameter_optimization import optimize_all_estimators


class ModelStatus(Enum):
    """Status of pretrained models."""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata for pretrained models."""
    model_id: str
    model_type: MLBaselineType
    version: str
    created_at: datetime
    training_data_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_extractor_config: Dict[str, Any]
    status: ModelStatus
    file_path: str
    checksum: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    license: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: MLBaselineType
    hyperparameters: Dict[str, Any]
    training_data_config: Dict[str, Any]
    validation_split: float = 0.2
    random_state: int = 42
    optimize_hyperparameters: bool = False
    optimization_trials: int = 50
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class PretrainedModelManager:
    """
    Manager for pretrained ML models.
    
    Handles creation, storage, loading, and management of pretrained models
    for Hurst exponent estimation.
    """
    
    def __init__(self, models_dir: Union[str, Path] = "pretrained_models"):
        """
        Initialize the pretrained model manager.
        
        Parameters:
        -----------
        models_dir : str or Path
            Directory to store pretrained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_path = self.models_dir / "models"
        self.metadata_path = self.models_dir / "metadata"
        self.cache_path = self.models_dir / "cache"
        
        for path in [self.models_path, self.metadata_path, self.cache_path]:
            path.mkdir(exist_ok=True)
        
        # Load existing metadata
        self._metadata_registry = self._load_metadata_registry()
    
    def _load_metadata_registry(self) -> Dict[str, ModelMetadata]:
        """Load existing metadata registry."""
        registry = {}
        
        for metadata_file in self.metadata_path.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    # Convert datetime strings back to datetime objects
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['model_type'] = MLBaselineType(data['model_type'])
                    data['status'] = ModelStatus(data['status'])
                    registry[data['model_id']] = ModelMetadata(**data)
            except Exception as e:
                warnings.warn(f"Failed to load metadata from {metadata_file}: {e}")
        
        return registry
    
    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata."""
        metadata_file = self.metadata_path / f"{metadata.model_id}.json"
        
        # Convert to dict and handle datetime serialization
        data = asdict(metadata)
        data['created_at'] = metadata.created_at.isoformat()
        data['model_type'] = metadata.model_type.value
        data['status'] = metadata.status.value
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_model_id(self, model_type: MLBaselineType, 
                          training_config: Dict[str, Any]) -> str:
        """Generate unique model ID."""
        # Create hash from model type and key config parameters
        config_str = f"{model_type.value}_{training_config.get('random_state', 42)}"
        if 'hyperparameters' in training_config:
            config_str += f"_{str(sorted(training_config['hyperparameters'].items()))}"
        
        hash_obj = hashlib.md5(config_str.encode())
        return f"{model_type.value}_{hash_obj.hexdigest()[:8]}"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def create_training_data(self, 
                           hurst_values: List[float] = None,
                           lengths: List[int] = None,
                           n_samples_per_config: int = 100,
                           generators: List[str] = None,
                           contaminations: List[str] = None,
                           biomedical_scenarios: List[str] = None,
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Create comprehensive training dataset.
        
        Parameters:
        -----------
        hurst_values : List[float], optional
            Hurst values to generate
        lengths : List[int], optional
            Time series lengths
        n_samples_per_config : int
            Number of samples per configuration
        generators : List[str], optional
            Data generators to use
        contaminations : List[str], optional
            Contamination types
        biomedical_scenarios : List[str], optional
            Biomedical scenarios
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
            (X, y, training_info) - features, targets, and metadata
        """
        if hurst_values is None:
            hurst_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        if lengths is None:
            lengths = [500, 1000, 2000]
        
        if generators is None:
            generators = ['fbm', 'fgn', 'arfima', 'mrw', 'fou']
        
        if contaminations is None:
            contaminations = ['none', 'noise', 'missing', 'artifacts']
        
        if biomedical_scenarios is None:
            biomedical_scenarios = ['eeg', 'ecg', 'respiratory']
        
        print(f"Creating comprehensive training dataset...")
        print(f"  - Hurst values: {hurst_values}")
        print(f"  - Lengths: {lengths}")
        print(f"  - Generators: {generators}")
        print(f"  - Contaminations: {contaminations}")
        print(f"  - Biomedical scenarios: {biomedical_scenarios}")
        print(f"  - Samples per config: {n_samples_per_config}")
        
        # Generate synthetic data
        from ..benchmark_core.generation import generate_grid
        
        samples = generate_grid(
            hurst_values=hurst_values,
            lengths=lengths,
            contaminations=contaminations,
            generators=generators,
            biomedical_scenarios=biomedical_scenarios
        )
        
        # Extract features
        extractor = TimeSeriesFeatureExtractor()
        X = []
        y = []
        
        print("Extracting features...")
        for i, sample in enumerate(samples):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(samples)} samples")
            
            features = extractor.extract_features(sample.data, sample.true_hurst)
            X.append(features.combined)
            y.append(sample.true_hurst)
        
        X = np.array(X)
        y = np.array(y)
        
        # Create training info
        training_info = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'hurst_range': (min(y), max(y)),
            'hurst_values': hurst_values,
            'lengths': lengths,
            'generators': generators,
            'contaminations': contaminations,
            'biomedical_scenarios': biomedical_scenarios,
            'n_samples_per_config': n_samples_per_config,
            'random_state': random_state,
            'feature_extractor_config': {
                'include_spectral': extractor.include_spectral,
                'include_wavelet': extractor.include_wavelet,
                'include_fractal': extractor.include_fractal,
                'include_biomedical': extractor.include_biomedical,
                'sampling_rate': extractor.sampling_rate
            }
        }
        
        print(f"Generated {len(X)} samples with {X.shape[1]} features")
        return X, y, training_info
    
    def train_model(self, 
                   training_config: TrainingConfig,
                   X: np.ndarray,
                   y: np.ndarray,
                   training_info: Dict[str, Any]) -> ModelMetadata:
        """
        Train a model and save it as pretrained.
        
        Parameters:
        -----------
        training_config : TrainingConfig
            Training configuration
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets
        training_info : Dict[str, Any]
            Training dataset information
            
        Returns:
        --------
        ModelMetadata
            Metadata for the trained model
        """
        print(f"Training {training_config.model_type.value} model...")
        
        # Generate model ID
        model_id = self._generate_model_id(training_config.model_type, asdict(training_config))
        
        # Check if model already exists
        if model_id in self._metadata_registry:
            print(f"Model {model_id} already exists, skipping training")
            return self._metadata_registry[model_id]
        
        try:
            # Create estimator
            estimator = MLBaselineFactory.create_estimator(
                training_config.model_type,
                **training_config.hyperparameters
            )
            
            # Optimize hyperparameters if requested
            if training_config.optimize_hyperparameters:
                print("Optimizing hyperparameters...")
                opt_results = optimize_all_estimators(
                    X, y,
                    estimator_types=[training_config.model_type.value],
                    n_trials=training_config.optimization_trials,
                    random_state=training_config.random_state
                )
                
                if training_config.model_type.value in opt_results:
                    best_params = opt_results[training_config.model_type.value].best_params
                    print(f"Best parameters: {best_params}")
                    # Update estimator with best parameters
                    estimator = MLBaselineFactory.create_estimator(
                        training_config.model_type,
                        **best_params
                    )
            
            # Train model
            print("Training model...")
            training_result = estimator.train(
                X, y,
                validation_split=training_config.validation_split,
                random_state=training_config.random_state
            )
            
            # Save model
            model_path = self.models_path / f"{model_id}.joblib"
            estimator.save_model(model_path)
            
            # Calculate checksum
            checksum = self._calculate_checksum(model_path)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=training_config.model_type,
                version="1.0.0",
                created_at=datetime.now(),
                training_data_info=training_info,
                performance_metrics={
                    'training_score': training_result.training_score,
                    'validation_score': training_result.validation_score,
                    'cv_mean': np.mean(training_result.cross_val_scores),
                    'cv_std': np.std(training_result.cross_val_scores)
                },
                hyperparameters=training_config.hyperparameters,
                feature_extractor_config=training_info['feature_extractor_config'],
                status=ModelStatus.TRAINED,
                file_path=str(model_path),
                checksum=checksum,
                description=training_config.description,
                tags=training_config.tags,
                author="Davian R. Chin",
                license="MIT"
            )
            
            # Save metadata
            self._save_metadata(metadata)
            self._metadata_registry[model_id] = metadata
            
            print(f"Model {model_id} trained and saved successfully")
            print(f"  Training score: {training_result.training_score:.4f}")
            print(f"  Validation score: {training_result.validation_score:.4f}")
            print(f"  CV score: {np.mean(training_result.cross_val_scores):.4f} ± {np.std(training_result.cross_val_scores):.4f}")
            
            return metadata
            
        except Exception as e:
            print(f"Failed to train model {model_id}: {e}")
            # Create failed metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=training_config.model_type,
                version="1.0.0",
                created_at=datetime.now(),
                training_data_info=training_info,
                performance_metrics={},
                hyperparameters=training_config.hyperparameters,
                feature_extractor_config=training_info['feature_extractor_config'],
                status=ModelStatus.FAILED,
                file_path="",
                checksum="",
                description=f"Failed: {str(e)}"
            )
            self._save_metadata(metadata)
            self._metadata_registry[model_id] = metadata
            raise
    
    def load_model(self, model_id: str) -> Tuple[BaseMLEstimator, ModelMetadata]:
        """
        Load a pretrained model.
        
        Parameters:
        -----------
        model_id : str
            ID of the model to load
            
        Returns:
        --------
        Tuple[BaseMLEstimator, ModelMetadata]
            Loaded model and its metadata
        """
        if model_id not in self._metadata_registry:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self._metadata_registry[model_id]
        
        if metadata.status == ModelStatus.FAILED:
            raise ValueError(f"Model {model_id} failed during training")
        
        # Load model
        model_path = Path(metadata.file_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        # Verify checksum
        current_checksum = self._calculate_checksum(model_path)
        if current_checksum != metadata.checksum:
            warnings.warn(f"Checksum mismatch for model {model_id}")
        
        # Create estimator and load model
        estimator = MLBaselineFactory.create_estimator(metadata.model_type)
        estimator.load_model(model_path)
        
        return estimator, metadata
    
    def list_models(self, 
                   model_type: Optional[MLBaselineType] = None,
                   status: Optional[ModelStatus] = None,
                   tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """
        List available models with optional filtering.
        
        Parameters:
        -----------
        model_type : MLBaselineType, optional
            Filter by model type
        status : ModelStatus, optional
            Filter by status
        tags : List[str], optional
            Filter by tags
            
        Returns:
        --------
        List[ModelMetadata]
            List of matching models
        """
        models = list(self._metadata_registry.values())
        
        if model_type is not None:
            models = [m for m in models if m.model_type == model_type]
        
        if status is not None:
            models = [m for m in models if m.status == status]
        
        if tags is not None:
            models = [m for m in models if m.tags and any(tag in m.tags for tag in tags)]
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def get_best_model(self, 
                      model_type: MLBaselineType,
                      metric: str = 'validation_score') -> Tuple[BaseMLEstimator, ModelMetadata]:
        """
        Get the best performing model of a given type.
        
        Parameters:
        -----------
        model_type : MLBaselineType
            Type of model to get
        metric : str
            Metric to use for ranking
            
        Returns:
        --------
        Tuple[BaseMLEstimator, ModelMetadata]
            Best model and its metadata
        """
        models = self.list_models(model_type=model_type, status=ModelStatus.TRAINED)
        
        if not models:
            raise ValueError(f"No trained models of type {model_type.value} found")
        
        # Find best model by metric
        best_model = max(models, key=lambda x: x.performance_metrics.get(metric, -np.inf))
        
        return self.load_model(best_model.model_id)
    
    def predict(self, 
               model_id: str,
               data: np.ndarray,
               return_metadata: bool = False) -> Union[float, Tuple[float, ModelMetadata]]:
        """
        Make prediction using a pretrained model.
        
        Parameters:
        -----------
        model_id : str
            ID of the model to use
        data : np.ndarray
            Time series data
        return_metadata : bool
            Whether to return model metadata
            
        Returns:
        --------
        Union[float, Tuple[float, ModelMetadata]]
            Prediction result and optionally metadata
        """
        estimator, metadata = self.load_model(model_id)
        
        # Extract features using the same configuration as training
        extractor = TimeSeriesFeatureExtractor(**metadata.feature_extractor_config)
        features = extractor.extract_features(data)
        
        # Make prediction
        prediction = estimator.predict(features.combined.reshape(1, -1))
        
        if return_metadata:
            return prediction.hurst_estimate, metadata
        else:
            return prediction.hurst_estimate
    
    def create_model_suite(self, 
                          training_configs: List[TrainingConfig],
                          X: np.ndarray,
                          y: np.ndarray,
                          training_info: Dict[str, Any]) -> List[ModelMetadata]:
        """
        Create a suite of pretrained models.
        
        Parameters:
        -----------
        training_configs : List[TrainingConfig]
            List of training configurations
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets
        training_info : Dict[str, Any]
            Training dataset information
            
        Returns:
        --------
        List[ModelMetadata]
            Metadata for all trained models
        """
        print(f"Creating model suite with {len(training_configs)} models...")
        
        results = []
        for i, config in enumerate(training_configs):
            print(f"\nTraining model {i+1}/{len(training_configs)}: {config.model_type.value}")
            try:
                metadata = self.train_model(config, X, y, training_info)
                results.append(metadata)
            except Exception as e:
                print(f"Failed to train {config.model_type.value}: {e}")
                continue
        
        print(f"\nSuccessfully trained {len(results)} models")
        return results
    
    def cleanup_models(self, 
                      keep_best: bool = True,
                      max_models_per_type: int = 5) -> None:
        """
        Clean up old or redundant models.
        
        Parameters:
        -----------
        keep_best : bool
            Whether to keep the best performing model of each type
        max_models_per_type : int
            Maximum number of models to keep per type
        """
        print("Cleaning up models...")
        
        for model_type in MLBaselineType:
            models = self.list_models(model_type=model_type, status=ModelStatus.TRAINED)
            
            if len(models) <= max_models_per_type:
                continue
            
            # Sort by performance
            models.sort(key=lambda x: x.performance_metrics.get('validation_score', -np.inf), reverse=True)
            
            # Keep best models
            models_to_keep = models[:max_models_per_type]
            models_to_remove = models[max_models_per_type:]
            
            for model in models_to_remove:
                print(f"Removing model {model.model_id}")
                
                # Remove model file
                model_path = Path(model.file_path)
                if model_path.exists():
                    model_path.unlink()
                
                # Remove metadata
                metadata_file = self.metadata_path / f"{model.model_id}.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                
                # Update registry
                if model.model_id in self._metadata_registry:
                    del self._metadata_registry[model.model_id]
        
        print("Cleanup completed")


def create_default_training_configs() -> List[TrainingConfig]:
    """Create default training configurations for all model types."""
    configs = []
    
    # Random Forest configurations
    configs.append(TrainingConfig(
        model_type=MLBaselineType.RANDOM_FOREST,
        hyperparameters={'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
        training_data_config={'n_samples_per_config': 50},
        description="Random Forest with moderate complexity",
        tags=['default', 'random_forest']
    ))
    
    configs.append(TrainingConfig(
        model_type=MLBaselineType.RANDOM_FOREST,
        hyperparameters={'n_estimators': 200, 'max_depth': 15, 'random_state': 42},
        training_data_config={'n_samples_per_config': 100},
        optimize_hyperparameters=True,
        optimization_trials=30,
        description="Random Forest with hyperparameter optimization",
        tags=['optimized', 'random_forest']
    ))
    
    # SVR configurations
    configs.append(TrainingConfig(
        model_type=MLBaselineType.SVR,
        hyperparameters={'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'},
        training_data_config={'n_samples_per_config': 50},
        description="SVR with RBF kernel",
        tags=['default', 'svr']
    ))
    
    configs.append(TrainingConfig(
        model_type=MLBaselineType.SVR,
        hyperparameters={'C': 10.0, 'epsilon': 0.01, 'kernel': 'rbf'},
        training_data_config={'n_samples_per_config': 100},
        optimize_hyperparameters=True,
        optimization_trials=30,
        description="SVR with hyperparameter optimization",
        tags=['optimized', 'svr']
    ))
    
    # Gradient Boosting configurations
    configs.append(TrainingConfig(
        model_type=MLBaselineType.GRADIENT_BOOSTING,
        hyperparameters={'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42},
        training_data_config={'n_samples_per_config': 50},
        description="Gradient Boosting with moderate complexity",
        tags=['default', 'gradient_boosting']
    ))
    
    configs.append(TrainingConfig(
        model_type=MLBaselineType.GRADIENT_BOOSTING,
        hyperparameters={'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': 42},
        training_data_config={'n_samples_per_config': 100},
        optimize_hyperparameters=True,
        optimization_trials=30,
        description="Gradient Boosting with hyperparameter optimization",
        tags=['optimized', 'gradient_boosting']
    ))
    
    return configs


def create_pretrained_suite(models_dir: Union[str, Path] = "pretrained_models",
                           force_retrain: bool = False) -> PretrainedModelManager:
    """
    Create a complete suite of pretrained models.
    
    Parameters:
    -----------
    models_dir : str or Path
        Directory to store models
    force_retrain : bool
        Whether to retrain existing models
        
    Returns:
    --------
    PretrainedModelManager
        Manager with trained models
    """
    manager = PretrainedModelManager(models_dir)
    
    # Check if models already exist
    existing_models = manager.list_models(status=ModelStatus.TRAINED)
    if existing_models and not force_retrain:
        print(f"Found {len(existing_models)} existing models")
        return manager
    
    print("Creating comprehensive training dataset...")
    
    # Create training data
    X, y, training_info = manager.create_training_data(
        hurst_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lengths=[500, 1000, 2000],
        n_samples_per_config=50,
        generators=['fbm', 'fgn', 'arfima', 'mrw', 'fou'],
        contaminations=['none', 'noise', 'missing', 'artifacts'],
        biomedical_scenarios=['eeg', 'ecg', 'respiratory']
    )
    
    # Create training configurations
    training_configs = create_default_training_configs()
    
    # Train all models
    results = manager.create_model_suite(training_configs, X, y, training_info)
    
    print(f"\nSuccessfully created {len(results)} pretrained models")
    return manager
