"""
Tests for pretrained model system.

This module provides comprehensive tests for the pretrained model system,
including model training, storage, loading, and inference functionality.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Import the pretrained model modules
from neurological_lrd_analysis.ml_baselines import (
    PretrainedModelManager,
    ModelMetadata,
    ModelStatus,
    TrainingConfig,
    PretrainedInference,
    PredictionResult,
    EnsembleResult,
    quick_predict,
    quick_ensemble_predict,
    MLBaselineType,
    create_default_training_configs,
    create_pretrained_suite
)

from neurological_lrd_analysis.benchmark_core.generation import (
    fbm_davies_harte,
    generate_grid
)


class TestPretrainedModelManager:
    """Test pretrained model manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = PretrainedModelManager(self.temp_dir)
        
        # Generate small training dataset
        self.X_train = []
        self.y_train = []
        
        for hurst in [0.3, 0.5, 0.7]:
            for _ in range(5):
                data = fbm_davies_harte(200, hurst, seed=None)
                from neurological_lrd_analysis.ml_baselines import TimeSeriesFeatureExtractor
                extractor = TimeSeriesFeatureExtractor()
                features = extractor.extract_features(data)
                self.X_train.append(features.combined)
                self.y_train.append(hurst)
        
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        
        self.training_info = {
            'n_samples': len(self.X_train),
            'n_features': self.X_train.shape[1],
            'hurst_range': (min(self.y_train), max(self.y_train)),
            'feature_extractor_config': {
                'include_spectral': True,
                'include_wavelet': True,
                'include_fractal': True,
                'include_biomedical': True,
                'sampling_rate': 250.0
            }
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.models_dir.exists()
        assert self.manager.models_path.exists()
        assert self.manager.metadata_path.exists()
        assert self.manager.cache_path.exists()
    
    def test_training_data_creation(self):
        """Test training data creation."""
        X, y, info = self.manager.create_training_data(
            hurst_values=[0.3, 0.5, 0.7],
            lengths=[200],
            n_samples_per_config=3
        )
        
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[0] == y.shape[0]
        assert 'n_samples' in info
        assert 'n_features' in info
    
    def test_model_training(self):
        """Test model training."""
        config = TrainingConfig(
            model_type=MLBaselineType.RANDOM_FOREST,
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            training_data_config={'n_samples_per_config': 5},
            description="Test model"
        )
        
        metadata = self.manager.train_model(config, self.X_train, self.y_train, self.training_info)
        
        assert metadata.model_id is not None
        assert metadata.model_type == MLBaselineType.RANDOM_FOREST
        assert metadata.status == ModelStatus.TRAINED
        assert metadata.performance_metrics['training_score'] > 0
        assert Path(metadata.file_path).exists()
    
    def test_model_loading(self):
        """Test model loading."""
        # Train a model first
        config = TrainingConfig(
            model_type=MLBaselineType.RANDOM_FOREST,
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            training_data_config={'n_samples_per_config': 5}
        )
        
        metadata = self.manager.train_model(config, self.X_train, self.y_train, self.training_info)
        
        # Load the model
        estimator, loaded_metadata = self.manager.load_model(metadata.model_id)
        
        assert estimator is not None
        assert loaded_metadata.model_id == metadata.model_id
        assert estimator.is_trained
    
    def test_model_listing(self):
        """Test model listing functionality."""
        # Train multiple models
        configs = [
            TrainingConfig(
                model_type=MLBaselineType.RANDOM_FOREST,
                hyperparameters={'n_estimators': 10, 'random_state': 42},
                training_data_config={'n_samples_per_config': 5}
            ),
            TrainingConfig(
                model_type=MLBaselineType.SVR,
                hyperparameters={'C': 1.0, 'epsilon': 0.1},
                training_data_config={'n_samples_per_config': 5}
            )
        ]
        
        for config in configs:
            self.manager.train_model(config, self.X_train, self.y_train, self.training_info)
        
        # List all models
        all_models = self.manager.list_models()
        assert len(all_models) == 2
        
        # List by type
        rf_models = self.manager.list_models(model_type=MLBaselineType.RANDOM_FOREST)
        assert len(rf_models) == 1
        assert rf_models[0].model_type == MLBaselineType.RANDOM_FOREST
        
        # List by status
        trained_models = self.manager.list_models(status=ModelStatus.TRAINED)
        assert len(trained_models) == 2
    
    def test_best_model_selection(self):
        """Test best model selection."""
        # Train multiple models of same type
        configs = [
            TrainingConfig(
                model_type=MLBaselineType.RANDOM_FOREST,
                hyperparameters={'n_estimators': 10, 'random_state': 42},
                training_data_config={'n_samples_per_config': 5}
            ),
            TrainingConfig(
                model_type=MLBaselineType.RANDOM_FOREST,
                hyperparameters={'n_estimators': 20, 'random_state': 42},
                training_data_config={'n_samples_per_config': 5}
            )
        ]
        
        for config in configs:
            self.manager.train_model(config, self.X_train, self.y_train, self.training_info)
        
        # Get best model
        estimator, metadata = self.manager.get_best_model(MLBaselineType.RANDOM_FOREST)
        assert estimator is not None
        assert metadata.model_type == MLBaselineType.RANDOM_FOREST
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Train a model
        config = TrainingConfig(
            model_type=MLBaselineType.RANDOM_FOREST,
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            training_data_config={'n_samples_per_config': 5}
        )
        
        metadata = self.manager.train_model(config, self.X_train, self.y_train, self.training_info)
        
        # Test prediction
        test_data = fbm_davies_harte(200, 0.6, seed=42)
        prediction = self.manager.predict(metadata.model_id, test_data)
        
        assert isinstance(prediction, float)
        assert 0 < prediction < 1
    
    def test_model_suite_creation(self):
        """Test model suite creation."""
        configs = create_default_training_configs()[:2]  # Use first 2 configs
        
        results = self.manager.create_model_suite(configs, self.X_train, self.y_train, self.training_info)
        
        assert len(results) > 0
        for result in results:
            assert result.status == ModelStatus.TRAINED


class TestPretrainedInference:
    """Test pretrained inference functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = PretrainedModelManager(self.temp_dir)
        
        # Generate training data
        self.X_train = []
        self.y_train = []
        
        for hurst in [0.3, 0.5, 0.7]:
            for _ in range(5):
                data = fbm_davies_harte(200, hurst, seed=None)
                from neurological_lrd_analysis.ml_baselines import TimeSeriesFeatureExtractor
                extractor = TimeSeriesFeatureExtractor()
                features = extractor.extract_features(data)
                self.X_train.append(features.combined)
                self.y_train.append(hurst)
        
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        
        self.training_info = {
            'n_samples': len(self.X_train),
            'n_features': self.X_train.shape[1],
            'hurst_range': (min(self.y_train), max(self.y_train)),
            'feature_extractor_config': {
                'include_spectral': True,
                'include_wavelet': True,
                'include_fractal': True,
                'include_biomedical': True,
                'sampling_rate': 250.0
            }
        }
        
        # Train a model
        config = TrainingConfig(
            model_type=MLBaselineType.RANDOM_FOREST,
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            training_data_config={'n_samples_per_config': 5}
        )
        
        self.manager.train_model(config, self.X_train, self.y_train, self.training_info)
        
        # Create inference system
        self.inference = PretrainedInference(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_inference_initialization(self):
        """Test inference system initialization."""
        assert self.inference.manager is not None
        assert self.inference._model_cache is not None
    
    def test_single_prediction(self):
        """Test single prediction."""
        test_data = fbm_davies_harte(200, 0.6, seed=42)
        
        result = self.inference.predict_single(test_data)
        
        assert isinstance(result, PredictionResult)
        assert 0 < result.hurst_estimate < 1
        assert result.model_id is not None
        assert result.model_type is not None
        assert result.prediction_time is not None
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        test_data_list = [
            fbm_davies_harte(200, 0.4, seed=42),
            fbm_davies_harte(200, 0.6, seed=43),
            fbm_davies_harte(200, 0.8, seed=44)
        ]
        
        results = self.inference.predict_batch(test_data_list, show_progress=False)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, PredictionResult)
            assert 0 < result.hurst_estimate < 1
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        test_data = fbm_davies_harte(200, 0.6, seed=42)
        
        result = self.inference.predict_ensemble(test_data)
        
        assert isinstance(result, EnsembleResult)
        assert 0 < result.mean_estimate < 1
        assert result.std_estimate >= 0
        assert len(result.individual_predictions) > 0
    
    def test_model_comparison(self):
        """Test model comparison."""
        test_data = fbm_davies_harte(200, 0.6, seed=42)
        
        results = self.inference.compare_models(test_data)
        
        assert len(results) > 0
        for model_type, result in results.items():
            assert isinstance(result, PredictionResult)
    
    def test_model_info(self):
        """Test model information retrieval."""
        # Get all models
        all_models = self.inference.get_model_info()
        assert len(all_models) > 0
        
        # Get specific model
        model_id = all_models[0].model_id
        model_info = self.inference.get_model_info(model_id)
        assert model_info.model_id == model_id
    
    def test_benchmark_models(self):
        """Test model benchmarking."""
        test_data = [
            fbm_davies_harte(200, 0.4, seed=42),
            fbm_davies_harte(200, 0.6, seed=43),
            fbm_davies_harte(200, 0.8, seed=44)
        ]
        true_hurst = [0.4, 0.6, 0.8]
        
        results = self.inference.benchmark_models(test_data, true_hurst)
        
        assert len(results) > 0
        for model_type, metrics in results.items():
            assert 'mse' in metrics
            assert 'mae' in metrics
            assert 'r2' in metrics
            assert 'correlation' in metrics


class TestQuickFunctions:
    """Test quick prediction functions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = PretrainedModelManager(self.temp_dir)
        
        # Generate training data
        self.X_train = []
        self.y_train = []
        
        for hurst in [0.3, 0.5, 0.7]:
            for _ in range(5):
                data = fbm_davies_harte(200, hurst, seed=None)
                from neurological_lrd_analysis.ml_baselines import TimeSeriesFeatureExtractor
                extractor = TimeSeriesFeatureExtractor()
                features = extractor.extract_features(data)
                self.X_train.append(features.combined)
                self.y_train.append(hurst)
        
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        
        self.training_info = {
            'n_samples': len(self.X_train),
            'n_features': self.X_train.shape[1],
            'hurst_range': (min(self.y_train), max(self.y_train)),
            'feature_extractor_config': {
                'include_spectral': True,
                'include_wavelet': True,
                'include_fractal': True,
                'include_biomedical': True,
                'sampling_rate': 250.0
            }
        }
        
        # Train a model
        config = TrainingConfig(
            model_type=MLBaselineType.RANDOM_FOREST,
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            training_data_config={'n_samples_per_config': 5}
        )
        
        self.manager.train_model(config, self.X_train, self.y_train, self.training_info)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_quick_predict(self):
        """Test quick prediction function."""
        test_data = fbm_davies_harte(200, 0.6, seed=42)
        
        prediction = quick_predict(test_data, self.temp_dir)
        
        assert isinstance(prediction, float)
        assert 0 < prediction < 1
    
    def test_quick_ensemble_predict(self):
        """Test quick ensemble prediction function."""
        test_data = fbm_davies_harte(200, 0.6, seed=42)
        
        mean_est, std_est = quick_ensemble_predict(test_data, self.temp_dir)
        
        assert isinstance(mean_est, float)
        assert isinstance(std_est, float)
        assert 0 < mean_est < 1
        assert std_est >= 0


class TestPretrainedSuite:
    """Test pretrained suite creation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_pretrained_suite(self):
        """Test creating a complete pretrained suite."""
        # This test might take a while, so we'll use a smaller configuration
        # Create a custom manager with simpler configuration
        manager = PretrainedModelManager(self.temp_dir)
        
        # Create training data with simpler configuration
        X, y, training_info = manager.create_training_data(
            hurst_values=[0.3, 0.5, 0.7],
            lengths=[200],
            n_samples_per_config=5,
            generators=['fbm', 'fgn'],  # Avoid problematic generators
            contaminations=['none', 'noise'],
            biomedical_scenarios=['eeg']
        )
        
        # Create simple training configs
        configs = [
            TrainingConfig(
                model_type=MLBaselineType.RANDOM_FOREST,
                hyperparameters={'n_estimators': 10, 'random_state': 42},
                training_data_config={'n_samples_per_config': 5},
                description="Test RF model"
            )
        ]
        
        # Train models
        results = manager.create_model_suite(configs, X, y, training_info)
        
        # Check that models were created
        models = manager.list_models(status=ModelStatus.TRAINED)
        assert len(models) > 0
        
        # Test that we can make predictions
        test_data = fbm_davies_harte(200, 0.6, seed=42)
        inference = PretrainedInference(self.temp_dir)
        
        result = inference.predict_single(test_data)
        assert 0 < result.hurst_estimate < 1


class TestErrorHandling:
    """Test error handling in pretrained model system."""
    
    def test_nonexistent_model(self):
        """Test handling of nonexistent models."""
        temp_dir = tempfile.mkdtemp()
        manager = PretrainedModelManager(temp_dir)
        
        with pytest.raises(ValueError, match="Model nonexistent not found"):
            manager.load_model("nonexistent")
        
        shutil.rmtree(temp_dir)
    
    def test_failed_model_training(self):
        """Test handling of failed model training."""
        temp_dir = tempfile.mkdtemp()
        manager = PretrainedModelManager(temp_dir)
        
        # Create invalid training data
        X_invalid = np.array([[1, 2, 3]])  # Too few samples
        y_invalid = np.array([0.5])
        
        training_info = {
            'n_samples': 1,
            'n_features': 3,
            'hurst_range': (0.5, 0.5),
            'feature_extractor_config': {
                'include_spectral': True,
                'include_wavelet': True,
                'include_fractal': True,
                'include_biomedical': True,
                'sampling_rate': 250.0
            }
        }
        
        config = TrainingConfig(
            model_type=MLBaselineType.RANDOM_FOREST,
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            training_data_config={'n_samples_per_config': 5}
        )
        
        with pytest.raises(Exception):
            manager.train_model(config, X_invalid, y_invalid, training_info)
        
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
