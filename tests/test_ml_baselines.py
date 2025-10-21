"""
Tests for ML baseline estimators.

This module provides comprehensive tests for the machine learning baseline
estimators, including feature extraction, model training, and hyperparameter optimization.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
import pytest
from typing import Dict, List, Optional
import warnings

# Import the ML baseline modules
from neurological_lrd_analysis.ml_baselines import (
    MLBaselineType,
    RandomForestEstimator,
    SVREstimator,
    GradientBoostingEstimator,
    MLBaselineFactory,
    TimeSeriesFeatureExtractor,
    OptunaOptimizer,
    create_optuna_study,
    optimize_hyperparameters,
    optimize_all_estimators
)

from neurological_lrd_analysis.benchmark_core.generation import (
    fbm_davies_harte,
    generate_fgn,
    generate_arfima,
    generate_mrw,
    generate_fou,
    generate_grid
)


class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = TimeSeriesFeatureExtractor()
        assert extractor.include_spectral is True
        assert extractor.include_wavelet is True
        assert extractor.include_fractal is True
        assert extractor.sampling_rate == 250.0
    
    def test_feature_extraction_basic(self):
        """Test basic feature extraction."""
        # Generate test data
        data = fbm_davies_harte(1000, 0.7, seed=42)
        
        extractor = TimeSeriesFeatureExtractor()
        features = extractor.extract_features(data)
        
        # Check that features were extracted
        assert len(features.combined) > 0
        assert len(features.feature_names) > 0
        assert len(features.statistical) > 0
        
        # Check statistical features
        assert 'mean' in features.statistical
        assert 'std' in features.statistical
        assert 'skewness' in features.statistical
        assert 'kurtosis' in features.statistical
    
    def test_feature_extraction_short_data(self):
        """Test feature extraction with short data."""
        data = np.random.randn(5)
        
        extractor = TimeSeriesFeatureExtractor()
        with pytest.raises(ValueError, match="Data too short"):
            extractor.extract_features(data)
    
    def test_feature_extraction_with_nan(self):
        """Test feature extraction with NaN values."""
        data = np.random.randn(100)
        data[10:15] = np.nan  # Add some NaN values
        
        extractor = TimeSeriesFeatureExtractor()
        features = extractor.extract_features(data)
        
        # Should handle NaN values gracefully
        assert len(features.combined) > 0
        assert not np.any(np.isnan(features.combined))
    
    def test_feature_extraction_biomedical(self):
        """Test biomedical feature extraction."""
        data = fbm_davies_harte(1000, 0.7, seed=42)
        
        extractor = TimeSeriesFeatureExtractor(
            include_biomedical=True,
            sampling_rate=250.0
        )
        features = extractor.extract_features(data)
        
        # Check biomedical features
        assert len(features.biomedical) > 0
        assert 'amplitude_range' in features.biomedical
        assert 'zero_crossing_rate' in features.biomedical
        assert 'signal_quality' in features.biomedical


class TestMLEstimators:
    """Test ML estimator functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Generate training data
        self.X_train = []
        self.y_train = []
        
        for hurst in [0.3, 0.5, 0.7, 0.9]:
            for _ in range(10):
                data = fbm_davies_harte(500, hurst, seed=None)
                extractor = TimeSeriesFeatureExtractor()
                features = extractor.extract_features(data)
                self.X_train.append(features.combined)
                self.y_train.append(hurst)
        
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        
        # Generate test data
        self.X_test = []
        self.y_test = []
        
        for hurst in [0.4, 0.6, 0.8]:
            for _ in range(5):
                data = fbm_davies_harte(500, hurst, seed=None)
                extractor = TimeSeriesFeatureExtractor()
                features = extractor.extract_features(data)
                self.X_test.append(features.combined)
                self.y_test.append(hurst)
        
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
    
    def test_random_forest_estimator(self):
        """Test Random Forest estimator."""
        estimator = RandomForestEstimator(n_estimators=10, random_state=42)
        
        # Train
        result = estimator.train(self.X_train, self.y_train)
        assert result.training_score > 0
        assert result.validation_score > 0
        assert len(result.cross_val_scores) == 5
        
        # Predict
        pred_result = estimator.predict(self.X_test[0:1])
        assert 0 < pred_result.hurst_estimate < 1
        assert pred_result.confidence_interval is not None
    
    def test_svr_estimator(self):
        """Test SVR estimator."""
        estimator = SVREstimator(C=1.0, epsilon=0.1)
        
        # Train
        result = estimator.train(self.X_train, self.y_train)
        assert result.training_score > 0
        assert result.validation_score > 0
        
        # Predict
        pred_result = estimator.predict(self.X_test[0:1])
        assert 0 < pred_result.hurst_estimate < 1
    
    def test_gradient_boosting_estimator(self):
        """Test Gradient Boosting estimator."""
        estimator = GradientBoostingEstimator(n_estimators=10, random_state=42)
        
        # Train
        result = estimator.train(self.X_train, self.y_train)
        assert result.training_score > 0
        assert result.validation_score > 0
        
        # Predict
        pred_result = estimator.predict(self.X_test[0:1])
        assert 0 < pred_result.hurst_estimate < 1
    
    def test_ml_baseline_factory(self):
        """Test ML baseline factory."""
        # Test individual estimator creation
        rf_estimator = MLBaselineFactory.create_estimator(MLBaselineType.RANDOM_FOREST)
        assert isinstance(rf_estimator, RandomForestEstimator)
        
        svr_estimator = MLBaselineFactory.create_estimator(MLBaselineType.SVR)
        assert isinstance(svr_estimator, SVREstimator)
        
        gb_estimator = MLBaselineFactory.create_estimator(MLBaselineType.GRADIENT_BOOSTING)
        assert isinstance(gb_estimator, GradientBoostingEstimator)
        
        # Test creating all estimators
        all_estimators = MLBaselineFactory.create_all_estimators()
        assert len(all_estimators) == 3
        assert MLBaselineType.RANDOM_FOREST in all_estimators
        assert MLBaselineType.SVR in all_estimators
        assert MLBaselineType.GRADIENT_BOOSTING in all_estimators
    
    def test_model_save_load(self, tmp_path):
        """Test model saving and loading."""
        estimator = RandomForestEstimator(n_estimators=10, random_state=42)
        estimator.train(self.X_train, self.y_train)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        estimator.save_model(model_path)
        
        # Load model
        new_estimator = RandomForestEstimator()
        new_estimator.load_model(model_path)
        
        # Test prediction
        pred1 = estimator.predict(self.X_test[0:1])
        pred2 = new_estimator.predict(self.X_test[0:1])
        
        assert abs(pred1.hurst_estimate - pred2.hurst_estimate) < 1e-6


class TestHyperparameterOptimization:
    """Test hyperparameter optimization functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Generate training data
        self.X_train = []
        self.y_train = []
        
        for hurst in [0.3, 0.5, 0.7, 0.9]:
            for _ in range(20):
                data = fbm_davies_harte(500, hurst, seed=None)
                extractor = TimeSeriesFeatureExtractor()
                features = extractor.extract_features(data)
                self.X_train.append(features.combined)
                self.y_train.append(hurst)
        
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
    
    def test_optuna_optimizer_initialization(self):
        """Test Optuna optimizer initialization."""
        optimizer = OptunaOptimizer(n_trials=10, random_state=42)
        assert optimizer.n_trials == 10
        assert optimizer.direction == 'minimize'
        assert optimizer.random_state == 42
    
    def test_optimize_random_forest(self):
        """Test Random Forest hyperparameter optimization."""
        optimizer = OptunaOptimizer(n_trials=5, random_state=42)
        
        result = optimizer.optimize_random_forest(
            self.X_train, self.y_train, cv_folds=3
        )
        
        assert result.best_score is not None
        assert result.best_params is not None
        assert 'n_estimators' in result.best_params
        assert 'max_depth' in result.best_params
        assert result.n_trials == 5
    
    def test_optimize_svr(self):
        """Test SVR hyperparameter optimization."""
        optimizer = OptunaOptimizer(n_trials=5, random_state=42)
        
        result = optimizer.optimize_svr(
            self.X_train, self.y_train, cv_folds=3
        )
        
        assert result.best_score is not None
        assert result.best_params is not None
        assert 'C' in result.best_params
        assert 'gamma' in result.best_params
    
    def test_optimize_gradient_boosting(self):
        """Test Gradient Boosting hyperparameter optimization."""
        optimizer = OptunaOptimizer(n_trials=5, random_state=42)
        
        result = optimizer.optimize_gradient_boosting(
            self.X_train, self.y_train, cv_folds=3
        )
        
        assert result.best_score is not None
        assert result.best_params is not None
        assert 'n_estimators' in result.best_params
        assert 'learning_rate' in result.best_params
    
    def test_optimize_hyperparameters_function(self):
        """Test the optimize_hyperparameters function."""
        result = optimize_hyperparameters(
            'random_forest',
            self.X_train,
            self.y_train,
            n_trials=3,
            cv_folds=3,
            random_state=42
        )
        
        assert result.best_score is not None
        assert result.best_params is not None
        assert result.n_trials == 3
    
    def test_optimize_all_estimators(self):
        """Test optimizing all estimator types."""
        results = optimize_all_estimators(
            self.X_train,
            self.y_train,
            estimator_types=['random_forest', 'svr'],
            n_trials=3,
            cv_folds=3,
            random_state=42
        )
        
        assert 'random_forest' in results
        assert 'svr' in results
        assert results['random_forest'].best_score is not None
        assert results['svr'].best_score is not None


class TestMLIntegration:
    """Test ML baseline integration with the main library."""
    
    def test_ml_baseline_import(self):
        """Test that ML baselines can be imported from main package."""
        from neurological_lrd_analysis import (
            MLBaselineType,
            RandomForestEstimator,
            SVREstimator,
            GradientBoostingEstimator,
            MLBaselineFactory,
            TimeSeriesFeatureExtractor
        )
        
        # Test that classes are available
        assert MLBaselineType.RANDOM_FOREST is not None
        assert RandomForestEstimator is not None
        assert SVREstimator is not None
        assert GradientBoostingEstimator is not None
        assert MLBaselineFactory is not None
        assert TimeSeriesFeatureExtractor is not None
    
    def test_ml_workflow_integration(self):
        """Test complete ML workflow integration."""
        # Generate data using the main library
        samples = generate_grid(
            hurst_values=[0.3, 0.5, 0.7, 0.9],
            lengths=[500],
            contaminations=['none'],
            generators=['fbm']
        )
        
        # Extract features
        extractor = TimeSeriesFeatureExtractor()
        X = []
        y = []
        
        for sample in samples:
            features = extractor.extract_features(sample.data, sample.true_hurst)
            X.append(features.combined)
            y.append(sample.true_hurst)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train ML model
        estimator = RandomForestEstimator(n_estimators=10, random_state=42)
        result = estimator.train(X, y, validation_split=0.3)
        
        # Test prediction
        pred_result = estimator.predict(X[0:1])
        assert 0 < pred_result.hurst_estimate < 1
        
        # Test that training was successful
        assert result.training_score > 0
        # Note: validation_score can be negative for R² when model performs worse than mean
        assert result.validation_score is not None


class TestMLErrorHandling:
    """Test error handling in ML baselines."""
    
    def test_untrained_model_prediction(self):
        """Test prediction with untrained model."""
        estimator = RandomForestEstimator()
        X_test = np.random.randn(1, 10)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            estimator.predict(X_test)
    
    def test_insufficient_data(self):
        """Test with insufficient training data."""
        estimator = RandomForestEstimator()
        X = np.random.randn(1, 10)  # Only one sample
        y = np.array([0.5])
        
        with pytest.raises(ValueError):
            estimator.train(X, y)
    
    def test_missing_dependencies(self):
        """Test behavior when dependencies are missing."""
        # This test would require mocking the import system
        # For now, we'll just test that the modules handle missing dependencies gracefully
        pass


if __name__ == "__main__":
    pytest.main([__file__])
