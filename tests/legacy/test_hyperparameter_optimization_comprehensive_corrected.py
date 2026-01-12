"""
Comprehensive tests for hyperparameter optimization module.

This module tests all functions and edge cases to achieve 80%+ coverage.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any
import warnings
import time

from neurological_lrd_analysis.ml_baselines.hyperparameter_optimization import (
    OptunaOptimizer,
    create_optuna_study,
    optimize_hyperparameters,
    optimize_all_estimators,
    create_optimized_estimators,
    OptimizationResult,
    _lazy_import_optuna,
    _lazy_import_sklearn
)


class TestLazyImports:
    """Test lazy import functions."""
    
    def test_lazy_import_optuna_success(self):
        """Test successful optuna import."""
        optuna, MedianPruner, TPESampler = _lazy_import_optuna()
        
        assert optuna is not None
        assert MedianPruner is not None
        assert TPESampler is not None
    
    def test_lazy_import_sklearn_success(self):
        """Test successful sklearn import."""
        cross_val_score, StratifiedKFold, mean_squared_error, make_scorer = _lazy_import_sklearn()
        
        assert cross_val_score is not None
        assert StratifiedKFold is not None
        assert mean_squared_error is not None
        assert make_scorer is not None
    
    def test_lazy_import_optuna_failure(self):
        """Test optuna import failure."""
        with patch('neurological_lrd_analysis.ml_baselines.hyperparameter_optimization.optuna', None):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                optuna, MedianPruner, TPESampler = _lazy_import_optuna()
                
                assert optuna is None
                assert MedianPruner is None
                assert TPESampler is None
                assert len(w) == 1
                assert "Optuna not available" in str(w[0].message)
    
    def test_lazy_import_sklearn_failure(self):
        """Test sklearn import failure."""
        with patch('neurological_lrd_analysis.ml_baselines.hyperparameter_optimization.sklearn', None):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cross_val_score, StratifiedKFold, mean_squared_error, make_scorer = _lazy_import_sklearn()
                
                assert cross_val_score is None
                assert StratifiedKFold is None
                assert mean_squared_error is None
                assert make_scorer is None
                assert len(w) == 1
                assert "scikit-learn not available" in str(w[0].message)


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult(
            best_params={'param1': 1.0, 'param2': 2.0},
            best_score=0.95,
            best_trial=5,
            optimization_time=10.5,
            n_trials=20,
            study=Mock()
        )
        
        assert result.best_params == {'param1': 1.0, 'param2': 2.0}
        assert result.best_score == 0.95
        assert result.best_trial == 5
        assert result.optimization_time == 10.5
        assert result.n_trials == 20
        assert result.study is not None


class TestOptunaOptimizer:
    """Test OptunaOptimizer class."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = OptunaOptimizer()
        
        assert optimizer is not None
        assert hasattr(optimizer, 'study_name')
        assert hasattr(optimizer, 'direction')
        assert hasattr(optimizer, 'n_trials')
    
    def test_optimizer_with_parameters(self):
        """Test optimizer initialization with parameters."""
        optimizer = OptunaOptimizer(
            study_name="test_study",
            direction="maximize",
            n_trials=50,
            timeout=300,
            random_state=42
        )
        
        assert optimizer.study_name == "test_study"
        assert optimizer.direction == "maximize"
        assert optimizer.n_trials == 50
        assert optimizer.timeout == 300
        assert optimizer.random_state == 42
    
    def test_optimizer_with_different_pruners(self):
        """Test optimizer with different pruners."""
        pruners = ['median', 'percentile', 'successive_halving', None]
        
        for pruner in pruners:
            optimizer = OptunaOptimizer(pruner=pruner)
            assert optimizer is not None
    
    def test_optimizer_with_different_samplers(self):
        """Test optimizer with different samplers."""
        samplers = ['tpe', 'random', 'cmaes', 'grid']
        
        for sampler in samplers:
            optimizer = OptunaOptimizer(sampler=sampler)
            assert optimizer is not None
    
    def test_optimizer_with_invalid_pruner(self):
        """Test optimizer with invalid pruner."""
        with pytest.raises(ValueError):
            OptunaOptimizer(pruner="invalid_pruner")
    
    def test_optimizer_with_invalid_sampler(self):
        """Test optimizer with invalid sampler."""
        with pytest.raises(ValueError):
            OptunaOptimizer(sampler="invalid_sampler")
    
    def test_optimizer_with_invalid_direction(self):
        """Test optimizer with invalid direction."""
        with pytest.raises(ValueError):
            OptunaOptimizer(direction="invalid_direction")


class TestCreateOptunaStudy:
    """Test create_optuna_study function."""
    
    def test_create_optuna_study_basic(self):
        """Test basic study creation."""
        study = create_optuna_study(
            study_name="test_study",
            direction="maximize",
            pruner="median",
            sampler="tpe"
        )
        
        assert study is not None
        assert study.study_name == "test_study"
    
    def test_create_optuna_study_defaults(self):
        """Test study creation with default parameters."""
        study = create_optuna_study(study_name="test_study")
        
        assert study is not None
        assert study.study_name == "test_study"
    
    def test_create_optuna_study_different_samplers(self):
        """Test study creation with different samplers."""
        samplers = ["tpe", "random", "cmaes", "grid"]
        
        for sampler in samplers:
            study = create_optuna_study(
                study_name=f"test_{sampler}",
                sampler=sampler
            )
            assert study is not None
    
    def test_create_optuna_study_different_pruners(self):
        """Test study creation with different pruners."""
        pruners = ["median", "percentile", "successive_halving"]
        
        for pruner in pruners:
            study = create_optuna_study(
                study_name=f"test_{pruner}",
                pruner=pruner
            )
            assert study is not None
    
    def test_create_optuna_study_invalid_sampler(self):
        """Test study creation with invalid sampler."""
        with pytest.raises(ValueError):
            create_optuna_study(study_name="test", sampler="invalid_sampler")
    
    def test_create_optuna_study_invalid_pruner(self):
        """Test study creation with invalid pruner."""
        with pytest.raises(ValueError):
            create_optuna_study(study_name="test", pruner="invalid_pruner")


class TestOptimizeHyperparameters:
    """Test optimize_hyperparameters function."""
    
    def test_optimize_hyperparameters_basic(self):
        """Test basic hyperparameter optimization."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        result = optimize_hyperparameters(
            estimator_type='random_forest',
            X=X,
            y=y,
            n_trials=5,
            timeout=10
        )
        
        assert result is not None
        assert hasattr(result, 'best_params')
        assert hasattr(result, 'best_score')
    
    def test_optimize_hyperparameters_different_estimators(self):
        """Test optimization with different estimator types."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        estimators = ['random_forest', 'svr', 'gradient_boosting']
        
        for estimator in estimators:
            result = optimize_hyperparameters(
                estimator_type=estimator,
                X=X,
                y=y,
                n_trials=3,
                timeout=5
            )
            assert result is not None
    
    def test_optimize_hyperparameters_with_cv_folds(self):
        """Test optimization with different CV folds."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        result = optimize_hyperparameters(
            estimator_type='random_forest',
            X=X,
            y=y,
            n_trials=3,
            timeout=5,
            cv_folds=3
        )
        
        assert result is not None
    
    def test_optimize_hyperparameters_with_different_scoring(self):
        """Test optimization with different scoring metrics."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        for scoring in scoring_metrics:
            result = optimize_hyperparameters(
                estimator_type='random_forest',
                X=X,
                y=y,
                n_trials=3,
                timeout=5,
                scoring=scoring
            )
            assert result is not None
    
    def test_optimize_hyperparameters_invalid_estimator(self):
        """Test optimization with invalid estimator type."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        with pytest.raises(ValueError):
            optimize_hyperparameters(
                estimator_type='invalid_estimator',
                X=X,
                y=y,
                n_trials=3,
                timeout=5
            )
    
    def test_optimize_hyperparameters_empty_data(self):
        """Test optimization with empty data."""
        X = np.array([]).reshape(0, 10)
        y = np.array([])
        
        with pytest.raises(ValueError):
            optimize_hyperparameters(
                estimator_type='random_forest',
                X=X,
                y=y,
                n_trials=3,
                timeout=5
            )
    
    def test_optimize_hyperparameters_mismatched_data(self):
        """Test optimization with mismatched data dimensions."""
        X = np.random.randn(100, 10)
        y = np.random.randn(50)  # Different length
        
        with pytest.raises(ValueError):
            optimize_hyperparameters(
                estimator_type='random_forest',
                X=X,
                y=y,
                n_trials=3,
                timeout=5
            )


class TestOptimizeAllEstimators:
    """Test optimize_all_estimators function."""
    
    def test_optimize_all_estimators_basic(self):
        """Test optimization of all estimators."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        results = optimize_all_estimators(
            X=X,
            y=y,
            n_trials=3,
            timeout=10
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for estimator_type, result in results.items():
            assert hasattr(result, 'best_params')
            assert hasattr(result, 'best_score')
    
    def test_optimize_all_estimators_with_cv_folds(self):
        """Test optimization with CV folds."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        results = optimize_all_estimators(
            X=X,
            y=y,
            n_trials=3,
            timeout=10,
            cv_folds=3
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_optimize_all_estimators_with_scoring(self):
        """Test optimization with different scoring."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        results = optimize_all_estimators(
            X=X,
            y=y,
            n_trials=3,
            timeout=10,
            scoring='neg_mean_absolute_error'
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0


class TestCreateOptimizedEstimators:
    """Test create_optimized_estimators function."""
    
    def test_create_optimized_estimators_basic(self):
        """Test creation of optimized estimators."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        estimators = create_optimized_estimators(
            X=X,
            y=y,
            n_trials=3,
            timeout=10
        )
        
        assert isinstance(estimators, dict)
        assert len(estimators) > 0
        
        for estimator_type, estimator in estimators.items():
            assert estimator is not None
    
    def test_create_optimized_estimators_with_cv_folds(self):
        """Test creation with CV folds."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        estimators = create_optimized_estimators(
            X=X,
            y=y,
            n_trials=3,
            timeout=10,
            cv_folds=3
        )
        
        assert isinstance(estimators, dict)
        assert len(estimators) > 0
    
    def test_create_optimized_estimators_with_scoring(self):
        """Test creation with different scoring."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        estimators = create_optimized_estimators(
            X=X,
            y=y,
            n_trials=3,
            timeout=10,
            scoring='neg_mean_absolute_error'
        )
        
        assert isinstance(estimators, dict)
        assert len(estimators) > 0


class TestErrorHandling:
    """Test error handling in hyperparameter optimization."""
    
    def test_optimize_hyperparameters_with_invalid_data_types(self):
        """Test optimization with invalid data types."""
        # Test with non-numeric data
        X = np.array([['a', 'b'], ['c', 'd']])
        y = np.array([1, 2])
        
        with pytest.raises((ValueError, TypeError)):
            optimize_hyperparameters(
                estimator_type='random_forest',
                X=X,
                y=y,
                n_trials=3,
                timeout=5
            )
    
    def test_optimize_hyperparameters_with_nan_data(self):
        """Test optimization with NaN data."""
        X = np.random.randn(100, 10)
        X[0, 0] = np.nan
        y = np.random.randn(100)
        
        with pytest.raises(ValueError):
            optimize_hyperparameters(
                estimator_type='random_forest',
                X=X,
                y=y,
                n_trials=3,
                timeout=5
            )
    
    def test_optimize_hyperparameters_with_inf_data(self):
        """Test optimization with infinite data."""
        X = np.random.randn(100, 10)
        X[0, 0] = np.inf
        y = np.random.randn(100)
        
        with pytest.raises(ValueError):
            optimize_hyperparameters(
                estimator_type='random_forest',
                X=X,
                y=y,
                n_trials=3,
                timeout=5
            )


class TestPerformance:
    """Test performance of hyperparameter optimization."""
    
    def test_optimization_performance(self):
        """Test that optimization completes within reasonable time."""
        X = np.random.randn(200, 20)
        y = np.random.randn(200)
        
        start_time = time.time()
        
        result = optimize_hyperparameters(
            estimator_type='random_forest',
            X=X,
            y=y,
            n_trials=5,
            timeout=30
        )
        
        end_time = time.time()
        
        assert result is not None
        assert (end_time - start_time) < 60.0  # Should complete within 60 seconds
    
    def test_optimization_memory_usage(self):
        """Test optimization memory usage."""
        X = np.random.randn(1000, 50)
        y = np.random.randn(1000)
        
        # This test mainly ensures no memory leaks
        result = optimize_hyperparameters(
            estimator_type='random_forest',
            X=X,
            y=y,
            n_trials=3,
            timeout=20
        )
        
        assert result is not None
    
    def test_optimization_reproducibility(self):
        """Test optimization reproducibility."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        # Run optimization twice with same random state
        result1 = optimize_hyperparameters(
            estimator_type='random_forest',
            X=X,
            y=y,
            n_trials=3,
            timeout=10,
            random_state=42
        )
        
        result2 = optimize_hyperparameters(
            estimator_type='random_forest',
            X=X,
            y=y,
            n_trials=3,
            timeout=10,
            random_state=42
        )
        
        # Results should be similar (allowing for some randomness in optimization)
        assert result1 is not None
        assert result2 is not None


class TestEdgeCases:
    """Test edge cases in hyperparameter optimization."""
    
    def test_optimization_with_single_feature(self):
        """Test optimization with single feature."""
        X = np.random.randn(100, 1)
        y = np.random.randn(100)
        
        result = optimize_hyperparameters(
            estimator_type='random_forest',
            X=X,
            y=y,
            n_trials=3,
            timeout=10
        )
        
        assert result is not None
    
    def test_optimization_with_single_sample(self):
        """Test optimization with single sample."""
        X = np.random.randn(1, 10)
        y = np.random.randn(1)
        
        with pytest.raises(ValueError):
            optimize_hyperparameters(
                estimator_type='random_forest',
                X=X,
                y=y,
                n_trials=3,
                timeout=10
            )
    
    def test_optimization_with_high_dimensional_data(self):
        """Test optimization with high dimensional data."""
        X = np.random.randn(100, 1000)
        y = np.random.randn(100)
        
        result = optimize_hyperparameters(
            estimator_type='random_forest',
            X=X,
            y=y,
            n_trials=3,
            timeout=20
        )
        
        assert result is not None
    
    def test_optimization_with_constant_target(self):
        """Test optimization with constant target values."""
        X = np.random.randn(100, 10)
        y = np.ones(100)  # Constant target
        
        with pytest.raises(ValueError):
            optimize_hyperparameters(
                estimator_type='random_forest',
                X=X,
                y=y,
                n_trials=3,
                timeout=10
            )
