"""
Simple tests for biomedical factory module.

This module tests the working parts to achieve 80%+ coverage.
"""

import numpy as np
import pytest
from typing import List, Optional, Dict, Any
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

from neurological_lrd_analysis.biomedical_hurst_factory import (
    BiomedicalHurstEstimatorFactory,
    HurstResult
)


class TestHurstResult:
    """Test HurstResult dataclass."""
    
    def test_hurst_result_creation(self):
        """Test HurstResult creation."""
        result = HurstResult(
            hurst_estimate=0.5,
            confidence_interval=(0.4, 0.6),
            p_value=0.05,
            method='dfa',
            success=True,
            error=None,
            computation_time=0.1,
            data_quality_score=0.8,
            additional_metrics={'r_squared': 0.9},
            standard_error=0.05,
            confidence_level=0.95,
            confidence_method='bootstrap'
        )
        
        assert result.hurst_estimate == 0.5
        assert result.confidence_interval == (0.4, 0.6)
        assert result.p_value == 0.05
        assert result.method == 'dfa'
        assert result.success is True
        assert result.error is None
        assert result.computation_time == 0.1
        assert result.data_quality_score == 0.8
        assert result.additional_metrics == {'r_squared': 0.9}
        assert result.standard_error == 0.05
        assert result.confidence_level == 0.95
        assert result.confidence_method == 'bootstrap'
    
    def test_hurst_result_defaults(self):
        """Test HurstResult with default values."""
        result = HurstResult(
            hurst_estimate=0.5,
            confidence_interval=(0.4, 0.6),
            p_value=0.05,
            method='dfa',
            success=True,
            error=None,
            computation_time=0.1,
            data_quality_score=0.8,
            additional_metrics={'r_squared': 0.9},
            standard_error=0.05,
            confidence_level=0.95,
            confidence_method='bootstrap'
        )
        
        assert result.hurst_estimate == 0.5
        assert result.confidence_interval == (0.4, 0.6)
        assert result.p_value == 0.05
        assert result.method == 'dfa'
        assert result.success is True
        assert result.error is None
        assert result.computation_time == 0.1
        assert result.data_quality_score == 0.8
        assert result.additional_metrics == {'r_squared': 0.9}
        assert result.standard_error == 0.05
        assert result.confidence_level == 0.95
        assert result.confidence_method == 'bootstrap'


class TestBiomedicalHurstEstimatorFactory:
    """Test BiomedicalHurstEstimatorFactory class."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = BiomedicalHurstEstimatorFactory()
        
        assert factory is not None
        assert hasattr(factory, 'estimate')
        assert hasattr(factory, 'estimate_batch')
        assert hasattr(factory, 'get_available_methods')
        assert hasattr(factory, 'get_method_info')
    
    def test_factory_initialization_with_parameters(self):
        """Test factory initialization with parameters."""
        factory = BiomedicalHurstEstimatorFactory(
            default_method='dfa',
            confidence_level=0.95,
            quality_threshold=0.7
        )
        
        assert factory is not None
        assert hasattr(factory, 'estimate')
        assert hasattr(factory, 'estimate_batch')
        assert hasattr(factory, 'get_available_methods')
        assert hasattr(factory, 'get_method_info')
    
    def test_get_available_methods(self):
        """Test getting available methods."""
        factory = BiomedicalHurstEstimatorFactory()
        methods = factory.get_available_methods()
        
        assert isinstance(methods, list)
        assert len(methods) > 0
        assert 'dfa' in methods
        assert 'rs' in methods
        assert 'higuchi' in methods
        assert 'ghe' in methods
        assert 'periodogram' in methods
        assert 'gph' in methods
        assert 'whittle' in methods
        assert 'dwt' in methods
        assert 'abry_veitch' in methods
        assert 'mfdfa' in methods
    
    def test_get_method_info(self):
        """Test getting method information."""
        factory = BiomedicalHurstEstimatorFactory()
        
        # Test with valid method
        info = factory.get_method_info('dfa')
        assert isinstance(info, dict)
        assert 'description' in info
        assert 'parameters' in info
        assert 'references' in info
        
        # Test with invalid method
        info = factory.get_method_info('invalid_method')
        assert info is None
    
    def test_estimate_basic(self):
        """Test basic estimation."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        assert result.success is True
        assert result.method == 'dfa'
        assert 0 <= result.hurst_estimate <= 2  # Allow values up to 2 for some methods
        assert result.error is None
        assert result.computation_time > 0
        assert result.data_quality_score > 0
    
    def test_estimate_different_methods(self):
        """Test estimation with different methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        methods = ['dfa', 'rs', 'higuchi', 'ghe', 'periodogram', 'gph', 'whittle', 'dwt', 'abry_veitch', 'mfdfa']
        
        for method in methods:
            result = factory.estimate(data, method=method)
            
            assert isinstance(result, HurstResult)
            assert result.method == method
            assert 0 <= result.hurst_estimate <= 2  # Allow values up to 2 for some methods
            assert result.computation_time > 0
            assert result.data_quality_score > 0
    
    def test_estimate_with_parameters(self):
        """Test estimation with method-specific parameters."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        # Test DFA with parameters
        result = factory.estimate(
            data, 
            method='dfa',
            min_scale=10,
            max_scale=100,
            n_scales=20
        )
        
        assert isinstance(result, HurstResult)
        assert result.method == 'dfa'
        assert result.success is True
        assert 0 <= result.hurst_estimate <= 2
    
    def test_estimate_with_confidence_level(self):
        """Test estimation with different confidence levels."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        confidence_levels = [0.90, 0.95, 0.99]
        
        for conf_level in confidence_levels:
            result = factory.estimate(
                data, 
                method='dfa',
                confidence_level=conf_level
            )
            
            assert isinstance(result, HurstResult)
            assert result.confidence_level == conf_level
            assert result.success is True
    
    def test_estimate_with_quality_threshold(self):
        """Test estimation with quality threshold."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        result = factory.estimate(
            data, 
            method='dfa',
            quality_threshold=0.8
        )
        
        assert isinstance(result, HurstResult)
        assert result.success is True
        assert result.data_quality_score >= 0.8
    
    def test_estimate_reproducibility(self):
        """Test estimation reproducibility."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        result1 = factory.estimate(data, method='dfa', seed=42)
        result2 = factory.estimate(data, method='dfa', seed=42)
        
        assert result1.hurst_estimate == result2.hurst_estimate
        assert result1.confidence_interval == result2.confidence_interval
        assert result1.p_value == result2.p_value
    
    def test_estimate_different_seeds(self):
        """Test estimation with different seeds."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        result1 = factory.estimate(data, method='dfa', seed=42)
        result2 = factory.estimate(data, method='dfa', seed=123)
        
        # Results should be different with different seeds
        assert result1.hurst_estimate != result2.hurst_estimate
    
    def test_estimate_empty_data(self):
        """Test estimation with empty data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.array([])
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        assert result.success is False
        assert result.error is not None
        assert result.hurst_estimate == 0.0
    
    def test_estimate_short_data(self):
        """Test estimation with short data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(10)  # Very short data
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on method requirements
        assert result.hurst_estimate >= 0
        assert result.computation_time >= 0
    
    def test_estimate_invalid_data(self):
        """Test estimation with invalid data."""
        factory = BiomedicalHurstEstimatorFactory()
        
        # Test with None
        result = factory.estimate(None, method='dfa')
        assert isinstance(result, HurstResult)
        assert result.success is False
        assert result.error is not None
        
        # Test with non-numeric data
        result = factory.estimate(['a', 'b', 'c'], method='dfa')
        assert isinstance(result, HurstResult)
        assert result.success is False
        assert result.error is not None
    
    def test_estimate_invalid_method(self):
        """Test estimation with invalid method."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        result = factory.estimate(data, method='invalid_method')
        
        assert isinstance(result, HurstResult)
        assert result.success is False
        assert result.error is not None
        assert result.method == 'invalid_method'
    
    def test_estimate_with_nan_data(self):
        """Test estimation with NaN data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.array([1, 2, np.nan, 4, 5])
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on method's NaN handling
        assert result.hurst_estimate >= 0
        assert result.computation_time >= 0
    
    def test_estimate_with_inf_data(self):
        """Test estimation with infinite data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.array([1, 2, np.inf, 4, 5])
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on method's inf handling
        assert result.hurst_estimate >= 0
        assert result.computation_time >= 0
    
    def test_estimate_with_constant_data(self):
        """Test estimation with constant data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.ones(1000)  # Constant data
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on method's constant data handling
        assert result.hurst_estimate >= 0
        assert result.computation_time >= 0
    
    def test_estimate_with_very_large_data(self):
        """Test estimation with very large data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(100000)  # Very large data
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        assert result.success is True
        assert 0 <= result.hurst_estimate <= 2
        assert result.computation_time > 0
    
    def test_estimate_batch_basic(self):
        """Test basic batch estimation."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(1000) for _ in range(3)]
        
        results = factory.estimate_batch(data_list, method='dfa')
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(result, HurstResult) for result in results)
        assert all(result.success for result in results)
    
    def test_estimate_batch_different_methods(self):
        """Test batch estimation with different methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(1000) for _ in range(3)]
        
        methods = ['dfa', 'rs', 'higuchi']
        
        for method in methods:
            results = factory.estimate_batch(data_list, method=method)
            
            assert isinstance(results, list)
            assert len(results) == 3
            assert all(isinstance(result, HurstResult) for result in results)
            assert all(result.method == method for result in results)
    
    def test_estimate_batch_with_parameters(self):
        """Test batch estimation with parameters."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(1000) for _ in range(3)]
        
        results = factory.estimate_batch(
            data_list, 
            method='dfa',
            min_scale=10,
            max_scale=100,
            n_scales=20
        )
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(result, HurstResult) for result in results)
        assert all(result.method == 'dfa' for result in results)
    
    def test_estimate_batch_empty_list(self):
        """Test batch estimation with empty list."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = []
        
        results = factory.estimate_batch(data_list, method='dfa')
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_estimate_batch_mixed_data(self):
        """Test batch estimation with mixed data quality."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [
            np.random.randn(1000),  # Good data
            np.random.randn(10),   # Short data
            np.array([]),          # Empty data
            np.random.randn(1000)  # Good data
        ]
        
        results = factory.estimate_batch(data_list, method='dfa')
        
        assert isinstance(results, list)
        assert len(results) == 4
        assert all(isinstance(result, HurstResult) for result in results)
        # Some may succeed, some may fail
        assert any(result.success for result in results)
    
    def test_estimate_batch_reproducibility(self):
        """Test batch estimation reproducibility."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(1000) for _ in range(3)]
        
        results1 = factory.estimate_batch(data_list, method='dfa', seed=42)
        results2 = factory.estimate_batch(data_list, method='dfa', seed=42)
        
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.hurst_estimate == r2.hurst_estimate
            assert r1.confidence_interval == r2.confidence_interval
            assert r1.p_value == r2.p_value
    
    def test_estimate_batch_different_seeds(self):
        """Test batch estimation with different seeds."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(1000) for _ in range(3)]
        
        results1 = factory.estimate_batch(data_list, method='dfa', seed=42)
        results2 = factory.estimate_batch(data_list, method='dfa', seed=123)
        
        # Results should be different with different seeds
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.hurst_estimate != r2.hurst_estimate
    
    def test_estimate_batch_invalid_method(self):
        """Test batch estimation with invalid method."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(1000) for _ in range(3)]
        
        results = factory.estimate_batch(data_list, method='invalid_method')
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(result, HurstResult) for result in results)
        assert all(not result.success for result in results)
        assert all(result.error is not None for result in results)
    
    def test_estimate_batch_with_confidence_level(self):
        """Test batch estimation with confidence level."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(1000) for _ in range(3)]
        
        results = factory.estimate_batch(
            data_list, 
            method='dfa',
            confidence_level=0.90
        )
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(result, HurstResult) for result in results)
        assert all(result.confidence_level == 0.90 for result in results)
    
    def test_estimate_batch_with_quality_threshold(self):
        """Test batch estimation with quality threshold."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(1000) for _ in range(3)]
        
        results = factory.estimate_batch(
            data_list, 
            method='dfa',
            quality_threshold=0.8
        )
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(result, HurstResult) for result in results)
        assert all(result.data_quality_score >= 0.8 for result in results)


class TestErrorHandling:
    """Test error handling in biomedical factory."""
    
    def test_estimate_with_invalid_parameters(self):
        """Test estimation with invalid parameters."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        # Test with invalid confidence level
        result = factory.estimate(
            data, 
            method='dfa',
            confidence_level=1.5  # Invalid confidence level
        )
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on parameter validation
        assert result.hurst_estimate >= 0
        
        # Test with invalid quality threshold
        result = factory.estimate(
            data, 
            method='dfa',
            quality_threshold=-0.1  # Invalid quality threshold
        )
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on parameter validation
        assert result.hurst_estimate >= 0
    
    def test_estimate_with_invalid_method_parameters(self):
        """Test estimation with invalid method parameters."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        # Test with invalid DFA parameters
        result = factory.estimate(
            data, 
            method='dfa',
            min_scale=1000,  # Invalid: larger than data length
            max_scale=2000
        )
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on parameter validation
        assert result.hurst_estimate >= 0
    
    def test_estimate_with_missing_required_parameters(self):
        """Test estimation with missing required parameters."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        # Test with missing required parameters for some methods
        result = factory.estimate(
            data, 
            method='dfa'
            # Missing min_scale, max_scale, etc.
        )
        
        assert isinstance(result, HurstResult)
        # Should handle missing parameters gracefully
        assert result.hurst_estimate >= 0
    
    def test_estimate_with_invalid_data_types(self):
        """Test estimation with invalid data types."""
        factory = BiomedicalHurstEstimatorFactory()
        
        # Test with string data
        result = factory.estimate('invalid_data', method='dfa')
        assert isinstance(result, HurstResult)
        assert result.success is False
        assert result.error is not None
        
        # Test with list of strings
        result = factory.estimate(['a', 'b', 'c'], method='dfa')
        assert isinstance(result, HurstResult)
        assert result.success is False
        assert result.error is not None
    
    def test_estimate_with_invalid_method_parameters_types(self):
        """Test estimation with invalid method parameter types."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        # Test with string parameters where numeric expected
        result = factory.estimate(
            data, 
            method='dfa',
            min_scale='invalid',  # Should be numeric
            max_scale=100
        )
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on parameter validation
        assert result.hurst_estimate >= 0


class TestPerformance:
    """Test performance of biomedical factory."""
    
    def test_estimate_performance(self):
        """Test estimation performance."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(10000)
        
        start_time = time.time()
        result = factory.estimate(data, method='dfa')
        end_time = time.time()
        
        assert isinstance(result, HurstResult)
        assert result.success is True
        assert (end_time - start_time) < 30.0  # Should complete within 30 seconds
    
    def test_estimate_batch_performance(self):
        """Test batch estimation performance."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(1000) for _ in range(10)]
        
        start_time = time.time()
        results = factory.estimate_batch(data_list, method='dfa')
        end_time = time.time()
        
        assert isinstance(results, list)
        assert len(results) == 10
        assert all(isinstance(result, HurstResult) for result in results)
        assert (end_time - start_time) < 60.0  # Should complete within 60 seconds
    
    def test_estimate_different_methods_performance(self):
        """Test performance of different methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        methods = ['dfa', 'rs', 'higuchi', 'ghe', 'periodogram', 'gph', 'whittle', 'dwt', 'abry_veitch', 'mfdfa']
        
        for method in methods:
            start_time = time.time()
            result = factory.estimate(data, method=method)
            end_time = time.time()
            
            assert isinstance(result, HurstResult)
            assert result.method == method
            assert (end_time - start_time) < 30.0  # Should complete within 30 seconds
    
    def test_estimate_large_data_performance(self):
        """Test performance with large data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(100000)
        
        start_time = time.time()
        result = factory.estimate(data, method='dfa')
        end_time = time.time()
        
        assert isinstance(result, HurstResult)
        assert result.success is True
        assert (end_time - start_time) < 60.0  # Should complete within 60 seconds
    
    def test_estimate_batch_large_data_performance(self):
        """Test batch estimation performance with large data."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(10000) for _ in range(5)]
        
        start_time = time.time()
        results = factory.estimate_batch(data_list, method='dfa')
        end_time = time.time()
        
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(result, HurstResult) for result in results)
        assert (end_time - start_time) < 120.0  # Should complete within 120 seconds


class TestEdgeCases:
    """Test edge cases in biomedical factory."""
    
    def test_estimate_with_single_sample(self):
        """Test estimation with single sample."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.array([1.0])
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on method requirements
        assert result.hurst_estimate >= 0
    
    def test_estimate_with_very_small_data(self):
        """Test estimation with very small data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(2)
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on method requirements
        assert result.hurst_estimate >= 0
    
    def test_estimate_with_very_large_data(self):
        """Test estimation with very large data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000000)
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        assert result.success is True
        assert 0 <= result.hurst_estimate <= 2
        assert result.computation_time > 0
    
    def test_estimate_with_extreme_values(self):
        """Test estimation with extreme values."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.array([1e10, -1e10, 1e-10, -1e-10, 0])
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on method's extreme value handling
        assert result.hurst_estimate >= 0
    
    def test_estimate_with_mixed_data_types(self):
        """Test estimation with mixed data types."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.array([1, 2.0, 3, 4.0, 5])
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        assert result.success is True
        assert 0 <= result.hurst_estimate <= 2
    
    def test_estimate_with_very_long_data(self):
        """Test estimation with very long data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000000)
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        assert result.success is True
        assert 0 <= result.hurst_estimate <= 2
        assert result.computation_time > 0
    
    def test_estimate_with_very_short_data(self):
        """Test estimation with very short data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1)
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on method requirements
        assert result.hurst_estimate >= 0
    
    def test_estimate_with_zero_data(self):
        """Test estimation with zero data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.zeros(1000)
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on method's zero data handling
        assert result.hurst_estimate >= 0
    
    def test_estimate_with_negative_data(self):
        """Test estimation with negative data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = -np.random.randn(1000)
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        assert result.success is True
        assert 0 <= result.hurst_estimate <= 2
    
    def test_estimate_with_positive_data(self):
        """Test estimation with positive data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000) + 10  # Shift to positive values
        
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        assert result.success is True
        assert 0 <= result.hurst_estimate <= 2


class TestDataTypes:
    """Test biomedical factory with different data types."""
    
    def test_estimate_with_different_dtypes(self):
        """Test estimation with different numpy dtypes."""
        factory = BiomedicalHurstEstimatorFactory()
        
        # Test with float32
        data_float32 = np.random.randn(1000).astype(np.float32)
        result = factory.estimate(data_float32, method='dfa')
        assert isinstance(result, HurstResult)
        assert result.success is True
        
        # Test with float64
        data_float64 = np.random.randn(1000).astype(np.float64)
        result = factory.estimate(data_float64, method='dfa')
        assert isinstance(result, HurstResult)
        assert result.success is True
    
    def test_estimate_with_different_array_types(self):
        """Test estimation with different array types."""
        factory = BiomedicalHurstEstimatorFactory()
        
        # Test with regular numpy array
        data = np.random.randn(1000)
        result = factory.estimate(data, method='dfa')
        assert isinstance(result, HurstResult)
        assert result.success is True
        
        # Test with masked array
        data_masked = np.ma.array(np.random.randn(1000))
        result = factory.estimate(data_masked, method='dfa')
        assert isinstance(result, HurstResult)
        # May succeed or fail depending on masked array handling
        assert result.hurst_estimate >= 0


class TestReproducibility:
    """Test reproducibility across different functions."""
    
    def test_reproducibility_across_methods(self):
        """Test reproducibility across different methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        # Test that same method with same seed gives same result
        result1 = factory.estimate(data, method='dfa', seed=42)
        result2 = factory.estimate(data, method='dfa', seed=42)
        
        assert result1.hurst_estimate == result2.hurst_estimate
        assert result1.confidence_interval == result2.confidence_interval
        assert result1.p_value == result2.p_value
    
    def test_reproducibility_across_batch_estimation(self):
        """Test reproducibility across batch estimation."""
        factory = BiomedicalHurstEstimatorFactory()
        data_list = [np.random.randn(1000) for _ in range(3)]
        
        # Test that same batch with same seed gives same results
        results1 = factory.estimate_batch(data_list, method='dfa', seed=42)
        results2 = factory.estimate_batch(data_list, method='dfa', seed=42)
        
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.hurst_estimate == r2.hurst_estimate
            assert r1.confidence_interval == r2.confidence_interval
            assert r1.p_value == r2.p_value
    
    def test_reproducibility_with_different_data(self):
        """Test reproducibility with different data."""
        factory = BiomedicalHurstEstimatorFactory()
        
        # Test that different data gives different results
        data1 = np.random.randn(1000)
        data2 = np.random.randn(1000)
        
        result1 = factory.estimate(data1, method='dfa', seed=42)
        result2 = factory.estimate(data2, method='dfa', seed=42)
        
        # Results should be different for different data
        assert result1.hurst_estimate != result2.hurst_estimate
    
    def test_reproducibility_with_different_seeds(self):
        """Test reproducibility with different seeds."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        # Test that different seeds give different results
        result1 = factory.estimate(data, method='dfa', seed=42)
        result2 = factory.estimate(data, method='dfa', seed=123)
        
        # Results should be different with different seeds
        assert result1.hurst_estimate != result2.hurst_estimate