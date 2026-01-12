"""
Comprehensive tests to improve biomedical factory coverage to 80%+.
"""

import numpy as np
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from neurological_lrd_analysis.biomedical_hurst_factory import (
    BiomedicalHurstEstimatorFactory,
    HurstResult,
    BiomedicalDataProcessor,
    ConfidenceEstimator,
    DFAEstimator,
    RSAnalysisEstimator,
    HiguchiEstimator,
    GHEEstimator,
    PeriodogramEstimator,
    GPHEstimator,
    WhittleMLEEstimator,
    DWTEstimator,
    AbryVeitchEstimator,
    MFDFAEstimator,
    BayesianHurstEstimator,
    EstimatorType,
    ConfidenceMethod,
)


class TestBiomedicalDataProcessor:
    """Test BiomedicalDataProcessor for better coverage."""
    
    def test_assess_data_quality_empty_data(self):
        """Test data quality assessment with empty data."""
        processor = BiomedicalDataProcessor()
        result = processor.assess_data_quality(np.array([]))
        
        assert result["data_length"] == 0
        assert result["missing_values"] == 0
        assert result["outlier_fraction"] == 0.0
        assert result["signal_to_noise_ratio"] == 0.0
        assert result["artifact_detection"]["eye_movements"] == 0
        assert result["artifact_detection"]["muscle_artifacts"] == 0
        assert result["artifact_detection"]["electrode_pop"] == 0
        assert result["artifact_detection"]["sudden_jumps"] == 0
        assert result["artifact_detection"]["jump_fraction"] == 0.0
    
    def test_assess_data_quality_all_nan(self):
        """Test data quality assessment with all NaN data."""
        processor = BiomedicalDataProcessor()
        data = np.full(100, np.nan)
        result = processor.assess_data_quality(data)
        
        assert result["data_length"] == 100
        assert result["missing_values"] == 100
        assert result["outlier_fraction"] == 0.0
        assert result["signal_to_noise_ratio"] == 0.0
    
    def test_assess_data_quality_with_outliers(self):
        """Test data quality assessment with outliers."""
        processor = BiomedicalDataProcessor()
        data = np.random.randn(1000)
        data[100:110] = 10.0  # Add outliers
        result = processor.assess_data_quality(data)
        
        assert result["data_length"] == 1000
        assert result["missing_values"] == 0
        assert result["outlier_fraction"] > 0.0
    
    def test_assess_data_quality_with_artifacts(self):
        """Test data quality assessment with artifacts."""
        processor = BiomedicalDataProcessor()
        data = np.random.randn(1000)
        # Add eye movement artifacts
        data[200:250] += 5.0
        result = processor.assess_data_quality(data)
        
        assert result["artifact_detection"]["eye_movements"] > 0
    
    def test_preprocess_data_basic(self):
        """Test basic data preprocessing."""
        processor = BiomedicalDataProcessor()
        data = np.random.randn(1000)
        processed = processor.preprocess_data(data)
        
        assert len(processed) == len(data)
        assert np.allclose(np.mean(processed), 0.0, atol=1e-10)
        assert np.allclose(np.std(processed), 1.0, atol=1e-10)
    
    def test_preprocess_data_with_nan(self):
        """Test preprocessing with NaN values."""
        processor = BiomedicalDataProcessor()
        data = np.random.randn(1000)
        data[100:110] = np.nan
        processed = processor.preprocess_data(data)
        
        assert len(processed) == len(data)
        assert not np.any(np.isnan(processed))
    
    def test_preprocess_data_with_outliers(self):
        """Test preprocessing with outliers."""
        processor = BiomedicalDataProcessor()
        data = np.random.randn(1000)
        data[100:110] = 10.0  # Add outliers
        processed = processor.preprocess_data(data, outlier_threshold=3.0)
        
        assert len(processed) == len(data)
        # Outliers should be clipped
        assert np.all(processed <= 3.0)
        assert np.all(processed >= -3.0)


class TestConfidenceEstimator:
    """Test ConfidenceEstimator for better coverage."""
    
    def test_bootstrap_confidence_basic(self):
        """Test basic bootstrap confidence estimation."""
        estimator = ConfidenceEstimator()
        data = np.random.randn(1000)
        
        # Mock the estimator to return a consistent result
        with patch.object(estimator, '_estimate_single', return_value=0.6):
            result = estimator.bootstrap_confidence(
                data, n_bootstrap=10, confidence_level=0.95, random_state=42
            )
            
            assert 'mean' in result
            assert 'std' in result
            assert 'confidence_interval' in result
            assert len(result['confidence_interval']) == 2
    
    def test_bootstrap_confidence_insufficient_data(self):
        """Test bootstrap with insufficient data."""
        estimator = ConfidenceEstimator()
        data = np.random.randn(5)  # Very small dataset
        
        result = estimator.bootstrap_confidence(
            data, n_bootstrap=10, confidence_level=0.95, random_state=42
        )
        
        # Should handle gracefully
        assert 'mean' in result
        assert 'std' in result
    
    def test_theoretical_confidence(self):
        """Test theoretical confidence estimation."""
        estimator = ConfidenceEstimator()
        
        result = estimator.theoretical_confidence(
            hurst_estimate=0.6,
            standard_error=0.05,
            n_samples=1000,
            confidence_level=0.95
        )
        
        assert len(result) == 2
        assert result[0] < result[1]
        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[1] <= 1.0
    
    def test_bayesian_confidence(self):
        """Test Bayesian confidence estimation."""
        estimator = ConfidenceEstimator()
        
        # Mock the Bayesian estimator
        with patch('neurological_lrd_analysis.biomedical_hurst_factory.BayesianHurstEstimator') as mock_bayesian:
            mock_bayesian.bayesian_confidence.return_value = (0.6, (0.55, 0.65), {})
            
            result = estimator.bayesian_confidence(
                data=np.random.randn(1000),
                estimator_type=EstimatorType.DFA,
                confidence_level=0.95,
                num_samples=100
            )
            
            assert 'mean' in result
            assert 'credible_interval' in result
            assert len(result['credible_interval']) == 2


class TestIndividualEstimators:
    """Test individual estimators for better coverage."""
    
    def test_dfa_estimator_edge_cases(self):
        """Test DFA estimator with edge cases."""
        estimator = DFAEstimator()
        
        # Test with very short data
        data = np.random.randn(10)
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_dfa_estimator_with_parameters(self):
        """Test DFA estimator with custom parameters."""
        estimator = DFAEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(
            data,
            min_scale=20,
            max_scale=200,
            polynomial_order=2,
            seed=42
        )
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
        assert 'scales' in result[1]
        assert 'fluctuations' in result[1]
    
    def test_rs_estimator_edge_cases(self):
        """Test R/S estimator with edge cases."""
        estimator = RSAnalysisEstimator()
        
        # Test with very short data
        data = np.random.randn(10)
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_higuchi_estimator_edge_cases(self):
        """Test Higuchi estimator with edge cases."""
        estimator = HiguchiEstimator()
        
        # Test with very short data
        data = np.random.randn(10)
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_ghe_estimator_edge_cases(self):
        """Test GHE estimator with edge cases."""
        estimator = GHEEstimator()
        
        # Test with very short data
        data = np.random.randn(10)
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_periodogram_estimator_edge_cases(self):
        """Test Periodogram estimator with edge cases."""
        estimator = PeriodogramEstimator()
        
        # Test with very short data
        data = np.random.randn(10)
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_gph_estimator_edge_cases(self):
        """Test GPH estimator with edge cases."""
        estimator = GPHEstimator()
        
        # Test with very short data
        data = np.random.randn(10)
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_whittle_mle_estimator_edge_cases(self):
        """Test Whittle MLE estimator with edge cases."""
        estimator = WhittleMLEEstimator()
        
        # Test with very short data
        data = np.random.randn(10)
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_dwt_estimator_edge_cases(self):
        """Test DWT estimator with edge cases."""
        estimator = DWTEstimator()
        
        # Test with very short data
        data = np.random.randn(10)
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_abry_veitch_estimator_edge_cases(self):
        """Test Abry-Veitch estimator with edge cases."""
        estimator = AbryVeitchEstimator()
        
        # Test with very short data
        data = np.random.randn(10)
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_mfdfa_estimator_edge_cases(self):
        """Test MFDFA estimator with edge cases."""
        estimator = MFDFAEstimator()
        
        # Test with very short data
        data = np.random.randn(10)
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)


class TestBayesianHurstEstimator:
    """Test BayesianHurstEstimator for better coverage."""
    
    def test_bayesian_confidence_basic(self):
        """Test basic Bayesian confidence estimation."""
        # Mock the estimator
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = (0.6, {})
        
        result = BayesianHurstEstimator.bayesian_confidence(
            mock_estimator,
            data=np.random.randn(1000),
            estimator_type=EstimatorType.DFA,
            num_samples=10
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert np.isfinite(result[0])  # mean
        assert len(result[1]) == 2    # credible interval
        assert isinstance(result[2], dict)  # additional results
    
    def test_bayesian_confidence_with_parameters(self):
        """Test Bayesian confidence with custom parameters."""
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = (0.6, {})
        
        result = BayesianHurstEstimator.bayesian_confidence(
            mock_estimator,
            data=np.random.randn(1000),
            estimator_type=EstimatorType.DFA,
            num_samples=5,
            burn_in=2,
            thin=1
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestFactoryAdvancedFeatures:
    """Test advanced factory features for better coverage."""
    
    def test_factory_with_custom_parameters(self):
        """Test factory with custom parameters."""
        factory = BiomedicalHurstEstimatorFactory(
            quality_threshold=0.5,
            confidence_level=0.99,
            confidence_method=ConfidenceMethod.THEORETICAL
        )
        
        data = np.random.randn(1000)
        result = factory.estimate(data, method='dfa')
        
        assert isinstance(result, HurstResult)
        assert result.confidence_level == 0.99
        assert result.confidence_method == ConfidenceMethod.THEORETICAL.value
    
    def test_factory_estimate_with_all_parameters(self):
        """Test factory estimate with all possible parameters."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        result = factory.estimate(
            data,
            method='dfa',
            confidence_level=0.90,
            confidence_method=ConfidenceMethod.BOOTSTRAP,
            quality_threshold=0.3,
            min_scale=20,
            max_scale=200,
            polynomial_order=2,
            seed=42
        )
        
        assert isinstance(result, HurstResult)
        assert result.confidence_level == 0.90
        assert result.confidence_method == ConfidenceMethod.BOOTSTRAP.value
    
    def test_factory_estimate_with_spectral_methods(self):
        """Test factory with spectral methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        spectral_methods = ['periodogram', 'gph', 'whittle']
        for method in spectral_methods:
            result = factory.estimate(data, method=method)
            assert isinstance(result, HurstResult)
            assert np.isfinite(result.hurst_estimate)
    
    def test_factory_estimate_with_wavelet_methods(self):
        """Test factory with wavelet methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        wavelet_methods = ['dwt', 'abry_veitch']
        for method in wavelet_methods:
            result = factory.estimate(data, method=method)
            assert isinstance(result, HurstResult)
            assert np.isfinite(result.hurst_estimate)
    
    def test_factory_estimate_with_multifractal_methods(self):
        """Test factory with multifractal methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        result = factory.estimate(data, method='mfdfa')
        assert isinstance(result, HurstResult)
        assert np.isfinite(result.hurst_estimate)
    
    def test_factory_estimate_with_different_confidence_methods(self):
        """Test factory with different confidence methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        confidence_methods = [
            ConfidenceMethod.BOOTSTRAP,
            ConfidenceMethod.THEORETICAL,
            ConfidenceMethod.BAYESIAN
        ]
        
        for method in confidence_methods:
            result = factory.estimate(
                data,
                method='dfa',
                confidence_method=method
            )
            assert isinstance(result, HurstResult)
            assert result.confidence_method == method.value
    
    def test_factory_estimate_with_quality_assessment(self):
        """Test factory with quality assessment."""
        factory = BiomedicalHurstEstimatorFactory(quality_threshold=0.5)
        data = np.random.randn(1000)
        
        result = factory.estimate(data, method='dfa')
        assert isinstance(result, HurstResult)
        assert hasattr(result, 'data_quality_score')
        assert result.data_quality_score >= 0.0
    
    def test_factory_estimate_with_artifacts(self):
        """Test factory with data containing artifacts."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        # Add artifacts
        data[100:150] += 5.0  # Eye movements
        data[300:310] += 10.0  # Muscle artifacts
        
        result = factory.estimate(data, method='dfa')
        assert isinstance(result, HurstResult)
        assert hasattr(result, 'additional_metrics')
        assert 'artifact_detection' in result.additional_metrics


class TestErrorHandling:
    """Test error handling for better coverage."""
    
    def test_factory_estimate_with_invalid_method(self):
        """Test factory with invalid method."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        with pytest.raises(ValueError):
            factory.estimate(data, method='invalid_method')
    
    def test_factory_estimate_with_none_data(self):
        """Test factory with None data."""
        factory = BiomedicalHurstEstimatorFactory()
        
        with pytest.raises((TypeError, ValueError)):
            factory.estimate(None)
    
    def test_factory_estimate_with_non_numeric_data(self):
        """Test factory with non-numeric data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = ['a', 'b', 'c']  # Non-numeric data
        
        with pytest.raises((TypeError, ValueError)):
            factory.estimate(data)
    
    def test_factory_estimate_with_infinite_data(self):
        """Test factory with infinite data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        data[100:110] = np.inf
        
        result = factory.estimate(data, method='dfa')
        assert isinstance(result, HurstResult)
        # Should handle gracefully
    
    def test_factory_estimate_with_very_large_data(self):
        """Test factory with very large data."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(10000)  # Large dataset
        
        result = factory.estimate(data, method='dfa')
        assert isinstance(result, HurstResult)
        assert np.isfinite(result.hurst_estimate)


class TestPerformanceAndEdgeCases:
    """Test performance and edge cases for better coverage."""
    
    def test_factory_estimate_performance(self):
        """Test factory estimation performance."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        import time
        start_time = time.time()
        result = factory.estimate(data, method='dfa')
        end_time = time.time()
        
        assert isinstance(result, HurstResult)
        assert result.computation_time > 0.0
        assert (end_time - start_time) > 0.0
    
    def test_factory_estimate_with_different_data_sizes(self):
        """Test factory with different data sizes."""
        factory = BiomedicalHurstEstimatorFactory()
        
        sizes = [100, 500, 1000, 2000]
        for size in sizes:
            data = np.random.randn(size)
            result = factory.estimate(data, method='dfa')
            assert isinstance(result, HurstResult)
            assert np.isfinite(result.hurst_estimate)
    
    def test_factory_estimate_with_different_hurst_values(self):
        """Test factory with different underlying Hurst values."""
        factory = BiomedicalHurstEstimatorFactory()
        
        # Generate data with different Hurst values
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        for h in hurst_values:
            # Generate fractional Brownian motion-like data
            data = np.cumsum(np.random.randn(1000)) * (h - 0.5)
            result = factory.estimate(data, method='dfa')
            assert isinstance(result, HurstResult)
            assert np.isfinite(result.hurst_estimate)
    
    def test_factory_estimate_reproducibility(self):
        """Test factory estimation reproducibility."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        # Run estimation twice with same seed
        result1 = factory.estimate(data, method='dfa', seed=42)
        result2 = factory.estimate(data, method='dfa', seed=42)
        
        assert isinstance(result1, HurstResult)
        assert isinstance(result2, HurstResult)
        # Results should be similar (allowing for small numerical differences)
        assert abs(result1.hurst_estimate - result2.hurst_estimate) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
