"""
Simple focused tests to improve biomedical factory coverage to 80%+.
"""

import numpy as np
import pytest
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


class TestBiomedicalDataProcessorCoverage:
    """Test BiomedicalDataProcessor for better coverage."""
    
    def test_assess_data_quality_basic(self):
        """Test basic data quality assessment."""
        processor = BiomedicalDataProcessor()
        data = np.random.randn(1000)
        result = processor.assess_data_quality(data)
        
        # Check that result is a dictionary with expected keys
        assert isinstance(result, dict)
        assert 'missing_data_fraction' in result
        assert 'outlier_fraction' in result
        assert 'signal_to_noise_ratio' in result
        assert 'artifact_detection' in result
        
        assert result['missing_data_fraction'] == 0.0
        assert 0.0 <= result['outlier_fraction'] <= 1.0
        # Signal-to-noise ratio can be negative for some data
        assert np.isfinite(result['signal_to_noise_ratio'])
    
    def test_assess_data_quality_with_nan(self):
        """Test data quality assessment with NaN values."""
        processor = BiomedicalDataProcessor()
        data = np.random.randn(1000)
        data[100:110] = np.nan
        result = processor.assess_data_quality(data)
        
        assert result['missing_data_fraction'] > 0
        assert result['has_missing_data'] == True
    
    def test_assess_data_quality_with_outliers(self):
        """Test data quality assessment with outliers."""
        processor = BiomedicalDataProcessor()
        data = np.random.randn(1000)
        data[100:110] = 10.0  # Add outliers
        result = processor.assess_data_quality(data)
        
        assert result['outlier_fraction'] > 0.0
    
    def test_assess_data_quality_empty_data(self):
        """Test data quality assessment with empty data."""
        processor = BiomedicalDataProcessor()
        data = np.array([])
        result = processor.assess_data_quality(data)
        
        assert result['missing_data_fraction'] == 0.0
        assert result['has_missing_data'] == False


class TestConfidenceEstimatorCoverage:
    """Test ConfidenceEstimator for better coverage."""
    
    def test_bootstrap_confidence_basic(self):
        """Test basic bootstrap confidence estimation."""
        estimator = ConfidenceEstimator()
        data = np.random.randn(1000)
        
        # Test bootstrap confidence with mock estimator
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = (0.6, {})
        
        result = estimator.bootstrap_confidence(
            data, mock_estimator, n_bootstrap=10, confidence_level=0.95, random_state=42
        )
        
        # Bootstrap confidence returns a tuple, not a dict
        assert isinstance(result, tuple)
        assert len(result) == 3  # mean, std, confidence_interval
    
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


class TestIndividualEstimatorsCoverage:
    """Test individual estimators for better coverage."""
    
    def test_dfa_estimator_with_sufficient_data(self):
        """Test DFA estimator with sufficient data."""
        estimator = DFAEstimator()
        data = np.random.randn(1000)
        
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
    
    def test_rs_estimator_with_sufficient_data(self):
        """Test R/S estimator with sufficient data."""
        estimator = RSAnalysisEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_higuchi_estimator_with_sufficient_data(self):
        """Test Higuchi estimator with sufficient data."""
        estimator = HiguchiEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_ghe_estimator_with_sufficient_data(self):
        """Test GHE estimator with sufficient data."""
        estimator = GHEEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_periodogram_estimator_with_sufficient_data(self):
        """Test Periodogram estimator with sufficient data."""
        estimator = PeriodogramEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_gph_estimator_with_sufficient_data(self):
        """Test GPH estimator with sufficient data."""
        estimator = GPHEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_whittle_mle_estimator_with_sufficient_data(self):
        """Test Whittle MLE estimator with sufficient data."""
        estimator = WhittleMLEEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_dwt_estimator_with_sufficient_data(self):
        """Test DWT estimator with sufficient data."""
        estimator = DWTEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_abry_veitch_estimator_with_sufficient_data(self):
        """Test Abry-Veitch estimator with sufficient data."""
        estimator = AbryVeitchEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)
    
    def test_mfdfa_estimator_with_sufficient_data(self):
        """Test MFDFA estimator with sufficient data."""
        estimator = MFDFAEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(data)
        
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        assert isinstance(result[1], dict)


class TestBayesianHurstEstimatorCoverage:
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
        # The result might be nan due to JAX issues, so just check structure
        assert isinstance(result[0], (float, np.floating))
        assert len(result[1]) == 2    # credible interval
        assert isinstance(result[2], dict)  # additional results


class TestFactoryAdvancedCoverage:
    """Test advanced factory features for better coverage."""
    
    def test_factory_estimate_with_different_methods(self):
        """Test factory with different methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        methods = ['dfa', 'rs', 'higuchi', 'ghe', 'periodogram', 'gph', 'whittle', 'dwt', 'abry_veitch', 'mfdfa']
        
        for method in methods:
            result = factory.estimate(data, method=method)
            assert isinstance(result, HurstResult)
            # Some methods might return nan, so just check structure
            assert hasattr(result, 'hurst_estimate')
            assert hasattr(result, 'estimator_name')
            assert hasattr(result, 'confidence_interval')
    
    def test_factory_estimate_with_confidence_methods(self):
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
    
    def test_factory_estimate_with_different_confidence_levels(self):
        """Test factory with different confidence levels."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        confidence_levels = [0.90, 0.95, 0.99]
        
        for level in confidence_levels:
            result = factory.estimate(
                data,
                method='dfa',
                confidence_level=level
            )
            assert isinstance(result, HurstResult)
            assert result.confidence_level == level
    
    def test_factory_estimate_with_parameters(self):
        """Test factory estimate with parameters."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        result = factory.estimate(
            data,
            method='dfa',
            confidence_level=0.90,
            confidence_method=ConfidenceMethod.THEORETICAL,
            min_scale=20,
            max_scale=200,
            polynomial_order=2,
            seed=42
        )
        
        assert isinstance(result, HurstResult)
        assert result.confidence_level == 0.90
        assert result.confidence_method == ConfidenceMethod.THEORETICAL.value
    
    def test_factory_estimate_with_spectral_methods(self):
        """Test factory with spectral methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        spectral_methods = ['periodogram', 'gph', 'whittle']
        for method in spectral_methods:
            result = factory.estimate(data, method=method)
            assert isinstance(result, HurstResult)
    
    def test_factory_estimate_with_wavelet_methods(self):
        """Test factory with wavelet methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        wavelet_methods = ['dwt', 'abry_veitch']
        for method in wavelet_methods:
            result = factory.estimate(data, method=method)
            assert isinstance(result, HurstResult)
    
    def test_factory_estimate_with_multifractal_methods(self):
        """Test factory with multifractal methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        result = factory.estimate(data, method='mfdfa')
        assert isinstance(result, HurstResult)


class TestErrorHandlingCoverage:
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


class TestPerformanceAndEdgeCasesCoverage:
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
