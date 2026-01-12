"""
Additional tests to improve biomedical factory coverage to 80%+.
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


class TestAdditionalCoverage:
    """Additional tests for better coverage."""
    
    def test_biomedical_data_processor_edge_cases(self):
        """Test BiomedicalDataProcessor edge cases."""
        processor = BiomedicalDataProcessor()
        
        # Test with all zeros
        data = np.zeros(1000)
        result = processor.assess_data_quality(data)
        assert isinstance(result, dict)
        
        # Test with all ones
        data = np.ones(1000)
        result = processor.assess_data_quality(data)
        assert isinstance(result, dict)
        
        # Test with constant data
        data = np.full(1000, 5.0)
        result = processor.assess_data_quality(data)
        assert isinstance(result, dict)
    
    def test_confidence_estimator_edge_cases(self):
        """Test ConfidenceEstimator edge cases."""
        estimator = ConfidenceEstimator()
        
        # Test theoretical confidence with edge values
        result = estimator.theoretical_confidence(
            hurst_estimate=0.0,
            standard_error=0.0,
            n_samples=1,
            confidence_level=0.95
        )
        assert len(result) == 2
        
        result = estimator.theoretical_confidence(
            hurst_estimate=1.0,
            standard_error=0.0,
            n_samples=1,
            confidence_level=0.95
        )
        assert len(result) == 2
    
    def test_individual_estimators_with_edge_cases(self):
        """Test individual estimators with edge cases."""
        # Test DFA with edge parameters
        estimator = DFAEstimator()
        data = np.random.randn(1000)
        
        result = estimator.estimate(
            data,
            min_scale=10,
            max_scale=50,
            polynomial_order=1
        )
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        
        # Test R/S with edge parameters
        estimator = RSAnalysisEstimator()
        result = estimator.estimate(
            data,
            min_scale=10,
            max_scale=50
        )
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        
        # Test Higuchi with edge parameters
        estimator = HiguchiEstimator()
        result = estimator.estimate(
            data,
            k_max=20
        )
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        
        # Test GHE with edge parameters
        estimator = GHEEstimator()
        result = estimator.estimate(
            data,
            q_values=[1, 2, 3]
        )
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        
        # Test Periodogram with edge parameters
        estimator = PeriodogramEstimator()
        result = estimator.estimate(
            data,
            low_freq_fraction=0.1,
            high_freq_cutoff=0.5
        )
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        
        # Test GPH with edge parameters
        estimator = GPHEstimator()
        result = estimator.estimate(
            data,
            low_freq_fraction=0.1,
            high_freq_cutoff=0.5
        )
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        
        # Test Whittle MLE with edge parameters
        estimator = WhittleMLEEstimator()
        result = estimator.estimate(
            data,
            low_freq_fraction=0.1,
            high_freq_cutoff=0.5
        )
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        
        # Test DWT with edge parameters
        estimator = DWTEstimator()
        result = estimator.estimate(
            data,
            wavelet='db4',
            max_level=5
        )
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
        
        # Test Abry-Veitch with edge parameters
        estimator = AbryVeitchEstimator()
        result = estimator.estimate(
            data,
            min_scale=10,
            max_scale=50
        )
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])
    
    def test_factory_with_edge_cases(self):
        """Test factory with edge cases."""
        factory = BiomedicalHurstEstimatorFactory()
        
        # Test with very short data
        data = np.random.randn(50)
        result = factory.estimate(data, method='dfa')
        assert isinstance(result, HurstResult)
        
        # Test with data containing NaN
        data = np.random.randn(1000)
        data[100:110] = np.nan
        result = factory.estimate(data, method='dfa')
        assert isinstance(result, HurstResult)
        
        # Test with data containing Inf
        data = np.random.randn(1000)
        data[100:110] = np.inf
        result = factory.estimate(data, method='dfa')
        assert isinstance(result, HurstResult)
        
        # Test with constant data
        data = np.ones(1000)
        result = factory.estimate(data, method='dfa')
        assert isinstance(result, HurstResult)
        
        # Test with zero variance data
        data = np.zeros(1000)
        result = factory.estimate(data, method='dfa')
        assert isinstance(result, HurstResult)
    
    def test_factory_with_different_confidence_levels(self):
        """Test factory with different confidence levels."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        confidence_levels = [0.80, 0.90, 0.95, 0.99]
        for level in confidence_levels:
            result = factory.estimate(
                data,
                method='dfa',
                confidence_level=level
            )
            assert isinstance(result, HurstResult)
            assert result.confidence_level == level
    
    def test_factory_with_different_confidence_methods(self):
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
    
    def test_factory_with_different_methods(self):
        """Test factory with different methods."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        methods = [
            'dfa', 'rs', 'higuchi', 'ghe', 
            'periodogram', 'gph', 'whittle', 
            'dwt', 'abry_veitch'
        ]
        
        for method in methods:
            result = factory.estimate(data, method=method)
            assert isinstance(result, HurstResult)
            assert hasattr(result, 'hurst_estimate')
            assert hasattr(result, 'estimator_name')
            assert hasattr(result, 'confidence_interval')
    
    def test_factory_with_parameters(self):
        """Test factory with various parameters."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        # Test with DFA parameters
        result = factory.estimate(
            data,
            method='dfa',
            min_scale=20,
            max_scale=200,
            polynomial_order=2,
            seed=42
        )
        assert isinstance(result, HurstResult)
        
        # Test with R/S parameters
        result = factory.estimate(
            data,
            method='rs',
            min_scale=20,
            max_scale=200,
            seed=42
        )
        assert isinstance(result, HurstResult)
        
        # Test with Higuchi parameters
        result = factory.estimate(
            data,
            method='higuchi',
            k_max=20,
            seed=42
        )
        assert isinstance(result, HurstResult)
        
        # Test with GHE parameters
        result = factory.estimate(
            data,
            method='ghe',
            q_values=[1, 2, 3],
            seed=42
        )
        assert isinstance(result, HurstResult)
        
        # Test with Periodogram parameters
        result = factory.estimate(
            data,
            method='periodogram',
            low_freq_fraction=0.1,
            high_freq_cutoff=0.5,
            seed=42
        )
        assert isinstance(result, HurstResult)
        
        # Test with GPH parameters
        result = factory.estimate(
            data,
            method='gph',
            low_freq_fraction=0.1,
            high_freq_cutoff=0.5,
            seed=42
        )
        assert isinstance(result, HurstResult)
        
        # Test with Whittle parameters
        result = factory.estimate(
            data,
            method='whittle',
            low_freq_fraction=0.1,
            high_freq_cutoff=0.5,
            seed=42
        )
        assert isinstance(result, HurstResult)
        
        # Test with DWT parameters
        result = factory.estimate(
            data,
            method='dwt',
            wavelet='db4',
            max_level=5,
            seed=42
        )
        assert isinstance(result, HurstResult)
        
        # Test with Abry-Veitch parameters
        result = factory.estimate(
            data,
            method='abry_veitch',
            min_scale=20,
            max_scale=200,
            seed=42
        )
        assert isinstance(result, HurstResult)
    
    def test_factory_error_handling(self):
        """Test factory error handling."""
        factory = BiomedicalHurstEstimatorFactory()
        
        # Test with invalid method
        data = np.random.randn(1000)
        with pytest.raises(ValueError):
            factory.estimate(data, method='invalid_method')
        
        # Test with None data
        with pytest.raises((TypeError, ValueError)):
            factory.estimate(None)
        
        # Test with non-numeric data
        with pytest.raises((TypeError, ValueError)):
            factory.estimate(['a', 'b', 'c'])
        
        # Test with empty data
        with pytest.raises((ValueError, TypeError)):
            factory.estimate(np.array([]))
    
    def test_factory_performance(self):
        """Test factory performance."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        import time
        start_time = time.time()
        result = factory.estimate(data, method='dfa')
        end_time = time.time()
        
        assert isinstance(result, HurstResult)
        assert result.computation_time > 0.0
        assert (end_time - start_time) > 0.0
    
    def test_factory_reproducibility(self):
        """Test factory reproducibility."""
        factory = BiomedicalHurstEstimatorFactory()
        data = np.random.randn(1000)
        
        # Test reproducibility with same seed
        result1 = factory.estimate(data, method='dfa', seed=42)
        result2 = factory.estimate(data, method='dfa', seed=42)
        
        assert isinstance(result1, HurstResult)
        assert isinstance(result2, HurstResult)
        # Results should be similar
        assert abs(result1.hurst_estimate - result2.hurst_estimate) < 0.1
    
    def test_factory_with_different_data_sizes(self):
        """Test factory with different data sizes."""
        factory = BiomedicalHurstEstimatorFactory()
        
        sizes = [100, 500, 1000, 2000]
        for size in sizes:
            data = np.random.randn(size)
            result = factory.estimate(data, method='dfa')
            assert isinstance(result, HurstResult)
    
    def test_factory_with_different_hurst_values(self):
        """Test factory with different underlying Hurst values."""
        factory = BiomedicalHurstEstimatorFactory()
        
        # Generate data with different Hurst values
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        for h in hurst_values:
            # Generate fractional Brownian motion-like data
            data = np.cumsum(np.random.randn(1000)) * (h - 0.5)
            result = factory.estimate(data, method='dfa')
            assert isinstance(result, HurstResult)
    
    def test_bayesian_estimator_edge_cases(self):
        """Test BayesianHurstEstimator edge cases."""
        # Mock the estimator
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = (0.6, {})
        
        # Test with different parameters
        result = BayesianHurstEstimator.bayesian_confidence(
            mock_estimator,
            data=np.random.randn(1000),
            estimator_type=EstimatorType.DFA,
            num_samples=5
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_individual_estimators_validation(self):
        """Test individual estimators validation."""
        # Test DFA with insufficient data
        estimator = DFAEstimator()
        with pytest.raises(ValueError):
            estimator.estimate(np.random.randn(10))
        
        # Test R/S with insufficient data
        estimator = RSAnalysisEstimator()
        with pytest.raises(ValueError):
            estimator.estimate(np.random.randn(10))
        
        # Test Higuchi with insufficient data
        estimator = HiguchiEstimator()
        with pytest.raises(ValueError):
            estimator.estimate(np.random.randn(10))
        
        # Test GHE with insufficient data
        estimator = GHEEstimator()
        with pytest.raises(ValueError):
            estimator.estimate(np.random.randn(10))
        
        # Test Periodogram with insufficient data
        estimator = PeriodogramEstimator()
        with pytest.raises(ValueError):
            estimator.estimate(np.random.randn(10))
        
        # Test GPH with insufficient data
        estimator = GPHEstimator()
        with pytest.raises(ValueError):
            estimator.estimate(np.random.randn(10))
        
        # Test Whittle MLE with insufficient data
        estimator = WhittleMLEEstimator()
        with pytest.raises(ValueError):
            estimator.estimate(np.random.randn(10))
        
        # Test DWT with insufficient data
        estimator = DWTEstimator()
        with pytest.raises(ValueError):
            estimator.estimate(np.random.randn(10))
        
        # Test Abry-Veitch with insufficient data
        estimator = AbryVeitchEstimator()
        with pytest.raises(ValueError):
            estimator.estimate(np.random.randn(10))
        
        # Test MFDFA with insufficient data
        estimator = MFDFAEstimator()
        with pytest.raises(ValueError):
            estimator.estimate(np.random.randn(10))


if __name__ == "__main__":
    pytest.main([__file__])
