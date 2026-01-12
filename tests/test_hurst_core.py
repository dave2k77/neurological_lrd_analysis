import numpy as np
import pytest
from typing import Tuple, Dict, Any, Optional

from neurological_lrd_analysis.biomedical_hurst_factory import (
    BiomedicalHurstEstimatorFactory,
    HurstResult,
    EstimatorType,
    ConfidenceMethod
)

class TestHurstResult:
    """Test HurstResult dataclass initialization and methods."""
    
    def test_hurst_result_full_initialization(self):
        """Test HurstResult with all fields."""
        result = HurstResult(
            hurst_estimate=0.6,
            estimator_name="DFA",
            confidence_interval=(0.55, 0.65),
            confidence_level=0.95,
            confidence_method="bootstrap",
            standard_error=0.02,
            bias_estimate=0.01,
            variance_estimate=0.0004,
            bootstrap_samples=np.array([0.58, 0.6, 0.62]),
            computation_time=0.1,
            memory_usage=10.5,
            convergence_flag=True,
            data_quality_score=0.9,
            missing_data_fraction=0.0,
            outlier_fraction=0.0,
            stationarity_p_value=0.8,
            regression_r_squared=0.99,
            scaling_range=(10, 100),
            goodness_of_fit=0.98,
            signal_to_noise_ratio=20.0,
            artifact_detection={"electrode_pop": False},
            additional_metrics={"custom_metric": 1.23}
        )
        
        assert result.hurst_estimate == 0.6
        assert result.estimator_name == "DFA"
        assert result.convergence_flag is True
        assert str(result).startswith("\nHurst Exponent Analysis Results")

class TestBiomedicalHurstEstimatorFactory:
    """Test BiomedicalHurstEstimatorFactory functionality."""
    
    @pytest.fixture
    def factory(self):
        return BiomedicalHurstEstimatorFactory()
        
    @pytest.fixture
    def sample_data(self):
        # Generate some synthetic fBm-like data (integrated white noise for simplicity in basic tests)
        np.random.seed(42)
        return np.cumsum(np.random.randn(500))

    def test_factory_creation(self, factory):
        assert factory is not None
        
    def test_estimate_dfa(self, factory, sample_data):
        result = factory.estimate(sample_data, method=EstimatorType.DFA)
        assert isinstance(result, HurstResult)
        assert result.estimator_name == "DFA"
        assert result.convergence_flag
        assert not np.isnan(result.hurst_estimate)
        assert "Convergence: Success" in str(result)
        
    def test_available_methods(self, factory):
        assert EstimatorType.DFA in factory.estimators
        assert EstimatorType.HIGUCHI in factory.estimators
        assert EstimatorType.TEMPORAL in factory.groups
        
    def test_error_handling_short_data(self, factory):
        short_data = np.array([1.0, 2.0])
        result = factory.estimate(short_data, method=EstimatorType.DFA)
        assert result.convergence_flag is False
        assert np.isnan(result.hurst_estimate)
        assert "Convergence: Failed" in str(result)
