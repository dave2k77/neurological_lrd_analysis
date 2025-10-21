"""
Comprehensive tests for biomedical scenarios module.

This module tests all biomedical scenario generation functions,
artifact addition, and grid generation functionality.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from neurological_lrd_analysis.benchmark_core.biomedical_scenarios import (
    BiomedicalScenario,
    generate_eeg_scenario,
    generate_ecg_scenario,
    generate_respiratory_scenario,
    add_eeg_artifacts,
    add_ecg_artifacts,
    add_respiratory_artifacts,
    generate_biomedical_scenario,
    generate_biomedical_grid
)


class TestBiomedicalScenario:
    """Test BiomedicalScenario dataclass."""
    
    def test_biomedical_scenario_creation(self):
        """Test creating a BiomedicalScenario instance."""
        scenario = BiomedicalScenario(
            scenario_type='eeg',
            sampling_rate=250.0,
            duration=10.0,
            hurst_range=(0.3, 0.7),
            typical_amplitude=50.0,
            noise_level=0.1,
            artifact_probability=0.05
        )
        
        assert scenario.scenario_type == 'eeg'
        assert scenario.sampling_rate == 250.0
        assert scenario.duration == 10.0
        assert scenario.hurst_range == (0.3, 0.7)
        assert scenario.typical_amplitude == 50.0
        assert scenario.noise_level == 0.1
        assert scenario.artifact_probability == 0.05


class TestEEGScenario:
    """Test EEG scenario generation."""
    
    def test_generate_eeg_scenario_basic(self):
        """Test basic EEG scenario generation."""
        data = generate_eeg_scenario(
            n=1000,
            hurst=0.5,
            scenario="rest",
            contamination_level=0.1,
            seed=42
        )
        
        assert len(data) == 1000
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))
    
    def test_generate_eeg_scenario_different_scenarios(self):
        """Test EEG generation for different scenarios."""
        scenarios = ["rest", "eyes_closed", "eyes_open", "sleep", "seizure"]
        
        for scenario in scenarios:
            data = generate_eeg_scenario(
                n=500,
                hurst=0.6,
                scenario=scenario,
                contamination_level=0.2,
                seed=42
            )
            
            assert len(data) == 500
            assert isinstance(data, np.ndarray)
            assert not np.any(np.isnan(data))
    
    def test_generate_eeg_scenario_different_hurst(self):
        """Test EEG generation for different Hurst values."""
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        
        for hurst in hurst_values:
            data = generate_eeg_scenario(
                n=500,
                hurst=hurst,
                scenario="rest",
                contamination_level=0.1,
                seed=42
            )
            
            assert len(data) == 500
            assert isinstance(data, np.ndarray)
    
    def test_generate_eeg_scenario_different_contamination(self):
        """Test EEG generation with different contamination levels."""
        contamination_levels = [0.0, 0.1, 0.5, 1.0]
        
        for contamination in contamination_levels:
            data = generate_eeg_scenario(
                n=500,
                hurst=0.5,
                scenario="rest",
                contamination_level=contamination,
                seed=42
            )
            
            assert len(data) == 500
            assert isinstance(data, np.ndarray)
    
    def test_generate_eeg_scenario_reproducibility(self):
        """Test EEG generation reproducibility with same seed."""
        data1 = generate_eeg_scenario(
            n=500,
            hurst=0.5,
            scenario="rest",
            contamination_level=0.1,
            seed=42
        )
        
        data2 = generate_eeg_scenario(
            n=500,
            hurst=0.5,
            scenario="rest",
            contamination_level=0.1,
            seed=42
        )
        
        np.testing.assert_array_equal(data1, data2)
    
    def test_generate_eeg_scenario_edge_cases(self):
        """Test EEG generation edge cases."""
        # Very short data
        data = generate_eeg_scenario(n=10, hurst=0.5, scenario="rest", seed=42)
        assert len(data) == 10
        
        # Extreme Hurst values
        data = generate_eeg_scenario(n=100, hurst=0.01, scenario="rest", seed=42)
        assert len(data) == 100
        
        data = generate_eeg_scenario(n=100, hurst=0.99, scenario="rest", seed=42)
        assert len(data) == 100


class TestECGScenario:
    """Test ECG scenario generation."""
    
    def test_generate_ecg_scenario_basic(self):
        """Test basic ECG scenario generation."""
        data = generate_ecg_scenario(
            n=1000,
            hurst=0.5,
            contamination_level=0.1,
            seed=42
        )
        
        assert len(data) == 1000
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))
    
    def test_generate_ecg_scenario_different_scenarios(self):
        """Test ECG generation for different scenarios."""
        scenarios = ["normal", "tachycardia", "bradycardia", "arrhythmia"]
        
        for scenario in scenarios:
            data = generate_ecg_scenario(
                n=500,
                hurst=0.6,
                contamination_level=0.2,
                seed=42
            )
            
            assert len(data) == 500
            assert isinstance(data, np.ndarray)
            assert not np.any(np.isnan(data))
    
    def test_generate_ecg_scenario_different_hurst(self):
        """Test ECG generation for different Hurst values."""
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        
        for hurst in hurst_values:
            data = generate_ecg_scenario(
                n=500,
                hurst=hurst,
                contamination_level=0.1,
                seed=42
            )
            
            assert len(data) == 500
            assert isinstance(data, np.ndarray)
    
    def test_generate_ecg_scenario_reproducibility(self):
        """Test ECG generation reproducibility with same seed."""
        data1 = generate_ecg_scenario(
            n=500,
            hurst=0.5,
            contamination_level=0.1,
            seed=42
        )
        
        data2 = generate_ecg_scenario(
            n=500,
            hurst=0.5,
            contamination_level=0.1,
            seed=42
        )
        
        np.testing.assert_array_equal(data1, data2)


class TestRespiratoryScenario:
    """Test respiratory scenario generation."""
    
    def test_generate_respiratory_scenario_basic(self):
        """Test basic respiratory scenario generation."""
        data = generate_respiratory_scenario(
            n=1000,
            hurst=0.5,
            contamination_level=0.1,
            seed=42
        )
        
        assert len(data) == 1000
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))
    
    def test_generate_respiratory_scenario_different_scenarios(self):
        """Test respiratory generation for different scenarios."""
        scenarios = ["normal", "shallow", "deep", "irregular"]
        
        for scenario in scenarios:
            data = generate_respiratory_scenario(
                n=500,
                hurst=0.6,
                contamination_level=0.2,
                seed=42
            )
            
            assert len(data) == 500
            assert isinstance(data, np.ndarray)
            assert not np.any(np.isnan(data))
    
    def test_generate_respiratory_scenario_reproducibility(self):
        """Test respiratory generation reproducibility with same seed."""
        data1 = generate_respiratory_scenario(
            n=500,
            hurst=0.5,
            contamination_level=0.1,
            seed=42
        )
        
        data2 = generate_respiratory_scenario(
            n=500,
            hurst=0.5,
            contamination_level=0.1,
            seed=42
        )
        
        np.testing.assert_array_equal(data1, data2)


class TestArtifactAddition:
    """Test artifact addition functions."""
    
    def test_add_eeg_artifacts(self):
        """Test EEG artifact addition."""
        # Create base signal
        base_signal = np.random.randn(1000)
        
        # Add artifacts
        contaminated_signal = add_eeg_artifacts(
            base_signal,
            contamination_level=0.2,
            seed=42
        )
        
        assert len(contaminated_signal) == len(base_signal)
        assert isinstance(contaminated_signal, np.ndarray)
        assert not np.any(np.isnan(contaminated_signal))
    
    def test_add_eeg_artifacts_different_levels(self):
        """Test EEG artifact addition with different contamination levels."""
        base_signal = np.random.randn(500)
        
        for level in [0.0, 0.1, 0.5, 1.0]:
            contaminated = add_eeg_artifacts(
                base_signal,
                contamination_level=level,
                seed=42
            )
            
            assert len(contaminated) == len(base_signal)
            assert isinstance(contaminated, np.ndarray)
    
    def test_add_ecg_artifacts(self):
        """Test ECG artifact addition."""
        base_signal = np.random.randn(1000)
        
        contaminated_signal = add_ecg_artifacts(
            base_signal,
            contamination_level=0.2,
            seed=42
        )
        
        assert len(contaminated_signal) == len(base_signal)
        assert isinstance(contaminated_signal, np.ndarray)
        assert not np.any(np.isnan(contaminated_signal))
    
    def test_add_respiratory_artifacts(self):
        """Test respiratory artifact addition."""
        base_signal = np.random.randn(1000)
        
        contaminated_signal = add_respiratory_artifacts(
            base_signal,
            contamination_level=0.2,
            seed=42
        )
        
        assert len(contaminated_signal) == len(base_signal)
        assert isinstance(contaminated_signal, np.ndarray)
        assert not np.any(np.isnan(contaminated_signal))
    
    def test_artifact_addition_reproducibility(self):
        """Test artifact addition reproducibility."""
        base_signal = np.random.randn(500)
        
        # Test EEG artifacts
        contaminated1 = add_eeg_artifacts(base_signal, contamination_level=0.3, seed=42)
        contaminated2 = add_eeg_artifacts(base_signal, contamination_level=0.3, seed=42)
        np.testing.assert_array_equal(contaminated1, contaminated2)
        
        # Test ECG artifacts
        contaminated1 = add_ecg_artifacts(base_signal, contamination_level=0.3, seed=42)
        contaminated2 = add_ecg_artifacts(base_signal, contamination_level=0.3, seed=42)
        np.testing.assert_array_equal(contaminated1, contaminated2)
        
        # Test respiratory artifacts
        contaminated1 = add_respiratory_artifacts(base_signal, contamination_level=0.3, seed=42)
        contaminated2 = add_respiratory_artifacts(base_signal, contamination_level=0.3, seed=42)
        np.testing.assert_array_equal(contaminated1, contaminated2)


class TestBiomedicalScenarioGeneration:
    """Test biomedical scenario generation."""
    
    def test_generate_biomedical_scenario_eeg(self):
        """Test biomedical scenario generation for EEG."""
        data = generate_biomedical_scenario(
            n=1000,
            hurst=0.5,
            scenario_type="eeg",
            scenario="rest",
            contamination_level=0.1,
            seed=42
        )
        
        assert len(data) == 1000
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
    
    def test_generate_biomedical_scenario_ecg(self):
        """Test biomedical scenario generation for ECG."""
        data = generate_biomedical_scenario(
            n=1000,
            hurst=0.5,
            scenario_type="ecg",
            scenario="normal",
            contamination_level=0.1,
            seed=42
        )
        
        assert len(data) == 1000
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
    
    def test_generate_biomedical_scenario_respiratory(self):
        """Test biomedical scenario generation for respiratory."""
        data = generate_biomedical_scenario(
            n=1000,
            hurst=0.5,
            scenario_type="respiratory",
            scenario="normal",
            contamination_level=0.1,
            seed=42
        )
        
        assert len(data) == 1000
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
    
    def test_generate_biomedical_scenario_different_hurst(self):
        """Test biomedical scenario generation for different Hurst values."""
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        
        for hurst in hurst_values:
            data = generate_biomedical_scenario(
                n=500,
                hurst=hurst,
                scenario_type="eeg",
                scenario="rest",
                contamination_level=0.1,
                seed=42
            )
            
            assert len(data) == 500
            assert isinstance(data, np.ndarray)
    
    def test_generate_biomedical_scenario_reproducibility(self):
        """Test biomedical scenario generation reproducibility."""
        data1 = generate_biomedical_scenario(
            n=500,
            hurst=0.5,
            scenario_type="eeg",
            scenario="rest",
            contamination_level=0.1,
            seed=42
        )
        
        data2 = generate_biomedical_scenario(
            n=500,
            hurst=0.5,
            scenario_type="eeg",
            scenario="rest",
            contamination_level=0.1,
            seed=42
        )
        
        np.testing.assert_array_equal(data1, data2)
    
    def test_generate_biomedical_scenario_edge_cases(self):
        """Test biomedical scenario generation edge cases."""
        # Very short data
        data = generate_biomedical_scenario(
            n=10,
            hurst=0.5,
            scenario_type="eeg",
            scenario="rest",
            contamination_level=0.1,
            seed=42
        )
        assert len(data) == 10
        
        # Zero contamination
        data = generate_biomedical_scenario(
            n=100,
            hurst=0.5,
            scenario_type="eeg",
            scenario="rest",
            contamination_level=0.0,
            seed=42
        )
        assert len(data) == 100
        
        # Maximum contamination
        data = generate_biomedical_scenario(
            n=100,
            hurst=0.5,
            scenario_type="eeg",
            scenario="rest",
            contamination_level=1.0,
            seed=42
        )
        assert len(data) == 100


class TestBiomedicalGrid:
    """Test biomedical grid generation."""
    
    def test_generate_biomedical_grid_basic(self):
        """Test basic biomedical grid generation."""
        grid = generate_biomedical_grid(
            scenario_type="eeg",
            hurst_values=[0.3, 0.5, 0.7],
            lengths=[500, 1000],
            contamination_levels=[0.1, 0.2],
            seed=42
        )
        
        assert isinstance(grid, list)
        assert len(grid) > 0  # Should have generated samples
        
        # Check that all items are TimeSeriesSample instances
        for sample in grid:
            assert hasattr(sample, 'data')
            assert hasattr(sample, 'true_hurst')
            assert hasattr(sample, 'length')
            assert isinstance(sample.data, np.ndarray)
            assert isinstance(sample.true_hurst, (int, float))
            assert isinstance(sample.length, int)
    
    def test_generate_biomedical_grid_different_parameters(self):
        """Test biomedical grid generation with different parameters."""
        # Test with different numbers
        grid = generate_biomedical_grid(
            scenario_type="eeg",
            hurst_values=[0.4, 0.6],
            lengths=[500],
            contamination_levels=[0.1],
            seed=42
        )
        
        assert len(grid) > 0
        assert all(hasattr(sample, 'data') for sample in grid)
    
    def test_generate_biomedical_grid_reproducibility(self):
        """Test biomedical grid generation reproducibility."""
        grid1 = generate_biomedical_grid(
            scenario_type="eeg",
            hurst_values=[0.5],
            lengths=[500],
            contamination_levels=[0.1],
            seed=42
        )
        
        grid2 = generate_biomedical_grid(
            scenario_type="eeg",
            hurst_values=[0.5],
            lengths=[500],
            contamination_levels=[0.1],
            seed=42
        )
        
        assert len(grid1) == len(grid2)
        for sample1, sample2 in zip(grid1, grid2):
            np.testing.assert_array_equal(sample1.data, sample2.data)
            assert sample1.true_hurst == sample2.true_hurst
            assert sample1.length == sample2.length
    
    def test_generate_biomedical_grid_edge_cases(self):
        """Test biomedical grid generation edge cases."""
        # Single scenario
        grid = generate_biomedical_grid(
            scenario_type="eeg",
            hurst_values=[0.5],
            lengths=[500],
            contamination_levels=[0.1],
            seed=42
        )
        
        assert len(grid) == 1
        assert hasattr(grid[0], 'data')
        
        # Multiple scenario types - test ECG
        grid = generate_biomedical_grid(
            scenario_type="ecg",
            hurst_values=[0.5],
            lengths=[500],
            contamination_levels=[0.1],
            seed=42
        )
        
        assert len(grid) > 0
        assert all(hasattr(sample, 'data') for sample in grid)


class TestErrorHandling:
    """Test error handling in biomedical scenarios."""
    
    def test_invalid_scenario_type(self):
        """Test handling of invalid scenario type."""
        with pytest.raises(ValueError, match="Unknown biomedical scenario"):
            data = generate_biomedical_scenario(
                n=100,
                hurst=0.5,
                scenario_type="invalid",
                contamination_level=0.1,
                seed=42
            )
    
    def test_invalid_scenario_name(self):
        """Test handling of invalid scenario name."""
        data = generate_biomedical_scenario(
            n=100,
            hurst=0.5,
            scenario_type="eeg",
            scenario="invalid",
            contamination_level=0.1,
            seed=42
        )
        
        assert len(data) == 100
        assert isinstance(data, np.ndarray)
    
    def test_extreme_contamination_levels(self):
        """Test handling of extreme contamination levels."""
        base_signal = np.random.randn(100)
        
        # Very high contamination
        contaminated = add_eeg_artifacts(base_signal, contamination_level=10.0, seed=42)
        assert len(contaminated) == len(base_signal)
        
        # Negative contamination (should be handled gracefully)
        contaminated = add_eeg_artifacts(base_signal, contamination_level=-0.1, seed=42)
        assert len(contaminated) == len(base_signal)
    
    def test_empty_parameter_lists(self):
        """Test handling of empty parameter lists."""
        # Empty hurst_values
        grid = generate_biomedical_grid(
            scenario_type="eeg",
            hurst_values=[],
            lengths=[500],
            contamination_levels=[0.1],
            seed=42
        )
        
        # Should handle gracefully
        assert isinstance(grid, list)
    
    def test_none_parameters(self):
        """Test handling of None parameters."""
        # None seed should be handled gracefully
        data = generate_eeg_scenario(
            n=100,
            hurst=0.5,
            scenario="rest",
            contamination_level=0.1,
            seed=None
        )
        
        assert len(data) == 100
        assert isinstance(data, np.ndarray)


class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_data_generation(self):
        """Test generation of large datasets."""
        # Large EEG data
        data = generate_eeg_scenario(
            n=10000,
            hurst=0.5,
            scenario="rest",
            contamination_level=0.1,
            seed=42
        )
        
        assert len(data) == 10000
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
    
    def test_grid_generation_performance(self):
        """Test performance of grid generation."""
        import time
        
        start_time = time.time()
        grid = generate_biomedical_grid(
            scenario_type="eeg",
            hurst_values=[0.3, 0.5, 0.7],
            lengths=[500, 1000],
            contamination_levels=[0.1, 0.2],
            seed=42
        )
        end_time = time.time()
        
        # Should complete in reasonable time (< 10 seconds)
        assert end_time - start_time < 10.0
        assert len(grid) > 0
        assert all(hasattr(sample, 'data') for sample in grid)
