"""
Simple tests for generation module.

This module tests the actual available generation functions.
"""

import numpy as np
import pytest
from typing import List, Optional
import tempfile
import os

from neurological_lrd_analysis.benchmark_core.generation import (
    TimeSeriesSample,
    fbm_davies_harte,
    generate_fgn,
    generate_arfima,
    generate_mrw,
    generate_fou,
    add_contamination,
    generate_grid
)


class TestTimeSeriesSample:
    """Test TimeSeriesSample dataclass."""
    
    def test_time_series_sample_creation(self):
        """Test TimeSeriesSample creation."""
        data = np.random.randn(100)
        sample = TimeSeriesSample(
            data=data,
            true_hurst=0.5,
            length=100,
            contamination='none',
            seed=42
        )
        
        assert sample.data is not None
        assert sample.true_hurst == 0.5
        assert sample.length == 100
        assert sample.contamination == 'none'
        assert sample.seed == 42
    
    def test_time_series_sample_defaults(self):
        """Test TimeSeriesSample with default values."""
        data = np.random.randn(50)
        sample = TimeSeriesSample(
            data=data,
            true_hurst=0.7,
            length=50,
            contamination='noise'
        )
        
        assert sample.data is not None
        assert sample.true_hurst == 0.7
        assert sample.length == 50
        assert sample.contamination == 'noise'
        assert sample.seed is None


class TestFbmDaviesHarte:
    """Test FBM generation using Davies-Harte method."""
    
    def test_fbm_davies_harte_basic(self):
        """Test basic FBM generation."""
        data = fbm_davies_harte(100, 0.5, seed=42)
        
        assert len(data) == 100
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))
    
    def test_fbm_davies_harte_different_hurst(self):
        """Test FBM generation with different Hurst values."""
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        
        for hurst in hurst_values:
            data = fbm_davies_harte(100, hurst, seed=42)
            assert len(data) == 100
            assert isinstance(data, np.ndarray)
            assert not np.any(np.isnan(data))
            assert not np.any(np.isinf(data))
    
    def test_fbm_davies_harte_reproducibility(self):
        """Test FBM generation reproducibility."""
        data1 = fbm_davies_harte(100, 0.5, seed=42)
        data2 = fbm_davies_harte(100, 0.5, seed=42)
        
        np.testing.assert_array_equal(data1, data2)
    
    def test_fbm_davies_harte_invalid_hurst(self):
        """Test FBM generation with invalid Hurst values."""
        with pytest.raises(ValueError):
            fbm_davies_harte(100, 0.0, seed=42)  # H must be > 0
        
        with pytest.raises(ValueError):
            fbm_davies_harte(100, 1.0, seed=42)  # H must be < 1
        
        with pytest.raises(ValueError):
            fbm_davies_harte(100, -0.1, seed=42)  # H must be > 0
    
    def test_fbm_davies_harte_invalid_length(self):
        """Test FBM generation with invalid length."""
        with pytest.raises(TypeError):
            fbm_davies_harte(0, 0.5, seed=42)  # n must be > 0
        
        with pytest.raises(TypeError):
            fbm_davies_harte(-1, 0.5, seed=42)  # n must be > 0


class TestGenerateFgn:
    """Test FGN generation."""
    
    def test_generate_fgn_basic(self):
        """Test basic FGN generation."""
        data = generate_fgn(100, 0.5, seed=42)
        
        assert len(data) == 100
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))
    
    def test_generate_fgn_different_hurst(self):
        """Test FGN generation with different Hurst values."""
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        
        for hurst in hurst_values:
            data = generate_fgn(100, hurst, seed=42)
            assert len(data) == 100
            assert isinstance(data, np.ndarray)
            assert not np.any(np.isnan(data))
            assert not np.any(np.isinf(data))
    
    def test_generate_fgn_reproducibility(self):
        """Test FGN generation reproducibility."""
        data1 = generate_fgn(100, 0.5, seed=42)
        data2 = generate_fgn(100, 0.5, seed=42)
        
        np.testing.assert_array_equal(data1, data2)


class TestGenerateArfima:
    """Test ARFIMA generation."""
    
    def test_generate_arfima_basic(self):
        """Test basic ARFIMA generation."""
        data = generate_arfima(100, 0.5, seed=42)
        
        assert len(data) == 100
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))
    
    def test_generate_arfima_different_hurst(self):
        """Test ARFIMA generation with different Hurst values."""
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        
        for hurst in hurst_values:
            data = generate_arfima(100, hurst, seed=42)
            assert len(data) == 100
            assert isinstance(data, np.ndarray)
            assert not np.any(np.isnan(data))
            assert not np.any(np.isinf(data))


class TestGenerateMrw:
    """Test MRW generation."""
    
    def test_generate_mrw_basic(self):
        """Test basic MRW generation."""
        data = generate_mrw(100, 0.5, seed=42)
        
        assert len(data) == 100
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))
    
    def test_generate_mrw_different_hurst(self):
        """Test MRW generation with different Hurst values."""
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        
        for hurst in hurst_values:
            data = generate_mrw(100, hurst, seed=42)
            assert len(data) == 100
            assert isinstance(data, np.ndarray)
            assert not np.any(np.isnan(data))
            assert not np.any(np.isinf(data))


class TestGenerateFou:
    """Test FOU generation."""
    
    def test_generate_fou_basic(self):
        """Test basic FOU generation."""
        data = generate_fou(100, 0.5, seed=42)
        
        assert len(data) == 100
        assert isinstance(data, np.ndarray)
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))
    
    def test_generate_fou_different_hurst(self):
        """Test FOU generation with different Hurst values."""
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        
        for hurst in hurst_values:
            data = generate_fou(100, hurst, seed=42)
            assert len(data) == 100
            assert isinstance(data, np.ndarray)
            assert not np.any(np.isnan(data))
            assert not np.any(np.isinf(data))


class TestAddContamination:
    """Test contamination addition."""
    
    def test_add_contamination_basic(self):
        """Test basic contamination addition."""
        data = np.random.randn(100)
        contaminated = add_contamination(
            data, 
            contamination_type='noise', 
            contamination_level=0.1, 
            seed=42
        )
        
        assert len(contaminated) == len(data)
        assert isinstance(contaminated, np.ndarray)
        assert not np.any(np.isnan(contaminated))
        assert not np.any(np.isinf(contaminated))
    
    def test_add_contamination_different_types(self):
        """Test contamination with different types."""
        data = np.random.randn(100)
        contamination_types = ['none', 'noise', 'missing', 'outliers']
        
        for cont_type in contamination_types:
            contaminated = add_contamination(
                data, 
                contamination_type=cont_type, 
                contamination_level=0.1, 
                seed=42
            )
            assert len(contaminated) == len(data)
            assert isinstance(contaminated, np.ndarray)
    
    def test_add_contamination_different_levels(self):
        """Test contamination with different levels."""
        data = np.random.randn(100)
        levels = [0.0, 0.1, 0.2, 0.5]
        
        for level in levels:
            contaminated = add_contamination(
                data, 
                contamination_type='noise', 
                contamination_level=level, 
                seed=42
            )
            assert len(contaminated) == len(data)
            assert isinstance(contaminated, np.ndarray)
    
    def test_add_contamination_reproducibility(self):
        """Test contamination reproducibility."""
        data = np.random.randn(100)
        
        contaminated1 = add_contamination(
            data, 
            contamination_type='noise', 
            contamination_level=0.1, 
            seed=42
        )
        contaminated2 = add_contamination(
            data, 
            contamination_type='noise', 
            contamination_level=0.1, 
            seed=42
        )
        
        np.testing.assert_array_equal(contaminated1, contaminated2)
    
    def test_add_contamination_edge_cases(self):
        """Test contamination edge cases."""
        data = np.random.randn(100)
        
        # Test with zero contamination
        contaminated = add_contamination(
            data, 
            contamination_type='noise', 
            contamination_level=0.0, 
            seed=42
        )
        np.testing.assert_array_equal(data, contaminated)
        
        # Test with maximum contamination
        contaminated = add_contamination(
            data, 
            contamination_type='noise', 
            contamination_level=1.0, 
            seed=42
        )
        assert len(contaminated) == len(data)
        assert isinstance(contaminated, np.ndarray)


class TestGenerateGrid:
    """Test grid generation."""
    
    def test_generate_grid_basic(self):
        """Test basic grid generation."""
        grid = generate_grid(
            hurst_values=[0.3, 0.5, 0.7],
            lengths=[100, 200],
            contaminations=['none', 'noise'],
            contamination_level=0.1,
            seed=42
        )
        
        assert isinstance(grid, list)
        assert len(grid) > 0
        assert all(isinstance(sample, TimeSeriesSample) for sample in grid)
    
    def test_generate_grid_different_parameters(self):
        """Test grid generation with different parameters."""
        grid = generate_grid(
            hurst_values=[0.4, 0.6],
            lengths=[50, 100],
            contaminations=['none'],
            contamination_level=0.2,
            generators=['fbm', 'fgn'],
            seed=42
        )
        
        assert isinstance(grid, list)
        assert len(grid) > 0
        assert all(isinstance(sample, TimeSeriesSample) for sample in grid)
    
    def test_generate_grid_reproducibility(self):
        """Test grid generation reproducibility."""
        grid1 = generate_grid(
            hurst_values=[0.5],
            lengths=[100],
            contaminations=['none'],
            seed=42
        )
        grid2 = generate_grid(
            hurst_values=[0.5],
            lengths=[100],
            contaminations=['none'],
            seed=42
        )
        
        assert len(grid1) == len(grid2)
        for sample1, sample2 in zip(grid1, grid2):
            np.testing.assert_array_equal(sample1.data, sample2.data)
    
    def test_generate_grid_edge_cases(self):
        """Test grid generation edge cases."""
        # Test with single values
        grid = generate_grid(
            hurst_values=[0.5],
            lengths=[100],
            contaminations=['none'],
            seed=42
        )
        
        assert isinstance(grid, list)
        assert len(grid) > 0
        assert all(isinstance(sample, TimeSeriesSample) for sample in grid)


class TestErrorHandling:
    """Test error handling in generation functions."""
    
    def test_fbm_davies_harte_invalid_parameters(self):
        """Test FBM generation with invalid parameters."""
        with pytest.raises(TypeError):
            fbm_davies_harte(-1, 0.5, seed=42)
        
        with pytest.raises(ValueError):
            fbm_davies_harte(100, 1.5, seed=42)
    
    def test_add_contamination_invalid_type(self):
        """Test contamination with invalid type."""
        data = np.random.randn(100)
        
        # The function may not raise an error for invalid types, just use a default
        contaminated = add_contamination(
            data, 
            contamination_type='invalid_type', 
            contamination_level=0.1, 
            seed=42
        )
        # Check that it still returns valid data
        assert len(contaminated) == len(data)
        assert isinstance(contaminated, np.ndarray)
    
    def test_generate_grid_invalid_parameters(self):
        """Test grid generation with invalid parameters."""
        # Test with empty lists - the function may handle this gracefully
        grid1 = generate_grid(
            hurst_values=[],  # Empty list
            lengths=[100],
            contaminations=['none'],
            seed=42
        )
        # Check that it returns an empty list or handles gracefully
        assert isinstance(grid1, list)
        
        grid2 = generate_grid(
            hurst_values=[0.5],
            lengths=[],  # Empty list
            contaminations=['none'],
            seed=42
        )
        # Check that it returns an empty list or handles gracefully
        assert isinstance(grid2, list)


class TestPerformance:
    """Test performance of generation functions."""
    
    def test_fbm_generation_performance(self):
        """Test FBM generation performance."""
        import time
        
        start_time = time.time()
        data = fbm_davies_harte(1000, 0.5, seed=42)
        end_time = time.time()
        
        assert len(data) == 1000
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds
    
    def test_contamination_performance(self):
        """Test contamination performance."""
        import time
        
        data = np.random.randn(1000)
        
        start_time = time.time()
        contaminated = add_contamination(
            data, 
            contamination_type='noise', 
            contamination_level=0.1, 
            seed=42
        )
        end_time = time.time()
        
        assert len(contaminated) == len(data)
        assert (end_time - start_time) < 2.0  # Should complete within 2 seconds
    
    def test_grid_generation_performance(self):
        """Test grid generation performance."""
        import time
        
        start_time = time.time()
        grid = generate_grid(
            hurst_values=[0.3, 0.5, 0.7],
            lengths=[100, 200],
            contaminations=['none', 'noise'],
            seed=42
        )
        end_time = time.time()
        
        assert len(grid) > 0
        assert (end_time - start_time) < 10.0  # Should complete within 10 seconds
