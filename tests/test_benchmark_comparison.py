"""
Tests for benchmark comparison system.

This module provides comprehensive tests for the benchmark comparison
between classical and ML methods for Hurst exponent estimation.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Import the benchmark modules
from neurological_lrd_analysis.ml_baselines.benchmark_comparison import (
    ClassicalMLBenchmark,
    BenchmarkResult,
    BenchmarkSummary,
    run_comprehensive_benchmark
)

from neurological_lrd_analysis.biomedical_hurst_factory import EstimatorType
from neurological_lrd_analysis.benchmark_core.generation import (
    fbm_davies_harte,
    generate_grid,
    TimeSeriesSample
)


class TestBenchmarkResult:
    """Test benchmark result data structures."""
    
    def test_benchmark_result_creation(self):
        """Test benchmark result creation."""
        result = BenchmarkResult(
            method_name="test_method",
            method_type="classical",
            hurst_estimate=0.6,
            confidence_interval=(0.5, 0.7),
            computation_time=0.1,
            error=0.05,
            metadata={"test": "value"}
        )
        
        assert result.method_name == "test_method"
        assert result.method_type == "classical"
        assert result.hurst_estimate == 0.6
        assert result.confidence_interval == (0.5, 0.7)
        assert result.computation_time == 0.1
        assert result.error == 0.05
        assert result.metadata["test"] == "value"
    
    def test_benchmark_summary_creation(self):
        """Test benchmark summary creation."""
        # Create mock results
        results = []
        for i in range(10):
            result = BenchmarkResult(
                method_name="test_method",
                method_type="classical",
                hurst_estimate=0.5 + i * 0.01,
                computation_time=0.1,
                error=0.01 + i * 0.001,
                metadata={"true_hurst": 0.5 + i * 0.01}
            )
            results.append(result)
        
        summary = BenchmarkSummary(
            method_name="test_method",
            method_type="classical",
            n_tests=10,
            mean_error=0.015,
            std_error=0.005,
            mean_absolute_error=0.015,
            root_mean_squared_error=0.016,
            correlation=0.95,
            mean_computation_time=0.1,
            success_rate=1.0,
            results=results
        )
        
        assert summary.method_name == "test_method"
        assert summary.n_tests == 10
        assert summary.mean_error == 0.015
        assert summary.success_rate == 1.0


class TestClassicalMLBenchmark:
    """Test the benchmark comparison system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple benchmark system
        self.benchmark = ClassicalMLBenchmark(
            pretrained_models_dir=self.temp_dir,
            classical_estimators=[EstimatorType.DFA, EstimatorType.RS_ANALYSIS],
            ml_estimators=['random_forest']
        )
        
        # Create simple test scenarios
        self.test_scenarios = []
        for hurst in [0.3, 0.5, 0.7]:
            data = fbm_davies_harte(200, hurst, seed=42)
            sample = TimeSeriesSample(
                data=data,
                true_hurst=hurst,
                length=len(data),
                contamination='none',
                seed=42
            )
            self.test_scenarios.append(sample)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_benchmark_initialization(self):
        """Test benchmark system initialization."""
        assert self.benchmark.pretrained_models_dir == Path(self.temp_dir)
        assert len(self.benchmark.classical_estimators) == 2
        assert len(self.benchmark.ml_estimators) == 1
        assert self.benchmark.classical_factory is not None
        assert self.benchmark.ml_inference is not None
    
    def test_create_test_scenarios(self):
        """Test test scenario creation."""
        scenarios = self.benchmark.create_test_scenarios(
            hurst_values=[0.3, 0.5, 0.7],
            lengths=[200],
            n_samples_per_config=2,
            include_contamination=False,
            include_biomedical=False
        )
        
        assert len(scenarios) > 0
        for scenario in scenarios:
            assert hasattr(scenario, 'data')
            assert hasattr(scenario, 'true_hurst')
            assert len(scenario.data) > 0
            assert 0 < scenario.true_hurst < 1
    
    def test_benchmark_classical_methods(self):
        """Test classical method benchmarking."""
        results = self.benchmark.benchmark_classical_methods(self.test_scenarios)
        
        assert len(results) > 0
        for method_name, method_results in results.items():
            assert isinstance(method_results, list)
            for result in method_results:
                assert isinstance(result, BenchmarkResult)
                assert result.method_type == 'classical'
                assert result.hurst_estimate > 0
                assert result.error is not None
    
    def test_calculate_summaries(self):
        """Test summary calculation."""
        # Create mock results
        mock_results = {
            'test_method': [
                BenchmarkResult(
                    method_name='test_method',
                    method_type='classical',
                    hurst_estimate=0.6,
                    computation_time=0.1,
                    error=0.05,
                    metadata={'true_hurst': 0.5}
                ),
                BenchmarkResult(
                    method_name='test_method',
                    method_type='classical',
                    hurst_estimate=0.7,
                    computation_time=0.1,
                    error=0.05,
                    metadata={'true_hurst': 0.6}
                )
            ]
        }
        
        summaries = self.benchmark.calculate_summaries(mock_results)
        
        assert 'test_method' in summaries
        summary = summaries['test_method']
        assert summary.method_name == 'test_method'
        assert summary.n_tests == 2
        assert summary.mean_error == 0.05
        assert summary.success_rate == 1.0
    
    def test_benchmark_save_results(self):
        """Test saving benchmark results."""
        # Create mock benchmark data
        mock_data = {
            'test_scenarios': self.test_scenarios,
            'classical_results': {},
            'ml_results': {},
            'all_results': {},
            'summaries': {},
            'benchmark_config': {}
        }
        
        results_dir = Path(self.temp_dir) / "test_results"
        self.benchmark.save_benchmark_results(mock_data, results_dir)
        
        # Check that files were created
        assert results_dir.exists()
        assert (results_dir / "benchmark_summary.json").exists()
        assert (results_dir / "benchmark_results.csv").exists()


class TestBenchmarkIntegration:
    """Test benchmark integration with the main library."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_benchmark_with_simple_data(self):
        """Test benchmark with simple synthetic data."""
        # Create simple test scenarios
        test_scenarios = []
        for hurst in [0.3, 0.5, 0.7]:
            data = fbm_davies_harte(200, hurst, seed=42)
            sample = TimeSeriesSample(
                data=data,
                true_hurst=hurst,
                length=len(data),
                contamination='none',
                seed=42
            )
            test_scenarios.append(sample)
        
        # Create benchmark system
        benchmark = ClassicalMLBenchmark(
            pretrained_models_dir=self.temp_dir,
            classical_estimators=[EstimatorType.DFA],
            ml_estimators=[]  # Skip ML for this test
        )
        
        # Run benchmark
        results = benchmark.run_comprehensive_benchmark(
            test_scenarios=test_scenarios,
            save_results=False
        )
        
        assert 'classical_results' in results
        assert 'summaries' in results
        assert len(results['summaries']) > 0
    
    def test_benchmark_error_handling(self):
        """Test benchmark error handling."""
        # Create benchmark with invalid configuration
        benchmark = ClassicalMLBenchmark(
            pretrained_models_dir="nonexistent_dir",
            classical_estimators=[EstimatorType.DFA],
            ml_estimators=[]
        )
        
        # Create test scenarios
        test_scenarios = []
        for hurst in [0.3, 0.5]:
            data = fbm_davies_harte(200, hurst, seed=42)
            sample = TimeSeriesSample(
                data=data,
                true_hurst=hurst,
                length=len(data),
                contamination='none',
                seed=42
            )
            test_scenarios.append(sample)
        
        # Run benchmark (should handle errors gracefully)
        results = benchmark.run_comprehensive_benchmark(
            test_scenarios=test_scenarios,
            save_results=False
        )
        
        # Should still return results structure
        assert 'classical_results' in results
        assert 'summaries' in results


class TestBenchmarkVisualization:
    """Test benchmark visualization functionality."""
    
    def test_benchmark_visualization_creation(self):
        """Test that benchmark visualizations can be created."""
        # Skip this test in headless environments
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create mock benchmark data
        mock_data = {
            'summaries': {
                'test_method': BenchmarkSummary(
                    method_name='test_method',
                    method_type='classical',
                    n_tests=10,
                    mean_error=0.05,
                    std_error=0.01,
                    mean_absolute_error=0.05,
                    root_mean_squared_error=0.06,
                    correlation=0.9,
                    mean_computation_time=0.1,
                    success_rate=1.0,
                    results=[]
                )
            }
        }
        
        # Create benchmark system
        benchmark = ClassicalMLBenchmark("temp_dir")
        
        # Test visualization creation (should not raise errors)
        try:
            benchmark.create_visualizations(mock_data, Path("temp_plot.png"))
            # If we get here, the visualization was created successfully
            assert True
        except Exception as e:
            # If visualization fails, it should be due to missing dependencies, not logic errors
            assert "matplotlib" in str(e).lower() or "seaborn" in str(e).lower()


class TestBenchmarkPerformance:
    """Test benchmark performance characteristics."""
    
    def test_benchmark_speed(self):
        """Test that benchmark runs in reasonable time."""
        import time
        
        # Create simple test scenarios
        test_scenarios = []
        for hurst in [0.3, 0.5]:
            data = fbm_davies_harte(100, hurst, seed=42)  # Short data for speed
            sample = TimeSeriesSample(
                data=data,
                true_hurst=hurst,
                length=len(data),
                contamination='none',
                seed=42
            )
            test_scenarios.append(sample)
        
        # Create benchmark system
        benchmark = ClassicalMLBenchmark(
            pretrained_models_dir="temp_dir",
            classical_estimators=[EstimatorType.DFA],
            ml_estimators=[]
        )
        
        # Time the benchmark
        start_time = time.time()
        results = benchmark.run_comprehensive_benchmark(
            test_scenarios=test_scenarios,
            save_results=False
        )
        end_time = time.time()
        
        # Should complete in reasonable time (less than 30 seconds for simple test)
        assert (end_time - start_time) < 30.0
        assert 'summaries' in results


if __name__ == "__main__":
    pytest.main([__file__])
