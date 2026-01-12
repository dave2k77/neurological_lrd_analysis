import numpy as np
import pytest
import os
import shutil
from pathlib import Path

from neurological_lrd_analysis.benchmark_core.generation import generate_grid, TimeSeriesSample
from neurological_lrd_analysis.benchmark_core.runner import (
    BenchmarkConfig, 
    run_benchmark_on_dataset, 
    analyze_benchmark_results,
    create_leaderboard
)

class TestBenchmarkingSystem:
    """Integration tests for the benchmarking system."""
    
    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "benchmark_results")
        
    def test_generate_grid(self):
        """Test generating a grid of test datasets."""
        hurst_values = [0.3, 0.7]
        lengths = [512]
        contaminations = ["none", "noise"]
        
        datasets = generate_grid(
            hurst_values=hurst_values,
            lengths=lengths,
            contaminations=contaminations,
            seed=42
        )
        
        # Expected: 2 (H) * 1 (L) * 2 (C) * 1 (Default Generator fbm) = 4
        assert len(datasets) == 4
        assert isinstance(datasets[0], TimeSeriesSample)
        assert datasets[0].length == 512
        
    def test_run_benchmark(self, output_dir):
        """Test running a benchmark on a small dataset."""
        # Create a tiny dataset
        samples = [
            TimeSeriesSample(
                data=np.cumsum(np.random.randn(500)),
                true_hurst=0.6,
                length=500,
                contamination="none",
                seed=42
            )
        ]
        
        config = BenchmarkConfig(
            output_dir=output_dir,
            n_bootstrap=10, # Fast for testing
            estimators=["DFA", "Higuchi"],
            save_results=True,
            verbose=False
        )
        
        results = run_benchmark_on_dataset(samples, config)
        
        assert len(results) == 2 # 2 estimators for 1 sample
        assert os.path.exists(os.path.join(output_dir, "benchmark_results.csv"))
        assert os.path.exists(os.path.join(output_dir, "benchmark_summary.txt"))
        
    def test_analyze_and_leaderboard(self):
        """Test results analysis and leaderboard creation."""
        from neurological_lrd_analysis.benchmark_core.runner import BenchmarkResult
        
        # Create mock results
        results = [
            BenchmarkResult(
                estimator="DFA",
                hurst_estimate=0.62,
                true_hurst=0.6,
                computation_time=0.1,
                convergence_flag=True,
                bias=0.02,
                absolute_error=0.02
            ),
            BenchmarkResult(
                estimator="Higuchi",
                hurst_estimate=0.55,
                true_hurst=0.6,
                computation_time=0.05,
                convergence_flag=True,
                bias=-0.05,
                absolute_error=0.05
            )
        ]
        
        analysis = analyze_benchmark_results(results)
        assert "DFA" in analysis
        assert analysis["DFA"]["mean_absolute_error"] == 0.02
        
        leaderboard = create_leaderboard(results)
        assert len(leaderboard) == 2
        assert leaderboard.iloc[0]["Estimator"] == "DFA" # DFA is more accurate in this mock
