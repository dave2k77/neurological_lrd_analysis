"""
Comprehensive Benchmark Comparison: Classical vs ML Models.

This module provides a comprehensive benchmarking framework for comparing
classical Hurst estimation methods with machine learning baseline models.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import time
import warnings
from pathlib import Path
import json

# Import classical methods
from ..biomedical_hurst_factory import (
    BiomedicalHurstEstimatorFactory, EstimatorType, HurstResult
)

# Import ML methods
from .pretrained_models import PretrainedModelManager, ModelStatus
from .inference import PretrainedInference, quick_predict, quick_ensemble_predict

# Import data generation
from ..benchmark_core.generation import (
    generate_grid, fbm_davies_harte, TimeSeriesSample
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test."""
    method_name: str
    method_type: str  # 'classical' or 'ml'
    hurst_estimate: float
    confidence_interval: Optional[Tuple[float, float]] = None
    computation_time: float = 0.0
    error: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results for a method."""
    method_name: str
    method_type: str
    n_tests: int
    mean_error: float
    std_error: float
    mean_absolute_error: float
    root_mean_squared_error: float
    correlation: float
    mean_computation_time: float
    success_rate: float
    results: List[BenchmarkResult]


class ClassicalMLBenchmark:
    """
    Comprehensive benchmark comparing classical and ML methods.
    
    Provides systematic comparison of classical Hurst estimation methods
    with machine learning baseline models across various test scenarios.
    """
    
    def __init__(self, 
                 pretrained_models_dir: Union[str, Path] = "pretrained_models",
                 classical_estimators: Optional[List[EstimatorType]] = None,
                 ml_estimators: Optional[List[str]] = None):
        """
        Initialize the benchmark system.
        
        Parameters:
        -----------
        pretrained_models_dir : str or Path
            Directory containing pretrained ML models
        classical_estimators : List[EstimatorType], optional
            Classical estimators to include
        ml_estimators : List[str], optional
            ML model types to include
        """
        self.pretrained_models_dir = Path(pretrained_models_dir)
        
        # Initialize classical estimator factory
        self.classical_factory = BiomedicalHurstEstimatorFactory()
        
        # Initialize ML inference system
        self.ml_inference = PretrainedInference(pretrained_models_dir)
        
        # Default classical estimators
        if classical_estimators is None:
            self.classical_estimators = [
                EstimatorType.DFA,
                EstimatorType.RS_ANALYSIS,
                EstimatorType.HIGUCHI,
                EstimatorType.GENERALIZED_HURST,
                EstimatorType.PERIODOGRAM,
                EstimatorType.GPH,
                EstimatorType.WHITTLE_MLE,
                EstimatorType.DWT,
                EstimatorType.ABRY_VEITCH,
                EstimatorType.MFDFA
            ]
        else:
            self.classical_estimators = classical_estimators
        
        # Default ML estimators
        if ml_estimators is None:
            self.ml_estimators = ['random_forest', 'svr', 'gradient_boosting']
        else:
            self.ml_estimators = ml_estimators
        
        # Results storage
        self.benchmark_results = {}
        self.summary_results = {}
    
    def create_test_scenarios(self, 
                            hurst_values: List[float] = None,
                            lengths: List[int] = None,
                            n_samples_per_config: int = 10,
                            include_contamination: bool = True,
                            include_biomedical: bool = True) -> List[TimeSeriesSample]:
        """
        Create comprehensive test scenarios.
        
        Parameters:
        -----------
        hurst_values : List[float], optional
            Hurst values to test
        lengths : List[int], optional
            Time series lengths
        n_samples_per_config : int
            Number of samples per configuration
        include_contamination : bool
            Whether to include contaminated data
        include_biomedical : bool
            Whether to include biomedical scenarios
            
        Returns:
        --------
        List[TimeSeriesSample]
            Test scenarios
        """
        if hurst_values is None:
            hurst_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        if lengths is None:
            lengths = [500, 1000, 2000]
        
        # Base generators
        generators = ['fbm', 'fgn', 'arfima', 'mrw', 'fou']
        
        # Contamination types
        contaminations = ['none']
        if include_contamination:
            contaminations.extend(['noise', 'missing', 'artifacts'])
        
        # Biomedical scenarios
        biomedical_scenarios = None
        if include_biomedical:
            biomedical_scenarios = ['eeg', 'ecg', 'respiratory']
        
        print(f"Creating test scenarios...")
        print(f"  - Hurst values: {hurst_values}")
        print(f"  - Lengths: {lengths}")
        print(f"  - Generators: {generators}")
        print(f"  - Contaminations: {contaminations}")
        print(f"  - Biomedical scenarios: {biomedical_scenarios}")
        print(f"  - Samples per config: {n_samples_per_config}")
        
        # Generate test scenarios
        samples = generate_grid(
            hurst_values=hurst_values,
            lengths=lengths,
            contaminations=contaminations,
            generators=generators,
            biomedical_scenarios=biomedical_scenarios
        )
        
        print(f"Generated {len(samples)} test scenarios")
        return samples
    
    def benchmark_classical_methods(self, 
                                  samples: List[TimeSeriesSample]) -> Dict[str, List[BenchmarkResult]]:
        """
        Benchmark classical Hurst estimation methods.
        
        Parameters:
        -----------
        samples : List[TimeSeriesSample]
            Test scenarios
            
        Returns:
        --------
        Dict[str, List[BenchmarkResult]]
            Results for each classical method
        """
        print(f"\nBenchmarking classical methods...")
        print(f"  - Methods: {[e.value for e in self.classical_estimators]}")
        print(f"  - Test scenarios: {len(samples)}")
        
        results = {}
        
        for estimator_type in self.classical_estimators:
            print(f"\n  Testing {estimator_type.value}...")
            method_results = []
            
            try:
                for i, sample in enumerate(samples):
                    if i % 50 == 0 and i > 0:
                        print(f"    Processed {i}/{len(samples)} samples")
                    
                    try:
                        start_time = time.time()
                        
                        # Estimate Hurst exponent using factory directly
                        result = self.classical_factory.estimate(sample.data, estimator_type)
                        
                        computation_time = time.time() - start_time
                        
                        # Calculate error
                        error = abs(result.hurst_estimate - sample.true_hurst)
                        
                        # Create benchmark result
                        benchmark_result = BenchmarkResult(
                            method_name=estimator_type.value,
                            method_type='classical',
                            hurst_estimate=result.hurst_estimate,
                            confidence_interval=result.confidence_interval,
                            computation_time=computation_time,
                            error=error,
                            metadata={
                                'true_hurst': sample.true_hurst,
                                'data_length': len(sample.data),
                                'generator': sample.generator,
                                'contamination': sample.contamination,
                                'biomedical_scenario': sample.biomedical_scenario
                            }
                        )
                        
                        method_results.append(benchmark_result)
                        
                    except Exception as e:
                        warnings.warn(f"Failed to estimate with {estimator_type.value} for sample {i}: {e}")
                        continue
                
                results[estimator_type.value] = method_results
                print(f"    Completed: {len(method_results)} successful estimates")
                
            except Exception as e:
                print(f"    Failed to create estimator {estimator_type.value}: {e}")
                results[estimator_type.value] = []
        
        return results
    
    def benchmark_ml_methods(self, 
                           samples: List[TimeSeriesSample]) -> Dict[str, List[BenchmarkResult]]:
        """
        Benchmark machine learning methods.
        
        Parameters:
        -----------
        samples : List[TimeSeriesSample]
            Test scenarios
            
        Returns:
        --------
        Dict[str, List[BenchmarkResult]]
            Results for each ML method
        """
        print(f"\nBenchmarking ML methods...")
        print(f"  - Methods: {self.ml_estimators}")
        print(f"  - Test scenarios: {len(samples)}")
        
        results = {}
        
        for ml_method in self.ml_estimators:
            print(f"\n  Testing {ml_method}...")
            method_results = []
            
            try:
                for i, sample in enumerate(samples):
                    if i % 50 == 0 and i > 0:
                        print(f"    Processed {i}/{len(samples)} samples")
                    
                    try:
                        start_time = time.time()
                        
                        # Predict using ML method
                        if ml_method == 'ensemble':
                            # Use ensemble prediction
                            mean_est, std_est = quick_ensemble_predict(
                                sample.data, self.pretrained_models_dir
                            )
                            hurst_estimate = mean_est
                            confidence_interval = (mean_est - std_est, mean_est + std_est)
                        else:
                            # Use single model prediction
                            hurst_estimate = quick_predict(
                                sample.data, self.pretrained_models_dir, ml_method
                            )
                            confidence_interval = None
                        
                        computation_time = time.time() - start_time
                        
                        # Calculate error
                        error = abs(hurst_estimate - sample.true_hurst)
                        
                        # Create benchmark result
                        benchmark_result = BenchmarkResult(
                            method_name=ml_method,
                            method_type='ml',
                            hurst_estimate=hurst_estimate,
                            confidence_interval=confidence_interval,
                            computation_time=computation_time,
                            error=error,
                            metadata={
                                'true_hurst': sample.true_hurst,
                                'data_length': len(sample.data),
                                'generator': sample.generator,
                                'contamination': sample.contamination,
                                'biomedical_scenario': sample.biomedical_scenario
                            }
                        )
                        
                        method_results.append(benchmark_result)
                        
                    except Exception as e:
                        warnings.warn(f"Failed to predict with {ml_method} for sample {i}: {e}")
                        continue
                
                results[ml_method] = method_results
                print(f"    Completed: {len(method_results)} successful predictions")
                
            except Exception as e:
                print(f"    Failed to benchmark {ml_method}: {e}")
                results[ml_method] = []
        
        return results
    
    def run_comprehensive_benchmark(self, 
                                  test_scenarios: Optional[List[TimeSeriesSample]] = None,
                                  save_results: bool = True,
                                  results_dir: Union[str, Path] = "benchmark_results") -> Dict[str, Any]:
        """
        Run comprehensive benchmark comparison.
        
        Parameters:
        -----------
        test_scenarios : List[TimeSeriesSample], optional
            Test scenarios to use
        save_results : bool
            Whether to save results to disk
        results_dir : str or Path
            Directory to save results
            
        Returns:
        --------
        Dict[str, Any]
            Complete benchmark results
        """
        print("=" * 80)
        print("COMPREHENSIVE CLASSICAL vs ML BENCHMARK")
        print("=" * 80)
        print("Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)")
        print("=" * 80)
        
        # Create test scenarios if not provided
        if test_scenarios is None:
            test_scenarios = self.create_test_scenarios()
        
        print(f"\nBenchmarking {len(test_scenarios)} test scenarios")
        print(f"Classical methods: {len(self.classical_estimators)}")
        print(f"ML methods: {len(self.ml_estimators)}")
        
        # Benchmark classical methods
        print(f"\n{'='*60}")
        print("CLASSICAL METHODS BENCHMARK")
        print(f"{'='*60}")
        classical_results = self.benchmark_classical_methods(test_scenarios)
        
        # Benchmark ML methods
        print(f"\n{'='*60}")
        print("ML METHODS BENCHMARK")
        print(f"{'='*60}")
        ml_results = self.benchmark_ml_methods(test_scenarios)
        
        # Combine results
        all_results = {**classical_results, **ml_results}
        
        # Calculate summaries
        print(f"\n{'='*60}")
        print("CALCULATING PERFORMANCE SUMMARIES")
        print(f"{'='*60}")
        summaries = self.calculate_summaries(all_results)
        
        # Create comprehensive results
        benchmark_data = {
            'test_scenarios': test_scenarios,
            'classical_results': classical_results,
            'ml_results': ml_results,
            'all_results': all_results,
            'summaries': summaries,
            'benchmark_config': {
                'classical_estimators': [e.value for e in self.classical_estimators],
                'ml_estimators': self.ml_estimators,
                'n_test_scenarios': len(test_scenarios)
            }
        }
        
        # Save results if requested
        if save_results:
            self.save_benchmark_results(benchmark_data, results_dir)
        
        # Print summary
        self.print_benchmark_summary(summaries)
        
        return benchmark_data
    
    def calculate_summaries(self, results: Dict[str, List[BenchmarkResult]]) -> Dict[str, BenchmarkSummary]:
        """Calculate performance summaries for all methods."""
        summaries = {}
        
        for method_name, method_results in results.items():
            if not method_results:
                continue
            
            # Extract data
            errors = [r.error for r in method_results if r.error is not None]
            computation_times = [r.computation_time for r in method_results]
            
            if not errors:
                continue
            
            # Calculate metrics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            mean_absolute_error = np.mean(np.abs(errors))
            root_mean_squared_error = np.sqrt(np.mean(np.array(errors)**2))
            
            # Calculate correlation with true values
            true_hurst = [r.metadata['true_hurst'] for r in method_results if r.metadata]
            estimates = [r.hurst_estimate for r in method_results]
            
            if len(true_hurst) > 1 and len(estimates) > 1:
                correlation = np.corrcoef(true_hurst, estimates)[0, 1]
            else:
                correlation = 0.0
            
            # Calculate success rate
            success_rate = len(method_results) / len(method_results) if method_results else 0.0
            
            # Determine method type
            method_type = 'classical' if method_name in [e.value for e in self.classical_estimators] else 'ml'
            
            # Create summary
            summary = BenchmarkSummary(
                method_name=method_name,
                method_type=method_type,
                n_tests=len(method_results),
                mean_error=mean_error,
                std_error=std_error,
                mean_absolute_error=mean_absolute_error,
                root_mean_squared_error=root_mean_squared_error,
                correlation=correlation,
                mean_computation_time=np.mean(computation_times),
                success_rate=success_rate,
                results=method_results
            )
            
            summaries[method_name] = summary
        
        return summaries
    
    def print_benchmark_summary(self, summaries: Dict[str, BenchmarkSummary]) -> None:
        """Print benchmark summary."""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        # Sort by mean absolute error
        sorted_summaries = sorted(summaries.items(), key=lambda x: x[1].mean_absolute_error)
        
        print(f"\n{'Method':<20} {'Type':<10} {'MAE':<8} {'RMSE':<8} {'Corr':<8} {'Time(ms)':<10} {'Success':<8}")
        print(f"{'-'*80}")
        
        for method_name, summary in sorted_summaries:
            print(f"{method_name:<20} {summary.method_type:<10} "
                  f"{summary.mean_absolute_error:<8.4f} {summary.root_mean_squared_error:<8.4f} "
                  f"{summary.correlation:<8.4f} {summary.mean_computation_time*1000:<10.1f} "
                  f"{summary.success_rate:<8.2f}")
        
        # Best performers
        print(f"\n{'='*60}")
        print("BEST PERFORMERS")
        print(f"{'='*60}")
        
        # Best overall
        if summaries:
            best_overall = min(summaries.items(), key=lambda x: x[1].mean_absolute_error)
            print(f"Best Overall: {best_overall[0]} (MAE: {best_overall[1].mean_absolute_error:.4f})")
            
            # Best classical
            classical_summaries = {k: v for k, v in summaries.items() if v.method_type == 'classical'}
            if classical_summaries:
                best_classical = min(classical_summaries.items(), key=lambda x: x[1].mean_absolute_error)
                print(f"Best Classical: {best_classical[0]} (MAE: {best_classical[1].mean_absolute_error:.4f})")
            
            # Best ML
            ml_summaries = {k: v for k, v in summaries.items() if v.method_type == 'ml'}
            if ml_summaries:
                best_ml = min(ml_summaries.items(), key=lambda x: x[1].mean_absolute_error)
                print(f"Best ML: {best_ml[0]} (MAE: {best_ml[1].mean_absolute_error:.4f})")
        else:
            print("No successful benchmark results to summarize.")
        
        # Speed comparison
        print(f"\n{'='*60}")
        print("SPEED COMPARISON")
        print(f"{'='*60}")
        
        if summaries:
            speed_sorted = sorted(summaries.items(), key=lambda x: x[1].mean_computation_time)
            print(f"Fastest: {speed_sorted[0][0]} ({speed_sorted[0][1].mean_computation_time*1000:.1f}ms)")
            print(f"Slowest: {speed_sorted[-1][0]} ({speed_sorted[-1][1].mean_computation_time*1000:.1f}ms)")
        else:
            print("No successful benchmark results for speed comparison.")
    
    def save_benchmark_results(self, 
                             benchmark_data: Dict[str, Any], 
                             results_dir: Union[str, Path]) -> None:
        """Save benchmark results to disk."""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary data
        summary_data = {}
        for method_name, summary in benchmark_data['summaries'].items():
            summary_data[method_name] = {
                'method_name': summary.method_name,
                'method_type': summary.method_type,
                'n_tests': summary.n_tests,
                'mean_error': summary.mean_error,
                'std_error': summary.std_error,
                'mean_absolute_error': summary.mean_absolute_error,
                'root_mean_squared_error': summary.root_mean_squared_error,
                'correlation': summary.correlation,
                'mean_computation_time': summary.mean_computation_time,
                'success_rate': summary.success_rate
            }
        
        # Save JSON summary
        with open(results_dir / "benchmark_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save detailed results as CSV
        all_results = []
        for method_name, method_results in benchmark_data['all_results'].items():
            for result in method_results:
                all_results.append({
                    'method_name': result.method_name,
                    'method_type': result.method_type,
                    'hurst_estimate': result.hurst_estimate,
                    'true_hurst': result.metadata['true_hurst'],
                    'error': result.error,
                    'computation_time': result.computation_time,
                    'data_length': result.metadata['data_length'],
                    'generator': result.metadata['generator'],
                    'contamination': result.metadata['contamination'],
                    'biomedical_scenario': result.metadata['biomedical_scenario']
                })
        
        df = pd.DataFrame(all_results)
        df.to_csv(results_dir / "benchmark_results.csv", index=False)
        
        print(f"\nBenchmark results saved to: {results_dir}")
    
    def create_visualizations(self, 
                            benchmark_data: Dict[str, Any],
                            save_path: Optional[Path] = None) -> None:
        """Create comprehensive visualizations of benchmark results."""
        print(f"\nCreating benchmark visualizations...")
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Classical vs ML Methods: Comprehensive Benchmark', fontsize=16, fontweight='bold')
        
        summaries = benchmark_data['summaries']
        
        # 1. Performance comparison (MAE)
        ax1 = axes[0, 0]
        methods = list(summaries.keys())
        mae_values = [summaries[m].mean_absolute_error for m in methods]
        colors = ['red' if summaries[m].method_type == 'classical' else 'blue' for m in methods]
        
        bars = ax1.barh(methods, mae_values, color=colors, alpha=0.7)
        ax1.set_xlabel('Mean Absolute Error')
        ax1.set_title('Performance Comparison (MAE)')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Classical'),
                          Patch(facecolor='blue', alpha=0.7, label='ML')]
        ax1.legend(handles=legend_elements)
        
        # 2. Speed comparison
        ax2 = axes[0, 1]
        speed_values = [summaries[m].mean_computation_time * 1000 for m in methods]  # Convert to ms
        bars = ax2.barh(methods, speed_values, color=colors, alpha=0.7)
        ax2.set_xlabel('Computation Time (ms)')
        ax2.set_title('Speed Comparison')
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlation comparison
        ax3 = axes[0, 2]
        corr_values = [summaries[m].correlation for m in methods]
        bars = ax3.barh(methods, corr_values, color=colors, alpha=0.7)
        ax3.set_xlabel('Correlation with True Values')
        ax3.set_title('Accuracy (Correlation)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Error distribution
        ax4 = axes[1, 0]
        classical_errors = []
        ml_errors = []
        
        for method_name, summary in summaries.items():
            if summary.method_type == 'classical':
                classical_errors.extend([r.error for r in summary.results if r.error is not None])
            else:
                ml_errors.extend([r.error for r in summary.results if r.error is not None])
        
        if classical_errors and ml_errors:
            ax4.hist(classical_errors, bins=30, alpha=0.7, label='Classical', color='red', density=True)
            ax4.hist(ml_errors, bins=30, alpha=0.7, label='ML', color='blue', density=True)
            ax4.set_xlabel('Absolute Error')
            ax4.set_ylabel('Density')
            ax4.set_title('Error Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance vs Speed scatter
        ax5 = axes[1, 1]
        mae_values = [summaries[m].mean_absolute_error for m in methods]
        speed_values = [summaries[m].mean_computation_time * 1000 for m in methods]
        
        for i, method in enumerate(methods):
            color = 'red' if summaries[method].method_type == 'classical' else 'blue'
            ax5.scatter(speed_values[i], mae_values[i], color=color, alpha=0.7, s=100)
            ax5.annotate(method, (speed_values[i], mae_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax5.set_xlabel('Computation Time (ms)')
        ax5.set_ylabel('Mean Absolute Error')
        ax5.set_title('Performance vs Speed')
        ax5.grid(True, alpha=0.3)
        
        # 6. Success rate comparison
        ax6 = axes[1, 2]
        success_rates = [summaries[m].success_rate for m in methods]
        bars = ax6.barh(methods, success_rates, color=colors, alpha=0.7)
        ax6.set_xlabel('Success Rate')
        ax6.set_title('Reliability (Success Rate)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to: {save_path}")
        
        plt.show()


def run_comprehensive_benchmark(pretrained_models_dir: Union[str, Path] = "pretrained_models",
                               results_dir: Union[str, Path] = "benchmark_results",
                               test_scenarios: Optional[List[TimeSeriesSample]] = None) -> Dict[str, Any]:
    """
    Run comprehensive benchmark comparison.
    
    Parameters:
    -----------
    pretrained_models_dir : str or Path
        Directory containing pretrained models
    results_dir : str or Path
        Directory to save results
    test_scenarios : List[TimeSeriesSample], optional
        Test scenarios to use
        
    Returns:
    --------
    Dict[str, Any]
        Complete benchmark results
    """
    # Create benchmark system
    benchmark = ClassicalMLBenchmark(pretrained_models_dir)
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        test_scenarios=test_scenarios,
        save_results=True,
        results_dir=results_dir
    )
    
    # Create visualizations
    results_path = Path(results_dir)
    benchmark.create_visualizations(results, results_path / "benchmark_visualization.png")
    
    return results


if __name__ == "__main__":
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    print("\nBenchmark completed successfully!")
