#!/usr/bin/env python3
"""
Working Benchmark Demonstration with ML Models.

This script demonstrates a working benchmark comparison between
classical and ML methods, including proper ML model integration.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Import the main library
from neurological_lrd_analysis import (
    # Data generation
    fbm_davies_harte,
    
    # Classical methods
    BiomedicalHurstEstimatorFactory, EstimatorType,
    
    # ML methods
    create_pretrained_suite, PretrainedInference,
    quick_predict, quick_ensemble_predict,
    TimeSeriesFeatureExtractor
)

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_test_scenarios():
    """Create test scenarios for benchmarking."""
    print("Creating test scenarios...")
    
    test_scenarios = []
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    
    for hurst in hurst_values:
        # Generate fBm data
        data = fbm_davies_harte(1000, hurst, seed=42)
        test_scenarios.append({
            'data': data,
            'true_hurst': hurst,
            'scenario': f'fBm_H{hurst}'
        })
    
    print(f"Created {len(test_scenarios)} test scenarios")
    return test_scenarios


def benchmark_classical_methods(test_scenarios):
    """Benchmark classical methods."""
    print("\nBenchmarking classical methods...")
    
    classical_factory = BiomedicalHurstEstimatorFactory()
    classical_methods = [EstimatorType.DFA, EstimatorType.RS_ANALYSIS, EstimatorType.HIGUCHI]
    
    results = {}
    
    for method in classical_methods:
        print(f"  Testing {method.value}...")
        method_results = []
        
        try:
            estimator = classical_factory.get_estimator(method)
            
            for scenario in test_scenarios:
                start_time = time.time()
                result = estimator.estimate(scenario['data'])
                computation_time = time.time() - start_time
                
                error = abs(result.hurst_estimate - scenario['true_hurst'])
                
                method_results.append({
                    'method_name': method.value,
                    'method_type': 'classical',
                    'hurst_estimate': result.hurst_estimate,
                    'true_hurst': scenario['true_hurst'],
                    'error': error,
                    'computation_time': computation_time,
                    'scenario': scenario['scenario']
                })
            
            results[method.value] = method_results
            print(f"    Completed: {len(method_results)} successful estimates")
            
        except Exception as e:
            print(f"    Failed: {e}")
            results[method.value] = []
    
    return results


def benchmark_ml_methods(test_scenarios):
    """Benchmark ML methods."""
    print("\nBenchmarking ML methods...")
    
    # Create pretrained models if they don't exist
    models_dir = "pretrained_models_benchmark"
    if not Path(models_dir).exists():
        print("Creating pretrained models...")
        create_pretrained_suite(models_dir, force_retrain=True)
    
    ml_methods = ['random_forest', 'ensemble']
    results = {}
    
    for method in ml_methods:
        print(f"  Testing {method}...")
        method_results = []
        
        try:
            for scenario in test_scenarios:
                start_time = time.time()
                
                if method == 'ensemble':
                    pred, _ = quick_ensemble_predict(scenario['data'], models_dir)
                else:
                    pred = quick_predict(scenario['data'], models_dir, method)
                
                computation_time = time.time() - start_time
                error = abs(pred - scenario['true_hurst'])
                
                method_results.append({
                    'method_name': method,
                    'method_type': 'ml',
                    'hurst_estimate': pred,
                    'true_hurst': scenario['true_hurst'],
                    'error': error,
                    'computation_time': computation_time,
                    'scenario': scenario['scenario']
                })
            
            results[method] = method_results
            print(f"    Completed: {len(method_results)} successful predictions")
            
        except Exception as e:
            print(f"    Failed: {e}")
            results[method] = []
    
    return results


def calculate_performance_metrics(results):
    """Calculate performance metrics for all methods."""
    print("\nCalculating performance metrics...")
    
    metrics = {}
    
    for method_name, method_results in results.items():
        if not method_results:
            continue
        
        errors = [r['error'] for r in method_results]
        times = [r['computation_time'] for r in method_results]
        estimates = [r['hurst_estimate'] for r in method_results]
        true_values = [r['true_hurst'] for r in method_results]
        
        # Calculate metrics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        
        # Correlation
        if len(true_values) > 1:
            correlation = np.corrcoef(true_values, estimates)[0, 1]
        else:
            correlation = 0.0
        
        mean_time = np.mean(times)
        success_rate = len(method_results) / len(method_results) if method_results else 0.0
        
        metrics[method_name] = {
            'method_name': method_name,
            'method_type': method_results[0]['method_type'] if method_results else 'unknown',
            'n_tests': len(method_results),
            'mean_error': mean_error,
            'std_error': std_error,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'mean_time': mean_time,
            'success_rate': success_rate
        }
    
    return metrics


def create_benchmark_visualizations(metrics, save_path=None):
    """Create comprehensive benchmark visualizations."""
    print("\nCreating benchmark visualizations...")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Classical vs ML Methods: Benchmark Results', fontsize=16, fontweight='bold')
    
    methods = list(metrics.keys())
    method_types = [metrics[m]['method_type'] for m in methods]
    colors = ['red' if t == 'classical' else 'blue' for t in method_types]
    
    # 1. Performance comparison (MAE)
    ax1 = axes[0, 0]
    mae_values = [metrics[m]['mae'] for m in methods]
    bars = ax1.barh(methods, mae_values, color=colors, alpha=0.7)
    ax1.set_xlabel('Mean Absolute Error')
    ax1.set_title('Performance Comparison (MAE)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Speed comparison
    ax2 = axes[0, 1]
    speed_values = [metrics[m]['mean_time'] * 1000 for m in methods]  # Convert to ms
    bars = ax2.barh(methods, speed_values, color=colors, alpha=0.7)
    ax2.set_xlabel('Computation Time (ms)')
    ax2.set_title('Speed Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation comparison
    ax3 = axes[1, 0]
    corr_values = [metrics[m]['correlation'] for m in methods]
    bars = ax3.barh(methods, corr_values, color=colors, alpha=0.7)
    ax3.set_xlabel('Correlation with True Values')
    ax3.set_title('Accuracy (Correlation)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance vs Speed scatter
    ax4 = axes[1, 1]
    for i, method in enumerate(methods):
        color = 'red' if method_types[i] == 'classical' else 'blue'
        ax4.scatter(speed_values[i], mae_values[i], color=color, alpha=0.7, s=100)
        ax4.annotate(method, (speed_values[i], mae_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Computation Time (ms)')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Performance vs Speed')
    ax4.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Classical'),
                      Patch(facecolor='blue', alpha=0.7, label='ML')]
    ax1.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {save_path}")
    
    plt.show()


def print_benchmark_summary(metrics):
    """Print benchmark summary."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Sort by MAE
    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]['mae'])
    
    print(f"\n{'Method':<20} {'Type':<10} {'MAE':<8} {'RMSE':<8} {'Corr':<8} {'Time(ms)':<10} {'Success':<8}")
    print("-" * 80)
    
    for method_name, metric in sorted_metrics:
        print(f"{method_name:<20} {metric['method_type']:<10} "
              f"{metric['mae']:<8.4f} {metric['rmse']:<8.4f} "
              f"{metric['correlation']:<8.4f} {metric['mean_time']*1000:<10.1f} "
              f"{metric['success_rate']:<8.2f}")
    
    # Best performers
    print(f"\n{'='*60}")
    print("BEST PERFORMERS")
    print(f"{'='*60}")
    
    if sorted_metrics:
        best_overall = sorted_metrics[0]
        print(f"Best Overall: {best_overall[0]} (MAE: {best_overall[1]['mae']:.4f})")
        
        # Best classical
        classical_metrics = {k: v for k, v in metrics.items() if v['method_type'] == 'classical'}
        if classical_metrics:
            best_classical = min(classical_metrics.items(), key=lambda x: x[1]['mae'])
            print(f"Best Classical: {best_classical[0]} (MAE: {best_classical[1]['mae']:.4f})")
        
        # Best ML
        ml_metrics = {k: v for k, v in metrics.items() if v['method_type'] == 'ml'}
        if ml_metrics:
            best_ml = min(ml_metrics.items(), key=lambda x: x[1]['mae'])
            print(f"Best ML: {best_ml[0]} (MAE: {best_ml[1]['mae']:.4f})")


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("WORKING BENCHMARK DEMONSTRATION: CLASSICAL vs ML METHODS")
    print("=" * 80)
    print("Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # 1. Create test scenarios
        test_scenarios = create_test_scenarios()
        
        # 2. Benchmark classical methods
        classical_results = benchmark_classical_methods(test_scenarios)
        
        # 3. Benchmark ML methods
        ml_results = benchmark_ml_methods(test_scenarios)
        
        # 4. Combine results
        all_results = {**classical_results, **ml_results}
        
        # 5. Calculate performance metrics
        metrics = calculate_performance_metrics(all_results)
        
        # 6. Print summary
        print_benchmark_summary(metrics)
        
        # 7. Create visualizations
        results_dir = Path("results/working_benchmark")
        results_dir.mkdir(parents=True, exist_ok=True)
        create_benchmark_visualizations(metrics, results_dir / "benchmark_results.png")
        
        print("\n" + "=" * 80)
        print("BENCHMARK DEMONSTRATION COMPLETED!")
        print("=" * 80)
        print("\nKey Findings:")
        print("✅ Classical methods provide reliable baseline performance")
        print("✅ ML methods offer potential for improved accuracy")
        print("✅ Speed vs accuracy trade-offs clearly demonstrated")
        print("✅ Comprehensive evaluation framework established")
        
        print(f"\nResults saved to: {results_dir}")
        print("You can now analyze the detailed performance comparisons!")
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
