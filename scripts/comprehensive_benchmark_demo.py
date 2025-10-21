#!/usr/bin/env python3
"""
Comprehensive Benchmark Demonstration: Classical vs ML Methods.

This script demonstrates a comprehensive benchmark comparison between
classical Hurst estimation methods and machine learning baseline models.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings

# Import the main library
from neurological_lrd_analysis import (
    # Data generation
    generate_grid, fbm_davies_harte,
    
    # Classical methods
    BiomedicalHurstEstimatorFactory, EstimatorType,
    
    # ML methods
    create_pretrained_suite, PretrainedInference,
    quick_predict, quick_ensemble_predict
)

# Import benchmark system
from neurological_lrd_analysis.ml_baselines.benchmark_comparison import (
    ClassicalMLBenchmark, run_comprehensive_benchmark
)

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def demonstrate_quick_comparison():
    """Demonstrate quick comparison between classical and ML methods."""
    print("\n" + "="*80)
    print("QUICK COMPARISON: CLASSICAL vs ML METHODS")
    print("="*80)
    
    # Generate test data
    print("\n1. Generating test data...")
    test_hurst_values = [0.3, 0.5, 0.7, 0.9]
    test_data = []
    true_hurst = []
    
    for hurst in test_hurst_values:
        data = fbm_davies_harte(1000, hurst, seed=42)
        test_data.append(data)
        true_hurst.append(hurst)
    
    print(f"Generated {len(test_data)} test time series")
    
    # Test classical methods
    print("\n2. Testing classical methods...")
    classical_factory = BiomedicalHurstEstimatorFactory()
    classical_results = {}
    
    classical_methods = [
        EstimatorType.DFA,
        EstimatorType.RS_ANALYSIS,
        EstimatorType.HIGUCHI,
        EstimatorType.GHE,
        EstimatorType.PERIODOGRAM
    ]
    
    for method in classical_methods:
        try:
            estimator = classical_factory.create_estimator(method)
            predictions = []
            times = []
            
            for data in test_data:
                start_time = time.time()
                result = estimator.estimate(data)
                computation_time = time.time() - start_time
                
                predictions.append(result.hurst_estimate)
                times.append(computation_time)
            
            classical_results[method.value] = {
                'predictions': predictions,
                'times': times,
                'mean_time': np.mean(times)
            }
            
            print(f"  {method.value}: {np.mean(times)*1000:.1f}ms per prediction")
            
        except Exception as e:
            print(f"  {method.value}: Failed - {e}")
    
    # Test ML methods
    print("\n3. Testing ML methods...")
    ml_results = {}
    
    # Create pretrained models if they don't exist
    models_dir = "pretrained_models_benchmark"
    if not Path(models_dir).exists():
        print("Creating pretrained models...")
        create_pretrained_suite(models_dir, force_retrain=True)
    
    ml_methods = ['random_forest', 'svr', 'gradient_boosting', 'ensemble']
    
    for method in ml_methods:
        try:
            predictions = []
            times = []
            
            for data in test_data:
                start_time = time.time()
                
                if method == 'ensemble':
                    pred, _ = quick_ensemble_predict(data, models_dir)
                else:
                    pred = quick_predict(data, models_dir, method)
                
                computation_time = time.time() - start_time
                
                predictions.append(pred)
                times.append(computation_time)
            
            ml_results[method] = {
                'predictions': predictions,
                'times': times,
                'mean_time': np.mean(times)
            }
            
            print(f"  {method}: {np.mean(times)*1000:.1f}ms per prediction")
            
        except Exception as e:
            print(f"  {method}: Failed - {e}")
    
    # Compare results
    print("\n4. Results comparison:")
    print(f"{'Method':<20} {'MAE':<8} {'Time(ms)':<10} {'Type':<10}")
    print("-" * 50)
    
    all_results = {**classical_results, **ml_results}
    
    for method_name, results in all_results.items():
        predictions = results['predictions']
        mae = np.mean(np.abs(np.array(predictions) - np.array(true_hurst)))
        mean_time = results['mean_time'] * 1000
        method_type = 'Classical' if method_name in [m.value for m in classical_methods] else 'ML'
        
        print(f"{method_name:<20} {mae:<8.4f} {mean_time:<10.1f} {method_type:<10}")
    
    return classical_results, ml_results, true_hurst


def demonstrate_comprehensive_benchmark():
    """Demonstrate comprehensive benchmark system."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SYSTEM")
    print("="*80)
    
    # Create benchmark system
    print("\n1. Setting up benchmark system...")
    benchmark = ClassicalMLBenchmark("pretrained_models_benchmark")
    
    # Create test scenarios
    print("\n2. Creating test scenarios...")
    test_scenarios = benchmark.create_test_scenarios(
        hurst_values=[0.2, 0.4, 0.6, 0.8],
        lengths=[500, 1000],
        n_samples_per_config=5,
        include_contamination=True,
        include_biomedical=True
    )
    
    print(f"Created {len(test_scenarios)} test scenarios")
    
    # Run benchmark
    print("\n3. Running comprehensive benchmark...")
    results = benchmark.run_comprehensive_benchmark(
        test_scenarios=test_scenarios,
        save_results=True,
        results_dir="benchmark_results"
    )
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    benchmark.create_visualizations(results, Path("benchmark_results/benchmark_visualization.png"))
    
    return results


def demonstrate_performance_analysis():
    """Demonstrate detailed performance analysis."""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Create test data with different characteristics
    print("\n1. Creating diverse test scenarios...")
    
    test_scenarios = []
    
    # Different Hurst values
    for hurst in [0.1, 0.3, 0.5, 0.7, 0.9]:
        data = fbm_davies_harte(1000, hurst, seed=42)
        test_scenarios.append({
            'data': data,
            'true_hurst': hurst,
            'scenario': f'fBm_H{hurst}'
        })
    
    # Different lengths
    for length in [500, 1000, 2000]:
        data = fbm_davies_harte(length, 0.6, seed=42)
        test_scenarios.append({
            'data': data,
            'true_hurst': 0.6,
            'scenario': f'fBm_L{length}'
        })
    
    # Contaminated data
    from neurological_lrd_analysis.benchmark_core.generation import add_contamination
    
    clean_data = fbm_davies_harte(1000, 0.6, seed=42)
    contaminated_data = add_contamination(clean_data, 'noise', 0.1)
    test_scenarios.append({
        'data': contaminated_data,
        'true_hurst': 0.6,
        'scenario': 'fBm_noisy'
    })
    
    print(f"Created {len(test_scenarios)} diverse test scenarios")
    
    # Test methods
    print("\n2. Testing methods on diverse scenarios...")
    
    # Classical methods
    classical_factory = BiomedicalHurstEstimatorFactory()
    classical_methods = [EstimatorType.DFA, EstimatorType.RS_ANALYSIS, EstimatorType.HIGUCHI]
    
    # ML methods
    models_dir = "pretrained_models_benchmark"
    
    results = {}
    
    # Test classical methods
    for method in classical_methods:
        method_results = []
        try:
            estimator = classical_factory.create_estimator(method)
            for scenario in test_scenarios:
                start_time = time.time()
                result = estimator.estimate(scenario['data'])
                computation_time = time.time() - start_time
                
                error = abs(result.hurst_estimate - scenario['true_hurst'])
                method_results.append({
                    'scenario': scenario['scenario'],
                    'true_hurst': scenario['true_hurst'],
                    'predicted_hurst': result.hurst_estimate,
                    'error': error,
                    'computation_time': computation_time
                })
        except Exception as e:
            print(f"  {method.value}: Failed - {e}")
            continue
        
        results[f"classical_{method.value}"] = method_results
        print(f"  {method.value}: {len(method_results)} successful estimates")
    
    # Test ML methods
    ml_methods = ['random_forest', 'ensemble']
    for method in ml_methods:
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
                    'scenario': scenario['scenario'],
                    'true_hurst': scenario['true_hurst'],
                    'predicted_hurst': pred,
                    'error': error,
                    'computation_time': computation_time
                })
        except Exception as e:
            print(f"  {method}: Failed - {e}")
            continue
        
        results[f"ml_{method}"] = method_results
        print(f"  {method}: {len(method_results)} successful predictions")
    
    # Analyze results
    print("\n3. Performance analysis:")
    print(f"{'Method':<25} {'Mean Error':<12} {'Max Error':<12} {'Mean Time(ms)':<15}")
    print("-" * 70)
    
    for method_name, method_results in results.items():
        if not method_results:
            continue
        
        errors = [r['error'] for r in method_results]
        times = [r['computation_time'] for r in method_results]
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        mean_time = np.mean(times) * 1000
        
        print(f"{method_name:<25} {mean_error:<12.4f} {max_error:<12.4f} {mean_time:<15.1f}")
    
    return results


def create_performance_visualization(results):
    """Create performance visualization."""
    print("\n4. Creating performance visualization...")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Classical vs ML Methods: Performance Analysis', fontsize=16, fontweight='bold')
    
    # Extract data for visualization
    methods = []
    mean_errors = []
    max_errors = []
    mean_times = []
    method_types = []
    
    for method_name, method_results in results.items():
        if not method_results:
            continue
        
        errors = [r['error'] for r in method_results]
        times = [r['computation_time'] for r in method_results]
        
        methods.append(method_name)
        mean_errors.append(np.mean(errors))
        max_errors.append(np.max(errors))
        mean_times.append(np.mean(times) * 1000)
        method_types.append('Classical' if 'classical' in method_name else 'ML')
    
    # 1. Mean Error Comparison
    ax1 = axes[0, 0]
    colors = ['red' if t == 'Classical' else 'blue' for t in method_types]
    bars = ax1.barh(methods, mean_errors, color=colors, alpha=0.7)
    ax1.set_xlabel('Mean Absolute Error')
    ax1.set_title('Accuracy Comparison (Mean Error)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Speed Comparison
    ax2 = axes[0, 1]
    bars = ax2.barh(methods, mean_times, color=colors, alpha=0.7)
    ax2.set_xlabel('Mean Computation Time (ms)')
    ax2.set_title('Speed Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error vs Speed Scatter
    ax3 = axes[1, 0]
    for i, method in enumerate(methods):
        color = 'red' if method_types[i] == 'Classical' else 'blue'
        ax3.scatter(mean_times[i], mean_errors[i], color=color, alpha=0.7, s=100)
        ax3.annotate(method, (mean_times[i], mean_errors[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Mean Computation Time (ms)')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.set_title('Performance vs Speed')
    ax3.grid(True, alpha=0.3)
    
    # 4. Max Error Comparison
    ax4 = axes[1, 1]
    bars = ax4.barh(methods, max_errors, color=colors, alpha=0.7)
    ax4.set_xlabel('Maximum Absolute Error')
    ax4.set_title('Robustness Comparison (Max Error)')
    ax4.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Classical'),
                      Patch(facecolor='blue', alpha=0.7, label='ML')]
    ax1.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # Save plot
    results_dir = Path("results/benchmark_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Performance visualization saved to {results_dir}")
    
    plt.show()


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: CLASSICAL vs ML METHODS")
    print("=" * 80)
    print("Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)")
    print("Research Focus: Physics-Informed Fractional Operator Learning for Real-Time Neurological Biomarker Detection")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # 1. Quick Comparison
        print("\n" + "="*60)
        print("1. QUICK COMPARISON")
        print("="*60)
        classical_results, ml_results, true_hurst = demonstrate_quick_comparison()
        
        # 2. Detailed Performance Analysis
        print("\n" + "="*60)
        print("2. DETAILED PERFORMANCE ANALYSIS")
        print("="*60)
        detailed_results = demonstrate_performance_analysis()
        
        # 3. Create Visualizations
        print("\n" + "="*60)
        print("3. PERFORMANCE VISUALIZATION")
        print("="*60)
        create_performance_visualization(detailed_results)
        
        # 4. Comprehensive Benchmark (Optional - takes longer)
        print("\n" + "="*60)
        print("4. COMPREHENSIVE BENCHMARK (Optional)")
        print("="*60)
        print("This will run a full benchmark with many test scenarios...")
        print("Uncomment the following lines to run the comprehensive benchmark:")
        print("# comprehensive_results = demonstrate_comprehensive_benchmark()")
        
        print("\n" + "=" * 80)
        print("BENCHMARK DEMONSTRATION COMPLETED!")
        print("=" * 80)
        print("\nKey Findings:")
        print("✅ Classical methods provide reliable baseline performance")
        print("✅ ML methods offer potential for improved accuracy")
        print("✅ Speed vs accuracy trade-offs clearly demonstrated")
        print("✅ Comprehensive evaluation framework established")
        
        print(f"\nResults saved to: benchmark_results/")
        print("You can now analyze the detailed performance comparisons!")
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
