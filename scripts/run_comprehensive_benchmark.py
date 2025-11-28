#!/usr/bin/env python3
"""
Comprehensive Benchmarking Script for Classical Estimators.

This script runs a comprehensive benchmark of classical LRD estimators across:
1. Pure Synthetic Data (fBm, fGn, ARFIMA)
2. Contaminated Data (Noise, Trends, Artifacts)
3. Biomedical Scenarios (EEG, ECG, Respiratory)

It leverages the integrated lrdbenchmark backend where available.
"""

import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neurological_lrd_analysis.benchmark_core.generation import generate_grid
from neurological_lrd_analysis.benchmark_core.runner import BenchmarkConfig, run_benchmark_on_dataset, create_leaderboard
from neurological_lrd_analysis.biomedical_hurst_factory import EstimatorType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark_run.log')
    ]
)
logger = logging.getLogger(__name__)

def run_comprehensive_benchmark():
    """Execute the comprehensive benchmark."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"benchmark_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting comprehensive benchmark. Results will be saved to {output_dir}")
    
    # =========================================================================
    # 1. Pure Synthetic Data Benchmark
    # =========================================================================
    logger.info("Generating Pure Synthetic Data...")
    pure_datasets = generate_grid(
        hurst_values=[0.4, 0.5, 0.6, 0.7, 0.8],
        lengths=[1024, 4096],
        contaminations=['none'],
        generators=['fbm', 'fgn', 'arfima'],
        seed=42
    )
    logger.info(f"Generated {len(pure_datasets)} pure datasets.")
    
    # =========================================================================
    # 2. Contaminated Data Benchmark
    # =========================================================================
    logger.info("Generating Contaminated Data...")
    contaminated_datasets = generate_grid(
        hurst_values=[0.5, 0.7, 0.8],
        lengths=[2048],
        contaminations=[
            'noise', 'trend', 'baseline_drift', 
            'electrode_pop', 'motion', 'powerline',
            'heavy_tail', 'neural_avalanche'
        ],
        contamination_level=0.1,
        generators=['fbm'],
        seed=123
    )
    logger.info(f"Generated {len(contaminated_datasets)} contaminated datasets.")
    
    # =========================================================================
    # 3. Biomedical Scenarios Benchmark
    # =========================================================================
    logger.info("Generating Biomedical Scenarios...")
    scenario_datasets = generate_grid(
        hurst_values=[0.6, 0.7, 0.8],
        lengths=[2048],
        contaminations=['none', 'noise'],
        contamination_level=0.05,
        biomedical_scenarios=[
            'eeg_rest', 'eeg_sleep', 
            'ecg_normal', 'ecg_tachycardia',
            'respiratory_rest'
        ],
        seed=456
    )
    logger.info(f"Generated {len(scenario_datasets)} scenario datasets.")
    
    # Combine all datasets
    all_datasets = pure_datasets + contaminated_datasets + scenario_datasets
    logger.info(f"Total datasets to benchmark: {len(all_datasets)}")
    
    # =========================================================================
    # Configure and Run Benchmark
    # =========================================================================
    config = BenchmarkConfig(
        output_dir=str(output_dir),
        n_bootstrap=0,  # Disable bootstrap for speed in this comprehensive run
        confidence_level=0.95,
        save_results=True,
        verbose=True
    )
    
    logger.info("Running benchmark (this may take some time)...")
    results = run_benchmark_on_dataset(all_datasets, config)
    
    # =========================================================================
    # Analysis and Reporting
    # =========================================================================
    logger.info("Analyzing results...")
    
    # Global Leaderboard
    leaderboard = create_leaderboard(results)
    leaderboard.to_csv(output_dir / "global_leaderboard.csv", index=False)
    print("\nGlobal Leaderboard:")
    print(leaderboard.to_string(index=False))
    
    # Segmented Analysis
    results_df = pd.DataFrame([r.to_dict() for r in results])
    
    # 1. Performance by Data Type (Pure vs Contaminated vs Scenario)
    # We need to infer type from contamination/generator strings if not explicitly stored
    def classify_type(row):
        if "eeg" in row['contamination'] or "ecg" in row['contamination'] or "respiratory" in row['contamination']:
            return "Biomedical Scenario"
        elif row['contamination'] == 'none':
            return "Pure Synthetic"
        else:
            return "Contaminated Synthetic"
            
    results_df['data_category'] = results_df.apply(classify_type, axis=1)
    
    print("\nPerformance by Data Category (MAE):")
    category_perf = results_df.groupby(['data_category', 'estimator'])['absolute_error'].mean().unstack()
    print(category_perf)
    category_perf.to_csv(output_dir / "performance_by_category.csv")
    
    # 2. Performance by Contamination Type
    print("\nPerformance by Contamination Type (MAE):")
    contam_perf = results_df.groupby(['contamination', 'estimator'])['absolute_error'].mean().unstack()
    print(contam_perf)
    contam_perf.to_csv(output_dir / "performance_by_contamination.csv")
    
    logger.info(f"Benchmark complete. All results saved to {output_dir}")

if __name__ == "__main__":
    run_comprehensive_benchmark()
