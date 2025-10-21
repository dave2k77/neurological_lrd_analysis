#!/usr/bin/env python3
"""
Demonstration script for pretrained model system.

This script demonstrates the complete pretrained model workflow including
model training, storage, loading, and inference for Hurst exponent estimation.

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
    
    # ML baselines
    MLBaselineType,
    PretrainedModelManager,
    PretrainedInference,
    TrainingConfig,
    create_default_training_configs,
    create_pretrained_suite,
    quick_predict,
    quick_ensemble_predict
)

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def demonstrate_model_training():
    """Demonstrate model training and storage."""
    print("\n" + "="*60)
    print("1. MODEL TRAINING AND STORAGE DEMONSTRATION")
    print("="*60)
    
    # Create model manager
    models_dir = "pretrained_models_demo"
    manager = PretrainedModelManager(models_dir)
    
    print(f"Created model manager in: {models_dir}")
    
    # Create comprehensive training dataset
    print("\nCreating comprehensive training dataset...")
    X, y, training_info = manager.create_training_data(
        hurst_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lengths=[500, 1000],
        n_samples_per_config=20,
        generators=['fbm', 'fgn', 'arfima'],
        contaminations=['none', 'noise', 'missing'],
        biomedical_scenarios=['eeg', 'ecg']
    )
    
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    print(f"Hurst range: {training_info['hurst_range']}")
    
    # Create training configurations
    print("\nCreating training configurations...")
    training_configs = create_default_training_configs()
    
    # Train models
    print(f"\nTraining {len(training_configs)} models...")
    results = manager.create_model_suite(training_configs, X, y, training_info)
    
    print(f"\nSuccessfully trained {len(results)} models:")
    for result in results:
        print(f"  - {result.model_type.value}: {result.model_id}")
        print(f"    Training score: {result.performance_metrics['training_score']:.4f}")
        print(f"    Validation score: {result.performance_metrics['validation_score']:.4f}")
    
    return manager, X, y


def demonstrate_model_inference(manager):
    """Demonstrate model inference capabilities."""
    print("\n" + "="*60)
    print("2. MODEL INFERENCE DEMONSTRATION")
    print("="*60)
    
    # Create inference system
    inference = PretrainedInference(manager.models_dir)
    
    # Generate test data
    print("\nGenerating test data...")
    test_hurst_values = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    test_data = []
    true_hurst = []
    
    for hurst in test_hurst_values:
        data = fbm_davies_harte(1000, hurst, seed=42)
        test_data.append(data)
        true_hurst.append(hurst)
    
    print(f"Generated {len(test_data)} test time series")
    
    # Single predictions
    print("\nSingle predictions:")
    single_predictions = []
    for i, data in enumerate(test_data):
        result = inference.predict_single(data)
        single_predictions.append(result.hurst_estimate)
        print(f"  True: {true_hurst[i]:.2f}, Predicted: {result.hurst_estimate:.3f}, "
              f"Model: {result.model_type}, Time: {result.prediction_time:.3f}s")
    
    # Batch predictions
    print("\nBatch predictions:")
    batch_results = inference.predict_batch(test_data, show_progress=True)
    batch_predictions = [r.hurst_estimate for r in batch_results]
    
    # Ensemble predictions
    print("\nEnsemble predictions:")
    ensemble_predictions = []
    ensemble_uncertainties = []
    
    for data in test_data:
        result = inference.predict_ensemble(data)
        ensemble_predictions.append(result.mean_estimate)
        ensemble_uncertainties.append(result.std_estimate)
        print(f"  Mean: {result.mean_estimate:.3f} ± {result.std_estimate:.3f}")
    
    # Model comparison
    print("\nModel comparison for first test case:")
    comparison = inference.compare_models(test_data[0])
    for model_type, result in comparison.items():
        print(f"  {model_type}: {result.hurst_estimate:.3f}")
    
    return {
        'test_data': test_data,
        'true_hurst': true_hurst,
        'single_predictions': single_predictions,
        'batch_predictions': batch_predictions,
        'ensemble_predictions': ensemble_predictions,
        'ensemble_uncertainties': ensemble_uncertainties
    }


def demonstrate_quick_functions(test_data, true_hurst):
    """Demonstrate quick prediction functions."""
    print("\n" + "="*60)
    print("3. QUICK PREDICTION FUNCTIONS DEMONSTRATION")
    print("="*60)
    
    # Quick single prediction
    print("\nQuick single prediction:")
    test_sample = test_data[0]
    quick_pred = quick_predict(test_sample, "pretrained_models_demo")
    print(f"  Quick prediction: {quick_pred:.3f} (True: {true_hurst[0]:.2f})")
    
    # Quick ensemble prediction
    print("\nQuick ensemble prediction:")
    mean_est, std_est = quick_ensemble_predict(test_sample, "pretrained_models_demo")
    print(f"  Ensemble: {mean_est:.3f} ± {std_est:.3f} (True: {true_hurst[0]:.2f})")
    
    # Batch quick predictions
    print("\nBatch quick predictions:")
    quick_predictions = []
    for i, data in enumerate(test_data[:3]):  # Test first 3 samples
        pred = quick_predict(data, "pretrained_models_demo")
        quick_predictions.append(pred)
        print(f"  Sample {i+1}: {pred:.3f} (True: {true_hurst[i]:.2f})")


def demonstrate_model_management(manager):
    """Demonstrate model management capabilities."""
    print("\n" + "="*60)
    print("4. MODEL MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    # List all models
    print("\nAll available models:")
    all_models = manager.list_models()
    for model in all_models:
        print(f"  - {model.model_id}: {model.model_type.value}")
        print(f"    Status: {model.status.value}")
        print(f"    Validation score: {model.performance_metrics.get('validation_score', 'N/A'):.4f}")
        print(f"    Created: {model.created_at}")
    
    # List models by type
    print(f"\nRandom Forest models:")
    rf_models = manager.list_models(model_type=MLBaselineType.RANDOM_FOREST)
    for model in rf_models:
        print(f"  - {model.model_id}")
    
    # Get best model
    print(f"\nBest Random Forest model:")
    try:
        best_estimator, best_metadata = manager.get_best_model(MLBaselineType.RANDOM_FOREST)
        print(f"  Model ID: {best_metadata.model_id}")
        print(f"  Validation score: {best_metadata.performance_metrics['validation_score']:.4f}")
        print(f"  Hyperparameters: {best_metadata.hyperparameters}")
    except ValueError as e:
        print(f"  No Random Forest models available: {e}")
    
    # Model information
    print(f"\nModel information:")
    if all_models:
        model_info = manager._metadata_registry[all_models[0].model_id]
        print(f"  Model ID: {model_info.model_id}")
        print(f"  Type: {model_info.model_type.value}")
        print(f"  Version: {model_info.version}")
        print(f"  Author: {model_info.author}")
        print(f"  License: {model_info.license}")
        print(f"  Description: {model_info.description}")
        print(f"  Tags: {model_info.tags}")


def create_visualizations(results):
    """Create visualizations of prediction results."""
    print("\n" + "="*60)
    print("5. CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pretrained Model Performance Analysis', fontsize=16, fontweight='bold')
    
    true_hurst = results['true_hurst']
    single_preds = results['single_predictions']
    ensemble_preds = results['ensemble_predictions']
    ensemble_uncertainties = results['ensemble_uncertainties']
    
    # 1. Predictions vs True values
    ax1 = axes[0, 0]
    ax1.scatter(true_hurst, single_preds, alpha=0.7, label='Single Model', s=60)
    ax1.scatter(true_hurst, ensemble_preds, alpha=0.7, label='Ensemble', s=60)
    
    # Perfect prediction line
    min_val = min(min(true_hurst), min(single_preds), min(ensemble_preds))
    max_val = max(max(true_hurst), max(single_preds), max(ensemble_preds))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('True Hurst Exponent')
    ax1.set_ylabel('Predicted Hurst Exponent')
    ax1.set_title('Predictions vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error analysis
    ax2 = axes[0, 1]
    single_errors = np.array(single_preds) - np.array(true_hurst)
    ensemble_errors = np.array(ensemble_preds) - np.array(true_hurst)
    
    ax2.scatter(true_hurst, single_errors, alpha=0.7, label='Single Model', s=60)
    ax2.scatter(true_hurst, ensemble_errors, alpha=0.7, label='Ensemble', s=60)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)
    
    ax2.set_xlabel('True Hurst Exponent')
    ax2.set_ylabel('Prediction Error')
    ax2.set_title('Error Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Uncertainty visualization
    ax3 = axes[1, 0]
    ax3.errorbar(true_hurst, ensemble_preds, yerr=ensemble_uncertainties, 
                fmt='o', alpha=0.7, capsize=5, label='Ensemble with Uncertainty')
    ax3.scatter(true_hurst, single_preds, alpha=0.7, label='Single Model', s=60)
    
    # Perfect prediction line
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('True Hurst Exponent')
    ax3.set_ylabel('Predicted Hurst Exponent')
    ax3.set_title('Uncertainty Quantification')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance metrics
    ax4 = axes[1, 1]
    
    # Calculate metrics
    single_mse = np.mean((np.array(single_preds) - np.array(true_hurst))**2)
    ensemble_mse = np.mean((np.array(ensemble_preds) - np.array(true_hurst))**2)
    
    single_mae = np.mean(np.abs(np.array(single_preds) - np.array(true_hurst)))
    ensemble_mae = np.mean(np.abs(np.array(ensemble_preds) - np.array(true_hurst)))
    
    single_corr = np.corrcoef(single_preds, true_hurst)[0, 1]
    ensemble_corr = np.corrcoef(ensemble_preds, true_hurst)[0, 1]
    
    metrics = ['MSE', 'MAE', 'Correlation']
    single_values = [single_mse, single_mae, single_corr]
    ensemble_values = [ensemble_mse, ensemble_mae, ensemble_corr]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, single_values, width, label='Single Model', alpha=0.8)
    ax4.bar(x + width/2, ensemble_values, width, label='Ensemble', alpha=0.8)
    
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Values')
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    results_dir = Path("results/pretrained_models_demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "pretrained_models_performance.png", dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to {results_dir}")
    
    plt.show()


def demonstrate_advanced_features(manager):
    """Demonstrate advanced features of the pretrained model system."""
    print("\n" + "="*60)
    print("6. ADVANCED FEATURES DEMONSTRATION")
    print("="*60)
    
    # Model benchmarking
    print("\nModel benchmarking:")
    inference = PretrainedInference(manager.models_dir)
    
    # Create test dataset
    test_data = []
    test_hurst = []
    for hurst in [0.3, 0.5, 0.7]:
        for _ in range(3):
            data = fbm_davies_harte(500, hurst, seed=None)
            test_data.append(data)
            test_hurst.append(hurst)
    
    # Benchmark all models
    benchmark_results = inference.benchmark_models(test_data, test_hurst)
    
    print("Benchmark Results:")
    for model_type, metrics in benchmark_results.items():
        print(f"  {model_type}:")
        print(f"    MSE: {metrics['mse']:.4f}")
        print(f"    MAE: {metrics['mae']:.4f}")
        print(f"    R²: {metrics['r2']:.4f}")
        print(f"    Correlation: {metrics['correlation']:.4f}")
    
    # Model cleanup demonstration
    print(f"\nModel cleanup (keeping best 2 models per type):")
    initial_count = len(manager.list_models())
    print(f"  Initial models: {initial_count}")
    
    manager.cleanup_models(keep_best=True, max_models_per_type=2)
    
    final_count = len(manager.list_models())
    print(f"  Final models: {final_count}")
    print(f"  Removed: {initial_count - final_count} models")


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("PRETRAINED MODEL SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)")
    print("Research Focus: Physics-Informed Fractional Operator Learning for Real-Time Neurological Biomarker Detection")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # 1. Model Training and Storage
        manager, X, y = demonstrate_model_training()
        
        # 2. Model Inference
        results = demonstrate_model_inference(manager)
        
        # 3. Quick Functions
        demonstrate_quick_functions(results['test_data'], results['true_hurst'])
        
        # 4. Model Management
        demonstrate_model_management(manager)
        
        # 5. Visualizations
        create_visualizations(results)
        
        # 6. Advanced Features
        demonstrate_advanced_features(manager)
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("✅ Model training and storage")
        print("✅ Single and batch predictions")
        print("✅ Ensemble predictions with uncertainty")
        print("✅ Quick prediction functions")
        print("✅ Model management and metadata")
        print("✅ Performance visualization")
        print("✅ Advanced benchmarking")
        print("✅ Model cleanup and optimization")
        
        print(f"\nPretrained models stored in: {manager.models_dir}")
        print("You can now use these models for quick Hurst exponent estimation!")
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
