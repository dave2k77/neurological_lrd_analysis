#!/usr/bin/env python3
"""
Demonstration script for ML baseline estimators.

This script demonstrates the usage of machine learning baseline estimators
for Hurst exponent estimation, including feature extraction, model training,
hyperparameter optimization, and performance evaluation.

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
    RandomForestEstimator,
    SVREstimator,
    GradientBoostingEstimator,
    MLBaselineFactory,
    TimeSeriesFeatureExtractor,
    OptunaOptimizer,
    optimize_hyperparameters,
    optimize_all_estimators
)

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def generate_training_data(n_samples_per_hurst: int = 50, 
                          hurst_values: list = None,
                          data_length: int = 1000) -> tuple:
    """
    Generate training data for ML models.
    
    Parameters:
    -----------
    n_samples_per_hurst : int
        Number of samples per Hurst value
    hurst_values : list, optional
        List of Hurst values to generate
    data_length : int
        Length of each time series
        
    Returns:
    --------
    tuple
        (X, y) feature matrix and target values
    """
    if hurst_values is None:
        hurst_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"Generating training data...")
    print(f"  - Hurst values: {hurst_values}")
    print(f"  - Samples per Hurst: {n_samples_per_hurst}")
    print(f"  - Data length: {data_length}")
    
    # Generate synthetic data
    samples = generate_grid(
        hurst_values=hurst_values,
        lengths=[data_length],
        contaminations=['none'],
        generators=['fbm']
    )
    
    # Extract features
    extractor = TimeSeriesFeatureExtractor()
    X = []
    y = []
    
    print("Extracting features...")
    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(samples)} samples")
        
        features = extractor.extract_features(sample.data, sample.true_hurst)
        X.append(features.combined)
        y.append(sample.true_hurst)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    return X, y


def train_ml_models(X: np.ndarray, y: np.ndarray, 
                   validation_split: float = 0.2) -> dict:
    """
    Train all ML baseline models.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    validation_split : float
        Fraction of data for validation
        
    Returns:
    --------
    dict
        Dictionary of trained models and results
    """
    print("\nTraining ML baseline models...")
    
    # Create all estimators
    estimators = MLBaselineFactory.create_all_estimators()
    results = {}
    
    for estimator_type, estimator in estimators.items():
        print(f"\nTraining {estimator_type.value}...")
        start_time = time.time()
        
        try:
            result = estimator.train(X, y, validation_split=validation_split)
            training_time = time.time() - start_time
            
            results[estimator_type] = {
                'estimator': estimator,
                'result': result,
                'training_time': training_time
            }
            
            print(f"  Training score: {result.training_score:.4f}")
            print(f"  Validation score: {result.validation_score:.4f}")
            print(f"  Cross-validation MSE: {np.mean(result.cross_val_scores):.4f} ± {np.std(result.cross_val_scores):.4f}")
            print(f"  Training time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"  Failed to train {estimator_type.value}: {e}")
            results[estimator_type] = None
    
    return results


def optimize_hyperparameters_demo(X: np.ndarray, y: np.ndarray, 
                                  n_trials: int = 20) -> dict:
    """
    Demonstrate hyperparameter optimization.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    n_trials : int
        Number of optimization trials
        
    Returns:
    --------
    dict
        Optimization results
    """
    print(f"\nOptimizing hyperparameters (n_trials={n_trials})...")
    
    # Optimize all estimators
    optimization_results = optimize_all_estimators(
        X, y,
        estimator_types=['random_forest', 'svr', 'gradient_boosting'],
        n_trials=n_trials,
        cv_folds=3,
        random_state=42
    )
    
    print("\nOptimization Results:")
    for estimator_type, result in optimization_results.items():
        print(f"\n{estimator_type.upper()}:")
        print(f"  Best score: {result.best_score:.4f}")
        print(f"  Best parameters: {result.best_params}")
        print(f"  Optimization time: {result.optimization_time:.2f}s")
        print(f"  Number of trials: {result.n_trials}")
    
    return optimization_results


def evaluate_models(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate trained models on test data.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        Test target values
        
    Returns:
    --------
    dict
        Evaluation results
    """
    print("\nEvaluating models on test data...")
    
    evaluation_results = {}
    
    for estimator_type, model_info in models.items():
        if model_info is None:
            continue
        
        estimator = model_info['estimator']
        print(f"\n{estimator_type.value.upper()}:")
        
        # Make predictions
        predictions = []
        for i in range(len(X_test)):
            pred_result = estimator.predict(X_test[i:i+1])
            predictions.append(pred_result.hurst_estimate)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        r2 = 1 - mse / np.var(y_test)
        rmse = np.sqrt(mse)
        
        # Calculate correlation
        correlation = np.corrcoef(predictions, y_test)[0, 1]
        
        evaluation_results[estimator_type] = {
            'predictions': predictions,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'correlation': correlation
        }
        
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Correlation: {correlation:.4f}")
    
    return evaluation_results


def create_visualizations(evaluation_results: dict, y_test: np.ndarray, 
                         save_path: Optional[Path] = None) -> None:
    """
    Create visualizations of model performance.
    
    Parameters:
    -----------
    evaluation_results : dict
        Evaluation results
    y_test : np.ndarray
        Test target values
    save_path : Path, optional
        Path to save plots
    """
    print("\nCreating visualizations...")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ML Baseline Estimators Performance', fontsize=16, fontweight='bold')
    
    # 1. Predictions vs True values
    ax1 = axes[0, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (estimator_type, results) in enumerate(evaluation_results.items()):
        predictions = results['predictions']
        ax1.scatter(y_test, predictions, alpha=0.6, label=estimator_type.value, 
                   color=colors[i % len(colors)], s=50)
    
    # Perfect prediction line
    min_val = min(y_test.min(), min(r['predictions'].min() for r in evaluation_results.values()))
    max_val = max(y_test.max(), max(r['predictions'].max() for r in evaluation_results.values()))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('True Hurst Exponent')
    ax1.set_ylabel('Predicted Hurst Exponent')
    ax1.set_title('Predictions vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax2 = axes[0, 1]
    errors_data = []
    labels_data = []
    
    for estimator_type, results in evaluation_results.items():
        errors = results['predictions'] - y_test
        errors_data.extend(errors)
        labels_data.extend([estimator_type.value] * len(errors))
    
    # Create box plot
    import pandas as pd
    df_errors = pd.DataFrame({'Error': errors_data, 'Model': labels_data})
    sns.boxplot(data=df_errors, x='Model', y='Error', ax=ax2)
    ax2.set_title('Error Distribution')
    ax2.set_ylabel('Prediction Error')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Performance metrics comparison
    ax3 = axes[1, 0]
    metrics = ['mse', 'mae', 'r2', 'correlation']
    metric_names = ['MSE', 'MAE', 'R²', 'Correlation']
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (estimator_type, results) in enumerate(evaluation_results.items()):
        values = [results[metric] for metric in metrics]
        ax3.bar(x + i * width, values, width, label=estimator_type.value, 
               color=colors[i % len(colors)], alpha=0.8)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Values')
    ax3.set_title('Performance Metrics Comparison')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(metric_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature importance (for Random Forest)
    ax4 = axes[1, 1]
    if MLBaselineType.RANDOM_FOREST in evaluation_results:
        # Get feature importance from the trained model
        # This would require access to the trained model
        ax4.text(0.5, 0.5, 'Feature Importance\n(Would require model access)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Feature Importance (Random Forest)')
    else:
        ax4.text(0.5, 0.5, 'Feature Importance\nNot available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Feature Importance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    
    plt.show()


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("ML Baseline Estimators Demonstration")
    print("=" * 80)
    print("Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)")
    print("Research Focus: Physics-Informed Fractional Operator Learning for Real-Time Neurological Biomarker Detection")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate training data
    print("\n1. Generating Training Data")
    print("-" * 40)
    X, y = generate_training_data(
        n_samples_per_hurst=30,
        hurst_values=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        data_length=1000
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    print("\n2. Training ML Models")
    print("-" * 40)
    models = train_ml_models(X_train, y_train, validation_split=0.2)
    
    # Optimize hyperparameters (optional)
    print("\n3. Hyperparameter Optimization")
    print("-" * 40)
    try:
        optimization_results = optimize_hyperparameters_demo(
            X_train, y_train, n_trials=10
        )
    except Exception as e:
        print(f"Hyperparameter optimization failed: {e}")
        print("Continuing with default parameters...")
        optimization_results = {}
    
    # Evaluate models
    print("\n4. Model Evaluation")
    print("-" * 40)
    evaluation_results = evaluate_models(models, X_test, y_test)
    
    # Create visualizations
    print("\n5. Creating Visualizations")
    print("-" * 40)
    results_dir = Path("results/ml_baselines_demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    create_visualizations(
        evaluation_results, y_test, 
        save_path=results_dir / "ml_baselines_performance.png"
    )
    
    # Summary
    print("\n6. Summary")
    print("-" * 40)
    print("ML Baseline Estimators Performance Summary:")
    print()
    
    for estimator_type, results in evaluation_results.items():
        print(f"{estimator_type.value.upper()}:")
        print(f"  R² Score: {results['r2']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  Correlation: {results['correlation']:.4f}")
        print()
    
    # Find best model
    best_model = max(evaluation_results.items(), key=lambda x: x[1]['r2'])
    print(f"Best performing model: {best_model[0].value.upper()}")
    print(f"Best R² score: {best_model[1]['r2']:.4f}")
    
    print("\n" + "=" * 80)
    print("Demonstration completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
