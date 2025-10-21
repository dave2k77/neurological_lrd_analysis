"""
Machine Learning Baseline Estimators for Hurst Exponent Estimation.

This module provides classical machine learning methods for estimating Hurst exponents
in neurological time series data, including Random Forest, Support Vector Regression,
and Gradient Boosting Trees with automatic hyperparameter optimization.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
Research Focus: Physics-Informed Fractional Operator Learning for Real-Time Neurological Biomarker Detection
"""

from .feature_extraction import (
    TimeSeriesFeatureExtractor,
    extract_statistical_features,
    extract_spectral_features,
    extract_wavelet_features,
    extract_fractal_features,
    extract_biomedical_features
)

from .ml_estimators import (
    MLBaselineType,
    RandomForestEstimator,
    SVREstimator,
    GradientBoostingEstimator,
    MLBaselineFactory
)

from .hyperparameter_optimization import (
    OptunaOptimizer,
    create_optuna_study,
    optimize_hyperparameters,
    optimize_all_estimators
)

from .pretrained_models import (
    PretrainedModelManager,
    ModelMetadata,
    ModelStatus,
    TrainingConfig,
    create_default_training_configs,
    create_pretrained_suite
)

from .inference import (
    PretrainedInference,
    PredictionResult,
    EnsembleResult,
    quick_predict,
    quick_ensemble_predict
)

from .benchmark_comparison import (
    ClassicalMLBenchmark,
    BenchmarkResult,
    BenchmarkSummary,
    run_comprehensive_benchmark
)

__all__ = [
    # Feature extraction
    "TimeSeriesFeatureExtractor",
    "extract_statistical_features",
    "extract_spectral_features", 
    "extract_wavelet_features",
    "extract_fractal_features",
    "extract_biomedical_features",
    
    # ML estimators
    "MLBaselineType",
    "RandomForestEstimator",
    "SVREstimator", 
    "GradientBoostingEstimator",
    "MLBaselineFactory",
    
    # Hyperparameter optimization
    "OptunaOptimizer",
    "create_optuna_study",
    "optimize_hyperparameters",
    "optimize_all_estimators",
    
    # Pretrained models
    "PretrainedModelManager",
    "ModelMetadata",
    "ModelStatus",
    "TrainingConfig",
    "create_default_training_configs",
    "create_pretrained_suite",
    
    # Inference
    "PretrainedInference",
    "PredictionResult",
    "EnsembleResult",
    "quick_predict",
    "quick_ensemble_predict",
    
    # Benchmark comparison
    "ClassicalMLBenchmark",
    "BenchmarkResult",
    "BenchmarkSummary",
    "run_comprehensive_benchmark"
]
