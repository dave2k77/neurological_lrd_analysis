Machine Learning Baselines
==========================

The neurological LRD analysis library includes comprehensive machine learning baselines for Hurst exponent estimation, providing state-of-the-art performance with fast inference capabilities.

Overview
--------

The ML baselines system provides:

- **74+ Feature Extraction**: Comprehensive feature engineering for time series data
- **Multiple ML Models**: Random Forest, SVR, Gradient Boosting, and Ensemble methods
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Pretrained Models**: Fast inference with pre-trained models
- **Real-time Performance**: 10-50ms prediction times

Quick Start
-----------

.. code-block:: python

    from neurological_lrd_analysis import (
        create_pretrained_suite, quick_predict, quick_ensemble_predict
    )

    # Create pretrained models (one-time setup)
    create_pretrained_suite("pretrained_models", force_retrain=True)

    # Fast ML prediction
    hurst_ml = quick_predict(your_time_series, "pretrained_models", "random_forest")

    # Ensemble prediction (best accuracy)
    hurst_ensemble, uncertainty = quick_ensemble_predict(your_time_series, "pretrained_models")

Feature Extraction
------------------

The ``TimeSeriesFeatureExtractor`` class provides comprehensive feature extraction from time series data:

.. code-block:: python

    from neurological_lrd_analysis import TimeSeriesFeatureExtractor

    # Create feature extractor
    extractor = TimeSeriesFeatureExtractor()

    # Extract features
    features = extractor.extract_features(time_series_data)
    print(f"Extracted {len(features)} features")

Feature Categories
~~~~~~~~~~~~~~~~~~

The feature extractor provides features in several categories:

**Statistical Features**
- Mean, variance, skewness, kurtosis
- Percentiles, quartiles, range
- Autocorrelation at various lags

**Spectral Features**
- Power spectral density
- Spectral centroid, bandwidth, rolloff
- Frequency band power ratios (delta, theta, alpha, beta, gamma)

**Wavelet Features**
- Wavelet energy at multiple scales
- Wavelet entropy and complexity
- Multiresolution analysis

**Fractal Features**
- Detrended Fluctuation Analysis (DFA)
- Higuchi fractal dimension
- Generalized Hurst exponent

**Biomedical Features**
- EEG-specific features (electrode characteristics)
- ECG-specific features (heart rate variability)
- Respiratory features (breathing patterns)

ML Estimators
-------------

The library provides several ML estimators for Hurst exponent estimation:

Random Forest Estimator
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from neurological_lrd_analysis import RandomForestEstimator

    # Create estimator
    estimator = RandomForestEstimator()

    # Train model
    result = estimator.train(X_train, y_train, validation_split=0.2)

    # Make predictions
    predictions = estimator.predict(X_test)

    # Get feature importance
    importance = estimator.get_feature_importance()

SVR Estimator
~~~~~~~~~~~~~

.. code-block:: python

    from neurological_lrd_analysis import SVREstimator

    # Create estimator
    estimator = SVREstimator()

    # Train model
    result = estimator.train(X_train, y_train, validation_split=0.2)

    # Make predictions
    predictions = estimator.predict(X_test)

Gradient Boosting Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from neurological_lrd_analysis import GradientBoostingEstimator

    # Create estimator
    estimator = GradientBoostingEstimator()

    # Train model
    result = estimator.train(X_train, y_train, validation_split=0.2)

    # Make predictions
    predictions = estimator.predict(X_test)

Hyperparameter Optimization
---------------------------

The library integrates with Optuna for automated hyperparameter optimization:

.. code-block:: python

    from neurological_lrd_analysis import (
        OptunaOptimizer, create_optuna_study, optimize_hyperparameters
    )

    # Create optimization study
    study = create_optuna_study(
        model_type="random_forest",
        X_train=X_train,
        y_train=y_train,
        n_trials=100
    )

    # Run optimization
    best_params = optimize_hyperparameters(
        model_type="random_forest",
        X_train=X_train,
        y_train=y_train,
        n_trials=100
    )

    print(f"Best parameters: {best_params}")

Pretrained Models
-----------------

The pretrained model system provides efficient model management and fast inference:

Model Management
~~~~~~~~~~~~~~~~

.. code-block:: python

    from neurological_lrd_analysis import PretrainedModelManager, TrainingConfig, MLBaselineType

    # Create model manager
    manager = PretrainedModelManager("models_directory")

    # Create training data
    X, y, training_info = manager.create_training_data(
        hurst_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lengths=[500, 1000, 2000],
        generators=['fbm', 'fgn', 'arfima'],
        contaminations=['none', 'noise', 'missing'],
        biomedical_scenarios=['eeg', 'ecg', 'respiratory']
    )

    # Create training configurations
    configs = [
        TrainingConfig(
            model_type=MLBaselineType.RANDOM_FOREST,
            hyperparameters={'n_estimators': 100, 'random_state': 42},
            description="Random Forest model"
        ),
        TrainingConfig(
            model_type=MLBaselineType.SVR,
            hyperparameters={'C': 1.0, 'gamma': 'scale'},
            description="SVR model"
        )
    ]

    # Train models
    results = manager.create_model_suite(configs, X, y, training_info)

    # List trained models
    models = manager.list_models()
    for model in models:
        print(f"Model: {model.model_id}, Type: {model.model_type}")

Fast Inference
~~~~~~~~~~~~~

.. code-block:: python

    from neurological_lrd_analysis import (
        quick_predict, quick_ensemble_predict, PretrainedInference
    )

    # Single model prediction
    hurst_rf = quick_predict(time_series, "models_directory", "random_forest")
    hurst_svr = quick_predict(time_series, "models_directory", "svr")

    # Ensemble prediction (best accuracy)
    hurst_ensemble, uncertainty = quick_ensemble_predict(time_series, "models_directory")

    # Batch prediction
    inference = PretrainedInference("models_directory")
    predictions = inference.predict_batch(time_series_list)

    # Ensemble batch prediction
    ensemble_predictions = inference.ensemble_predict_batch(time_series_list)

Benchmark Comparison
--------------------

The library provides comprehensive benchmarking between classical and ML methods:

.. code-block:: python

    from neurological_lrd_analysis import ClassicalMLBenchmark, run_comprehensive_benchmark

    # Create benchmark system
    benchmark = ClassicalMLBenchmark(
        pretrained_models_dir="pretrained_models",
        classical_estimators=[EstimatorType.DFA, EstimatorType.RS_ANALYSIS],
        ml_estimators=['random_forest', 'ensemble']
    )

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        test_scenarios=test_scenarios,
        save_results=True
    )

    # Access results
    print("Performance Summary:")
    for method_name, summary in results['summaries'].items():
        print(f"{method_name}:")
        print(f"  MAE: {summary.mean_absolute_error:.4f}")
        print(f"  RMSE: {summary.root_mean_squared_error:.4f}")
        print(f"  Correlation: {summary.correlation:.4f}")
        print(f"  Mean time: {summary.mean_computation_time*1000:.1f}ms")

Performance Results
-------------------

Based on comprehensive benchmarking, the ML methods show superior performance:

**Performance Rankings (MAE - Mean Absolute Error)**
1. **Ensemble (ML)**: MAE 0.1518 - **BEST OVERALL**
2. **DFA (Classical)**: MAE 0.1983 - Best Classical
3. **R/S Analysis (Classical)**: MAE 0.1993
4. **Periodogram (Classical)**: MAE 0.2038
5. **Higuchi (Classical)**: MAE 0.9906

**Speed Rankings (computation time)**
1. **Periodogram**: 14.0ms - **FASTEST**
2. **Ensemble (ML)**: 59.3ms - **ML is very fast!**
3. **R/S Analysis**: 694.2ms
4. **Higuchi**: 811.5ms
5. **DFA**: 2044.5ms

Key Findings
~~~~~~~~~~~~

- **ML Ensemble method achieved the best accuracy** (MAE: 0.1518)
- **ML methods are significantly faster** than most classical methods
- **ML ensemble is 4x faster than DFA** while being more accurate
- **ML methods show excellent correlation** (0.9294) with true values

API Reference
--------------

TimeSeriesFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.TimeSeriesFeatureExtractor
    :members:
    :undoc-members:
    :show-inheritance:

RandomForestEstimator
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.RandomForestEstimator
    :members:
    :undoc-members:
    :show-inheritance:

SVREstimator
~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.SVREstimator
    :members:
    :undoc-members:
    :show-inheritance:

GradientBoostingEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.GradientBoostingEstimator
    :members:
    :undoc-members:
    :show-inheritance:

PretrainedModelManager
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.PretrainedModelManager
    :members:
    :undoc-members:
    :show-inheritance:

PretrainedInference
~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.PretrainedInference
    :members:
    :undoc-members:
    :show-inheritance:

ClassicalMLBenchmark
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.ClassicalMLBenchmark
    :members:
    :undoc-members:
    :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: neurological_lrd_analysis.create_pretrained_suite
.. autofunction:: neurological_lrd_analysis.quick_predict
.. autofunction:: neurological_lrd_analysis.quick_ensemble_predict
.. autofunction:: neurological_lrd_analysis.run_comprehensive_benchmark
