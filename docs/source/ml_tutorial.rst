Machine Learning Tutorial
=========================

This tutorial provides a comprehensive guide to using the machine learning baselines in the neurological LRD analysis library.

Prerequisites
-------------

Before starting this tutorial, ensure you have the library installed:

.. code-block:: bash

    pip install neurological-lrd-analysis

You'll also need some additional dependencies for ML functionality:

.. code-block:: bash

    pip install optuna joblib scikit-learn

Tutorial Overview
-----------------

This tutorial covers:

1. **Feature Extraction**: Extracting 74+ features from time series data
2. **ML Model Training**: Training Random Forest, SVR, and Gradient Boosting models
3. **Hyperparameter Optimization**: Using Optuna for automated tuning
4. **Pretrained Models**: Creating and using pre-trained models
5. **Fast Inference**: Real-time prediction capabilities
6. **Benchmarking**: Comparing classical and ML methods

Step 1: Feature Extraction
--------------------------

The first step in using ML methods is to extract features from your time series data.

.. code-block:: python

    import numpy as np
    from neurological_lrd_analysis import TimeSeriesFeatureExtractor, fbm_davies_harte

    # Generate sample time series data
    data = fbm_davies_harte(1000, 0.7, seed=42)

    # Create feature extractor
    extractor = TimeSeriesFeatureExtractor()

    # Extract features
    features = extractor.extract_features(data)
    print(f"Extracted {len(features)} features")

    # Display feature names and values
    for name, value in features.items():
        print(f"{name}: {value:.4f}")

Feature Categories
~~~~~~~~~~~~~~~~~~

The feature extractor provides features in several categories:

**Statistical Features**
- Basic statistics: mean, variance, skewness, kurtosis
- Distribution features: percentiles, quartiles, range
- Autocorrelation features: at various lags

**Spectral Features**
- Power spectral density features
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
- EEG-specific features
- ECG-specific features
- Respiratory features

Step 2: Training ML Models
--------------------------

Now let's train ML models using the extracted features.

.. code-block:: python

    from neurological_lrd_analysis import (
        RandomForestEstimator, SVREstimator, GradientBoostingEstimator
    )

    # Generate training data
    X_train = []
    y_train = []

    for hurst in [0.3, 0.5, 0.7, 0.9]:
        for _ in range(10):  # 10 samples per Hurst value
            data = fbm_davies_harte(1000, hurst, seed=np.random.randint(0, 10000))
            features = extractor.extract_features(data)
            X_train.append(list(features.values()))
            y_train.append(hurst)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train Random Forest
    rf_estimator = RandomForestEstimator()
    rf_result = rf_estimator.train(X_train, y_train, validation_split=0.2)
    print(f"Random Forest - Training score: {rf_result.training_score:.4f}")
    print(f"Random Forest - Validation score: {rf_result.validation_score:.4f}")

    # Train SVR
    svr_estimator = SVREstimator()
    svr_result = svr_estimator.train(X_train, y_train, validation_split=0.2)
    print(f"SVR - Training score: {svr_result.training_score:.4f}")
    print(f"SVR - Validation score: {svr_result.validation_score:.4f}")

    # Train Gradient Boosting
    gb_estimator = GradientBoostingEstimator()
    gb_result = gb_estimator.train(X_train, y_train, validation_split=0.2)
    print(f"Gradient Boosting - Training score: {gb_result.training_score:.4f}")
    print(f"Gradient Boosting - Validation score: {gb_result.validation_score:.4f}")

Step 3: Hyperparameter Optimization
------------------------------------

Use Optuna to automatically find the best hyperparameters for your models.

.. code-block:: python

    from neurological_lrd_analysis import (
        OptunaOptimizer, create_optuna_study, optimize_hyperparameters
    )

    # Optimize Random Forest hyperparameters
    print("Optimizing Random Forest hyperparameters...")
    rf_study = create_optuna_study(
        model_type="random_forest",
        X_train=X_train,
        y_train=y_train,
        n_trials=50
    )

    print(f"Best Random Forest parameters: {rf_study.best_params}")
    print(f"Best Random Forest score: {rf_study.best_value:.4f}")

    # Optimize SVR hyperparameters
    print("Optimizing SVR hyperparameters...")
    svr_study = create_optuna_study(
        model_type="svr",
        X_train=X_train,
        y_train=y_train,
        n_trials=50
    )

    print(f"Best SVR parameters: {svr_study.best_params}")
    print(f"Best SVR score: {svr_study.best_value:.4f}")

Step 4: Creating Pretrained Models
-----------------------------------

Create a comprehensive suite of pretrained models for fast inference.

.. code-block:: python

    from neurological_lrd_analysis import (
        create_pretrained_suite, PretrainedModelManager, TrainingConfig, MLBaselineType
    )

    # Create pretrained model suite
    print("Creating pretrained model suite...")
    manager = create_pretrained_suite(
        models_dir="pretrained_models",
        force_retrain=True
    )

    # List created models
    models = manager.list_models()
    print(f"Created {len(models)} pretrained models:")
    for model in models:
        print(f"  - {model.model_id}: {model.model_type.value}")
        print(f"    Validation score: {model.performance_metrics.get('validation_score', 'N/A'):.4f}")

Step 5: Fast Inference
----------------------

Use the pretrained models for fast inference on new data.

.. code-block:: python

    from neurological_lrd_analysis import (
        quick_predict, quick_ensemble_predict, PretrainedInference
    )

    # Generate test data
    test_data = fbm_davies_harte(1000, 0.6, seed=123)

    # Single model predictions
    print("Single model predictions:")
    hurst_rf = quick_predict(test_data, "pretrained_models", "random_forest")
    print(f"Random Forest prediction: {hurst_rf:.4f}")

    hurst_svr = quick_predict(test_data, "pretrained_models", "svr")
    print(f"SVR prediction: {hurst_svr:.4f}")

    # Ensemble prediction (best accuracy)
    print("Ensemble prediction:")
    hurst_ensemble, uncertainty = quick_ensemble_predict(test_data, "pretrained_models")
    print(f"Ensemble prediction: {hurst_ensemble:.4f} ± {uncertainty:.4f}")

    # Batch prediction
    print("Batch prediction:")
    inference = PretrainedInference("pretrained_models")
    test_data_list = [fbm_davies_harte(1000, h, seed=123+i) for h in [0.4, 0.6, 0.8] for i in range(3)]
    predictions = inference.predict_batch(test_data_list)
    print(f"Batch predictions: {predictions}")

Step 6: Comprehensive Benchmarking
---------------------------------

Compare classical and ML methods using the comprehensive benchmark system.

.. code-block:: python

    from neurological_lrd_analysis import (
        ClassicalMLBenchmark, run_comprehensive_benchmark,
        BiomedicalHurstEstimatorFactory, EstimatorType
    )

    # Create test scenarios
    test_scenarios = []
    for hurst in [0.3, 0.5, 0.7, 0.9]:
        for length in [500, 1000, 2000]:
            data = fbm_davies_harte(length, hurst, seed=42)
            test_scenarios.append({
                'data': data,
                'true_hurst': hurst,
                'length': length,
                'scenario': f'fBm_H{hurst}_L{length}'
            })

    # Create benchmark system
    benchmark = ClassicalMLBenchmark(
        pretrained_models_dir="pretrained_models",
        classical_estimators=[EstimatorType.DFA, EstimatorType.RS_ANALYSIS, EstimatorType.HIGUCHI],
        ml_estimators=['random_forest', 'svr', 'ensemble']
    )

    # Run comprehensive benchmark
    print("Running comprehensive benchmark...")
    results = benchmark.run_comprehensive_benchmark(
        test_scenarios=test_scenarios,
        save_results=True
    )

    # Display results
    print("\nBenchmark Results:")
    print("=" * 60)
    print(f"{'Method':<15} {'Type':<10} {'MAE':<8} {'RMSE':<8} {'Corr':<8} {'Time(ms)':<10}")
    print("-" * 60)

    for method_name, summary in results['summaries'].items():
        print(f"{method_name:<15} {summary.method_type:<10} "
              f"{summary.mean_absolute_error:<8.4f} {summary.root_mean_squared_error:<8.4f} "
              f"{summary.correlation:<8.4f} {summary.mean_computation_time*1000:<10.1f}")

Step 7: Advanced Usage
-----------------------

Explore advanced features of the ML baselines system.

Feature Importance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get feature importance from trained models
    rf_importance = rf_estimator.get_feature_importance()
    print("Random Forest Feature Importance (top 10):")
    for i, importance in enumerate(rf_importance[:10]):
        print(f"  Feature {i}: {importance:.4f}")

Cross-Validation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Perform cross-validation analysis
    cv_scores = rf_estimator.get_cv_scores()
    print(f"Random Forest CV scores: {cv_scores}")
    print(f"Mean CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

Model Persistence
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Save trained models
    rf_estimator.save_model("rf_model.pkl")
    svr_estimator.save_model("svr_model.pkl")

    # Load saved models
    from neurological_lrd_analysis import RandomForestEstimator, SVREstimator

    loaded_rf = RandomForestEstimator()
    loaded_rf.load_model("rf_model.pkl")

    loaded_svr = SVREstimator()
    loaded_svr.load_model("svr_model.pkl")

    # Use loaded models for prediction
    test_features = extractor.extract_features(test_data)
    test_features_array = np.array([list(test_features.values())])
    
    rf_pred = loaded_rf.predict(test_features_array)
    svr_pred = loaded_svr.predict(test_features_array)
    
    print(f"Loaded RF prediction: {rf_pred[0]:.4f}")
    print(f"Loaded SVR prediction: {svr_pred[0]:.4f}")

Performance Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create performance comparison plot
    methods = list(results['summaries'].keys())
    mae_values = [results['summaries'][m].mean_absolute_error for m in methods]
    time_values = [results['summaries'][m].mean_computation_time * 1000 for m in methods]

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.barh(methods, mae_values)
    plt.xlabel('Mean Absolute Error')
    plt.title('Performance Comparison (MAE)')
    
    plt.subplot(1, 2, 2)
    plt.barh(methods, time_values)
    plt.xlabel('Computation Time (ms)')
    plt.title('Speed Comparison')
    
    plt.tight_layout()
    plt.savefig('ml_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()

Best Practices
--------------

1. **Feature Engineering**: Always use the comprehensive feature extractor for best results
2. **Hyperparameter Optimization**: Use Optuna for automated tuning
3. **Model Selection**: Ensemble methods typically provide the best accuracy
4. **Validation**: Always use proper train/validation splits
5. **Persistence**: Save trained models for reuse
6. **Benchmarking**: Compare ML methods with classical methods

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Import Errors**
- Ensure all dependencies are installed: `pip install optuna joblib scikit-learn`
- Check that the library is properly installed: `pip install neurological-lrd-analysis`

**Memory Issues**
- Reduce the number of features or samples
- Use smaller hyperparameter search spaces
- Consider using fewer models in the ensemble

**Performance Issues**
- Use pretrained models for fast inference
- Consider using fewer features for real-time applications
- Optimize hyperparameters for your specific use case

**Model Training Issues**
- Ensure sufficient training data
- Check for data quality issues
- Use proper validation splits

Next Steps
----------

- Explore the **API Reference** for detailed documentation
- Check out the **Benchmarking Guide** for performance analysis
- Try the **Jupyter Notebooks** for interactive examples
- Contribute to the project on **GitHub**

For more information, see the complete documentation at https://neurological-lrd-analysis.readthedocs.io/.
