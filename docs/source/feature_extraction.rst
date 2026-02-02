Feature Extraction
==================

The ``TimeSeriesFeatureExtractor`` class provides comprehensive feature extraction from time series data, extracting 74+ features across multiple categories for machine learning applications.

Overview
--------

The feature extractor is designed to capture the essential characteristics of time series data that are relevant for Hurst exponent estimation. It provides features in several categories:

- **Statistical Features**: Basic statistical measures
- **Spectral Features**: Frequency domain characteristics
- **Wavelet Features**: Time-frequency analysis
- **Fractal Features**: Self-similarity measures
- **Biomedical Features**: Domain-specific characteristics

Basic Usage
-----------

.. code-block:: python

    from neurological_lrd_analysis import TimeSeriesFeatureExtractor
    import numpy as np

    # Create feature extractor
    extractor = TimeSeriesFeatureExtractor()

    # Generate sample data
    data = np.random.randn(1000)

    # Extract features
    features = extractor.extract_features(data)
    print(f"Extracted {len(features)} features")

    # Display feature names and values
    for name, value in features.items():
        print(f"{name}: {value:.4f}")

Feature Categories
------------------

Statistical Features
~~~~~~~~~~~~~~~~~~~~

Basic statistical measures of the time series:

- **Mean**: Average value
- **Variance**: Measure of spread
- **Skewness**: Measure of asymmetry
- **Kurtosis**: Measure of tail heaviness
- **Percentiles**: 25th, 50th, 75th, 90th, 95th percentiles
- **Range**: Difference between max and min
- **Interquartile Range**: Difference between 75th and 25th percentiles

Autocorrelation Features
~~~~~~~~~~~~~~~~~~~~~~~~

Autocorrelation at various lags:

- **Autocorrelation at lag 1**: First-order autocorrelation
- **Autocorrelation at lag 2**: Second-order autocorrelation
- **Autocorrelation at lag 5**: Fifth-order autocorrelation
- **Autocorrelation at lag 10**: Tenth-order autocorrelation

Spectral Features
~~~~~~~~~~~~~~~~~

Frequency domain characteristics:

- **Spectral Centroid**: Center of mass of the spectrum
- **Spectral Bandwidth**: Measure of spectral spread
- **Spectral Rolloff**: Frequency below which 85% of energy lies
- **Spectral Flatness**: Measure of noisiness
- **Zero Crossing Rate**: Rate of sign changes

Frequency Band Power
~~~~~~~~~~~~~~~~~~~~

Power in different frequency bands (biomedical relevance):

- **Delta Power**: 0.5-4 Hz (deep sleep, unconsciousness)
- **Theta Power**: 4-8 Hz (light sleep, meditation)
- **Alpha Power**: 8-13 Hz (relaxed wakefulness)
- **Beta Power**: 13-30 Hz (active concentration)
- **Gamma Power**: 30-100 Hz (high-level cognitive processing)

Power Ratios
~~~~~~~~~~~~

Relative power in different bands:

- **Delta Ratio**: Delta power / Total power
- **Theta Ratio**: Theta power / Total power
- **Alpha Ratio**: Alpha power / Total power
- **Beta Ratio**: Beta power / Total power
- **Gamma Ratio**: Gamma power / Total power

Wavelet Features
~~~~~~~~~~~~~~~~

Time-frequency analysis using wavelets:

- **Wavelet Energy**: Energy at different scales
- **Wavelet Entropy**: Measure of complexity
- **Wavelet Complexity**: Measure of irregularity
- **Multiresolution Analysis**: Features at multiple scales

Fractal Features
~~~~~~~~~~~~~~~~

Self-similarity and fractal characteristics:

- **Detrended Fluctuation Analysis (DFA)**: Long-range correlations
- **Higuchi Fractal Dimension**: Measure of complexity
- **Generalized Hurst Exponent**: Multifractal analysis
- **Sample Entropy**: Measure of regularity

Biomedical Features
~~~~~~~~~~~~~~~~~~~

Domain-specific features for biomedical signals:

- **EEG Features**: Electrode characteristics, brain activity patterns
- **ECG Features**: Heart rate variability, cardiac rhythms
- **Respiratory Features**: Breathing patterns, respiratory variability

Advanced Usage
--------------

Custom Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Extract specific feature categories
    extractor = TimeSeriesFeatureExtractor()

    # Extract only statistical features
    statistical_features = extractor.extract_statistical_features(data)

    # Extract only spectral features
    spectral_features = extractor.extract_spectral_features(data)

    # Extract only wavelet features
    wavelet_features = extractor.extract_wavelet_features(data)

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Extract features from multiple time series
    data_list = [np.random.randn(1000) for _ in range(10)]
    
    all_features = []
    for data in data_list:
        features = extractor.extract_features(data)
        all_features.append(list(features.values()))
    
    # Convert to numpy array for ML training
    X = np.array(all_features)
    print(f"Feature matrix shape: {X.shape}")

Feature Selection
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get feature names for selection
    feature_names = list(extractor.extract_features(data).keys())
    print(f"Available features: {len(feature_names)}")

    # Select specific features
    selected_features = ['mean', 'variance', 'skewness', 'kurtosis']
    selected_indices = [feature_names.index(f) for f in selected_features]
    
    # Extract only selected features
    X_selected = X[:, selected_indices]
    print(f"Selected features shape: {X_selected.shape}")

Performance Considerations
--------------------------

Memory Usage
~~~~~~~~~~~~

The feature extractor is designed to be memory-efficient:

- Features are computed on-demand
- No unnecessary data is stored
- Efficient algorithms for large datasets

Computation Time
~~~~~~~~~~~~~~~~

Feature extraction time scales with data length:

- **Short series** (< 1000 points): ~1-10ms
- **Medium series** (1000-5000 points): ~10-100ms
- **Long series** (> 5000 points): ~100ms-1s

Optimization Tips
~~~~~~~~~~~~~~~~~

For best performance:

1. **Use appropriate data length**: 1000-2000 points is optimal
2. **Batch processing**: Extract features for multiple series at once
3. **Feature selection**: Use only relevant features for your application
4. **Memory management**: Process data in chunks for very large datasets

API Reference
--------------

TimeSeriesFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.TimeSeriesFeatureExtractor
    :members:
    :undoc-members:
    :show-inheritance:

Methods
~~~~~~~

.. automethod:: neurological_lrd_analysis.TimeSeriesFeatureExtractor.extract_features
.. autofunction:: neurological_lrd_analysis.ml_baselines.extract_statistical_features
.. autofunction:: neurological_lrd_analysis.ml_baselines.extract_spectral_features
.. autofunction:: neurological_lrd_analysis.ml_baselines.extract_wavelet_features
.. autofunction:: neurological_lrd_analysis.ml_baselines.extract_fractal_features
.. autofunction:: neurological_lrd_analysis.ml_baselines.extract_biomedical_features

Examples
--------

Complete Feature Extraction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from neurological_lrd_analysis import TimeSeriesFeatureExtractor, fbm_davies_harte
    import numpy as np

    # Generate sample data
    data = fbm_davies_harte(1000, 0.7, seed=42)

    # Create feature extractor
    extractor = TimeSeriesFeatureExtractor()

    # Extract all features
    features = extractor.extract_features(data)

    # Display feature summary
    print(f"Extracted {len(features)} features")
    print(f"Feature categories:")
    print(f"  Statistical: {len([f for f in features.keys() if 'statistical' in f])}")
    print(f"  Spectral: {len([f for f in features.keys() if 'spectral' in f])}")
    print(f"  Wavelet: {len([f for f in features.keys() if 'wavelet' in f])}")
    print(f"  Fractal: {len([f for f in features.keys() if 'fractal' in f])}")

    # Show top features by value
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\nTop 10 features by absolute value:")
    for name, value in sorted_features[:10]:
        print(f"  {name}: {value:.4f}")

Feature Analysis for ML
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Analyze feature importance for ML
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # Generate training data
    X_train = []
    y_train = []

    for hurst in [0.3, 0.5, 0.7, 0.9]:
        for _ in range(20):
            data = fbm_davies_harte(1000, hurst, seed=np.random.randint(0, 10000))
            features = extractor.extract_features(data)
            X_train.append(list(features.values()))
            y_train.append(hurst)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Get feature importance
    feature_names = list(features.keys())
    importance = rf.feature_importances_

    # Show most important features
    importance_pairs = list(zip(feature_names, importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    print("Most important features for Hurst estimation:")
    for name, imp in importance_pairs[:10]:
        print(f"  {name}: {imp:.4f}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Memory Errors**
- Reduce data length or process in chunks
- Use feature selection to reduce dimensionality

**Computation Time**
- Use shorter time series for real-time applications
- Consider using only essential features

**Feature Quality**
- Ensure data is properly preprocessed
- Check for NaN or infinite values
- Use appropriate data length (1000-2000 points recommended)

Best Practices
~~~~~~~~~~~~~~

1. **Data Preprocessing**: Clean and normalize data before feature extraction
2. **Feature Selection**: Use domain knowledge to select relevant features
3. **Validation**: Always validate features on test data
4. **Documentation**: Keep track of which features are used in your models
