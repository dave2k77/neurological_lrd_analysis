API Reference
=============

This section provides comprehensive documentation for all classes, methods, and functions in the Neurological LRD Analysis library.

Core Classes
------------

BiomedicalHurstEstimatorFactory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.BiomedicalHurstEstimatorFactory
   :members:
   :undoc-members:
   :show-inheritance:

HurstResult
~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.HurstResult
   :members:
   :undoc-members:
   :show-inheritance:

BiomedicalDataProcessor
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.BiomedicalDataProcessor
   :members:
   :undoc-members:
   :show-inheritance:

Enumerations
------------

EstimatorType
~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.EstimatorType
   :members:
   :undoc-members:

ConfidenceMethod
~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.ConfidenceMethod
   :members:
   :undoc-members:

Machine Learning Baselines
----------------------------

TimeSeriesFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.ml_baselines.feature_extraction.TimeSeriesFeatureExtractor
   :members:
   :undoc-members:
   :show-inheritance:

RandomForestEstimator
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.ml_baselines.ml_estimators.RandomForestEstimator
   :members:
   :undoc-members:
   :show-inheritance:

SVREstimator
~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.ml_baselines.ml_estimators.SVREstimator
   :members:
   :undoc-members:
   :show-inheritance:

GradientBoostingEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.ml_baselines.ml_estimators.GradientBoostingEstimator
   :members:
   :undoc-members:
   :show-inheritance:

OptunaOptimizer
~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.ml_baselines.hyperparameter_optimization.OptunaOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

PretrainedModelManager
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.ml_baselines.pretrained_models.PretrainedModelManager
   :members:
   :undoc-members:
   :show-inheritance:

PretrainedInference
~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.ml_baselines.inference.PretrainedInference
   :members:
   :undoc-members:
   :show-inheritance:

ClassicalMLBenchmark
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.ml_baselines.benchmark_comparison.ClassicalMLBenchmark
   :members:
   :undoc-members:
   :show-inheritance:

Data Generation
---------------

Synthetic Data Generators
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: neurological_lrd_analysis.benchmark_core.generation
   :members:
   :undoc-members:

Biomedical Scenarios
~~~~~~~~~~~~~~~~~~~~

.. automodule:: neurological_lrd_analysis.benchmark_core.biomedical_scenarios
   :members:
   :undoc-members:

Benchmarking
------------

Configuration and Results
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: neurological_lrd_analysis.benchmark_core.runner
   :members:
   :undoc-members:

Registry System
---------------

Estimator Registry
~~~~~~~~~~~~~~~~~~

.. automodule:: neurological_lrd_analysis.benchmark_registry.registry
   :members:
   :undoc-members:

Backend Selection
-----------------

Backend Selector
~~~~~~~~~~~~~~~~

.. automodule:: neurological_lrd_analysis.benchmark_backends.selector
   :members:
   :undoc-members:

Individual Estimators
---------------------

Temporal Estimators
~~~~~~~~~~~~~~~~~~~

DFA Estimator
^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.DFAEstimator
   :members:
   :undoc-members:
   :show-inheritance:

R/S Analysis Estimator
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.RSAnalysisEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Higuchi Estimator
^^^^^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.HiguchiEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Generalized Hurst Exponent Estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.GHEEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Spectral Estimators
~~~~~~~~~~~~~~~~~~~

Periodogram Estimator
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.PeriodogramEstimator
   :members:
   :undoc-members:
   :show-inheritance:

GPH Estimator
^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.GPHEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Whittle MLE Estimator
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.WhittleMLEEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Wavelet Estimators
~~~~~~~~~~~~~~~~~~

DWT Estimator
^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.DWTEstimator
   :members:
   :undoc-members:
   :show-inheritance:

NDWT Estimator
^^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.NDWTEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Abry-Veitch Estimator
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.AbryVeitchEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Multifractal Estimators
~~~~~~~~~~~~~~~~~~~~~~~

MFDFA Estimator
^^^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.MFDFAEstimator
   :members:
   :undoc-members:
   :show-inheritance:

MF-DMA Estimator
^^^^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.MFDMAEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Bayesian Estimators
~~~~~~~~~~~~~~~~~~~

Bayesian Hurst Estimator
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: neurological_lrd_analysis.biomedical_hurst_factory.BayesianHurstEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Utility Machine Learning Functions
---------------------------

Feature Extraction Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: neurological_lrd_analysis.ml_baselines.feature_extraction.extract_statistical_features
.. autofunction:: neurological_lrd_analysis.ml_baselines.feature_extraction.extract_spectral_features
.. autofunction:: neurological_lrd_analysis.ml_baselines.feature_extraction.extract_wavelet_features
.. autofunction:: neurological_lrd_analysis.ml_baselines.feature_extraction.extract_fractal_features
.. autofunction:: neurological_lrd_analysis.ml_baselines.feature_extraction.extract_biomedical_features

Hyperparameter Optimization Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: neurological_lrd_analysis.ml_baselines.hyperparameter_optimization.create_optuna_study
.. autofunction:: neurological_lrd_analysis.ml_baselines.hyperparameter_optimization.optimize_hyperparameters
.. autofunction:: neurological_lrd_analysis.ml_baselines.hyperparameter_optimization.optimize_all_estimators

Pretrained Model Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: neurological_lrd_analysis.ml_baselines.pretrained_models.create_default_training_configs
.. autofunction:: neurological_lrd_analysis.ml_baselines.pretrained_models.create_pretrained_suite

Inference Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: neurological_lrd_analysis.ml_baselines.inference.quick_predict
.. autofunction:: neurological_lrd_analysis.ml_baselines.inference.quick_ensemble_predict

Benchmark Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: neurological_lrd_analysis.ml_baselines.benchmark_comparison.run_comprehensive_benchmark

Functions
-----------------

Helper Functions
~~~~~~~~~~~~~~~~

.. automodule:: neurological_lrd_analysis.biomedical_hurst_factory
   :members:
   :undoc-members:
   :exclude-members: BiomedicalHurstEstimatorFactory, HurstResult, BiomedicalDataProcessor

Data Structures
---------------

TimeSeriesSample
~~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.benchmark_core.generation.TimeSeriesSample
   :members:
   :undoc-members:
   :show-inheritance:

BenchmarkConfig
~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.benchmark_core.runner.BenchmarkConfig
   :members:
   :undoc-members:
   :show-inheritance:

BenchmarkResult
~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.benchmark_core.runner.BenchmarkResult
   :members:
   :undoc-members:
   :show-inheritance:

ScoringWeights
~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.benchmark_core.runner.ScoringWeights
   :members:
   :undoc-members:
   :show-inheritance:

BaseEstimator
~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.benchmark_registry.registry.BaseEstimator
   :members:
   :undoc-members:
   :show-inheritance:

EstimatorResult
~~~~~~~~~~~~~~~

.. autoclass:: neurological_lrd_analysis.benchmark_registry.registry.EstimatorResult
   :members:
   :undoc-members:
   :show-inheritance:

Constants
---------

BIOMEDICAL_SCENARIOS
~~~~~~~~~~~~~~~~~~~~

.. data:: neurological_lrd_analysis.benchmark_core.biomedical_scenarios.BIOMEDICAL_SCENARIOS

   Dictionary containing predefined biomedical scenarios with their configurations.

   Available scenarios:
   
   * ``eeg_rest``: Resting state EEG
   * ``eeg_eyes_closed``: Eyes closed EEG
   * ``eeg_sleep``: Sleep state EEG
   * ``eeg_parkinsonian``: Parkinson's disease EEG
   * ``eeg_epileptic``: Epileptic EEG
   * ``ecg_normal``: Normal heart rate ECG
   * ``ecg_tachycardia``: Tachycardia ECG
   * ``respiratory_rest``: Resting respiratory signal

   Each scenario includes:
   
   * ``scenario_type``: Type of biomedical signal
   * ``hurst_range``: Expected Hurst exponent range
   * ``typical_amplitude``: Typical signal amplitude
   * ``noise_level``: Typical noise level
   * ``auto_contamination``: Automatic contamination type (if applicable)
