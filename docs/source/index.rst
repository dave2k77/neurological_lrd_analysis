Welcome to Neurological LRD Analysis's documentation!
=====================================================

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.11+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/PyPI-neurological--lrd--analysis-green.svg
   :target: https://pypi.org/project/neurological-lrd-analysis/
   :alt: PyPI

A comprehensive library for estimating Hurst exponents in neurological time series data, featuring **classical methods**, **machine learning baselines**, and **comprehensive benchmarking capabilities**.

**Research Context**: This library is developed as part of PhD research in Biomedical Engineering at the University of Reading, UK by Davian R. Chin, focusing on **Physics-Informed Fractional Operator Learning for Real-Time Neurological Biomarker Detection: A Framework for Memory-Driven EEG Analysis**.

Features
--------

* **Classical Methods**: DFA, R/S Analysis, Higuchi, Generalized Hurst Exponent, Periodogram, GPH, Whittle MLE, DWT, Abry-Veitch, MFDFA
* **Machine Learning Baselines**: Random Forest, SVR, Gradient Boosting, Ensemble methods with 74+ feature extraction
* **Comprehensive Benchmarking**: Direct comparison between classical and ML methods with real performance data
* **Fast Inference**: 10-50ms prediction times with pretrained models
* **Neurological Scenarios**: EEG, ECG, respiratory signals with realistic artifacts
* **GPU Acceleration**: JAX and NumPyro integration for Bayesian inference
* **Clinical Relevance**: Specialized for Parkinson's disease, epilepsy, and neurological conditions

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install neurological-lrd-analysis

Basic usage:

**Classical Methods:**

.. code-block:: python

   from neurological_lrd_analysis import BiomedicalHurstEstimatorFactory, EstimatorType

   # Create factory instance
   factory = BiomedicalHurstEstimatorFactory()

   # Estimate Hurst exponent using DFA
   result = factory.estimate(
       data=your_time_series,
       method=EstimatorType.DFA,
       confidence_method="bootstrap",
       n_bootstrap=100
   )

   print(f"Hurst exponent: {result.hurst_estimate:.3f}")

**Machine Learning Methods:**

.. code-block:: python

   from neurological_lrd_analysis import (
       create_pretrained_suite, quick_predict, quick_ensemble_predict
   )

   # Create pretrained models (one-time setup)
   create_pretrained_suite("pretrained_models", force_retrain=True)

   # Fast ML prediction (10-50ms)
   hurst_ml = quick_predict(your_time_series, "pretrained_models", "random_forest")

   # Ensemble prediction (best accuracy)
   hurst_ensemble, uncertainty = quick_ensemble_predict(your_time_series, "pretrained_models")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   ENVIRONMENT_SETUP
   TUTORIAL
   ml_tutorial

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   api_reference
   BENCHMARKING_GUIDE
   CONFIGURATION_GUIDE

.. toctree::
   :maxdepth: 2
   :caption: Machine Learning

   ml_baselines
   feature_extraction
   NUMPYRO_INTEGRATION

.. toctree::
   :maxdepth: 2
   :caption: Research

   comprehensive-lrd-estimators-paper
   Techniques for Estimating the Hurst Exponent
   wavelet-based-lrd-estimators

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
