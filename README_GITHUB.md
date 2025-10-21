# Neurological LRD Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/badge/PyPI-neurological--lrd--analysis-green.svg)](https://pypi.org/project/neurological-lrd-analysis/)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://neurological-lrd-analysis.readthedocs.io/)
[![GitHub Actions](https://github.com/dave2k77/neurological_lrd_analysis/workflows/CI/badge.svg)](https://github.com/dave2k77/neurological_lrd_analysis/actions)
[![Codecov](https://codecov.io/gh/dave2k77/neurological_lrd_analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/dave2k77/neurological_lrd_analysis)

A comprehensive library for estimating Hurst exponents in neurological time series data, featuring **classical methods**, **machine learning baselines**, and **comprehensive benchmarking capabilities**.

**Developed as part of PhD research in Biomedical Engineering at the University of Reading, UK** by Davian R. Chin, focusing on **Physics-Informed Fractional Operator Learning for Real-Time Neurological Biomarker Detection: A Framework for Memory-Driven EEG Analysis**.

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install neurological-lrd-analysis

# Install with GPU support
pip install neurological-lrd-analysis[gpu]

# Install with development dependencies
pip install neurological-lrd-analysis[dev]

# Or install from source
git clone https://github.com/dave2k77/neurological_lrd_analysis.git
cd neurological_lrd_analysis
pip install -e .
```

### Basic Usage

#### Classical Methods
```python
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
print(f"Confidence interval: {result.confidence_interval}")
```

#### Machine Learning Methods
```python
from neurological_lrd_analysis import (
    create_pretrained_suite, quick_predict, quick_ensemble_predict
)

# Create pretrained models (one-time setup)
create_pretrained_suite("pretrained_models", force_retrain=True)

# Fast ML prediction (10-50ms)
hurst_ml = quick_predict(your_time_series, "pretrained_models", "random_forest")

# Ensemble prediction (best accuracy)
hurst_ensemble, uncertainty = quick_ensemble_predict(your_time_series, "pretrained_models")
```

#### Comprehensive Benchmarking
```python
from neurological_lrd_analysis import ClassicalMLBenchmark

# Run comprehensive benchmark
benchmark = ClassicalMLBenchmark("pretrained_models")
results = benchmark.run_comprehensive_benchmark()

# Access results
print("Performance Summary:")
for method, metrics in results['summaries'].items():
    print(f"{method}: MAE={metrics.mean_absolute_error:.4f}, "
          f"Time={metrics.mean_computation_time*1000:.1f}ms")
```

## ✨ Key Features

### 🧮 **Multiple Estimation Methods**
- **Classical Methods**: DFA, R/S Analysis, Higuchi, Generalized Hurst Exponent, Periodogram, GPH, Whittle MLE, DWT, Abry-Veitch, MFDFA
- **ML Methods**: Random Forest, Support Vector Regression, Gradient Boosting, Ensemble approaches
- **Hybrid Approaches**: Classical + ML ensemble methods for optimal performance

### 🤖 **Machine Learning Baselines**
- **Feature Extraction**: 74+ comprehensive features (statistical, spectral, wavelet, fractal, biomedical)
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Pretrained Models**: Fast inference with pre-trained ML models
- **Ensemble Methods**: Weighted combination of multiple ML models
- **Real-time Inference**: 10-50ms prediction times

### 🏆 **Comprehensive Benchmarking**
- **Classical vs ML Comparison**: Direct performance comparison between methods
- **Real Benchmark Results**: Actual performance data with CSV exports
- **Visualization System**: Publication-ready plots and analysis
- **Performance Metrics**: MAE, RMSE, correlation, computation time, success rate

### 🏥 **Biomedical Scenarios**
- **EEG**: Rest, eyes closed/open, sleep, seizure patterns
- **ECG**: Normal heart rate, tachycardia, realistic QRS complexes
- **Respiratory**: Breathing patterns, irregular breathing
- **Neurological**: Memory-driven EEG analysis, neurological biomarker detection
- **Artifacts**: Electrode pops, motion, baseline drift, powerline interference

## 📊 Benchmark Results

### Performance Rankings (MAE - Mean Absolute Error)
1. **🥇 Ensemble (ML)**: MAE 0.1518 - **BEST OVERALL**
2. **🥈 DFA (Classical)**: MAE 0.1983 - Best Classical
3. **🥉 R/S Analysis (Classical)**: MAE 0.1993
4. **4th Periodogram (Classical)**: MAE 0.2038
5. **5th Higuchi (Classical)**: MAE 0.9906

### Speed Rankings (computation time)
1. **🥇 Periodogram**: 14.0ms - **FASTEST**
2. **🥈 Ensemble (ML)**: 59.3ms - **ML is very fast!**
3. **🥉 R/S Analysis**: 694.2ms
4. **4th Higuchi**: 811.5ms
5. **5th DFA**: 2044.5ms

### Key Findings
- **ML Ensemble method achieved the best accuracy** (MAE: 0.1518)
- **ML methods are significantly faster** than most classical methods
- **ML ensemble is 4x faster than DFA** while being more accurate
- **ML methods show excellent correlation** (0.9294) with true values

## 🎯 Use Cases

### Real-time Applications
```python
# Fast ML prediction for real-time systems
hurst_ml = quick_predict(eeg_data, "pretrained_models", "random_forest")
# 10-50ms prediction time
```

### Research Applications
```python
# Comprehensive analysis with uncertainty quantification
hurst_ensemble, uncertainty = quick_ensemble_predict(eeg_data, "pretrained_models")
print(f"Hurst: {hurst_ensemble:.3f} ± {uncertainty:.3f}")
```

### Benchmarking
```python
# Compare all methods
benchmark = ClassicalMLBenchmark("pretrained_models")
results = benchmark.run_comprehensive_benchmark()
# Results saved to CSV with visualizations
```

## 📖 Documentation

- **[Complete Documentation](https://neurological-lrd-analysis.readthedocs.io/)** - Comprehensive guides and API reference
- **[ML Baselines Tutorial](https://neurological-lrd-analysis.readthedocs.io/en/latest/ml_tutorial.html)** - Step-by-step ML tutorial
- **[Feature Extraction Guide](https://neurological-lrd-analysis.readthedocs.io/en/latest/feature_extraction.html)** - 74+ feature extraction
- **[Benchmarking Guide](https://neurological-lrd-analysis.readthedocs.io/en/latest/benchmarking.html)** - Performance comparison

## 🧪 Examples and Demos

### Classical Methods
```bash
# Run biomedical scenarios demonstration
python scripts/biomedical_scenarios_demo.py

# Run neurological conditions demonstration
python scripts/neurological_conditions_demo.py

# Run NumPyro integration demonstration
python scripts/numpyro_integration_demo.py
```

### Machine Learning Methods
```bash
# Run ML baselines demonstration
python scripts/ml_baselines_demo.py

# Run pretrained models demonstration
python scripts/pretrained_models_demo.py

# Run comprehensive benchmark demonstration
python scripts/comprehensive_benchmark_demo.py
```

### Benchmark Results
```bash
# View actual benchmark results
ls results/comprehensive_benchmark/
# - benchmark_results.csv (detailed results)
# - performance_metrics.csv (summary metrics)
# - comprehensive_benchmark_visualization.png
# - detailed_performance_analysis.png
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run ML baselines tests
python -m pytest tests/test_ml_baselines.py -v

# Run pretrained models tests
python -m pytest tests/test_pretrained_models.py -v

# Run benchmark comparison tests
python -m pytest tests/test_benchmark_comparison.py -v
```

## 📊 Project Structure

```
neurological_lrd_analysis/
├── neurological_lrd_analysis/     # Main package
│   ├── biomedical_hurst_factory.py    # Classical methods
│   ├── benchmark_core/                # Core benchmarking
│   ├── benchmark_backends/            # Backend selection
│   ├── benchmark_registry/            # Estimator registry
│   └── ml_baselines/                  # Machine learning baselines
│       ├── feature_extraction.py       # 74+ feature extraction
│       ├── ml_estimators.py          # ML model implementations
│       ├── hyperparameter_optimization.py  # Optuna integration
│       ├── pretrained_models.py       # Model management system
│       ├── inference.py              # Fast inference system
│       └── benchmark_comparison.py   # Classical vs ML comparison
├── docs/                         # Documentation
├── notebooks/                    # Interactive Jupyter tutorials
│   ├── 01_Biomedical_Time_Series_Analysis.ipynb
│   ├── 02_Hurst_Estimator_Creation_and_Validation.ipynb
│   ├── 03_Benchmarking_and_Leaderboards.ipynb
│   └── README.md
├── scripts/                      # Demo scripts and utilities
│   ├── ml_baselines_demo.py          # ML baselines demonstration
│   ├── pretrained_models_demo.py     # Pretrained models demo
│   └── comprehensive_benchmark_demo.py # Full benchmark demo
├── tests/                        # Test suite
│   ├── test_ml_baselines.py          # ML baselines tests
│   ├── test_pretrained_models.py     # Pretrained models tests
│   └── test_benchmark_comparison.py  # Benchmark comparison tests
├── examples/                     # Example notebooks and code
├── results/                      # Benchmark results and outputs
│   └── comprehensive_benchmark/       # Actual benchmark results
│       ├── benchmark_results.csv      # Detailed results
│       ├── performance_metrics.csv    # Summary metrics
│       └── *.png                     # Visualizations
├── pyproject.toml               # Package configuration
├── setup.py                     # Setup script
└── requirements.txt             # Dependencies
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/dave2k77/neurological_lrd_analysis.git
cd neurological_lrd_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest tests/ -v

# Run linting
black neurological_lrd_analysis/
isort neurological_lrd_analysis/
flake8 neurological_lrd_analysis/

# Run type checking
mypy neurological_lrd_analysis/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [JAX](https://github.com/google/jax) for GPU acceleration
- Uses [NumPyro](https://github.com/pyro-ppl/numpyro) for Bayesian inference
- Leverages [PyWavelets](https://github.com/PyWavelets/pywavelets) for wavelet analysis
- Inspired by research in physics-informed machine learning

## 📚 Citation

If you use this library in your research, please cite:

```bibtex
@software{neurological_lrd_analysis,
  title={Neurological LRD Analysis: A Comprehensive Library for Physics-Informed Fractional Operator Learning and Real-Time Neurological Biomarker Detection},
  author={Davian R. Chin},
  year={2025},
  institution={University of Reading, UK},
  email={d.r.chin@pgr.reading.ac.uk},
  orcid={https://orcid.org/0009-0003-9434-3919},
  note={PhD Research in Biomedical Engineering: A Framework for Memory-Driven EEG Analysis},
  url={https://github.com/dave2k77/neurological_lrd_analysis}
}
```

## 🔗 Links

- [Documentation](https://neurological-lrd-analysis.readthedocs.io/)
- [Issue Tracker](https://github.com/dave2k77/neurological_lrd_analysis/issues)
- [PyPI Package](https://pypi.org/project/neurological-lrd-analysis/)
- [GitHub Releases](https://github.com/dave2k77/neurological_lrd_analysis/releases)
- [Research Paper](docs/comprehensive-lrd-estimators-paper.md)

## 📦 Releases and Versioning

This project follows [Semantic Versioning](https://semver.org/) and uses GitHub Actions for automated releases to PyPI.

### Creating a New Release

1. **Bump version**: `python scripts/bump_version.py 0.4.1`
2. **Commit changes**: `git add . && git commit -m "Bump version to 0.4.1"`
3. **Create tag**: `git tag v0.4.1`
4. **Push to GitHub**: `git push origin main --tags`
5. **GitHub Actions** will automatically:
   - Run tests
   - Build the package
   - Publish to PyPI
   - Create a GitHub release

### Installation Options

- **Stable releases**: `pip install neurological-lrd-analysis`
- **Development versions**: `pip install git+https://github.com/dave2k77/neurological_lrd_analysis.git`
- **Test versions**: `pip install --index-url https://test.pypi.org/simple/ neurological-lrd-analysis`

---

**Author:** Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)  
**Email:** d.r.chin@pgr.reading.ac.uk  
**ORCiD:** [https://orcid.org/0009-0003-9434-3919](https://orcid.org/0009-0003-9434-3919)
