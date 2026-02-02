# Neurological LRD Analysis - Documentation Index

## 📚 Complete Documentation Suite

This document provides an organized index of all documentation available for the Neurological LRD Analysis project.

## 🚀 Getting Started

### Essential Reading
1. **[README.md](../README.md)** - Project overview, quick start, and key features
2. **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Detailed setup instructions and troubleshooting
3. **[TUTORIAL.md](TUTORIAL.md)** - Comprehensive tutorial with examples

### Quick Setup
```bash
# Automated setup (recommended)
bash scripts/setup_venv.sh
source neurological_env/bin/activate

# Verify installation
python -c "from neurological_lrd_analysis import BiomedicalHurstEstimatorFactory; print('Ready!')"
```

### Biomedical Scenarios Demo
```bash
# Run biomedical scenarios demonstration
python scripts/biomedical_scenarios_demo.py

# Run neurological conditions demonstration
python scripts/neurological_conditions_demo.py

# Run benchmark with custom parameters
python scripts/run_benchmark.py --hurst-values 0.5,0.7 --lengths 512,1024
```

## 📖 Reference Documentation

### Core API
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation with all classes, methods, and parameters
- **[BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md)** - Enhanced benchmarking with statistical reporting
- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - Application-specific configuration and scoring customization

### Project Structure
- **[project_instructions.md](project_instructions.md)** - Project requirements and specifications

## 🧪 Testing and Validation

### Test Documentation
- **[README_Tests.md](../tests/README_Tests.md)** - Test suite overview
- **[Summary_Comprehensive_Test_Plan.md](Summary_Comprehensive_Test_Plan.md)** - Test plan summary

### Reports and Analysis
- **[reports/TEST_COVERAGE_SUMMARY.md](reports/TEST_COVERAGE_SUMMARY.md)** - Recent test coverage results
- **[reports/ML_BENCHMARK_RESULTS.md](reports/ML_BENCHMARK_RESULTS.md)** - Machine Learning baseline results
- **[reports/DOCUMENTATION_SUMMARY.md](reports/DOCUMENTATION_SUMMARY.md)** - Documentation status report

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific test categories
python -m pytest tests/ -m unit
python -m pytest tests/ -m integration
python -m pytest tests/ -m performance
```

## 🔬 Research and Implementation

### Research Documentation
- **[comprehensive-lrd-estimators-paper.md](comprehensive-lrd-estimators-paper.md)** - Comprehensive LRD estimators research
- **[Techniques for Estimating the Hurst Exponent.md](Techniques%20for%20Estimating%20the%20Hurst%20Exponent.md)** - Hurst estimation techniques
- **[wavelet-based-lrd-estimators.md](wavelet-based-lrd-estimators.md)** - Wavelet-based LRD estimators

### Implementation Documentation
- **[API_Reference_Guide.md](API_Reference_Guide.md)** - Implementation API guide
- **[comprehensive-hurst-library.md](comprehensive-hurst-library.md)** - Comprehensive Hurst library documentation
- **[GPU-Acceleration-Strategy.md](GPU-Acceleration-Strategy.md)** - GPU acceleration strategy

## 📊 Data Models and Analysis

### Data Models
- **[biomedical_data_models_generative_factory_architecture.md](biomedical_data_models_generative_factory_architecture.md)** - Data models architecture
- **[biomedical_time_series_data_analysis_plotting_utility.md](biomedical_time_series_data_analysis_plotting_utility.md)** - Plotting utilities
- **[biomedical-framework-api-documentation.md](biomedical-framework-api-documentation.md)** - Framework API documentation
- **[biomedical-framework-index.md](biomedical-framework-index.md)** - Framework index
- **[biomedical-framework-quick-reference.md](biomedical-framework-quick-reference.md)** - Quick reference guide
- **[biomedical-generative-model-documentation.md](biomedical-generative-model-documentation.md)** - Generative model documentation
- **[biomedical-plotting-system-documentation.md](biomedical-plotting-system-documentation.md)** - Plotting system documentation
- **[Complete_API_Documentation_Biomedical_Time_Series_Analysis.md](Complete_API_Documentation_Biomedical_Time_Series_Analysis.md)** - Complete API documentation
- **[design_algorithms_for_generating_synthetic_data.md](design_algorithms_for_generating_synthetic_data.md)** - Synthetic data generation algorithms

### Research
- **[time-series-algorithms-comprehensive.md](time-series-algorithms-comprehensive.md)** - Time series algorithms research
- **[time-series-models-biomedicine-neuroscience.md](time-series-models-biomedicine-neuroscience.md)** - Biomedical time series models

## 🎯 Usage Examples

### Basic Usage
```python
from neurological_lrd_analysis import BiomedicalHurstEstimatorFactory, EstimatorType
import numpy as np

# Create factory
factory = BiomedicalHurstEstimatorFactory()

# Generate test data
data = np.cumsum(np.random.randn(1000))

# Single method estimation
result = factory.estimate(data, EstimatorType.DFA)
print(f"Hurst: {result.hurst_estimate:.3f}")

# Group estimation
group_result = factory.estimate(data, EstimatorType.ALL)
print(f"Ensemble: {group_result.ensemble_estimate:.3f}")
```

### Advanced Usage
```python
# Custom parameters
result = factory.estimate(
    data, 
    EstimatorType.DFA,
    min_window=20,
    max_window=200,
    confidence_method="bootstrap",
    n_bootstrap=1000
)

# Data quality assessment
from neurological_lrd_analysis import BiomedicalDataProcessor
processor = BiomedicalDataProcessor()
quality = processor.assess_data_quality(data)
print(f"Quality score: {quality['data_quality_score']:.3f}")
```

## 🔧 Development

### Project Structure
```
neurological_lrd_analysis/
├── neurological_lrd_analysis/     # Main package
├── benchmark_core/                # Benchmarking infrastructure
├── benchmark_backends/            # Backend selection
├── benchmark_registry/            # Estimator registry
├── ml_baselines/                 # Machine Learning models
├── docs/                         # Documentation
├── tests/                        # Test suite
├── pyproject.toml                # Project configuration
├── scripts/                      # Setup and demo scripts
└── README.md                     # Project overview
```

### Development Workflow
1. **Setup**: Run `bash scripts/setup_venv.sh` and activate environment
2. **Develop**: Make changes to code
3. **Test**: Run `python -m pytest tests/ -v`
4. **Document**: Update relevant documentation
5. **Validate**: Run benchmarks and examples

## 📈 Performance Benchmarks

### Recent Benchmark Results
- **Total Estimators**: 12+ implemented (Classical + ML)
- **Success Rate**: 100% across all methods
- **Best Performance**: Hybrid ML ensembles provide highest accuracy
- **Computation Time**: Classical: < 1s; ML Inference: 10-50ms

### Benchmarking
```bash
python scripts/run_benchmark.py --biomedical-scenarios eeg_rest --contaminations noise
```

## 🆘 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated
2. **Data Too Short**: Check minimum length requirements
3. **Missing Dependencies**: Install optional packages as needed
4. **Performance Issues**: Consider using GPU acceleration

### Getting Help
- Check [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for setup issues
- Review [TUTORIAL.md](TUTORIAL.md) for usage examples
- Consult [API_REFERENCE.md](API_REFERENCE.md) for detailed API information

## 🔄 Version History

### Current Version: v0.4.2
- ✅ Machine Learning baselines and pretrained models
- ✅ Physics-informed fractional operator features
- ✅ Comprehensive benchmarking system
- ✅ JAX and NumPyro integration
- ✅ Enhanced biomedical scenarios

### Previous Versions
- **v0.3.0**: Convergence analysis and lazy imports
- **v0.2.0**: Wavelet and multifractal estimators
- **v0.1.0**: Initial release with classical estimators

## 📝 Contributing

### Documentation Updates
When adding new features or making changes:

1. **Update API Reference**: Modify [API_REFERENCE.md](API_REFERENCE.md)
2. **Update Tutorial**: Add examples to [TUTORIAL.md](TUTORIAL.md)
3. **Update README**: Keep [README.md](../README.md) current
4. **Update Index**: Maintain this documentation index

### Documentation Standards
- Use clear, concise language
- Include code examples
- Provide troubleshooting information
- Keep documentation up-to-date with code changes

## 📞 Support and Contact

- **Author**: Davian R. Chin
- **Email**: d.r.chin@pgr.reading.ac.uk
- **Research**: PhD in Biomedical Engineering, University of Reading, UK

---

**Last Updated**: February 2026  
**Documentation Version**: 2.0.0  
**Project Version**: 0.4.2
