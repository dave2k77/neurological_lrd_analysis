# Jupyter Notebooks for Neurological LRD Analysis

**Author:** Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)  
**Email:** d.r.chin@pgr.reading.ac.uk  
**ORCiD:** [https://orcid.org/0009-0003-9434-3919](https://orcid.org/0009-0003-9434-3919)  
**Research Focus:** Physics-Informed Fractional Operator Learning for Real-Time Neurological Biomarker Detection

---

## Overview

This directory contains comprehensive Jupyter notebooks designed for demonstration, training, and development purposes. These notebooks provide hands-on tutorials for using the Neurological LRD Analysis library in biomedical and neurological research applications.

## Notebooks

### 1. [01_Biomedical_Time_Series_Analysis.ipynb](01_Biomedical_Time_Series_Analysis.ipynb)

**Purpose:** Comprehensive tutorial for biomedical time series data generation and analysis

**Topics Covered:**
- Installation and setup of the Neurological LRD Analysis library
- Understanding long-range dependence in biomedical signals
- Biomedical time series generation (EEG, ECG, Respiratory)
- Neurological condition simulation (Parkinson's, Epilepsy, etc.)
- Contamination methods (artifacts, noise, missing data)
- Data quality assessment
- Long-range dependence analysis (Classical Methods)
- Machine Learning Baselines (Random Forest, SVR, Gradient Boosting)
- Hyperparameter Optimization (Optuna Integration)
- Pretrained Models (Training, Storage, Loading)
- Comprehensive Benchmarking (Classical vs ML Methods)
- Biomedical scenario comparisons
- Clinical interpretation of results

**Target Audience:** Researchers, clinicians, and students working with neurological and biomedical time series data

**Key Features:**
- Step-by-step demonstrations
- Real-world biomedical scenarios
- Comprehensive visualizations
- Clinical interpretations
- Best practices for data analysis
- ML integration for improved accuracy

---

### 2. [02_Hurst_Estimator_Creation_and_Validation.ipynb](02_Hurst_Estimator_Creation_and_Validation.ipynb)

**Purpose:** Comprehensive tutorial for Hurst estimator creation, usage, and validation

**Topics Covered:**
- Understanding Hurst estimators and their applications
- Estimator factory usage with built-in methods
- Custom estimator development
- Estimator validation techniques
- Performance comparison across methods
- Uncertainty quantification (confidence intervals)
- Best practices for reliable estimation
- Error analysis and debugging

**Target Audience:** Researchers developing new estimation methods or validating existing ones

**Key Features:**
- Detailed method explanations
- Code examples for custom estimators
- Validation frameworks
- Performance benchmarking
- Statistical analysis techniques

---

### 3. [03_Benchmarking_and_Leaderboards.ipynb](03_Benchmarking_and_Leaderboards.ipynb)

**Purpose:** Comprehensive tutorial for systematic benchmarking and leaderboard creation

**Topics Covered:**
- Understanding benchmarking principles for Hurst estimation
- Benchmark setup and configuration
- Systematic data generation for testing
- Performance metrics (accuracy, speed, robustness)
- Leaderboard creation and ranking
- Statistical significance testing
- Publication-ready visualizations
- Custom scoring functions for specific applications

**Target Audience:** Researchers comparing estimation methods and validating new algorithms

**Key Features:**
- Systematic comparison frameworks
- Statistical analysis tools
- Automated leaderboard generation
- Publication-quality plots
- Performance optimization guidance

---

## Getting Started

### Prerequisites

1. **Python Environment**: Python 3.11 or higher
2. **Library Installation**: `pip install neurological-lrd-analysis`
3. **Jupyter Notebook**: JupyterLab or Jupyter Notebook
4. **Additional Dependencies**:
   ```bash
   pip install matplotlib seaborn pandas scipy scikit-learn
   ```

### Running the Notebooks

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dave2k77/neurological_lrd_analysis.git
   cd neurological_lrd_analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Jupyter**:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

4. **Open and run notebooks** in the order listed above

### Notebook Structure

Each notebook follows a consistent structure:
- **Introduction and Overview**
- **Theoretical Background**
- **Practical Examples**
- **Advanced Topics**
- **Summary and Next Steps**

## Educational Use

These notebooks are designed for:
- **Academic Courses**: Biomedical signal processing, time series analysis
- **Research Training**: PhD students, postdocs, research assistants
- **Clinical Education**: Healthcare professionals learning signal analysis
- **Workshop Materials**: Conference tutorials, training sessions

## Research Applications

The notebooks support research in:
- **Neurological Biomarker Development**
- **Clinical Signal Processing**
- **Physics-Informed Machine Learning**
- **Fractional Calculus Applications**
- **Real-Time Monitoring Systems**

### 4. [04_ML_Baselines_Tutorial.ipynb](04_ML_Baselines_Tutorial.ipynb)

**Purpose:** Comprehensive tutorial for Machine Learning baselines in Hurst exponent estimation

**Topics Covered:**
- Feature extraction (74+ features from time series)
- ML baselines (Random Forest, SVR, Gradient Boosting)
- Hyperparameter optimization (Optuna integration)
- Pretrained models (Training, storage, loading)
- Inference systems (Single, batch, ensemble predictions)
- Comprehensive benchmarking (Classical vs ML methods)
- Performance analysis (Accuracy, speed, memory)

**Target Audience:** Researchers and practitioners who want to leverage machine learning for improved Hurst estimation

**Key Features:**
- Complete ML workflow demonstration
- Feature engineering best practices
- Model optimization techniques
- Production-ready inference systems
- Performance benchmarking

---

### 5. [05_Hyperparameter_Optimization_Tutorial.ipynb](05_Hyperparameter_Optimization_Tutorial.ipynb)

**Purpose:** Advanced tutorial for hyperparameter optimization with Optuna

**Topics Covered:**
- Optuna integration (Study creation and optimization)
- Hyperparameter spaces (Defining search spaces for different models)
- Optimization strategies (Different optimization algorithms)
- Multi-objective optimization (Balancing accuracy vs speed)
- Visualization (Optimization progress and results)
- Best practices (Tips for effective hyperparameter tuning)

**Target Audience:** Researchers who want to maximize ML model performance

**Key Features:**
- Advanced optimization techniques
- Multi-objective optimization
- Visualization of optimization progress
- Best practices for hyperparameter tuning

---

### 6. [06_Pretrained_Models_Tutorial.ipynb](06_Pretrained_Models_Tutorial.ipynb)

**Purpose:** Tutorial for pretrained model system and inference deployment

**Topics Covered:**
- Model training (Creating and training model suites)
- Model storage (Efficient model persistence and metadata)
- Model loading (Loading pretrained models for inference)
- Inference systems (Single, batch, and ensemble predictions)
- Model management (Organizing and tracking model versions)
- Production deployment (Best practices for production systems)

**Target Audience:** Practitioners who need efficient model deployment and inference

**Key Features:**
- Production-ready model deployment
- Efficient inference systems
- Model versioning and management
- Best practices for production systems

---

## Customization

Each notebook can be customized for specific research needs:
- **Parameter Modification**: Adjust sampling rates, signal lengths, contamination levels
- **Method Extension**: Add new estimation methods or validation techniques
- **Application Focus**: Tailor examples to specific biomedical domains
- **Visualization**: Modify plots for publication requirements

## Contributing

Contributions to improve these notebooks are welcome:
- **Bug Fixes**: Report and fix issues
- **New Examples**: Add relevant use cases
- **Documentation**: Improve explanations and comments
- **Performance**: Optimize code examples

## Support

For questions or support:
- **Email**: d.r.chin@pgr.reading.ac.uk
- **GitHub Issues**: [Repository Issues](https://github.com/dave2k77/neurological_lrd_analysis/issues)
- **Documentation**: [ReadTheDocs](https://neurological-lrd-analysis.readthedocs.io/)

---

## Citation

If you use these notebooks in your research, please cite:

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

---

**Last Updated:** October 2025  
**Version:** 0.4.2  
**License:** MIT
