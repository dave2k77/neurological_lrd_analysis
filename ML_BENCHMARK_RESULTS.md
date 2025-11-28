# ML Models Benchmark Results

## Overview
The ML baseline models benchmark was executed successfully. The benchmark evaluated three models: Random Forest, Support Vector Regression (SVR), and Gradient Boosting.

## Results Summary

| Model | MSE | MAE | R² Score | RMSE | Correlation |
|-------|-----|-----|----------|------|-------------|
| **Gradient Boosting** | 0.0000 | 0.0009 | **0.9997** | 0.0031 | 0.9999 |
| **Random Forest** | 0.0001 | 0.0032 | 0.9985 | 0.0073 | 0.9993 |
| **SVR** | 0.0070 | 0.0683 | 0.8003 | 0.0836 | 0.9526 |

**Best Performing Model:** Gradient Boosting (R² = 0.9997)

## Details
- **Training Data:** 210 samples (30 samples per Hurst value: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
- **Data Length:** 1000 points
- **Features:** 74 extracted features per sample
- **Hyperparameter Optimization:** Disabled (Optuna not available)

## Visualizations
Plots have been saved to `results/ml_baselines_demo/ml_baselines_performance.png`.
