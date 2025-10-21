"""
Simple tests for visualization module.

This module tests the actual available visualization functions.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
import tempfile
import os

from neurological_lrd_analysis.benchmark_core.visualization import (
    create_accuracy_comparison_plot,
    create_bias_distribution_plot,
    create_error_vs_hurst_plot,
    create_efficiency_analysis_plot,
    create_uncertainty_quantification_plot,
    create_focused_analysis_report,
    create_summary_dashboard
)


class MockBenchmarkResult:
    """Mock benchmark result object for testing."""
    def __init__(self, estimator, mae=0.05, rmse=0.08, correlation=0.95, 
                 true_hurst=0.5, estimated_hurst=0.52, computation_time=0.1,
                 success=True, bias=0.02, error=0.03, convergence_flag=True,
                 hurst_estimate=0.52, absolute_error=0.02, confidence_interval=(0.50, 0.54),
                 relative_error=0.04, standard_error=0.01, p_value=0.05):
        self.estimator = estimator
        self.mae = mae
        self.rmse = rmse
        self.correlation = correlation
        self.true_hurst = true_hurst
        self.estimated_hurst = estimated_hurst
        self.computation_time = computation_time
        self.success = success
        self.bias = bias
        self.error = error
        self.convergence_flag = convergence_flag
        self.hurst_estimate = hurst_estimate
        self.absolute_error = absolute_error
        self.confidence_interval = confidence_interval
        self.relative_error = relative_error
        self.standard_error = standard_error
        self.p_value = p_value


class TestAccuracyComparisonPlot:
    """Test accuracy comparison plotting functions."""
    
    def test_create_accuracy_comparison_plot_basic(self):
        """Test basic accuracy comparison plot creation."""
        results = [
            MockBenchmarkResult("DFA", mae=0.05, rmse=0.07),
            MockBenchmarkResult("R/S", mae=0.06, rmse=0.08),
            MockBenchmarkResult("Higuchi", mae=0.04, rmse=0.06)
        ]
        
        fig = create_accuracy_comparison_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_accuracy_comparison_plot_different_results(self):
        """Test accuracy comparison plot creation with different results."""
        results = [
            MockBenchmarkResult("Method 1", mae=0.03, rmse=0.05, correlation=0.95),
            MockBenchmarkResult("Method 2", mae=0.04, rmse=0.06, correlation=0.92),
            MockBenchmarkResult("Method 3", mae=0.05, rmse=0.07, correlation=0.90)
        ]
        
        fig = create_accuracy_comparison_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_accuracy_comparison_plot_empty_results(self):
        """Test accuracy comparison plot creation with empty results."""
        try:
            fig = create_accuracy_comparison_plot([])
            assert fig is None or hasattr(fig, 'savefig')
            if fig:
                plt.close(fig)
        except (ValueError, KeyError):
            # Expected behavior for empty results
            pass


class TestBiasDistributionPlot:
    """Test bias distribution plotting functions."""
    
    def test_create_bias_distribution_plot_basic(self):
        """Test basic bias distribution plot creation."""
        results = [
            MockBenchmarkResult("DFA", bias=0.02, true_hurst=0.5, estimated_hurst=0.52),
            MockBenchmarkResult("R/S", bias=0.03, true_hurst=0.5, estimated_hurst=0.53),
            MockBenchmarkResult("Higuchi", bias=0.01, true_hurst=0.5, estimated_hurst=0.51)
        ]
        
        fig = create_bias_distribution_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_bias_distribution_plot_different_data(self):
        """Test bias distribution plot creation with different data."""
        results = [
            MockBenchmarkResult("Method A", bias=0.01, true_hurst=0.3, estimated_hurst=0.31),
            MockBenchmarkResult("Method B", bias=0.02, true_hurst=0.7, estimated_hurst=0.72),
            MockBenchmarkResult("Method C", bias=0.015, true_hurst=0.9, estimated_hurst=0.915)
        ]
        
        fig = create_bias_distribution_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)


class TestErrorVsHurstPlot:
    """Test error vs Hurst plotting functions."""
    
    def test_create_error_vs_hurst_plot_basic(self):
        """Test basic error vs Hurst plot creation."""
        results = [
            MockBenchmarkResult("DFA", error=0.02, true_hurst=0.3),
            MockBenchmarkResult("DFA", error=0.03, true_hurst=0.5),
            MockBenchmarkResult("DFA", error=0.04, true_hurst=0.7),
            MockBenchmarkResult("R/S", error=0.025, true_hurst=0.3),
            MockBenchmarkResult("R/S", error=0.035, true_hurst=0.5),
            MockBenchmarkResult("R/S", error=0.045, true_hurst=0.7)
        ]
        
        fig = create_error_vs_hurst_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_error_vs_hurst_plot_different_data(self):
        """Test error vs Hurst plot creation with different data."""
        results = [
            MockBenchmarkResult("Method X", error=0.01, true_hurst=0.2),
            MockBenchmarkResult("Method X", error=0.02, true_hurst=0.4),
            MockBenchmarkResult("Method X", error=0.03, true_hurst=0.6),
            MockBenchmarkResult("Method Y", error=0.015, true_hurst=0.2),
            MockBenchmarkResult("Method Y", error=0.025, true_hurst=0.4),
            MockBenchmarkResult("Method Y", error=0.035, true_hurst=0.6)
        ]
        
        fig = create_error_vs_hurst_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)


class TestEfficiencyAnalysisPlot:
    """Test efficiency analysis plotting functions."""
    
    def test_create_efficiency_analysis_plot_basic(self):
        """Test basic efficiency analysis plot creation."""
        results = [
            MockBenchmarkResult("DFA", computation_time=0.1, mae=0.05),
            MockBenchmarkResult("R/S", computation_time=0.2, mae=0.06),
            MockBenchmarkResult("Higuchi", computation_time=0.15, mae=0.04)
        ]
        
        fig = create_efficiency_analysis_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_efficiency_analysis_plot_different_metrics(self):
        """Test efficiency analysis plot creation with different metrics."""
        results = [
            MockBenchmarkResult("Fast Method", computation_time=0.05, mae=0.08),
            MockBenchmarkResult("Accurate Method", computation_time=0.3, mae=0.03),
            MockBenchmarkResult("Balanced Method", computation_time=0.1, mae=0.05)
        ]
        
        fig = create_efficiency_analysis_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)


class TestUncertaintyQuantificationPlot:
    """Test uncertainty quantification plotting functions."""
    
    def test_create_uncertainty_quantification_plot_basic(self):
        """Test basic uncertainty quantification plot creation."""
        results = [
            MockBenchmarkResult("DFA", true_hurst=0.5, estimated_hurst=0.52, error=0.02),
            MockBenchmarkResult("R/S", true_hurst=0.5, estimated_hurst=0.53, error=0.03),
            MockBenchmarkResult("Higuchi", true_hurst=0.5, estimated_hurst=0.51, error=0.01)
        ]
        
        fig = create_uncertainty_quantification_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_uncertainty_quantification_plot_different_data(self):
        """Test uncertainty quantification plot creation with different data."""
        results = [
            MockBenchmarkResult("Method A", true_hurst=0.3, estimated_hurst=0.32, error=0.02),
            MockBenchmarkResult("Method B", true_hurst=0.7, estimated_hurst=0.72, error=0.02),
            MockBenchmarkResult("Method C", true_hurst=0.9, estimated_hurst=0.91, error=0.01)
        ]
        
        fig = create_uncertainty_quantification_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)


class TestFocusedAnalysisReport:
    """Test focused analysis report functions."""
    
    def test_create_focused_analysis_report_basic(self):
        """Test basic focused analysis report creation."""
        results = [
            MockBenchmarkResult("DFA", mae=0.05, rmse=0.08, correlation=0.95),
            MockBenchmarkResult("R/S", mae=0.06, rmse=0.09, correlation=0.92),
            MockBenchmarkResult("Higuchi", mae=0.04, rmse=0.07, correlation=0.96)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # create_focused_analysis_report returns None, it just creates files
            create_focused_analysis_report(results, output_dir=temp_dir)
            # Check that the function completed without error
            assert True  # If we get here, the function completed successfully
    
    def test_create_focused_analysis_report_different_data(self):
        """Test focused analysis report creation with different data."""
        results = [
            MockBenchmarkResult("Method 1", mae=0.03, rmse=0.05, correlation=0.98),
            MockBenchmarkResult("Method 2", mae=0.04, rmse=0.06, correlation=0.95),
            MockBenchmarkResult("Method 3", mae=0.05, rmse=0.07, correlation=0.93)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # create_focused_analysis_report returns None, it just creates files
            create_focused_analysis_report(results, output_dir=temp_dir)
            # Check that the function completed without error
            assert True  # If we get here, the function completed successfully


class TestSummaryDashboard:
    """Test summary dashboard functions."""
    
    def test_create_summary_dashboard_basic(self):
        """Test basic summary dashboard creation."""
        results = [
            MockBenchmarkResult("DFA", mae=0.05, rmse=0.08, correlation=0.95, computation_time=0.1),
            MockBenchmarkResult("R/S", mae=0.06, rmse=0.09, correlation=0.92, computation_time=0.2),
            MockBenchmarkResult("Higuchi", mae=0.04, rmse=0.07, correlation=0.96, computation_time=0.15)
        ]
        
        dashboard = create_summary_dashboard(results)
        
        assert dashboard is not None
        assert hasattr(dashboard, 'savefig')
        plt.close(dashboard)
    
    def test_create_summary_dashboard_different_data(self):
        """Test summary dashboard creation with different data."""
        results = [
            MockBenchmarkResult("Fast Method", mae=0.08, rmse=0.12, correlation=0.90, computation_time=0.05),
            MockBenchmarkResult("Accurate Method", mae=0.03, rmse=0.05, correlation=0.98, computation_time=0.3),
            MockBenchmarkResult("Balanced Method", mae=0.05, rmse=0.08, correlation=0.95, computation_time=0.1)
        ]
        
        dashboard = create_summary_dashboard(results)
        
        assert dashboard is not None
        assert hasattr(dashboard, 'savefig')
        plt.close(dashboard)


class TestErrorHandling:
    """Test error handling in visualization functions."""
    
    def test_create_accuracy_comparison_plot_invalid_data(self):
        """Test accuracy comparison plot with invalid data."""
        try:
            fig = create_accuracy_comparison_plot({"invalid": "data"})
            assert fig is None or hasattr(fig, 'savefig')
            if fig:
                plt.close(fig)
        except (ValueError, AttributeError, KeyError):
            # Expected behavior for invalid data
            pass
    
    def test_create_bias_distribution_plot_invalid_data(self):
        """Test bias distribution plot with invalid data."""
        try:
            fig = create_bias_distribution_plot({"invalid": "data"})
            assert fig is None or hasattr(fig, 'savefig')
            if fig:
                plt.close(fig)
        except (ValueError, AttributeError, KeyError):
            # Expected behavior for invalid data
            pass
    
    def test_create_error_vs_hurst_plot_invalid_data(self):
        """Test error vs Hurst plot with invalid data."""
        try:
            fig = create_error_vs_hurst_plot({"invalid": "data"})
            assert fig is None or hasattr(fig, 'savefig')
            if fig:
                plt.close(fig)
        except (ValueError, AttributeError, KeyError):
            # Expected behavior for invalid data
            pass


class TestPerformance:
    """Test performance of visualization functions."""
    
    def test_plot_creation_performance(self):
        """Test that plot creation completes within reasonable time."""
        results = [
            MockBenchmarkResult("DFA", mae=0.05, rmse=0.08),
            MockBenchmarkResult("R/S", mae=0.06, rmse=0.09),
            MockBenchmarkResult("Higuchi", mae=0.04, rmse=0.07)
        ]
        
        import time
        start_time = time.time()
        fig = create_accuracy_comparison_plot(results)
        end_time = time.time()
        
        assert fig is not None
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds
        plt.close(fig)
    
    def test_report_creation_performance(self):
        """Test that report creation completes within reasonable time."""
        results = [
            MockBenchmarkResult("DFA", mae=0.05, rmse=0.08),
            MockBenchmarkResult("R/S", mae=0.06, rmse=0.09),
            MockBenchmarkResult("Higuchi", mae=0.04, rmse=0.07)
        ]
        
        import time
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # create_focused_analysis_report returns None, it just creates files
            create_focused_analysis_report(results, output_dir=temp_dir)
            end_time = time.time()
        
        # Check that the function completed without error
        assert True  # If we get here, the function completed successfully
        assert (end_time - start_time) < 10.0  # Should complete within 10 seconds
    
    def test_dashboard_creation_performance(self):
        """Test that dashboard creation completes within reasonable time."""
        results = [
            MockBenchmarkResult("DFA", mae=0.05, rmse=0.08, computation_time=0.1),
            MockBenchmarkResult("R/S", mae=0.06, rmse=0.09, computation_time=0.2),
            MockBenchmarkResult("Higuchi", mae=0.04, rmse=0.07, computation_time=0.15)
        ]
        
        import time
        start_time = time.time()
        dashboard = create_summary_dashboard(results)
        end_time = time.time()
        
        assert dashboard is not None
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds
        plt.close(dashboard)