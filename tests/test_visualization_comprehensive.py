"""
Comprehensive tests for visualization module.

This module tests all visualization functionality,
plotting functions, and error handling.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, mock_open
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


class TestAccuracyComparisonPlot:
    """Test accuracy comparison plotting functions."""
    
    def test_create_accuracy_comparison_plot_basic(self):
        """Test basic accuracy comparison plot creation."""
        results = {
            "DFA": {"mae": 0.05, "rmse": 0.07},
            "R/S": {"mae": 0.06, "rmse": 0.08},
            "Higuchi": {"mae": 0.04, "rmse": 0.06}
        }
        
        fig = create_accuracy_comparison_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_hurst_plot_different_data_sizes(self):
        """Test Hurst plot creation with different data sizes."""
        data_sizes = [100, 500, 1000, 5000]
        
        for size in data_sizes:
            data = np.random.randn(size)
            hurst_values = [0.3, 0.5, 0.7]
            
            fig = create_hurst_plot(data, hurst_values)
            assert fig is not None
            plt.close(fig)
    
    def test_create_hurst_plot_different_hurst_values(self):
        """Test Hurst plot creation with different Hurst values."""
        data = np.random.randn(1000)
        
        hurst_sets = [
            [0.3, 0.5, 0.7],
            [0.1, 0.9],
            [0.2, 0.4, 0.6, 0.8],
            [0.5]
        ]
        
        for hurst_values in hurst_sets:
            fig = create_hurst_plot(data, hurst_values)
            assert fig is not None
            plt.close(fig)
    
    def test_create_hurst_plot_with_labels(self):
        """Test Hurst plot creation with custom labels."""
        data = np.random.randn(1000)
        hurst_values = [0.3, 0.5, 0.7]
        
        fig = create_hurst_plot(
            data, 
            hurst_values,
            title="Custom Hurst Plot",
            xlabel="Time",
            ylabel="Value"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_hurst_plot_with_style(self):
        """Test Hurst plot creation with custom style."""
        data = np.random.randn(1000)
        hurst_values = [0.3, 0.5, 0.7]
        
        fig = create_hurst_plot(
            data,
            hurst_values,
            style='seaborn-v0_8',
            figsize=(12, 8)
        )
        
        assert fig is not None
        plt.close(fig)


class TestComparisonPlot:
    """Test comparison plotting functions."""
    
    def test_create_comparison_plot_basic(self):
        """Test basic comparison plot creation."""
        data1 = np.random.randn(1000)
        data2 = np.random.randn(1000)
        
        fig = create_comparison_plot(data1, data2, "Data 1", "Data 2")
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_comparison_plot_multiple_datasets(self):
        """Test comparison plot creation with multiple datasets."""
        datasets = {
            "Dataset 1": np.random.randn(1000),
            "Dataset 2": np.random.randn(1000),
            "Dataset 3": np.random.randn(1000)
        }
        
        fig = create_comparison_plot(datasets)
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_comparison_plot_with_metadata(self):
        """Test comparison plot creation with metadata."""
        data1 = np.random.randn(1000)
        data2 = np.random.randn(1000)
        
        metadata = {
            "Dataset 1": {"hurst": 0.3, "length": 1000},
            "Dataset 2": {"hurst": 0.7, "length": 1000}
        }
        
        fig = create_comparison_plot(
            data1, 
            data2, 
            "Data 1", 
            "Data 2",
            metadata=metadata
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_comparison_plot_different_styles(self):
        """Test comparison plot creation with different styles."""
        data1 = np.random.randn(1000)
        data2 = np.random.randn(1000)
        
        styles = ['default', 'seaborn-v0_8', 'ggplot', 'bmh']
        
        for style in styles:
            fig = create_comparison_plot(
                data1, 
                data2, 
                "Data 1", 
                "Data 2",
                style=style
            )
            assert fig is not None
            plt.close(fig)


class TestBenchmarkPlot:
    """Test benchmark plotting functions."""
    
    def test_create_benchmark_plot_basic(self):
        """Test basic benchmark plot creation."""
        results = {
            "DFA": {"mae": 0.05, "rmse": 0.07, "time": 0.1},
            "R/S": {"mae": 0.06, "rmse": 0.08, "time": 0.2},
            "Higuchi": {"mae": 0.04, "rmse": 0.06, "time": 0.15}
        }
        
        fig = create_benchmark_plot(results)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_benchmark_plot_different_metrics(self):
        """Test benchmark plot creation with different metrics."""
        results = {
            "Method 1": {"accuracy": 0.95, "speed": 0.8, "memory": 0.9},
            "Method 2": {"accuracy": 0.90, "speed": 0.9, "memory": 0.8},
            "Method 3": {"accuracy": 0.85, "speed": 0.7, "memory": 0.95}
        }
        
        fig = create_benchmark_plot(results, metrics=["accuracy", "speed", "memory"])
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_benchmark_plot_with_error_bars(self):
        """Test benchmark plot creation with error bars."""
        results = {
            "Method 1": {
                "mae": 0.05, 
                "mae_std": 0.01,
                "time": 0.1,
                "time_std": 0.02
            },
            "Method 2": {
                "mae": 0.06,
                "mae_std": 0.015,
                "time": 0.2,
                "time_std": 0.03
            }
        }
        
        fig = create_benchmark_plot(results, show_error_bars=True)
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_benchmark_plot_radar_chart(self):
        """Test benchmark plot creation as radar chart."""
        results = {
            "Method 1": {"accuracy": 0.95, "speed": 0.8, "memory": 0.9},
            "Method 2": {"accuracy": 0.90, "speed": 0.9, "memory": 0.8}
        }
        
        fig = create_benchmark_plot(results, plot_type="radar")
        
        assert fig is not None
        plt.close(fig)


class TestScatterPlot:
    """Test scatter plotting functions."""
    
    def test_create_scatter_plot_basic(self):
        """Test basic scatter plot creation."""
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        fig = create_scatter_plot(x, y)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_scatter_plot_with_colors(self):
        """Test scatter plot creation with colors."""
        x = np.random.randn(100)
        y = np.random.randn(100)
        colors = np.random.randn(100)
        
        fig = create_scatter_plot(x, y, c=colors)
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_scatter_plot_with_sizes(self):
        """Test scatter plot creation with sizes."""
        x = np.random.randn(100)
        y = np.random.randn(100)
        sizes = np.random.randint(10, 100, 100)
        
        fig = create_scatter_plot(x, y, s=sizes)
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_scatter_plot_with_regression(self):
        """Test scatter plot creation with regression line."""
        x = np.random.randn(100)
        y = 2 * x + np.random.randn(100) * 0.1
        
        fig = create_scatter_plot(x, y, show_regression=True)
        
        assert fig is not None
        plt.close(fig)


class TestHistogramPlot:
    """Test histogram plotting functions."""
    
    def test_create_histogram_plot_basic(self):
        """Test basic histogram plot creation."""
        data = np.random.randn(1000)
        
        fig = create_histogram_plot(data)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_histogram_plot_multiple_datasets(self):
        """Test histogram plot creation with multiple datasets."""
        data1 = np.random.randn(1000)
        data2 = np.random.randn(1000) + 1
        
        fig = create_histogram_plot([data1, data2], labels=["Data 1", "Data 2"])
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_histogram_plot_different_bins(self):
        """Test histogram plot creation with different bin counts."""
        data = np.random.randn(1000)
        
        bin_counts = [10, 20, 50, 100]
        
        for bins in bin_counts:
            fig = create_histogram_plot(data, bins=bins)
            assert fig is not None
            plt.close(fig)
    
    def test_create_histogram_plot_with_density(self):
        """Test histogram plot creation with density."""
        data = np.random.randn(1000)
        
        fig = create_histogram_plot(data, density=True)
        
        assert fig is not None
        plt.close(fig)


class TestCorrelationPlot:
    """Test correlation plotting functions."""
    
    def test_create_correlation_plot_basic(self):
        """Test basic correlation plot creation."""
        data = np.random.randn(100, 5)
        
        fig = create_correlation_plot(data)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_correlation_plot_with_labels(self):
        """Test correlation plot creation with labels."""
        data = np.random.randn(100, 4)
        labels = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
        
        fig = create_correlation_plot(data, labels=labels)
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_correlation_plot_heatmap(self):
        """Test correlation plot creation as heatmap."""
        data = np.random.randn(100, 5)
        
        fig = create_correlation_plot(data, plot_type="heatmap")
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_correlation_plot_network(self):
        """Test correlation plot creation as network."""
        data = np.random.randn(100, 5)
        
        fig = create_correlation_plot(data, plot_type="network")
        
        assert fig is not None
        plt.close(fig)


class TestTimeSeriesPlot:
    """Test time series plotting functions."""
    
    def test_create_time_series_plot_basic(self):
        """Test basic time series plot creation."""
        data = np.random.randn(1000)
        
        fig = create_time_series_plot(data)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_time_series_plot_multiple_series(self):
        """Test time series plot creation with multiple series."""
        data1 = np.random.randn(1000)
        data2 = np.random.randn(1000) + 1
        
        fig = create_time_series_plot([data1, data2], labels=["Series 1", "Series 2"])
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_time_series_plot_with_time_axis(self):
        """Test time series plot creation with time axis."""
        data = np.random.randn(1000)
        time_axis = np.linspace(0, 10, 1000)
        
        fig = create_time_series_plot(data, time_axis=time_axis)
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_time_series_plot_with_annotations(self):
        """Test time series plot creation with annotations."""
        data = np.random.randn(1000)
        annotations = {
            100: "Event 1",
            500: "Event 2",
            800: "Event 3"
        }
        
        fig = create_time_series_plot(data, annotations=annotations)
        
        assert fig is not None
        plt.close(fig)


class TestSavePlot:
    """Test plot saving functions."""
    
    def test_save_plot_basic(self):
        """Test basic plot saving."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_plot(fig, tmp.name)
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)
        
        plt.close(fig)
    
    def test_save_plot_different_formats(self):
        """Test plot saving in different formats."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        
        formats = ['png', 'pdf', 'svg', 'jpg']
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=f'.{fmt}', delete=False) as tmp:
                save_plot(fig, tmp.name, format=fmt)
                assert os.path.exists(tmp.name)
                os.unlink(tmp.name)
        
        plt.close(fig)
    
    def test_save_plot_with_dpi(self):
        """Test plot saving with different DPI settings."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        
        dpi_values = [72, 150, 300]
        
        for dpi in dpi_values:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                save_plot(fig, tmp.name, dpi=dpi)
                assert os.path.exists(tmp.name)
                os.unlink(tmp.name)
        
        plt.close(fig)
    
    def test_save_plot_with_metadata(self):
        """Test plot saving with metadata."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        
        metadata = {
            "Title": "Test Plot",
            "Author": "Test Author",
            "Date": "2024-01-01"
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_plot(fig, tmp.name, metadata=metadata)
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)
        
        plt.close(fig)


class TestPlotGrid:
    """Test plot grid functions."""
    
    def test_create_plot_grid_basic(self):
        """Test basic plot grid creation."""
        data = [np.random.randn(100) for _ in range(4)]
        
        fig = create_plot_grid(data, nrows=2, ncols=2)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)
    
    def test_create_plot_grid_different_layouts(self):
        """Test plot grid creation with different layouts."""
        data = [np.random.randn(100) for _ in range(6)]
        
        layouts = [
            (2, 3),
            (3, 2),
            (1, 6),
            (6, 1)
        ]
        
        for nrows, ncols in layouts:
            fig = create_plot_grid(data, nrows=nrows, ncols=ncols)
            assert fig is not None
            plt.close(fig)
    
    def test_create_plot_grid_with_titles(self):
        """Test plot grid creation with titles."""
        data = [np.random.randn(100) for _ in range(4)]
        titles = ["Plot 1", "Plot 2", "Plot 3", "Plot 4"]
        
        fig = create_plot_grid(data, nrows=2, ncols=2, titles=titles)
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_plot_grid_with_shared_axes(self):
        """Test plot grid creation with shared axes."""
        data = [np.random.randn(100) for _ in range(4)]
        
        fig = create_plot_grid(
            data, 
            nrows=2, 
            ncols=2, 
            sharex=True, 
            sharey=True
        )
        
        assert fig is not None
        plt.close(fig)


class TestFormatPlotStyle:
    """Test plot style formatting functions."""
    
    def test_format_plot_style_basic(self):
        """Test basic plot style formatting."""
        style = format_plot_style("default")
        
        assert isinstance(style, dict)
        assert "figure" in style
        assert "axes" in style
    
    def test_format_plot_style_different_styles(self):
        """Test plot style formatting for different styles."""
        styles = ["default", "seaborn-v0_8", "ggplot", "bmh", "classic"]
        
        for style_name in styles:
            style = format_plot_style(style_name)
            assert isinstance(style, dict)
            assert "figure" in style
            assert "axes" in style
    
    def test_format_plot_style_custom_parameters(self):
        """Test plot style formatting with custom parameters."""
        style = format_plot_style(
            "default",
            figsize=(12, 8),
            dpi=150,
            fontsize=12
        )
        
        assert isinstance(style, dict)
        assert "figure" in style
        assert "axes" in style
    
    def test_format_plot_style_color_scheme(self):
        """Test plot style formatting with color scheme."""
        style = format_plot_style("default", color_scheme="viridis")
        
        assert isinstance(style, dict)
        assert "figure" in style
        assert "axes" in style


class TestErrorHandling:
    """Test error handling in visualization functions."""
    
    def test_create_hurst_plot_invalid_data(self):
        """Test Hurst plot creation with invalid data."""
        # Empty data
        try:
            fig = create_hurst_plot(np.array([]), [0.5])
            assert fig is None or hasattr(fig, 'savefig')
            if fig:
                plt.close(fig)
        except (ValueError, IndexError):
            # Expected behavior for empty data
            pass
        
        # Data with NaN
        try:
            data = np.array([1, 2, np.nan, 4, 5])
            fig = create_hurst_plot(data, [0.5])
            assert fig is None or hasattr(fig, 'savefig')
            if fig:
                plt.close(fig)
        except (ValueError, RuntimeError):
            # Expected behavior for data with NaN
            pass
    
    def test_create_comparison_plot_mismatched_data(self):
        """Test comparison plot creation with mismatched data."""
        data1 = np.random.randn(100)
        data2 = np.random.randn(200)  # Different length
        
        try:
            fig = create_comparison_plot(data1, data2, "Data 1", "Data 2")
            assert fig is None or hasattr(fig, 'savefig')
            if fig:
                plt.close(fig)
        except (ValueError, IndexError):
            # Expected behavior for mismatched data
            pass
    
    def test_create_benchmark_plot_empty_results(self):
        """Test benchmark plot creation with empty results."""
        try:
            fig = create_benchmark_plot({})
            assert fig is None or hasattr(fig, 'savefig')
            if fig:
                plt.close(fig)
        except (ValueError, KeyError):
            # Expected behavior for empty results
            pass
    
    def test_save_plot_invalid_path(self):
        """Test plot saving with invalid path."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        
        try:
            save_plot(fig, "/invalid/path/plot.png")
            # If it doesn't raise an error, check that it handles gracefully
        except (OSError, IOError):
            # Expected behavior for invalid path
            pass
        
        plt.close(fig)
    
    def test_create_plot_grid_insufficient_data(self):
        """Test plot grid creation with insufficient data."""
        data = [np.random.randn(100)]  # Only one dataset
        
        try:
            fig = create_plot_grid(data, nrows=2, ncols=2)
            assert fig is None or hasattr(fig, 'savefig')
            if fig:
                plt.close(fig)
        except (ValueError, IndexError):
            # Expected behavior for insufficient data
            pass


class TestPerformance:
    """Test performance characteristics."""
    
    def test_plot_creation_performance(self):
        """Test that plot creation is reasonably fast."""
        import time
        
        data = np.random.randn(1000)
        
        start_time = time.time()
        fig = create_hurst_plot(data, [0.5])
        end_time = time.time()
        
        # Should complete in less than 1 second
        assert end_time - start_time < 1.0
        assert fig is not None
        plt.close(fig)
    
    def test_large_data_plotting_performance(self):
        """Test plotting performance with large datasets."""
        import time
        
        data = np.random.randn(10000)
        
        start_time = time.time()
        fig = create_time_series_plot(data)
        end_time = time.time()
        
        # Should complete in less than 2 seconds
        assert end_time - start_time < 2.0
        assert fig is not None
        plt.close(fig)
    
    def test_multiple_plot_creation_performance(self):
        """Test performance of creating multiple plots."""
        import time
        
        datasets = [np.random.randn(1000) for _ in range(10)]
        
        start_time = time.time()
        
        for data in datasets:
            fig = create_hurst_plot(data, [0.5])
            plt.close(fig)
        
        end_time = time.time()
        
        # Should complete 10 plots in less than 5 seconds
        assert end_time - start_time < 5.0
    
    def test_plot_saving_performance(self):
        """Test plot saving performance."""
        import time
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.random.randn(1000))
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            start_time = time.time()
            save_plot(fig, tmp.name)
            end_time = time.time()
            
            # Should complete in less than 0.5 seconds
            assert end_time - start_time < 0.5
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)
        
        plt.close(fig)
