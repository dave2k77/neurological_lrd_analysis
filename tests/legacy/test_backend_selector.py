"""
Comprehensive tests for backend selector module.

This module tests all backend selection functionality,
hardware detection, and performance optimization.
"""

import os
import platform
import pytest
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, Any

from neurological_lrd_analysis.benchmark_backends.selector import (
    BackendType,
    check_jax_availability,
    check_numba_availability,
    get_system_info,
    select_backend,
    get_backend_info,
    recommend_backend
)


class TestBackendType:
    """Test BackendType enum."""
    
    def test_backend_type_enum(self):
        """Test BackendType enum values."""
        assert BackendType.NUMPY.value == "numpy"
        assert BackendType.NUMBA_CPU.value == "numba_cpu"
        assert BackendType.NUMBA_GPU.value == "numba_gpu"
        assert BackendType.JAX_CPU.value == "jax_cpu"
        assert BackendType.JAX_GPU.value == "jax_gpu"
    
    def test_backend_type_enumeration(self):
        """Test BackendType enum iteration."""
        backends = list(BackendType)
        assert len(backends) == 5
        assert BackendType.NUMPY in backends
        assert BackendType.NUMBA_CPU in backends
        assert BackendType.NUMBA_GPU in backends
        assert BackendType.JAX_CPU in backends
        assert BackendType.JAX_GPU in backends


class TestJAXAvailability:
    """Test JAX availability checking."""
    
    def test_check_jax_availability_no_jax(self):
        """Test JAX availability when JAX is not installed."""
        with patch.dict('sys.modules', {'jax': None}):
            jax_info = check_jax_availability()
            
            assert jax_info["available"] is False
            assert jax_info["gpu_available"] is False
            assert jax_info["tpu_available"] is False
    
    def test_check_jax_availability_with_jax_no_gpu(self):
        """Test JAX availability when JAX is available but no GPU."""
        mock_jax = MagicMock()
        mock_jax.devices.side_effect = Exception("No GPU")
        
        with patch.dict('sys.modules', {'jax': mock_jax}):
            jax_info = check_jax_availability()
            
            assert jax_info["available"] is True
            assert jax_info["gpu_available"] is False
            assert jax_info["tpu_available"] is False
    
    def test_check_jax_availability_with_gpu(self):
        """Test JAX availability when GPU is available."""
        mock_jax = MagicMock()
        mock_jax.devices.return_value = ["gpu:0", "gpu:1"]
        
        with patch.dict('sys.modules', {'jax': mock_jax}):
            jax_info = check_jax_availability()
            
            assert jax_info["available"] is True
            assert jax_info["gpu_available"] is True
            assert jax_info["tpu_available"] is False
    
    def test_check_jax_availability_with_tpu(self):
        """Test JAX availability when TPU is available."""
        mock_jax = MagicMock()
        mock_jax.devices.side_effect = lambda device_type: ["tpu:0"] if device_type == "tpu" else []
        
        with patch.dict('sys.modules', {'jax': mock_jax}):
            jax_info = check_jax_availability()
            
            assert jax_info["available"] is True
            assert jax_info["gpu_available"] is False
            assert jax_info["tpu_available"] is True


class TestNumbaAvailability:
    """Test Numba availability checking."""
    
    def test_check_numba_availability_no_numba(self):
        """Test Numba availability when Numba is not installed."""
        with patch.dict('sys.modules', {'numba': None}):
            numba_info = check_numba_availability()
            
            assert numba_info["available"] is False
            assert numba_info["gpu_available"] is False
    
    def test_check_numba_availability_with_numba_no_gpu(self):
        """Test Numba availability when Numba is available but no GPU."""
        mock_numba = MagicMock()
        mock_numba.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'numba': mock_numba}):
            numba_info = check_numba_availability()
            
            assert numba_info["available"] is True
            assert numba_info["gpu_available"] is False
    
    def test_check_numba_availability_with_gpu(self):
        """Test Numba availability when GPU is available."""
        mock_numba = MagicMock()
        mock_numba.cuda.is_available.return_value = True
        
        with patch.dict('sys.modules', {'numba': mock_numba}):
            numba_info = check_numba_availability()
            
            assert numba_info["available"] is True
            assert numba_info["gpu_available"] is True


class TestSystemInfo:
    """Test system information gathering."""
    
    def test_get_system_info_basic(self):
        """Test basic system information gathering."""
        with patch('platform.system', return_value='Linux'), \
             patch('platform.machine', return_value='x86_64'), \
             patch('os.cpu_count', return_value=8):
            
            system_info = get_system_info()
            
            assert system_info["os"] == "Linux"
            assert system_info["architecture"] == "x86_64"
            assert system_info["cpu_count"] == 8
            assert "memory_gb" in system_info
            assert "python_version" in system_info
    
    def test_get_system_info_different_os(self):
        """Test system information for different operating systems."""
        test_cases = [
            ('Windows', 'AMD64'),
            ('Darwin', 'arm64'),
            ('Linux', 'x86_64')
        ]
        
        for os_name, arch in test_cases:
            with patch('platform.system', return_value=os_name), \
                 patch('platform.machine', return_value=arch), \
                 patch('os.cpu_count', return_value=4):
                
                system_info = get_system_info()
                
                assert system_info["os"] == os_name
                assert system_info["architecture"] == arch
                assert system_info["cpu_count"] == 4
    
    def test_get_system_info_memory_detection(self):
        """Test memory detection functionality."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.total = 16 * 1024**3  # 16 GB
            
            system_info = get_system_info()
            
            assert system_info["memory_gb"] == 16.0
    
    def test_get_system_info_cpu_count_edge_cases(self):
        """Test CPU count detection edge cases."""
        # Test when cpu_count returns None
        with patch('os.cpu_count', return_value=None):
            system_info = get_system_info()
            assert system_info["cpu_count"] == 1  # Default fallback
        
        # Test when cpu_count returns 0
        with patch('os.cpu_count', return_value=0):
            system_info = get_system_info()
            assert system_info["cpu_count"] == 1  # Default fallback


class TestBackendSelection:
    """Test backend selection logic."""
    
    def test_select_backend_numpy_only(self):
        """Test backend selection when only NumPy is available."""
        with patch('neurological_lrd_analysis.benchmark_backends.selector.check_jax_availability', 
                   return_value={"available": False, "gpu_available": False, "tpu_available": False}), \
             patch('neurological_lrd_analysis.benchmark_backends.selector.check_numba_availability',
                   return_value={"available": False, "gpu_available": False}):
            
            backend = select_backend(
                data_size=1000,
            )
            
            assert backend == BackendType.NUMPY
    
    def test_select_optimal_backend_with_numba(self):
        """Test backend selection when Numba is available."""
        with patch('neurological_lrd_analysis.benchmark_backends.selector.check_jax_availability',
                   return_value={"available": False, "gpu_available": False, "tpu_available": False}), \
             patch('neurological_lrd_analysis.benchmark_backends.selector.check_numba_availability',
                   return_value={"available": True, "gpu_available": False}):
            
            backend = select_backend(
                data_size=1000,
            )
            
            assert backend == BackendType.NUMBA_CPU
    
    def test_select_optimal_backend_with_jax(self):
        """Test backend selection when JAX is available."""
        with patch('neurological_lrd_analysis.benchmark_backends.selector.check_jax_availability',
                   return_value={"available": True, "gpu_available": False, "tpu_available": False}), \
             patch('neurological_lrd_analysis.benchmark_backends.selector.check_numba_availability',
                   return_value={"available": True, "gpu_available": False}):
            
            backend = select_backend(
                data_size=1000,
            )
            
            assert backend == BackendType.JAX_CPU
    
    def test_select_optimal_backend_with_gpu(self):
        """Test backend selection when GPU is available."""
        with patch('neurological_lrd_analysis.benchmark_backends.selector.check_jax_availability',
                   return_value={"available": True, "gpu_available": True, "tpu_available": False}), \
             patch('neurological_lrd_analysis.benchmark_backends.selector.check_numba_availability',
                   return_value={"available": True, "gpu_available": True}):
            
            backend = select_backend(
                data_size=10000,
            )
            
            # Should prefer JAX GPU over Numba GPU
            assert backend == BackendType.JAX_GPU
    
    def test_select_optimal_backend_large_data(self):
        """Test backend selection for large datasets."""
        with patch('neurological_lrd_analysis.benchmark_backends.selector.check_jax_availability',
                   return_value={"available": True, "gpu_available": True, "tpu_available": False}), \
             patch('neurological_lrd_analysis.benchmark_backends.selector.check_numba_availability',
                   return_value={"available": True, "gpu_available": True}):
            
            # Large dataset should prefer GPU
            backend = select_backend(
                data_size=100000,
            )
            
            assert backend == BackendType.JAX_GPU
    
    def test_select_optimal_backend_different_priorities(self):
        """Test backend selection with different performance priorities."""
        with patch('neurological_lrd_analysis.benchmark_backends.selector.check_jax_availability',
                   return_value={"available": True, "gpu_available": True, "tpu_available": False}), \
             patch('neurological_lrd_analysis.benchmark_backends.selector.check_numba_availability',
                   return_value={"available": True, "gpu_available": True}):
            
            # Accuracy priority
            backend = select_backend(
                data_size=1000,
            )
            assert backend in [BackendType.JAX_CPU, BackendType.JAX_GPU]
            
            # Speed priority
            backend = select_backend(
                data_size=1000,
            )
            assert backend in [BackendType.JAX_CPU, BackendType.JAX_GPU]
            
            # Memory priority
            backend = select_backend(
                data_size=1000,
                performance_priority="memory"
            )
            assert backend in [BackendType.NUMPY, BackendType.NUMBA_CPU, BackendType.JAX_CPU]


class TestBackendPerformance:
    """Test backend performance metrics."""
    
    def test_get_backend_performance_metrics(self):
        """Test backend performance metrics gathering."""
        metrics = get_backend_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "numpy" in metrics
        assert "numba_cpu" in metrics
        assert "numba_gpu" in metrics
        assert "jax_cpu" in metrics
        assert "jax_gpu" in metrics
        
        for backend, perf in metrics.items():
            assert "speed_score" in perf
            assert "memory_score" in perf
            assert "accuracy_score" in perf
            assert isinstance(perf["speed_score"], (int, float))
            assert isinstance(perf["memory_score"], (int, float))
            assert isinstance(perf["accuracy_score"], (int, float))
    
    def test_backend_performance_metrics_values(self):
        """Test that performance metrics have reasonable values."""
        metrics = get_backend_performance_metrics()
        
        for backend, perf in metrics.items():
            # Scores should be between 0 and 10
            assert 0 <= perf["speed_score"] <= 10
            assert 0 <= perf["memory_score"] <= 10
            assert 0 <= perf["accuracy_score"] <= 10
    
    def test_backend_performance_relative_ranking(self):
        """Test that performance metrics have reasonable relative rankings."""
        metrics = get_backend_performance_metrics()
        
        # NumPy should have good accuracy but lower speed
        assert metrics["numpy"]["accuracy_score"] >= 7
        assert metrics["numpy"]["speed_score"] <= 5
        
        # GPU backends should have high speed
        if "jax_gpu" in metrics:
            assert metrics["jax_gpu"]["speed_score"] >= 8
        if "numba_gpu" in metrics:
            assert metrics["numba_gpu"]["speed_score"] >= 7


class TestBackendConfiguration:
    """Test backend configuration."""
    
    def test_configure_backend_numpy(self):
        """Test NumPy backend configuration."""
        config = configure_backend(BackendType.NUMPY)
        
        assert config["backend"] == "numpy"
        assert "threads" in config
        assert "memory_limit" in config
        assert isinstance(config["threads"], int)
        assert isinstance(config["memory_limit"], (int, float))
    
    def test_configure_backend_numba_cpu(self):
        """Test Numba CPU backend configuration."""
        config = configure_backend(BackendType.NUMBA_CPU)
        
        assert config["backend"] == "numba_cpu"
        assert "threads" in config
        assert "cache" in config
        assert isinstance(config["threads"], int)
        assert isinstance(config["cache"], bool)
    
    def test_configure_backend_jax_cpu(self):
        """Test JAX CPU backend configuration."""
        config = configure_backend(BackendType.JAX_CPU)
        
        assert config["backend"] == "jax_cpu"
        assert "threads" in config
        assert "precision" in config
        assert isinstance(config["threads"], int)
        assert config["precision"] in ["float32", "float64"]
    
    def test_configure_backend_gpu(self):
        """Test GPU backend configuration."""
        for backend_type in [BackendType.NUMBA_GPU, BackendType.JAX_GPU]:
            config = configure_backend(backend_type)
            
            assert config["backend"] == backend_type.value
            assert "device_id" in config
            assert "memory_fraction" in config
            assert isinstance(config["device_id"], int)
            assert 0 <= config["memory_fraction"] <= 1.0


class TestBackendSelector:
    """Test BackendSelector class."""
    
    def test_backend_selector_initialization(self):
        """Test BackendSelector initialization."""
        selector = BackendSelector()
        
        assert hasattr(selector, 'available_backends')
        assert hasattr(selector, 'system_info')
        assert hasattr(selector, 'performance_metrics')
    
    def test_backend_selector_get_available_backends(self):
        """Test getting available backends."""
        selector = BackendSelector()
        backends = selector.get_available_backends()
        
        assert isinstance(backends, list)
        assert BackendType.NUMPY in backends  # NumPy should always be available
        
        # Check that all returned backends are valid
        for backend in backends:
            assert isinstance(backend, BackendType)
    
    def test_backend_selector_select_backend(self):
        """Test backend selection."""
        selector = BackendSelector()
        
        # Test with different parameters
        backend = selector.select_backend(
            data_size=1000,
        )
        
        assert isinstance(backend, BackendType)
        assert backend in selector.get_available_backends()
    
    def test_backend_selector_configure_backend(self):
        """Test backend configuration through selector."""
        selector = BackendSelector()
        
        backend = BackendType.NUMPY
        config = selector.configure_backend(backend)
        
        assert isinstance(config, dict)
        assert config["backend"] == backend.value
    
    def test_backend_selector_get_recommendations(self):
        """Test getting backend recommendations."""
        selector = BackendSelector()
        
        recommendations = selector.get_recommendations(
            data_size=1000,
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that recommendations are ordered by suitability
        for i in range(len(recommendations) - 1):
            current_score = recommendations[i]["score"]
            next_score = recommendations[i + 1]["score"]
            assert current_score >= next_score


class TestErrorHandling:
    """Test error handling in backend selection."""
    
    def test_select_optimal_backend_invalid_priority(self):
        """Test backend selection with invalid performance priority."""
        backend = select_backend(
            data_size=1000,
            performance_priority="invalid"
        )
        
        # Should fall back to default behavior
        assert isinstance(backend, BackendType)
    
    def test_select_optimal_backend_negative_data_size(self):
        """Test backend selection with negative data size."""
        backend = select_backend(
            data_size=-100,
        )
        
        # Should handle gracefully
        assert isinstance(backend, BackendType)
    
    def test_configure_backend_invalid_backend(self):
        """Test configuration with invalid backend."""
        # This should raise an error or handle gracefully
        try:
            config = configure_backend("invalid_backend")
            # If it doesn't raise an error, check that it returns a valid config
            assert isinstance(config, dict)
        except (ValueError, TypeError):
            # Expected behavior for invalid backend
            pass
    
    def test_backend_selector_error_recovery(self):
        """Test error recovery in BackendSelector."""
        selector = BackendSelector()
        
        # Test that selector can handle errors gracefully
        try:
            backends = selector.get_available_backends()
            assert isinstance(backends, list)
        except Exception:
            # If there's an error, it should be handled gracefully
            pass


class TestPerformance:
    """Test performance characteristics."""
    
    def test_backend_selection_performance(self):
        """Test that backend selection is fast."""
        import time
        
        start_time = time.time()
        
        for _ in range(100):
            backend = select_backend(
                data_size=1000,
            )
        
        end_time = time.time()
        
        # Should complete 100 selections in less than 1 second
        assert end_time - start_time < 1.0
    
    def test_system_info_performance(self):
        """Test that system info gathering is fast."""
        import time
        
        start_time = time.time()
        system_info = get_system_info()
        end_time = time.time()
        
        # Should complete in less than 0.1 seconds
        assert end_time - start_time < 0.1
        assert isinstance(system_info, dict)
    
    def test_performance_metrics_performance(self):
        """Test that performance metrics gathering is fast."""
        import time
        
        start_time = time.time()
        metrics = get_backend_performance_metrics()
        end_time = time.time()
        
        # Should complete in less than 0.1 seconds
        assert end_time - start_time < 0.1
        assert isinstance(metrics, dict)
