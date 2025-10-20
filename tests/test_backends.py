from neurological_lrd_analysis.benchmark_backends.selector import select_backend


def test_backend_selection_cpu():
    # On CPU-only environments, should return numpy, numba_cpu, or jax_cpu
    b = select_backend(1000, real_time=False)
    assert b in {"numpy", "numba_cpu", "jax_cpu", "numba_gpu", "jax_gpu"}


