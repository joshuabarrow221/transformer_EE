"""Optional cuML FFT adapter; importing this module never initializes CUDA.

CPU PCA/SV scaling remain shared with the reference pipeline. Only the reduced
float32 matrix enters CUDA. No automatic fallback or GPU package installation
is allowed: a requested GPU fit must actually use the selected CUDA device.
"""
import numpy as np


def runtime(device=0):
    """Fail early on missing dependencies/driver and record the actual device."""
    try:
        import cupy as cp
        import cuml
        if device < 0 or device >= cp.cuda.runtime.getDeviceCount():
            raise ValueError(f'CUDA device index {device} is unavailable')
        with cp.cuda.Device(device):
            # A real allocation catches driver/runtime mismatches before PCA.
            cp.zeros(1, dtype=cp.float32)
            cp.cuda.get_current_stream().synchronize()
            props = cp.cuda.runtime.getDeviceProperties(device)
        name = props['name']
        return cp, cuml, dict(cuml=cuml.__version__, cupy=cp.__version__,
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(), device_index=device,
            device_name=name.decode() if isinstance(name, bytes) else str(name),
            device_total_bytes=int(props['totalGlobalMem']))
    except Exception as exc:
        raise RuntimeError('cuML GPU backend unavailable. Install compatible cuML/CuPy '
                           'and NVIDIA drivers; see GPU.md. No CPU fallback was used.') from exc


def parameters(n, seed, perplexity, iterations, learning_rate=200.0):
    """Explicit cuML settings, rather than input-size-dependent adaptive tuning.

    cuML's learning-rate convention/schedule differs from openTSNE's auto mode;
    this is a separately audited backend, not a bitwise replacement. A fixed,
    configurable cuML rate is preferable to claiming identical optimization.
    Reject unsupported neighbor counts rather than accepting cuML's clipping.
    """
    k = min(n - 1, int(3 * perplexity))
    if not 0 < perplexity < n or iterations < 300 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid cuML perplexity, iterations, or learning rate')
    if k < 1 or k > 1023:
        raise ValueError('cuML requires 1 <= min(n-1, int(3*perplexity)) <= 1023')
    return dict(n_components=2, perplexity=float(perplexity), method='fft',
        init='pca', random_state=seed, max_iter=iterations, n_neighbors=k,
        early_exaggeration=12.0, late_exaggeration=1.0, exaggeration_iter=250,
        learning_rate=float(learning_rate), learning_rate_method='none',
        pre_momentum=0.5, post_momentum=0.8, min_grad_norm=0.0,
        metric='euclidean', square_distances=True, output_type='numpy')


def fit(work, seed, perplexity, iterations, device=0, learning_rate=200.0):
    """Run GPU neighbors/optimization and return ordinary portable NumPy data.

    The surrounding plot runner is a subprocess: its exit releases CUDA
    allocations. Grids within that process may reuse allocator reservations. Do not globally clear CuPy's allocator, which
    could belong to another caller. GPU exceptions propagate without retrying
    on CPU or changing the sample size.
    """
    cp, cuml, environment = runtime(device)
    settings = parameters(len(work), seed, perplexity, iterations, learning_rate)
    with cp.cuda.Device(device):
        from cuml.manifold import TSNE
        matrix = cp.asarray(work, dtype=cp.float32, order='C')
        estimator = TSNE(**settings)
        coordinates = np.asarray(estimator.fit_transform(matrix), dtype=np.float32)
        cp.cuda.get_current_stream().synchronize()
        divergence = float(estimator.kl_divergence_)
        actual_iterations = getattr(estimator, 'n_iter_', None)
    info = dict(backend='cuML FFT', backend_version=cuml.__version__,
        gpu_environment=environment, gpu_parameters=settings,
        actual_iterations=int(actual_iterations) if actual_iterations is not None else None,
        neighbors='cuML internal GPU search', knn_k=settings['n_neighbors'],
        learning_rate_convention='cuML fixed input rate; differs from openTSNE auto',
        early_exaggeration_iterations=250)
    return coordinates, divergence, info
