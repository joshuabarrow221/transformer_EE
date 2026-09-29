# A100 host validation, 2026-09-28/29 UTC

The merged `dev/tsne-cuda` code passed synthetic CUDA validation on an NVIDIA
A100 80GB PCIe exposed as a **MIG 4g.40gb** device (42,412,802,048 usable device
bytes), driver 595.71.05. This allocation has about 40 GB, not the full 80 GB.

## Results

- Plotting suite with `TRANSFORMEREE_TEST_GPU=1`: 21 passed, one expected skip
  (the test requiring a CPU-only host). The three local workflow regression
  tests in the repository's `tests/` directory also passed.
- Real cuML FFT fit: 1,500 synthetic events, 64 input features reduced to 50
  by CPU PCA, perplexity 30, all 1,000 requested iterations completed. All
  coordinates finite; bounded 1,000-event 10-neighbor trustworthiness 0.96213,
  KL divergence 0.55849. Observed PCA time 0.090 s and fit/diagnostic time
  0.856 s in the isolated test; these are not production-scale benchmarks.
- End-to-end synthetic checkpoint/CSV run: 48 selected events, CUDA inference,
  cuML FFT t-SNE, coordinate export, PNG/PDF grids, cache reuse, and
  coordinate-only PNG/PDF redraw all passed. This tiny model has 24 latent
  dimensions and does not need dimensionality reduction; the 1,500-event test
  above exercises PCA explicitly.
- CPU/GPU maximum absolute differences: head-input features 1.91e-6,
  penultimate activations 3.58e-7, predictions 1.19e-7. All pass
  `rtol=2e-4, atol=1e-5`. TF32 is disabled by the GPU inference runner.

PCA and Matplotlib rendering run on CPU; model inference and cuML t-SNE run
on CUDA. No scientific t-SNE code changes were needed for these host checks.
Real supplied checkpoints/data and full-study memory/performance remain to
be checked when the sample/model tarball arrives.

## Current host environment

Use the tested launcher from the repository root:

```bash
/tmp/transformer-ee-tsne-env/bin/tsne-python plotting/tSNE/run.py --help
TRANSFORMEREE_TEST_GPU=1 PYTHONPATH=plotting/tSNE \
  /tmp/transformer-ee-tsne-env/bin/tsne-python -m unittest discover \
  -s plotting/tSNE/tests -v
```

The environment uses Python 3.10.20, existing CUDA PyTorch 2.4.0.post301,
cuML 26.2.0, CuPy 13.6.0, NumPy 1.26.4, and openTSNE 1.0.4. It inherits the
host's Python packages but installs RAPIDS and its CUDA libraries separately.
The inherited Conda CuPy distribution metadata causes a duplicate-package
warning; the imported CuPy implementation is the environment's 13.6.0 wheel.
The base training environment was not modified.

The environment occupies about 5.6 GB and is under **`/tmp`**, because the home
volume had only 4.3 GB free. It can disappear on host/container restart. Move
future production environments and large tarballs/runs to a persistent volume
with adequate free space. Local logs, plots, metrics, and the full package
inventory are under `plotting/tSNE/runs/host-validation-20260928/` (ignored by
Git). Its synthetic checkpoint is randomly initialized and is not a physics
model.

## Recreating this host setup

These commands assume the existing `/opt/conda` installation and are specific
to this host. For a different host use the [official RAPIDS installation
instructions](https://docs.rapids.ai/install/) and a matching PyTorch build.
Choose an environment location with at least 6 GB free for installed packages
and additional installation scratch space.

```bash
TSNE_ENV=/tmp/transformer-ee-tsne-env
/opt/conda/bin/python -m venv --system-site-packages "$TSNE_ENV"
"$TSNE_ENV/bin/python" -m pip install --no-cache-dir \
  'cuml-cu12==26.2.0' 'cupy-cuda12x==13.6.0' 'numpy<2' \
  -r plotting/tSNE/requirements.txt

"$TSNE_ENV/bin/python" - "$TSNE_ENV" <<'PY'
from pathlib import Path
import shlex
import sys
root = Path(sys.argv[1]).resolve()
site = root / 'lib/python3.10/site-packages'
paths = sorted(str(p) for p in (site / 'nvidia').glob('*/lib'))
paths += sorted(str(p) for p in site.glob('*/lib64'))
paths += ['/opt/conda/lib']
cublas = site / 'nvidia/cublas/lib'
preload = ':'.join(['/opt/conda/lib/libstdc++.so.6',
                    str(cublas / 'libcublasLt.so.12'),
                    str(cublas / 'libcublas.so.12')])
launcher = root / 'bin/tsne-python'
launcher.write_text(
    '#!/usr/bin/env bash\n'
    'export LD_LIBRARY_PATH=' + shlex.quote(':'.join(paths))
    + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n'
    'export LD_PRELOAD=' + shlex.quote(preload)
    + '${LD_PRELOAD:+:$LD_PRELOAD}\n'
    'exec ' + shlex.quote(str(root / 'bin/python')) + ' "$@"\n'
)
launcher.chmod(0o755)
PY
```

The launcher selects the Conda C++ runtime and newer wheel cuBLAS libraries
before loading PyTorch. Without it, this host can load an older system C++
runtime or older Conda cuBLAS first, producing `CXXABI_1.3.15` or
`cublasSetEnvironmentMode` errors. This process-local setup changes no driver,
system CUDA installation, or global shell configuration.
