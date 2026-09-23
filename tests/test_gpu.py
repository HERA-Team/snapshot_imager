"""
GPU (CuPy + cuFINUFFT) tests.

These are skipped automatically unless CuPy and cuFINUFFT are installed and a
CUDA device is available, so they do not run in CI. Run them on a GPU node
with ``pytest -m gpu``.
"""

import numpy as np
import pytest

from snapshot_imager import (
    snapshot_imager_mfs_type_1,
    snapshot_imager_mfs_type_3,
    snapshot_imager_type1,
    snapshot_imager_type3,
)

cp = pytest.importorskip("cupy")
pytest.importorskip("cufinufft")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - depends on the CUDA runtime
    _HAS_DEVICE = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _HAS_DEVICE, reason="no CUDA device available"),
]


@pytest.mark.parametrize(
    "imager",
    [
        snapshot_imager_type1,
        snapshot_imager_type3,
        snapshot_imager_mfs_type_1,
        snapshot_imager_mfs_type_3,
    ],
)
def test_gpu_matches_cpu(imager, imaging_data_small):
    kwargs = {"npix": 16, "fov": 10.0, "verbose": False}
    cpu = imager(imaging_data_small, use_cupy=False, **kwargs)
    gpu = imager(imaging_data_small, use_cupy=True, **kwargs)
    scale = np.abs(cpu.images).max()
    np.testing.assert_allclose(gpu.images, cpu.images, atol=1e-6 * scale)
