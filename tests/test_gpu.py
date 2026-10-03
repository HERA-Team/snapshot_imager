"""
GPU (CuPy + cuFINUFFT) tests.

These are skipped automatically unless CuPy and cuFINUFFT are installed and a
CUDA device is available, so they do not run in CI. Run them on a GPU node
with ``pytest -m gpu``.
"""

import numpy as np
import pytest
from helpers import make_hermitian_pair, make_point_source

from snapshot_imager import (
    ImagingData,
    dirty_image,
    dirty_image_points,
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


def _assert_close(gpu, cpu, rtol):
    assert gpu.images.dtype == cpu.images.dtype
    np.testing.assert_array_equal(np.isnan(gpu.images), np.isnan(cpu.images))
    scale = np.nanmax(np.abs(cpu.images))
    np.testing.assert_allclose(
        np.nan_to_num(gpu.images), np.nan_to_num(cpu.images), atol=rtol * scale
    )


@pytest.mark.parametrize(
    "imager",
    [
        snapshot_imager_type1,
        snapshot_imager_type3,
        snapshot_imager_mfs_type_1,
        snapshot_imager_mfs_type_3,
    ],
)
def test_legacy_gpu_matches_cpu(imager, imaging_data_small):
    kwargs = {"npix": 16, "fov": 10.0, "verbose": False}
    cpu = imager(imaging_data_small, use_cupy=False, **kwargs)
    gpu = imager(imaging_data_small, use_cupy=True, **kwargs)
    _assert_close(gpu, cpu, rtol=1e-6)


@pytest.mark.parametrize("method", ["type1", "type3"])
@pytest.mark.parametrize("mfs", [False, True])
@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("hermitian", [True, False])
def test_dirty_image_gpu_matches_cpu(method, mfs, dtype, hermitian):
    ps = make_point_source(
        npix=33, fov=180.0, l_idx=20, m_idx=9, dtype=dtype, hermitian=hermitian
    )
    kwargs = {"npix": 33, "fov": 180.0, "method": method, "mfs": mfs}
    cpu = dirty_image(ps.data, **kwargs)
    gpu = dirty_image(ps.data, use_gpu=True, **kwargs)
    _assert_close(gpu, cpu, rtol=1e-6 if dtype == np.complex128 else 1e-4)


def test_rm_phasor_gpu_matches_cpu(imaging_data_small):
    d = imaging_data_small
    phasor = np.exp(1j * np.linspace(0, np.pi, d.nfreqs))
    cpu = dirty_image(d, 16, 10.0, rm_phasor=phasor)
    gpu = dirty_image(d, 16, 10.0, rm_phasor=phasor, use_gpu=True)
    _assert_close(gpu, cpu, rtol=1e-6)


def test_fully_flagged_snapshot_is_zero_on_gpu(imaging_data_small):
    d = imaging_data_small
    weights = d.weights.copy()
    weights[:, 0, :] = 0.0
    data = ImagingData(d.vis, weights, d.uvw, d.times, d.freqs)
    gpu = dirty_image(data, 16, 10.0, use_gpu=True)
    assert np.all(gpu.images[0] == 0)


@pytest.mark.parametrize("constant_weights", [True, False])
@pytest.mark.parametrize("mfs", [False, True])
def test_psf_gpu_matches_cpu(constant_weights, mfs):
    hermitian, _ = make_hermitian_pair(nbls=15, ntimes=4, nfreqs=5)
    if constant_weights:
        weights = np.repeat(hermitian.weights[:, :1], hermitian.ntimes, axis=1)
        hermitian = ImagingData(
            hermitian.vis, weights, hermitian.uvw, hermitian.times, hermitian.freqs,
            hermitian=True,
        )
    kwargs = {"npix": 33, "fov": 180.0, "mfs": mfs, "return_psf": True}
    cpu = dirty_image(hermitian, **kwargs)
    gpu = dirty_image(hermitian, use_gpu=True, **kwargs)
    _assert_close(gpu, cpu, rtol=1e-6)
    scale = np.nanmax(np.abs(cpu.psf))
    np.testing.assert_allclose(
        np.nan_to_num(gpu.psf), np.nan_to_num(cpu.psf), atol=1e-6 * scale
    )


@pytest.mark.parametrize("method", ["direct", "type3"])
@pytest.mark.parametrize("mfs", [False, True])
@pytest.mark.parametrize("w_term", [None, "unprojected"])
@pytest.mark.parametrize("per_time", [False, True])
def test_points_gpu_matches_cpu(method, mfs, w_term, per_time):
    hermitian, _ = make_hermitian_pair(nbls=15, ntimes=3, nfreqs=4)
    rng = np.random.default_rng(0)
    shape = (hermitian.ntimes, 20) if per_time else (20,)
    l, m = rng.uniform(-0.5, 0.5, shape), rng.uniform(-0.5, 0.5, shape)
    kwargs = {"mfs": mfs, "method": method, "w_term": w_term}
    cpu = dirty_image_points(hermitian, l, m, **kwargs).values
    gpu = dirty_image_points(hermitian, l, m, use_gpu=True, **kwargs).values
    assert gpu.dtype == cpu.dtype
    np.testing.assert_allclose(gpu, cpu, atol=1e-6 * np.abs(cpu).max())
