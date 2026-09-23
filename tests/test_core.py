"""Tests for snapshot_imager.core utilities."""

import sys
import types

import finufft
import numpy as np
import pytest

from snapshot_imager import estimate_memory_requirements, get_nufft_library
from snapshot_imager.core import (
    prepare_weighted_visibilities,
    validate_imaging_inputs,
)


class TestGetNufftLibrary:
    def test_cpu(self):
        xp, nufft_lib, use_gpu = get_nufft_library(use_cupy=False)
        assert xp is np
        assert nufft_lib is finufft
        assert use_gpu is False

    def test_falls_back_to_cpu_without_cupy(self, monkeypatch):
        # A None entry in sys.modules makes `import cupy` raise ImportError
        monkeypatch.setitem(sys.modules, "cupy", None)
        with pytest.warns(RuntimeWarning, match="falling back to CPU"):
            xp, nufft_lib, use_gpu = get_nufft_library(use_cupy=True)
        assert xp is np
        assert nufft_lib is finufft
        assert use_gpu is False

    def test_falls_back_to_cpu_without_gpu_device(self, monkeypatch):
        """CuPy and cuFINUFFT installed, but no CUDA device available."""
        fake_cupy = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                runtime=types.SimpleNamespace(getDeviceCount=lambda: 0)
            )
        )
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
        monkeypatch.setitem(sys.modules, "cufinufft", types.SimpleNamespace())
        with pytest.warns(RuntimeWarning, match="no CUDA device"):
            xp, nufft_lib, use_gpu = get_nufft_library(use_cupy=True)
        assert xp is np
        assert nufft_lib is finufft
        assert use_gpu is False

    def test_uses_gpu_when_available(self, monkeypatch):
        fake_cupy = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
            )
        )
        fake_cufinufft = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
        monkeypatch.setitem(sys.modules, "cufinufft", fake_cufinufft)
        xp, nufft_lib, use_gpu = get_nufft_library(use_cupy=True)
        assert xp is fake_cupy
        assert nufft_lib is fake_cufinufft
        assert use_gpu is True


class TestPrepareWeightedVisibilities:
    @pytest.fixture
    def arrays(self):
        rng = np.random.default_rng(0)
        vis = rng.standard_normal((6, 4, 5)) + 1j * rng.standard_normal((6, 4, 5))
        weights = rng.uniform(0, 1, (6, 4, 5))
        return vis, weights

    def test_by_frequency(self, arrays):
        vis, weights = arrays
        out = prepare_weighted_visibilities(vis, weights, freq_idx=2)
        assert out.shape == (4, 6)  # (ntimes, nbls)
        np.testing.assert_allclose(out, (vis[:, :, 2] * weights[:, :, 2]).T)

    def test_by_time(self, arrays):
        vis, weights = arrays
        out = prepare_weighted_visibilities(vis, weights, time_idx=1)
        assert out.shape == (5, 6)  # (nfreqs, nbls)
        np.testing.assert_allclose(out, (vis[:, 1, :] * weights[:, 1, :]).T)

    def test_requires_an_index(self, arrays):
        with pytest.raises(ValueError, match="Must specify"):
            prepare_weighted_visibilities(*arrays)

    def test_rejects_both_indices(self, arrays):
        with pytest.raises(ValueError, match="Cannot specify both"):
            prepare_weighted_visibilities(*arrays, freq_idx=0, time_idx=0)


class TestValidateImagingInputs:
    @pytest.fixture
    def inputs(self, imaging_data):
        d = imaging_data
        return {"vis": d.vis, "weights": d.weights, "u": d.u, "v": d.v}

    def test_valid_inputs_pass(self, inputs):
        validate_imaging_inputs(**inputs, npix=16, fov=10.0)

    def test_weights_shape_mismatch(self, inputs):
        inputs["weights"] = inputs["weights"][:-1]
        with pytest.raises(ValueError, match="doesn't match weights shape"):
            validate_imaging_inputs(**inputs, npix=16, fov=10.0)

    @pytest.mark.parametrize("key", ["u", "v"])
    def test_uv_shape_mismatch(self, inputs, key):
        inputs[key] = inputs[key][:-1]
        with pytest.raises(ValueError, match=f"{key} shape"):
            validate_imaging_inputs(**inputs, npix=16, fov=10.0)

    @pytest.mark.parametrize("npix", [0, -4])
    def test_bad_npix(self, inputs, npix):
        with pytest.raises(ValueError, match="npix must be positive"):
            validate_imaging_inputs(**inputs, npix=npix, fov=10.0)

    @pytest.mark.parametrize("fov", [0.0, -1.0, 180.1])
    def test_bad_fov(self, inputs, fov):
        with pytest.raises(ValueError, match="fov must be between"):
            validate_imaging_inputs(**inputs, npix=16, fov=fov)


class TestEstimateMemoryRequirements:
    def test_values(self):
        est = estimate_memory_requirements(nbls=100, ntimes=10, nfreqs=20, npix=64)
        itemsize = np.dtype(np.complex128).itemsize
        assert est["input_data"] == pytest.approx(100 * 10 * 20 * itemsize / 1e9)
        assert est["output_images"] == pytest.approx(10 * 20 * 64 * 64 * itemsize / 1e9)
        assert est["total"] == pytest.approx(
            est["input_data"] + est["output_images"] + est["working_memory"]
        )

    def test_single_precision_is_half(self):
        kwargs = {"nbls": 100, "ntimes": 10, "nfreqs": 20, "npix": 64}
        double = estimate_memory_requirements(**kwargs)
        single = estimate_memory_requirements(**kwargs, dtype=np.complex64)
        assert single["total"] == pytest.approx(double["total"] / 2)
