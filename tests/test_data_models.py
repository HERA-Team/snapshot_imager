"""Tests for snapshot_imager.data_models."""

import numpy as np
import pytest

from snapshot_imager import ImageResult, ImagingData


class TestImagingData:
    def test_shape_properties(self, imaging_data):
        nbls, ntimes, nfreqs = 20, 5, 8
        assert imaging_data.shape == (nbls, ntimes, nfreqs)
        assert imaging_data.nbls == nbls
        assert imaging_data.ntimes == ntimes
        assert imaging_data.nfreqs == nfreqs

    def test_uvw_coordinate_properties(self, imaging_data):
        nbls, nfreqs = 20, 8
        assert imaging_data.u.shape == (nbls, nfreqs)
        assert imaging_data.v.shape == (nbls, nfreqs)
        assert imaging_data.w.shape == (nbls, nfreqs)

    def test_uvw_properties_index_uvw_axis(self, imaging_data):
        np.testing.assert_array_equal(imaging_data.u, imaging_data.uvw[:, 0, :])
        np.testing.assert_array_equal(imaging_data.v, imaging_data.uvw[:, 1, :])
        np.testing.assert_array_equal(imaging_data.w, imaging_data.uvw[:, 2, :])

    @pytest.mark.parametrize(
        "bad_weights_shape",
        [
            (20, 5, 9),  # wrong nfreqs
            (20, 6, 8),  # wrong ntimes
            (21, 5, 8),  # wrong nbls
        ],
    )
    def test_invalid_weights_shape_raises(self, imaging_data, bad_weights_shape):
        with pytest.raises(ValueError, match="weights shape"):
            ImagingData(
                vis=imaging_data.vis,
                weights=np.ones(bad_weights_shape),
                uvw=imaging_data.uvw,
                times=imaging_data.times,
                freqs=imaging_data.freqs,
            )

    def test_invalid_uvw_shape_raises(self, imaging_data):
        bad_uvw = np.ones((20, 4, 8))  # axis 1 should be 3
        with pytest.raises(ValueError, match="uvw shape"):
            ImagingData(
                vis=imaging_data.vis,
                weights=imaging_data.weights,
                uvw=bad_uvw,
                times=imaging_data.times,
                freqs=imaging_data.freqs,
            )

    def test_invalid_times_length_raises(self, imaging_data):
        with pytest.raises(ValueError, match="times length"):
            ImagingData(
                vis=imaging_data.vis,
                weights=imaging_data.weights,
                uvw=imaging_data.uvw,
                times=np.linspace(2459000, 2459000.05, 99),
                freqs=imaging_data.freqs,
            )

    def test_invalid_freqs_length_raises(self, imaging_data):
        with pytest.raises(ValueError, match="freqs length"):
            ImagingData(
                vis=imaging_data.vis,
                weights=imaging_data.weights,
                uvw=imaging_data.uvw,
                times=imaging_data.times,
                freqs=np.linspace(100e6, 150e6, 99),
            )


class TestImageResult:
    def test_shape_properties(self):
        images = np.zeros((3, 4, 16, 16))
        result = ImageResult(
            images=images,
            l_coords=np.zeros(16),
            m_coords=np.zeros(16),
            fov=10.0,
            npix=16,
        )
        assert result.shape == (3, 4, 16, 16)
        assert result.ntimes == 3
        assert result.nfreqs == 4
