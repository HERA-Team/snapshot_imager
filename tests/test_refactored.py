"""
Tests for the snapshot_imager package.

Run with: pytest tests/
"""

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
import astropy.units as u

from snapshot_imager import (
    ImagingData,
    ImageResult,
    phase_track_to_source,
    snapshot_imager_type1,
    snapshot_imager_type3,
    snapshot_imager_mfs_type_1,
    snapshot_imager_mfs_type_3,
    compute_image_grid,
    compute_baseline_extent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def imaging_data():
    """Standard-sized ImagingData for most tests."""
    nbls, ntimes, nfreqs = 20, 5, 8
    rng = np.random.default_rng(42)
    vis = rng.standard_normal((nbls, ntimes, nfreqs)) + 1j * rng.standard_normal((nbls, ntimes, nfreqs))
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = rng.standard_normal((nbls, 3, nfreqs)) * 50
    times = np.linspace(2459000, 2459000.05, ntimes)
    freqs = np.linspace(100e6, 150e6, nfreqs)
    return ImagingData(vis, weights, uvw, times, freqs)


@pytest.fixture
def imaging_data_small():
    """Small ImagingData for slower MFS tests."""
    nbls, ntimes, nfreqs = 10, 2, 4
    rng = np.random.default_rng(42)
    vis = rng.standard_normal((nbls, ntimes, nfreqs)) + 1j * rng.standard_normal((nbls, ntimes, nfreqs))
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = rng.standard_normal((nbls, 3, nfreqs)) * 50
    times = np.linspace(2459000, 2459000.05, ntimes)
    freqs = np.linspace(100e6, 150e6, nfreqs)
    return ImagingData(vis, weights, uvw, times, freqs)


@pytest.fixture
def telescope_location():
    """HERA telescope location."""
    return EarthLocation(lat=-30.7215 * u.deg, lon=21.4283 * u.deg, height=1051 * u.m)


# ---------------------------------------------------------------------------
# ImagingData
# ---------------------------------------------------------------------------

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

    @pytest.mark.parametrize("bad_weights_shape", [
        (20, 5, 9),   # wrong nfreqs
        (20, 6, 8),   # wrong ntimes
        (21, 5, 8),   # wrong nbls
    ])
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


# ---------------------------------------------------------------------------
# Imagers
# ---------------------------------------------------------------------------

class TestType1Imager:

    def test_output_shape(self, imaging_data):
        result = snapshot_imager_type1(imaging_data, npix=32, fov=10.0, verbose=False)
        assert result.shape == (5, 8, 32, 32)

    def test_result_metadata(self, imaging_data):
        result = snapshot_imager_type1(imaging_data, npix=32, fov=10.0, verbose=False)
        assert result.npix == 32
        assert result.fov == 10.0
        assert len(result.l_coords) == 32
        assert len(result.m_coords) == 32

    def test_returns_image_result(self, imaging_data):
        result = snapshot_imager_type1(imaging_data, npix=32, fov=10.0, verbose=False)
        assert isinstance(result, ImageResult)

    def test_images_are_finite(self, imaging_data):
        result = snapshot_imager_type1(imaging_data, npix=32, fov=10.0, verbose=False)
        assert np.all(np.isfinite(result.images))


class TestType3Imager:

    def test_output_shape(self, imaging_data):
        result = snapshot_imager_type3(imaging_data, npix=32, fov=10.0, verbose=False)
        assert result.shape == (5, 8, 32, 32)

    def test_returns_image_result(self, imaging_data):
        result = snapshot_imager_type3(imaging_data, npix=32, fov=10.0, verbose=False)
        assert isinstance(result, ImageResult)

    def test_images_are_finite(self, imaging_data):
        result = snapshot_imager_type3(imaging_data, npix=32, fov=10.0, verbose=False)
        assert np.all(np.isfinite(result.images))


class TestMFSType1Imager:

    def test_output_shape(self, imaging_data_small):
        # MFS collapses all frequencies into a single wideband image per time step
        result = snapshot_imager_mfs_type_1(imaging_data_small, npix=16, fov=10.0, verbose=False)
        assert result.shape == (2, 1, 16, 16)

    def test_returns_image_result(self, imaging_data_small):
        result = snapshot_imager_mfs_type_1(imaging_data_small, npix=16, fov=10.0, verbose=False)
        assert isinstance(result, ImageResult)


class TestMFSType3Imager:

    def test_output_shape(self, imaging_data_small):
        # MFS collapses all frequencies into a single wideband image per time step
        result = snapshot_imager_mfs_type_3(imaging_data_small, npix=16, fov=10.0, verbose=False)
        assert result.shape == (2, 1, 16, 16)

    def test_returns_image_result(self, imaging_data_small):
        result = snapshot_imager_mfs_type_3(imaging_data_small, npix=16, fov=10.0, verbose=False)
        assert isinstance(result, ImageResult)


# ---------------------------------------------------------------------------
# Phase tracking
# ---------------------------------------------------------------------------

class TestPhaseTracking:

    def test_output_shape_preserved(self, imaging_data, telescope_location):
        vis_phased = phase_track_to_source(
            vis=imaging_data.vis,
            uvw=imaging_data.uvw,
            times=imaging_data.times,
            ra_src=299.868,
            dec_src=40.734,
            telescope_loc=telescope_location,
        )
        assert vis_phased.shape == imaging_data.vis.shape

    def test_phase_rotation_modifies_data(self, imaging_data, telescope_location):
        vis_phased = phase_track_to_source(
            vis=imaging_data.vis,
            uvw=imaging_data.uvw,
            times=imaging_data.times,
            ra_src=299.868,
            dec_src=40.734,
            telescope_loc=telescope_location,
        )
        assert not np.allclose(vis_phased, imaging_data.vis)

    def test_amplitudes_preserved(self, imaging_data, telescope_location):
        """Phase rotation should not change visibility amplitudes."""
        vis_phased = phase_track_to_source(
            vis=imaging_data.vis,
            uvw=imaging_data.uvw,
            times=imaging_data.times,
            ra_src=299.868,
            dec_src=40.734,
            telescope_loc=telescope_location,
        )
        np.testing.assert_allclose(
            np.abs(vis_phased), np.abs(imaging_data.vis), rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestImageGrid:

    def test_coordinate_array_lengths(self):
        lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix=64, fov=20.0)
        assert len(lcoords) == 64
        assert len(mcoords) == 64

    def test_grid_shapes(self):
        lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix=64, fov=20.0)
        assert lgrid.shape == (64, 64)
        assert mgrid.shape == (64, 64)

    def test_coords_within_fov(self):
        npix, fov = 64, 20.0
        lcoords, mcoords, _, _ = compute_image_grid(npix=npix, fov=fov)
        extent = np.sin(np.deg2rad(fov / 2))
        assert np.all(np.abs(lcoords) <= extent)
        assert np.all(np.abs(mcoords) <= extent)


class TestBaselineExtent:

    def test_positive_result(self):
        rng = np.random.default_rng(0)
        u = rng.standard_normal((100, 10)) * 50
        v = rng.standard_normal((100, 10)) * 30
        assert compute_baseline_extent(u, v) > 0

    def test_dominated_by_largest_axis(self):
        rng = np.random.default_rng(0)
        u = rng.standard_normal((100, 10)) * 50
        v = rng.standard_normal((100, 10)) * 30
        umax = compute_baseline_extent(u, v)
        assert umax >= np.max(np.abs(u))
        assert umax >= np.max(np.abs(v))
