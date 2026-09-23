"""Tests for snapshot_imager.coordinates."""

import numpy as np
import pytest

from snapshot_imager import (
    compute_baseline_extent,
    compute_image_grid,
    phase_track_to_source,
)


class TestPhaseTracking:
    @pytest.fixture
    def phased(self, imaging_data, telescope_location):
        return phase_track_to_source(
            vis=imaging_data.vis,
            uvw=imaging_data.uvw,
            times=imaging_data.times,
            ra_src=299.868,
            dec_src=40.734,
            telescope_loc=telescope_location,
        )

    def test_output_shape_preserved(self, imaging_data, phased):
        assert phased.shape == imaging_data.vis.shape

    def test_phase_rotation_modifies_data(self, imaging_data, phased):
        assert not np.allclose(phased, imaging_data.vis)

    def test_amplitudes_preserved(self, imaging_data, phased):
        """Phase rotation should not change visibility amplitudes."""
        np.testing.assert_allclose(np.abs(phased), np.abs(imaging_data.vis), rtol=1e-10)

    def test_accepts_astropy_time(self, imaging_data, telescope_location, phased):
        from astropy.time import Time

        phased_time = phase_track_to_source(
            vis=imaging_data.vis,
            uvw=imaging_data.uvw,
            times=Time(imaging_data.times, format="jd"),
            ra_src=299.868,
            dec_src=40.734,
            telescope_loc=telescope_location,
        )
        np.testing.assert_allclose(phased_time, phased)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Sign mismatch: phase_track_to_source applies exp(-2*pi*i*(u*l+v*m+w*n)) "
            "while the imagers use the exp(+2*pi*i*(u*l+v*m)) kernel, so tracking "
            "moves the source to twice its offset instead of the image center."
        ),
    )
    def test_tracked_source_moves_to_image_center(self, telescope_location):
        import astropy.units as u
        from astropy.coordinates import AltAz, SkyCoord
        from astropy.time import Time

        from snapshot_imager import ImagingData, snapshot_imager_type1

        times = np.array([2459000.3])
        frame = AltAz(obstime=Time(times, format="jd"), location=telescope_location)
        zenith = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame[0]).icrs
        ra, dec = zenith.ra.deg + 3.0, zenith.dec.deg + 2.0

        src = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).transform_to(frame)
        l0 = (np.cos(src.alt.rad) * np.sin(src.az.rad)).item()
        m0 = (np.cos(src.alt.rad) * np.cos(src.az.rad)).item()

        rng = np.random.default_rng(3)
        bl = rng.uniform(-60, 60, (40, 3))
        bl[:, 2] = 0.0
        uvw = np.concatenate([bl, -bl])[:, :, None]
        vis = np.exp(-2j * np.pi * (uvw[:, 0] * l0 + uvw[:, 1] * m0))[:, None, :]
        weights = np.ones(vis.shape)
        freqs = np.array([150e6])

        tracked = phase_track_to_source(vis, uvw, times, ra, dec, telescope_location)
        result = snapshot_imager_type1(
            ImagingData(tracked, weights, uvw, times, freqs),
            npix=64,
            fov=30.0,
            verbose=False,
        )
        image = result.images[0, 0].real
        m_idx, l_idx = np.unravel_index(np.argmax(image), image.shape)
        assert (m_idx, l_idx) == (32, 32)


class TestImageGrid:
    def test_coordinate_array_lengths(self):
        lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix=64, fov=20.0)
        assert len(lcoords) == 64
        assert len(mcoords) == 64

    def test_grid_shapes(self):
        lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix=64, fov=20.0)
        assert lgrid.shape == (64, 64)
        assert mgrid.shape == (64, 64)

    @pytest.mark.parametrize("flat_projection", [True, False])
    def test_coords_within_fov(self, flat_projection):
        npix, fov = 64, 20.0
        lcoords, mcoords, _, _ = compute_image_grid(
            npix=npix, fov=fov, flat_projection=flat_projection
        )
        extent = np.sin(np.deg2rad(fov / 2))
        assert np.all(np.abs(lcoords) <= extent)
        assert np.all(np.abs(mcoords) <= extent)

    def test_center_pixel_is_origin(self):
        npix = 64
        lcoords, mcoords, _, _ = compute_image_grid(npix=npix, fov=20.0)
        assert lcoords[npix // 2] == pytest.approx(0.0, abs=1e-15)
        assert mcoords[npix // 2] == pytest.approx(0.0, abs=1e-15)

    def test_grid_orientation(self):
        """Rows index m and columns index l."""
        lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix=16, fov=20.0)
        np.testing.assert_array_equal(lgrid[3], lcoords)
        np.testing.assert_array_equal(mgrid[:, 3], mcoords)


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
