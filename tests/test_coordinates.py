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

    def test_tracked_source_moves_to_image_center(self, telescope_location):
        """
        A source simulated with the pyuvdata/pyuvsim convention images at its
        true (l, m) position, and phase tracking moves it to the image center.
        """
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
        # pyuvsim: V = exp(+2πi uvw·lmn) with uvw = enu(ant2) - enu(ant1)
        vis = np.exp(2j * np.pi * (uvw[:, 0] * l0 + uvw[:, 1] * m0))[:, None, :]
        weights = np.ones(vis.shape)
        freqs = np.array([150e6])

        def peak_lm(v):
            result = snapshot_imager_type1(
                ImagingData(v, weights, uvw, times, freqs),
                npix=64,
                fov=30.0,
                verbose=False,
            )
            image = result.images[0, 0].real
            m_idx, l_idx = np.unravel_index(np.argmax(image), image.shape)
            return result.l_coords[l_idx], result.m_coords[m_idx]

        # Before tracking: the source is at its true position (to within a pixel)
        pixel = 2 * np.sin(np.deg2rad(15.0)) / 64
        l_peak, m_peak = peak_lm(vis)
        assert abs(l_peak - l0) <= pixel and abs(m_peak - m0) <= pixel

        # After tracking: the source is at the image center
        tracked = phase_track_to_source(vis, uvw, times, ra, dec, telescope_location)
        assert peak_lm(tracked) == (0.0, 0.0)


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

    @pytest.mark.parametrize("flat_projection", [True, False])
    @pytest.mark.parametrize("npix", [64, 65])
    def test_center_pixel_is_origin(self, npix, flat_projection):
        lcoords, mcoords, _, _ = compute_image_grid(
            npix=npix, fov=20.0, flat_projection=flat_projection
        )
        assert lcoords[npix // 2] == 0.0
        assert mcoords[npix // 2] == 0.0

    @pytest.mark.parametrize("npix", [64, 65])
    def test_uniform_pixel_spacing(self, npix):
        fov = 20.0
        lcoords, _, _, _ = compute_image_grid(npix=npix, fov=fov)
        spacing = 2 * np.sin(np.deg2rad(fov / 2)) / npix
        np.testing.assert_allclose(np.diff(lcoords), spacing)

    def test_odd_npix_grid_is_symmetric(self):
        lcoords, _, _, _ = compute_image_grid(npix=65, fov=20.0)
        np.testing.assert_allclose(lcoords, -lcoords[::-1], atol=1e-15)

    def test_even_npix_grid_is_unchanged(self):
        """Even-npix grids match the previous linspace definition."""
        npix, fov = 64, 20.0
        extent = np.sin(np.deg2rad(fov / 2))
        lcoords, _, _, _ = compute_image_grid(npix=npix, fov=fov)
        np.testing.assert_allclose(
            lcoords, np.linspace(-extent, extent, npix, endpoint=False), atol=1e-15
        )

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


class TestRadecToLmn:
    @pytest.fixture
    def time(self):
        from astropy.time import Time

        return Time(2459000.3, format="jd")

    def _radec_of(self, alt, az, time, location):
        import astropy.units as u
        from astropy.coordinates import AltAz, SkyCoord

        frame = AltAz(obstime=time, location=location)
        icrs = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame=frame).icrs
        return icrs.ra.deg, icrs.dec.deg

    @pytest.mark.parametrize(
        "alt, az, expected",
        [
            (90.0, 0.0, (0.0, 0.0, 1.0)),  # zenith
            (60.0, 90.0, (0.5, 0.0, np.sqrt(3) / 2)),  # East
            (30.0, 0.0, (0.0, np.sqrt(3) / 2, 0.5)),  # North
            (-20.0, 180.0, (0.0, -np.cos(np.deg2rad(20)), -np.sin(np.deg2rad(20)))),
        ],
    )
    def test_known_directions(self, telescope_location, time, alt, az, expected):
        from snapshot_imager import radec_to_lmn

        ra, dec = self._radec_of(alt, az, time, telescope_location)
        l, m, n = radec_to_lmn(ra, dec, time, telescope_location)
        np.testing.assert_allclose([l.item(), m.item(), n.item()], expected, atol=1e-8)

    def test_shapes_and_normalization(self, telescope_location):
        from snapshot_imager import radec_to_lmn

        times = np.linspace(2459000.2, 2459000.3, 5)
        l, m, n = radec_to_lmn([10.0, 20.0, 30.0], [-30.0, -20.0, -10.0], times, telescope_location)
        assert l.shape == m.shape == n.shape == (5, 3)
        np.testing.assert_allclose(l**2 + m**2 + n**2, 1.0)

    def test_mismatched_ra_dec(self, telescope_location):
        from snapshot_imager import radec_to_lmn

        with pytest.raises(ValueError, match="same shape"):
            radec_to_lmn([10.0, 20.0], [10.0], 2459000.3, telescope_location)


def test_phase_tracking_below_horizon_uses_negative_n(imaging_data, telescope_location):
    """n is the sine of the elevation, including for sources below the horizon."""
    from snapshot_imager import radec_to_lmn

    d = imaging_data
    # Cygnus A is below HERA's horizon at these times
    l, m, n = radec_to_lmn(299.868, 40.734, d.times, telescope_location)
    assert np.all(n < 0)
    tracked = phase_track_to_source(
        d.vis, d.uvw, d.times, 299.868, 40.734, telescope_location
    )
    phase = np.exp(
        -2j
        * np.pi
        * (
            d.uvw[:, None, 0, :] * l[None, :, 0, None]
            + d.uvw[:, None, 1, :] * m[None, :, 0, None]
            + d.uvw[:, None, 2, :] * n[None, :, 0, None]
        )
    )
    np.testing.assert_allclose(tracked, d.vis * phase)
