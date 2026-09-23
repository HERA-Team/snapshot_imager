"""Tests for snapshot_imager.dirty_image_points."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from helpers import make_hermitian_pair

import snapshot_imager.imager as imager_module
from snapshot_imager import (
    ImagingData,
    PointsResult,
    compute_image_grid,
    dirty_image,
    dirty_image_points,
    radec_to_lmn,
)

C = 299792458.0


def _grid_points(npix, fov):
    _, _, lgrid, mgrid = compute_image_grid(npix, fov)
    return lgrid.ravel(), mgrid.ravel()


# ---------------------------------------------------------------------------
# Consistency with dirty_image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["direct", "type3"])
@pytest.mark.parametrize("mfs", [False, True])
@pytest.mark.parametrize("layout", ["hermitian", "explicit"])
@pytest.mark.parametrize("npix", [16, 17])
def test_matches_dirty_image_on_grid(method, mfs, layout, npix):
    hermitian, explicit = make_hermitian_pair(nbls=15, ntimes=3, nfreqs=4)
    data = hermitian if layout == "hermitian" else explicit
    l, m = _grid_points(npix, 60.0)

    result = dirty_image_points(data, l, m, mfs=mfs, method=method, eps=1e-12)
    expected = dirty_image(data, npix, 60.0, mfs=mfs, eps=1e-12).images

    assert isinstance(result, PointsResult)
    assert result.values.dtype == expected.dtype
    ref = expected.reshape(result.values.shape)
    np.testing.assert_allclose(result.values, ref, atol=1e-10 * np.abs(ref).max())


@pytest.mark.parametrize("method", ["direct", "type3"])
@pytest.mark.parametrize("mfs", [False, True])
def test_psf_matches_dirty_image_psf(method, mfs):
    hermitian, _ = make_hermitian_pair(nbls=15, ntimes=3, nfreqs=4)
    l, m = _grid_points(17, 60.0)
    result = dirty_image_points(hermitian, l, m, mfs=mfs, psf=True, method=method, eps=1e-12)
    expected = dirty_image(hermitian, 17, 60.0, mfs=mfs, return_psf=True, eps=1e-12).psf
    ref = expected.reshape(result.values.shape)
    np.testing.assert_allclose(result.values, ref, atol=1e-10 * np.abs(ref).max())


def test_psf_peak_and_offsets_beyond_the_sky():
    hermitian, _ = make_hermitian_pair(nbls=15, ntimes=3, nfreqs=4)
    result = dirty_image_points(hermitian, [0.0, 1.5], [0.0, 0.5], psf=True)
    peak = result.values[:, :, 0]
    has_weight = result.sum_weights > 0
    np.testing.assert_allclose(peak[has_weight], 1.0, rtol=1e-12)
    # Offsets are not directions, so they are never masked
    assert np.all(np.isfinite(result.values))


@pytest.mark.parametrize("mfs", [False, True])
def test_direct_and_type3_agree(mfs):
    hermitian, _ = make_hermitian_pair(nbls=20, ntimes=2, nfreqs=3)
    rng = np.random.default_rng(1)
    r, phi = np.sqrt(rng.uniform(0, 1, 50)), rng.uniform(0, 2 * np.pi, 50)
    l, m = r * np.cos(phi), r * np.sin(phi)
    a = dirty_image_points(hermitian, l, m, mfs=mfs, method="direct").values
    b = dirty_image_points(hermitian, l, m, mfs=mfs, method="type3", eps=1e-12).values
    np.testing.assert_allclose(a, b, atol=1e-10 * np.abs(a).max())


@pytest.mark.parametrize("method", ["direct", "type3"])
@pytest.mark.parametrize("mfs", [False, True])
def test_time_dependent_positions(method, mfs):
    hermitian, _ = make_hermitian_pair(nbls=15, ntimes=3, nfreqs=4)
    rng = np.random.default_rng(2)
    l = rng.uniform(-0.5, 0.5, (3, 7))
    m = rng.uniform(-0.5, 0.5, (3, 7))
    result = dirty_image_points(hermitian, l, m, mfs=mfs, method=method, eps=1e-12)
    for ti in range(3):
        at_t = dirty_image_points(hermitian, l[ti], m[ti], mfs=mfs, method=method, eps=1e-12)
        np.testing.assert_allclose(
            result.values[ti], at_t.values[ti], atol=1e-10 * np.abs(at_t.values).max()
        )


def test_complex64_gives_float32():
    single, _ = make_hermitian_pair(dtype=np.complex64)
    double, _ = make_hermitian_pair()
    l, m = _grid_points(8, 40.0)
    a = dirty_image_points(single, l, m).values
    b = dirty_image_points(double, l, m).values
    assert a.dtype == np.float32
    np.testing.assert_allclose(a, b, atol=1e-4 * np.abs(b).max())


def test_empty_points():
    hermitian, _ = make_hermitian_pair(nbls=10, ntimes=2, nfreqs=3)
    result = dirty_image_points(hermitian, [], [])
    assert result.values.shape == (2, 3, 0)


# ---------------------------------------------------------------------------
# Visibility masking and metadata
# ---------------------------------------------------------------------------


def test_directions_outside_the_sky_are_nan():
    hermitian, _ = make_hermitian_pair(nbls=10, ntimes=2, nfreqs=3)
    result = dirty_image_points(hermitian, [0.9, 0.1, 0.1], [0.9, 0.1, 0.1], n=[0.0, 0.5, -0.5])
    assert np.all(np.isnan(result.values[:, :, 0]))  # l^2 + m^2 > 1
    assert np.all(np.isfinite(result.values[:, :, 1]))
    assert np.all(np.isnan(result.values[:, :, 2]))  # below the horizon


def test_result_metadata():
    hermitian, explicit = make_hermitian_pair(nbls=10, ntimes=2, nfreqs=3)
    result = dirty_image_points(hermitian, [0.1], [0.2])
    np.testing.assert_array_equal(result.times, hermitian.times)
    np.testing.assert_array_equal(result.freqs, hermitian.freqs)
    np.testing.assert_array_equal(result.l, [0.1])
    assert result.npoints == 1
    # Implied conjugates count toward the summed weights
    np.testing.assert_allclose(
        result.sum_weights, dirty_image_points(explicit, [0.1], [0.2]).sum_weights
    )
    mfs = dirty_image_points(hermitian, [0.1], [0.2], mfs=True)
    assert mfs.shape == (2, 1, 1)
    np.testing.assert_allclose(mfs.freqs, [np.mean(hermitian.freqs)])


# ---------------------------------------------------------------------------
# w-term
# ---------------------------------------------------------------------------


def _non_coplanar_source(l0, m0, zenith_phased=False):
    """Unit point source seen by an array with ~0.8 m antenna height scatter."""
    rng = np.random.default_rng(5)
    freqs = np.array([150e6])
    pos = rng.uniform(-80, 80, (12, 3))
    pos[:, 2] = rng.normal(0, 0.8, 12)
    a1, a2 = np.triu_indices(12, 1)
    uvw = ((pos[a2] - pos[a1]) * freqs[0] / C)[:, :, None]
    n0 = np.sqrt(1 - l0**2 - m0**2)
    # pyuvsim convention for unprojected data: V = exp(+2πi (u l + v m + w n))
    vis = np.exp(2j * np.pi * (uvw[:, 0] * l0 + uvw[:, 1] * m0 + uvw[:, 2] * n0))
    if zenith_phased:
        vis = vis * np.exp(-2j * np.pi * uvw[:, 2])
    vis = vis[:, None, :]
    return ImagingData(
        vis, np.ones(vis.shape), uvw, np.array([2459000.0]), freqs, hermitian=True
    )


@pytest.mark.parametrize("method", ["direct", "type3"])
@pytest.mark.parametrize("l0, m0", [(0.55, -0.30), (0.95, 0.25)])
@pytest.mark.parametrize(
    "zenith_phased, w_term", [(False, "unprojected"), (True, "zenith")]
)
def test_w_term_recovers_non_coplanar_source(method, l0, m0, zenith_phased, w_term):
    data = _non_coplanar_source(l0, m0, zenith_phased=zenith_phased)
    with_w = dirty_image_points(data, [l0], [m0], w_term=w_term, method=method, eps=1e-10)
    without_w = dirty_image_points(data, [l0], [m0], method=method, eps=1e-10)
    wrong = "zenith" if w_term == "unprojected" else "unprojected"
    wrong_w = dirty_image_points(data, [l0], [m0], w_term=wrong, method=method, eps=1e-10)
    assert with_w.values.item() == pytest.approx(1.0, abs=1e-8)
    assert without_w.values.item() < 0.9
    assert wrong_w.values.item() < 0.9


def test_w_term_uses_given_n():
    data = _non_coplanar_source(0.55, -0.30)
    n0 = np.sqrt(1 - 0.55**2 - 0.30**2)
    a = dirty_image_points(data, [0.55], [-0.30], w_term="unprojected")
    b = dirty_image_points(data, [0.55], [-0.30], [n0], w_term="unprojected")
    np.testing.assert_allclose(a.values, b.values)
    np.testing.assert_allclose(a.n, [n0])


# ---------------------------------------------------------------------------
# Catalog sources: radec_to_lmn -> dirty_image_points
# ---------------------------------------------------------------------------


def test_tracks_a_drifting_catalog_source(telescope_location):
    """A source simulated at an RA/Dec is recovered at every time as it drifts."""
    times = 2459000.3 + np.arange(4) * 10 / 86400 * 30  # 5-minute steps
    frame = AltAz(obstime=Time(times[0], format="jd"), location=telescope_location)
    zenith = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame).icrs
    ra, dec = zenith.ra.deg + 2.0, zenith.dec.deg - 3.0
    l, m, n = radec_to_lmn(ra, dec, times, telescope_location)

    rng = np.random.default_rng(4)
    pos = rng.uniform(-60, 60, (15, 3))
    pos[:, 2] = 0.0
    a1, a2 = np.triu_indices(15, 1)
    freqs = np.array([120e6, 150e6])
    uvw = (pos[a2] - pos[a1])[:, :, None] * freqs[None, None, :] / C
    vis = np.exp(
        2j * np.pi * (uvw[:, None, 0, :] * l[None] + uvw[:, None, 1, :] * m[None])
    )
    data = ImagingData(vis, np.ones(vis.shape), uvw, times, freqs, hermitian=True)

    result = dirty_image_points(data, l, m, n)
    np.testing.assert_allclose(result.values[:, :, 0], 1.0, atol=1e-12)
    # A fixed direction does not follow the source
    fixed = dirty_image_points(data, l[0], m[0])
    assert fixed.values[-1, :, 0].max() < 0.99


# ---------------------------------------------------------------------------
# Method selection and validation
# ---------------------------------------------------------------------------


def test_auto_method(monkeypatch):
    chosen = []
    original = imager_module.PointTransform

    def spy(*args, **kwargs):
        chosen.append(kwargs["method"])
        return original(*args, **kwargs)

    monkeypatch.setattr(imager_module, "PointTransform", spy)
    hermitian, _ = make_hermitian_pair(nbls=10, ntimes=1, nfreqs=2)
    dirty_image_points(hermitian, [0.1], [0.1])
    monkeypatch.setattr(imager_module, "_DIRECT_MAX_TERMS", 5)
    dirty_image_points(hermitian, [0.1], [0.1])
    assert chosen == ["direct", "type3"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"l": [0.1, 0.2], "m": [0.1]}, "same shape"),
        ({"l": np.zeros((5, 2)), "m": np.zeros((5, 2))}, "time-dependent"),
        ({"l": [0.1], "m": [0.1], "n": [0.5, 0.5]}, "n must have"),
        ({"l": [0.1], "m": [0.1], "method": "fft"}, "method must be one of"),
        ({"l": [0.1], "m": [0.1], "w_term": "projected"}, "w_term must be one of"),
        ({"l": [0.1], "m": [0.1], "psf": True, "w_term": "zenith"}, "not supported"),
    ],
)
def test_invalid_arguments(kwargs, match):
    hermitian, _ = make_hermitian_pair(nbls=10, ntimes=2, nfreqs=2)
    with pytest.raises(ValueError, match=match):
        dirty_image_points(hermitian, **kwargs)
