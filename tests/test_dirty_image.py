"""Tests for snapshot_imager.dirty_image and its legacy wrappers."""

import inspect

import numpy as np
import pytest
from helpers import make_hermitian_pair, make_point_source

import snapshot_imager._engine as engine
from snapshot_imager import (
    compute_image_grid,
    dirty_image,
    snapshot_imager_mfs_type_1,
    snapshot_imager_mfs_type_3,
    snapshot_imager_type1,
    snapshot_imager_type3,
)
from snapshot_imager._engine import GridTransform, get_backend

# (legacy function, equivalent dirty_image keyword arguments)
LEGACY = [
    (snapshot_imager_type1, {"method": "type1"}),
    (snapshot_imager_type3, {"method": "type3"}),
    (snapshot_imager_mfs_type_1, {"method": "type1", "mfs": True}),
    (snapshot_imager_mfs_type_3, {"method": "type3", "mfs": True}),
]


# ---------------------------------------------------------------------------
# dirty_image
# ---------------------------------------------------------------------------


class TestDirtyImage:
    def test_defaults(self, imaging_data_small):
        result = dirty_image(imaging_data_small)
        assert result.npix == 256
        assert result.fov == 180.0
        assert result.shape == (2, 4, 256, 256)

    @pytest.mark.parametrize("method", ["type1", "type3"])
    def test_coords_are_1d(self, imaging_data_small, method):
        result = dirty_image(imaging_data_small, 16, 10.0, method=method)
        lcoords, mcoords, _, _ = compute_image_grid(16, 10.0)
        np.testing.assert_array_equal(result.l_coords, lcoords)
        np.testing.assert_array_equal(result.m_coords, mcoords)

    def test_times_and_freqs_per_channel(self, imaging_data_small):
        d = imaging_data_small
        result = dirty_image(d, 16, 10.0)
        np.testing.assert_array_equal(result.times, d.times)
        np.testing.assert_array_equal(result.freqs, d.freqs)

    @pytest.mark.parametrize("combine", ["mfs", "rm_phasor"])
    def test_combined_images_report_mean_frequency(self, imaging_data_small, combine):
        d = imaging_data_small
        kwargs = (
            {"mfs": True}
            if combine == "mfs"
            else {"rm_phasor": np.ones(d.nfreqs, dtype=complex)}
        )
        result = dirty_image(d, 16, 10.0, **kwargs)
        assert result.shape == (d.ntimes, 1, 16, 16)
        np.testing.assert_allclose(result.freqs, [np.mean(d.freqs)])

    @pytest.mark.parametrize("method", ["type1", "type3"])
    @pytest.mark.parametrize("mfs", [False, True])
    def test_point_source(self, method, mfs):
        ps = make_point_source(npix=33)
        result = dirty_image(
            ps.data, ps.npix, ps.fov, method=method, mfs=mfs, eps=1e-12
        )
        peak = np.nanmax(result.images.real, axis=(-2, -1))
        m_idx, l_idx = np.unravel_index(
            np.nanargmax(result.images[0, 0].real), (ps.npix, ps.npix)
        )
        assert (m_idx, l_idx) == (ps.m_idx, ps.l_idx)
        expected = ps.data.weights[:, 0, :].sum() if mfs else 1.0
        np.testing.assert_allclose(peak, expected, rtol=1e-8)

    def test_invalid_method(self, imaging_data_small):
        with pytest.raises(ValueError, match="method must be one of"):
            dirty_image(imaging_data_small, 16, 10.0, method="type2")

    def test_rm_phasor_with_mfs_raises(self, imaging_data_small):
        with pytest.raises(ValueError, match="cannot be combined with mfs"):
            dirty_image(
                imaging_data_small,
                16,
                10.0,
                mfs=True,
                rm_phasor=np.ones(imaging_data_small.nfreqs),
            )

    def test_rm_phasor_wrong_shape_raises(self, imaging_data_small):
        with pytest.raises(ValueError, match="rm_phasor must have shape"):
            dirty_image(imaging_data_small, 16, 10.0, rm_phasor=np.ones(3))

    def test_rm_phasor_complex128_with_complex64_data(self):
        """A double-precision phasor accumulates into single-precision images."""
        kwargs = {"npix": 16, "fov": 20.0, "l_idx": 11, "m_idx": 4}
        single = make_point_source(dtype=np.complex64, **kwargs)
        double = make_point_source(**kwargs)
        phasor = np.exp(1j * np.linspace(0, np.pi, single.data.nfreqs))
        assert phasor.dtype == np.complex128

        result = dirty_image(single.data, 16, 20.0, rm_phasor=phasor)
        expected = dirty_image(double.data, 16, 20.0, rm_phasor=phasor)

        assert result.images.dtype == np.complex64
        scale = np.abs(expected.images).max()
        np.testing.assert_allclose(result.images, expected.images, atol=1e-4 * scale)


# ---------------------------------------------------------------------------
# Hermitian data (conjugate baselines implied)
# ---------------------------------------------------------------------------


class TestHermitian:
    @pytest.mark.parametrize("npix", [32, 33])
    @pytest.mark.parametrize("mfs", [False, True])
    @pytest.mark.parametrize("method", ["type1", "type3"])
    def test_matches_explicit_conjugates(self, method, mfs, npix):
        hermitian, explicit = make_hermitian_pair()
        kwargs = {"method": method, "mfs": mfs, "eps": 1e-12}
        a = dirty_image(hermitian, npix, 180.0, **kwargs).images
        b = dirty_image(explicit, npix, 180.0, **kwargs).images

        assert a.dtype == np.float64
        assert b.dtype == np.complex128
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        scale = np.nanmax(np.abs(b))
        np.testing.assert_allclose(a, b.real, atol=1e-10 * scale)
        np.testing.assert_allclose(b.imag, 0.0, atol=1e-10 * scale)

    def test_rm_phasor_output_is_complex(self):
        hermitian, explicit = make_hermitian_pair()
        phasor = np.exp(1j * np.linspace(0, np.pi, hermitian.nfreqs))
        a = dirty_image(hermitian, 16, 20.0, rm_phasor=phasor, eps=1e-12).images
        b = dirty_image(explicit, 16, 20.0, rm_phasor=phasor, eps=1e-12).images
        assert a.dtype == np.complex128
        np.testing.assert_allclose(a, b, atol=1e-10 * np.abs(b).max())

    @pytest.mark.parametrize("mfs", [False, True])
    def test_complex64_gives_float32(self, mfs):
        hermitian, explicit = make_hermitian_pair(dtype=np.complex64)
        a = dirty_image(hermitian, 16, 20.0, mfs=mfs).images
        b = dirty_image(explicit, 16, 20.0, mfs=mfs).images
        assert a.dtype == np.float32
        np.testing.assert_allclose(a, b.real, atol=1e-4 * np.abs(b).max())

    @pytest.mark.parametrize("mfs", [False, True])
    def test_point_source(self, mfs):
        ps = make_point_source(npix=33, hermitian=True)
        result = dirty_image(ps.data, ps.npix, ps.fov, mfs=mfs, eps=1e-12)
        peak = result.images[:, :, ps.m_idx, ps.l_idx]
        # Per channel: unit peak. MFS: summed weights of every visibility,
        # including the implied conjugates.
        expected = 2 * ps.data.weights[:, 0, :].sum() if mfs else 1.0
        np.testing.assert_allclose(peak, expected, rtol=1e-9)
        assert np.all(result.images[0, 0] <= peak[0, 0] + 1e-9)


# ---------------------------------------------------------------------------
# Channel chunking and defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hermitian_layout", [True, False])
@pytest.mark.parametrize("rm_phasor", [False, True])
def test_channel_chunking_does_not_change_images(
    monkeypatch, hermitian_layout, rm_phasor
):
    hermitian, explicit = make_hermitian_pair(nfreqs=8)
    data = hermitian if hermitian_layout else explicit
    kwargs = {"eps": 1e-12}
    if rm_phasor:
        kwargs["rm_phasor"] = np.exp(1j * np.arange(data.nfreqs))
    expected = dirty_image(data, 16, 20.0, **kwargs).images

    per_channel_bytes = data.nbls * data.ntimes * 16
    for channels_per_chunk in (1, 3):  # 3 does not divide 8 channels
        monkeypatch.setattr(
            engine, "_CHUNK_BYTES", channels_per_chunk * per_channel_bytes
        )
        result = dirty_image(data, 16, 20.0, **kwargs).images
        np.testing.assert_allclose(result, expected, atol=1e-12 * np.abs(expected).max())


def test_default_eps_is_1e_6():
    for func in (
        dirty_image,
        snapshot_imager_type1,
        snapshot_imager_type3,
        snapshot_imager_mfs_type_1,
        snapshot_imager_mfs_type_3,
    ):
        assert inspect.signature(func).parameters["eps"].default == 1e-6


@pytest.mark.parametrize("mfs", [False, True])
def test_default_eps_accuracy(mfs):
    hermitian, _ = make_hermitian_pair(nbls=40)
    precise = dirty_image(hermitian, 64, 180.0, mfs=mfs, eps=1e-12).images
    default = dirty_image(hermitian, 64, 180.0, mfs=mfs).images
    scale = np.nanmax(np.abs(precise))
    np.testing.assert_allclose(
        np.nan_to_num(default), np.nan_to_num(precise), atol=5e-5 * scale
    )


# ---------------------------------------------------------------------------
# Legacy wrappers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("legacy, kwargs", LEGACY)
def test_legacy_matches_dirty_image(imaging_data_small, legacy, kwargs):
    """The legacy functions return what dirty_image returns."""
    old = legacy(imaging_data_small, npix=17, fov=20.0, verbose=False)
    new = dirty_image(imaging_data_small, 17, 20.0, **kwargs)
    # Not bitwise: FINUFFT's multithreaded spreading sums in a run-dependent
    # order, so repeated runs can differ at the ~1e-15 level.
    scale = np.abs(new.images).max()
    np.testing.assert_allclose(old.images, new.images, rtol=0, atol=1e-12 * scale)
    assert (old.npix, old.fov) == (new.npix, new.fov)
    if kwargs["method"] == "type3":
        # Legacy Type 3 functions report the flattened coordinates of every pixel
        _, _, lgrid, mgrid = compute_image_grid(17, 20.0)
        np.testing.assert_array_equal(old.l_coords, lgrid.ravel())
        np.testing.assert_array_equal(old.m_coords, mgrid.ravel())
    else:
        np.testing.assert_array_equal(old.l_coords, new.l_coords)
        np.testing.assert_array_equal(old.m_coords, new.m_coords)


@pytest.mark.parametrize(
    "legacy, npix, fov",
    [
        (snapshot_imager_type1, 200, 180),
        (snapshot_imager_type3, 200, 180),
        (snapshot_imager_mfs_type_1, 200, 10),
        (snapshot_imager_mfs_type_3, 200, 10),
    ],
)
def test_legacy_defaults_unchanged(legacy, npix, fov):
    ps = make_point_source(nbls=5, ntimes=1, nfreqs=1)
    result = legacy(ps.data, verbose=False)
    assert (result.npix, result.fov) == (npix, fov)


def test_legacy_invalid_modeord(imaging_data_small):
    with pytest.raises(ValueError, match="modeord must be 0 or 1"):
        snapshot_imager_type1(imaging_data_small, npix=16, modeord=2, verbose=False)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestGridTransform:
    def test_type1_plan_is_reused(self):
        transform = GridTransform(get_backend(), 16, 20.0, n_trans=2)
        plan = transform._plan
        rng = np.random.default_rng(0)
        for _ in range(3):
            transform.set_points(*rng.uniform(-20, 20, (2, 30)))
            assert transform._plan is plan

    def test_matches_direct_sum(self):
        """One transform, repeatedly re-pointed, matches a brute-force DFT."""
        npix, fov = 9, 30.0
        lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
        rng = np.random.default_rng(1)
        for method in ("type1", "type3"):
            transform = GridTransform(
                get_backend(), npix, fov, n_trans=2, method=method, uv_extent=40.0
            )
            for _ in range(2):
                u, v = rng.uniform(-40, 40, (2, 25))
                c = rng.standard_normal((2, 25)) + 1j * rng.standard_normal((2, 25))
                transform.set_points(u, v)
                kernel = np.exp(
                    -2j * np.pi * (u[:, None, None] * lgrid + v[:, None, None] * mgrid)
                )
                expected = np.einsum("tj,jml->tml", c, kernel)
                np.testing.assert_allclose(transform(c), expected, atol=1e-9)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            GridTransform(get_backend(), 16, 20.0, n_trans=1, method="type2")

    def test_type3_requires_uv_extent(self):
        with pytest.raises(ValueError, match="uv_extent"):
            GridTransform(get_backend(), 16, 20.0, n_trans=1, method="type3")
