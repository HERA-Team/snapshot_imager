"""Tests for the imaging algorithms in snapshot_imager.imager."""

import numpy as np
import pytest

from snapshot_imager import (
    ImageResult,
    ImagingData,
    snapshot_imager_mfs_type_1,
    snapshot_imager_mfs_type_3,
    snapshot_imager_type1,
    snapshot_imager_type3,
)

PER_FREQ_IMAGERS = [snapshot_imager_type1, snapshot_imager_type3]
MFS_IMAGERS = [snapshot_imager_mfs_type_1, snapshot_imager_mfs_type_3]
ALL_IMAGERS = PER_FREQ_IMAGERS + MFS_IMAGERS


# ---------------------------------------------------------------------------
# Output structure
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

    def test_rm_phasor_sums_over_frequency(self, imaging_data):
        """With rm_phasor, channels are weighted by the phasor and summed."""
        nfreqs = imaging_data.nfreqs
        phasor = np.exp(1j * np.linspace(0, np.pi, nfreqs))

        per_freq = snapshot_imager_type1(imaging_data, npix=16, fov=10.0, verbose=False)
        summed = snapshot_imager_type1(
            imaging_data, npix=16, fov=10.0, verbose=False, rm_phasor=phasor
        )

        assert summed.shape == (imaging_data.ntimes, 1, 16, 16)
        expected = np.sum(per_freq.images * phasor[None, :, None, None], axis=1)
        np.testing.assert_allclose(summed.images[:, 0], expected, atol=1e-10)

    def test_fully_flagged_channel_is_zero(self, imaging_data):
        weights = imaging_data.weights.copy()
        weights[:, :, 3] = 0.0
        data = ImagingData(
            imaging_data.vis,
            weights,
            imaging_data.uvw,
            imaging_data.times,
            imaging_data.freqs,
        )
        result = snapshot_imager_type1(data, npix=16, fov=10.0, verbose=False)
        assert np.all(result.images[:, 3] == 0)
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

    def test_coords_are_flattened_grid(self, imaging_data):
        result = snapshot_imager_type3(imaging_data, npix=16, fov=10.0, verbose=False)
        assert result.l_coords.shape == (16 * 16,)
        assert result.m_coords.shape == (16 * 16,)


class TestMFSType1Imager:
    def test_output_shape(self, imaging_data_small):
        # MFS collapses all frequencies into a single wideband image per time step
        result = snapshot_imager_mfs_type_1(
            imaging_data_small, npix=16, fov=10.0, verbose=False
        )
        assert result.shape == (2, 1, 16, 16)

    def test_returns_image_result(self, imaging_data_small):
        result = snapshot_imager_mfs_type_1(
            imaging_data_small, npix=16, fov=10.0, verbose=False
        )
        assert isinstance(result, ImageResult)


class TestMFSType3Imager:
    def test_output_shape(self, imaging_data_small):
        # MFS collapses all frequencies into a single wideband image per time step
        result = snapshot_imager_mfs_type_3(
            imaging_data_small, npix=16, fov=10.0, verbose=False
        )
        assert result.shape == (2, 1, 16, 16)

    def test_returns_image_result(self, imaging_data_small):
        result = snapshot_imager_mfs_type_3(
            imaging_data_small, npix=16, fov=10.0, verbose=False
        )
        assert isinstance(result, ImageResult)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("imager", ALL_IMAGERS)
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"npix": 0, "fov": 10.0}, "npix must be positive"),
        ({"npix": 16, "fov": 0.0}, "fov must be between"),
        ({"npix": 16, "fov": 200.0}, "fov must be between"),
    ],
)
def test_invalid_parameters_raise(imager, imaging_data_small, kwargs, match):
    with pytest.raises(ValueError, match=match):
        imager(imaging_data_small, verbose=False, **kwargs)


# ---------------------------------------------------------------------------
# Correctness against a simulated point source
# ---------------------------------------------------------------------------


def _peak_index(image):
    return np.unravel_index(np.argmax(image.real), image.shape)


@pytest.mark.parametrize("imager", PER_FREQ_IMAGERS)
def test_point_source_location_and_flux(imager, point_source):
    """A unit point source on a pixel images to that pixel with peak 1."""
    ps = point_source
    result = imager(ps.data, npix=ps.npix, fov=ps.fov, verbose=False)

    for ti in range(ps.data.ntimes):
        for fi in range(ps.data.nfreqs):
            image = result.images[ti, fi]
            assert _peak_index(image) == (ps.m_idx, ps.l_idx)
            assert image[ps.m_idx, ps.l_idx].real == pytest.approx(1.0, abs=1e-8)
            # Hermitian baseline coverage -> real image
            np.testing.assert_allclose(image.imag, 0.0, atol=1e-8)


@pytest.mark.parametrize("imager", MFS_IMAGERS)
def test_mfs_point_source_location(imager, point_source):
    """The MFS image of a point source peaks at the source pixel."""
    ps = point_source
    result = imager(ps.data, npix=ps.npix, fov=ps.fov, verbose=False)

    total_weight = ps.data.weights[:, 0, :].sum()
    for ti in range(ps.data.ntimes):
        image = result.images[ti, 0]
        assert _peak_index(image) == (ps.m_idx, ps.l_idx)
        # MFS images are not normalized: the peak equals the summed weights
        assert image[ps.m_idx, ps.l_idx].real == pytest.approx(total_weight, rel=1e-8)
        np.testing.assert_allclose(image.imag, 0.0, atol=1e-6)


def test_type1_matches_type3(imaging_data):
    """Type 1 and Type 3 NUFFTs evaluate the same DFT on the same grid."""
    kwargs = {"npix": 32, "fov": 10.0, "verbose": False}
    t1 = snapshot_imager_type1(imaging_data, **kwargs)
    t3 = snapshot_imager_type3(imaging_data, **kwargs)
    np.testing.assert_allclose(t1.images, t3.images, atol=1e-8)


def test_mfs_type1_matches_mfs_type3(imaging_data_small):
    kwargs = {"npix": 16, "fov": 10.0, "verbose": False}
    t1 = snapshot_imager_mfs_type_1(imaging_data_small, **kwargs)
    t3 = snapshot_imager_mfs_type_3(imaging_data_small, **kwargs)
    np.testing.assert_allclose(t1.images, t3.images, atol=1e-8)


def test_type1_matches_direct_dft(imaging_data_small):
    """Compare the Type 1 imager against a brute-force DFT."""
    npix, fov = 8, 10.0
    result = snapshot_imager_type1(
        imaging_data_small, npix=npix, fov=fov, verbose=False
    )

    lgrid, mgrid = np.meshgrid(result.l_coords, result.m_coords)
    d = imaging_data_small
    for fi in range(d.nfreqs):
        kernel = np.exp(
            2j
            * np.pi
            * (
                d.u[:, fi, None, None] * lgrid[None]
                + d.v[:, fi, None, None] * mgrid[None]
            )
        )
        weighted = d.vis[:, :, fi] * d.weights[:, :, fi]
        dirty = np.einsum("bt,bml->tml", weighted, kernel)
        psf = np.einsum("bt,bml->tml", d.weights[:, :, fi], kernel)
        expected = dirty / psf.real.max()
        np.testing.assert_allclose(result.images[:, fi], expected, atol=1e-8)
