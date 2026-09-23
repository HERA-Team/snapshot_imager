"""Tests for snapshot_imager.preprocessing."""

import numpy as np
import pytest
from astropy import constants
from hera_cal.datacontainer import DataContainer

from snapshot_imager import (
    ImagingData,
    snapshot_imager_type1,
    unpack_data_containers,
    unpack_uvdata,
)

NTIMES, NFREQS = 4, 6
ANTPAIRS = [(0, 1), (0, 2), (1, 2)]
POL = "ee"


@pytest.fixture
def containers():
    """Synthetic hera_cal DataContainers for data, flags and nsamples."""
    rng = np.random.default_rng(7)
    shape = (NTIMES, NFREQS)
    data, flags, nsamples = {}, {}, {}
    for ap in ANTPAIRS:
        key = ap + (POL,)
        data[key] = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        flags[key] = rng.uniform(size=shape) < 0.2
        nsamples[key] = rng.integers(1, 4, size=shape).astype(float)

    data, flags, nsamples = DataContainer(data), DataContainer(flags), DataContainer(nsamples)
    data.freqs = np.linspace(100e6, 110e6, NFREQS)
    data.times = 2459000.0 + np.arange(NTIMES) / 1440.0
    data.antpos = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([14.6, 0.0, 0.0]),
        2: np.array([7.3, 12.6, 0.0]),
    }
    return data, flags, nsamples


class TestUnpackDataContainers:
    def test_returns_imaging_data(self, containers):
        out = unpack_data_containers(*containers, pol=POL, antpairs=ANTPAIRS)
        assert isinstance(out, ImagingData)
        # Each baseline is included together with its conjugate
        assert out.shape == (2 * len(ANTPAIRS), NTIMES, NFREQS)
        assert out.uvw.shape == (2 * len(ANTPAIRS), 3, NFREQS)

    def test_defaults_come_from_data_container(self, containers):
        data = containers[0]
        out = unpack_data_containers(*containers, pol=POL)
        assert out.nbls == 2 * len(data.antpairs())
        np.testing.assert_array_equal(out.freqs, data.freqs)
        np.testing.assert_array_equal(out.times, data.times)

    def test_conjugate_baselines(self, containers):
        data = containers[0]
        out = unpack_data_containers(*containers, pol=POL, antpairs=ANTPAIRS)
        for i, ap in enumerate(ANTPAIRS):
            np.testing.assert_allclose(out.vis[2 * i], data[ap + (POL,)])
            np.testing.assert_allclose(out.vis[2 * i + 1], np.conj(data[ap + (POL,)]))
            np.testing.assert_allclose(out.uvw[2 * i + 1], -out.uvw[2 * i])
            np.testing.assert_array_equal(out.weights[2 * i + 1], out.weights[2 * i])

    def test_uvw_in_wavelengths(self, containers):
        data = containers[0]
        out = unpack_data_containers(*containers, pol=POL, antpairs=ANTPAIRS)
        for i, (a1, a2) in enumerate(ANTPAIRS):
            blvec = data.antpos[a2] - data.antpos[a1]
            expected = blvec[:, None] * data.freqs[None, :] / constants.c.value
            np.testing.assert_allclose(out.uvw[2 * i], expected)

    def test_weights_use_nsamples_and_flags(self, containers):
        _, flags, nsamples = containers
        out = unpack_data_containers(*containers, pol=POL, antpairs=ANTPAIRS)
        for i, ap in enumerate(ANTPAIRS):
            key = ap + (POL,)
            expected = nsamples[key] * (~flags[key]).astype(float)
            np.testing.assert_allclose(out.weights[2 * i], expected)

    def test_weights_without_nsamples(self, containers):
        _, flags, _ = containers
        out = unpack_data_containers(
            *containers, pol=POL, antpairs=ANTPAIRS, weight_by_nsamples=False
        )
        for i, ap in enumerate(ANTPAIRS):
            expected = (~flags[ap + (POL,)]).astype(float)
            np.testing.assert_allclose(out.weights[2 * i], expected)

    def test_time_and_freq_slices(self, containers):
        data = containers[0]
        tslice, fslice = slice(1, 3), slice(2, 5)
        out = unpack_data_containers(
            *containers,
            pol=POL,
            antpairs=ANTPAIRS,
            time_slice=tslice,
            freq_slice=fslice,
        )
        assert out.shape == (2 * len(ANTPAIRS), 2, 3)
        np.testing.assert_array_equal(out.times, data.times[tslice])
        np.testing.assert_array_equal(out.freqs, data.freqs[fslice])
        np.testing.assert_allclose(out.vis[0], data[ANTPAIRS[0] + (POL,)][tslice, fslice])

    def test_explicit_antpos_and_freqs_override_container(self, containers):
        antpos = {k: 2 * v for k, v in containers[0].antpos.items()}
        freqs = containers[0].freqs * 1.5
        out = unpack_data_containers(
            *containers, pol=POL, antpairs=ANTPAIRS, antpos=antpos, freqs=freqs
        )
        blvec = antpos[1] - antpos[0]
        np.testing.assert_allclose(
            out.uvw[0], blvec[:, None] * freqs[None, :] / constants.c.value
        )

    def test_output_can_be_imaged(self, containers):
        out = unpack_data_containers(*containers, pol=POL, antpairs=ANTPAIRS)
        result = snapshot_imager_type1(out, npix=16, fov=90.0, verbose=False)
        assert result.shape == (NTIMES, NFREQS, 16, 16)
        assert np.all(np.isfinite(result.images))


def test_unpack_uvdata_not_implemented():
    with pytest.raises(NotImplementedError):
        unpack_uvdata(object())
