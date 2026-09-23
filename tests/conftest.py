"""Shared fixtures for the snapshot_imager test suite."""

from dataclasses import dataclass

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.utils import iers

from snapshot_imager import ImagingData, compute_image_grid

# Keep tests hermetic: use astropy's bundled IERS tables (which cover the
# test epochs) rather than downloading IERS-A from the network.
iers.conf.auto_download = False


def _random_imaging_data(nbls, ntimes, nfreqs, seed=42):
    rng = np.random.default_rng(seed)
    vis = rng.standard_normal((nbls, ntimes, nfreqs)) + 1j * rng.standard_normal(
        (nbls, ntimes, nfreqs)
    )
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = rng.standard_normal((nbls, 3, nfreqs)) * 50
    times = np.linspace(2459000, 2459000.05, ntimes)
    freqs = np.linspace(100e6, 150e6, nfreqs)
    return ImagingData(vis, weights, uvw, times, freqs)


@pytest.fixture
def imaging_data():
    """Standard-sized random ImagingData for most tests."""
    return _random_imaging_data(nbls=20, ntimes=5, nfreqs=8)


@pytest.fixture
def imaging_data_small():
    """Small random ImagingData for slower MFS tests."""
    return _random_imaging_data(nbls=10, ntimes=2, nfreqs=4)


@pytest.fixture
def telescope_location():
    """HERA telescope location."""
    return EarthLocation(lat=-30.7215 * u.deg, lon=21.4283 * u.deg, height=1051 * u.m)


@dataclass
class PointSource:
    """Simulated point-source observation and where it should appear."""

    data: ImagingData
    npix: int
    fov: float
    l_idx: int  # expected column (l) index of the source in the image
    m_idx: int  # expected row (m) index of the source in the image


@pytest.fixture
def point_source():
    """
    Unit-flux point source placed exactly on an image pixel.

    Baselines are coplanar (w = 0) and include their conjugates so the
    dirty image is real. Visibilities follow V = exp(-2*pi*i*(u*l0 + v*m0)).
    """
    npix, fov = 32, 20.0
    nbls, ntimes, nfreqs = 30, 2, 3
    rng = np.random.default_rng(1)

    freqs = np.linspace(100e6, 120e6, nfreqs)
    bl = rng.uniform(-40, 40, (nbls, 3))
    bl[:, 2] = 0.0
    uvw = bl[:, :, None] * (freqs / freqs[0])[None, None, :]
    uvw = np.concatenate([uvw, -uvw])

    lcoords, mcoords, _, _ = compute_image_grid(npix, fov)
    l_idx, m_idx = 20, 9
    l0, m0 = lcoords[l_idx], mcoords[m_idx]

    vis = np.exp(-2j * np.pi * (uvw[:, 0, :] * l0 + uvw[:, 1, :] * m0))
    vis = np.repeat(vis[:, None, :], ntimes, axis=1)
    weights = np.ones(vis.shape)
    times = 2459000.0 + np.arange(ntimes) / 1440.0

    return PointSource(
        data=ImagingData(vis, weights, uvw, times, freqs),
        npix=npix,
        fov=fov,
        l_idx=l_idx,
        m_idx=m_idx,
    )
