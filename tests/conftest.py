"""Shared fixtures for the snapshot_imager test suite."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.utils import iers

from helpers import make_point_source

from snapshot_imager import ImagingData

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


@pytest.fixture(params=[32, 33], ids=["even-npix", "odd-npix"])
def point_source(request):
    """Unit point source on a pixel, for both even and odd image sizes."""
    return make_point_source(npix=request.param)
