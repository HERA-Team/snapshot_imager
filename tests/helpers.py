"""Simulation helpers shared by the test suite."""

from dataclasses import dataclass

import numpy as np

from snapshot_imager import ImagingData, compute_image_grid


@dataclass
class PointSource:
    """Simulated point-source observation and where it should appear."""

    data: ImagingData
    npix: int
    fov: float
    l_idx: int  # expected column (l) index of the source in the image
    m_idx: int  # expected row (m) index of the source in the image


def make_point_source(
    npix=32,
    fov=20.0,
    l_idx=20,
    m_idx=9,
    nbls=30,
    ntimes=2,
    nfreqs=3,
    dtype=np.complex128,
    seed=1,
):
    """
    Unit-flux point source placed exactly on an image pixel.

    Visibilities follow the pyuvdata/pyuvsim convention used for HERA data,
    V = exp(+2*pi*i*(u*l0 + v*m0 + w*n0)) with uvw = xyz(ant2) - xyz(ant1).
    Baselines are coplanar (w = 0) and include their conjugates (the first
    ``nbls`` rows are the baselines, the next ``nbls`` their conjugates), so
    the dirty image is real.
    """
    rng = np.random.default_rng(seed)

    freqs = np.linspace(100e6, 120e6, nfreqs)
    bl = rng.uniform(-40, 40, (nbls, 3))
    bl[:, 2] = 0.0
    uvw = bl[:, :, None] * (freqs / freqs[0])[None, None, :]
    uvw = np.concatenate([uvw, -uvw])

    lcoords, mcoords, _, _ = compute_image_grid(npix, fov)
    l0, m0 = lcoords[l_idx], mcoords[m_idx]

    vis = np.exp(2j * np.pi * (uvw[:, 0, :] * l0 + uvw[:, 1, :] * m0))
    vis = np.repeat(vis[:, None, :], ntimes, axis=1).astype(dtype)
    weights = np.ones(vis.shape)
    times = 2459000.0 + np.arange(ntimes) / 1440.0

    return PointSource(
        data=ImagingData(vis, weights, uvw, times, freqs),
        npix=npix,
        fov=fov,
        l_idx=l_idx,
        m_idx=m_idx,
    )
