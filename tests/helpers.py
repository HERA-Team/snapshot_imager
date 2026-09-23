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
    hermitian=False,
):
    """
    Unit-flux point source placed exactly on an image pixel.

    Visibilities follow the pyuvdata/pyuvsim convention used for HERA data,
    V = exp(+2*pi*i*(u*l0 + v*m0 + w*n0)) with uvw = xyz(ant2) - xyz(ant1).
    Baselines are coplanar (w = 0) and include their conjugates (the first
    ``nbls`` rows are the baselines, the next ``nbls`` their conjugates), so
    the dirty image is real. With ``hermitian=True`` only the first ``nbls``
    rows are kept and the conjugates are implied.
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
    if hermitian:
        vis, weights, uvw = vis[:nbls], weights[:nbls], uvw[:nbls]

    return PointSource(
        data=ImagingData(vis, weights, uvw, times, freqs, hermitian=hermitian),
        npix=npix,
        fov=fov,
        l_idx=l_idx,
        m_idx=m_idx,
    )


def make_hermitian_pair(nbls=12, ntimes=3, nfreqs=5, seed=3, dtype=np.complex128):
    """
    The same random visibilities in both layouts.

    Returns ``(hermitian, explicit)``: ``hermitian`` holds one row per
    baseline with the conjugates implied; ``explicit`` also stores each
    conjugate (-uvw, conj(vis), same weights). Weights are random, with some
    flags and one fully flagged snapshot.
    """
    rng = np.random.default_rng(seed)
    shape = (nbls, ntimes, nfreqs)
    vis = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    weights = rng.uniform(0.5, 2.0, shape)
    weights[rng.uniform(size=shape) < 0.2] = 0.0
    weights[:, 0, 1] = 0.0
    uvw = rng.uniform(-30, 30, (nbls, 3, nfreqs))
    times = 2459000.0 + np.arange(ntimes) / 1440.0
    freqs = np.linspace(100e6, 120e6, nfreqs)

    hermitian = ImagingData(vis, weights, uvw, times, freqs, hermitian=True)
    explicit = ImagingData(
        np.concatenate([vis, np.conj(vis)]),
        np.concatenate([weights, weights]),
        np.concatenate([uvw, -uvw]),
        times,
        freqs,
    )
    return hermitian, explicit
