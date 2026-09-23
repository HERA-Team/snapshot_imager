"""
Snapshot imaging with non-uniform FFTs (NUFFTs).

:func:`dirty_image` is the main entry point. It makes dirty images on a regular
(l, m) grid, either one image per channel or one multi-frequency synthesis
(MFS) image combining all channels, using FINUFFT on the CPU or cuFINUFFT on
the GPU.

Sign convention: visibilities are assumed to follow the pyuvdata/pyuvsim
convention used for HERA data, uvw = xyz(ant2) - xyz(ant1) and
V ∝ exp(+2πi (u*l + v*m + w*n)), so the dirty image is formed with the
kernel exp(-2πi (u*l + v*m)).

Pixels outside the visible sky (l**2 + m**2 > 1, which only occurs for
fov > 90 degrees) are set to NaN.

The functions ``snapshot_imager_type1``, ``snapshot_imager_type3``,
``snapshot_imager_mfs_type_1`` and ``snapshot_imager_mfs_type_3`` are kept for
backwards compatibility; they are thin wrappers around :func:`dirty_image`
with their original defaults and outputs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ._engine import (
    METHODS,
    GridTransform,
    get_backend,
    image_mfs,
    image_per_channel,
)
from .coordinates import _below_horizon, compute_baseline_extent, compute_image_grid
from .core import _nufft_dtypes, validate_imaging_inputs
from .data_models import ImageResult, ImagingData


def dirty_image(
    data: ImagingData,
    npix: int = 256,
    fov: float = 180.0,
    *,
    mfs: bool = False,
    method: str = "type1",
    eps: float = 1e-13,
    use_gpu: bool = False,
    rm_phasor: np.ndarray | None = None,
    verbose: bool = False,
) -> ImageResult:
    """
    Make dirty images from visibility data.

    Parameters
    ----------
    data : ImagingData
        Visibilities, weights, and uvw coordinates to image.
    npix : int, optional
        Number of pixels along each image axis. Default is 256.
    fov : float, optional
        Field of view in degrees (0 < fov <= 180), centered on the phase
        center. Default is 180 (the whole visible sky).
    mfs : bool, optional
        If False (default), make one image per channel, each snapshot
        normalized by its summed weights. If True, combine all channels into
        one multi-frequency synthesis image per time (not normalized).
    method : {"type1", "type3"}, optional
        NUFFT used to evaluate the image. "type1" (default) is much faster on
        a regular grid; "type3" evaluates the same sum directly at the pixel
        positions and is mainly useful for validation.
    eps : float, optional
        NUFFT tolerance. Default is 1e-13. Single-precision (complex64) data
        is imaged in single precision with a tolerance of at least 1e-6.
    use_gpu : bool, optional
        Image on the GPU with CuPy and cuFINUFFT. Falls back to the CPU with a
        warning if they are unavailable. Default is False.
    rm_phasor : np.ndarray, optional
        Complex array of shape (nfreqs,). If given (only with ``mfs=False``),
        each channel's image is multiplied by ``rm_phasor[fi]`` and the
        channels are summed, e.g. for rotation-measure synthesis.
    verbose : bool, optional
        Show a progress bar. Default is False.

    Returns
    -------
    ImageResult
        ``images`` has shape (ntimes, nfreqs, npix, npix), or
        (ntimes, 1, npix, npix) for MFS or with ``rm_phasor``, indexed
        ``images[time, freq, m, l]``. ``l_coords`` and ``m_coords`` are the
        pixel direction cosines, shape (npix,). With ``mfs=False``, a unit
        point source has peak 1 and fully flagged snapshots are zero.
        Pixels below the horizon are NaN.

    Notes
    -----
    Pixel ``npix // 2`` along each axis is the phase center (l = m = 0); see
    :func:`~snapshot_imager.compute_image_grid` for the grid definition.
    """
    validate_imaging_inputs(data.vis, data.weights, data.u, data.v, npix, fov)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if rm_phasor is not None:
        if mfs:
            raise ValueError("rm_phasor cannot be combined with mfs=True")
        rm_phasor = np.asarray(rm_phasor)
        if rm_phasor.shape != (data.nfreqs,):
            raise ValueError(
                f"rm_phasor must have shape ({data.nfreqs},), got {rm_phasor.shape}"
            )

    backend = get_backend(use_gpu)
    complex_dtype, _, eps = _nufft_dtypes(data.vis.dtype, eps)
    transform = GridTransform(
        backend,
        npix,
        fov,
        n_trans=1 if mfs else data.ntimes,
        method=method,
        eps=eps,
        complex_dtype=complex_dtype,
        uv_extent=compute_baseline_extent(data.u, data.v) if method == "type3" else None,
    )

    if mfs:
        images = image_mfs(data, transform, verbose=verbose)
    else:
        images = image_per_channel(data, transform, rm_phasor=rm_phasor, verbose=verbose)

    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
    images[:, :, _below_horizon(lgrid, mgrid)] = np.nan

    if mfs or rm_phasor is not None:
        freqs = np.array([np.mean(data.freqs)])
    else:
        freqs = np.asarray(data.freqs)

    return ImageResult(
        images=images,
        l_coords=lcoords,
        m_coords=mcoords,
        fov=fov,
        npix=npix,
        times=np.asarray(data.times),
        freqs=freqs,
    )


# ---------------------------------------------------------------------------
# Legacy interface
# ---------------------------------------------------------------------------


def _flattened_coords(result: ImageResult) -> ImageResult:
    """Legacy Type 3 results report the flattened (l, m) of every pixel."""
    _, _, lgrid, mgrid = compute_image_grid(result.npix, result.fov)
    return replace(result, l_coords=np.ravel(lgrid), m_coords=np.ravel(mgrid))


def snapshot_imager_type1(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 180,
    eps: float = 1e-13,
    use_cupy: bool = False,
    modeord: int = 0,
    verbose: bool = True,
    rm_phasor: np.ndarray | None = None,
) -> ImageResult:
    """
    Per-channel snapshot imager using a Type 1 NUFFT.

    Kept for backwards compatibility; equivalent to
    ``dirty_image(imaging_data, npix, fov, method="type1", ...)``.

    Parameters
    ----------
    imaging_data : ImagingData
        Prepared visibility data
    npix : int, optional
        Number of pixels per dimension. Default is 200.
    fov : float, optional
        Field of view in degrees. Default is 180.
    eps : float, optional
        FINUFFT tolerance (precision). Default is 1e-13.
    use_cupy : bool, optional
        Whether to use GPU acceleration. Default is False.
    modeord : int, optional
        Pixel ordering: 0 (default) is centered, matching ``l_coords`` and
        ``m_coords``; 1 is FFT-style ordering (apply ``np.fft.fftshift`` over
        the last two axes to match the returned coordinates).
    verbose : bool, optional
        Show a progress bar. Default is True.
    rm_phasor : np.ndarray, optional
        Optional complex array of shape (nfreqs,). If provided, each channel's
        image is multiplied by ``rm_phasor[fi]`` and the channels are summed
        (e.g. for rotation-measure synthesis), so the output has shape
        (ntimes, 1, npix, npix). Default is None.

    Returns
    -------
    ImageResult
        See :func:`dirty_image`.
    """
    if modeord not in (0, 1):
        raise ValueError(f"modeord must be 0 or 1, got {modeord}")
    result = dirty_image(
        imaging_data,
        npix,
        fov,
        method="type1",
        eps=eps,
        use_gpu=use_cupy,
        rm_phasor=rm_phasor,
        verbose=verbose,
    )
    if modeord == 1:
        result.images = np.fft.ifftshift(result.images, axes=(-2, -1))
    return result


def snapshot_imager_type3(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 180,
    eps: float = 1e-13,
    use_cupy: bool = False,
    verbose: bool = True,
) -> ImageResult:
    """
    Per-channel snapshot imager using a Type 3 NUFFT.

    Kept for backwards compatibility; equivalent to
    ``dirty_image(imaging_data, npix, fov, method="type3", ...)`` except that
    ``l_coords`` and ``m_coords`` are the flattened coordinates of every pixel
    (shape (npix*npix,)).

    Parameters
    ----------
    imaging_data : ImagingData
        Prepared visibility data
    npix : int, optional
        Number of pixels per dimension. Default is 200.
    fov : float, optional
        Field of view in degrees. Default is 180.
    eps : float, optional
        FINUFFT tolerance (precision). Default is 1e-13.
    use_cupy : bool, optional
        Whether to use GPU acceleration. Default is False.
    verbose : bool, optional
        Show a progress bar. Default is True.

    Returns
    -------
    ImageResult
        See :func:`dirty_image`.
    """
    result = dirty_image(
        imaging_data,
        npix,
        fov,
        method="type3",
        eps=eps,
        use_gpu=use_cupy,
        verbose=verbose,
    )
    return _flattened_coords(result)


def snapshot_imager_mfs_type_1(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 10,
    eps: float = 1e-13,
    use_cupy: bool = False,
    verbose: bool = True,
) -> ImageResult:
    """
    Multi-frequency synthesis (MFS) snapshot imager using a Type 1 NUFFT.

    Kept for backwards compatibility; equivalent to
    ``dirty_image(imaging_data, npix, fov, mfs=True, method="type1", ...)``.

    Parameters
    ----------
    imaging_data : ImagingData
        Prepared visibility data
    npix : int, optional
        Number of pixels per dimension. Default is 200.
    fov : float, optional
        Field of view in degrees. Default is 10.
    eps : float, optional
        FINUFFT tolerance (precision). Default is 1e-13.
    use_cupy : bool, optional
        Whether to use GPU acceleration. Default is False.
    verbose : bool, optional
        Show a progress bar. Default is True.

    Returns
    -------
    ImageResult
        One wideband image per time, shape (ntimes, 1, npix, npix). MFS images
        are not normalized: a unit point source peaks at the summed weights.
    """
    return dirty_image(
        imaging_data,
        npix,
        fov,
        mfs=True,
        method="type1",
        eps=eps,
        use_gpu=use_cupy,
        verbose=verbose,
    )


def snapshot_imager_mfs_type_3(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 10,
    eps: float = 1e-13,
    use_cupy: bool = False,
    verbose: bool = True,
) -> ImageResult:
    """
    Multi-frequency synthesis (MFS) snapshot imager using a Type 3 NUFFT.

    Kept for backwards compatibility; equivalent to
    ``dirty_image(imaging_data, npix, fov, mfs=True, method="type3", ...)``
    except that ``l_coords`` and ``m_coords`` are the flattened coordinates of
    every pixel (shape (npix*npix,)).

    Parameters
    ----------
    imaging_data : ImagingData
        Prepared visibility data
    npix : int, optional
        Number of pixels per dimension. Default is 200.
    fov : float, optional
        Field of view in degrees. Default is 10.
    eps : float, optional
        FINUFFT tolerance (precision). Default is 1e-13.
    use_cupy : bool, optional
        Whether to use GPU acceleration. Default is False.
    verbose : bool, optional
        Show a progress bar. Default is True.

    Returns
    -------
    ImageResult
        One wideband image per time, shape (ntimes, 1, npix, npix). MFS images
        are not normalized: a unit point source peaks at the summed weights.
    """
    result = dirty_image(
        imaging_data,
        npix,
        fov,
        mfs=True,
        method="type3",
        eps=eps,
        use_gpu=use_cupy,
        verbose=verbose,
    )
    return _flattened_coords(result)
