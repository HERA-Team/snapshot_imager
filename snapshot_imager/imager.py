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
    _DIRECT_MAX_TERMS,
    METHODS,
    POINT_METHODS,
    GridTransform,
    PointTransform,
    evaluate_points,
    get_backend,
    image_mfs,
    image_per_channel,
    weights_constant_in_time,
)
from .coordinates import _below_horizon, compute_baseline_extent, compute_image_grid
from .core import _nufft_dtypes, validate_imaging_inputs
from .data_models import ImageResult, ImagingData, PointsResult


def dirty_image(
    data: ImagingData,
    npix: int = 256,
    fov: float = 180.0,
    *,
    mfs: bool = False,
    method: str = "type1",
    eps: float = 1e-6,
    use_gpu: bool = False,
    rm_phasor: np.ndarray | None = None,
    return_psf: bool = False,
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
        NUFFT tolerance: the relative error of the image. Default is 1e-6,
        far below the noise in any dirty image; use a smaller value (down to
        ~1e-14) for validation. Single-precision (complex64) data is imaged in
        single precision with a tolerance of at least 1e-6.
    use_gpu : bool, optional
        Image on the GPU with CuPy and cuFINUFFT. Falls back to the CPU with a
        warning if they are unavailable. Default is False.
    rm_phasor : np.ndarray, optional
        Complex array of shape (nfreqs,). If given (only with ``mfs=False``),
        each channel's image is multiplied by ``rm_phasor[fi]`` and the
        channels are summed, e.g. for rotation-measure synthesis.
    return_psf : bool, optional
        Also compute the synthesized beam (``ImageResult.psf``): the image of
        unit visibilities, with the same shape and normalization as the
        images. It is transformed together with the visibilities, and only
        once for all times when the weights don't change with time. Default
        is False.
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

        For Hermitian data (``data.hermitian``, the default from
        :func:`~snapshot_imager.unpack_data_containers`) the images are
        real-valued (float64, or float32 for complex64 data), except with
        ``rm_phasor``, whose summed images are complex. Otherwise the images
        are complex.

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
    # Number of synthesized-beam transforms per channel: one if the weights
    # (and hence the beam) are the same at every time, else one per time
    psf_rows = 0
    if return_psf:
        psf_rows = 1 if weights_constant_in_time(data) else data.ntimes
    transform = GridTransform(
        backend,
        npix,
        fov,
        n_trans=1 if mfs else data.ntimes + psf_rows,
        method=method,
        eps=eps,
        complex_dtype=complex_dtype,
        uv_extent=compute_baseline_extent(data.u, data.v) if method == "type3" else None,
    )

    if mfs:
        images, psf = image_mfs(data, transform, psf_rows=psf_rows, verbose=verbose)
    else:
        images, psf = image_per_channel(
            data, transform, rm_phasor=rm_phasor, psf_rows=psf_rows, verbose=verbose
        )

    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
    below_horizon = _below_horizon(lgrid, mgrid)
    images[:, :, below_horizon] = np.nan
    if psf is not None:
        psf[:, :, below_horizon] = np.nan
        if psf.shape != images.shape:
            psf = np.broadcast_to(psf, images.shape)

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
        psf=psf,
        sum_weights=_sum_weights(data),
    )


def _sum_weights(data: ImagingData) -> np.ndarray:
    """Summed weights per (time, channel), counting implied conjugates."""
    return data.weights.sum(axis=0) * (2 if data.hermitian else 1)


_W_TERMS = (None, "unprojected", "zenith")


def dirty_image_points(
    data: ImagingData,
    l,
    m,
    n=None,
    *,
    mfs: bool = False,
    psf: bool = False,
    w_term: str | None = None,
    method: str = "auto",
    eps: float = 1e-6,
    use_gpu: bool = False,
    verbose: bool = False,
) -> PointsResult:
    """
    Evaluate the dirty image (or the synthesized beam) at arbitrary directions.

    The values are exactly what :func:`dirty_image` would give at those
    directions, without evaluating a full grid: for example at catalog source
    positions (from :func:`~snapshot_imager.radec_to_lmn`), or along the
    horizon.

    Parameters
    ----------
    data : ImagingData
        Visibilities, weights, and uvw coordinates.
    l, m : array_like
        Direction cosines toward East and North, shape (npoints,) for the same
        directions at every time, or (ntimes, npoints) for directions that
        change with time (e.g. catalog sources drifting through the beam).
        With ``psf=True`` these are offsets from the source.
    n : array_like, optional
        Up direction cosines, same shape as ``l``. Used for the w-term and to
        mask directions below the horizon (n < 0). If omitted,
        ``n = sqrt(1 - l**2 - m**2)``.
    mfs : bool, optional
        Combine all channels (unnormalized, as in ``dirty_image(mfs=True)``)
        instead of evaluating each channel.
    psf : bool, optional
        Evaluate the synthesized beam at the offsets ``(l, m)`` instead of the
        image: the response to a unit point source at the phase center, with
        the same normalization as the images (per-channel peak 1).
    w_term : {None, "unprojected", "zenith"}, optional
        Whether and how to include the w-term, using ``data.uvw``:

        - None (default): ignore w, a 2D transform as in :func:`dirty_image`.
        - "unprojected": kernel exp(-2πi (u l + v m + w n)), for drift-scan
          data in the pyuvdata/pyuvsim convention, where a source in direction
          (l, m, n) has V ∝ exp(+2πi (u l + v m + w n)). Here ignoring w costs
          the most coherence near zenith (n ≈ 1).
        - "zenith": kernel exp(-2πi (u l + v m + w (n - 1))), for data whose
          zenith phase exp(2πi w) has been removed, either by phasing to
          zenith or by per-antenna calibration gains absorbing it. Here
          ignoring w costs the most coherence near the horizon.

        The w-term is only meaningful when each row has a well-defined w: for
        redundantly averaged data the heights of the averaged baselines
        differ, so the group's w is approximate. Not supported with
        ``psf=True``.
    method : {"auto", "direct", "type3"}, optional
        "direct" evaluates the sum exactly (a matrix product; best for tens to
        thousands of directions), "type3" uses a Type 3 NUFFT with tolerance
        ``eps`` (best for many directions). "auto" (default) picks "direct"
        when (baselines x directions) is small.
    eps : float, optional
        NUFFT tolerance for "type3". Default is 1e-6.
    use_gpu : bool, optional
        Use CuPy (and cuFINUFFT for "type3"). Default is False.
    verbose : bool, optional
        Show a progress bar. Default is False.

    Returns
    -------
    PointsResult
        ``values`` of shape (ntimes, nfreqs, npoints), or (ntimes, 1, npoints)
        for MFS; real-valued for Hermitian data. Directions outside the visible
        sky (l**2 + m**2 > 1, or n < 0) are NaN, except for ``psf=True``.
    """
    l = np.asarray(l, dtype=float)
    m = np.asarray(m, dtype=float)
    if l.shape != m.shape or l.ndim not in (1, 2):
        raise ValueError(
            f"l and m must have the same shape, (npoints,) or (ntimes, npoints); "
            f"got {l.shape} and {m.shape}"
        )
    if l.ndim == 2 and l.shape[0] != data.ntimes:
        raise ValueError(
            f"time-dependent l, m must have shape ({data.ntimes}, npoints), got {l.shape}"
        )
    if n is not None:
        n = np.asarray(n, dtype=float)
        if n.shape != l.shape:
            raise ValueError(f"n must have the same shape as l, got {n.shape}")
    if method not in POINT_METHODS:
        raise ValueError(f"method must be one of {POINT_METHODS}, got {method!r}")
    if w_term not in _W_TERMS:
        raise ValueError(f"w_term must be one of {_W_TERMS}, got {w_term!r}")
    include_w = w_term is not None
    if psf and include_w:
        raise ValueError(
            "w_term is not supported with psf=True: with a w-term the beam "
            "is not a function of offset alone"
        )

    visible = None
    if not psf:
        visible = l**2 + m**2 <= 1
        if n is not None:
            visible &= n >= 0
    n_used = n
    if include_w and n_used is None:
        n_used = np.sqrt(np.clip(1 - l**2 - m**2, 0, None))
    n_target = None
    if include_w:
        n_used = np.where(visible, n_used, 0.0)
        n_target = n_used - 1 if w_term == "zenith" else n_used

    npoints = l.shape[-1]
    if method == "auto":
        uv_points = data.nbls * (data.nfreqs if mfs else 1)
        method = "direct" if uv_points * npoints <= _DIRECT_MAX_TERMS else "type3"

    backend = get_backend(use_gpu)
    complex_dtype, _, eps = _nufft_dtypes(data.vis.dtype, eps)
    per_time = l.ndim == 2
    transform = PointTransform(
        backend,
        n_trans=1 if (mfs or per_time) else data.ntimes,
        method=method,
        eps=eps,
        complex_dtype=complex_dtype,
        include_w=include_w,
        uv_extent=compute_baseline_extent(data.u, data.v),
        w_extent=float(np.max(np.abs(data.w))) if include_w else 1.0,
    )
    values = evaluate_points(
        data, transform, l, m, n_target, mfs=mfs, psf=psf, verbose=verbose
    )

    if visible is not None and not visible.all():
        mask = visible if per_time else np.broadcast_to(visible, (data.ntimes, npoints))
        values[~np.broadcast_to(mask[:, None, :], values.shape)] = np.nan

    freqs = np.array([np.mean(data.freqs)]) if mfs else np.asarray(data.freqs)
    return PointsResult(
        values=values,
        l=l,
        m=m,
        n=n if n is not None else n_used,
        times=np.asarray(data.times),
        freqs=freqs,
        sum_weights=_sum_weights(data),
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
    eps: float = 1e-6,
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
        FINUFFT tolerance (precision). Default is 1e-6.
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
    eps: float = 1e-6,
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
        FINUFFT tolerance (precision). Default is 1e-6.
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
    eps: float = 1e-6,
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
        FINUFFT tolerance (precision). Default is 1e-6.
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
    eps: float = 1e-6,
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
        FINUFFT tolerance (precision). Default is 1e-6.
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
