"""
Snapshot imaging algorithms using Non-Uniform FFT (NUFFT).

This module implements different snapshot imaging algorithms for radio
interferometry data. All algorithms use FINUFFT for efficient computation
and support optional GPU acceleration via CuPy.

Sign convention: visibilities are assumed to follow the pyuvdata/pyuvsim
convention used for HERA data, uvw = xyz(ant2) - xyz(ant1) and
V ∝ exp(+2πi (u*l + v*m + w*n)), so the dirty image is formed with the
kernel exp(-2πi (u*l + v*m)) (FINUFFT ``isign=-1``).

Pixels outside the visible sky (l**2 + m**2 > 1, which only occurs for
fov > 90 degrees) are set to NaN in every imager's output.
"""
import numpy as np
import tqdm

from .data_models import ImagingData, ImageResult
from .coordinates import _below_horizon, compute_image_grid
from .core import (
    _normalize_by_weights,
    _nufft_dtypes,
    get_nufft_library,
    prepare_weighted_visibilities,
    validate_imaging_inputs,
)

# Imaging kernel sign: exp(-2πi(ul + vm)) for V ∝ exp(+2πi(ul + vm + wn)).
_ISIGN = -1


def snapshot_imager_type1(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 180,
    eps: float = 1e-13,
    use_cupy: bool = False,
    modeord: int = 0,
    verbose=True,
    rm_phasor: np.ndarray = None,
) -> ImageResult:
    """
    Snapshot imager using Type 1 NUFFT with plan reuse.
    
    Type 1 NUFFT transforms from non-uniform points (UV coordinates) to
    a uniform grid (image pixels). This is the most efficient approach
    for snapshot imaging when the output grid is regular.
    
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
        Mode ordering: 0 for FINUFFT default (centered, matching ``l_coords``
        and ``m_coords``), 1 for FFT-style ordering (apply ``np.fft.fftshift``
        over the last two axes to match the returned coordinates).
        Default is 0.
    rm_phasor : np.ndarray, optional
        Optional complex array of shape (nfreqs,). If provided, each channel's
        image is multiplied by ``rm_phasor[fi]`` and the channels are summed
        (e.g. for rotation-measure synthesis), so the output has shape
        (ntimes, 1, npix, npix). Default is None.

    Returns
    -------
    ImageResult
        Container with image cube and coordinate information. Each snapshot
        is normalized by its summed weights (the peak of its synthesized
        beam), so a unit point source has peak 1; fully flagged snapshots are
        zero. Pixels below the horizon are NaN.

    Notes
    -----
    This algorithm creates a FINUFFT Plan once per frequency channel and
    processes all time samples together using the n_trans parameter for
    optimal performance.

    The UV coordinates are scaled by 4π·sin(fov/2)/npix for Type 1 NUFFT.
    """
    # Validate inputs
    validate_imaging_inputs(
        imaging_data.vis,
        imaging_data.weights,
        imaging_data.u,
        imaging_data.v,
        npix,
        fov
    )
    
    # Get appropriate libraries
    xp, nufft_lib, use_gpu = get_nufft_library(use_cupy)
    
    # Extract dimensions
    nbls, ntimes, nfreqs = imaging_data.shape
    complex_dtype, real_dtype, eps = _nufft_dtypes(imaging_data.vis.dtype, eps)

    # Set up the image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)

    # Pre-allocate output array
    if rm_phasor is not None:
        image_stack = np.zeros((ntimes, 1, npix, npix), dtype=complex_dtype)
    else:
        image_stack = np.zeros((ntimes, nfreqs, npix, npix), dtype=complex_dtype)

    # Compute the normalization factor for Type 1 NUFFT
    l_max = np.sin(np.deg2rad(fov / 2))
    norm_factor = 4 * np.pi / npix * l_max

    # Process each frequency
    for fi in tqdm.tqdm(range(nfreqs), desc="Imaging frequencies", disable=not verbose):
        # Get and scale UV coordinates for this frequency
        u_scaled = (imaging_data.u[:, fi] * norm_factor).astype(real_dtype)
        v_scaled = (imaging_data.v[:, fi] * norm_factor).astype(real_dtype)

        # Weighted data for all times, shape (ntimes, nbls)
        weighted_data = prepare_weighted_visibilities(
            imaging_data.vis,
            imaging_data.weights,
            freq_idx=fi
        ).astype(complex_dtype, copy=False)

        # Peak of each snapshot's synthesized beam, shape (ntimes,)
        sum_weights = imaging_data.weights[:, :, fi].sum(axis=0)

        plan = nufft_lib.Plan(
            1,
            (npix, npix),
            n_trans=ntimes,
            eps=eps,
            isign=_ISIGN,
            dtype=complex_dtype,
            modeord=modeord
        )

        if use_gpu:
            plan.setpts(xp.asarray(u_scaled), xp.asarray(v_scaled))
            output = xp.asnumpy(plan.execute(xp.asarray(weighted_data)))
        else:
            plan.setpts(u_scaled, v_scaled)
            output = plan.execute(weighted_data)

        # Type 1 output is indexed (time, l, m); images are stored as (time, m, l)
        output = _normalize_by_weights(
            np.transpose(output, axes=(0, 2, 1)), sum_weights
        )
        if rm_phasor is not None:
            image_stack[:, 0, :, :] += output * rm_phasor[fi]
        else:
            image_stack[:, fi, :, :] = output

    below_horizon = _below_horizon(lgrid, mgrid)
    if modeord == 1:
        below_horizon = np.fft.ifftshift(below_horizon)
    image_stack[:, :, below_horizon] = np.nan

    return ImageResult(
        images=image_stack,
        l_coords=lcoords,
        m_coords=mcoords,
        fov=fov,
        npix=npix
    )


def snapshot_imager_type3(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 180,
    eps: float = 1e-13,
    use_cupy: bool = False,
    verbose=True,
) -> ImageResult:
    """
    Snapshot imager using Type 3 NUFFT with plan reuse.
    
    Type 3 NUFFT transforms from non-uniform points to non-uniform points.
    This can be useful for irregular output grids or when evaluating at
    specific sky positions.
    
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
    
    Returns
    -------
    ImageResult
        Container with image cube and coordinate information. Each snapshot
        is normalized by its summed weights (the peak of its synthesized
        beam), so a unit point source has peak 1; fully flagged snapshots are
        zero. Pixels below the horizon are NaN.

    Notes
    -----
    Type 3 NUFFT is generally slower than Type 1 for regular output grids,
    but provides more flexibility in choosing output positions.
    
    The output coordinates are flattened (1D arrays) for Type 3.
    """
    # Validate inputs
    validate_imaging_inputs(
        imaging_data.vis,
        imaging_data.weights,
        imaging_data.u,
        imaging_data.v,
        npix,
        fov
    )
    
    # Get appropriate libraries
    xp, nufft_lib, use_gpu = get_nufft_library(use_cupy)
    
    # Extract dimensions
    nbls, ntimes, nfreqs = imaging_data.shape
    complex_dtype, real_dtype, eps = _nufft_dtypes(imaging_data.vis.dtype, eps)

    # Set up the image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
    lgrid_flat = np.ravel(lgrid)
    mgrid_flat = np.ravel(mgrid)

    # Get maximum baseline extent for scaling
    umax = max(np.max(np.abs(imaging_data.u)), np.max(np.abs(imaging_data.v)))

    # Pre-compute grid coordinates scaled by umax
    lgrid_scaled = (lgrid_flat * umax).astype(real_dtype)
    mgrid_scaled = (mgrid_flat * umax).astype(real_dtype)

    # Normalization factor
    norm_factor = 2 * np.pi / umax

    # Pre-allocate output array
    image_stack = np.zeros((ntimes, nfreqs, npix, npix), dtype=complex_dtype)

    if use_gpu:
        # Target points are the same for every frequency
        lgrid_scaled = xp.asarray(lgrid_scaled)
        mgrid_scaled = xp.asarray(mgrid_scaled)

    # Process each frequency
    for fi in tqdm.tqdm(range(nfreqs), desc="Imaging frequencies", disable=not verbose):
        # Scale UV coordinates for this frequency
        u_scaled = (imaging_data.u[:, fi] * norm_factor).astype(real_dtype)
        v_scaled = (imaging_data.v[:, fi] * norm_factor).astype(real_dtype)

        # Weighted data for all times, shape (ntimes, nbls)
        weighted_data = prepare_weighted_visibilities(
            imaging_data.vis,
            imaging_data.weights,
            freq_idx=fi
        ).astype(complex_dtype, copy=False)

        # Peak of each snapshot's synthesized beam, shape (ntimes,)
        sum_weights = imaging_data.weights[:, :, fi].sum(axis=0)

        # Type 3 plan in 2 dimensions
        plan = nufft_lib.Plan(
            3,
            2,
            n_trans=ntimes,
            eps=eps,
            isign=_ISIGN,
            dtype=complex_dtype
        )

        if use_gpu:
            plan.setpts(
                xp.asarray(u_scaled), xp.asarray(v_scaled),
                s=lgrid_scaled, t=mgrid_scaled
            )
            output = xp.asnumpy(plan.execute(xp.asarray(weighted_data)))
        else:
            plan.setpts(u_scaled, v_scaled, s=lgrid_scaled, t=mgrid_scaled)
            output = plan.execute(weighted_data)

        # Reshape and store
        image_stack[:, fi, :, :] = _normalize_by_weights(
            output.reshape(ntimes, npix, npix), sum_weights
        )

    image_stack[:, :, _below_horizon(lgrid, mgrid)] = np.nan

    return ImageResult(
        images=image_stack,
        l_coords=lgrid_flat,
        m_coords=mgrid_flat,
        fov=fov,
        npix=npix
    )

def snapshot_imager_mfs_type_1(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 10,
    eps: float = 1e-13,
    use_cupy: bool = False,
    verbose=True,
) -> ImageResult:
    """
    Multi-frequency synthesis (MFS) snapshot imager using Type 1 NUFFT with plan reuse.
    
    This implementation uses FINUFFT plans for efficient Type 1 NUFFT computation,
    processing all baselines across all frequencies together for each time step.
    Supports optional GPU acceleration via CuPy/cuFINUFFT.
    
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
    
    Returns
    -------
    ImageResult
        Container with image cube and coordinate information
    
    Notes
    -----
    For each time step, all baselines across all frequency channels are
    concatenated into a single set of non-uniform UV points, producing one
    wideband MFS image. The output image cube has shape
    (ntimes, 1, npix, npix).

    MFS images are not normalized: each pixel is the weighted sum of the
    visibilities (a unit point source peaks at the summed weights). Pixels
    below the horizon are NaN.
    """
    # Validate inputs
    validate_imaging_inputs(
        imaging_data.vis,
        imaging_data.weights,
        imaging_data.u,
        imaging_data.v,
        npix,
        fov
    )
    
    # Get appropriate libraries
    xp, nufft_lib, use_gpu = get_nufft_library(use_cupy)
    
    # Extract dimensions
    nbls, ntimes, nfreqs = imaging_data.shape
    complex_dtype, real_dtype, eps = _nufft_dtypes(imaging_data.vis.dtype, eps)

    # Set up the image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)

    # Normalization factor
    l_max = np.sin(np.deg2rad(fov / 2))
    norm_factor = 4 * np.pi / npix * l_max

    # Pre-allocate output array: one MFS image per time step
    image_stack = np.zeros((ntimes, 1, npix, npix), dtype=complex_dtype)

    # Process each time step, combining all baselines and frequencies
    for ti in tqdm.tqdm(range(ntimes), desc="Imaging Times", disable=not verbose):
        # Ravel UV coords in (nfreqs, nbls) order to match the weighted data below
        u_scaled = np.ravel(imaging_data.u.T * norm_factor).astype(real_dtype)
        v_scaled = np.ravel(imaging_data.v.T * norm_factor).astype(real_dtype)

        # Prepare weighted data: shape (nfreqs, nbls) -> ravel to (nfreqs*nbls,)
        weighted_data = np.ravel(prepare_weighted_visibilities(
            imaging_data.vis,
            imaging_data.weights,
            time_idx=ti
        )).astype(complex_dtype, copy=False)

        plan = nufft_lib.Plan(
            1,
            (npix, npix),
            n_trans=1,
            eps=eps,
            isign=_ISIGN,
            dtype=complex_dtype,
            modeord=0,
        )

        if use_gpu:
            plan.setpts(xp.asarray(u_scaled), xp.asarray(v_scaled))
            output = xp.asnumpy(plan.execute(xp.asarray(weighted_data)))
        else:
            plan.setpts(u_scaled, v_scaled)
            output = plan.execute(weighted_data)

        # Type 1 output is transposed relative to image convention
        image_stack[ti, 0, :, :] = output.T

    image_stack[:, :, _below_horizon(lgrid, mgrid)] = np.nan

    return ImageResult(
        images=image_stack,
        l_coords=lcoords,
        m_coords=mcoords,
        fov=fov,
        npix=npix
    )

def snapshot_imager_mfs_type_3(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 10,
    eps: float = 1e-13,
    use_cupy: bool = False,
    verbose=True,
) -> ImageResult:
    """
    Multi-frequency synthesis (MFS) snapshot imager using Type 3 NUFFT with plan reuse.
    
    This implementation uses FINUFFT plans for efficient Type 3 NUFFT computation,
    processing all baselines across all frequencies together for each time step.
    Supports optional GPU acceleration via CuPy/cuFINUFFT.
    
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
    
    Returns
    -------
    ImageResult
        Container with image cube and coordinate information
    
    Notes
    -----
    For each time step, all baselines across all frequency channels are
    concatenated into a single set of non-uniform UV points, producing one
    wideband MFS image. The output image cube has shape
    (ntimes, 1, npix, npix).

    MFS images are not normalized: each pixel is the weighted sum of the
    visibilities (a unit point source peaks at the summed weights). Pixels
    below the horizon are NaN.
    """
    # Validate inputs
    validate_imaging_inputs(
        imaging_data.vis,
        imaging_data.weights,
        imaging_data.u,
        imaging_data.v,
        npix,
        fov
    )
    
    # Get appropriate libraries
    xp, nufft_lib, use_gpu = get_nufft_library(use_cupy)
    
    # Extract dimensions
    nbls, ntimes, nfreqs = imaging_data.shape
    complex_dtype, real_dtype, eps = _nufft_dtypes(imaging_data.vis.dtype, eps)

    # Set up the image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
    lgrid_flat = np.ravel(lgrid)
    mgrid_flat = np.ravel(mgrid)

    # Get maximum baseline extent for scaling
    umax = max(np.max(np.abs(imaging_data.u)), np.max(np.abs(imaging_data.v)))

    # Pre-compute grid coordinates scaled by umax
    lgrid_scaled = (lgrid_flat * umax).astype(real_dtype)
    mgrid_scaled = (mgrid_flat * umax).astype(real_dtype)

    # Compute the normalization factor for Type 3 NUFFT
    norm_factor = 2 * np.pi / umax

    # Pre-allocate output array: one MFS image per time step
    image_stack = np.zeros((ntimes, 1, npix, npix), dtype=complex_dtype)

    if use_gpu:
        # Target points are the same for every time step
        lgrid_scaled = xp.asarray(lgrid_scaled)
        mgrid_scaled = xp.asarray(mgrid_scaled)

    # Process each time step, combining all baselines and frequencies
    for ti in tqdm.tqdm(range(ntimes), desc="Imaging Times", disable=not verbose):
        # Ravel UV coords in (nfreqs, nbls) order to match the weighted data below
        u_scaled = np.ravel(imaging_data.u.T * norm_factor).astype(real_dtype)
        v_scaled = np.ravel(imaging_data.v.T * norm_factor).astype(real_dtype)

        # Prepare weighted data: shape (nfreqs, nbls) -> ravel to (nfreqs*nbls,)
        weighted_data = np.ravel(prepare_weighted_visibilities(
            imaging_data.vis,
            imaging_data.weights,
            time_idx=ti
        )).astype(complex_dtype, copy=False)

        # Type 3 plan in 2 dimensions
        plan = nufft_lib.Plan(
            3,
            2,
            n_trans=1,
            eps=eps,
            isign=_ISIGN,
            dtype=complex_dtype
        )

        if use_gpu:
            plan.setpts(
                xp.asarray(u_scaled), xp.asarray(v_scaled),
                s=lgrid_scaled, t=mgrid_scaled
            )
            output = xp.asnumpy(plan.execute(xp.asarray(weighted_data)))
        else:
            plan.setpts(u_scaled, v_scaled, s=lgrid_scaled, t=mgrid_scaled)
            output = plan.execute(weighted_data)

        image_stack[ti, 0, :, :] = output.reshape(npix, npix)

    image_stack[:, :, _below_horizon(lgrid, mgrid)] = np.nan

    return ImageResult(
        images=image_stack,
        l_coords=lgrid_flat,
        m_coords=mgrid_flat,
        fov=fov,
        npix=npix
    )