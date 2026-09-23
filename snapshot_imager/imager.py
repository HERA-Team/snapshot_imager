"""
Snapshot imaging algorithms using Non-Uniform FFT (NUFFT).

This module implements different snapshot imaging algorithms for radio
interferometry data. All algorithms use FINUFFT for efficient computation
and support optional GPU acceleration via CuPy.
"""
import numpy as np
import tqdm

from .data_models import ImagingData, ImageResult
from .coordinates import compute_image_grid
from .core import (
    get_nufft_library,
    prepare_weighted_visibilities,
    validate_imaging_inputs,
)


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
        Mode ordering: 0 for FINUFFT default (fftshift), 1 for FFT-style.
        Default is 0.
    rm_phasor : np.ndarray, optional
        Optional array of shape (nfreqs,) containing phasor values to remove
        from the visibilities before imaging. If provided, the phasor will be
        divided out from the visibilities for each frequency channel. Default is None.
    
    Returns
    -------
    ImageResult
        Container with image cube and coordinate information
    
    Notes
    -----
    This algorithm creates a FINUFFT Plan once per frequency channel and
    processes all time samples together using the n_trans parameter for
    optimal performance.
    
    The UV coordinates are scaled by 4π/npix for Type 1 NUFFT.
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
    
    # Set up the image grid
    lcoords, mcoords, _, _ = compute_image_grid(npix, fov)
    
    # Pre-allocate output array
    if rm_phasor is not None:
        image_stack = np.zeros((ntimes, 1, npix, npix), dtype=complex)
    else:
        image_stack = np.zeros((ntimes, nfreqs, npix, npix), dtype=complex)

    # Compute the normalization factor for Type 1 NUFFT
    l_max = np.sin(np.deg2rad(fov / 2))
    norm_factor = 4 * np.pi / npix * l_max

    # Process each frequency
    for fi in tqdm.tqdm(range(nfreqs), desc="Imaging frequencies", disable=not verbose):
        # Get and scale UV coordinates for this frequency
        u_scaled = imaging_data.u[:, fi] * norm_factor
        v_scaled = imaging_data.v[:, fi] * norm_factor
        
        if use_gpu:
            # Transfer UV coords to GPU once per frequency
            u_gpu = xp.asarray(u_scaled)
            v_gpu = xp.asarray(v_scaled)
            
            # Create Type 1 plan
            plan = nufft_lib.Plan(
                1,
                (npix, npix),
                n_trans=ntimes,
                eps=eps,
                dtype=xp.complex64 if imaging_data.vis.dtype == np.complex64 else xp.complex128,
                modeord=modeord
            )
            
            # Set the non-uniform points
            plan.setpts(u_gpu, v_gpu)
            
            # Prepare weighted data for all times
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                freq_idx=fi
            )
            data_gpu = xp.asarray(weighted_data)
            
            # Execute transform for all time steps at once
            output_gpu = plan.execute(data_gpu)

            # Compute weight normalisation
            norm_weights = prepare_weighted_visibilities(
                np.ones_like(imaging_data.vis),
                imaging_data.weights,
                freq_idx=fi
            )
            wgts_gpu = xp.asarray(norm_weights)
            
            # Execute transform for all time steps at once
            wgts_output_gpu = plan.execute(wgts_gpu)
            norm = wgts_output_gpu.real.max()
            
            # Transfer back to CPU, reshape, and store results
            output_cpu = np.transpose(xp.asnumpy(output_gpu), axes=(0, 2, 1))
            output_cpu = np.divide(
                output_cpu,
                norm,
                out=np.zeros_like(output_cpu),
                where=norm != 0
            )
            if rm_phasor is not None:
                image_stack[:, 0, :, :] += output_cpu * rm_phasor[fi]
            else:
                image_stack[:, fi, :, :] = output_cpu
            
        else:
            # CPU version
            plan = nufft_lib.Plan(
                1,
                (npix, npix),
                n_trans=ntimes,
                eps=eps,
                dtype=np.complex64 if imaging_data.vis.dtype == np.complex64 else np.complex128,
                modeord=modeord
            )
            
            # Set the non-uniform points
            plan.setpts(u_scaled, v_scaled)
            
            # Prepare weighted data for all times
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                freq_idx=fi
            )
            
            # Execute transform
            output = plan.execute(weighted_data)

            # Compute weight normalisation
            norm_weights = prepare_weighted_visibilities(
                np.ones_like(imaging_data.vis),
                imaging_data.weights,
                freq_idx=fi
            )
            
            # Execute transform
            wgts_output = plan.execute(norm_weights)
            norm = wgts_output.real.max()
            
            output = np.divide(
                output,
                norm,
                out=np.zeros_like(output),
                where=norm != 0
            )

            # Store results, Type 1 is transposed compared to Type 3
            if rm_phasor is not None:
                image_stack[:, 0, :, :] += np.transpose(output, axes=(0, 2, 1)) * rm_phasor[fi]
            else:
                image_stack[:, fi, :, :] = np.transpose(output, axes=(0, 2, 1))
    
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
        Container with image cube and coordinate information
    
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
    
    # Set up the image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
    lgrid_flat = np.ravel(lgrid)
    mgrid_flat = np.ravel(mgrid)

    # Get maximum baseline extent for scaling
    umax = max(np.max(np.abs(imaging_data.u)), np.max(np.abs(imaging_data.v)))
    
    # Pre-compute grid coordinates scaled by umax
    lgrid_scaled = lgrid_flat * umax
    mgrid_scaled = mgrid_flat * umax
    
    # Normalization factor
    norm_factor = 2 * np.pi / umax
    
    # Pre-allocate output array
    image_stack = np.zeros((ntimes, nfreqs, npix, npix), dtype=complex)
    
    # Process each frequency
    for fi in tqdm.tqdm(range(nfreqs), desc="Imaging frequencies", disable=not verbose):
        # Scale UV coordinates for this frequency
        u_scaled = imaging_data.u[:, fi] * norm_factor
        v_scaled = imaging_data.v[:, fi] * norm_factor
        
        if use_gpu:
            # Transfer coordinates to GPU once per frequency
            u_gpu = xp.asarray(u_scaled)
            v_gpu = xp.asarray(v_scaled)
            lgrid_gpu = xp.asarray(lgrid_scaled)
            mgrid_gpu = xp.asarray(mgrid_scaled)
            
            # Create Type 3 plan
            plan = nufft_lib.Plan(
                nufft_type=3,
                n_modes=2,  # Dimension (2D transform)
                n_trans=ntimes,
                eps=eps,
                dtype=xp.complex64 if imaging_data.vis.dtype == np.complex64 else xp.complex128
            )
            
            # Set source and target points
            plan.setpts(u_gpu, v_gpu, s=lgrid_gpu, t=mgrid_gpu)
            
            # Prepare weighted data
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                freq_idx=fi
            )
            data_gpu = xp.asarray(weighted_data)

            # Execute transform
            output_gpu = plan.execute(data_gpu)

            # Prepare weighted data
            weights = prepare_weighted_visibilities(
                np.ones_like(imaging_data.vis),
                imaging_data.weights,
                freq_idx=fi
            )
            wgts_gpu = xp.asarray(weights)

            # Execute transform
            wgts_output_gpu = plan.execute(wgts_gpu)
            norm = wgts_output_gpu.real.max()
            
            # Transfer back and reshape
            output_cpu = xp.asnumpy(output_gpu / norm)
            image_stack[:, fi, :, :] = output_cpu.reshape(ntimes, npix, npix)
            
        else:
            # CPU version
            plan = nufft_lib.Plan(
                nufft_type=3,
                n_modes_or_dim=2,
                n_trans=ntimes,
                eps=eps,
                dtype=np.complex64 if imaging_data.vis.dtype == np.complex64 else np.complex128
            )
            
            # Set source and target points
            plan.setpts(u_scaled, v_scaled, s=lgrid_scaled, t=mgrid_scaled)
            
            # Prepare weighted data
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                freq_idx=fi
            )
            # Execute transform
            output = plan.execute(weighted_data)

            # Prepare weighted data
            weights = prepare_weighted_visibilities(
                np.ones_like(imaging_data.vis),
                imaging_data.weights,
                freq_idx=fi
            )
            # Execute transform
            wgts_output = plan.execute(weights)
            norm = wgts_output.real.max()
            output = np.divide(
                output,
                norm,
                out=np.zeros_like(output),
                where=norm != 0
            )
            
            # Reshape and store
            image_stack[:, fi, :, :] = output.reshape(ntimes, npix, npix)
    
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
    
    # Set up the image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
    lgrid_flat = np.ravel(lgrid)
    mgrid_flat = np.ravel(mgrid)

    # Normalization factor
    l_max = np.max([np.abs(lgrid_flat).max(), np.abs(mgrid_flat).max()])
    norm_factor = 4 * np.pi / npix * l_max

    # Pre-allocate output array: one MFS image per time step
    image_stack = np.zeros((ntimes, 1, npix, npix), dtype=complex)

    # Process each time step, combining all baselines and frequencies
    for ti in tqdm.tqdm(range(ntimes), desc="Imaging Times", disable=not verbose):
        # Ravel UV coords in (nfreqs, nbls) order to match the weighted data below
        u_scaled = np.ravel(imaging_data.u.T * norm_factor)
        v_scaled = np.ravel(imaging_data.v.T * norm_factor)

        # Prepare weighted data: shape (nfreqs, nbls) -> ravel to (nfreqs*nbls,)
        weighted_data = prepare_weighted_visibilities(
            imaging_data.vis,
            imaging_data.weights,
            time_idx=ti
        )

        if use_gpu:
            u_gpu = xp.asarray(u_scaled)
            v_gpu = xp.asarray(v_scaled)

            plan = nufft_lib.Plan(
                1,
                (npix, npix),
                n_trans=1,
                eps=eps,
                dtype=xp.complex64 if imaging_data.vis.dtype == np.complex64 else xp.complex128,
                modeord=0,
            )
            plan.setpts(u_gpu, v_gpu)

            data_gpu = xp.asarray(np.ravel(weighted_data))
            output_gpu = plan.execute(data_gpu)
            # Type 1 output is transposed relative to image convention
            image_stack[ti, 0, :, :] = xp.asnumpy(output_gpu).T

        else:
            plan = nufft_lib.Plan(
                1,
                (npix, npix),
                n_trans=1,
                eps=eps,
                dtype=np.complex64 if imaging_data.vis.dtype == np.complex64 else np.complex128,
                modeord=0,
            )
            plan.setpts(u_scaled, v_scaled)

            output = plan.execute(np.ravel(weighted_data))
            # Type 1 output is transposed relative to image convention
            image_stack[ti, 0, :, :] = output.T

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
    processing all time samples together using the n_trans parameter. Supports
    optional GPU acceleration via CuPy/cuFINUFFT.
    
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
    
    # Set up the image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
    lgrid_flat = np.ravel(lgrid)
    mgrid_flat = np.ravel(mgrid)

    # Get maximum baseline extent for scaling
    umax = max(np.max(np.abs(imaging_data.u)), np.max(np.abs(imaging_data.v)))
    
    # Pre-compute grid coordinates scaled by umax
    lgrid_scaled = lgrid_flat * umax
    mgrid_scaled = mgrid_flat * umax
        
    # Compute the normalization factor for Type 3 NUFFT
    norm_factor = 2 * np.pi / umax
    
    # Pre-allocate output array: one MFS image per time step
    image_stack = np.zeros((ntimes, 1, npix, npix), dtype=complex)

    # Process each time step, combining all baselines and frequencies
    for ti in tqdm.tqdm(range(ntimes), desc="Imaging Times", disable=not verbose):
        # Ravel UV coords in (nfreqs, nbls) order to match the weighted data below
        u_scaled = np.ravel(imaging_data.u.T * norm_factor)
        v_scaled = np.ravel(imaging_data.v.T * norm_factor)

        # Prepare weighted data: shape (nfreqs, nbls) -> ravel to (nfreqs*nbls,)
        weighted_data = prepare_weighted_visibilities(
            imaging_data.vis,
            imaging_data.weights,
            time_idx=ti
        )

        if use_gpu:
            u_gpu = xp.asarray(u_scaled)
            v_gpu = xp.asarray(v_scaled)
            lgrid_gpu = xp.asarray(lgrid_scaled)
            mgrid_gpu = xp.asarray(mgrid_scaled)

            plan = nufft_lib.Plan(
                nufft_type=3,
                n_modes=2,
                n_trans=1,
                eps=eps,
                dtype=xp.complex64 if imaging_data.vis.dtype == np.complex64 else xp.complex128
            )
            plan.setpts(u_gpu, v_gpu, s=lgrid_gpu, t=mgrid_gpu)

            data_gpu = xp.asarray(np.ravel(weighted_data))
            output_gpu = plan.execute(data_gpu)
            image_stack[ti, 0, :, :] = xp.asnumpy(output_gpu).reshape(npix, npix)

        else:
            plan = nufft_lib.Plan(
                nufft_type=3,
                n_modes_or_dim=2,
                n_trans=1,
                eps=eps,
                dtype=np.complex64 if imaging_data.vis.dtype == np.complex64 else np.complex128
            )
            plan.setpts(u_scaled, v_scaled, s=lgrid_scaled, t=mgrid_scaled)

            output = plan.execute(np.ravel(weighted_data))
            image_stack[ti, 0, :, :] = output.reshape(npix, npix)

    return ImageResult(
        images=image_stack,
        l_coords=lgrid_flat,
        m_coords=mgrid_flat,
        fov=fov,
        npix=npix
    )