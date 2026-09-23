"""
Core imaging utilities shared across different imaging algorithms.

This module contains common functionality used by multiple imaging methods,
including library selection, plan management, and coordinate scaling.
"""
import numpy as np


def get_nufft_library(use_cupy: bool = False):
    """
    Get the appropriate NUFFT library and array library.
    
    Parameters
    ----------
    use_cupy : bool, optional
        Whether to attempt GPU acceleration with CuPy/cuFINUFFT.
        Default is False.
    
    Returns
    -------
    xp : module
        Array library (numpy or cupy)
    nufft_lib : module
        NUFFT library (finufft or cufinufft)
    use_gpu : bool
        Whether GPU acceleration is actually being used
    
    Notes
    -----
    If CuPy/cuFINUFFT are requested but not available, falls back to
    CPU version with a warning.
    """
    if use_cupy:
        try:
            import cupy as cp
            import cufinufft
            xp = cp
            nufft_lib = cufinufft
            use_gpu = True
            print("Using GPU acceleration with CuPy/cuFINUFFT")
        except ImportError:
            print("Warning: CuPy/cuFINUFFT not available, falling back to CPU")
            import numpy as np
            import finufft
            xp = np
            nufft_lib = finufft
            use_gpu = False
    else:
        import numpy as np
        import finufft
        xp = np
        nufft_lib = finufft
        use_gpu = False
    
    return xp, nufft_lib, use_gpu

def prepare_weighted_visibilities(
    vis: np.ndarray,
    weights: np.ndarray,
    freq_idx: int | None = None,
    time_idx: int | None = None
) -> np.ndarray:
    """
    Prepare weighted visibilities for a specific frequency.
    
    Parameters
    ----------
    vis : np.ndarray
        Visibility data, shape (nbls, ntimes, nfreqs)
    weights : np.ndarray
        Visibility weights, shape (nbls, ntimes, nfreqs)
    freq_idx : int
        Frequency channel index
    
    Returns
    -------
    np.ndarray
        Weighted visibilities, shape (ntimes, nbls)
    
    Notes
    -----
    The output is transposed to (ntimes, nbls) for compatibility with
    FINUFFT's n_trans parameter.
    """
    if freq_idx is None and time_idx is None:
        raise ValueError("Must specify either freq_idx or time_idx")
    if freq_idx is not None and time_idx is not None:
        raise ValueError("Cannot specify both freq_idx and time_idx")
    
    if freq_idx is not None:
        weighted_data = vis[:, :, freq_idx] * weights[:, :, freq_idx]
    else:
        weighted_data = vis[:, time_idx, :] * weights[:, time_idx, :]

    # Transpose to (ntimes/nfreqs, nbls). FINUFFT requires C-contiguous input
    # and would otherwise copy (and warn) on every execute call.
    return np.ascontiguousarray(weighted_data.T)


# Smallest NUFFT tolerance used for single-precision data
_MIN_EPS_SINGLE = 1e-6


def _nufft_dtypes(vis_dtype, eps: float):
    """
    Choose NUFFT precision to match the visibilities.

    Returns ``(complex_dtype, real_dtype, eps)``. Single-precision (complex64)
    visibilities use a single-precision NUFFT with float32 points. Tolerances
    tighter than ~1e-6 are not achievable in single precision (FINUFFT warns
    on every plan, and Type 3 accuracy actually degrades on older FINUFFT), so
    ``eps`` is floored at ``_MIN_EPS_SINGLE`` there.
    """
    if np.dtype(vis_dtype) == np.complex64:
        return np.complex64, np.float32, max(eps, _MIN_EPS_SINGLE)
    return np.complex128, np.float64, eps


def _normalize_by_weights(images: np.ndarray, sum_weights: np.ndarray) -> np.ndarray:
    """
    Normalize each snapshot image by its summed visibility weights.

    For non-negative weights the synthesized beam (PSF) of a snapshot peaks at
    the phase center with value ``sum(weights)``, so this puts every snapshot
    in units where a unit point source has peak 1. Snapshots with no weight
    (fully flagged) are set to zero.

    Parameters
    ----------
    images : np.ndarray
        Images with shape (ntimes, npix, npix).
    sum_weights : np.ndarray
        Summed weights for each snapshot, shape (ntimes,).
    """
    sum_weights = np.asarray(sum_weights)[:, None, None]
    return np.divide(
        images, sum_weights, out=np.zeros_like(images), where=sum_weights != 0
    )


def validate_imaging_inputs(
    vis: np.ndarray,
    weights: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    npix: int,
    fov: float,
):
    """
    Validate imaging input parameters.
    
    Parameters
    ----------
    vis : np.ndarray
        Visibility data
    weights : np.ndarray
        Visibility weights
    u : np.ndarray
        U-coordinates
    v : np.ndarray
        V-coordinates
    npix : int
        Number of pixels
    fov : float
        Field of view in degrees
    
    Raises
    ------
    ValueError
        If any inputs are invalid
    """
    if vis.shape != weights.shape:
        raise ValueError(
            f"vis shape {vis.shape} doesn't match weights shape {weights.shape}"
        )
    
    nbls, ntimes, nfreqs = vis.shape
    
    if u.shape != (nbls, nfreqs):
        raise ValueError(
            f"u shape {u.shape} doesn't match expected shape ({nbls}, {nfreqs})"
        )
    
    if v.shape != (nbls, nfreqs):
        raise ValueError(
            f"v shape {v.shape} doesn't match expected shape ({nbls}, {nfreqs})"
        )
    
    if npix <= 0:
        raise ValueError(f"npix must be positive, got {npix}")
    
    if fov <= 0 or fov > 180:
        raise ValueError(f"fov must be between 0 and 180 degrees, got {fov}")


def estimate_memory_requirements(
    nbls: int,
    ntimes: int,
    nfreqs: int,
    npix: int,
    dtype: np.dtype = np.complex128
) -> dict:
    """
    Estimate memory requirements for imaging.
    
    Parameters
    ----------
    nbls : int
        Number of baselines
    ntimes : int
        Number of time samples
    nfreqs : int
        Number of frequency channels
    npix : int
        Number of pixels per dimension
    dtype : np.dtype, optional
        Data type for complex arrays
    
    Returns
    -------
    dict
        Dictionary with memory estimates in GB:
        - 'input_data': Input visibility data
        - 'output_images': Output image cube
        - 'working_memory': Estimated working memory
        - 'total': Total estimated memory
    """
    bytes_per_element = np.dtype(dtype).itemsize
    
    input_data = nbls * ntimes * nfreqs * bytes_per_element / 1e9
    output_images = ntimes * nfreqs * npix * npix * bytes_per_element / 1e9
    
    # Rough estimate for FINUFFT working memory
    # This is approximate and depends on the NUFFT algorithm
    working_memory = max(input_data, output_images) * 2
    
    total = input_data + output_images + working_memory
    
    return {
        'input_data': input_data,
        'output_images': output_images,
        'working_memory': working_memory,
        'total': total
    }
